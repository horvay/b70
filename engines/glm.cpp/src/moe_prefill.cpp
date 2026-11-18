// Routed experts for prompts (M >= 256 rows): DPAS GEMMs over 32-row blocks per expert, fp32 accumulation, per-pair y in
// fp16 before the combine (docs/forward-spec.md §6.5). A port of moe_prefill in llm/glm/xpu/csrc/exl3_moe.n128.sycl.
// Built as its own library (libglm_moe_pf.so) with large-GRF code generation, which the decode GEMV must not get.
#include "moe.h"
#include "../kernels/exl3_moe_kernels.h"

namespace glm {

using namespace glm_moe;
using fp16 = sycl::half;

void moe_align(sycl::queue& q, const int* ids, const float* w, int npairs, int topk, int E, int pad, int* act_expert,
               int* act_start, int* act_cnt, int* n_act, int* act_pstart, int* pair_tok, float* pair_w, int* pair_of,
               int* blk_a, int* n_blk);

void moe_prefill(sycl::queue& q, Workspace& ws, const half* x, int M, const int* sel, const float* w, int topk,
                 const int64_t* ptrs, int H, int I, int kc, int E, half* out, float act_lim) {
    constexpr int MB = 32;
    ws.reset();
    Geom gm{H, I, kc};
    GLM_CHECK(I % 128 == 0 && H % (4 * 128) == 0 && (2 * I / 16) % 8 == 0, "moe prefill geometry");
    int npairs = M * topk, A = std::min(npairs, E);
    int mxb = (npairs + MB - 1) / MB + A;          // sum over experts of ceil(count / MB)
    int* ib = ws.take<int>(4 * A + 1 + 2 * npairs + 16 + mxb + 1);
    float* pw = ws.take<float>(npairs);
    int *ae = ib, *as = ib + A, *ac = ib + 2 * A, *aps = ib + 3 * A, *na = ib + 4 * A;
    int* pt = ib + 4 * A + 1;
    int* po = pt + npairs;
    int* blk_a = po + npairs + 16;
    int* n_blk = blk_a + mxb;
    moe_align(q, sel, w, npairs, topk, E, MB, ae, as, ac, na, aps, pt, pw, po, blk_a, n_blk);
    Route rt{ae, as, ac, na, aps, pt, pw};
    fp16* ablk = ws.take<fp16>((size_t)mxb * (I / 16) * MB * 16);
    fp16* y = ws.take<fp16>((size_t)npairs * H);
    sycl::ext::oneapi::experimental::properties props{sycl::ext::intel::experimental::grf_size<256>};
    // with 256 registers an Xe-core holds 32 hardware threads: split each row block's columns over S work-groups
    int T1 = 2 * (I / 16) / 4, S = 1;
    while (T1 / S > 32 && (I / 128) % (2 * S) == 0) S *= 2;
    GLM_CHECK(T1 / S <= 32, "moe prefill: gate/up work-group too large for large-GRF mode");
    int T2 = (H / 4 / 16) / 4;
    auto run = [&](auto kt) {
        constexpr int KK = decltype(kt)::value;
        GateUpDpasKernel<KK, MB, fp16> k{x, H, ptrs, rt, blk_a, n_blk, ablk, H, I, gm.off_suh_g(), gm.off_svh_g(),
                                         gm.off_suh_d(), S, act_lim};
        q.submit([&](sycl::handler& h) { h.parallel_for(sycl::nd_range<1>((size_t)mxb * T1, T1 / S), props, k); });
        DownDpasKernel<KK, MB, fp16> kd{ablk, ptrs, rt, blk_a, n_blk, y, H, I, gm.gu_bytes(), gm.off_svh_d()};
        q.submit([&](sycl::handler& h) { h.parallel_for(sycl::nd_range<1>((size_t)mxb * 4 * T2, T2), props, kd); });
    };
    if (kc == 3) run(std::integral_constant<int, 3>{});
    else if (kc == 4) run(std::integral_constant<int, 4>{});
    else if (kc == 25) run(std::integral_constant<int, 25>{});
    else if (kc == 35) run(std::integral_constant<int, 35>{});
    else GLM_CHECK(false, "moe prefill: K code %d", kc);
    int n = M * (H / 128), local = 8;
    CombineKernel<fp16, fp16> ck{y, po, out, M, H, topk, H};
    q.submit([&](sycl::handler& h) { h.parallel_for(sycl::nd_range<1>((n + local - 1) / local * local, local), ck); });
}

}  // namespace glm
