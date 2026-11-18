// Host launchers for the routed-expert kernels, without torch: a port of moe_forward_impl from
// llm/glm/xpu/csrc/exl3_moe.n128.sycl with the PyTorch stack's settings (glm_xpu/ops.py moe_ops: gate|up k-splits 8,
// down 4, pair mode for M = 1, grouped GEMV above). The DPAS prompt path (M >= 256) is not ported yet.
#include "moe.h"
#include "../kernels/exl3_moe_kernels.h"

namespace glm {

using namespace glm_moe;
using fp16 = sycl::half;

void moe_align(sycl::queue& q, const int* ids, const float* w, int npairs, int topk, int E, int pad, int* act_expert,
               int* act_start, int* act_cnt, int* n_act, int* act_pstart, int* pair_tok, float* pair_w, int* pair_of,
               int* blk_a = nullptr, int* n_blk = nullptr);
// prompt path (M >= 256): libglm_moe_pf.so, built with large-GRF code generation
void moe_prefill(sycl::queue& q, Workspace& ws, const half* x, int M, const int* sel, const float* w, int topk,
                 const int64_t* ptrs, int H, int I, int kc, int E, half* out, float act_lim);

namespace {
constexpr int kGuP = 8, kDnP = 4, kPairMaxM = 1;

template <int K, int MR, int GP>
void gateup_p(sycl::queue& q, const fp16* x, int xs, const int64_t* ptrs, Route rt, fp16* abuf, int H, int I,
              const Geom& gm, int max_act, float lim) {
    GLM_CHECK((H / 16) % (GP * 8) == 0, "moe gate/up: H/16 split");
    GateUpKernel<K, MR, GP, fp16> k{x, xs, ptrs, rt, abuf, H, I, gm.off_suh_g(), gm.off_svh_g(), gm.off_suh_d(),
                                    (int)((gm.blob() - gm.off_suh_g()) / 4), lim};
    constexpr int T = 4 * GP;
    size_t n = (size_t)max_act * (I / 128) * T;
    q.submit([&](sycl::handler& h) { h.parallel_for(sycl::nd_range<1>(n, T), k); });
}

// gate/up k-splits: whole 8-tile-row (one Hadamard block) runs each. 8 for GLM-5.3 (H/16 = 256), 10 for
// Qwen3.8-Flash-Next (H/16 = 160)
template <int K, int MR>
void gateup(sycl::queue& q, const fp16* x, int xs, const int64_t* ptrs, Route rt, fp16* abuf, int H, int I,
            const Geom& gm, int max_act, float lim) {
    if ((H / 16) % (kGuP * 8) == 0) gateup_p<K, MR, kGuP>(q, x, xs, ptrs, rt, abuf, H, I, gm, max_act, lim);
    else gateup_p<K, MR, 10>(q, x, xs, ptrs, rt, abuf, H, I, gm, max_act, lim);
}

template <int K, int MR, int DP>
void down_p(sycl::queue& q, const fp16* abuf, const int64_t* ptrs, Route rt, float* y, int H, int I, const Geom& gm,
            int max_act) {
    constexpr int RB = DownKernel<K, MR, DP>::RB;
    GLM_CHECK((I / 16) % DP == 0 && ((I / 16) / DP) % RB == 0, "moe down: I/16 split");
    DownKernel<K, MR, DP> k{abuf, ptrs, rt, y, H, I, gm.gu_bytes(), gm.off_svh_d()};
    constexpr int T = 2 * DP;
    size_t n = (size_t)max_act * (H / 128) * T;
    q.submit([&](sycl::handler& h) { h.parallel_for(sycl::nd_range<1>(n, T), k); });
}

// down k-splits: 4 for GLM-5.3 (I/16 = 128), 5 for Qwen3.8-Flash-Next (I/16 = 40: 8 tile rows each, one batch)
template <int K, int MR>
void down(sycl::queue& q, const fp16* abuf, const int64_t* ptrs, Route rt, float* y, int H, int I, const Geom& gm,
          int max_act) {
    if ((I / 16) % (kDnP * DownKernel<K, MR, kDnP>::RB) == 0) down_p<K, MR, kDnP>(q, abuf, ptrs, rt, y, H, I, gm, max_act);
    else down_p<K, MR, 5>(q, abuf, ptrs, rt, y, H, I, gm, max_act);
}

template <int K>
void run(sycl::queue& q, int MR, const fp16* x, const int64_t* pp, Route rt, fp16* ab, float* y, int H, int I,
         const Geom& gm, int max_act, float lim) {
    if (MR == 1) { gateup<K, 1>(q, x, H, pp, rt, ab, H, I, gm, max_act, lim); down<K, 1>(q, ab, pp, rt, y, H, I, gm, max_act); }
    else if (MR == 2) { gateup<K, 2>(q, x, H, pp, rt, ab, H, I, gm, max_act, lim); down<K, 2>(q, ab, pp, rt, y, H, I, gm, max_act); }
    else { gateup<K, 4>(q, x, H, pp, rt, ab, H, I, gm, max_act, lim); down<K, 4>(q, ab, pp, rt, y, H, I, gm, max_act); }
}

// K codes the routed-expert kernels are built for: GLM-5.3 (4, 2.5, 3.5), Qwen3.8-Flash-Next (3)
constexpr bool kc_built(int kc) { return kc == 3 || kc == 4 || kc == 25 || kc == 35; }

void run_any(sycl::queue& q, int kc, int MR, const fp16* x, const int64_t* pp, Route rt, fp16* ab, float* y, int H,
             int I, const Geom& gm, int max_act, float lim) {
    switch (kc) {
        case 3: run<3>(q, MR, x, pp, rt, ab, y, H, I, gm, max_act, lim); break;
        case 4: run<4>(q, MR, x, pp, rt, ab, y, H, I, gm, max_act, lim); break;
        case 25: run<25>(q, MR, x, pp, rt, ab, y, H, I, gm, max_act, lim); break;
        default: run<35>(q, MR, x, pp, rt, ab, y, H, I, gm, max_act, lim); break;
    }
}
}  // namespace

size_t moe_record_bytes(int H, int I, int kc) { return Geom{H, I, kc}.blob(); }

void moe_forward(sycl::queue& q, Workspace& ws, const half* x, int M, const int* sel, const float* w, int topk,
                 const int64_t* ptrs, int H, int I, int kc, int E, half* out, float act_lim) {
    GLM_CHECK(M >= 1, "moe: M=%d", M);
    if (M >= 256) return moe_prefill(q, ws, x, M, sel, w, topk, ptrs, H, I, kc, E, out, act_lim);
    GLM_CHECK(kc_built(kc), "moe: K code %d not built (3, 4, 25, 35 are)", kc);
    GLM_CHECK(E <= 1024 && H % 128 == 0 && I % 128 == 0, "moe: shape");
    ws.reset();
    Geom gm{H, I, kc};
    int npairs = M * topk;
    bool pair = M <= kPairMaxM;
    Route rt{};
    const int* pair_of = nullptr;
    int max_act = pair ? npairs : std::min(npairs, E);
    if (pair) {
        rt = Route{sel, nullptr, nullptr, nullptr, nullptr, nullptr, w, topk, npairs};
    } else {
        int A = std::min(npairs, E);
        int* ib = ws.take<int>(4 * A + 1 + 2 * npairs + 16);
        float* pw = ws.take<float>(npairs);
        int *ae = ib, *as = ib + A, *ac = ib + 2 * A, *aps = ib + 3 * A, *na = ib + 4 * A;
        int* pt = ib + 4 * A + 1;
        int* po = pt + npairs;
        moe_align(q, sel, w, npairs, topk, E, 1, ae, as, ac, na, aps, pt, pw, po);
        rt = Route{ae, as, ac, na, aps, pt, pw};
        pair_of = po;
    }
    fp16* ab = ws.take<fp16>((size_t)npairs * I);
    float* y = ws.take<float>((size_t)npairs * H);
    int MR = pair ? 1 : (M <= 64 ? 2 : 4);
    if (kc == 4 && MR > 2) MR = 2;     // K=4 MR=4 gave wrong output on the B70 (the PyTorch stack's note)
    run_any(q, kc, MR, x, ptrs, rt, ab, y, H, I, gm, max_act, act_lim);
    int n = M * (H / 128), local = 8;
    CombineKernel<fp16> ck{y, pair_of, out, M, H, topk, H};
    q.submit([&](sycl::handler& h) { h.parallel_for(sycl::nd_range<1>((n + local - 1) / local * local, local), ck); });
}


MoeSplit moe_split_begin(sycl::queue& q, Workspace& ws, const half* x, const int* sel_p, const float* w_p,
                         const int* pair_of, int topk, int n0, const int64_t* ptrs, int H, int I, int kc, int E,
                         float act_lim) {
    GLM_CHECK(kc_built(kc), "moe: K code %d not built (3, 4, 25, 35 are)", kc);
    GLM_CHECK(n0 >= 0 && n0 <= topk, "moe split: %d of %d", n0, topk);
    ws.reset();
    MoeSplit s;
    s.sel_p = sel_p;
    s.w_p = w_p;
    s.pair_of = pair_of;
    s.n0 = n0;
    s.topk = topk;
    s.act_lim = act_lim;
    s.ab = ws.take<fp16>((size_t)topk * I);
    s.y = ws.take<float>((size_t)topk * H);
    if (n0) {
        Route rt{sel_p, nullptr, nullptr, nullptr, nullptr, nullptr, w_p, topk, n0};
        run_any(q, kc, 1, x, ptrs, rt, s.ab, s.y, H, I, Geom{H, I, kc}, n0, act_lim);
    }
    return s;
}

void moe_split_end(sycl::queue& q, const MoeSplit& s, const half* x, const int64_t* ptrs, int H, int I, int kc, int E,
                   half* out) {
    int n1 = s.topk - s.n0;
    if (n1) {
        // positions n0 .. topk - 1: a launch over pairs 0 .. n1 - 1 of the shifted arrays (M = 1: every pair's token is 0)
        Route rt{s.sel_p + s.n0, nullptr, nullptr, nullptr, nullptr, nullptr, s.w_p + s.n0, s.topk, n1};
        run_any(q, kc, 1, x, ptrs, rt, s.ab + (size_t)s.n0 * I, s.y + (size_t)s.n0 * H, H, I, Geom{H, I, kc}, n1,
                s.act_lim);
    }
    int n = H / 128, local = 8;
    CombineKernel<fp16> ck{s.y, s.pair_of, out, 1, H, s.topk, H};
    q.submit([&](sycl::handler& h) { h.parallel_for(sycl::nd_range<1>((n + local - 1) / local * local, local), ck); });
}

}  // namespace glm
