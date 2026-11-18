// Host launchers for the routed-expert kernels, without torch: a port of moe_forward_impl from
// llm/glm/xpu/csrc/exl3_moe.n128.sycl with the PyTorch stack's settings (glm_xpu/ops.py moe_ops: gate|up k-splits 8,
// down 4, pair mode for M = 1, grouped GEMV above). The DPAS prompt path (M >= 256) is not ported yet.
#include "../src/moe.h"
#include <chrono>
#include <vector>
#include <cstdio>
#include "../kernels/exl3_moe_kernels.h"

namespace glm {

using namespace glm_moe;
using fp16 = sycl::half;
namespace review_trace {
struct Rec { sycl::event ev; int phase; };
static bool active = false;
static std::vector<Rec> events;
static double host_s = 0;
void add(sycl::event ev, int phase) { if (active) events.push_back({ev,phase}); }
}
extern "C" void review_trace_reset() { review_trace::events.clear(); review_trace::host_s=0; review_trace::active=true; }
extern "C" void review_trace_report(int tokens) {
 using namespace review_trace;
 double ns[3]={}; double gap=0; uint64_t prev=0; int calls=0;
 for(auto &r:events) {
  auto st=r.ev.get_profiling_info<sycl::info::event_profiling::command_start>();
  auto en=r.ev.get_profiling_info<sycl::info::event_profiling::command_end>();
  ns[r.phase]+=en-st;
  if (r.phase==0) {prev=0; ++calls;}
  if(prev) gap += st-prev;
  prev=en;
 }
 fprintf(stderr,"REVIEW_MOE tokens=%d calls=%d gu=%.4f dn=%.4f combine=%.4f internal_gap=%.4f host_submit=%.4f ms/token\n",tokens,calls,ns[0]*1e-6/tokens,ns[1]*1e-6/tokens,ns[2]*1e-6/tokens,gap*1e-6/tokens,host_s*1e3/tokens);
 events.clear(); active=false;
}

void moe_align(sycl::queue& q, const int* ids, const float* w, int npairs, int topk, int E, int pad, int* act_expert,
               int* act_start, int* act_cnt, int* n_act, int* act_pstart, int* pair_tok, float* pair_w, int* pair_of,
               int* blk_a = nullptr, int* n_blk = nullptr);
// prompt path (M >= 256): libglm_moe_pf.so, built with large-GRF code generation
void moe_prefill(sycl::queue& q, Workspace& ws, const half* x, int M, const int* sel, const float* w, int topk,
                 const int64_t* ptrs, int H, int I, int kc, int E, half* out);

namespace {
constexpr int kGuP = 8, kDnP = 4, kPairMaxM = 1;

template <int K, int MR>
void gateup(sycl::queue& q, const fp16* x, int xs, const int64_t* ptrs, Route rt, fp16* abuf, int H, int I,
            const Geom& gm, int max_act) {
    GateUpKernel<K, MR, kGuP, fp16> k{x, xs, ptrs, rt, abuf, H, I, gm.off_suh_g(), gm.off_svh_g(), gm.off_suh_d(),
                                      (int)((gm.blob() - gm.off_suh_g()) / 4)};
    constexpr int T = 4 * kGuP;
    size_t n = (size_t)max_act * (I / 128) * T;
    review_trace::add(q.submit([&](sycl::handler& h) { h.parallel_for(sycl::nd_range<1>(n, T), k); }), 0);
}

template <int K, int MR>
void down(sycl::queue& q, const fp16* abuf, const int64_t* ptrs, Route rt, float* y, int H, int I, const Geom& gm,
          int max_act) {
    constexpr int RB = DownKernel<K, MR, kDnP>::RB;
    GLM_CHECK((I / 16) % kDnP == 0 && ((I / 16) / kDnP) % RB == 0, "moe down: I/16 split");
    DownKernel<K, MR, kDnP> k{abuf, ptrs, rt, y, H, I, gm.gu_bytes(), gm.off_svh_d()};
    constexpr int T = 2 * kDnP;
    size_t n = (size_t)max_act * (H / 128) * T;
    review_trace::add(q.submit([&](sycl::handler& h) { h.parallel_for(sycl::nd_range<1>(n, T), k); }), 1);
}

template <int K>
void run(sycl::queue& q, int MR, const fp16* x, const int64_t* pp, Route rt, fp16* ab, float* y, int H, int I,
         const Geom& gm, int max_act) {
    if (MR == 1) { gateup<K, 1>(q, x, H, pp, rt, ab, H, I, gm, max_act); down<K, 1>(q, ab, pp, rt, y, H, I, gm, max_act); }
    else if (MR == 2) { gateup<K, 2>(q, x, H, pp, rt, ab, H, I, gm, max_act); down<K, 2>(q, ab, pp, rt, y, H, I, gm, max_act); }
    else { gateup<K, 4>(q, x, H, pp, rt, ab, H, I, gm, max_act); down<K, 4>(q, ab, pp, rt, y, H, I, gm, max_act); }
}
}  // namespace

size_t moe_record_bytes(int H, int I, int kc) { return Geom{H, I, kc}.blob(); }

void moe_forward(sycl::queue& q, Workspace& ws, const half* x, int M, const int* sel, const float* w, int topk,
                 const int64_t* ptrs, int H, int I, int kc, int E, half* out) {
    GLM_CHECK(M >= 1, "moe: M=%d", M);
    if (M >= 256) return moe_prefill(q, ws, x, M, sel, w, topk, ptrs, H, I, kc, E, out);
    GLM_CHECK(kc == 4 || kc == 25 || kc == 35, "moe: K code %d not built (4, 25, 35 are)", kc);
    GLM_CHECK(E <= 1024 && H % 128 == 0 && I % 128 == 0, "moe: shape");
    auto review_t0 = std::chrono::steady_clock::now();
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
    if (kc == 4) run<4>(q, MR, x, ptrs, rt, ab, y, H, I, gm, max_act);
    else if (kc == 25) run<25>(q, MR, x, ptrs, rt, ab, y, H, I, gm, max_act);
    else run<35>(q, MR, x, ptrs, rt, ab, y, H, I, gm, max_act);
    int n = M * (H / 128), local = 8;
    CombineKernel<fp16> ck{y, pair_of, out, M, H, topk, H};
    review_trace::add(q.submit([&](sycl::handler& h) { h.parallel_for(sycl::nd_range<1>((n + local - 1) / local * local, local), ck); }), 2);
    if (review_trace::active) review_trace::host_s += std::chrono::duration<double>(std::chrono::steady_clock::now()-review_t0).count();
}

}  // namespace glm
