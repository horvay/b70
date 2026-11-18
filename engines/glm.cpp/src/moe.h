#pragma once
// Routed experts: out[m] = fp16( sum_k w[m,k] * expert_{sel[m,k]}(x[m]) ) over a device pointer table (one record per
// expert, VRAM or host memory read zero-copy), docs/forward-spec.md §6.5. Store record layout: kernels/exl3_moe_kernels.h
// Geom (gate|up trellis, down trellis, suh_g, suh_u, svh_g, svh_u, suh_d, svh_d).
#include "rt.h"

namespace glm {

constexpr int kcode(double K) { return K == (int)K ? (int)K : 10 * (int)K + 5; }   // 2.5 -> 25, 3.5 -> 35
size_t moe_record_bytes(int H, int I, int kc);

// x fp16 [M, H]; sel int32 [M, topk]; w fp32 [M, topk]; ptrs int64 [E] (device addresses of the records)
void moe_forward(sycl::queue& q, Workspace& ws, const half* x, int M, const int* sel, const float* w, int topk,
                 const int64_t* ptrs, int H, int I, int kc, int E, half* out, float act_lim = 10.f);

// Decode (M = 1) in two passes, bit-identical to moe_forward: the experts of the first n0 positions of a permutation of
// the picks now (their records are readable), the rest at moe_split_end (after their reads land), then the combine in
// the original pick order. sel_p / w_p / pair_of: ops::permute_picks's output (device, topk each). The pointer rows of
// the second pass's experts may change in between (in queue order).
struct MoeSplit {
    const int* sel_p = nullptr; const float* w_p = nullptr; const int* pair_of = nullptr;
    half* ab = nullptr; float* y = nullptr;
    int n0 = 0, topk = 0;
    float act_lim = 10.f;
};
MoeSplit moe_split_begin(sycl::queue& q, Workspace& ws, const half* x, const int* sel_p, const float* w_p,
                         const int* pair_of, int topk, int n0, const int64_t* ptrs, int H, int I, int kc, int E,
                         float act_lim = 10.f);
void moe_split_end(sycl::queue& q, const MoeSplit& s, const half* x, const int64_t* ptrs, int H, int I, int kc, int E,
                   half* out);

}  // namespace glm
