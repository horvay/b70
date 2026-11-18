#pragma once
// SYCL kernels for Ripple-65B (rp.h), raw device pointers, row-major. Row r of a call sits at position
// pdev ? pdev[r] : p0 + r (pdev: a recorded decode graph reads the positions at run time).
#include "rp.h"

namespace glm::rpops {

// cos, sin of the 64 RoPE pairs at each row's position: tab fp32 [R, 64, 2]
void rope_rows(sycl::queue& q, int R, const int* pdev, int p0, float* tab);
// After the projections: q [R, 80, 128] in place = fp16(rope(fp16(rms(q) * q_norm))); k, v [R, 8, 128] -> K = fp16(rope(fp16(rms(k)
// * k_norm))), V = v, written to Kd / Vd rows: span > 0: game(sdev; null: 0) * span + (ring ? p % ring : p); span == 0: row r
// (in place is fine). rope null: a global layer. qb null: K and V only.
void post(sycl::queue& q, half* qb, const half* kb, const half* vb, const bf16* qn, const bf16* kn, const float* rope, int R,
          const int* pdev, int p0, half* Kd, half* Vd, int span, bool ring, const int* sdev);
// y = fp16(lerp(e, h, w)) [R, D]
void lerp(sycl::queue& q, const float* e, const half* h, float w, half* y, int R);
// o = fp16(o * fp16(sigmoid(g))), n values
void gate_mul(sycl::queue& q, half* o, const half* g, size_t n);
// m = fp16(fp16(silu(min(g, 20))) * clamp(u, +-20)), in place over g allowed
void silu_mul(sycl::queue& q, const half* g, const half* u, half* m, size_t n);

// y [R, D] = fp16(x [R, D] fp32), a row of the target's stream as the drafter's fc reads it (y row stride ys)
void to16(sycl::queue& q, const float* x, half* y, int R, int ys);
// DFlash2's grouped dynamic convolution over a draft block: out[t] = sum_tap (base[tap] + dyn[t][tap]) * x[t - tap], tap < 4, t - tap >= 0
// base bf16 [4, D]; dyn fp16 [R, 4, 320] with row stride dys; out fp16 [R, D] (x fp16: prepare), or added to res (x fp32: finish)
void dynconv_prepare(sycl::queue& q, const half* x, const half* dyn, int dys, const bf16* base, half* out, int R);
void dynconv_finish(sycl::queue& q, const float* x, const half* dyn, int dys, const bf16* base, float* res, int R);
// The drafter's MLP: no clamps, and silu(g) * u reaches 1e5 and more where fp16 stops at 65K (an inf is a NaN out of down_proj). So
// m = fp16(fp16(silu(g)) * u * scale[r]), scale[r] a power of 2 putting the row's largest value under 2^14 (never above 1), R rows of I;
// mul_rows brings down_proj's output back: y [R, D] /= scale[r]
void silu_scaled(sycl::queue& q, const half* g, const half* u, half* m, float* scale, int R);
void mul_rows(sycl::queue& q, float* y, const float* scale, int R);
// the drafter's attention: its rings of span slots, a window of win (a block's rows also see the block's later rows), o16 [R, 80, 128];
// ns pieces of `blocks` blocks: span = ns * blocks * 32
void attn_draft(sycl::queue& q, const half* q16, const half* K, const half* V, int R, const int* pdev, const int* sdev, int span, int win,
                int blocks, float* part, half* o16);

// Decode / verify attention (split over keys, then combined), o16 [R, 80, 128]. part: attn_part_floats(R) floats.
size_t attn_part_floats(int R);
void attn_sliding(sycl::queue& q, const half* q16, const half* K, const half* V, int R, const int* pdev, const int* sdev, float* part, half* o16);
void attn_global(sycl::queue& q, const half* q16, const half* K, const half* V, int R, const int* pdev, const int* sdev, int TS, float* part, half* o16);

// ring rows <-> a prompt's contiguous K / V ([*, 8, 128])
void ring_gather(sycl::queue& q, const half* ring, int p_first, int n, half* dst);
void ring_scatter(sycl::queue& q, const half* src, int p_first, int n, half* ring);

}  // namespace glm::rpops

namespace glm::rpfa {
// Prompt attention (kernels/g4_fa.h with Ripple's score scale): query row i (q [Lq, 80, 128]) at position qbase + i, key j (k / v
// [Lk, 8, 128]) at kbase + j; a row sees keys at or before its position, within the last win positions when win > 0; o may alias q
void prompt(sycl::queue& q, const half* qb, const half* k, const half* v, int Lq, int Lk, int qbase, int kbase, int win, half* o);
// decode pieces (kernels/rp_dec.h): partial m, l, O per row, query head, piece
void dec_sliding(sycl::queue& q, const half* qb, const half* K, const half* V, int R, const int* pdev, const int* sdev, int p0,
                 float* pm, float* pl, float* po, int ns, int blocks);
void dec_global(sycl::queue& q, const half* qb, const half* K, const half* V, int R, const int* pdev, const int* sdev, int p0,
                int TS, float* pm, float* pl, float* po, int ns);
void dec_draft(sycl::queue& q, const half* qb, const half* K, const half* V, int R, const int* pdev, const int* sdev, int span, int win,
               float* pm, float* pl, float* po, int ns, int blocks);
}  // namespace glm::rpfa
