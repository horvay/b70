#pragma once
// SYCL kernels for Gemma 4 (g4.h), raw device pointers, row-major. Row r of a call sits at position
// pdev ? pdev[r] : p0 + r (pdev: a recorded decode graph reads the positions at run time).
#include "g4.h"

namespace glm::g4ops {

// y = fp16(rms(x) * w), x fp32 [R, D], w bf16
void norm16(sycl::queue& q, const float* x, const bf16* w, half* y, int R, int dim = g4::D);
// y = fp16(rms(x) * w), x fp16 [R, dim], in place allowed
void add_norm_rows(sycl::queue& q, const half* x, const bf16* w, half* y, int R, int dim = g4::D);
// x = (x + rms(a) * wa) * scale (fp32), then y = fp16(rms(x) * wn) when wn is set; a fp16 or fp32 [R, D].
// y may alias a (each row is read whole before it is written).
template <class TA>
void add_norm(sycl::queue& q, float* x, const TA* a, const bf16* wa, float scale, const bf16* wn, half* y, int R,
              int dim = g4::D);

// cos, sin of the sliding layers' 128 RoPE pairs at each row's position: tab fp32 [R, 128, 2] (once per step: every
// sliding layer's rows sit at the same positions)
void rope_rows(sycl::queue& q, int R, const int* pdev, int p0, float* tab);
// Sliding layer, after the projections: q [R, 32, 256] in place = fp16(rope(fp16(rms(q) * q_norm))); k, v [R, 16, 256]
// -> K = fp16(rope(fp16(rms(k) * k_norm))), V = fp16(rms(v)) written to row dst(r) of Kd / Vd ([*, 16, 256]):
// ring > 0: slot pos % ring of the row's game (sdev; null: game 0; a game has ring rows); ring == 0: row r (in place is
// fine).
void sliding_post(sycl::queue& q, half* qb, const half* kb, const half* vb, const bf16* qn, const bf16* kn,
                  const float* srope, int R, const int* pdev, int p0, half* Kd, half* Vd, int ring, const int* sdev = nullptr);
// Full layer: q [R, 32, 512] in place = fp16(rope(fp16(rms(q) * q_norm))) (proportional RoPE from rope [T, 64, 2]);
// P = k [R, 4, 512] copied to Pc[pos], Rc[pos][h] = 1 / rms(P head), Kr[pos][h] = fp16(rope(P * k_norm)), its 128 rotated
// dims (j | j + 256, j < 64; attention multiplies by r); sdev: each row's game, a game has TS rows of cache
void full_post(sycl::queue& q, half* qb, const half* kb, const bf16* qn, const bf16* kn, const float* rope, int R,
               const int* pdev, int p0, half* Pc, half* Kr, float* Rc, const int* sdev = nullptr, int TS = 0);
// Full layer keys k0 .. k0 + n - 1 rebuilt for a prompt: K = fp16(rope(rms(P) * k_norm)), V = fp16(rms(P)),
// [n, 4, 512] each
void full_kv(sycl::queue& q, const half* Pc, const float* Rc, const bf16* kn, const float* rope, int k0, int n,
             half* K, half* V);

// The drafter's query heads alone (it reads the target's K / V): as the query part of sliding_post / full_post
void sliding_q(sycl::queue& q, half* qb, const bf16* qn, const float* srope, int R);
void full_q(sycl::queue& q, half* qb, const bf16* qn, const float* rope, int R, const int* pdev);

// Decode / verify attention (split over keys, then combined), o16 [R, 32, D]. part: attn_part_floats(R) floats.
size_t attn_part_floats(int R);
// sliding: K / V ring [RING, 16, 256], keys max(0, p - 1023) .. p
void attn_sliding_dec(sycl::queue& q, const half* q16, const half* K, const half* V, int R, const int* pdev, const int* sdev,
                      int p0, float* part, half* o16);
// full: keys 0 .. p from the cache (kernels/g4_dec.h); kn: k_norm fp16 [512]
void attn_full_dec(sycl::queue& q, const half* q16, const half* Pc, const half* Kr, const float* Rc, const half* kn, int R,
                   const int* pdev, const int* sdev, int TS, int p0, float* part, half* o16);

// ring rows <-> a prompt's contiguous K / V ([*, 16, 256])
// dst[i] = ring[(p_first + i) % RING] for i < n
void ring_gather(sycl::queue& q, const half* ring, int p_first, int n, half* dst);
// ring[(p_first + i) % RING] = src[i] for i < n
void ring_scatter(sycl::queue& q, const half* src, int p_first, int n, half* ring);

// m = fp16(fp16(gelu_tanh(g)) * u), in place over g allowed
void gelu_mul(sycl::queue& q, const half* g, const half* u, half* m, size_t n);
// logits fp16 [Q, V] = softcap(x [Q, D] @ W^T), W fp16 [V, D]
void head_logits(sycl::queue& q, const half* x, const half* W, int Q, half* out);
// in place: l = 30 * tanh(l / 30)
void softcap(sycl::queue& q, half* l, size_t n);
// z [R, 2 D] = a [R, D] | b [R, D]
void concat(sycl::queue& q, const half* a, const half* b, half* z, int R, int D);
// fp32 attention accumulators -> fp16 rows of o (row stride 32 * D)
void f32_to_f16(sycl::queue& q, const float* a, half* b, size_t n);
void fill(sycl::queue& q, float* a, float v, size_t n);

}  // namespace glm::g4ops

namespace glm::g4fa {
// Prompt attention (ESIMD, DPAS), kernels/g4_fa.h. Query row i (q [Lq, 32, D], row stride 32 * D) sits at position
// qbase + i; key j (k / v [Lk, Hk, D]) at kbase + j; a row sees keys at or before its position, and within the last
// win positions when win > 0.
// sliding (D = 256, Hk = 16): normalised fp16 out (o may alias q)
void sliding(sycl::queue& q, const half* qb, const half* k, const half* v, int Lq, int Lk, int qbase, int kbase,
             int win, half* o);
// full (D = 512, Hk = 4), one key block: merged into the running fp32 o32 [Lq, 32, 512] and lse [Lq, 32]
// (natural log; start from 0 and -inf)
void full_block(sycl::queue& q, const half* qb, const half* k, const half* v, int Lq, int Lk, int qbase, int kbase,
                float* o32, float* lse);
// decode pieces (kernels/g4_dec.h): partial m, l, O per row, query head, piece
void sliding_dec(sycl::queue& q, const half* qb, const half* K, const half* V, int R, const int* pdev, const int* sdev, int p0,
                 float* pm, float* pl, float* po, int ns, int blocks);
void full_dec(sycl::queue& q, const half* qb, const half* Pc, const half* Kr, const float* Rc, const half* kn, int R,
              const int* pdev, const int* sdev, int TS, int p0, float* pm, float* pl, float* po, int ns);
// y [Q, N] = x [Q, K] (row stride xs) @ W^T, W fp16 [N, K], Q <= 8, DPAS; softcap: 30 tanh(y / 30)
void dense16(sycl::queue& q, const half* x, int Q, int xs, const half* w, int K, int N, half* y, bool softcap = false);
void dense32(sycl::queue& q, const half* x, int Q, int xs, const half* w, int K, int N, float* y);
}  // namespace glm::g4fa
