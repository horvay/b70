#pragma once
// SYCL kernels for the Qwen3.8-Flash-Next blocks (docs/qwen4-spec.md), raw device pointers in, row-major:
// hyper-connection sites (§4), Gated DeltaNet (§6), QSA attention (§7), routing (§8) and the PLE (§5.4).
#include "q4.h"

namespace glm::q4ops {

// --- §4 gated-residual sites. X: [rows][4][D] fp32 streams
// n16[r, s*D + d] = fp16(rms(X[r, s]) * w1p[s*D + d])
void site_norm(sycl::queue& q, const float* X, const float* w1p, half* n16, int rows);
// t [rows, nd] (n16 @ [down | inject]^T): a16 = fp16(silu(t[:320] / 4)); inj = 2 * sigmoid(t[320 + s] / 4) (nd = 324)
void site_gates(sycl::queue& q, const float* t, int nd, half* a16, float* inj, int rows);
// y[r, d] = fp16(mean_s sigmoid(gpre[r, s*D + d]) * n16[r, s*D + d])
void site_combine(sycl::queue& q, const float* gpre, const half* n16, half* y, int rows);
// X[r, s, :] += inj[r, s] * o[r, :]
void site_apply(sycl::queue& q, float* X, const float* inj, const float* o, int rows);
// X[r, s, :] = e[r, :]
void expand(sycl::queue& q, const float* e, float* X, int rows);

// --- §6 Gated DeltaNet. ab [rows, 96] (b | a); beta = bf16(sigmoid(b)), g = neg_a * softplus(a + dt_bias)
void gdn_gates(sycl::queue& q, const float* ab, const float* neg_a, const float* dtb, float* beta, float* g, int rows);
// depthwise causal conv (kernel 4) + SiLU over qkv fp32 [rows, 10240] (inputs rounded to bf16), window state bf16
// [10240][3] updated; out bf16 [rows, 10240]
void gdn_conv(sycl::queue& q, const float* qkv, bf16* state, const bf16* w, bf16* out, int rows);
// delta rule over rows tokens: rec fp32 [48][128 k][128 v] updated in place; core bf16 [rows, 48, 128]
void gdn_delta(sycl::queue& q, const bf16* c, const float* beta, const float* g, float* rec, bf16* core, int rows);
// out[r, j, :] = fp16(rms(core[r, j]) * w * sigmoid(z[r, j, :]))
void gdn_norm(sycl::queue& q, const bf16* core, const float* z, const float* w, half* out, int rows);

// --- §7 QSA attention, rows at positions p0 .. p0 + rows - 1
// qg fp16 [rows, 24 x (256 query | 256 gate)], k, v fp16 [rows, 512], ik fp16 [rows, 640] (4 indexer q heads | raw key).
// Writes qn16 [rows, 24, 256] (normed, roped), K/V cache rows, qi fp32 [rows, 4, 128] (normed, roped), rk cache rows.
// pdev (decode graphs): p0 read from device memory at run time instead of the argument
void attn_prep(sycl::queue& q, const half* qg, const half* k, const half* v, const half* ik, const float* qn,
               const float* kn, const float* iqn, int rows, int p0, half* qn16, half* Kc, half* Vc, float* qi, half* rk,
               const int* pdev = nullptr);
// pooled keys for blocks b0 .. b1 - 1: rope(Norm1p(fp16(mean(rk[4b .. 4b+3])), ikn), 4b)
void attn_pool(sycl::queue& q, const half* rk, const float* ikn, half* pk, int b0, int b1);
// one decode token at position *pdev: the block it completes, if any
void attn_pool_dec(sycl::queue& q, const half* rk, const float* ikn, half* pk, const int* pdev);
// rows' token lists: dense rows (p <= 2050) get cnt = -1 (all of 0 .. p); others the top-512 blocks by
// sum_h relu(qi[h] . pk[b]) / sqrt(128) (ties to the lower block) plus the tail 4 nb .. p. sc: scratch
// [rows, nbcap] fp32 (nbcap >= the pools any row sees). list int32 [rows, NSEL]
void attn_select(sycl::queue& q, const float* qi, const half* pk, int rows, int p0, int nbcap, float* sc, int* list,
                 int* cnt, const int* pdev = nullptr);
// softmax(q k / 16) v over each row's tokens per head (head h reads kv head h / 12), times sigmoid(gate):
// o16 [rows, 24 * 256]. part: scratch, attn_part_floats(rows) floats
size_t attn_part_floats(int rows);
void attn_core(sycl::queue& q, const half* qn16, const half* Kc, const half* Vc, const int* list, const int* cnt,
               const half* qg, int rows, int p0, float* part, half* o16, const int* pdev = nullptr);

// --- §8 routing: logits fp16 [rows, stride] (the first 512 columns): top-10 by logit (ties to the lower index),
// w = softmax over the ten
void route(sycl::queue& q, const half* rl, int stride, int rows, int* sel, float* w);
// per-expert pick counts of one layer: use[e] += picks; stats[0] += picks, stats[1] += picks of RAM-resident experts
void count_use(sycl::queue& q, const int* sel, int rows, int* use, const uint8_t* in_vram, unsigned long long* stats);
// a = fp16(silu(g) * u)
void swiglu(sycl::queue& q, const half* g, const half* u, half* a, size_t n);
// out = fp32(moe) + sigmoid(rl[r, 512]) * sh
void moe_combine(sycl::queue& q, const half* moe, const float* sh, const half* rl, int stride, float* out, int rows);

// --- §5.4 PLE. kv fp32 [rows, 12800] (key | value) -> gv fp32 [rows, 4D] (sigmoid gate x value), cn fp16 normed conv
// input; then the dilated causal conv (taps t-9, t-6, t-3, t) with state fp16 [4D][9] and X += gv + silu(conv)
void ple_gate(sycl::queue& q, const float* kv, const float* X, const float* nk, const float* nq, const float* nc,
              float* gv, half* cn, int rows);
void ple_conv(sycl::queue& q, const half* cn, const float* gv, const float* w, half* state, float* X, int rows);

}  // namespace glm::q4ops
