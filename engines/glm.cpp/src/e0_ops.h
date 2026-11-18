#pragma once
// SYCL kernels for Edge0 (docs/edge0-spec.md): Q8_0 linears and the latent MoE's NVFP4 experts.
#include "rt.h"

namespace glm {

// A Q8_0 weight [N, K] (rows = outputs) split for alignment: int8 values [N, K] and one fp16 scale per 32 [N, K / 32].
struct Q8W {
    Tensor qs, d;
    int N = 0, K = 0;
    size_t bytes() const { return qs.bytes() + d.bytes(); }
};

namespace e0ops {

constexpr int ELL = 1792, G = 8, GE = G * ELL, EI = 640;   // latent width, D/U groups, their concatenation, expert ff
constexpr int NV_ROW_GU = ELL / 64 * 36;                     // bytes of one gate/up row (1008)
constexpr int NV_ROW_D = EI / 64 * 36;                       // bytes of one down row (360)
constexpr size_t NV_DOWN_OFF = (size_t)2 * EI * NV_ROW_GU;   // 1,290,240
constexpr size_t NV_TAIL = NV_DOWN_OFF + (size_t)ELL * NV_ROW_D;   // 1,935,360: {s_gate, s_up, s_down, 0} fp32
constexpr size_t NV_REC = NV_TAIL + 16;                      // 1,935,376 bytes a record
constexpr int MV_MAX = 16;                                   // rows a matvec takes; more go through the GEMM path

// y[M, N] = x[M, K] . W^T, M <= MV_MAX. TX: half or float; TY: half or float
template <class TX, class TY>
void q8_matvec(sycl::queue& q, const TX* x, int M, const Q8W& w, TY* y);
// one row, groups of ELL skipped where gmask's bit is clear: gmode 1 = output rows grouped (D), 2 = input columns
// grouped (U). Skipped D outputs are left as they were (no pick reads them); skipped U inputs are zero anyway
template <class TX>
void q8_matvec_grouped(sycl::queue& q, const TX* x, const Q8W& w, float* y, const uint32_t* gmask, int gmode);
// *mask = OR over the picks of 1 << label[sel[p]]
void group_mask(sycl::queue& q, const int* sel, const int* label, int P, uint32_t* mask);
// out [N, K] fp16
void q8_dequant(sycl::queue& q, const Q8W& w, half* out);
// any M: the matvec for few rows, else dequantize into scratch (N * K halves) and one fp16 GEMM
template <class TY>
void q8_linear(sycl::queue& q, const half* x, int M, const Q8W& w, TY* y, half* scratch);

// --- the engine's expert record layout ("tiled"): each matrix (gate [640, 1792], up, down [1792, 640]) as tiles of 16
// rows x one 16-column sub-block, 144 B each: the 16 rows' UE4M3 scales, then their 8 code bytes each (as in the file:
// byte i = values i (low nibble) and i + 8). Tile (G, s) of a matrix with C columns at ((G * C / 16) + s) * 144. Same
// offsets of the matrices and the tail as the file's record; one tile is a DPAS B operand.
void nv_repack(const char* raw, char* out);         // host: the file's record -> tiled
// device: n records at src (device memory, file layout, NV_REC apart) -> tiled at dst[j] (device memory; a record
// for pinned RAM goes through VRAM and a copy: this kernel's small writes over PCIe crawl)
void nv_repack_dev(sycl::queue& q, const char* src, const int64_t* dst, int n);
constexpr int NV_TILE = 144;

// --- latent MoE, decode rows (M <= MV_MAX), tiled records: a work-group per (pick, 16 output rows), a lane a row.
// lead (null: every pick its own expert): picks sharing an expert are computed by its first pick's group, the
// weights decoded once. As nv_gate_up / nv_down otherwise
void nv_gate_up_t(sycl::queue& q, const float* xl, const int* sel, const int* lead, const int* label,
                  const int64_t* ptrs, int M, float* h);
void nv_down_t(sycl::queue& q, const float* h, const int* sel, const int* lead, const int64_t* ptrs, int M, float* yl);

// --- latent MoE, decode rows (M <= MV_MAX). sel/w [M, 10] (route), label [E] (this layer), ptrs [E] (records).
// h [M*10, 640] fp32 = silu(s_g * (G . x)) * (s_u * (U . x)), x = xl[r, label[e] * ELL ...] (fp32 [M, GE])
void nv_gate_up(sycl::queue& q, const float* xl, const int* sel, const int* label, const int64_t* ptrs, int M, float* h);
// yl [M*10, ELL] = s_d * (Down . h)
void nv_down(sycl::queue& q, const float* h, const int* sel, const int64_t* ptrs, int M, float* yl);
// several rows (verify windows): each distinct expert's weights read once for all picks of it. lead [M*10]: the
// first pick of the same expert (pick_leaders); then as nv_gate_up / nv_down
void pick_leaders(sycl::queue& q, const int* sel, int P, int* lead);
void nv_gate_up_g(sycl::queue& q, const float* xl, const int* sel, const int* lead, const int* label, const int64_t* ptrs,
                  int M, float* h);
void nv_down_g(sycl::queue& q, const float* h, const int* sel, const int* lead, const int64_t* ptrs, int M, float* yl);
// z [M, GE]: z[r, g*ELL + c] = sum over the row's picks k with label g of w[r, k] * yl[r*10 + k, c]
void latent_combine(sycl::queue& q, const float* yl, const int* sel, const float* w, const int* label, int M, float* z);

// --- latent MoE, prompt rows: one expert at a time over the rows routed to it
// rows[i]: the token row of the expert's i-th pick, wk[i]: its weight. A [n, width] = src[rows[i], col0 ...]
// (width a multiple of 8)
void gather_rows(sycl::queue& q, const half* src, int stride, int col0, const int* rows, int n, int width, half* A);
// out[rows[i], :] += wk[i] * O[i, :]   (O fp32 [n, width]; rows distinct within one call)
void scatter_rows(sycl::queue& q, const float* O, const int* rows, const float* wk, int n, int width, float* out);
// one tiled record -> gate|up fp16 [1280, ELL], down fp16 [ELL, 640] (no global scales)
void nv_dequant(sycl::queue& q, const char* rec, half* gu, half* dn);
// h16 [n, 640] = fp16(silu(s_g * GU[:, :640]) * (s_u * GU[:, 640:])), GU fp32 [n, 1280]; the scales read from rec
void nv_act(sycl::queue& q, const float* GU, const char* rec, int n, half* h16);
// z[rows[i], col0 + c] += wk[i] * Y[i, c] (Y fp32 [n, ELL]; rows distinct within one call)
void scatter_add(sycl::queue& q, const float* Y, const int* rows, const float* wk, int n, float* z, int col0);

// --- latent MoE, prompt rows in batches of experts (grouped GEMMs)
// Xb[i, :] = xl16[rows[i], col[i] .. col[i] + ELL)
void gather_picks(sycl::queue& q, const half* xl16, const int* rows, const int* col, int n, half* Xb);
// recs[j] (device array of tiled record addresses) -> gu [cnt][1280, ELL], dn [cnt][ELL, 640] fp16
void nv_dequant_batch(sycl::queue& q, const int64_t* recs, int cnt, half* gu, half* dn);
// h16 [n, 640] = fp16(silu(sgu[2i] * GU[i, :640]) * (sgu[2i+1] * GU[i, 640:]))
void nv_act_rows(sycl::queue& q, const float* GU, const float* sgu, int n, half* h16);

// --- latent MoE, prompt rows on the XMX units (e0_dpas.cpp): blocks of up to 64 rows of one expert, block b = rows
// br0[b] .. br0[b] + bn[b] - 1 of X / H / Y, its tiled record at brec[b]
// H [rows, 640] fp16 = silu(s_g * (X . G^T)) * (s_u * (X . U^T)), X [rows, ELL] fp16
// rows: X's (H's) row count (its 2D block loads read rows past it as zero)
void nv_gate_up_dpas(sycl::queue& q, const half* X, int rows, const int64_t* brec, const int* br0, const int* bn, int nb,
                     half* H);
// Y [rows, ELL] fp16 = s_d * (H . Down^T)
void nv_down_dpas(sycl::queue& q, const half* H, int rows, const int64_t* brec, const int* br0, const int* bn, int nb,
                  half* Y);

// out[row_ids[i], :] += sum over k in off[i] .. off[i+1]-1 of w[pick[k]] * O[pick[k], :] (fp32, width D), in that
// order: a row once per call however many of its picks the call holds (scatter_rows needs distinct rows)
void combine_rows(sycl::queue& q, const float* O, const int* row_ids, const int* off, const int* pick, const float* w,
                  int nrows, int width, float* out);

// --- int8 KV cache (QSA layers): K, V int8 [T, 2, 256] with an fp16 scale per 64 values [T, 2, 4]
// (absmax / 127; codes rounded against the stored fp16 scale, Strata's kv_q8). As q4ops::attn_prep / attn_core otherwise (same arguments, same partials).
void attn_prep_i8(sycl::queue& q, const half* qg, const half* k, const half* v, const half* ik, const float* qn,
                  const float* kn, const float* iqn, int rows, int p0, half* qn16, int8_t* Kq, half* Ks, int8_t* Vq,
                  half* Vs, float* qi, half* rk, const int* pdev = nullptr);
void attn_core_i8(sycl::queue& q, const half* qn16, const int8_t* Kq, const half* Ks, const int8_t* Vq, const half* Vs,
                  const int* list, const int* cnt, const half* qg, int rows, int p0, float* part, half* o16,
                  const int* pdev = nullptr);

// --- MTP draft layer (Q2_0 experts at full width, dense attention over its own fp16 cache)
// Q2_0 record (Strata's mtp_rt.py): codes of gate/up rows interleaved (2r gate r, 2r+1 up r) [1280][40][16], codes of
// down [2560][10][16], then the fp16 scales of the same rows [1280][40], [2560][10]; value = (code - 1) * d
constexpr size_t Q2_REC = 1382400;
// h [M*10, 640] = silu(G . y_r) * (U . y_r), y fp16 [M, 2560]
void q2_gate_up(sycl::queue& q, const half* y, const int* sel, const int64_t* ptrs, int M, float* h);
// out [M, 2560] fp32 = sum_k w[r, k] * (Down_k . h[r*10 + k])
void q2_down_sum(sycl::queue& q, const float* h, const int* sel, const float* w, const int64_t* ptrs, int M, float* out);
// rows of qg [rows, 24 x (256 | 256)], k, v [rows, 512]: q normed + roped into qn16 [rows, 24, 256] (when qg), K normed +
// roped and V into the fp16 cache at positions p0 + r
void mtp_prep(sycl::queue& q, const half* qg, const half* k, const half* v, const float* qn, const float* kn, int rows,
              int p0, half* qn16, half* Kc, half* Vc);
// one row at position p: softmax(q k / 16) v over cache positions 0..p (dense), times sigmoid(gate): o16 [24 * 256].
// part: mtp_attn_floats(p) floats
size_t mtp_attn_floats(int max_tokens);
void mtp_attn(sycl::queue& q, const half* qn16, const half* Kc, const half* Vc, const half* qg, int p, float* part,
              half* o16);
// y [rows, n] fp16 = rms(x[r]) * w (w carries the +1), x fp32
void rms_rows(sycl::queue& q, const float* x, const float* w, int rows, int n, half* y);
// R[t, s, :] = h[t * 4 + s, :] + e[t, :]   (fp32, D = 2560)
void add_streams(sycl::queue& q, const float* h, const float* e, int rows, float* R);
// out[0] = argmax over logits [n] (ties to the lower index), out[1] = bits of its softmax probability
void argmax_prob(sycl::queue& q, const half* logits, int n, int* out);

// the QSA pooled keys of blocks that rows at positions *pdev + r complete (q4ops::attn_pool_dec for several rows)
void attn_pool_rows(sycl::queue& q, const half* rk, const float* ikn, half* pk, const int* pdev, int rows);

// q4ops::route with a work-group of 512 a row (one lane an expert ranks itself): top-10 by logit, ties to the lower
// index, w = softmax over the ten
void route512(sycl::queue& q, const half* rl, int stride, int rows, int* sel, float* w);

// --- shared expert and output
// out[r, :] += sigmoid(rl[r * stride + 512]) * sh[r, :]
void add_shared(sycl::queue& q, float* out, const float* sh, const half* rl, int stride, int rows);
void f32_to_f16(sycl::queue& q, const float* x, half* y, size_t n);

}  // namespace e0ops
}  // namespace glm
