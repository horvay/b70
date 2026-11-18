#pragma once
// SYCL kernels for Qwen3.8-27B (docs/qwen35-spec.md), raw device pointers, row-major. The Gated DeltaNet's gates, conv
// and delta rule are q4_ops'; the sizes are the same.
#include "q35.h"

namespace glm::q35ops {

// y[r, d] = fp16(rms(x[r]) * w1p[d]), x fp32 [rows, D]
void norm16(sycl::queue& q, const float* x, const float* w1p, half* y, int rows);
// x += o, fp32 [rows, D]
void add(sycl::queue& q, float* x, const float* o, int rows);
// out[r, j, :] = fp16(rms(core[r, j]) * w * silu(z[r, j, :])): this model's output gate is swish
void gdn_norm(sycl::queue& q, const bf16* core, const float* z, const float* w, half* out, int rows);

// One token per row, each row its own sequence sdev[r] (decode): the conv window bf16 [seqs][10240][3], and the delta
// rule's rec fp32 [seqs][48][128][128]. Prompts use q4_ops', whose rows run in order over one sequence.
void gdn_conv_rows(sycl::queue& q, const float* qkv, bf16* state, const bf16* w, bf16* out, int rows, const int* sdev);
void gdn_delta_rows(sycl::queue& q, const bf16* c, const float* beta, const float* g, float* rec, bf16* core, int rows,
                    const int* sdev);

// Decode rows grouped by sequence (MTP verify: a sequence's rows together and in order). desc [rows][3]: a group's first
// row holds {rows in the group, state slot read, state slot written}, every other row {0, -, -}. A group's rows run in
// order from the read slot's state, the state after its last row goes to the written slot (the same slot: in place).
// The arithmetic of each row is gdn_conv_rows' / gdn_delta_rows' exactly, so a group of n rows gives what n steps of one
// row do.
void gdn_conv_seq(sycl::queue& q, const float* qkv, bf16* state, const bf16* w, bf16* out, int rows, const int* desc);
void gdn_delta_seq(sycl::queue& q, const bf16* c, const float* beta, const float* g, float* rec, bf16* core, int rows,
                   const int* desc);

// y[r * ys + d] = fp16(rms(x[r]) * w1p[d]), x fp32 or fp16 [rows, D] (MTP: the two halves of fc's input)
template <class T> void norm_rows(sycl::queue& q, const T* x, const float* w1p, half* y, int ys, int rows);
// ring[(sdev[r] * RING + pdev[r] % RING) * D ..] = src[r * D ..] (the post-norm hidden of each decode row, for MTP)
void put_rows(sycl::queue& q, const half* src, int rows, half* ring, const int* pdev, const int* sdev, int RING);
// out[r] = ring[src[r]] (src < MH), mh[src[r] - MH] (src >= MH) or zeros (src < 0), fp16 [D] rows (MTP inputs)
constexpr int MH = 1 << 20;
void gather_hidden(sycl::queue& q, const int* src, int rows, const half* ring, const half* mh, half* out);
// out[2 r] = argmax of logits row r (ties to the lower index), out[2 r + 1] = bits of its softmax probability
void argmax_rows(sycl::queue& q, const half* logits, int rows, int n, int* out);

// qg fp16 [rows, 24 x (256 query | 256 gate)], k, v fp16 [rows, 4 * 256]. Writes qn16 [rows, 24, 256] and K / V cache
// rows, normed and roped at pid [rows, 3] (T, H, W). Row r sits at position p0 + r of the cache given, or (decode) at
// pdev[r] of sequence sdev[r], a sequence holding TS rows.
void attn_prep(sycl::queue& q, const half* qg, const half* k, const half* v, const float* qn, const float* kn, int rows,
               int p0, const int* pid, half* qn16, half* Kc, half* Vc, const int* pdev = nullptr, const int* sdev = nullptr,
               int TS = 0);
// softmax(q k / 16) v over each row's keys 0 .. p, per head (head h reads kv head h / 6), times sigmoid(gate): o16
// [rows, 24 * 256]. nch: key pieces of CH to run (rows' keys must fit); part: attn_part_floats(rows, nch) floats.
size_t attn_part_floats(int rows, int nch);
void attn_core(sycl::queue& q, const half* qn16, const half* Kc, const half* Vc, const half* qg, int rows, int p0, int nch,
               float* part, half* o16, const int* pdev = nullptr, const int* sdev = nullptr, int TS = 0);

}  // namespace glm::q35ops
