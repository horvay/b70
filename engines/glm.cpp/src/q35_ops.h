#pragma once
// SYCL kernels for Qwen3.8-27B (docs/qwen35-spec.md), raw device pointers, row-major. The Gated DeltaNet's gates, conv
// and delta rule are q4_ops'; the sizes are the same.
#include "q35.h"

namespace glm::q35ops {

// y[r, d] = fp16(rms(x[r]) * w1p[d]), x fp32 [rows, D]
void norm16(sycl::queue& q, const float* x, const float* w1p, half* y, int rows);
// x += o, fp32 [rows, D]
void add(sycl::queue& q, float* x, const float* o, int rows);
// x += o, then y = norm16(x): one launch, bit for bit the two
void add_norm16(sycl::queue& q, float* x, const float* o, const float* w1p, half* y, int rows);
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
// gdn_delta_seq's and gdn_replay's kernel (set before anything runs or records; Q35_DELTA): 1 a lane per value column
// (contiguous state loads, sums over a thread's 32 key rows then 4 quarters), 0 a lane per key row (sub-group sums)
extern int gdn_delta_variant;
void gdn_delta_seq(sycl::queue& q, const bf16* c, const float* beta, const float* g, float* rec, bf16* core, int rows,
                   const int* desc);

// A rejected draft's replay over every GDN layer in two launches (Q35Model::replay_dev): per layer its saved step
// inputs (qkv kq, beta kb, g kg), its conv window and recurrent state, conv weights; cv: scratch bf16 [nl, rows, 10240]
struct GdnReplayLayer {
    const float* kq; const float* kb; const float* kg;
    bf16* conv; const bf16* conv_w; float* rec;
};
void gdn_replay(sycl::queue& q, const GdnReplayLayer* layers, int nl, bf16* cv, int rows, const int* desc);

// y[r * ys + d] = fp16(rms(x[r]) * w1p[d]), x fp32 or fp16 [rows, D] (MTP: the two halves of fc's input)
template <class T> void norm_rows(sycl::queue& q, const T* x, const float* w1p, half* y, int ys, int rows);
// ring[(sdev[r] * RING + pdev[r] % RING) * D ..] = src[r * D ..] (the post-norm hidden of each decode row, for MTP)
void put_rows(sycl::queue& q, const half* src, int rows, half* ring, const int* pdev, const int* sdev, int RING);
// out[r] = ring[src[r]] (src < MH), mh[src[r] - MH] (src >= MH) or zeros (src < 0), fp16 [D] rows (MTP inputs)
constexpr int MH = 1 << 20;
void gather_hidden(sycl::queue& q, const int* src, int rows, const half* ring, const half* mh, half* out);
// dst [rows_t][dst_tn][words] = src [rows_t][map[j]][words] (uint16 words): an EXL3 trellis' column tiles gathered
// (MTP's draft head from blocks of the lm_head)
void gather_col_tiles(sycl::queue& q, const uint16_t* src, uint16_t* dst, int rows_t, int src_tn, int dst_tn, int words,
                      const int* map);
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
// Decode / verify rows on the ESIMD kernel (kernels/q35_dec.h, src/q35_pf.cpp): row r at position pdev[r] of sequence
// sdev[r] (TS rows a sequence) over its keys 0 .. pdev[r] in ns pieces; part: dec_part_floats(rows, ns) floats. Then
// attn_dec_combine: o16 [rows, 24 * 256] = the pieces merged, times sigmoid(gate). Each row's result depends on its own
// position and keys only (MTP's exactness).
size_t dec_part_floats(int rows, int ns);
void attn_dec(sycl::queue& q, const half* qn16, const half* Kc, const half* Vc, int rows, const int* pdev, const int* sdev,
              int TS, float* part, int ns);
void attn_dec_combine(sycl::queue& q, const float* part, int rows, int ns, const half* qg, half* o16);
// Prompt rows at positions p0 .. p0 + rows - 1 of one sequence (src/q35_pf.cpp): flash attention with DPAS over the
// cache Kc / Vc [T, 4, 256] (keys 0 .. each row's position), o fp16 [rows, 24, 256] before the gate
// variant: 1 kernels/g4_fa.h, 5 kernels/q35_fa.h, 13 the same with K^T from kt (kt_fill), 15 with V from vt too
// (vt_fill); kt / vt hold this layer's keys 0 .. p0 + rows - 1 of the sequence, TS keys a row
void attn_prompt(sycl::queue& q, const half* qn16, const half* Kc, const half* Vc, int rows, int p0, half* o, int variant = 1,
                 const half* kt = nullptr, int TS = 0, const half* vt = nullptr);
// kt [4][128 d pairs][TS][2] = the keys from..to - 1 of Kc [T, 4, 256], pair-transposed (attn_prompt 13 - 15)
void kt_fill(sycl::queue& q, const half* Kc, half* kt, int TS, int from, int to);
// vt [4][TS / 2 key pairs][256][2] = the values from..to - 1 of Vc [T, 4, 256], key pairs interleaved (from rounded
// down to even: a pair is written whole)
void vt_fill(sycl::queue& q, const half* Vc, half* vt, int TS, int from, int to);
// timing experiments on kernels/q35_fa.h (q35_gen fabench): dbg bits drop parts of the kernel (wrong results)
void attn_bench(sycl::queue& q, const half* qn16, const half* Kc, const half* Vc, int rows, int p0, half* o, int dbg);
// o[r, h, :] *= sigmoid(gate[r, h, :]), the gate half of qg; in place
void attn_gate(sycl::queue& q, const half* qg, half* o, int rows);

// q4ops::gdn_conv for a prompt's rows (one sequence), all rows at once; bit-identical
void gdn_conv_prompt(sycl::queue& q, const float* qkv, bf16* state, const bf16* w, bf16* out, int rows);
// The delta rule over a prompt's rows (in order, one sequence) in chunks of 64, kernels/q35_gdn.h: c bf16 [rows, 10240]
// (q | k | v after the conv), beta, g fp32 [rows, 48], rec fp32 [48][128 k][128 v] updated, core bf16 [rows, 48, 128].
// The same function as q4ops::gdn_delta, summed in another order with fp16 matrix inputs. scratch: gdn_chunk_bytes.
size_t gdn_chunk_bytes(int rows);
void gdn_chunked(sycl::queue& q, const bf16* c, const float* beta, const float* g, float* rec, bf16* core, int rows,
                 void* scratch);
// its parts (src/q35_pf.cpp): per chunk A, Mq and the decays; the chunks in order
void gdn_prep(sycl::queue& q, const half* qh, const half* kh, const float* beta, const float* g, float* A, half* Mq,
              float* bt, float* gam, float* wl, int n, int npad, int nc);
void gdn_scan(sycl::queue& q, const bf16* cv, const half* qh, const half* kh, const float* A, const half* Mq,
              const float* bt, const float* gam, const float* wl, float* rec, bf16* core, int n, int npad, int nc);

}  // namespace glm::q35ops
