#pragma once
// EXL3 (trellis, mul1 codebook) linear: y = svh * Had(Had(x * suh) @ W_hat) / 128, on the ESIMD kernels in
// kernels/exl3_esimd.h. Rows M <= 8 run the vector GEMV (fp16 tile partials), 9..128 the DPAS kernel (fp32), as the
// PyTorch stack's exl3_gemm_small (llm/glm/xpu/csrc/exl3_ops.sycl).
#include <algorithm>
#include <string>
#include "rt.h"
#include "lora.h"
#include "st.h"

namespace glm {

// rows above this take the reconstruct + GEMM path (a model's choice: DPAS blocks past 32 rows spill registers on Gemma)
extern int exl3_large_above;
// that path in int8 (W8A8, default on; GLM_EXL3_INT8=0: fp16): the Hadamard-domain activations quantized per row,
// the trellis decoded to int8 at one static scale (|mul1 codebook value| <= 3.453125), an s8 x s8 -> s32 oneDNN GEMM
// with both scales applied in its epilogue (fp16 out), then the output Hadamard as before. Decode rows are untouched.
// GLM_EXL3_INT8_CHECK=1 also runs the fp16 path on each call and prints the relative error (slow; a check).
extern bool exl3_int8;
std::string exl3_i8_stages_report();      // GLM_EXL3_I8_STAGES=1: the int8 path's time by stage since the last report
// the int8 path keeps the decoded weights of the last linears it ran in an arena of this many MB (a model's choice; 0:
// off; GLM_EXL3_I8_CACHE_MB overrides): a prompt read in pieces decodes each trellis once per layer, not per piece
extern size_t exl3_i8_cache_mb;
// rows up to this take the vector kernel, above it DPAS (a model's choice: DPAS wins sooner at low K)
extern int exl3_vec_max;
// the vector GEMV splits K for M rows as it does for one row, so each row's sum runs in the same order whatever M is
// (Qwen3.5 MTP: a verify step's logits equal one-row steps'); GLM_EXL3_SAMESPLIT=0/1 overrides
extern bool exl3_same_split;

struct Exl3Linear {
    Tensor trellis;   // I16 [in/16, out/16, 16K] (16K + 8 at half-integer K)
    Tensor suh;       // F16 [in]
    Tensor svh;       // F16 [out]
    Tensor sonb;      // I32 zeros [out/128] (one Hadamard group)
    int K = 0, in = 0, out = 0;      // K: the bitrate 1..8, or 10 KA + 5 for KA + 0.5 (kernels/exl3_esimd.h's code)
    // a merge of linears that read the same input (merge()): S input Hadamards (suh [S, in]), shard s's outputs are
    // columns shard_col[s] .. shard_col[s + 1] - 1, sonb says which shard each 128 columns read
    int S = 1;
    int shard_col[5] = {};
    int cb = 2;                      // codebook: 2 mul1, 0 the original one (no mul1 tensor: some packs' heads)
    const Lora* lora = nullptr;      // GM LoRA (applied after the base output)

    static bool present(const SafeTensors& st, const std::string& prefix) { return st.has(prefix + ".trellis"); }
    static Exl3Linear load(const SafeTensors& st, const std::string& prefix);
    // one linear of up to 4 that read the same input, at the same K and codebook (false: they cannot; nothing changed).
    // The parts' weights move into it (their tensors are freed); forward_split gives each its own output.
    static bool can_merge(const std::vector<const Exl3Linear*>& parts);
    static Exl3Linear merge(const std::vector<Exl3Linear*>& parts);
    // a merged linear's outputs: shard s's columns to ys[s] ([M, its width], row-major), one call
    void forward_split(sycl::queue& q, Workspace& ws, const half* x, int M, void* const* ys, DType ydt) const;
    // K's code from a trellis' uint16 words per tile (GLM_CHECK on a rate not built)
    static int k_code(int words, bool mul1);
    // x: fp16 [M, in] row-major (x_stride elements between rows); y: [M, out] fp16 or fp32, row-major.
    // n_cols (a multiple of 128, at most exl3_large_above rows, no LoRA): only the first n_cols outputs, y [M, n_cols]
    void forward(sycl::queue& q, Workspace& ws, const half* x, int M, void* y, DType ydt, int x_stride = -1,
                 int n_cols = -1) const;
    // workspace bytes forward() needs for M rows
    size_t ws_bytes(int M) const {
        return (M > 128 ? std::max(((size_t)S * M * in + (size_t)M * out + (size_t)in * std::min(out, 16384)) * 2 + 3 * 256,
                                   i8_ws_bytes(M))
                        : (size_t)128 * S * in * 2 + (size_t)16 * M * out * 4 + 2 * 256) +
               (lora ? lora->ws_bytes(M) : 0);
    }
    // the int8 path: xq int8 [S, Mp, in], sx fp32 [S, Mp], part fp16 [Mp, out], W int8 [in, <= 32768 columns]; Mp: M
    // padded to a multiple of kI8RowPad, so a handful of GEMM shapes serve every prompt piece
    static constexpr int kI8RowPad = 256, kI8Slice = 32768;
    size_t i8_ws_bytes(int M) const {
        size_t Mp = (size_t)(M + kI8RowPad - 1) / kI8RowPad * kI8RowPad;
        return S * (Mp * in + Mp * 4) + Mp * out * 2 + (size_t)in * std::min(out, kI8Slice) + (4 + 2 * S) * 256;
    }

private:
    // ys (a merged linear's outputs, one a shard) or y
    void forward_base(sycl::queue& q, Workspace& ws, const half* x, int M, void* y, DType ydt, int x_stride,
                      int n_cols = -1, void* const* ys = nullptr) const;
    void forward_large(sycl::queue& q, Workspace& ws, const half* x, int M, void* y, DType ydt, int x_stride,
                       void* const* ys = nullptr) const;
    void forward_large_f16(sycl::queue& q, Workspace& ws, const half* x, int M, void* y, DType ydt, int x_stride,
                           void* const* ys = nullptr) const;
    void forward_large_i8(sycl::queue& q, Workspace& ws, const half* x, int M, void* y, DType ydt, int x_stride,
                          void* const* ys = nullptr) const;
    void had_out(sycl::queue& q, const half* part, int M, void* y, DType ydt, void* const* ys = nullptr) const;   // fp16 part [M, out] -> y
public:
};

}  // namespace glm
