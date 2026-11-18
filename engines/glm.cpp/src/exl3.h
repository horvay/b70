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
    int cb = 2;                      // codebook: 2 mul1, 0 the original one (no mul1 tensor: some packs' heads)
    const Lora* lora = nullptr;      // GM LoRA (applied after the base output)

    static bool present(const SafeTensors& st, const std::string& prefix) { return st.has(prefix + ".trellis"); }
    static Exl3Linear load(const SafeTensors& st, const std::string& prefix);
    // K's code from a trellis' uint16 words per tile (GLM_CHECK on a rate not built)
    static int k_code(int words, bool mul1);
    // x: fp16 [M, in] row-major (x_stride elements between rows); y: [M, out] fp16 or fp32, row-major.
    // n_cols (a multiple of 128, at most exl3_large_above rows, no LoRA): only the first n_cols outputs, y [M, n_cols]
    void forward(sycl::queue& q, Workspace& ws, const half* x, int M, void* y, DType ydt, int x_stride = -1,
                 int n_cols = -1) const;
    // workspace bytes forward() needs for M rows
    size_t ws_bytes(int M) const {
        return (M > 128 ? ((size_t)M * in + (size_t)M * out + (size_t)in * std::min(out, 16384)) * 2 + 3 * 256
                        : (size_t)128 * in * 2 + (size_t)16 * M * out * 4 + 2 * 256) +
               (lora ? lora->ws_bytes(M) : 0);
    }

private:
    void forward_base(sycl::queue& q, Workspace& ws, const half* x, int M, void* y, DType ydt, int x_stride,
                      int n_cols = -1) const;
    void forward_large(sycl::queue& q, Workspace& ws, const half* x, int M, void* y, DType ydt, int x_stride) const;
public:
};

}  // namespace glm
