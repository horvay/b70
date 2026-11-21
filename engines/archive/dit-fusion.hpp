#ifndef GGML_SYCL_DIT_FUSION_HPP
#define GGML_SYCL_DIT_FUSION_HPP

#include "common.hpp"

// Element-wise chains of diffusion transformers (MOSS-Image) run as one kernel each. Every chain is bandwidth bound,
// so one pass over the activations instead of one per op:
//   norm_mod  rms_norm(x) * w * m1 + m0            RMS_NORM, MUL, MUL, ADD (adaLN modulation)
//   norm_rope rope(rms_norm(x) * w)                 RMS_NORM, MUL, VIEW x2, CONCAT, MUL, MUL, ADD (rotate-half RoPE
//                                                   as x * rc + swap_halves(x) * rs)
//   situ      tanh(a*g + b) * sigmoid(g) * tanh(c*h + d)   SCALE, TANH, SIGMOID, MUL, SCALE, TANH, MUL
//   mul_add   x + y * g                             MUL, ADD (gated residual)
// and the runtime LoRA of sd.cpp: the up projection accumulates into the base matmul's output (lora_gemm_acc), and the
// LoRA matmuls can run in a reduced-precision compute mode (GGML_SYCL_LORA_GEMM_MODE).
// Returns the number of nodes after node_idx that were consumed (0 = no match, -1 = node_idx alone); writes the label
// of the fused kernel to *label for the profiler.
int ggml_sycl_try_dit_fusion(ggml_backend_sycl_context & ctx, ggml_cgraph * cgraph, int node_idx, const char ** label);

#endif  // GGML_SYCL_DIT_FUSION_HPP
