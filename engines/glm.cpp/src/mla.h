#pragma once
// MLA attention (NoPE, 1-head latent MQA) with the DSA sparse indexer: the decode graph path of
// docs/forward-spec.md §5.4 (gathered attention over the selected tokens at every context length).
#include "exl3.h"
#include "rt.h"
#include "st.h"

namespace glm {

struct MlaLayer {
    static constexpr int H = 64, DC = 512, DN = 256, DV = 256, QR = 1536;
    static constexpr int HI = 32, DI = 128, TOPK = 2048, POOL = 4;
    Exl3Linear q_a, q_b, kv_a, o, wq_b;
    Tensor q_a_norm, kv_a_norm;      // bf16 [1536], [512]
    Tensor kv_b;                     // fp16 [32768, 512]: row h*512 + j: j < 256 W_UK[h, j, :], else W_UV[h, j-256, :]
    Tensor wk, k_norm_w, k_norm_b, weights_proj, gate;   // fp16 indexer
    Tensor ape;                      // fp32 [4, 128]

    static MlaLayer load(const SafeTensors& st, const std::string& prefix);
};

// One sequence's cache planes of one MLA layer (contiguous; position = row)
struct MlaCache {
    Tensor latent;   // fp16 [T, 512]
    Tensor kidx;     // fp16 [T, 256]: indexer key at 0, pool gate at 128
    Tensor kpool;    // fp16 [T / 4, 128]
    int max_tokens = 0;
    void init(int T);
};

// x: fp16 [Q, 4096] (after input_layernorm), rows at positions *pos_dev .. *pos_dev + Q - 1 -> out fp32 [Q, 4096].
// The kernels read the position from device memory (pos_dev) and size their grids by the cache, not the context, so
// a recorded decode step replays at any position; pos0 (the same value on the host) is only checked.
void mla_forward(sycl::queue& q, Workspace& lin, Workspace& act, const MlaLayer& L, MlaCache& c, const half* x, int Q,
                 int pos0, const int* pos_dev, float* out);

}  // namespace glm
