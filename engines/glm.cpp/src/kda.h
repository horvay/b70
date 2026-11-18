#pragma once
// KDA linear attention (GatedDeltaNet with a per-channel forget gate), docs/forward-spec.md §4: recurrent algorithm
// for S tokens at a time (decode, MTP verify with history, short prefill pieces).
#include "exl3.h"
#include "rt.h"
#include "st.h"

namespace glm {

struct KdaLayer {
    static constexpr int H = 64, DK = 128, DV = 128, F = 3 * H * DK, CONV = 4;
    Exl3Linear qkv, o;
    Tensor conv_w;              // bf16 [24576, 4]
    Tensor b_proj, f_a, f_b, g_a, g_b;     // fp16 [out, in]
    Tensor A_log, dt_bias;      // fp32 [64], [8192]
    Tensor o_norm;              // bf16 [128]
    const Lora *lb = nullptr, *lfa = nullptr, *lfb = nullptr, *lga = nullptr, *lgb = nullptr;   // fp16 linears' LoRAs

    Tensor w3;                  // fp16 [320, 4096]: g_a | b_proj | f_a stacked (one GEMV; built by fuse())
    static KdaLayer load(const SafeTensors& st, const std::string& prefix);
    // after the LoRA is attached / merged: stack the three projections on the block input when none has a runtime LoRA
    void fuse();
};

// Per-sequence state of one KDA layer: conv_state bf16 [F][4 + hist], recurrent fp32 [hist + 1][H][DK][DV].
struct KdaState {
    Tensor conv, rec;
    int hist = 0;
    int last_history = 0;
    void init(int max_history);
    void zero();
};

// y: fp16 [S, 4096] (after input_layernorm) -> out fp32 [S, 4096]. history: keep per-token states for a rewind.
// lin: the linears' scratch (reset by each linear); act: this layer's activations (reset here)
void kda_forward(sycl::queue& q, Workspace& lin, Workspace& act, const KdaLayer& L, KdaState& st, const half* y, int S,
                 float* out, bool history);
// after an MTP verify with history: drop the last r tokens' state (r = 0 still restores the conv window)
void kda_rewind(sycl::queue& q, KdaState& st, int r);

}  // namespace glm
