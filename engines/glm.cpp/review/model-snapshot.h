#pragma once
// GLM-5.3-Flash trunk: embedding (host bf16 table), 45 blocks (mHC around KDA or MLA attention, then a dense MLP or
// MoE with a shared expert), the mean over the 4 streams, the final norm and the lm_head. docs/forward-spec.md §2.
#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include "exl3.h"
#include "kda.h"
#include "mla.h"
#include "ops.h"
#include "st.h"
#include "tier.h"

namespace glm {

struct Block {
    int layer = 0;
    bool is_mla = false, is_moe = false;
    std::unique_ptr<KdaLayer> kda;
    std::unique_ptr<MlaLayer> mla;
    Tensor hc_fn[2], hc_base[2], hc_scale[2];   // [0] attention site, [1] MLP site; fn as fp16 (decode)
    Tensor hc_fn32[2];                           // fp32 fn (mixes of more than 32 rows, as the PyTorch stack)
    Tensor in_norm, post_norm;                   // bf16 [4096]
    Exl3Linear gate, up, down;                   // dense MLP, or the MoE's shared expert
    Tensor router, bias_c, esb;                  // MoE: fp16 [288, 4096]; fp16 mean-centred bias; fp32 bias
    int tier_li = -1, kc = 0, next_moe = -1;     // next_moe: the next MoE layer (prefetch prediction)
};

struct ModelConfig {
    int hidden = 4096, layers = 45, vocab = 154880, experts = 288, topk = 8, moe_inter = 2048;
    float rsf = 2.5f, eps = 1e-5f;
    std::vector<bool> mla, moe;
};

class Model {
public:
    // warm_stats: routing profile (llm/glm/xpu/data/route_stats_glm53.json) for the start-up fill, or empty.
    // tc.vram_bytes 0: what is free after the weights, less reserve_gib
    // lora: adapter folder(s), comma-separated, or empty; lora_scale multiplies alpha / r
    Model(const std::string& model_dir, const std::string& store_bin, int max_tokens, TierConfig tc,
          const std::string& warm_stats = "", double reserve_gib = 3.0, const std::string& lora = "",
          float lora_scale = 1.0f);
    // run Q tokens at the current position (decode / forced continuation); logits fp16 [Q, vocab] on the device
    const half* forward(const int* tokens, int Q);
    // a prompt (no logits): n tokens at the current position, layer by layer over all rows (attention and MLPs in
    // pieces of prefill_chunk rows, each MoE layer's experts staged once for all rows); the prompt's last token then
    // goes through forward(), as the PyTorch stack's generator does
    // ckpt_at (a position inside (pos, pos + n]): snapshot the recurrent (KDA) state there, for prefix reuse
    void prefill(const int* tokens, int n, int ckpt_at = -1);
    int prefill_chunk = 2048;
    // prefix reuse: the attention cache stays valid below any position; the recurrent state is restored from the
    // latest checkpoint at or below `upto` (false: none, the caller starts over with reset())
    bool restore(int upto);
    int latest_checkpoint() const { return ckpts_.empty() ? -1 : ckpts_.back().pos; }
    void reset();
    int position() const { return pos; }
    int max_context() const { return max_tokens; }
    const ModelConfig& config() const { return cfg; }
    Tier& tier() { return *tier_; }
    // GLM_PROF=1: wall time per phase (each phase waits for the GPU, so the total runs slower than without)
    std::string prof_report(int tokens);
    void prof_reset() { std::fill(prof_s.begin(), prof_s.end(), 0.0); }
    // debug: when set, forward() records per layer [attn_in | attn_out | mlp_in | mlp_out | R (4 x 4096)] of row 0
    std::vector<float>* dump = nullptr;
    // debug probes, one sublayer of layer L on host data at position 0 (fresh state): attention (y fp16 -> fp32),
    // MLP (y -> fp32, with the picks it used), mHC mix + norm of a site (R fp32 [4 x 4096] -> y)
    void probe_attn(int L, const std::vector<half>& yin, std::vector<float>& out);
    void probe_mlp(int L, const std::vector<half>& yin, std::vector<float>& out, std::vector<int>& picks);
    void probe_mix(int L, int site, const std::vector<float>& Rin, std::vector<half>& yout);

private:
    ModelConfig cfg;
    std::unique_ptr<SafeTensors> st;
    std::unique_ptr<Tier> tier_;
    std::unique_ptr<LoraSet> loras_;
    std::vector<Block> blocks;
    const bf16* embed = nullptr;                 // host [vocab, 4096]
    Tensor final_norm;
    Exl3Linear lm_head;
    std::vector<KdaState> kda_st;               // per layer (empty for MLA layers)
    std::vector<MlaCache> mla_c;
    Workspace lin, act, mws;
    int pos = 0, max_tokens = 0, qmax = 8;
    // activations
    Tensor x_in, R, collapsed, y, attn_out, mlp_out, g, u, a, sh, moe_out, post, comb, hc_part, sel, w, rlogits, nlogits, picks,
        fin, fin16, logits;
    std::vector<float> x_host;
    int64_t* picks_h = nullptr;      // pinned: [sel Q*k | exact Q*k | pred Q*w1]
    static constexpr int kPfWidth = 12;     // predicted picks of the next MoE layer (prefetch)

    bool prof_on = false, prof_gpu = false;
    std::vector<std::pair<int, sycl::event>> prof_ev;     // GLM_PROF=2: (phase ending here, barrier)
    void prof_collect();
    std::vector<double> prof_s;
    std::chrono::steady_clock::time_point prof_t;
    void mark(int phase);
    void block_forward(Block& b, int Q);
    void mlp_forward(Block& b, int Q);
    void mix(Block& b, int site, float* Rr, int rows, float* post_, float* comb_, half* y_);
    std::vector<int> picks_all;
    struct Checkpoint { int pos = -1; std::vector<void*> conv, rec; };     // pinned host copies per KDA layer
    std::vector<Checkpoint> ckpts_;                                          // oldest first, at most 2
    Checkpoint& new_checkpoint(int at);
};

}  // namespace glm
