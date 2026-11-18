#pragma once
// Qwen3.8-Flash-Next (qwen4_exp) text trunk, docs/qwen4-spec.md: embedding (host bf16 table) expanded into 4 fp32
// streams, 48 blocks (a gated-residual hyper-connection site in front of Gated DeltaNet or QSA attention, another in
// front of the MoE with a shared expert), the hashed n-gram PLE in front of layer 1, the combine-less mixer and the
// lm_head (no final norm). Every routed expert is resident: a VRAM slot or a pinned RAM record the kernels read in
// place, both filled from the safetensors at load (no NVMe store).
#include <memory>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <string>
#include <vector>
#include "exl3.h"
#include "lm.h"
#include "st.h"

namespace glm {

namespace q4 {
constexpr int D = 2560, S = 4, SD = S * D, R = 320, L = 48, V = 248320, E = 512, TOPK = 10, I = 640;
// Gated DeltaNet
constexpr int GK = 16, GV = 48, GDK = 128, GDV = 128, GQKV = 2 * GK * GDK + GV * GDV, GCONV = 4;
// attention + QSA indexer
constexpr int AH = 24, AKV = 2, AD = 256, ROT = 64, IH = 4, ID = 128, BLK = 4, BUDGET = 512, NSEL = BUDGET * BLK + BLK - 1;
constexpr float EPS = 1e-6f;
constexpr int EOS = 248044;
}  // namespace q4

// one gated-residual site: (1 + hc_norm) fp32 [4D]; down | inject fp16 [320 (+4), 4D]; up fp16 [4D, 320]
struct Q4Site {
    Tensor w1p, wdi, wu;
    int nd = 0;                      // rows of wdi: 324 (site) or 320 (the final mixer: no inject)
};

struct Q4Gdn {
    Exl3Linear qkv, z, o;
    Tensor ab;                       // fp16 [96, 2560]: in_proj_b rows, then in_proj_a
    Tensor conv_w;                   // bf16 [10240, 4]
    Tensor neg_a, dt_bias, norm_w;   // fp32 [48] (-exp(A_log)), [48], [128]
};

struct Q4Attn {
    Exl3Linear q, k, v, o, idx;
    Tensor qn, kn, iqn, ikn;         // fp32 (1 + w): [256], [256], [128], [128]
};

struct Q4Block {
    int layer = 0;
    bool attn = false;
    Q4Site site[2];                  // [0] attention, [1] MoE
    std::unique_ptr<Q4Gdn> gdn;
    std::unique_ptr<Q4Attn> att;
    Tensor router;                   // fp16 [513, 2560]: the 512 experts, then shared_expert_gate
    Exl3Linear sg, su, sd;           // shared expert
};

// per-sequence state of one GDN layer: conv window bf16 [10240][3] (last three qkv columns), recurrent fp32 [48][128][128]
struct Q4GdnState {
    Tensor conv, rec;
    void init();
    void zero();
};

// per attention layer: K (normed, roped) / V fp16 [T, 2, 256], indexer raw keys fp16 [T, 128], pooled keys fp16
// [T/4, 128] (normed, roped at the block start)
struct Q4Cache {
    Tensor k, v, rk, pk;
    void init(int max_tokens);
};

class NgramTable;

// one routed expert's record for the MoE kernels (layer li, expert e), moe_record_bytes(D, I, 3) bytes at dst
void q4_build_record(const SafeTensors& st, int li, int e, char* dst);

class Q4Model : public LM {
public:
    // max_tokens: context; reserve_gib: VRAM kept free beside the expert slots; hot: expert ranking files
    // (flashnext/hot/*.json, hottest first), comma-separated, empty for an even spread
    Q4Model(const std::string& model_dir, int max_tokens, double reserve_gib = 2.0, const std::string& hot = "");
    ~Q4Model() override;
    const half* forward(const int* tokens, int Q) override;
    void prefill(const int* tokens, int n, int ckpt_at = -1) override;
    bool restore(int upto) override;
    void reset() override;
    int position() const override { return pos; }
    int max_context() const override { return max_tokens; }
    int vocab() const override { return q4::V; }
    int prefill_chunk = 2048;        // rows per attention / GDN / MoE piece of a prompt

    // debug: row 0's [site input y | sublayer out] per sublayer, and the streams after each block, when set
    std::vector<float>* dump = nullptr;
    // routed picks since the last call: total and RAM-resident, and the share of RAM picks the hottest n RAM experts
    // account for (n = 256, 1024, 4096); clears the counters
    std::string expert_stats();
    size_t swaps() const { return swapped_; }

private:
    std::unique_ptr<SafeTensors> st;
    std::unique_ptr<NgramTable> ngram;
    std::vector<Q4Block> blocks;
    Q4Site mixer;
    Exl3Linear lm_head;
    const bf16* embed = nullptr;     // host [V, D]
    // PLE (layer 1)
    Tensor ple_wkv;                  // fp16 [12800, 2560]: key_proj rows, then value_proj
    Tensor ple_nk, ple_nq, ple_nc;   // fp32 (1 + w) [4D]
    Tensor ple_cw;                   // fp32 [4D, 4]
    Tensor ple_state;                // fp16 [4D][9]: the last 9 normed conv inputs
    int ids_ctx[2] = {q4::EOS, q4::EOS};
    std::vector<Q4GdnState> gst;
    std::vector<Q4Cache> kv;

    // experts: ptrs int64 [L, E] (VRAM slot or RAM record), staged rows for prompts
    size_t rec_bytes = 0;
    char* vram_pool = nullptr;
    char* ram_pool = nullptr;
    std::vector<int8_t> in_vram;     // [L * E]
    std::vector<int64_t> ptrs_h;
    Tensor ptrs_dev;
    Tensor use_dev, in_vram_dev, stats_dev;   // int32 [L, E] decode pick counts; uint8 [L, E]; u64 [2] picks, RAM picks
    void load_experts(const std::string& hot, double reserve_gib);
    // adaptive residency: every rebal_every decode tokens, the most-picked RAM experts trade places with the
    // least-picked VRAM ones (counts decay by half each time), at most rebal_max pairs, between tokens
    int rebal_every = 32, rebal_max = 64, since_rebal = 0;
    std::vector<float> score_;                // decayed decode picks per (layer, expert)
    char* swap_buf = nullptr;                 // pinned [rebal_max] records
    size_t swapped_ = 0;
    void rebalance();
    bool counting = false;                    // count_use runs (decode only: a prompt touches nearly every expert)

    Workspace lin, act, mws;
    int pos = 0, max_tokens = 0, qmax = 8;
    // activations (prefill_chunk rows, except X: qmax rows for decode)
    Tensor X, n16, t324, a16, gpre, y16, sub, inj, logits, rl, sel, wts, moe16, sg16, su16, sa16, sh32, emb16, kv32, gv32,
        cn16, fin16;
    float* x_host = nullptr;         // pinned [prefill_chunk, D]: embedding rows on their way up
    half* e_host = nullptr;          // pinned [prefill_chunk, D]: n-gram rows
    sycl::event x_ev, e_ev;          // their last uploads (the buffers are rewritten after these)
    // decode (one token) as one recorded graph: the position comes from pos_dev (copied from pinned pos_h in it)
    Tensor pos_dev;
    int* pos_h = nullptr;
    bool graphs_on = true, recording = false;
    std::unique_ptr<sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>>
        dec_graph;
    sycl::event dec_ev;
    void host_rows(const int* tokens, int n);         // embedding rows -> x_host, n-gram rows -> e_host (PLE ids advance)
    void forward_dev(int Q, const int* pdev);         // the device part of forward() (recorded for Q = 1)

    void embed_rows(const int* tokens, int n, float* Xd);
    void site_mix(const Q4Site& s, const float* Xd, int rows, half* y, float* injp);
    void site_apply(float* Xd, const float* injp, const float* o, int rows);
    void gdn_forward(const Q4Gdn& g, Q4GdnState& s, const half* y, int rows, float* out);
    void attn_forward(const Q4Attn& a, Q4Cache& c, const half* y, int rows, int p0, float* out, const int* pdev = nullptr);
    void moe_forward_(const Q4Block& b, const half* y, int rows, float* out, const int64_t* ptrs);
    void ple_host(const int* tokens, int rows);       // n-gram rows -> e_host, uploaded to emb16
    void ple_dev(int rows, float* Xd);
    // tokens null: the PLE's rows are already in emb16 (forward())
    void block_forward(Q4Block& b, float* Xd, int rows, int p0, const int64_t* ptrs, const int* tokens,
                       const int* pdev = nullptr);
    void grab(const void* p, int n, bool f16);

    struct Checkpoint { int pos = -1; std::vector<void*> conv, rec; void* ple = nullptr; int ids[2]; };
    std::vector<Checkpoint> ckpts_;
    Checkpoint& new_checkpoint(int at);
    void save_checkpoint(Checkpoint& c);
};

// The hashed n-gram embedding table (ngram_embedding.safetensors, docs/qwen4-spec.md §5): 16 rows of 160 values per
// token, read from the file with pread on a thread pool and decoded on the host.
class NgramTable {
public:
    explicit NgramTable(const std::string& path);
    ~NgramTable();
    // hist: [ctx0, ctx1, tok_0 .. tok_{n-1}] -> emb fp16 [n, 2560]
    void lookup(const int* hist, int n, half* emb);
private:
    int fd = -1;
    int64_t base = 0;                // file offset of row 0
    int K = 5, row_words = 51;
    int64_t mult[3] = {};
    int64_t vsize[16] = {}, offs[16] = {};
    std::vector<float> head_bias;    // [16, 160]
    int threads = 16;
};

}  // namespace glm
