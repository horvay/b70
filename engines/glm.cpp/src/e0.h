#pragma once
// Edge0-145B (Edge0/Edge0-145B-NVFP4): Qwen3.8-Flash-Next's trunk (q4.h, the same kernels) with a latent MoE in every
// layer, from the release folder as shipped (docs/edge0-spec.md):
//   model-skeleton.gguf   trunk: Q8_0 linears, BF16 small projections, F32 norms (llama.cpp's qwen4exp conversion:
//                         norms carry their +1, ssm_a is -exp(A_log), GDN value heads in tiled order: undone at load)
//   latent_dense_q8.bin   per layer 8 groups of D [1792, 2560] and U [2560, 1792], Q8_0
//   labels.bin            int32 [48, 512]: each expert's D/U group
//   experts.bin           [48, 512] records of 1,935,376 B: NVFP4 gate [640, 1792] | up | down [1792, 640] + 4 fp32
//                         {s_gate, s_up, s_down, 0}, each scale multiplying its projection's output
//   ple-fp8.gguf          the n-gram table: 320,001,536 rows of 160 FP8 E4M3 bytes times strata.ple.scale
//   profile.bin           STRP v1: (layer, expert) pairs hottest first, the VRAM slots' order
// MoE(x) = sum_k w_k U_{g_k} A_k(D_{g_k} x) + sigmoid(gate . x) shared(x), g_k = label[e_k].
#include <memory>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <string>
#include <vector>
#include <oneapi/mkl/types.hpp>
#include "e0_ops.h"
#include "gguf.h"
#include "lm.h"
#include "q4.h"

namespace glm {

struct E0Gdn {
    Q8W qkv, z, o;
    Tensor ab, conv_w, neg_a, dt_bias, norm_w;      // as Q4Gdn
};

struct E0Attn {
    Q8W q, k, v, o;
    Tensor idx;                                     // fp16 [640, 2560]: indexer q (512 rows) | k (128)
    Tensor qn, kn, iqn, ikn;                        // fp32 (1 + w)
};

struct E0Block {
    int layer = 0;
    bool attn = false;
    Q4Site site[2];
    std::unique_ptr<E0Gdn> gdn;
    std::unique_ptr<E0Attn> att;
    Tensor router;                                  // fp16 [513, 2560]: the 512 experts, then the shared expert's gate
    Q8W sg, su, sd;                                 // shared expert
    Q8W dcat, ucat;                                 // D: [8 * 1792, 2560]; U: [2560, 8 * 1792] (groups side by side)
};

// a QSA layer's cache: K / V int8 [T, 2, 256] with fp16 scales per 64 values [T, 2, 4] (or fp16 K / V with E0_KV16=1), the indexer's
// raw keys fp16 [T, 128] and pooled keys fp16 [T / 4 + 1, 128]
struct E0Cache {
    Tensor k, v, ks, vs, rk, pk;
    bool i8 = true;
    void init(int max_tokens, bool int8);
};

// the FP8 n-gram table: 16 rows per token, pread on a thread pool
class E0Ngram {
public:
    E0Ngram(const std::string& path, const Gguf& skel);
    ~E0Ngram();
    void lookup(const int* hist, int n, half* emb);     // as NgramTable::lookup
private:
    int fd = -1;
    uint64_t base = 0;
    float scale = 1.0f;
    int64_t mult[3] = {}, vsize[16] = {}, offs[16] = {};
    float lut[256];
};

// The MTP draft layer (mtp/: dense.txt + dense.bin, experts.bin Q2_0): one full-attention block over the trunk's last
// stream stack and the next token's embedding (docs/edge0-spec.md section 7)
struct E0Mtp {
    Q8W fc_emb, fc_hid, q, k, v, o, sg, su, sd;
    Tensor w_pe, w_ph, qn, kn;                       // fp32 norm weights (+1 applied)
    Q4Site site_attn, site_mlp, mixer;
    Tensor router;                                   // fp16 [513, 2560]
    char* experts = nullptr;                         // 512 Q2_0 records in VRAM
    Tensor ptrs;                                     // int64 [512]
    Tensor Kc, Vc;                                   // fp16 [max_tokens, 2, 256]
};

class E0Model : public LM {
public:
    // dir: the release folder (experts.bin and ple-fp8.gguf in it, or E0_EXPERTS / E0_PLE)
    E0Model(const std::string& dir, int max_tokens, double reserve_gib = 2.0);
    ~E0Model() override;
    const half* forward(const int* tokens, int Q) override;
    void prefill(const int* tokens, int n, int ckpt_at = -1) override;
    bool restore(int upto) override;
    void reset() override;
    int position() const override { return pos; }
    int max_context() const override { return max_tokens; }
    int vocab() const override { return q4::V; }
    int prefill_chunk = 2048;

    int draft(int tok, int hidden_row, int n, int* out) override;
    float spec_min_p = 0.5f;                         // a draft below this probability ends the chain (E0_SPEC_MIN_P)

    std::vector<float>* dump = nullptr;
    std::string expert_stats();
    size_t swaps() const { return swapped_; }

private:
    std::unique_ptr<Gguf> skel;
    std::unique_ptr<E0Ngram> ngram;
    std::vector<E0Block> blocks;
    Q4Site mixer;
    Q8W lm_head;
    const uint8_t* embed = nullptr;                 // host Q8_0 [V, D] (the skeleton's mapping)
    Tensor ple_wkv, ple_nk, ple_nq, ple_nc, ple_cw, ple_state;
    int ids_ctx[2] = {q4::EOS, q4::EOS};
    std::vector<Q4GdnState> gst;
    std::vector<E0Cache> kv;
    bool kv8 = true;

    // experts
    size_t rec_bytes = 0;                           // a slot (the record rounded up)
    std::vector<char*> vram_chunks;                 // the VRAM slots' allocations (checked pieces)
    char* ram_pool = nullptr;
    std::vector<int8_t> in_vram;
    std::vector<int64_t> ptrs_h;
    std::vector<int> labels_h;                      // [L, E]
    std::vector<float> scales_h;                    // [L, E, 3]
    Tensor ptrs_dev, labels_dev;
    Tensor use_dev, in_vram_dev, stats_dev;
    void load_experts(const std::string& path, const std::string& profile, double reserve_gib);
    int rebal_every = 32, rebal_max = 64, since_rebal = 0;
    std::vector<float> score_;
    char* swap_buf = nullptr;
    size_t swapped_ = 0;
    void rebalance();
    bool counting = false;

    Workspace act;
    int pos = 0, max_tokens = 0, qmax = 8;
    Tensor X, n16, t324, a16, gpre, y16, sub, inj, logits, rl, sel, wts, sg16, su16, sa16, sh32, emb16, kv32, gv32, cn16,
        fin16;
    // latent MoE buffers: decode (MV_MAX rows) and prompt (prefill_chunk rows)
    Tensor xl32, h32, yl32, z32, deq, elist;
    // prompts: the MoE runs over up to moe_rows rows at once (each RAM expert crosses PCIe once per piece), D and U
    // applied per expert from their fp16 copies (ddeq [GE, D], udeq [D, GE])
    int moe_rows = 8192, moe_batch = 32, pick_cap = 8192;
    bool group_skip = true;                         // decode: D / U skip the groups no pick uses (E0_GROUP_SKIP=0)
    bool moe_dpas = true;                           // prompts: the XMX kernels (E0_MOE_PATH=expert: one expert at
                                                    // a time through oneMKL)
    void moe_prompt_dpas(const E0Block& b, int rows, const std::vector<int>& cnt, float* out);
    std::vector<int> dp_lst, dp_sorted, dp_br0, dp_bn, dp_crow, dp_coff, dp_cpick;
    std::vector<int64_t> dp_brec;
    Tensor blk_rec, blk_r0, blk_n;
    int blk_cap = 0;
    bool moe_grouped = false;                       // E0_MOE_GROUPED=1: batches of experts through oneMKL's grouped
                                                    // GEMMs (slower than one expert at a time so far)
    bool stage_on = false;                          // E0_STAGE=1: RAM records staged through VRAM on the copy engine
                                                    // (measured no faster: 739 vs 778 tok/s at 4K)
    Tensor stage;                                   // 8 VRAM slots for RAM records on their way to the dequantizer
    Tensor ddeq, udeq, xl16, bgu, bdn, Xb, GUb, h16b, Yb, Yh, Ob, yall, injall, outall;
    // grouped GEMMs: device-visible pointer arrays [3 GEMMs][E] and the host parameter arrays (alive per layer)
    int64_t* recs_sh = nullptr;
    const half** pa_sh = nullptr;
    const half** pb_sh = nullptr;
    float** pc_sh = nullptr;
    std::vector<int64_t> gm_m, gm_n, gm_k, gm_lda, gm_ldb, gm_ldc, gm_gs;
    std::vector<float> gm_alpha, gm_beta;
    std::vector<oneapi::mkl::transpose> gm_ta, gm_tb;
    std::vector<int> sel_h;
    std::vector<float> wts_h;
    float* x_host = nullptr;
    half* e_host = nullptr;
    sycl::event x_ev, e_ev;
    Tensor pos_dev;
    int* pos_h = nullptr;
    bool graphs_on = true, recording = false;
    std::unique_ptr<sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>>
        dec_graphs[9];                              // by row count (1 .. qmax)
    sycl::event dec_ev;

    void embed_host(const int* tokens, int n);
    void host_rows(const int* tokens, int n);
    void forward_dev(int Q, const int* pdev);
    void embed_rows(const int* tokens, int n, float* Xd);
    void site_mix(const Q4Site& s, const float* Xd, int rows, half* y, float* injp);
    void gdn_forward(const E0Gdn& g, Q4GdnState& s, const half* y, int rows, float* out);
    void attn_forward(const E0Attn& a, E0Cache& c, const half* y, int rows, int p0, float* out, const int* pdev);
    void moe_forward_(const E0Block& b, const half* y, int rows, float* out);
    void ple_host(const int* tokens, int rows);
    void ple_dev(int rows, float* Xd);
    void block_forward(E0Block& b, float* Xd, int rows, int p0, const int* tokens, const int* pdev = nullptr);
    // block_forward up to the MoE's input: y [rows, D] fp16 and the site's inject [rows, 4] at the given places
    void pre_moe(E0Block& b, float* Xd, int rows, int p0, const int* tokens, const int* pdev, half* y, float* injp);
    void grab(const void* p, int n, bool f16);

    // MTP
    std::unique_ptr<E0Mtp> mtp;
    int mtp_done = 0;                               // MTP cells (K/V) filled for positions below this
    int pend_pos = -1;                              // a cell whose stack is known but not its next token (mtp_pend)
    Tensor mtp_pend, mtp_in, mtp_R, mtp_emb, mtp_e2, mtp_h2, mtp_hn, mtp_en, mtp_k, mtp_v, mtp_qg, mtp_qn, mtp_o16,
        mtp_o, mtp_part, mtp_h, mtp_moe, mtp_logits, mtp_best;
    float* mtp_emb_h = nullptr;                     // pinned embedding rows
    std::vector<int> fwd_toks;                      // the last forward()'s tokens
    int fwd_p0 = 0;
    void load_mtp(const std::string& dir);
    // MTP cells p0 .. p0 + T - 1 from stacks Xin [T, S, D] and their next tokens; full: the whole block for the last
    // row, its stack left in mtp_R and its draft in mtp_best ({token, probability bits})
    void mtp_run(const float* Xin, const int* toks, int T, int p0, bool full);
    // the prompt's cells (rows with a known next token), the last row's stack kept for later (pend_pos)
    void mtp_prefill(const float* Xp, const int* toks, int rows, int p0, bool last_known);

    // verify windows (forward() with Q > 1): the recurrent state before it and the per-row inputs to replay a prefix
    int vwin_p0 = 0, vwin_Q = 0;
    bool saving = false;
    std::vector<int> vwin_toks;
    int ids_bak[2] = {0, 0};
    std::vector<Tensor> rec_bak, conv_bak, sv_qkv, sv_cv, sv_beta, sv_g;
    Tensor ple_bak, sv_cn, rp_scratch;
    bool restore_window(int upto);

    struct Checkpoint { int pos = -1; std::vector<void*> conv, rec; void* ple = nullptr; int ids[2]; };
    std::vector<Checkpoint> ckpts_;
    Checkpoint& new_checkpoint(int at);
    void save_checkpoint(Checkpoint& c);
};

}  // namespace glm
