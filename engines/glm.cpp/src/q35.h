#pragma once
// Qwen3.8-27B (model_type qwen3_5) on EXL3, docs/qwen35-spec.md: 64 layers (96 in the 40B upscale; any count works), each x += mixer(rms(x)) then
// x += MLP(rms(x)); the mixer is Gated DeltaNet (48 layers, the kernels of q4_ops) or full attention with an output
// gate (16 layers, L % 4 == 3). fp32 residual stream, fp16 activations into every linear, as exllamav3. RoPE is the
// interleaved 3-axis mRoPE: text has one position on all three axes, an image's tokens T, T + row, T + col, and the
// text after it continues from T + max(rows, cols). The vision tower is q35v.h. MTP drafting (the pack's mtp.* layer)
// and the GDN state rollback for its verify steps: docs/qwen35-spec.md section 7.
#include <memory>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <string>
#include <vector>
#include "exl3.h"
#include "lm.h"
#include "q4.h"
#include "st.h"

namespace glm {

namespace q35 {
constexpr int D = 5120, V = 248320, I = 17408;     // the layer count comes from config.json (27B: 64, 40B: 96)
constexpr int AH = 24, AKV = 4, AD = 256, ROT = 64;     // attention: query heads, KV heads, head dim, rotated dims
constexpr float EPS = 1e-6f;
constexpr int IMAGE_PAD = 248056;
constexpr int CH = 128;                                 // keys per piece of the plain attention kernel
}  // namespace q35

struct Q35Attn {
    Exl3Linear q, k, v, o;
    Exl3Linear qkv;                  // q, k, v merged (Exl3Linear::merge; then q, k, v are empty), or empty
    Tensor qn, kn;                   // fp32 (1 + w) [256]
};

struct Q35Block {
    int layer = 0;
    bool attn = false;
    Tensor w1, w2;                   // fp32 (1 + w) [D]: input_layernorm, post_attention_layernorm
    std::unique_ptr<Q4Gdn> gdn;
    std::unique_ptr<Q35Attn> att;
    Exl3Linear g, u, d;              // mlp gate, up, down
    Exl3Linear gu, gdn_in;           // merged: gate + up, GDN in_proj_qkv + in_proj_z (or empty)
};

// K (normed, roped), V fp16 [seqs, T, 4, 256]
struct Q35Cache {
    Tensor k, v;
    void init(int seqs, int max_tokens);
};

// A GDN layer's recurrent state, per slot: conv window bf16 [slots, 10240, 3], rec fp32 [slots, 48, 128, 128]. With MTP
// each sequence has two slots: a decode step reads one and writes the other, so a rejected draft's rows can be replayed
// from the state before the step (docs/qwen35-spec.md section 7).
struct Q35GdnState {
    Tensor conv, rec;
    Tensor kq, kb, kg;               // the last decode step's per-row inputs: qkv fp32 [qmax, 10240], beta, g fp32 [qmax, 48]
    void init(int slots, int qmax);
};

// The MTP draft layer (mtp.*): fc(cat(norm_e(embed(token p)), norm_h(hidden p - 1))), one gated full-attention block
// with its own KV cache (kv[L]), mtp.norm, the shared lm_head. hidden: the trunk's post-final-norm state.
struct Q35Mtp {
    Exl3Linear fc;                   // 10240 -> 5120
    Tensor pe, ph, fn;               // fp32 (1 + w): pre_fc_norm_embedding, pre_fc_norm_hidden, norm
    Q35Block blk;                    // blk.layer = L
};

// A page, encoded: emb fp32 [rows * cols, D], in the order the prompt's image tokens take them.
struct Q35Image {
    int rows = 0, cols = 0;
    std::vector<float> emb;
};

class Q35Model : public MultiLM {
public:
    // seqs: sequences held at once (G4Engine runs them together), max_tokens: the context of each
    Q35Model(const std::string& model_dir, int max_tokens, int seqs = 1);
    ~Q35Model() override;
    const half* forward(const int* tokens, int Q) override;
    void prefill(const int* tokens, int n, int ckpt_at = -1) override;
    bool restore(int upto) override;
    void reset() override;
    int position() const override { return seqs_[cur_].pos; }
    int max_context() const override { return max_tokens; }
    int vocab() const override { return q35::V; }
    void select(int seq) override { cur_ = seq; }
    int games() const override { return (int)seqs_.size(); }
    int position_of(int seq) const override { return seqs_[seq].pos; }
    // a sequence's rows together and in order (MTP verify: its token then its drafts); one recorded graph per row count
    const half* forward_rows(const int* seq, const int* tokens, int Q) override;
    // MTP draft chains for sequences seq[i], whose sampled token tok[i] sits at their position (row: unused, the
    // hidden state comes from the sequence's ring); greedy, a chain ends below mtp_min_p
    void draft_rows(const int* seq, const int* tok, const int* row, const int* n, int R,
                    std::vector<std::vector<int>>& out) override;
    bool has_mtp() const { return (bool)mtp; }
    float mtp_min_p = 0.25f;         // a chain ends where its drafts' probability product falls below this (Q35_MTP_MIN_P)
    // the draft head: by default the lm_head's 128-id blocks in mtp_blocks (the ids the model emits most, and the block of
    // <|im_end|> / <think> / </think>), gathered into a head of their own; Q35_MTP_HEAD=N: the first N ids instead (the
    // old head, 65536), 248320 = all; Q35_MTP_BLOCKS=file: block numbers from a file
    int mtp_head = 65536;            // prefix mode: columns (a multiple of 128); 0: block mode
    std::vector<int> mtp_blocks;     // block mode: block numbers, ascending
    long long mtp_rows = 0;          // MTP rows run (catch-up + drafts), for the tests

    // The selected sequence's images for its next prompt, in prompt order; reset() keeps them, the next set_images
    // replaces them. A prompt's image tokens take their embeddings and positions from these.
    void set_images(std::vector<Q35Image> imgs);
    const SafeTensors& tensors() const { return *st; }

    int prefill_chunk = 2048;        // rows per attention / GDN / MLP piece of a prompt
    // prompt kernels (Q35_PF_ATTN, Q35_PF_GDN; q35_gen kl compares them): attention 0 = the per-row decode kernel,
    // else a q35ops::attn_prompt variant (5: kernels/q35_fa.h); GDN 1 = the chunked delta rule (kernels/q35_gdn.h) and
    // the all-rows conv, 0 = the recurrent kernels
    int pf_attn = 15, pf_gdn = 1;
    // decode / verify attention (Q35_DEC_ATTN): 1 the ESIMD kernel (kernels/q35_dec.h) over dec_ns pieces of each row's
    // keys (Q35_DEC_NS), 0 the plain SYCL kernel (attn_core); set at load (the partials are sized for it)
    int dec_attn = 1, dec_ns = 32;
    // attention 13 and 15 read K^T from ktp, the selected sequence's keys of one layer pair-transposed
    // [4][128][TS][2], 15 also V from vtp [4][TS / 2][256][2] (2 KB a token each, TS = max_tokens rounded up to 16):
    // filled from the cache at a layer's first piece of a prefill call, then appended piece by piece
    Tensor ktp, vtp;
    const void* kt_cache = nullptr;  // the layer (its cache) ktp holds, for kt_seq, keys below kt_valid
    int kt_seq = -1, kt_valid = 0;
    std::vector<float>* dump = nullptr;   // debug: row 0's streams after each layer
    // a rejected draft's GDN replay (Q35_REPLAY): 1 every layer in two launches (q35ops::gdn_replay), 0 two per layer
    int replay_mode = 1;
    uint64_t state_hash();           // test: a hash of the selected sequence's live GDN state (conv windows, recurrent)

private:
    std::unique_ptr<SafeTensors> st;
    std::vector<Q35Block> blocks;
    Exl3Linear lm_head;
    Tensor norm_w;                   // fp32 (1 + w) [D]
    const bf16* embed = nullptr;     // host [V, D]
    std::vector<Q35GdnState> gst;
    std::vector<Q35Cache> kv;

    Workspace lin, act;
    int L = 0;                       // decoder layers, from config.json
    int max_tokens = 0, qmax = 8;
    Tensor X, y16, sub, g16, u16, a16, fin16, logits, part, pid_dev, pos_dev, seq_dev;
    float* x_host = nullptr;         // pinned [prefill_chunk, D]
    int* pid_h = nullptr;            // pinned [8 * prefill_chunk, 3]: T, H, W rope positions of the rows on their way up
    int* pos_h = nullptr;            // pinned [qmax]: each decode row's position, and
    int* seq_h = nullptr;            // pinned [qmax]: its sequence
    sycl::event x_ev, dec_ev;
    bool graphs_on = true;
    using ExecGraph = sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>;
    std::vector<std::unique_ptr<ExecGraph>> dec_graphs;    // by row count

    struct Checkpoint {
        int pos = -1, rope = 0, img = 0, img_row = 0;
        std::vector<void*> conv, rec;
        void* hid = nullptr;         // MTP: the hidden state, token and rope position at pos - 1
        bool hid_ok = false;
        int htok = 0, hpid[3] = {};
    };
    static constexpr int RING = 16;  // MTP: the last positions' hidden states, tokens and rope positions kept per sequence
    struct Seq {
        int pos = 0;
        int rope_next = 0, img_i = 0, img_row = 0;         // where the next token's position and embedding come from
        std::vector<Q35Image> images;
        std::vector<Checkpoint> ckpts;
        int slot = 0;                                      // its GDN state's live slot (of nslot)
        // the last decode step's rows of this sequence, while restore() can still replay a prefix of them
        int win_p0 = -1, win_first = 0, win_n = 0, win_rope = 0;
        // MTP
        int mtp_done = 0;                                  // MTP K/V right for positions below this
        int ring_pos[RING], ring_tok[RING], ring_pid[RING][3];
        Seq() { for (auto& p : ring_pos) p = -1; }
    };
    std::vector<Seq> seqs_;
    int cur_ = 0;
    // embeddings and rope positions, the sequence's cursors advance
    void rows_host(Seq& s, const int* tokens, int n, float* dst, int* pid);

    // prompt rows (in order, over the selected sequence): pdev null; decode rows: pdev, sdev per row
    void gdn_forward(const Q4Gdn& g, const Exl3Linear& gin, Q35GdnState& s, const half* y, int rows, float* out,
                     const int* sdev);
    void attn_forward(const Q35Attn& a, Q35Cache& c, const half* y, int rows, int p0, const int* pid, float* out,
                      const int* pdev, const int* sdev);
    void block_forward(Q35Block& b, float* Xd, int rows, int p0, const int* pid, const int* pdev, const int* sdev,
                       bool add_first = false, bool defer_last = false);
    void forward_dev(int Q);
    // linears that read the same input become one call (Q35_MERGE, default 1): fewer, larger launches
    void merge_linears(Q35Block& b);
    int merge_mask = 7;              // Q35_MERGE: 1 gate + up, 2 GDN qkv + z, 4 attention q + k + v (0: none)
    void grab(const float* p, int n);

    Checkpoint& new_checkpoint(Seq& s, int at);
    void save_checkpoint(Seq& s, Checkpoint& c);

    // GDN state slots and the decode steps' row groups
    int nslot = 1;
    int cslot(int s) const { return s * nslot + seqs_[s].slot; }
    size_t conv_slot_bytes() const { return (size_t)q4::GQKV * (q4::GCONV - 1) * 2; }
    size_t rec_slot_bytes() const { return (size_t)q4::GV * q4::GDK * q4::GDV * 4; }
    Tensor desc_dev, rdesc_dev;
    int* desc_h = nullptr;           // pinned [qmax][3]: this step's row groups (gdn_conv_seq)
    int* rdesc_h = nullptr;          // pinned [qmax][3]: a replay's
    int win_Q = 0;                   // rows of the step the windows belong to
    std::vector<std::unique_ptr<ExecGraph>> rep_graphs;    // by replay_mode and row count
    Tensor rep_layers, rep_cv;       // gdn_replay's per-layer pointers, its conv outputs
    int rep_nl = 0;
    sycl::event rep_ev;
    void replay_dev(int Q);
    bool restore_window(Seq& s, int upto);

    // MTP
    std::unique_ptr<Q35Mtp> mtp;
    Exl3Linear draft_head;           // block mode: the lm_head's columns of mtp_blocks
    void build_draft_head();
    int draft_id(int col) const { return mtp_head ? col : mtp_blocks[col / 128] * 128 + col % 128; }
    Tensor hring;                    // fp16 [seqs, RING, D]: post-norm hidden of the positions in ring_pos
    Tensor hfin;                     // fp16 [prefill_chunk + 1, D]: a prompt piece's hidden states, the one before first
    Tensor mx, mhin, mh16, mlog, mbest, mpos_dev, mseq_dev, mpid_dev, msrc_dev;
    float* memb_h = nullptr;         // pinned [MQ, D]: MTP rows' token embeddings
    int* mpos_h = nullptr;           // pinned [6 MQ]: positions, sequences, T H W, hidden sources
    int* mbest_h = nullptr;          // pinned [2 * qmax]
    int MQ = 0;                      // MTP decode rows a pass
    void load_mtp();
    void emb_row(int token, float* dst) const;
    // MTP rows r: token tok[r] at position pos[r] of sequence seq[r], rope pid, hidden mhin row r, into kv[L];
    // the last nhead rows' drafts to mbest_h. Pieces of at most MQ rows; rows of a sequence in order.
    // hid: hring row (s * RING + slot), q35ops::MH + j for mh16 row j (the pass before), -1 zeros
    struct MRow { int seq, pos, tok, pid[3]; int hid; };
    void mtp_decode(const std::vector<MRow>& rows, int nhead);
    void mtp_pass_dev(int n, int nhead);
    std::vector<std::unique_ptr<ExecGraph>> mtp_graphs;    // by n * (MQ + 1) + nhead
    // rows (catch-up) for positions [from, to) of sequence s from its ring; the ones it lacks are skipped
    void mtp_catchup(int s, int to, std::vector<MRow>& rows);
    void ring_put(int s, int p, int tok, const int* pid, const half* hid_dev);
    // a prompt piece's MTP rows (positions p0 .. p0 + nr - 1, hidden hfin rows 0 .. nr - 1) over the selected sequence
    void mtp_prefill_rows(const int* tokens, int nr, int p0, const int* pid_dev_rows, float* Xm);
};

}  // namespace glm
