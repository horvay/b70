#pragma once
// Gemma 4 31B dense (model_type gemma4, e.g. Gemma 4 31B-based) on EXL3, text only. The forward is
// llm/exl3conv/notes/gemma4-forward.md §2 with exllamav3's rounding points: fp32 residual stream, fp16 activations into
// every linear, fp32 norms / softmax / RoPE.
//
// Layers: 50 sliding-window (32 query heads x 256, 16 KV heads, window 1024 including the query, RoPE theta 1e4) and
// 10 full-attention (L % 6 == 5: 32 x 512, 4 KV heads, no v_proj, proportional RoPE: dims j and j + 256 rotate for
// j < 64, theta 1e6). Attention scale 1. Every layer: x = (x + post_attn(attn(in(x))) + post_ff(mlp(pre_ff(x)))) *
// layer_scalar, the MLP gelu_tanh(gate) * up. Embedding E[id] * 73.5 (rounded to bf16), tied fp16 lm_head, final
// logit softcap 30.
//
// Caches, per sequence:
//  - sliding layers: K (normed, roped) and V (normed) fp16 in a ring of RING positions (the window plus room for a
//    verify step's rows), slot p % RING;
//  - full layers: the raw k_proj output P fp16 [T, 4, 512], 1 / rms(P) per head, and Kr, the 128 dims RoPE rotates of
//    k_norm * P. K and V are functions of P, so decode reads P once for both and multiplies by r in the kernel (prompts
//    rebuild K and V into a scratch per key block). 5 KB a token and layer against 8 KB for K and V.
// Prompts run layer-major over super-chunks of rows (one dequant of each layer's weights per super-chunk).
#include <memory>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <string>
#include <vector>
#include "exl3.h"
#include "lm.h"
#include "st.h"

namespace glm {

namespace g4 {
constexpr int D = 5376, L = 60, NH = 32, I = 21504, V = 262144;
constexpr int SKV = 16, SD = 256;              // sliding layers: KV heads, head dim
constexpr int FKV = 4, FD = 512, FROT = 64;    // full layers: KV heads, head dim, rotated pairs
constexpr int WIN = 1024;                      // sliding window (keys p - 1023 .. p)
constexpr int RING = 1024 + 128;               // sliding cache slots: the window + up to 129 rows in one step
constexpr float EPS = 1e-6f, EMBED_MUL = 73.5f, SOFTCAP = 30.0f;
constexpr int QMAX = 16;                       // rows of one forward() (decode, draft verify)
inline bool is_full(int li) { return li % 6 == 5; }
}  // namespace g4

struct G4Layer {
    bool full = false;
    Exl3Linear q, k, v, o, g, u, d;            // v only on sliding layers
    Tensor w_in, w_pa, w_pf, w_ff;             // bf16 [D]: input, post_attention, pre_feedforward, post_feedforward
    Tensor qn, kn;                             // bf16 [head dim]: q_norm, k_norm
    Tensor kn16;                               // full layers: k_norm fp16 [512]
    float scalar = 1.0f;                       // layer_scalar
};

// every game's caches, game after game in one buffer (TS rows of a full layer's, RING of a sliding layer's)
struct G4Cache {
    std::vector<Tensor> k, v;                  // per sliding layer: fp16 [S * RING, 16, 256]
    std::vector<Tensor> p, kr, r;              // per full layer: fp16 [S * TS, 4, 512], fp16 [S * TS, 4, 128], fp32 [S * TS, 4] (1 / rms)
    void init(int TS, int S);
};

// Google's assistant model (gemma-4-31B-it-assistant): 4 layers of hidden 1024 whose query heads read the target's last
// sliding and last full layers' caches (no K / V of its own, nothing written). A step: z = pre_projection(E[tok] * 73.5 |
// h), the layers, y = final norm, logits = y @ its own tied embedding, h = post_projection(y); greedy. h starts as the
// target's normed hidden state. Plain fp16 weights.
struct G4DraftLayer {
    bool full = false;
    Tensor q, o, g, u, d;                      // fp16 [out, in]
    Tensor w_in, w_pa, w_pf, w_ff, qn;         // bf16
    float scalar = 1.0f;
};

class G4Model : public MultiLM {
public:
    // max_tokens: context of each game; drafter: an assistant model's folder, or empty; games: sequences held at once
    G4Model(const std::string& model_dir, int max_tokens, const std::string& drafter = "", int games = 1);
    ~G4Model() override;
    const half* forward(const int* tokens, int Q) override;
    void prefill(const int* tokens, int n, int ckpt_at = -1) override;
    void prefill_turn(const int* tokens, int n, int ckpt_at) override;
    bool restore(int upto) override;
    void reset() override;
    int position() const override { return pos; }
    int max_context() const override { return max_tokens; }
    int vocab() const override { return g4::V; }
    int draft(int tok, int hidden_row, int n, int* out) override;
    // the sequence forward / prefill / restore / reset / position / draft act on
    void select(int game);
    int games() const { return games_; }
    int position_of(int game) const { return game == cur_ ? pos : pos_[game]; }
    int reach(int game, int upto) const override;
    // Q rows of several games, each game's rows together and in order, read at each game's position; logits [Q, V]
    const half* forward_rows(const int* game, const int* tokens, int Q);
    // draft chains for R rows of the last forward_rows: chain i starts from its game's token tok[i], from hidden row
    // row[i], and stops at n[i] tokens or a doubtful one; out[i] gets them. Chains run together, row by row.
    void draft_rows(const int* game, const int* tok, const int* row, const int* n, int R, std::vector<std::vector<int>>& out);

    // llm/g4draft's trainer. While a capture is set, a prompt writes each row's final normed hidden state (fin, fp16 [*, D])
    // and the last sliding layer's K / V (fp16 [*, 16, 256]) at the row's position in it (rows: its capacity).
    struct Capture { half* fin = nullptr; half* sk = nullptr; half* sv = nullptr; int rows = 0; };
    void set_capture(const Capture* c) { capture_ = c; }
    // the last full layer's K / V ([n, 4, 512] fp16 each) at positions k0 .. k0 + n - 1 of the selected game
    void last_full_kv(int k0, int n, half* K, half* V);
    // logits fp16 [R, V] (soft-capped, as forward's) of R final-normed rows
    void head_rows(const half* fin, int R, half* out);
    // E[id] * 73.5 rounded as the model rounds it, fp32 [n, D]
    void embed_rows(const int* tokens, int n, float* dst) { embed_host(tokens, n, dst); }
    void set_turn_checkpoints(int n) { turn_ckpts = n; }

    int super_chunk = 8192;                    // prompt rows per layer-major pass (GLM_PREFILL_SC)
    int q_chunk = 2048;                        // query rows per attention launch in prompts
    int key_block = 16384;                     // full-layer keys rebuilt per pass in prompts
    float draft_margin = 3.0f;                 // a draft chain ends at a token this close to its runner-up (logits)



private:
    std::unique_ptr<SafeTensors> st;
    std::vector<G4Layer> layers;
    Tensor w_norm;                             // bf16 [D]
    Tensor head16;                             // fp16 [V, D] (a 16-bit head), or
    Exl3Linear head_q;                         // an EXL3 head
    bool head_exl3 = false;
    const bf16* embed = nullptr;               // host [V, D] (mapped)
    Tensor rope_full;                          // fp32 [max_tokens, 64, 2]: cos, sin of the full layers' rotated pairs
    G4Cache cache;
    int pos = 0, max_tokens = 0;               // the selected game's position; context of each game
    int games_ = 1, cur_ = 0, TS = 0;          // games; the selected one; a game's rows in a full layer's cache
    std::vector<int> pos_;                     // the other games' positions

    Workspace lin;                             // Exl3Linear temporaries
    // decode / verify rows (QMAX)
    Tensor X, h16, q16, k16, v16, o16, a16, g16, u16, d32, fin16, logits, part, srope_d;
    // prompt super-chunk (buffers shared between steps where one's output replaces another's: see prefill_sc)
    Tensor Xp, hp, qp, kp, vp, gp, up, fo32, flse, srope_p;
    float* x_host = nullptr;                   // pinned [super_chunk, D]: embedding rows
    int* pos_h = nullptr;                      // pinned [2 * QMAX]: row positions, row games
    Tensor pos_dev;                            // int32 [2 * QMAX]: the same
    sycl::event x_ev, dec_ev;
    bool graphs_on = true;
    std::vector<std::unique_ptr<sycl::ext::oneapi::experimental::command_graph<
        sycl::ext::oneapi::experimental::graph_state::executable>>> graphs;   // by Q

    void embed_host(const int* tokens, int n, float* dst);
    void decode_layer(int li, float* x, half* h, int Q, const int* pdev, const int* sdev, half* fin);
    void forward_dev(int Q);
    void prefill_sc(const int* tokens, int n);

    // drafter
    std::unique_ptr<SafeTensors> dst;
    std::vector<G4DraftLayer> dlayers;
    Tensor dpre, dpost, dhead, dw_norm;        // fp16 [1024, 10752], [5376, 1024], [V, 1024]; bf16 [1024]
    Tensor dz, de, dx, dh, dq, dok, dg, du, dd, dy, dprev, dlog, dpos;   // DQ rows; dpos int32 [3, DQ]: rope position, last key, game
    Tensor dsrope;
    half* dz_h = nullptr;                      // pinned [DQ, D]: the tokens' embeddings, fp16
    int* dpos_h = nullptr;                     // pinned: dpos
    half* dlog_h = nullptr;                    // pinned [DQ, V]
    std::vector<std::unique_ptr<sycl::ext::oneapi::experimental::command_graph<
        sycl::ext::oneapi::experimental::graph_state::executable>>> dgraphs;   // by rows
    void load_drafter(const std::string& dir);
    void draft_dev(int R);

    struct Checkpoint { int pos = -1; bool turn = false; std::vector<void*> k, v; };
    std::vector<std::vector<Checkpoint>> ckpts_;   // per game: at most turn_ckpts at turn starts and 1 at a prompt's end
    int turn_ckpts = 2;                            // G4_TURN_CKPTS
    bool ckpt_turn_ = false;                       // the checkpoint of the prefill under way is a turn's start
    const Capture* capture_ = nullptr;
    void save_checkpoint(int at, bool turn);
};

}  // namespace glm
