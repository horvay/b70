#pragma once
// Ripple-65B (model_type ricursive) on EXL3, text only. The forward is the checkpoint's modeling_ricursive.py with exllamav3's
// rounding points, as Gemma 4 (g4.h): fp32 residual stream, fp16 activations into every linear, fp32 norms / softmax / RoPE.
//
// Layers: 80, each 80 query heads x 128 over 8 KV heads. 64 sliding-window (window 1024 including the query, RoPE theta 1e4),
// 16 global (L % 5 == 4: no RoPE). Scores q . k / sqrt(128). A layer, n = input norm(x):
//     q, k = q_proj(n), k_proj(n), rms norm per head (weights q_norm, k_norm), RoPE on the sliding layers
//     v = v_proj(lerp(E, n, sigmoid(value_res_lambda))), E the row's scaled embedding
//     x += post_attn(o_proj(attention * sigmoid(g_proj(n))))
//     x += post_ff(down(silu(min(gate(m), 20)) * clamp(up(m), +-20))), m = pre_ff(x)
// Embedding E[id] * embed_scale (rounded to bf16), its own fp16 or EXL3 lm_head, no softcap.
//
// Caches, per sequence: sliding layers K and V fp16 in a ring of RING positions, slot p % RING; global layers K and V fp16 of
// every position (64 KB a token). Prompts run layer-major over super-chunks of rows.
#include <memory>
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <string>
#include <vector>
#include "exl3.h"
#include "lm.h"
#include "st.h"

namespace glm {

namespace rp {
constexpr int D = 5120, L = 80, NH = 80, NKV = 8, HD = 128, I = 40960, V = 100352;
constexpr int QD = NH * HD, KD = NKV * HD;
constexpr int WIN = 1024;                      // sliding window (keys p - 1023 .. p)
constexpr int RING = 1024 + 128;               // sliding cache slots: the window + up to 129 rows in one step
constexpr int QMAX = 16;                       // rows of one forward()
constexpr float EPS = 1e-6f, CLAMP = 20.0f;
// DFlash2 drafter: 5 layers like the target's, a block of 16 rows (anchor, then masks), reading taps of the target's stream
constexpr int DL = 5, DBLOCK = 16, DTOPK = 32, DRANK = 256, DMASK = 100278;
constexpr int DTAPS[5] = {1, 20, 39, 58, 77};
constexpr int DFC = 5 * D;
constexpr int DRING_S = 65 * 64, DWIN_S = 4096;          // sliding layers' ring, window
constexpr int DRING_G = 65 * 256, DWIN_G = 16384;        // the last layer: no RoPE, the 16K window it was trained on
inline bool is_global(int li) { return li % 5 == 4; }
inline int global_index(int li) { return li / 5; }
inline int sliding_index(int li) { return li - li / 5; }
}  // namespace rp

// an EXL3 linear, or a plain fp16 weight [out, in] (a 16-bit head; bf16 weights, loaded as fp16, for tests on the original
// checkpoint)
struct RpLin {
    Exl3Linear e;
    Tensor w;
    int in = 0, out = 0;
    bool dense = false;
    static RpLin load(const SafeTensors& st, const std::string& prefix);
    size_t ws_bytes(int M) const { return dense ? 0 : e.ws_bytes(M); }
    size_t bytes() const { return dense ? w.bytes() : e.trellis.bytes() + e.suh.bytes() + e.svh.bytes(); }
    // x row stride xs (-1: in)
    void forward(sycl::queue& q, Workspace& ws, const half* x, int M, void* y, DType ydt, int xs = -1) const;
};

struct RpDraftLayer {
    bool global = false;
    RpLin q, k, v, o, gate, up, down, ac, mc;  // ac, mc: the convolutions' kernel projections [2 sides, 4 taps, 320 groups]
    Tensor ac_base, mc_base;                   // bf16 [2, 4, D]
    Tensor w_in, w_pa, qn, kn;
    Tensor K, V;                               // fp16 [S * ring, 8, 128]
    int ring = 0, win = 0;
};

struct RpLayer {
    bool global = false;
    RpLin q, k, v, g, o, gate, up, down;
    Tensor w_in, w_pa, w_pf, w_ff;             // bf16 [D]
    Tensor qn, kn;                             // bf16 [128]
    float blend = 0.0f;                        // sigmoid(value_res_lambda)
};

class RpModel : public MultiLM {
public:
    // max_tokens: context of each game; drafter: a DFlash2 folder, or empty; games: sequences held at once; the card's room
    // decides what fits
    RpModel(const std::string& model_dir, int max_tokens, const std::string& drafter = "", int games = 1);
    ~RpModel() override;
    const half* forward(const int* tokens, int Q) override;
    void prefill(const int* tokens, int n, int ckpt_at = -1) override;
    bool restore(int upto) override;
    void reset() override;
    int position() const override { return pos; }
    int max_context() const override { return max_tokens; }
    int vocab() const override { return rp::V; }
    void select(int game) override;
    int games() const override { return games_; }
    int position_of(int game) const override { return game == cur_ ? pos : pos_[game]; }
    const half* forward_rows(const int* game, const int* tokens, int Q) override;
    int draft(int tok, int hidden_row, int n, int* out) override;
    // tests: the drafter's rows after its first `layers` layers, raw fp32 / fp16 files in dir
    void dump_draft(const std::string& dir, int tok, int layers);
    int draft_debug_layers = 99;
    void draft_rows(const int* game, const int* tok, const int* row, const int* n, int R, std::vector<std::vector<int>>& out) override;

    int super_chunk = 8192;                    // prompt rows per layer-major pass (GLM_PREFILL_SC)
    int q_chunk = 2048;                        // query rows per attention launch in prompts
    float draft_margin = 1.0f;                 // a draft chain ends at a token this close to its runner-up (RP_DRAFT_MARGIN)

private:
    std::unique_ptr<SafeTensors> st;
    int nl = rp::L;                            // layers (RP_TEST_LAYERS: the first few of a checkpoint, for tests)
    std::vector<RpLayer> layers;
    Tensor w_norm;                             // bf16 [D]
    RpLin head;
    const bf16* embed = nullptr;               // host [V, D] (mapped)
    float embed_scale = 1.0f;
    std::vector<Tensor> ck, cv;                // sliding: fp16 [S * RING, 8, 128] per layer
    std::vector<Tensor> gk, gv;                // global: fp16 [S * TS, 8, 128] per layer
    int pos = 0, max_tokens = 0;
    int games_ = 1, cur_ = 0, TS = 0;
    std::vector<int> pos_;

    Workspace lin;
    // decode rows
    Tensor X, E0, h16, vin, q16, k16, v16, gt16, o16, a16, g16, u16, d32, fin16, logits, part, srope_d;
    // prompt super-chunk
    Tensor Xp, E0p, hp, vinp, qp, gtp, kp, vp, gp, up, srope_p;
    float* x_host = nullptr;                   // pinned [super_chunk, D]: embedding rows
    int* pos_h = nullptr;                      // pinned [2 * QMAX]: row positions, row games
    Tensor pos_dev;                            // int32 [2 * QMAX]: the same
    sycl::event x_ev, dec_ev;
    bool graphs_on = true;
    std::vector<std::unique_ptr<sycl::ext::oneapi::experimental::command_graph<
        sycl::ext::oneapi::experimental::graph_state::executable>>> graphs;   // by Q

    void embed_host(const int* tokens, int n, float* dst);
    void decode_layer(int li, float* x, const float* e0, half* h, int Q, const int* pdev, const int* sdev, half* fin);
    size_t draft_bytes(const std::string& dir) const;
    void forward_dev(int Q);
    void prefill_sc(const int* tokens, int n);

    // drafter
    std::unique_ptr<SafeTensors> dst;
    std::vector<RpDraftLayer> dlayers;
    RpLin dfc, dsel;
    Tensor dw_hn, dw_norm;                     // bf16 [D]: hidden_norm, norm
    const bf16* dpred = nullptr;               // host [V, 256] (mapped)
    const bf16* dsucc = nullptr;
    int tap_of[rp::L];                         // a layer's tap index, or -1
    Tensor tap16, tapn, dk16, dv16;            // verify rows: taps fp16 [QMAX, DFC], fc + hidden_norm [QMAX, D], K, V
    Tensor tapp, tapnp, dkp, dvp;              // prompt rows: the same
    Tensor dX, dy, dy2, dq, dok, dgt, dup, dd, dcv, dstate, dgate, dlog, dpos, dsrope, dscale;     // a block's rows
    float* dx_h = nullptr;                     // pinned [DBLOCK, D]
    int* dpos_h = nullptr;                     // pinned [2 * DBLOCK]: row positions, games
    half* dlog_h = nullptr;                    // pinned [DBLOCK, V]
    half* dgate_h = nullptr;                   // pinned [DBLOCK, 256]
    std::vector<std::unique_ptr<sycl::ext::oneapi::experimental::command_graph<
        sycl::ext::oneapi::experimental::graph_state::executable>>> dgraph;
    void load_drafter(const std::string& dir);
    void draft_dev();
    void draft_kv_rows(const half* tap, half* tn, half* k, half* v, int n, const int* pdev, int p0, const int* sdev, const float* rope, bool prompt);
    int draft_chain(int game, int tok, int n, int* out);

    struct Checkpoint { int pos = -1; std::vector<void*> k, v, dk, dv; };
    std::vector<std::vector<Checkpoint>> ckpts_;   // per game: at most 3 (one game), 1 (several)
    void save_checkpoint(int at);
};

}  // namespace glm
