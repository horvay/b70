#pragma once
// Fine-tuning Gemma 4's assistant drafter (gemma-4-31B-it-assistant) against a target model's own output (llm/g4draft).
//
// The target (G4Model) reads a training sequence with its capture on: every row's final normed hidden state, the last
// sliding layer's K / V, and (rebuilt here) the last full layer's K / V. A draft chain at anchor position p is then
// replayed as the server runs it: step k reads token S[p + k - 1] and h (the target's hidden state at p - 1 for the first
// step, the drafter's own post_projection output after that), RoPE at p, keys up to p - 1. Its loss is the KL divergence
// from the target's next-token distribution at row p + k - 1 (temperature 1, soft-capped logits), backpropagated through
// the whole chain. Everything trains in fp32 (oneMKL sgemm); the head (the tied embedding) stays frozen.
//
// Metrics replay acceptance exactly: the recorded tokens are the target's own samples, and the server accepts a draft
// when the sampled token equals it, so a chain's accepted length is its count of leading steps whose argmax equals the
// recorded token (with the server's margin rule: the chain stops after a step whose top two logits are within margin).
#include <memory>
#include <string>
#include <vector>
#include "g4.h"

namespace glm {

struct G4DraftStats {
    double kl = 0;                 // sum over valid row-steps
    int rows = 0;                  // valid row-steps
    int anchors = 0;
    double accepted = 0;           // leading matches, no margin rule (steps cap)
    double accepted_m = 0;         // with the margin rule
    double drafted_m = 0;          // tokens drafted with the margin rule
    std::vector<double> hit;       // per step: argmax == recorded token (teacher forced)
    std::vector<int> hit_n;
    std::vector<double> thit;      // per step: the target's own argmax == recorded token (the ceiling of a greedy drafter)
    double oracle = 0;             // leading steps a drafter guessing the target's argmax would get accepted
    void add(const G4DraftStats& o);
};

class G4DraftTrainer {
public:
    // anchors: chain starts per chunk; steps: chain length; ctx: the longest sequence
    G4DraftTrainer(G4Model& target, const std::string& drafter_dir, int anchors, int steps, int ctx);
    ~G4DraftTrainer();
    // capture buffers for the target (set_capture) and the full layer's keys of a freshly read sequence [0, len)
    const G4Model::Capture* capture() const { return &cap_; }
    void begin_sequence(int len);
    // anchors p (ascending, each with p >= 1 and p + 1 < len) of the current sequence; backward too when train
    G4DraftStats chunk(const std::vector<int>& seq, const std::vector<int>& anchors, bool train, float margin);
    // AdamW over the gradients summed since the last step (scaled by 1 / chunks); returns the grad norm before clipping
    double step(float lr, int chunks, float clip = 1.0f);
    // drafter folder with the trained tensors (bf16), everything else copied
    void save_model(const std::string& out_dir) const;
    // optimizer state (params, moments, step) to resume a run
    void save_state(const std::string& path) const;
    bool load_state(const std::string& path);
    // gradient check: P += eps * u(seed) (u a fixed pseudo-random direction, entries in [-1, 1]); sum_i G_i u_i; G = 0
    void perturb(float eps, uint32_t seed);
    double grad_dot(uint32_t seed);
    void zero_grad();
    int64_t trainable() const { return np_; }
    int opt_steps() const { return t_; }

private:
    G4Model& m_;
    std::string src_dir_;
    int A_, K_, ctx_;
    G4Model::Capture cap_;
    struct Layer {
        bool full = false;
        int hd = 0;
        // offsets into the flat parameter buffer
        int64_t q = 0, o = 0, g = 0, u = 0, d = 0, w_in = 0, w_pa = 0, w_pf = 0, w_ff = 0, qn = 0;
        float scalar = 1.0f;
    };
    std::vector<Layer> L_;
    int64_t pre_ = 0, post_ = 0, wnorm_ = 0, np_ = 0;
    std::vector<std::pair<std::string, int64_t>> names_;   // trainable tensor names and their offsets
    float *P_ = nullptr, *G_ = nullptr, *M_ = nullptr, *V_ = nullptr;
    float* head_ = nullptr;                                 // frozen [V, 1024]
    int t_ = 0;
    // sequence: the full layer's K / V fp32 [ctx, 4, 512]; sliding block fp32
    float *fk_ = nullptr, *fv_ = nullptr, *sk_ = nullptr, *sv_ = nullptr;
    half *tmpk_ = nullptr, *tmpv_ = nullptr;
    int sblock_ = 0;
    // activations: one arena, sliced per step and layer
    float* arena_ = nullptr;
    struct Act;
    std::vector<std::unique_ptr<Act>> acts_;               // per step
    // per chunk
    float *Eh_ = nullptr, *Ed_ = nullptr;                   // host pinned / device [A, D] target embeddings
    half* tl_ = nullptr;                                    // target logits [A + K, V]
    float* dl_ = nullptr;                                   // drafter logits / their gradient [A, V]
    float *S_ = nullptr, *dP_ = nullptr;                    // attention scores [budget]
    float *qp_ = nullptr, *op_ = nullptr, *dop_ = nullptr;  // permuted queries / outputs [A, 32, 512]
    float* delta_ = nullptr;                                // [A, 32]
    float *bx_ = nullptr, *bx2_ = nullptr, *bh_ = nullptr, *bq_ = nullptr, *bo_ = nullptr, *bg_ = nullptr, *bu_ = nullptr,
          *bm_ = nullptr, *bd_ = nullptr, *bz_ = nullptr, *tq_ = nullptr, *dyacc_ = nullptr;    // backward scratch
    int *pos_d_ = nullptr, *row_d_ = nullptr;               // per row: anchor position, target-logit row
    float* w_d_ = nullptr;                                  // per row: loss weight
    float* rowstat_ = nullptr;                              // per row: kl, gap
    int* rowarg_ = nullptr;                                 // per row: argmax
    size_t sbudget_ = 0;

    void forward_step(int k, int R, const int* fin_rows, bool first);
    void backward_step(int k, int R, bool last);
    void attn_fwd(const Layer& l, float* qr, float* o, float* lse, int R, int k0, int nk);
    void attn_bwd(const Layer& l, const float* qr, const float* o, const float* lse, float* d_o, float* dq, int R, int k0,
                  int nk);
};

}  // namespace glm
