#pragma once
// What the engine needs from a model: GLM-5.3-Flash (model.h) or Qwen3.8-Flash-Next (q4.h). One sequence; the
// recurrent state lives in the model, the attention caches cover positions 0 .. position() - 1.
#include <vector>
#include "rt.h"

namespace glm {

class LM {
public:
    virtual ~LM() = default;
    // Q tokens at the current position; logits fp16 [Q, vocab()] on the device
    virtual const half* forward(const int* tokens, int Q) = 0;
    // n prompt tokens at the current position, no logits; ckpt_at (inside (pos, pos + n]): snapshot the recurrent
    // state there for prefix reuse
    virtual void prefill(const int* tokens, int n, int ckpt_at = -1) = 0;
    // back to the latest checkpoint at or below upto (false: none; the caller starts over with reset())
    virtual bool restore(int upto) = 0;
    virtual void reset() = 0;
    virtual int position() const = 0;
    virtual int max_context() const = 0;
    virtual int vocab() const = 0;
    // up to n tokens likely to follow tok, written to out, from the state of row hidden_row of the last forward(); tok
    // is the sampled token that row predicted. Returns how many (0: no drafter).
    virtual int draft(int tok, int hidden_row, int n, int* out) { (void)tok; (void)hidden_row; (void)n; (void)out; return 0; }
    int draft_tokens = 0;                // drafts per verify step (0: none)
};

// Several sequences at once (G4Engine schedules them): forward, prefill, restore, reset, position and draft act on the
// selected one; forward_rows reads rows of several together, each at its own position.
class MultiLM : public LM {
public:
    virtual void select(int seq) = 0;
    virtual int games() const = 0;
    virtual int position_of(int seq) const = 0;
    // where restore(upto) would leave sequence `seq`, without going there; -1 when it could not. A model that cannot
    // say answers upto, and the engine then ranks its sequences by their shared prefix alone
    virtual int reach(int seq, int upto) const { (void)seq; return upto; }
    // prefill whose checkpoint (at ckpt_at) is where a turn starts: a model that tells the two apart keeps these beside
    // the one at a prompt's end, for a later prompt that branches at that turn
    virtual void prefill_turn(const int* tokens, int n, int ckpt_at) { prefill(tokens, n, ckpt_at); }
    // Q rows, each sequence's rows together and in order; logits [Q, vocab()]
    virtual const half* forward_rows(const int* seq, const int* tokens, int Q) = 0;
    // draft chains from rows of the last forward_rows (no drafter: never asked, drafts stay empty)
    virtual void draft_rows(const int* seq, const int* tok, const int* row, const int* n, int R,
                            std::vector<std::vector<int>>& out) {
        (void)seq; (void)tok; (void)row; (void)n; (void)R; out.clear();
    }
};

}  // namespace glm
