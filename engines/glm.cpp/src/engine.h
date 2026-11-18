#pragma once
// One generation at a time over the model, with prefix reuse between requests: a prompt sharing a prefix with what the
// model last read continues from there (or from the latest recurrent-state checkpoint inside the shared prefix), so a
// chat follow-up only reads its new tokens plus at most ckpt_every.
#include <atomic>
#include <functional>
#include <string>
#include <vector>
#include "lm.h"
#include "sampler.h"

namespace glm {

struct GenRequest {
    std::vector<int> prompt;
    SamplingParams sampling;
    int max_new = 256;
    std::vector<int> stop_ids;                          // end the reply (not emitted)
    std::function<bool(int id)> on_token;               // false: stop (a stop string matched, the client left)
    std::function<void(int done, int total, int cached)> on_progress;   // while the prompt is read
    const std::atomic<bool>* cancelled = nullptr;
    bool fresh = false;                                 // no prefix reuse: the prompt's images are not in its token ids
    // G4Engine: where the prompt's last turns start, ascending. The next prompt mostly branches at one of them (a
    // reply's thought, a prefilled thought and a hidden pass's instruction are all dropped), so each gets a checkpoint.
    std::vector<int> turn_marks;
    std::function<void()> before;                       // on the engine's thread, ahead of the prompt (image encoding)
};

struct GenResult {
    int prompt_tokens = 0, completion_tokens = 0, cached_tokens = 0;
    int game = 0, shared_tokens = 0;                    // G4Engine: the game it ran in, and the prefix they shared
    std::string finish = "stop";                        // stop | length | cancelled
    double prompt_s = 0, total_s = 0, decode_tok_s = 0;
    int drafted = 0, accepted = 0;                      // draft tokens read, and those the sampled reply agreed with
    int stop_id = -1;                                   // the stop token that ended the reply
};

class Engine {
public:
    explicit Engine(LM& m) : m_(m) {}
    GenResult run(const GenRequest& r);
    int ckpt_every = 256;                                // checkpoint at the last multiple of this in each prompt

private:
    LM& m_;
    std::vector<int> seq_;                               // the tokens at positions 0 .. position() - 1
    half* logits_h_ = nullptr;
    int logits_cap_ = 1;                                 // rows logits_h_ holds
};

}  // namespace glm
