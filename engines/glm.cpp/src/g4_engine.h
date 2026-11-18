#pragma once
// Several conversations at once over a MultiLM's sequences (Gemma 4, Qwen3.8-27B): a request goes to the free game that
// can bring back the most of its tokens; each step reads a piece of one prompt (decode never stalls longer than a piece) and one
// batched decode step for every game past its prompt, with the drafts their chains are sure of.
#include <functional>
#include <memory>
#include <vector>
#include "engine.h"
#include "g4.h"
#include "sampler.h"

namespace glm {

class G4Engine {
public:
    explicit G4Engine(MultiLM& m);
    bool has_free() const;                       // a request can start now
    bool busy() const;                           // step() has work
    void submit(GenRequest r, std::function<void(const GenResult&)> done);
    void step();
    // cancelled requests leave their games now (step() does this too): called before a waiting request picks a game,
    // so one that was cancelled and sent again longer goes back to the game that holds its tokens
    void reap();
    // the longest prefix the prompt shares with a game that is running a request (running) or free (not)
    int shared_prefix(const std::vector<int>& prompt, bool running) const;
    int max_rows = 8;                            // rows of a decode step (more goes to the slower batched kernels)
    int piece = 2048;                            // prompt tokens between decode steps, while other games decode
    int mark_gap = 128;                          // a checkpoint this close behind another is not worth its copy
    double stale_s = 1800;                       // a game left alone this long is no conversation worth keeping

private:
    struct Game {
        bool active = false, reading = false;
        GenRequest r;
        std::function<void(const GenResult&)> done;
        std::vector<int> seq;                    // the tokens at positions 0 .. position - 1
        int next = 0;                            // reading: the prompt's next token to read
        int start = 0;
        std::vector<int> marks;                  // reading: the turn starts ahead that get a checkpoint, ascending
        bool end_mark = true;                    // reading: the prompt's end gets one too
        std::unique_ptr<Sampler> sm;
        int tok = 0, seed_row = -1;
        GenResult res;
        double t0 = 0, t_first = 0;
        uint64_t used = 0;                       // when this game last finished
        double t_used = 0;                       // the same, in seconds
        int continued = 0;                       // requests in a row that went on from what the game held
    };
    MultiLM& m_;
    std::vector<Game> games_;
    uint64_t clock_ = 0;
    half* logits_h_ = nullptr;
    std::vector<std::vector<int>> drafts_;
    void finish(int g);
};

}  // namespace glm
