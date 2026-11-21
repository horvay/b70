#pragma once
// Sampling's selection on the card (G4Engine: Gemma 4, Ripple, Qwen3.5 / 3.8): for each row of fp16 logits a kernel keeps
// the pool, every token within `window` logits of the row's best (after the row's penalties), at most CAP of the best,
// and sends back their ids and raw logits, the pool's size and what the window held beyond it (count, and the sum of
// exp((x - max) * inv_t) of those tokens above the sampler's floor). Sampler::sample_pool then runs the sampler's own code
// on the pool. A row's whole logits come back only when the pool cannot settle it (Sampler::sample_pool returns -1).
#include <vector>
#include "rt.h"

namespace glm {

struct PoolRow {
    float window = 0;      // raw logit units below the row's (penalized) best that the pool must cover
    float inv_t = 1;       // 1 / temperature, for the leftover mass (0: none needed)
    float log_floor = 0;   // the sampler's floor, log units after temperature (leftover mass counts tokens above it)
    float rep = 1;         // repetition penalty of the row's listed ids
    int pen_off = 0, pen_n = 0;    // the row's penalized ids in the step's list
};

struct PenEntry {
    int id;
    float sub;             // presence + frequency * count, as the host computes it
};

class GpuPool {
public:
    static constexpr int CAP = 4096, NB = 1024;
    GpuPool(int V, int qmax);
    ~GpuPool();
    // rows [Q, V] of logits on the device; blocks until the pools are on the host
    void run(sycl::queue& q, const half* logits, int Q, const std::vector<PoolRow>& rows, const std::vector<PenEntry>& pen);
    int n(int r) const { return meta_h_[r * 4]; }                       // pool size (-1: the window's top bin alone overflowed)
    int rest_n(int r) const { return meta_h_[r * 4 + 1]; }              // window tokens left out
    float rest_sum(int r) const;                                        // their exp((x - gmax) * inv_t) above the floor
    float gmax(int r) const;                                            // the row's best, penalized, raw units
    const int* ids(int r) const { return ids_h_ + (size_t)r * CAP; }
    const half* raw(int r) const { return raw_h_ + (size_t)r * CAP; }

private:
    int V_, qmax_;
    Tensor ids_d_, raw_d_, meta_d_, rows_d_, pen_d_, scratch_;
    int* ids_h_ = nullptr;
    half* raw_h_ = nullptr;
    int* meta_h_ = nullptr;
    PoolRow* rows_h_ = nullptr;
    PenEntry* pen_h_ = nullptr;
    int pen_cap_ = 0;
};

}  // namespace glm
