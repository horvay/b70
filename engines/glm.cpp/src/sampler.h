#pragma once
// Token sampling on the host from a row of fp16 logits, in exllamav3's ComboSampler order: repetition penalty
// (Transformers style over all past ids), presence / frequency penalties, temperature, min-p, top-k, top-p, sample.
// temperature 0 or top_k 1: argmax.
#include <cstdint>
#include <random>
#include <vector>
#include "rt.h"
#include "sample_gpu.h"

namespace glm {

struct SamplingParams {
    float temperature = 1.0f, top_p = 1.0f, min_p = 0.0f;
    int top_k = 0;
    float rep_p = 1.0f, pres_p = 0.0f, freq_p = 0.0f;
    bool has_seed = false;
    uint64_t seed = 0;
};

// the host's fast paths over a row of logits (AVX2, F16C)
void f16_to_f32(const half* in, float* out, size_t n);
// the first index of the largest value; best, and the largest of the others
int top2(const float* x, size_t n, float& best, float& second);

// the k largest of a row, in no order
void top_k(const float* x, size_t n, int k, int* idx, float* val);

class Sampler {
public:
    // vocab: ids at or above it (the lm_head's padding) are never sampled
    Sampler(const SamplingParams& p, int vocab);
    // past: every token so far (prompt and generated), for the penalties; seen() adds them incrementally
    void seen(int id);
    int sample(const half* logits);
    // The GPU pool (sample_gpu.h) for a row: the window it must cover, and the penalized ids (pending: tokens sampled
    // before this row in the same step, not yet seen())
    void pool_row(PoolRow& pr, std::vector<PenEntry>& pen, const int* pending, int npending) const;
    // sample() over a row's pool (ids, raw logits, n; rest_n window tokens left out, rest_sum their mass above the floor
    // relative to gmax): the same token sample() gives, by the same code, or -1 when the pool cannot settle it (the
    // caller then passes the whole row to sample()). Only top_k unset with top_p < 1 uses rest_sum (the top-p total).
    int sample_pool(const int* ids, const half* raw, int n, int rest_n, float rest_sum, float gmax);
    // rows settled by pools, rows that needed the whole row, rows whose top-p total took the GPU's rest_sum
    static long long n_pool, n_full, n_rest;

private:
    SamplingParams p_;
    int V_;
    std::mt19937_64 rng_;
    std::vector<int> count_;          // past occurrences per id
    std::vector<int> past_;           // distinct ids seen
    std::vector<float> l_;
    std::vector<int> idx_;
    std::vector<float> pv_;           // a pool's values
    std::vector<int> pi_;             // pool positions
    float penalize(int id, float x) const;
};

}  // namespace glm
