#pragma once
// Token sampling on the host from a row of fp16 logits, in exllamav3's ComboSampler order: repetition penalty
// (Transformers style over all past ids), presence / frequency penalties, temperature, min-p, top-k, top-p, sample.
// temperature 0 or top_k 1: argmax.
#include <cstdint>
#include <random>
#include <vector>
#include "rt.h"

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

private:
    SamplingParams p_;
    int V_;
    std::mt19937_64 rng_;
    std::vector<int> count_;          // past occurrences per id
    std::vector<int> past_;           // distinct ids seen
    std::vector<float> l_;
    std::vector<int> idx_;
};

}  // namespace glm
