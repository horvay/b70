#include "sampler.h"
#include <algorithm>
#include <cmath>
#include <immintrin.h>

namespace glm {

Sampler::Sampler(const SamplingParams& p, int vocab) : p_(p), V_(vocab), count_(vocab, 0), l_(vocab) {
    rng_.seed(p.has_seed ? p.seed : std::random_device{}());
}

void Sampler::seen(int id) {
    if (id < 0 || id >= V_) return;
    if (count_[id]++ == 0) past_.push_back(id);
}

void f16_to_f32(const half* in, float* out, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(out + i, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(in + i))));
    for (; i < n; i++) out[i] = (float)in[i];
}

void top_k(const float* x, size_t n, int k, int* idx, float* val) {
    // a min-heap of the k best; AVX2 skips the runs below its root (almost everything: a few hundred updates a row)
    for (int i = 0; i < k; i++) { idx[i] = i; val[i] = x[i]; }
    auto down = [&](int i) {
        for (;;) {
            int l = 2 * i + 1, r = l + 1, m = i;
            if (l < k && val[l] < val[m]) m = l;
            if (r < k && val[r] < val[m]) m = r;
            if (m == i) return;
            std::swap(val[i], val[m]);
            std::swap(idx[i], idx[m]);
            i = m;
        }
    };
    for (int i = k / 2 - 1; i >= 0; i--) down(i);
    size_t j = k;
    for (; j + 8 <= n; j += 8) {
        int mask = _mm256_movemask_ps(_mm256_cmp_ps(_mm256_loadu_ps(x + j), _mm256_set1_ps(val[0]), _CMP_GT_OQ));
        while (mask) {
            int b = __builtin_ctz(mask);
            mask &= mask - 1;
            if (x[j + b] > val[0]) { val[0] = x[j + b]; idx[0] = (int)(j + b); down(0); }
        }
    }
    for (; j < n; j++)
        if (x[j] > val[0]) { val[0] = x[j]; idx[0] = (int)j; down(0); }
}

int top2(const float* x, size_t n, float& best, float& second) {
    // the largest value by vectors, then its first index and the largest of the rest
    __m256 mv = _mm256_set1_ps(-INFINITY);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) mv = _mm256_max_ps(mv, _mm256_loadu_ps(x + i));
    float m8[8];
    _mm256_storeu_ps(m8, mv);
    float mx = *std::max_element(m8, m8 + 8);
    for (; i < n; i++) mx = std::max(mx, x[i]);
    int at = (int)(std::find(x, x + n, mx) - x);
    __m256 sv = _mm256_set1_ps(-INFINITY);
    i = 0;
    for (; i + 8 <= (size_t)at; i += 8) sv = _mm256_max_ps(sv, _mm256_loadu_ps(x + i));
    float s = -INFINITY;
    for (; i < (size_t)at; i++) s = std::max(s, x[i]);
    i = at + 1;
    for (; i + 8 <= n; i += 8) sv = _mm256_max_ps(sv, _mm256_loadu_ps(x + i));
    for (; i < n; i++) s = std::max(s, x[i]);
    _mm256_storeu_ps(m8, sv);
    best = mx;
    second = std::max(s, *std::max_element(m8, m8 + 8));
    return at;
}

int Sampler::sample(const half* logits) {
    f16_to_f32(logits, l_.data(), V_);
    // penalties over the past ids
    if (p_.rep_p != 1.0f || p_.pres_p != 0.0f || p_.freq_p != 0.0f)
        for (int v : past_) {
            float& x = l_[v];
            if (p_.rep_p != 1.0f) x = x > 0 ? x / p_.rep_p : x * p_.rep_p;
            x -= p_.pres_p + p_.freq_p * count_[v];
        }
    if (p_.temperature <= 0.0f || p_.top_k == 1) {
        float bv, sv;
        return top2(l_.data(), V_, bv, sv);
    }
    // temperature, softmax
    float inv_t = 1.0f / p_.temperature, mx = -INFINITY;
    {
        __m256 m = _mm256_set1_ps(-INFINITY), it = _mm256_set1_ps(inv_t);
        int v = 0;
        for (; v + 8 <= V_; v += 8) {
            __m256 x = _mm256_mul_ps(_mm256_loadu_ps(&l_[v]), it);
            _mm256_storeu_ps(&l_[v], x);
            m = _mm256_max_ps(m, x);
        }
        float m8[8];
        _mm256_storeu_ps(m8, m);
        mx = *std::max_element(m8, m8 + 8);
        for (; v < V_; v++) { l_[v] *= inv_t; mx = std::max(mx, l_[v]); }
    }
    // candidates: min-p against the top probability (exp(0) = 1 after the max shift), and a floor that keeps the
    // sort short without changing top-p / top-k results (tokens below 1e-9 of the top never reach either). The test
    // runs on the log value, so only the few survivors pay for an exp (262K exps were 2 ms a token).
    float floor_p = std::max(p_.min_p, 1e-9f);
    const float log_floor = mx + std::log(floor_p);
    idx_.clear();
    for (int v = 0; v < V_; v++)
        if (l_[v] >= log_floor) { l_[v] = std::exp(l_[v] - mx); idx_.push_back(v); }
    // the order is total (ties by id), so with top-k the first k of a partial sort are the full sort's
    auto cmp = [&](int a, int b) { return l_[a] > l_[b] || (l_[a] == l_[b] && a < b); };
    if (p_.top_k > 0 && (size_t)p_.top_k < idx_.size())
        std::partial_sort(idx_.begin(), idx_.begin() + p_.top_k, idx_.end(), cmp);
    else
        std::sort(idx_.begin(), idx_.end(), cmp);
    size_t n = idx_.size();
    if (p_.top_k > 0) n = std::min(n, (size_t)p_.top_k);
    if (p_.top_p < 1.0f) {
        double kept = 0, tot = 0;
        for (size_t i = 0; i < n; i++) tot += l_[idx_[i]];
        // renormalised over the min-p / top-k survivors, as the chained steps do
        size_t m = 0;
        for (; m < n; m++) {
            kept += l_[idx_[m]];
            if (kept >= p_.top_p * tot) { m++; break; }
        }
        n = std::max<size_t>(1, m);
    }
    double tot = 0;
    for (size_t i = 0; i < n; i++) tot += l_[idx_[i]];
    std::uniform_real_distribution<double> U(0.0, tot);
    double r = U(rng_), acc = 0;
    for (size_t i = 0; i < n; i++) {
        acc += l_[idx_[i]];
        if (r < acc) return idx_[i];
    }
    return idx_[n - 1];
}

}  // namespace glm
