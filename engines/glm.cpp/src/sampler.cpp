#include "sampler.h"
#include <algorithm>
#include <atomic>
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

long long Sampler::n_pool = 0, Sampler::n_full = 0, Sampler::n_rest = 0;

// sample()'s penalty of an id seen before, the same float operations
float Sampler::penalize(int id, float x) const {
    if (p_.rep_p != 1.0f) x = x > 0 ? x / p_.rep_p : x * p_.rep_p;
    x -= p_.pres_p + p_.freq_p * count_[id];
    return x;
}

void Sampler::pool_row(PoolRow& pr, std::vector<PenEntry>& pen, const int* pending, int npending) const {
    const bool greedy = p_.temperature <= 0.0f || p_.top_k == 1;
    const bool penal = p_.rep_p != 1.0f || p_.pres_p != 0.0f || p_.freq_p != 0.0f;
    pr.rep = p_.rep_p;
    pr.pen_off = (int)pen.size();
    if (penal) {
        // the counts this row's sampling will see: the past, and the tokens sampled before it in this step
        std::vector<std::pair<int, int>> extra;
        for (int i = 0; i < npending; i++) {
            const int id = pending[i];
            if (id < 0 || id >= V_) continue;
            auto e = std::find_if(extra.begin(), extra.end(), [&](auto& p) { return p.first == id; });
            if (e == extra.end()) extra.push_back({id, 1});
            else e->second++;
        }
        for (int v : past_) {
            int c = count_[v];
            for (auto& e : extra) if (e.first == v) { c += e.second; e.second = 0; }
            pen.push_back({v, p_.pres_p + p_.freq_p * c});
        }
        for (auto& e : extra) if (e.second) pen.push_back({e.first, p_.pres_p + p_.freq_p * e.second});
    }
    pr.pen_n = (int)pen.size() - pr.pen_off;
    if (greedy) {
        // the ties at the top (0.01: GPU and host penalties may differ by an ulp)
        pr.window = 0.01f;
        pr.inv_t = 0.0f;
        pr.log_floor = 0.0f;
    } else {
        const float lf = std::log(std::max(p_.min_p, 1e-9f));
        pr.window = -lf * p_.temperature * 1.001f + 0.02f;     // every token above the floor, with room for roundings
        pr.inv_t = 1.0f / p_.temperature;
        pr.log_floor = lf;
    }
}

int Sampler::sample_pool(const int* ids, const half* raw, int n, int rest_n, float rest_sum, float gmax) {
    if (n <= 0) return -1;
    const bool penal = p_.rep_p != 1.0f || p_.pres_p != 0.0f || p_.freq_p != 0.0f;
    pv_.resize(n);
    float xmin = INFINITY;
    for (int i = 0; i < n; i++) {
        float x = (float)raw[i];
        if (penal && count_[ids[i]] > 0) x = penalize(ids[i], x);
        pv_[i] = x;
        xmin = std::min(xmin, x);
    }
    if (p_.temperature <= 0.0f || p_.top_k == 1) {
        // top2's answer: the largest value, its lowest id
        int best = -1;
        for (int i = 0; i < n; i++)
            if (best < 0 || pv_[i] > pv_[best] || (pv_[i] == pv_[best] && ids[i] < ids[best])) best = i;
        n_pool++;
        return ids[best];
    }
    const bool complete = rest_n == 0;
    const float inv_t = 1.0f / p_.temperature;
    float mx = -INFINITY;
    for (int i = 0; i < n; i++) { pv_[i] *= inv_t; mx = std::max(mx, pv_[i]); }
    const float floor_p = std::max(p_.min_p, 1e-9f);
    const float log_floor = mx + std::log(floor_p);
    pi_.clear();
    for (int i = 0; i < n; i++)
        if (pv_[i] >= log_floor) pi_.push_back(i);
    // the k-th candidate must sit clear of the pool's edge, where a token left out could rank by a rounding
    const float edge = xmin * inv_t;
    std::vector<float> xs(n);
    for (int i : pi_) { xs[i] = pv_[i]; pv_[i] = std::exp(pv_[i] - mx); }
    auto cmp = [&](int a, int b) { return pv_[a] > pv_[b] || (pv_[a] == pv_[b] && ids[a] < ids[b]); };
    size_t m = pi_.size();
    if (p_.top_k > 0 && (size_t)p_.top_k < pi_.size()) {
        std::partial_sort(pi_.begin(), pi_.begin() + p_.top_k, pi_.end(), cmp);
        if (!complete && !(xs[pi_[p_.top_k - 1]] > edge + 1e-3f * std::max(1.0f, inv_t))) return -1;
    } else {
        if (!complete) {
            if (p_.top_k > 0) return -1;              // fewer candidates in the pool than k: the k-th is outside
            if (p_.top_p >= 1.0f) return -1;          // the whole window is a candidate
        }
        std::sort(pi_.begin(), pi_.end(), cmp);
    }
    if (p_.top_k > 0) m = std::min(m, (size_t)p_.top_k);
    bool used_rest = false;
    if (p_.top_p < 1.0f) {
        double kept = 0, tot = 0;
        for (size_t i = 0; i < m; i++) tot += pv_[pi_[i]];
        if (!complete && p_.top_k <= 0) {
            // the window's tokens past the pool, summed on the card, after the pool's own (they are all smaller)
            tot += (double)rest_sum * std::exp((double)gmax * inv_t - (double)mx);
            used_rest = true;
        }
        size_t c = 0;
        bool cut = false;
        for (; c < m; c++) {
            kept += pv_[pi_[c]];
            if (kept >= p_.top_p * tot) { c++; cut = true; break; }
        }
        if (!cut && !complete && p_.top_k <= 0) return -1;     // the cut lies past the pool
        m = std::max<size_t>(1, c);
    }
    double tot = 0;
    for (size_t i = 0; i < m; i++) tot += pv_[pi_[i]];
    std::uniform_real_distribution<double> U(0.0, tot);
    double r = U(rng_), acc = 0;
    n_pool++;
    if (used_rest) n_rest++;
    for (size_t i = 0; i < m; i++) {
        acc += pv_[pi_[i]];
        if (r < acc) return ids[pi_[i]];
    }
    return ids[pi_[m - 1]];
}

}  // namespace glm
