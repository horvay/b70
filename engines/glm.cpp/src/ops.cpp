#include "ops.h"
#include <cmath>
#include <oneapi/mkl/blas.hpp>

namespace glm::ops {

namespace {
constexpr int HD = 4 * 4096;      // mHC: 4 streams of 4096
using sycl::nd_item;
using sycl::nd_range;

template <class T> float f(T v) { return static_cast<float>(v); }
}  // namespace

// ------------------------------------------------------------------------------------------------ RMSNorm
template <class TX, class TY>
void rms_norm(sycl::queue& q, const TX* x, const bf16* w, TY* y, int R, int D, float eps) {
    constexpr int WG = 256;
    q.parallel_for(nd_range<1>(R * WG, WG), [=](nd_item<1> it) {
        int r = it.get_group(0), t = it.get_local_id(0);
        const TX* xr = x + (size_t)r * D;
        auto ld = [&](int c) {
            float v = f(xr[c]);
            if constexpr (std::is_same_v<TX, half>) v = sycl::fmin(sycl::fmax(v, -65504.f), 65504.f);
            return v;
        };
        float acc = 0;
        for (int c = t; c < D; c += WG) { float v = ld(c); acc += v * v; }
        acc = sycl::reduce_over_group(it.get_group(), acc, sycl::plus<float>());
        float rmf = 1.0f / sycl::sqrt(acc / D + eps);
        for (int c = t; c < D; c += WG) {
            float v = ld(c);
            if (w) v = v * (f(w[c]) + 0.0f);
            y[(size_t)r * D + c] = static_cast<TY>(v * rmf);
        }
    });
}
template void rms_norm<half, half>(sycl::queue&, const half*, const bf16*, half*, int, int, float);
template void rms_norm<float, half>(sycl::queue&, const float*, const bf16*, half*, int, int, float);
template void rms_norm<half, float>(sycl::queue&, const half*, const bf16*, float*, int, int, float);
template void rms_norm<float, float>(sycl::queue&, const float*, const bf16*, float*, int, int, float);

// ------------------------------------------------------------------------------------------------ mHC
// chunks per row as the PyTorch stack's hc_mix_num_chunks (R = 1: 64 chunks of 256 columns)
int hc_mix_chunks(int R) {
    int chunks_a = std::min(128, std::max(1, 512 / std::max(R, 1)));
    int chunk_cols = ((HD / chunks_a) + 255) / 256 * 256;
    return (HD + chunk_cols - 1) / chunk_cols;
}

void hc_mix(sycl::queue& q, const float* streams, const HcSite& s, float* post, float* comb, half* collapsed,
            float* part, int R, float rms_eps, float hc_eps, int iters) {
    constexpr int WG = 256;
    const int nc = hc_mix_chunks(R);
    const int chunk_cols = HD / nc;
    // partials: part[r][chunk][0..23] = sum fn[m, c] * s[c], part[..][24] = sum s^2
    q.parallel_for(nd_range<1>((size_t)R * nc * WG, WG), [=](nd_item<1> it) {
        int g = it.get_group(0), t = it.get_local_id(0);
        int r = g / nc, ch = g % nc;
        const float* sr = streams + (size_t)r * HD;
        float acc[25];
        for (int m = 0; m < 25; m++) acc[m] = 0;
        for (int c = ch * chunk_cols + t; c < (ch + 1) * chunk_cols; c += WG) {
            float sv = sr[c];
            acc[24] += sv * sv;
            if (s.fn_f16) {
                const half* fn = static_cast<const half*>(s.fn);
                for (int m = 0; m < 24; m++) acc[m] += f(fn[(size_t)m * HD + c]) * sv;
            } else {
                const float* fn = static_cast<const float*>(s.fn);
                for (int m = 0; m < 24; m++) acc[m] += fn[(size_t)m * HD + c] * sv;
            }
        }
        float* out = part + ((size_t)r * nc + ch) * 25;
        for (int m = 0; m < 25; m++) {
            float v = sycl::reduce_over_group(it.get_group(), acc[m], sycl::plus<float>());
            if (t == 0) out[m] = v;
        }
    });
    // final: mix, sinkhorn, post / comb, collapsed
    const float* base = s.base;
    const float* scale = s.scale;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sh(sycl::range<1>(32), h);
        h.parallel_for(nd_range<1>((size_t)R * WG, WG), [=](nd_item<1> it) {
            int r = it.get_group(0), t = it.get_local_id(0);
            if (t < 25) {
                float v = 0;
                for (int ch = 0; ch < nc; ch++) v += part[((size_t)r * nc + ch) * 25 + t];
                sh[t] = v;
            }
            sycl::group_barrier(it.get_group());
            if (t == 0) {
                float rmr = 1.0f / sycl::sqrt(sh[24] / HD + rms_eps);
                float z[24], sig[24];
                for (int m = 0; m < 24; m++) {
                    float sc = scale[m < 4 ? 0 : (m < 8 ? 1 : 2)];
                    z[m] = sh[m] * rmr * sc + base[m];
                    sig[m] = 1.0f / (1.0f + sycl::exp(-z[m]));
                }
                for (int hh = 0; hh < 4; hh++) post[r * 4 + hh] = 2.0f * sig[4 + hh];
                float C[4][4];
                for (int i = 0; i < 4; i++) {
                    float mx = z[8 + 4 * i];
                    for (int j = 1; j < 4; j++) mx = sycl::fmax(mx, z[8 + 4 * i + j]);
                    float e[4], se = 0;
                    for (int j = 0; j < 4; j++) { e[j] = sycl::exp(z[8 + 4 * i + j] - mx); se += e[j]; }
                    for (int j = 0; j < 4; j++) C[i][j] = e[j] / se + hc_eps;
                }
                auto colpass = [&] {
                    for (int j = 0; j < 4; j++) {
                        float cs = C[0][j] + C[1][j] + C[2][j] + C[3][j];
                        for (int i = 0; i < 4; i++) C[i][j] = C[i][j] / (cs + hc_eps);
                    }
                };
                colpass();
                for (int k = 0; k < iters - 1; k++) {
                    for (int i = 0; i < 4; i++) {
                        float rs = C[i][0] + C[i][1] + C[i][2] + C[i][3];
                        for (int j = 0; j < 4; j++) C[i][j] = C[i][j] / (rs + hc_eps);
                    }
                    colpass();
                }
                for (int i = 0; i < 4; i++)
                    for (int j = 0; j < 4; j++) comb[r * 16 + i * 4 + j] = C[i][j];
                for (int hh = 0; hh < 4; hh++) sh[hh] = sig[hh] + hc_eps;     // pre
            }
            sycl::group_barrier(it.get_group());
            const float* sr = streams + (size_t)r * HD;
            float p0 = sh[0], p1 = sh[1], p2 = sh[2], p3 = sh[3];
            for (int d = t; d < 4096; d += WG)
                collapsed[(size_t)r * 4096 + d] =
                    static_cast<half>(sr[d] * p0 + sr[4096 + d] * p1 + sr[8192 + d] * p2 + sr[12288 + d] * p3);
        });
    });
}

void hc_apply(sycl::queue& q, float* streams, const float* y, const float* post, const float* comb, int R) {
    q.parallel_for(sycl::range<2>(R, 4096), [=](sycl::item<2> it) {
        int r = it[0], d = it[1];
        float* s = streams + (size_t)r * HD;
        float x0 = s[d], x1 = s[4096 + d], x2 = s[8192 + d], x3 = s[12288 + d];
        float yv = y[(size_t)r * 4096 + d];
        const float* C = comb + r * 16;     // [g][h]
        for (int h = 0; h < 4; h++)
            s[h * 4096 + d] = post[r * 4 + h] * yv + (C[0 * 4 + h] * x0 + C[1 * 4 + h] * x1 + C[2 * 4 + h] * x2 +
                                                      C[3 * 4 + h] * x3);
    });
}

void hc_mean(sycl::queue& q, const float* streams, float* out, int R) {
    q.parallel_for(sycl::range<2>(R, 4096), [=](sycl::item<2> it) {
        const float* s = streams + (size_t)it[0] * HD;
        int d = it[1];
        out[(size_t)it[0] * 4096 + d] = (((s[d] + s[4096 + d]) + s[8192 + d]) + s[12288 + d]) / 4.0f;
    });
}

void hc_expand(sycl::queue& q, const float* x, float* streams, int R) {
    q.parallel_for(sycl::range<2>(R, 4096), [=](sycl::item<2> it) {
        float v = x[(size_t)it[0] * 4096 + it[1]];
        float* s = streams + (size_t)it[0] * HD;
        for (int h = 0; h < 4; h++) s[h * 4096 + it[1]] = v;
    });
}

// ------------------------------------------------------------------------------------------------ MLP pieces
void swiglu(sycl::queue& q, const half* g, const half* u, half* a, size_t n, float lim) {
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        float gv = f(g[i]), uv = f(u[i]);
        float s = gv / (1.0f + sycl::exp(-gv));
        s = sycl::fmin(s, lim);
        uv = sycl::fmin(sycl::fmax(uv, -lim), lim);
        a[i] = static_cast<half>(s * uv);
    });
}

template <class TY>
void gemv_f16(sycl::queue& q, const half* x, int M, int K, const half* W, int N, TY* y, bool round16) {
    if (M > 16) {
        // many rows: one fp16 GEMM (fp32 accumulation, fp16 out, as oneDNN in the PyTorch stack), widened if needed
        static thread_local std::pair<half*, size_t> tmp{nullptr, 0};
        half* out16;
        if constexpr (std::is_same_v<TY, half>) {
            out16 = y;
        } else {
            size_t need = (size_t)M * N;
            if (tmp.second < need) {
                q.wait();                  // a pending kernel may still read the old buffer
                if (tmp.first) sycl::free(tmp.first, q);
                tmp = {sycl::malloc_device<half>(need, q), need};
                GLM_CHECK(tmp.first, "gemv_f16: %zu-byte fp16 staging allocation failed", need * sizeof(half));
            }
            out16 = tmp.first;
        }
        oneapi::mkl::blas::row_major::gemm(q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::trans, M, N, K,
                                           half(1.0f), x, K, W, K, half(0.0f), out16, N);
        if constexpr (!std::is_same_v<TY, half>) {
            (void)round16;     // the GEMM output is fp16 either way
            q.parallel_for(sycl::range<1>((size_t)M * N), [=](sycl::id<1> i) { y[i] = f(out16[i]); });
        }
        return;
    }
    if (N < 1024) {
        // few outputs (router, gates, LoRA A): a 128-wide work-group per output so the dot over K fills the GPU
        constexpr int WGN = 128;
        q.parallel_for(nd_range<1>((size_t)N * WGN, WGN), [=](nd_item<1> it) {
            int n = it.get_group(0), l = it.get_local_id(0);
            const half* wr = W + (size_t)n * K;
            for (int m = 0; m < M; m++) {
                const half* xr = x + (size_t)m * K;
                float acc = 0;
                for (int k = l; k < K; k += WGN) acc += f(xr[k]) * f(wr[k]);
                acc = sycl::reduce_over_group(it.get_group(), acc, sycl::plus<float>());
                if (l == 0) {
                    if constexpr (std::is_same_v<TY, float>) y[(size_t)m * N + n] = round16 ? f(static_cast<half>(acc)) : acc;
                    else y[(size_t)m * N + n] = static_cast<TY>(acc);
                }
            }
        });
        return;
    }
    // one sub-group of 16 per output column n, all M rows; W row n is contiguous
    constexpr int SG = 16, WG = 128, PER = WG / SG;
    size_t groups = (N + PER - 1) / PER;
    q.parallel_for(nd_range<1>(groups * WG, WG), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        auto sg = it.get_sub_group();
        int n = it.get_group(0) * PER + sg.get_group_id()[0];
        int l = sg.get_local_id()[0];
        if (n >= N) return;
        const half* wr = W + (size_t)n * K;
        for (int m = 0; m < M; m++) {
            const half* xr = x + (size_t)m * K;
            float acc = 0;
            for (int k = l; k < K; k += SG) acc += f(xr[k]) * f(wr[k]);
            acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
            if (l == 0) {
                if constexpr (std::is_same_v<TY, float>) y[(size_t)m * N + n] = round16 ? f(static_cast<half>(acc)) : acc;
                else y[(size_t)m * N + n] = static_cast<TY>(acc);
            }
        }
    });
}
template void gemv_f16<half>(sycl::queue&, const half*, int, int, const half*, int, half*, bool);
template void gemv_f16<float>(sycl::queue&, const half*, int, int, const half*, int, float*, bool);

void route_exact(sycl::queue& q, const half* y, int Q, const half* gate, const half* bias_c, int E, int k, float rsf,
                 half* logits, int* sel, float* w) {
    gemv_f16<half>(q, y, Q, 4096, gate, E, logits);
    q.parallel_for(sycl::range<1>(Q), [=](sycl::id<1> r) {
        const half* lg = logits + (size_t)r * E;
        float key[512];
        bool used[512];
        for (int e = 0; e < E; e++) {
            float s = 1.0f / (1.0f + sycl::exp(-f(lg[e])));
            key[e] = s + f(bias_c[e]);
            used[e] = false;
        }
        int idx[16];
        float sw[16], sum = 0;
        // top-k descending; ties to the lower index (torch.topk on XPU: see the parity notes)
        for (int j = 0; j < k; j++) {
            int best = -1;
            for (int e = 0; e < E; e++)
                if (!used[e] && (best < 0 || key[e] > key[best])) best = e;
            used[best] = true;
            idx[j] = best;
            sw[j] = 1.0f / (1.0f + sycl::exp(-f(lg[best])));
            sum += sw[j];
        }
        for (int j = 0; j < k; j++) {
            sel[r * k + j] = idx[j];
            w[r * k + j] = f(static_cast<half>(sw[j] * rsf / (sum + 1e-20f)));
        }
    });
}

void route_exact2(sycl::queue& q, const half* lc, const half* ln, int Q, const half* bias_c, const float* esb_n, int E,
                  int k, int w1, float rsf, int* sel, float* w, int64_t* picks) {
    constexpr int WG = 512;
    GLM_CHECK(E <= WG && k <= 16, "route_exact2: %d experts, top-%d", E, k);
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<int, 1> idx(sycl::range<1>(16), h);
        h.parallel_for(sycl::nd_range<1>((size_t)Q * WG, WG), [=](sycl::nd_item<1> it) {
            int r = it.get_group(0), e = it.get_local_id(0);
            auto grp = it.get_group();
            bool on = e < E;
            auto pack = [](float v, int i) {
                uint32_t b = sycl::bit_cast<uint32_t>(v);
                b = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
                return ((uint64_t)b << 32) | (uint32_t)(0xFFFFFFFFu - (uint32_t)i);
            };
            const half* lg = lc + (size_t)r * E;
            float key = on ? 1.0f / (1.0f + sycl::exp(-f(lg[e]))) + f(bias_c[e]) : -INFINITY;
            bool used = !on;
            for (int t = 0; t < k; t++) {
                uint64_t m = sycl::reduce_over_group(grp, used ? uint64_t(0) : pack(key, e), sycl::maximum<uint64_t>());
                int i = (int)(0xFFFFFFFFu - (uint32_t)m);
                if (e == i) used = true;
                if (e == 0) idx[t] = i;
            }
            sycl::group_barrier(grp);
            if (e == 0) {
                float sw[16], sum = 0;
                for (int t = 0; t < k; t++) {
                    sw[t] = 1.0f / (1.0f + sycl::exp(-f(lg[idx[t]])));
                    sum += sw[t];
                }
                for (int t = 0; t < k; t++) {
                    sel[r * k + t] = idx[t];
                    w[r * k + t] = f(static_cast<half>(sw[t] * rsf / (sum + 1e-20f)));
                    picks[r * k + t] = idx[t];
                    picks[(size_t)Q * k + r * k + t] = idx[t];
                }
            }
            if (ln) {
                float kn = on ? 1.0f / (1.0f + sycl::exp(-f(ln[(size_t)r * E + e]))) + (esb_n ? esb_n[e] : 0.0f)
                              : -INFINITY;
                used = !on;
                for (int t = 0; t < w1; t++) {
                    uint64_t m = sycl::reduce_over_group(grp, used ? uint64_t(0) : pack(kn, e), sycl::maximum<uint64_t>());
                    int i = (int)(0xFFFFFFFFu - (uint32_t)m);
                    if (e == i) used = true;
                    if (e == 0) picks[(size_t)2 * Q * k + r * w1 + t] = i;
                }
            }
        });
    });
}

void route_tier(sycl::queue& q, const half* lc, const half* ln, int Q, const float* esb, const float* esb_n,
                const float* bonus, const float* bonus_n, int E, int k, int w1, float rsf, int* sel, float* w,
                int64_t* picks) {
    // one work-group per row, one work-item per expert; each top-k round is a group max over (score, ~index) packed
    // into 64 bits, so ties go to the lower index (as the k rounds of argmax-and-mask before)
    constexpr int WG = 512;
    GLM_CHECK(E <= WG, "route_tier: %d experts", E);
    q.parallel_for(sycl::nd_range<1>((size_t)Q * WG, WG), [=](sycl::nd_item<1> it) {
        int r = it.get_group(0), e = it.get_local_id(0);
        auto grp = it.get_group();
        bool on = e < E;
        auto pack = [](float v, int i) {
            uint32_t b = sycl::bit_cast<uint32_t>(v);
            b = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
            return ((uint64_t)b << 32) | (uint32_t)(0xFFFFFFFFu - (uint32_t)i);
        };
        float s = on ? 1.0f / (1.0f + sycl::exp(-f(lc[(size_t)r * E + e]))) : 0.0f;
        float key = on ? s + (esb ? esb[e] : 0.0f) : -INFINITY;
        float kb = on ? key + (bonus ? bonus[e] : 0.0f) : -INFINITY;
        // exact top-k
        bool used = !on;
        for (int t = 0; t < k; t++) {
            uint64_t m = sycl::reduce_over_group(grp, used ? uint64_t(0) : pack(key, e), sycl::maximum<uint64_t>());
            int i = (int)(0xFFFFFFFFu - (uint32_t)m);
            if (e == i) used = true;
            if (e == 0) picks[(size_t)Q * k + r * k + t] = i;
        }
        // selected top-k (with the bonus) and the weights
        used = !on;
        float sum = 0, mine = 0;
        int slot = -1;
        for (int t = 0; t < k; t++) {
            uint64_t m = sycl::reduce_over_group(grp, used ? uint64_t(0) : pack(kb, e), sycl::maximum<uint64_t>());
            int i = (int)(0xFFFFFFFFu - (uint32_t)m);
            if (e == i) { used = true; slot = t; mine = s; }
            if (e == 0) { sel[r * k + t] = i; picks[r * k + t] = i; }
        }
        sum = sycl::reduce_over_group(grp, slot >= 0 ? mine : 0.0f, sycl::plus<float>());
        if (slot >= 0) w[r * k + slot] = mine * rsf / (sum + 1e-20f);
        if (ln) {
            float kn = on ? 1.0f / (1.0f + sycl::exp(-f(ln[(size_t)r * E + e]))) + (esb_n ? esb_n[e] : 0.0f) +
                            (bonus_n ? bonus_n[e] : 0.0f)
                          : -INFINITY;
            used = !on;
            for (int t = 0; t < w1; t++) {
                uint64_t m = sycl::reduce_over_group(grp, used ? uint64_t(0) : pack(kn, e), sycl::maximum<uint64_t>());
                int i = (int)(0xFFFFFFFFu - (uint32_t)m);
                if (e == i) used = true;
                if (e == 0) picks[(size_t)2 * Q * k + r * w1 + t] = i;
            }
        }
    });
}

void permute_picks(sycl::queue& q, const int* sel, const float* w, const int* perm, int n, int* sel_p, float* w_p,
                   int* pair_of) {
    GLM_CHECK(n <= 16, "permute_picks: %d", n);
    struct P { int v[16]; } pm{};
    for (int j = 0; j < n; j++) pm.v[j] = perm[j];
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> j) {
        int k = pm.v[j];
        sel_p[j] = sel[k];
        w_p[j] = w[k];
        pair_of[k] = (int)j;
    });
}

void f32_to_f16(sycl::queue& q, const float* x, half* y, size_t n) {
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { y[i] = static_cast<half>(x[i]); });
}

void add_f32(sycl::queue& q, float* y, const float* x, size_t n) {
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { y[i] += x[i]; });
}

}  // namespace glm::ops
