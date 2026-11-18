#include "g4_draft.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <oneapi/mkl/blas.hpp>
#include "json.h"
#include "st.h"

namespace glm {

using sycl::nd_item;
using sycl::nd_range;
using sycl::range;

namespace {
constexpr int D = g4::D, V = g4::V, NH = g4::NH;
constexpr int DH = 1024, DI = 8192, Z = 2 * D;
constexpr float EPS = 1e-6f;

// ------------------------------------------------------------------------------------------------ GEMM (fp32, row-major)

void gemm(sycl::queue& q, bool ta, bool tb, int64_t M, int64_t N, int64_t K, float alpha, const float* A, int64_t lda,
          const float* B, int64_t ldb, float beta, float* C, int64_t ldc) {
    using namespace oneapi::mkl;
    blas::row_major::gemm(q, ta ? transpose::trans : transpose::nontrans, tb ? transpose::trans : transpose::nontrans, M, N,
                          K, alpha, A, lda, B, ldb, beta, C, ldc);
}
// y [R, out] = x [R, in] W^T (W [out, in])
void lin(sycl::queue& q, const float* x, int R, int in, const float* W, int out, float* y, float beta = 0) {
    gemm(q, false, true, R, out, in, 1, x, in, W, in, beta, y, out);
}
// dx [R, in] (+)= dy [R, out] W ; dy's row stride lds
void lin_dx(sycl::queue& q, const float* dy, int lds, int R, int out, const float* W, int in, float* dx, float beta) {
    gemm(q, false, false, R, in, out, 1, dy, lds, W, in, beta, dx, in);
}
// dW [out, in] += dy^T x
void lin_dw(sycl::queue& q, const float* dy, int lds, int R, int out, const float* x, int in, float* dW) {
    gemm(q, true, false, out, in, R, 1, dy, lds, x, in, 1, dW, in);
}

// ------------------------------------------------------------------------------------------------ norms

// y = x r w (w may be null), r = 1 / sqrt(mean(x^2) + eps) -> rinv[r]
void rms_fwd(sycl::queue& q, const float* x, const float* w, float* y, float* rinv, int R, int Dm) {
    constexpr int NW = 256;
    q.parallel_for(nd_range<1>((size_t)R * NW, NW), [=](nd_item<1> it) {
        const int r = it.get_group(0), t = it.get_local_id(0);
        const float* xr = x + (size_t)r * Dm;
        float* yr = y + (size_t)r * Dm;
        float acc = 0;
        for (int c = t; c < Dm; c += NW) acc += xr[c] * xr[c];
        acc = sycl::reduce_over_group(it.get_group(), acc, sycl::plus<float>());
        const float rm = 1.0f / sycl::sqrt(acc / Dm + EPS);
        if (t == 0) rinv[r] = rm;
        for (int c = t; c < Dm; c += NW) yr[c] = xr[c] * rm * (w ? w[c] : 1.0f);
    });
}

// xo = (xi + rms(a) wa) * scale; ra[r] = 1 / rms(a)
void addnorm_fwd(sycl::queue& q, const float* xi, const float* a, const float* wa, float scale, float* xo, float* ra, int R,
                 int Dm) {
    constexpr int NW = 256;
    q.parallel_for(nd_range<1>((size_t)R * NW, NW), [=](nd_item<1> it) {
        const int r = it.get_group(0), t = it.get_local_id(0);
        const float* ar = a + (size_t)r * Dm;
        float acc = 0;
        for (int c = t; c < Dm; c += NW) acc += ar[c] * ar[c];
        acc = sycl::reduce_over_group(it.get_group(), acc, sycl::plus<float>());
        const float rm = 1.0f / sycl::sqrt(acc / Dm + EPS);
        if (t == 0) ra[r] = rm;
        for (int c = t; c < Dm; c += NW) xo[(size_t)r * Dm + c] = (xi[(size_t)r * Dm + c] + ar[c] * rm * wa[c]) * scale;
    });
}

// y = x r w: dx (= or +=) r (w dy) - x r^3 mean(x w dy)
void rms_bwd(sycl::queue& q, const float* x, const float* w, const float* rinv, const float* dy, float* dx, bool acc, int R,
             int Dm) {
    constexpr int NW = 256;
    q.parallel_for(nd_range<1>((size_t)R * NW, NW), [=](nd_item<1> it) {
        const int r = it.get_group(0), t = it.get_local_id(0);
        const float* xr = x + (size_t)r * Dm;
        const float* gr = dy + (size_t)r * Dm;
        float* o = dx + (size_t)r * Dm;
        const float rm = rinv[r];
        float s = 0;
        for (int c = t; c < Dm; c += NW) s += xr[c] * w[c] * gr[c];
        s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
        const float k = rm * rm * rm * s / Dm;
        for (int c = t; c < Dm; c += NW) {
            const float v = rm * w[c] * gr[c] - xr[c] * k;
            o[c] = acc ? o[c] + v : v;
        }
    });
}

// dw[c] += sum_r dy[r, c] x[r, c] rinv[r]
void rms_dw(sycl::queue& q, const float* x, const float* rinv, const float* dy, float* dw, int R, int Dm) {
    constexpr int NW = 64;
    q.parallel_for(nd_range<1>((size_t)Dm * NW, NW), [=](nd_item<1> it) {
        const int c = it.get_group(0), t = it.get_local_id(0);
        float s = 0;
        for (int r = t; r < R; r += NW) s += dy[(size_t)r * Dm + c] * x[(size_t)r * Dm + c] * rinv[r];
        s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
        if (t == 0) dw[c] += s;
    });
}

// dw[c] += sum over rows of t[row, c] (t [rows, Dm])
void colsum(sycl::queue& q, const float* t, float* dw, int rows, int Dm) {
    constexpr int NW = 256;
    q.parallel_for(nd_range<1>((size_t)Dm * NW, NW), [=](nd_item<1> it) {
        const int c = it.get_group(0), l = it.get_local_id(0);
        float s = 0;
        for (int r = l; r < rows; r += NW) s += t[(size_t)r * Dm + c];
        s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
        if (l == 0) dw[c] += s;
    });
}

void scale_ip(sycl::queue& q, float* x, float s, size_t n) {
    q.parallel_for(range<1>(n), [=](sycl::id<1> i) { x[i] *= s; });
}

// ------------------------------------------------------------------------------------------------ query heads

// per (row, head): n = rms(q) w; rotate pairs (j, j + hd/2) for j < nrot by p * base^(-2j / hd). A work-group of hd/2
// lanes, lane j holds dims j and j + hd/2.
void qhead_fwd(sycl::queue& q, const float* qraw, const float* w, float* qr, float* rq, const int* pos, int R, int hd,
               int nrot, float base) {
    const int h2 = hd / 2;
    q.parallel_for(nd_range<1>((size_t)R * NH * h2, h2), [=](nd_item<1> it) {
        const size_t u = it.get_group(0);
        const int j = it.get_local_id(0), r = u / NH;
        const float* src = qraw + u * hd;
        float a = src[j], b = src[j + h2];
        float acc = sycl::reduce_over_group(it.get_group(), a * a + b * b, sycl::plus<float>());
        const float rm = 1.0f / sycl::sqrt(acc / hd + EPS);
        if (j == 0) rq[u] = rm;
        float na = a * rm * w[j], nb = b * rm * w[j + h2];
        if (j < nrot) {
            const float inv = sycl::pow(base, -2.0f * j / hd);
            const float th = (float)pos[r] * inv;
            const float c = sycl::cos(th), s = sycl::sin(th);
            const float ra = na * c - nb * s;
            nb = nb * c + na * s;
            na = ra;
        }
        qr[u * hd + j] = na;
        qr[u * hd + j + h2] = nb;
    });
}

// backward of qhead_fwd: dq from dqr; tq [R * NH, hd] = dn * q * r (the q_norm weight's gradient terms)
void qhead_bwd(sycl::queue& q, const float* qraw, const float* w, const float* rq, const int* pos, const float* dqr,
               float* dq, float* tq, int R, int hd, int nrot, float base) {
    const int h2 = hd / 2;
    q.parallel_for(nd_range<1>((size_t)R * NH * h2, h2), [=](nd_item<1> it) {
        const size_t u = it.get_group(0);
        const int j = it.get_local_id(0), r = u / NH;
        float da = dqr[u * hd + j], db = dqr[u * hd + j + h2];
        if (j < nrot) {
            const float inv = sycl::pow(base, -2.0f * j / hd);
            const float th = (float)pos[r] * inv;
            const float c = sycl::cos(th), s = sycl::sin(th);
            const float na = da * c + db * s;
            db = -da * s + db * c;
            da = na;
        }
        const float a = qraw[u * hd + j], b = qraw[u * hd + j + h2];
        const float rm = rq[u];
        tq[u * hd + j] = da * a * rm;
        tq[u * hd + j + h2] = db * b * rm;
        const float ga = da * w[j], gb = db * w[j + h2];
        float s = sycl::reduce_over_group(it.get_group(), ga * a + gb * b, sycl::plus<float>());
        const float k = rm * rm * rm * s / hd;
        dq[u * hd + j] = rm * ga - a * k;
        dq[u * hd + j + h2] = rm * gb - b * k;
    });
}

// ------------------------------------------------------------------------------------------------ MLP

inline float gelu_t(float z, float& dg) {
    const float k = 0.7978845608028654f, c = 0.044715f;
    const float t = sycl::tanh(k * (z + c * z * z * z));
    dg = 0.5f * (1.0f + t) + 0.5f * z * (1.0f - t * t) * k * (1.0f + 3.0f * c * z * z);
    return 0.5f * z * (1.0f + t);
}
void gelu_fwd(sycl::queue& q, const float* g, const float* u, float* m, size_t n) {
    q.parallel_for(range<1>(n), [=](sycl::id<1> i) {
        float dg;
        m[i] = gelu_t(g[i], dg) * u[i];
    });
}
void gelu_bwd(sycl::queue& q, const float* g, const float* u, const float* dm, float* dgo, float* duo, size_t n) {
    q.parallel_for(range<1>(n), [=](sycl::id<1> i) {
        float dg;
        const float ge = gelu_t(g[i], dg);
        duo[i] = dm[i] * ge;
        dgo[i] = dm[i] * u[i] * dg;
    });
}

// ------------------------------------------------------------------------------------------------ attention

// [R, NH, hd] -> [G][R][gs][hd] and back (G key heads, gs query heads each)
void permute_in(sycl::queue& q, const float* src, float* dst, int R, int G, int hd) {
    const int gs = NH / G;
    q.parallel_for(range<1>((size_t)R * NH * hd), [=](sycl::id<1> id) {
        size_t i = id[0];
        const int d = i % hd;
        const size_t rh = i / hd;
        const int h = rh % NH, r = rh / NH, g = h / gs, k = h % gs;
        dst[(((size_t)g * R + r) * gs + k) * hd + d] = src[i];
    });
}
void permute_out(sycl::queue& q, const float* src, float* dst, int R, int G, int hd) {
    const int gs = NH / G;
    q.parallel_for(range<1>((size_t)R * NH * hd), [=](sycl::id<1> id) {
        size_t i = id[0];
        const int d = i % hd;
        const size_t rh = i / hd;
        const int h = rh % NH, r = rh / NH, g = h / gs, k = h % gs;
        dst[i] = src[(((size_t)g * R + r) * gs + k) * hd + d];
    });
}

// the keys row rr of a score block sees: row r = r0 + rr / gs at position p sees p - 1 back to 0 (full) or p - 1024
inline void key_range(int p, bool full, int k0, int nk, int& lo, int& hi) {
    hi = sycl::min(p - 1 - k0, nk - 1);
    lo = sycl::max((full ? 0 : sycl::max(0, p - g4::WIN)) - k0, 0);
}

// scores -> probabilities in place; lse [R, NH]
void softmax_rows(sycl::queue& q, float* S, int M, int nk, int gs, int r0, int g, const int* pos, int k0, bool full, float* lse) {
    constexpr int NW = 256;
    q.parallel_for(nd_range<1>((size_t)M * NW, NW), [=](nd_item<1> it) {
        const int rr = it.get_group(0), t = it.get_local_id(0);
        const int r = r0 + rr / gs, h = g * gs + rr % gs;
        int lo, hi;
        key_range(pos[r], full, k0, nk, lo, hi);
        float* row = S + (size_t)rr * nk;
        float mx = -INFINITY;
        for (int j = lo + t; j <= hi; j += NW) mx = sycl::fmax(mx, row[j]);
        mx = sycl::reduce_over_group(it.get_group(), mx, sycl::maximum<float>());
        float s = 0;
        for (int j = lo + t; j <= hi; j += NW) s += sycl::exp(row[j] - mx);
        s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
        const float l = mx + sycl::log(s);
        if (t == 0) lse[(size_t)r * NH + h] = l;
        for (int j = t; j < nk; j += NW) row[j] = (j >= lo && j <= hi) ? sycl::exp(row[j] - l) : 0.0f;
    });
}
// scores -> probabilities from a known lse
void probs_rows(sycl::queue& q, float* S, int M, int nk, int gs, int r0, int g, const int* pos, int k0, bool full, const float* lse) {
    q.parallel_for(range<2>((size_t)M, (size_t)nk), [=](sycl::item<2> it) {
        const int rr = it.get_id(0), j = it.get_id(1);
        const int r = r0 + rr / gs, h = g * gs + rr % gs;
        int lo, hi;
        key_range(pos[r], full, k0, nk, lo, hi);
        float& v = S[(size_t)rr * nk + j];
        v = (j >= lo && j <= hi) ? sycl::exp(v - lse[(size_t)r * NH + h]) : 0.0f;
    });
}
// dP -> dS = P (dP - delta) in place
void dsoft_rows(sycl::queue& q, const float* P, float* dP, int M, int nk, int gs, int r0, int g, const float* delta) {
    q.parallel_for(range<2>((size_t)M, (size_t)nk), [=](sycl::item<2> it) {
        const int rr = it.get_id(0), j = it.get_id(1);
        const int r = r0 + rr / gs, h = g * gs + rr % gs;
        const size_t i = (size_t)rr * nk + j;
        dP[i] = P[i] * (dP[i] - delta[(size_t)r * NH + h]);
    });
}
// delta [R * NH] = sum_d dO * O
void row_dot(sycl::queue& q, const float* a, const float* b, float* out, int rows, int hd) {
    constexpr int NW = 64;
    q.parallel_for(nd_range<1>((size_t)rows * NW, NW), [=](nd_item<1> it) {
        const int r = it.get_group(0), t = it.get_local_id(0);
        float s = 0;
        for (int d = t; d < hd; d += NW) s += a[(size_t)r * hd + d] * b[(size_t)r * hd + d];
        s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
        if (t == 0) out[r] = s;
    });
}

void h2f(sycl::queue& q, const half* a, float* b, size_t n) {
    q.parallel_for(range<1>(n), [=](sycl::id<1> i) { b[i] = static_cast<float>(a[i]); });
}

// ------------------------------------------------------------------------------------------------ loss

// per row: target logits (fp16, soft-capped) at tl row trow[r]; drafter logits dl in place -> w (softmax(dl) - softmax(tl));
// kl[r], arg[r] (drafter argmax), gap[r] (its top two logits' difference)
void kl_rows(sycl::queue& q, float* dl, const half* tl, const int* trow, const float* w, float* kl, int* arg, float* gap,
             int* targ, int R) {
    constexpr int NW = 1024;
    q.parallel_for(nd_range<1>((size_t)R * NW, NW), [=](nd_item<1> it) {
        const int r = it.get_group(0), t = it.get_local_id(0);
        auto grp = it.get_group();
        float* d = dl + (size_t)r * V;
        const half* tt = tl + (size_t)trow[r] * V;
        float md = -INFINITY, mt = -INFINITY;
        for (int j = t; j < V; j += NW) {
            md = sycl::fmax(md, d[j]);
            mt = sycl::fmax(mt, static_cast<float>(tt[j]));
        }
        md = sycl::reduce_over_group(grp, md, sycl::maximum<float>());
        mt = sycl::reduce_over_group(grp, mt, sycl::maximum<float>());
        float sd = 0, st = 0, second = -INFINITY;
        int am = V, cnt = 0, at = V;
        for (int j = t; j < V; j += NW) {
            const float v = d[j];
            sd += sycl::exp(v - md);
            const float tv = static_cast<float>(tt[j]);
            st += sycl::exp(tv - mt);
            if (tv == mt) at = sycl::min(at, j);
            if (v == md) { am = sycl::min(am, j); cnt++; }
            else second = sycl::fmax(second, v);
        }
        sd = sycl::reduce_over_group(grp, sd, sycl::plus<float>());
        st = sycl::reduce_over_group(grp, st, sycl::plus<float>());
        am = sycl::reduce_over_group(grp, am, sycl::minimum<int>());
        at = sycl::reduce_over_group(grp, at, sycl::minimum<int>());
        cnt = sycl::reduce_over_group(grp, cnt, sycl::plus<int>());
        second = sycl::reduce_over_group(grp, second, sycl::maximum<float>());
        const float ld = md + sycl::log(sd), lt = mt + sycl::log(st);
        const float wr = w[r];
        float k = 0;
        for (int j = t; j < V; j += NW) {
            const float lpt = static_cast<float>(tt[j]) - lt, lpd = d[j] - ld;
            const float pt = sycl::exp(lpt);
            k += pt * (lpt - lpd);
            d[j] = wr * (sycl::exp(lpd) - pt);
        }
        k = sycl::reduce_over_group(grp, k, sycl::plus<float>());
        if (t == 0) {
            kl[r] = k;
            arg[r] = am;
            targ[r] = at;
            gap[r] = cnt > 1 ? 0.0f : md - second;
        }
    });
}

// ------------------------------------------------------------------------------------------------ input, optimizer

// z [R, 2D] = E [R, D] | h, h: the target's fin row p - 1 (first step) or hprev [R, D]
void fill_z(sycl::queue& q, float* z, const float* E, const half* fin, const float* hprev, const int* pos, int R) {
    q.parallel_for(range<1>((size_t)R * D), [=](sycl::id<1> id) {
        const size_t i = id[0];
        const int r = i / D, c = i % D;
        z[(size_t)r * Z + c] = E[i];
        z[(size_t)r * Z + D + c] = fin ? static_cast<float>(fin[(size_t)(pos[r] - 1) * D + c]) : hprev[i];
    });
}

float sum_squares(sycl::queue& q, const float* x, size_t n, float* acc_dev) {
    q.memset(acc_dev, 0, sizeof(float));
    constexpr int NW = 256;
    const size_t groups = 4096;
    q.parallel_for(nd_range<1>(groups * NW, NW), [=](nd_item<1> it) {
        float s = 0;
        for (size_t i = it.get_global_id(0); i < n; i += groups * NW) s += x[i] * x[i];
        s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
        if (it.get_local_id(0) == 0) {
            sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                a(*acc_dev);
            a.fetch_add(s);
        }
    });
    float h = 0;
    q.memcpy(&h, acc_dev, sizeof(float)).wait();
    return h;
}

void adamw(sycl::queue& q, float* p, float* g, float* m, float* v, size_t n, float lr, float gscale, float b1, float b2,
           float eps, float c1, float c2) {
    q.parallel_for(range<1>(n), [=](sycl::id<1> i) {
        const float gi = g[i] * gscale;
        const float mi = b1 * m[i] + (1 - b1) * gi;
        const float vi = b2 * v[i] + (1 - b2) * gi * gi;
        m[i] = mi;
        v[i] = vi;
        p[i] -= lr * (mi / c1) / (sycl::sqrt(vi / c2) + eps);
        g[i] = 0;
    });
}

float bf16_to_f(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}
uint16_t f_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    if ((u & 0x7fffffff) > 0x7f800000) return 0x7fc0;
    u += 0x7fff + ((u >> 16) & 1);
    return (uint16_t)(u >> 16);
}

template <class T> T* dev_alloc(size_t n, sycl::queue& q, const char* what) {
    T* p = sycl::malloc_device<T>(n, q);
    GLM_CHECK(p, "g4draft: VRAM for %s (%.2f GiB)", what, n * sizeof(T) / 1073741824.0);
    Dev::get().vram_used += n * sizeof(T);
    return p;
}
}  // namespace

// ------------------------------------------------------------------------------------------------ activations

struct G4DraftTrainer::Act {
    float *z, *x0;
    struct L { float *xin, *rin, *hn, *qraw, *rq, *qr, *o, *lse, *a, *ra, *x1, *r1, *h2, *g, *u, *m, *d, *rd, *xo; };
    std::vector<L> l;
    float *xf, *rf, *y, *dyh, *hout;
};

void G4DraftStats::add(const G4DraftStats& o) {
    kl += o.kl;
    rows += o.rows;
    anchors += o.anchors;
    accepted += o.accepted;
    accepted_m += o.accepted_m;
    drafted_m += o.drafted_m;
    oracle += o.oracle;
    if (hit.size() < o.hit.size()) { hit.resize(o.hit.size(), 0); hit_n.resize(o.hit.size(), 0); thit.resize(o.hit.size(), 0); }
    for (size_t i = 0; i < o.hit.size(); i++) { hit[i] += o.hit[i]; hit_n[i] += o.hit_n[i]; thit[i] += o.thit[i]; }
}

G4DraftTrainer::G4DraftTrainer(G4Model& target, const std::string& dir, int anchors, int steps, int ctx)
    : m_(target), src_dir_(dir), A_(anchors), K_(steps), ctx_(ctx) {
    auto& q = Dev::get().q;
    Json c = Json::parse(read_file(dir + "/config.json"));
    GLM_CHECK(c.get_s("model_type", "") == "gemma4_assistant" && c.get_i("backbone_hidden_size", 0) == D,
              "%s: not the assistant of this model", dir.c_str());
    const Json& tc = c["text_config"];
    const int nl = (int)tc.get_i("num_hidden_layers", 0);
    GLM_CHECK(tc.get_i("hidden_size", 0) == DH && tc.get_i("intermediate_size", 0) == DI, "%s: shape", dir.c_str());
    SafeTensors st(dir);
    // the flat parameter layout
    auto add = [&](const std::string& name, int64_t n) {
        const StTensor& t = st.at(name);
        GLM_CHECK(t.dt == DType::BF16 && (int64_t)(t.bytes / 2) == n, "%s: expected bf16 of %lld", name.c_str(), (long long)n);
        names_.push_back({name, np_});
        int64_t off = np_;
        np_ += n;
        return off;
    };
    pre_ = add("pre_projection.weight", (int64_t)DH * Z);
    post_ = add("post_projection.weight", (int64_t)D * DH);
    wnorm_ = add("model.norm.weight", DH);
    L_.resize(nl);
    for (int li = 0; li < nl; li++) {
        Layer& l = L_[li];
        std::string p = "model.layers." + std::to_string(li) + ".", ap = p + "self_attn.";
        l.full = tc["layer_types"][li].s() == "full_attention";
        l.hd = l.full ? g4::FD : g4::SD;
        l.q = add(ap + "q_proj.weight", (int64_t)NH * l.hd * DH);
        l.o = add(ap + "o_proj.weight", (int64_t)DH * NH * l.hd);
        l.g = add(p + "mlp.gate_proj.weight", (int64_t)DI * DH);
        l.u = add(p + "mlp.up_proj.weight", (int64_t)DI * DH);
        l.d = add(p + "mlp.down_proj.weight", (int64_t)DH * DI);
        l.w_in = add(p + "input_layernorm.weight", DH);
        l.w_pa = add(p + "post_attention_layernorm.weight", DH);
        l.w_pf = add(p + "pre_feedforward_layernorm.weight", DH);
        l.w_ff = add(p + "post_feedforward_layernorm.weight", DH);
        l.qn = add(ap + "q_norm.weight", l.hd);
        l.scalar = bf16_to_f(*static_cast<const uint16_t*>(st.at(p + "layer_scalar").data));
    }
    P_ = dev_alloc<float>(np_, q, "parameters");
    G_ = dev_alloc<float>(np_, q, "gradients");
    M_ = dev_alloc<float>(np_, q, "moments");
    V_ = dev_alloc<float>(np_, q, "moments");
    {
        std::vector<float> h;
        for (auto& [name, off] : names_) {
            const StTensor& t = st.at(name);
            const uint16_t* b = static_cast<const uint16_t*>(t.data);
            h.resize(t.bytes / 2);
            for (size_t i = 0; i < h.size(); i++) h[i] = bf16_to_f(b[i]);
            upload(P_ + off, h.data(), h.size() * 4);
        }
        const StTensor& e = st.at("model.embed_tokens.weight");
        GLM_CHECK(e.dt == DType::BF16 && e.shape[0] == V && e.shape[1] == DH, "embed_tokens: expected bf16 [V, 1024]");
        head_ = dev_alloc<float>((size_t)V * DH, q, "head");
        const uint16_t* b = static_cast<const uint16_t*>(e.data);
        const size_t rows = 16384;
        h.resize(rows * DH);
        for (size_t r0 = 0; r0 < (size_t)V; r0 += rows) {
            for (size_t i = 0; i < rows * DH; i++) h[i] = bf16_to_f(b[r0 * DH + i]);
            upload(head_ + r0 * DH, h.data(), rows * DH * 4);
        }
    }
    q.memset(G_, 0, np_ * 4);
    q.memset(M_, 0, np_ * 4);
    q.memset(V_, 0, np_ * 4);

    // capture and sequence buffers
    cap_.rows = ctx;
    cap_.fin = dev_alloc<half>((size_t)ctx * D, q, "capture");
    cap_.sk = dev_alloc<half>((size_t)ctx * g4::SKV * g4::SD, q, "capture");
    cap_.sv = dev_alloc<half>((size_t)ctx * g4::SKV * g4::SD, q, "capture");
    const size_t fkv = (size_t)g4::FKV * g4::FD;
    fk_ = dev_alloc<float>((size_t)ctx * fkv, q, "full keys");
    fv_ = dev_alloc<float>((size_t)ctx * fkv, q, "full keys");
    tmpk_ = dev_alloc<half>((size_t)4096 * fkv, q, "key staging");
    tmpv_ = dev_alloc<half>((size_t)4096 * fkv, q, "key staging");
    sblock_ = g4::WIN + A_ + K_ + 64;
    sk_ = dev_alloc<float>((size_t)sblock_ * g4::SKV * g4::SD, q, "sliding keys");
    sv_ = dev_alloc<float>((size_t)sblock_ * g4::SKV * g4::SD, q, "sliding keys");

    // activations
    size_t per = (size_t)Z + DH;
    for (auto& l : L_) per += (size_t)DH * 7 + 4 + (size_t)NH * l.hd * 3 + NH * 2 + (size_t)DI * 3;
    per += (size_t)DH * 3 + 1 + D;
    arena_ = dev_alloc<float>(per * A_ * K_, q, "activations");
    float* a = arena_;
    auto take = [&](size_t n) { float* p = a; a += n * A_; return p; };
    for (int k = 0; k < K_; k++) {
        auto act = std::make_unique<Act>();
        act->z = take(Z);
        act->x0 = take(DH);
        float* xin = act->x0;
        for (auto& l : L_) {
            Act::L s;
            s.xin = xin;
            s.rin = take(1);
            s.hn = take(DH);
            s.qraw = take((size_t)NH * l.hd);
            s.rq = take(NH);
            s.qr = take((size_t)NH * l.hd);
            s.o = take((size_t)NH * l.hd);
            s.lse = take(NH);
            s.a = take(DH);
            s.ra = take(1);
            s.x1 = take(DH);
            s.r1 = take(1);
            s.h2 = take(DH);
            s.g = take(DI);
            s.u = take(DI);
            s.m = take(DI);
            s.d = take(DH);
            s.rd = take(1);
            s.xo = take(DH);
            xin = s.xo;
            act->l.push_back(s);
        }
        act->xf = xin;
        act->rf = take(1);
        act->y = take(DH);
        act->dyh = take(DH);
        act->hout = take(D);
        acts_.push_back(std::move(act));
    }
    GLM_CHECK((size_t)(a - arena_) <= per * A_ * K_, "g4draft: arena overflow");

    // one pinned buffer and one device buffer per step: a step's upload must not see the next step's rows
    Eh_ = sycl::malloc_host<float>((size_t)A_ * D * K_, q);
    Ed_ = dev_alloc<float>((size_t)A_ * D * K_, q, "embeddings");
    tl_ = dev_alloc<half>((size_t)(A_ + K_) * V, q, "target logits");
    dl_ = dev_alloc<float>((size_t)A_ * V, q, "drafter logits");
    sbudget_ = (size_t)48 << 20;   // floats a score block may take (192 MB)
    S_ = dev_alloc<float>(sbudget_, q, "scores");
    dP_ = dev_alloc<float>(sbudget_, q, "scores");
    const size_t qn = (size_t)A_ * NH * g4::FD;
    qp_ = dev_alloc<float>(qn, q, "attention");
    op_ = dev_alloc<float>(qn, q, "attention");
    dop_ = dev_alloc<float>(qn, q, "attention");
    tq_ = dev_alloc<float>(qn, q, "attention");
    bq_ = dev_alloc<float>(qn, q, "backward");
    bo_ = dev_alloc<float>(qn, q, "backward");
    delta_ = dev_alloc<float>((size_t)A_ * NH, q, "attention");
    bx_ = dev_alloc<float>((size_t)A_ * DH, q, "backward");
    bx2_ = dev_alloc<float>((size_t)A_ * DH, q, "backward");
    bh_ = dev_alloc<float>((size_t)A_ * DH, q, "backward");
    dyacc_ = dev_alloc<float>((size_t)A_ * DH, q, "backward");
    bg_ = dev_alloc<float>((size_t)A_ * DI, q, "backward");
    bu_ = dev_alloc<float>((size_t)A_ * DI, q, "backward");
    bm_ = dev_alloc<float>((size_t)A_ * DI, q, "backward");
    bz_ = dev_alloc<float>((size_t)A_ * Z, q, "backward");
    pos_d_ = dev_alloc<int>(A_, q, "rows");
    row_d_ = dev_alloc<int>((size_t)A_ * K_, q, "rows");
    w_d_ = dev_alloc<float>((size_t)A_ * K_, q, "rows");
    rowstat_ = dev_alloc<float>((size_t)A_ * K_ * 2 + 1, q, "rows");
    rowarg_ = dev_alloc<int>((size_t)A_ * K_ * 2, q, "rows");     // drafter argmax, then the target's
    q.wait();
    fprintf(stderr, "g4draft: %d layers, %.1fM trainable, head frozen; %d anchors x %d steps a chunk; VRAM %.2f GiB\n", nl,
            np_ / 1e6, A_, K_, Dev::get().vram_used / 1073741824.0);
}

G4DraftTrainer::~G4DraftTrainer() {
    auto& q = Dev::get().q;
    q.wait();
    for (void* p : std::initializer_list<void*>{P_, G_, M_, V_, head_, cap_.fin, cap_.sk, cap_.sv, fk_, fv_, tmpk_, tmpv_, sk_,
                                                sv_, arena_, Ed_, tl_, dl_, S_, dP_, qp_, op_, dop_, tq_, bq_, bo_, delta_, bx_,
                                                bx2_, bh_, dyacc_, bg_, bu_, bm_, bz_, pos_d_, row_d_, w_d_, rowstat_, rowarg_})
        if (p) sycl::free(p, q);
    if (Eh_) sycl::free(Eh_, q);
}

void G4DraftTrainer::begin_sequence(int len) {
    GLM_CHECK(len <= ctx_, "sequence of %d tokens (context %d)", len, ctx_);
    auto& q = Dev::get().q;
    const size_t fkv = (size_t)g4::FKV * g4::FD;
    for (int k0 = 0; k0 < len; k0 += 4096) {
        const int n = std::min(4096, len - k0);
        m_.last_full_kv(k0, n, tmpk_, tmpv_);
        h2f(q, tmpk_, fk_ + (size_t)k0 * fkv, (size_t)n * fkv);
        h2f(q, tmpv_, fv_ + (size_t)k0 * fkv, (size_t)n * fkv);
    }
}

// ------------------------------------------------------------------------------------------------ attention (GEMMs)

namespace {
struct KeyBlock { const float *K, *Vv; int G, k0, nk; bool full; };
}

void G4DraftTrainer::attn_fwd(const Layer& l, float* qr, float* o, float* lse, int R, int k0, int nk) {
    auto& q = Dev::get().q;
    const int G = l.full ? g4::FKV : g4::SKV, gs = NH / G, hd = l.hd;
    const float* Kb = l.full ? fk_ : sk_;
    const float* Vb = l.full ? fv_ : sv_;
    permute_in(q, qr, qp_, R, G, hd);
    const int Rc = std::max(1, std::min(R, (int)(sbudget_ / ((size_t)gs * nk))));
    for (int g = 0; g < G; g++)
        for (int r0 = 0; r0 < R; r0 += Rc) {
            const int rc = std::min(Rc, R - r0), M = rc * gs;
            const float* Qg = qp_ + ((size_t)g * R + r0) * gs * hd;
            gemm(q, false, true, M, nk, hd, 1, Qg, hd, Kb + (size_t)g * hd, (int64_t)G * hd, 0, S_, nk);
            softmax_rows(q, S_, M, nk, gs, r0, g, pos_d_, k0, l.full, lse);
            gemm(q, false, false, M, hd, nk, 1, S_, nk, Vb + (size_t)g * hd, (int64_t)G * hd, 0,
                 op_ + ((size_t)g * R + r0) * gs * hd, hd);
        }
    permute_out(q, op_, o, R, G, hd);
}

void G4DraftTrainer::attn_bwd(const Layer& l, const float* qr, const float* o, const float* lse, float* d_o, float* dq, int R,
                              int k0, int nk) {
    auto& q = Dev::get().q;
    const int G = l.full ? g4::FKV : g4::SKV, gs = NH / G, hd = l.hd;
    const float* Kb = l.full ? fk_ : sk_;
    const float* Vb = l.full ? fv_ : sv_;
    row_dot(q, d_o, o, delta_, R * NH, hd);
    permute_in(q, qr, qp_, R, G, hd);
    permute_in(q, d_o, dop_, R, G, hd);
    const int Rc = std::max(1, std::min(R, (int)(sbudget_ / ((size_t)gs * nk))));
    for (int g = 0; g < G; g++)
        for (int r0 = 0; r0 < R; r0 += Rc) {
            const int rc = std::min(Rc, R - r0), M = rc * gs;
            const size_t off = ((size_t)g * R + r0) * gs * hd;
            gemm(q, false, true, M, nk, hd, 1, qp_ + off, hd, Kb + (size_t)g * hd, (int64_t)G * hd, 0, S_, nk);
            probs_rows(q, S_, M, nk, gs, r0, g, pos_d_, k0, l.full, lse);
            gemm(q, false, true, M, nk, hd, 1, dop_ + off, hd, Vb + (size_t)g * hd, (int64_t)G * hd, 0, dP_, nk);
            dsoft_rows(q, S_, dP_, M, nk, gs, r0, g, delta_);
            gemm(q, false, false, M, hd, nk, 1, dP_, nk, Kb + (size_t)g * hd, (int64_t)G * hd, 0, op_ + off, hd);
        }
    permute_out(q, op_, dq, R, G, hd);
}

// ------------------------------------------------------------------------------------------------ steps

namespace {
// the chunk's key blocks (set by chunk(), read by the steps)
int g_sk0 = 0, g_snk = 0, g_fnk = 0;
}

void G4DraftTrainer::forward_step(int k, int R, const int*, bool first) {
    auto& q = Dev::get().q;
    Act& A = *acts_[k];
    fill_z(q, A.z, Ed_ + (size_t)k * A_ * D, first ? cap_.fin : nullptr, first ? nullptr : acts_[k - 1]->hout, pos_d_, R);
    lin(q, A.z, R, Z, P_ + pre_, DH, A.x0);
    for (size_t li = 0; li < L_.size(); li++) {
        const Layer& l = L_[li];
        Act::L& s = A.l[li];
        const int qd = NH * l.hd;
        rms_fwd(q, s.xin, P_ + l.w_in, s.hn, s.rin, R, DH);
        lin(q, s.hn, R, DH, P_ + l.q, qd, s.qraw);
        qhead_fwd(q, s.qraw, P_ + l.qn, s.qr, s.rq, pos_d_, R, l.hd, l.full ? g4::FROT : l.hd / 2, l.full ? 1e6f : 1e4f);
        if (l.full) attn_fwd(l, s.qr, s.o, s.lse, R, 0, g_fnk);
        else attn_fwd(l, s.qr, s.o, s.lse, R, g_sk0, g_snk);
        lin(q, s.o, R, qd, P_ + l.o, DH, s.a);
        addnorm_fwd(q, s.xin, s.a, P_ + l.w_pa, 1.0f, s.x1, s.ra, R, DH);
        rms_fwd(q, s.x1, P_ + l.w_pf, s.h2, s.r1, R, DH);
        lin(q, s.h2, R, DH, P_ + l.g, DI, s.g);
        lin(q, s.h2, R, DH, P_ + l.u, DI, s.u);
        gelu_fwd(q, s.g, s.u, s.m, (size_t)R * DI);
        lin(q, s.m, R, DI, P_ + l.d, DH, s.d);
        addnorm_fwd(q, s.x1, s.d, P_ + l.w_ff, l.scalar, s.xo, s.rd, R, DH);
    }
    rms_fwd(q, A.xf, P_ + wnorm_, A.y, A.rf, R, DH);
    lin(q, A.y, R, DH, head_, V, dl_);
    kl_rows(q, dl_, tl_, row_d_ + (size_t)k * A_, w_d_ + (size_t)k * A_, rowstat_ + (size_t)k * A_,
            rowarg_ + (size_t)k * A_, rowstat_ + (size_t)(K_ + k) * A_, rowarg_ + (size_t)(K_ + k) * A_, R);
    gemm(q, false, false, R, DH, V, 1, dl_, V, head_, DH, 0, A.dyh, DH);
    lin(q, A.y, R, DH, P_ + post_, D, A.hout);
}

void G4DraftTrainer::backward_step(int k, int R, bool last) {
    auto& q = Dev::get().q;
    Act& A = *acts_[k];
    q.memcpy(dyacc_, A.dyh, (size_t)R * DH * 4);
    if (!last) {
        // the next step's h came from this step's post_projection: its gradient is the h part of that step's dz (bz_)
        lin_dx(q, bz_ + D, Z, R, D, P_ + post_, DH, dyacc_, 1.0f);
        lin_dw(q, bz_ + D, Z, R, D, A.y, DH, G_ + post_);
    }
    rms_bwd(q, A.xf, P_ + wnorm_, A.rf, dyacc_, bx_, false, R, DH);
    rms_dw(q, A.xf, A.rf, dyacc_, G_ + wnorm_, R, DH);
    for (int li = (int)L_.size() - 1; li >= 0; li--) {
        const Layer& l = L_[li];
        Act::L& s = A.l[li];
        const int qd = NH * l.hd;
        // xo = (x1 + rms(d) w_ff) * scalar
        if (l.scalar != 1.0f) scale_ip(q, bx_, l.scalar, (size_t)R * DH);
        rms_bwd(q, s.d, P_ + l.w_ff, s.rd, bx_, bx2_, false, R, DH);
        rms_dw(q, s.d, s.rd, bx_, G_ + l.w_ff, R, DH);
        lin_dx(q, bx2_, DH, R, DH, P_ + l.d, DI, bm_, 0.0f);
        lin_dw(q, bx2_, DH, R, DH, s.m, DI, G_ + l.d);
        gelu_bwd(q, s.g, s.u, bm_, bg_, bu_, (size_t)R * DI);
        lin_dx(q, bg_, DI, R, DI, P_ + l.g, DH, bh_, 0.0f);
        lin_dx(q, bu_, DI, R, DI, P_ + l.u, DH, bh_, 1.0f);
        lin_dw(q, bg_, DI, R, DI, s.h2, DH, G_ + l.g);
        lin_dw(q, bu_, DI, R, DI, s.h2, DH, G_ + l.u);
        rms_bwd(q, s.x1, P_ + l.w_pf, s.r1, bh_, bx_, true, R, DH);
        rms_dw(q, s.x1, s.r1, bh_, G_ + l.w_pf, R, DH);
        // x1 = xin + rms(a) w_pa
        rms_bwd(q, s.a, P_ + l.w_pa, s.ra, bx_, bx2_, false, R, DH);
        rms_dw(q, s.a, s.ra, bx_, G_ + l.w_pa, R, DH);
        lin_dx(q, bx2_, DH, R, DH, P_ + l.o, qd, bq_, 0.0f);
        lin_dw(q, bx2_, DH, R, DH, s.o, qd, G_ + l.o);
        if (l.full) attn_bwd(l, s.qr, s.o, s.lse, bq_, bo_, R, 0, g_fnk);
        else attn_bwd(l, s.qr, s.o, s.lse, bq_, bo_, R, g_sk0, g_snk);
        qhead_bwd(q, s.qraw, P_ + l.qn, s.rq, pos_d_, bo_, bq_, tq_, R, l.hd, l.full ? g4::FROT : l.hd / 2,
                  l.full ? 1e6f : 1e4f);
        colsum(q, tq_, G_ + l.qn, R * NH, l.hd);
        lin_dx(q, bq_, qd, R, qd, P_ + l.q, DH, bh_, 0.0f);
        lin_dw(q, bq_, qd, R, qd, s.hn, DH, G_ + l.q);
        rms_bwd(q, s.xin, P_ + l.w_in, s.rin, bh_, bx_, true, R, DH);
        rms_dw(q, s.xin, s.rin, bh_, G_ + l.w_in, R, DH);
    }
    // x0 = z pre^T: dz for the step before (its h part), and pre's gradient
    lin_dw(q, bx_, DH, R, DH, A.z, Z, G_ + pre_);
    lin_dx(q, bx_, DH, R, DH, P_ + pre_, Z, bz_, 0.0f);
}

G4DraftStats G4DraftTrainer::chunk(const std::vector<int>& seq, const std::vector<int>& anchors, bool train, float margin) {
    auto& q = Dev::get().q;
    const int R = (int)anchors.size(), len = (int)seq.size();
    GLM_CHECK(R >= 1 && R <= A_, "chunk of %d anchors (at most %d)", R, A_);
    const int pmin = anchors.front(), pmax = anchors.back();
    GLM_CHECK(pmin >= 1 && pmax + 1 < len, "anchors %d .. %d of %d", pmin, pmax, len);
    // target logits for rows pmin .. pmin + nrow - 1 (row p + k - 1 for step k)
    const int nrow = std::min(len - 1, pmax + K_ - 1) - pmin + 1;
    GLM_CHECK(nrow <= A_ + K_, "anchors span %d rows", nrow);
    m_.head_rows(cap_.fin + (size_t)pmin * D, nrow, tl_);
    // rows: positions, target rows and weights (a step past the sequence's last token has none)
    std::vector<int> trow((size_t)A_ * K_, 0);
    std::vector<float> w((size_t)A_ * K_, 0.0f);
    int valid = 0;
    for (int k = 0; k < K_; k++)
        for (int r = 0; r < R; r++) {
            const int p = anchors[r];
            if (p + k + 1 <= len - 1) {
                trow[(size_t)k * A_ + r] = p + k - pmin;
                w[(size_t)k * A_ + r] = 1.0f;
                valid++;
            }
        }
    for (auto& x : w) x = x > 0 ? 1.0f / valid : 0.0f;
    upload(pos_d_, anchors.data(), (size_t)R * 4);
    upload(row_d_, trow.data(), trow.size() * 4);
    upload(w_d_, w.data(), w.size() * 4);
    // key blocks: the sliding layer's keys pmin - 1024 .. pmax - 1 as fp32; the full layer's 0 .. pmax - 1
    g_sk0 = std::max(0, pmin - g4::WIN);
    g_snk = pmax - g_sk0;
    g_fnk = pmax;
    GLM_CHECK(g_snk <= sblock_, "sliding block of %d keys", g_snk);
    const size_t skv = (size_t)g4::SKV * g4::SD;
    h2f(q, cap_.sk + (size_t)g_sk0 * skv, sk_, (size_t)g_snk * skv);
    h2f(q, cap_.sv + (size_t)g_sk0 * skv, sv_, (size_t)g_snk * skv);

    std::vector<int> tok(R);
    for (int k = 0; k < K_; k++) {
        for (int r = 0; r < R; r++) tok[r] = seq[std::min(anchors[r] + k, len - 1)];
        float* eh = Eh_ + (size_t)k * A_ * D;
        m_.embed_rows(tok.data(), R, eh);
        q.memcpy(Ed_ + (size_t)k * A_ * D, eh, (size_t)R * D * 4);
        forward_step(k, R, nullptr, k == 0);
    }
    if (train)
        for (int k = K_ - 1; k >= 0; k--) backward_step(k, R, k == K_ - 1);

    // metrics
    std::vector<float> st((size_t)A_ * K_ * 2);
    std::vector<int> arg((size_t)A_ * K_ * 2);
    q.memcpy(st.data(), rowstat_, st.size() * 4);
    q.memcpy(arg.data(), rowarg_, arg.size() * 4);
    q.wait();
    G4DraftStats s;
    s.hit.assign(K_, 0);
    s.hit_n.assign(K_, 0);
    s.thit.assign(K_, 0);
    s.anchors = R;
    for (int r = 0; r < R; r++) {
        const int p = anchors[r];
        bool run = true, run_m = true, run_t = true;
        for (int k = 0; k < K_; k++) {
            if (p + k + 1 > len - 1) break;
            const size_t i = (size_t)k * A_ + r;
            const bool hit = arg[i] == seq[p + k + 1];
            s.kl += st[i];
            s.rows++;
            s.hit[k] += hit;
            s.hit_n[k]++;
            const bool thit = arg[(size_t)(K_ + k) * A_ + r] == seq[p + k + 1];
            s.thit[k] += thit;
            if (run_t && thit) s.oracle++;
            else run_t = false;
            if (run && hit) s.accepted++;
            else run = false;
            if (run_m) {
                s.drafted_m++;
                if (hit) s.accepted_m++;
                if (!hit || st[(size_t)(K_ + k) * A_ + r] < margin) run_m = false;
            }
        }
    }
    return s;
}

double G4DraftTrainer::step(float lr, int chunks, float clip) {
    auto& q = Dev::get().q;
    const float inv = 1.0f / std::max(1, chunks);
    const double norm = std::sqrt((double)sum_squares(q, G_, np_, rowstat_ + (size_t)A_ * K_ * 2)) * inv;
    const float gscale = inv * (norm > clip ? (float)(clip / norm) : 1.0f);
    t_++;
    const float b1 = 0.9f, b2 = 0.999f;
    adamw(q, P_, G_, M_, V_, np_, lr, gscale, b1, b2, 1e-8f, 1.0f - std::pow(b1, t_), 1.0f - std::pow(b2, t_));
    q.wait();
    return norm;
}

namespace {
inline float hash_u(uint64_t i, uint32_t seed) {
    uint64_t x = i * 0x9E3779B97F4A7C15ull ^ ((uint64_t)seed << 32 | seed);
    x ^= x >> 33; x *= 0xff51afd7ed558ccdull; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ull; x ^= x >> 33;
    return (float)((x >> 40) & 0xffffff) / 8388608.0f - 1.0f;
}
}  // namespace

void G4DraftTrainer::perturb(float eps, uint32_t seed) {
    float* p = P_;
    Dev::get().q.parallel_for(range<1>(np_), [=](sycl::id<1> i) { p[i] += eps * hash_u(i[0], seed); }).wait();
}

double G4DraftTrainer::grad_dot(uint32_t seed) {
    auto& q = Dev::get().q;
    constexpr int NW = 256;
    const size_t groups = 4096;
    float* part = sycl::malloc_device<float>(groups, q);
    const float* g = G_;
    const int64_t n = np_;
    q.parallel_for(nd_range<1>(groups * NW, NW), [=](nd_item<1> it) {
        float s = 0;
        for (size_t i = it.get_global_id(0); i < (size_t)n; i += groups * NW) s += g[i] * hash_u(i, seed);
        s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
        if (it.get_local_id(0) == 0) part[it.get_group(0)] = s;
    });
    std::vector<float> h(groups);
    q.memcpy(h.data(), part, groups * 4).wait();
    sycl::free(part, q);
    double r = 0;
    for (float v : h) r += v;
    return r;
}

void G4DraftTrainer::zero_grad() { Dev::get().q.memset(G_, 0, np_ * 4).wait(); }

// ------------------------------------------------------------------------------------------------ saving

void G4DraftTrainer::save_model(const std::string& out) const {
    namespace fs = std::filesystem;
    fs::create_directories(out);
    for (auto& e : fs::directory_iterator(src_dir_))
        if (e.is_regular_file() && e.path().filename() != "model.safetensors")
            fs::copy_file(e.path(), fs::path(out) / e.path().filename(), fs::copy_options::overwrite_existing);
    // the source file's bytes with the trained tensors written over their ranges (same dtypes and shapes)
    const std::string src = src_dir_ + "/model.safetensors", dst = out + "/model.safetensors.tmp";
    fs::copy_file(src, dst, fs::copy_options::overwrite_existing);
    FILE* f = fopen(dst.c_str(), "r+b");
    GLM_CHECK(f, "cannot write %s", dst.c_str());
    uint64_t hn = 0;
    GLM_CHECK(fread(&hn, 8, 1, f) == 1, "%s: header", dst.c_str());
    std::string hs(hn, '\0');
    GLM_CHECK(fread(hs.data(), 1, hn, f) == hn, "%s: header", dst.c_str());
    Json h = Json::parse(hs);
    std::vector<float> buf;
    std::vector<uint16_t> b16;
    for (auto& [name, off] : names_) {
        const Json& e = h[name];
        const int64_t a = e["data_offsets"][0].i(), b = e["data_offsets"][1].i();
        const size_t n = (b - a) / 2;
        buf.resize(n);
        b16.resize(n);
        download(buf.data(), P_ + off, n * 4);
        for (size_t i = 0; i < n; i++) b16[i] = f_to_bf16(buf[i]);
        fseek(f, (long)(8 + hn + a), SEEK_SET);
        GLM_CHECK(fwrite(b16.data(), 2, n, f) == n, "%s: write", dst.c_str());
    }
    fclose(f);
    fs::rename(dst, out + "/model.safetensors");
}

void G4DraftTrainer::save_state(const std::string& path) const {
    const std::string tmp = path + ".tmp";
    FILE* f = fopen(tmp.c_str(), "wb");
    GLM_CHECK(f, "cannot write %s", tmp.c_str());
    const uint64_t magic = 0x3154415453344447ull, n = np_;   // "G4DSTAT1"
    fwrite(&magic, 8, 1, f);
    fwrite(&n, 8, 1, f);
    fwrite(&t_, 4, 1, f);
    std::vector<float> buf((size_t)16 << 20);
    for (const float* src : {P_, M_, V_})
        for (size_t i = 0; i < n; i += buf.size()) {
            const size_t c = std::min(buf.size(), n - i);
            download(buf.data(), src + i, c * 4);
            GLM_CHECK(fwrite(buf.data(), 4, c, f) == c, "%s: write", tmp.c_str());
        }
    GLM_CHECK(fclose(f) == 0, "%s: close", tmp.c_str());
    std::filesystem::rename(tmp, path);
}

bool G4DraftTrainer::load_state(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    uint64_t magic = 0, n = 0;
    GLM_CHECK(fread(&magic, 8, 1, f) == 1 && fread(&n, 8, 1, f) == 1 && fread(&t_, 4, 1, f) == 1, "%s: header", path.c_str());
    GLM_CHECK(magic == 0x3154415453344447ull && n == (uint64_t)np_, "%s: not a state of this drafter", path.c_str());
    std::vector<float> buf((size_t)16 << 20);
    for (float* dst : {P_, M_, V_})
        for (size_t i = 0; i < n; i += buf.size()) {
            const size_t c = std::min(buf.size(), n - i);
            GLM_CHECK(fread(buf.data(), 4, c, f) == c, "%s: short", path.c_str());
            upload(dst + i, buf.data(), c * 4);
        }
    fclose(f);
    return true;
}

}  // namespace glm
