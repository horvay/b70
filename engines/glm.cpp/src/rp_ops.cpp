#include "rp_ops.h"
#include <cmath>
#include "ops.h"

namespace glm::rpops {

using namespace rp;
using sycl::nd_item;
using sycl::nd_range;
using sycl::range;

namespace {
template <class T> inline float f(T v) { return static_cast<float>(v); }
constexpr float NEG_INF = -INFINITY;
constexpr int BLK_S = 2, NS_S = RING / (32 * BLK_S), NS_G = 64, NS_D = 65;   // decode attention: pieces of keys
static_assert(RING % (32 * BLK_S) == 0);
constexpr int WG = 128, SGS = 16, PER = WG / SGS;

inline float clamp16(float v) { return sycl::fmin(sycl::fmax(v, -65504.f), 65504.f); }
inline size_t units_range(size_t units) { return (units + PER - 1) / PER * WG; }
using V16 = sycl::vec<half, 16>;
using V8 = sycl::vec<half, 8>;
inline const V16& ld(const half* p) { return *reinterpret_cast<const V16*>(p); }
inline const V8& ld8(const half* p) { return *reinterpret_cast<const V8*>(p); }

const float* inv_freq() {
    static const float* inv = [] {
        auto& q = Dev::get().q;
        float* d = sycl::malloc_device<float>(HD / 2, q);
        std::vector<float> h(HD / 2);
        for (int j = 0; j < HD / 2; j++) h[j] = (float)std::pow(10000.0, -2.0 * j / (double)HD);
        q.memcpy(d, h.data(), h.size() * 4).wait();
        return d;
    }();
    return inv;
}
}  // namespace

void rope_rows(sycl::queue& q, int R, const int* pdev, int p0, float* tab) {
    const float* inv = inv_freq();
    q.parallel_for(range<1>((size_t)R * (HD / 2)), [=](sycl::id<1> i) {
        int r = i / (HD / 2), j = i % (HD / 2);
        int p = pdev ? pdev[r] : p0 + r;
        float th = (float)p * inv[j];
        tab[i * 2] = sycl::cos(th);
        tab[i * 2 + 1] = sycl::sin(th);
    });
}

void post(sycl::queue& q, half* qb, const half* kb, const half* vb, const bf16* qn, const bf16* kn, const float* rope, int R,
          const int* pdev, int p0, half* Kd, half* Vd, int span, bool ring, const int* sdev) {
    const int NU = (qb ? NH : 0) + 2 * NKV;     // query heads, then key heads, then value heads
    const int xb = qb ? 0 : NH;
    const size_t units = (size_t)R * NU;
    q.parallel_for(nd_range<1>(units_range(units), WG), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        auto sg = it.get_sub_group();
        size_t u = (size_t)it.get_group(0) * PER + sg.get_group_id()[0];
        if (u >= units) return;
        int l = sg.get_local_id()[0];        // a lane: 8 dims
        int r = u / NU, x = u % NU + xb;
        int p = pdev ? pdev[r] : p0 + r;
        size_t slot = span ? (size_t)(sdev ? sdev[r] : 0) * span + (ring ? p % span : p) : r;
        if (x >= NH + NKV) {                 // V: the projection as it is
            int h = x - NH - NKV;
            *reinterpret_cast<V8*>(Vd + ((size_t)slot * NKV + h) * HD + l * 8) = ld8(vb + ((size_t)r * NKV + h) * HD + l * 8);
            return;
        }
        const half* src;
        half* dst;
        const bf16* w;
        if (x < NH) {
            src = dst = qb + ((size_t)r * NH + x) * HD;
            w = qn;
        } else {
            int h = x - NH;
            src = kb + ((size_t)r * NKV + h) * HD;
            dst = Kd + ((size_t)slot * NKV + h) * HD;
            w = kn;
        }
        V8 sv = ld8(src + l * 8);
        float acc = 0;
        for (int i = 0; i < 8; i++) { float v = clamp16(f(sv[i])); acc += v * v; }
        acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
        float rm = 1.0f / sycl::sqrt(acc / HD + EPS);
        V8 out;
        for (int i = 0; i < 8; i++) out[i] = half(clamp16(f(sv[i])) * rm * f(w[l * 8 + i]));
        if (rope) {
            // pairs (j, j + 64): lane l and lane l ^ 8
            const float* cs = rope + ((size_t)r * (HD / 2) + (l & 7) * 8) * 2;
            for (int i = 0; i < 8; i++) {
                float mine = f(out[i]);
                float other = sycl::permute_group_by_xor(sg, mine, 8);
                float c = cs[i * 2], s = cs[i * 2 + 1];
                out[i] = half(l < 8 ? mine * c - other * s : mine * c + other * s);
            }
        }
        *reinterpret_cast<V8*>(dst + l * 8) = out;
    });
}

void lerp(sycl::queue& q, const float* e, const half* h, float w, half* y, int R) {
    q.parallel_for(range<1>((size_t)R * D), [=](sycl::id<1> i) {
        float a = e[i];
        y[i] = half(a + w * (f(h[i]) - a));
    });
}

void to16(sycl::queue& q, const float* x, half* y, int R, int ys) {
    q.parallel_for(range<1>((size_t)R * D), [=](sycl::id<1> i) { y[(i / D) * ys + i % D] = half(x[i]); });
}

namespace {
template <class TX, class TY, bool ADD>
void dynconv(sycl::queue& q, const TX* x, const half* dyn, int dys, const bf16* base, TY* out, int R) {
    constexpr int G = D / 16;
    q.parallel_for(range<1>((size_t)R * D), [=](sycl::id<1> id) {
        int t = id / D, d = id % D;
        float acc = 0;
        for (int tap = 0; tap < 4 && tap <= t; tap++)
            acc += (f(base[(size_t)tap * D + d]) + f(dyn[(size_t)t * dys + tap * G + d / 16])) * f(x[(size_t)(t - tap) * D + d]);
        size_t o = (size_t)t * D + d;
        if constexpr (ADD) out[o] += acc;
        else out[o] = TY(acc);
    });
}
}  // namespace

void dynconv_prepare(sycl::queue& q, const half* x, const half* dyn, int dys, const bf16* base, half* out, int R) {
    dynconv<half, half, false>(q, x, dyn, dys, base, out, R);
}
void dynconv_finish(sycl::queue& q, const float* x, const half* dyn, int dys, const bf16* base, float* res, int R) {
    dynconv<float, float, true>(q, x, dyn, dys, base, res, R);
}

void silu_scaled(sycl::queue& q, const half* g, const half* u, half* m, float* scale, int R) {
    constexpr int NW = 256;
    q.parallel_for(nd_range<1>((size_t)R * NW, NW), [=](nd_item<1> it) {
        const int r = it.get_group(0), t = it.get_local_id(0);
        const size_t base = (size_t)r * I;
        auto prod = [&](size_t c) {
            float z = f(g[base + c]);
            return f(half(z / (1.0f + sycl::exp(-z)))) * f(u[base + c]);
        };
        float mx = 0;
        for (int c = t; c < I; c += NW) mx = sycl::fmax(mx, sycl::fabs(prod(c)));
        mx = sycl::reduce_over_group(it.get_group(), mx, sycl::maximum<float>());
        float sc = mx < 16384.0f ? 1.0f : sycl::pow(2.0f, -sycl::ceil(sycl::log2(mx / 16384.0f)) - 1.0f);
        if (t == 0) scale[r] = sc;
        sycl::group_barrier(it.get_group());
        for (int c = t; c < I; c += NW) m[base + c] = half(prod(c) * sc);
    });
}

void mul_rows(sycl::queue& q, float* y, const float* scale, int R) {
    q.parallel_for(range<1>((size_t)R * D), [=](sycl::id<1> i) { y[i] /= scale[i / D]; });
}

void gate_mul(sycl::queue& q, half* o, const half* g, size_t n) {
    q.parallel_for(range<1>(n / 16), [=](sycl::id<1> id) {
        size_t i = id[0] * 16;
        V16 gv = ld(g + i), ov = ld(o + i), out;
        for (int k = 0; k < 16; k++) {
            float s = 1.0f / (1.0f + sycl::exp(-f(gv[k])));
            out[k] = half(f(half(s)) * f(ov[k]));
        }
        *reinterpret_cast<V16*>(o + i) = out;
    });
}

void silu_mul(sycl::queue& q, const half* g, const half* u, half* m, size_t n) {
    q.parallel_for(range<1>(n / 16), [=](sycl::id<1> id) {
        size_t i = id[0] * 16;
        V16 gv = ld(g + i), uv = ld(u + i), out;
        for (int k = 0; k < 16; k++) {
            float z = sycl::fmin(f(gv[k]), CLAMP);
            float up = sycl::fmin(sycl::fmax(f(uv[k]), -CLAMP), CLAMP);
            float si = z / (1.0f + sycl::exp(-z));
            out[k] = half(f(half(si)) * up);
        }
        *reinterpret_cast<V16*>(m + i) = out;
    });
}

// ------------------------------------------------------------------------------------------------ decode attention

// part layout of one call: m [R * NH * NS], l [R * NH * NS], o [R * NH * NS * HD]
size_t attn_part_floats(int R) { return (size_t)R * NH * NS_D * (HD + 2); }

namespace {
// o16 = sum_s 2^(m_s - M) o_s / sum_s 2^(m_s - M) l_s per row and head (m in base 2)
template <int NS>
void combine(sycl::queue& q, const float* part, int R, half* o16) {
    const size_t nm = (size_t)R * NH * NS;
    const float* pm = part;
    const float* pl = part + nm;
    const float* po = part + 2 * nm;
    constexpr int NW = 16;
    q.parallel_for(nd_range<1>((size_t)R * NH * NW, NW), [=](nd_item<1> it) {
        size_t rh = it.get_group(0);
        int t = it.get_local_id(0);
        const float* mp = pm + rh * NS;
        const float* lp = pl + rh * NS;
        float M = NEG_INF;
        for (int s = 0; s < NS; s++) M = sycl::fmax(M, mp[s]);
        float den = 0;
        float e[NS];
        for (int s = 0; s < NS; s++) {
            e[s] = mp[s] == NEG_INF ? 0.0f : sycl::exp2(mp[s] - M);
            den += e[s] * lp[s];
        }
        float inv = 1.0f / den;
        half* out = o16 + rh * HD;
        const float* os = po + rh * NS * HD;
        for (int d = t; d < HD; d += NW) {
            float acc = 0;
            for (int s = 0; s < NS; s++) acc += e[s] * os[(size_t)s * HD + d];
            out[d] = half(acc * inv);
        }
    });
}
}  // namespace

void attn_sliding(sycl::queue& q, const half* q16, const half* K, const half* V_, int R, const int* pdev, const int* sdev, float* part, half* o16) {
    const size_t nm = (size_t)R * NH * NS_S;
    rpfa::dec_sliding(q, q16, K, V_, R, pdev, sdev, 0, part, part + nm, part + 2 * nm, NS_S, BLK_S);
    combine<NS_S>(q, part, R, o16);
}

void attn_global(sycl::queue& q, const half* q16, const half* K, const half* V_, int R, const int* pdev, const int* sdev, int TS, float* part, half* o16) {
    const size_t nm = (size_t)R * NH * NS_G;
    rpfa::dec_global(q, q16, K, V_, R, pdev, sdev, 0, TS, part, part + nm, part + 2 * nm, NS_G);
    combine<NS_G>(q, part, R, o16);
}

void attn_draft(sycl::queue& q, const half* q16, const half* K, const half* V_, int R, const int* pdev, const int* sdev, int span, int win,
                int blocks, float* part, half* o16) {
    const size_t nm = (size_t)R * NH * NS_D;
    GLM_CHECK(span == NS_D * blocks * 32, "attn_draft: ring of %d", span);
    rpfa::dec_draft(q, q16, K, V_, R, pdev, sdev, span, win, part, part + nm, part + 2 * nm, NS_D, blocks);
    combine<NS_D>(q, part, R, o16);
}

// ------------------------------------------------------------------------------------------------ ring <-> rows

void ring_gather(sycl::queue& q, const half* ring, int p_first, int n, half* dst) {
    if (n <= 0) return;
    const size_t row = KD;
    q.parallel_for(range<1>((size_t)n * row / 16), [=](sycl::id<1> id) {
        size_t i = id[0] * 16;
        int t = i / row;
        size_t c = i % row;
        size_t slot = (size_t)(p_first + t) % RING;
        *reinterpret_cast<V16*>(dst + (size_t)t * row + c) = ld(ring + slot * row + c);
    });
}

void ring_scatter(sycl::queue& q, const half* src, int p_first, int n, half* ring) {
    if (n <= 0) return;
    const size_t row = KD;
    q.parallel_for(range<1>((size_t)n * row / 16), [=](sycl::id<1> id) {
        size_t i = id[0] * 16;
        int t = i / row;
        size_t c = i % row;
        size_t slot = (size_t)(p_first + t) % RING;
        *reinterpret_cast<V16*>(ring + slot * row + c) = ld(src + (size_t)t * row + c);
    });
}

}  // namespace glm::rpops
