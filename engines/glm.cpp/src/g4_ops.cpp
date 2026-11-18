#include "g4_ops.h"
#include <cmath>
#include "ops.h"

namespace glm::g4ops {

using namespace g4;
using sycl::nd_item;
using sycl::nd_range;
using sycl::range;

namespace {
template <class T> inline float f(T v) { return static_cast<float>(v); }
constexpr float NEG_INF = -INFINITY;
constexpr int BLK_S = 3, NS_S = RING / (32 * BLK_S), NS_F = 64;   // decode attention: pieces of keys (RING = 1152: 36 blocks)
static_assert(RING % (32 * BLK_S) == 0);
constexpr int WG = 128, SGS = 16, PER = WG / SGS;

inline float clamp16(float v) { return sycl::fmin(sycl::fmax(v, -65504.f), 65504.f); }
inline size_t units_range(size_t units) { return (units + PER - 1) / PER * WG; }
using V16 = sycl::vec<half, 16>;
inline const V16& ld(const half* p) { return *reinterpret_cast<const V16*>(p); }
}  // namespace

void norm16(sycl::queue& q, const float* x, const bf16* w, half* y, int R, int dim) {
    ops::rms_norm<float, half>(q, x, w, y, R, dim, EPS);
}

void add_norm_rows(sycl::queue& q, const half* x, const bf16* w, half* y, int R, int dim) {
    ops::rms_norm<half, half>(q, x, w, y, R, dim, EPS);
}

template <class TA>
void add_norm(sycl::queue& q, float* x, const TA* a, const bf16* wa, float scale, const bf16* wn, half* y, int R, int dim) {
    constexpr int NW = 256;
    const int D = dim;
    q.parallel_for(nd_range<1>((size_t)R * NW, NW), [=](nd_item<1> it) {
        int r = it.get_group(0), t = it.get_local_id(0);
        const TA* ar = a + (size_t)r * D;
        float* xr = x + (size_t)r * D;
        auto rd = [&](int c) {
            float v = f(ar[c]);
            if constexpr (std::is_same_v<TA, half>) v = clamp16(v);
            return v;
        };
        float acc = 0;
        for (int c = t; c < D; c += NW) { float v = rd(c); acc += v * v; }
        acc = sycl::reduce_over_group(it.get_group(), acc, sycl::plus<float>());
        float rm = 1.0f / sycl::sqrt(acc / D + EPS);
        float ax = 0;
        for (int c = t; c < D; c += NW) {
            float nx = (xr[c] + rd(c) * rm * f(wa[c])) * scale;
            xr[c] = nx;
            ax += nx * nx;
        }
        if (wn) {
            ax = sycl::reduce_over_group(it.get_group(), ax, sycl::plus<float>());     // also: every read of a is done
            float rn = 1.0f / sycl::sqrt(ax / D + EPS);
            half* yr = y + (size_t)r * D;
            for (int c = t; c < D; c += NW) yr[c] = half(xr[c] * rn * f(wn[c]));
        }
    });
}
template void add_norm<half>(sycl::queue&, float*, const half*, const bf16*, float, const bf16*, half*, int, int);
template void add_norm<float>(sycl::queue&, float*, const float*, const bf16*, float, const bf16*, half*, int, int);

// ------------------------------------------------------------------------------------------------ RoPE

const float* sliding_inv_freq() {
    static const float* inv = [] {
        auto& q = Dev::get().q;
        float* d = sycl::malloc_device<float>(SD / 2, q);
        std::vector<float> h(SD / 2);
        for (int j = 0; j < SD / 2; j++) h[j] = (float)std::pow(10000.0, -2.0 * j / (double)SD);
        q.memcpy(d, h.data(), h.size() * 4).wait();
        return d;
    }();
    return inv;
}

void rope_rows(sycl::queue& q, int R, const int* pdev, int p0, float* tab) {
    const float* inv = sliding_inv_freq();
    q.parallel_for(range<1>((size_t)R * (SD / 2)), [=](sycl::id<1> i) {
        int r = i / (SD / 2), j = i % (SD / 2);
        int p = pdev ? pdev[r] : p0 + r;
        float th = (float)p * inv[j];
        tab[i * 2] = sycl::cos(th);
        tab[i * 2 + 1] = sycl::sin(th);
    });
}

// ------------------------------------------------------------------------------------------------ after the projections

namespace {
void sliding_post_(sycl::queue& q, half* qb, const half* kb, const half* vb, const bf16* qn, const bf16* kn,
                   const float* srope, int R, const int* pdev, int p0, half* Kd, half* Vd, int ring, bool qonly, const int* sdev) {
    constexpr int NU = NH + 2 * SKV;     // query heads, then key heads, then value heads
    const size_t units = (size_t)R * (qonly ? NH : NU);
    q.parallel_for(nd_range<1>(units_range(units), WG), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        auto sg = it.get_sub_group();
        size_t u = (size_t)it.get_group(0) * PER + sg.get_group_id()[0];
        if (u >= units) return;
        int l = sg.get_local_id()[0];
        int r = qonly ? u / NH : u / NU, x = qonly ? u % NH : u % NU;
        int p = qonly ? 0 : pdev ? pdev[r] : p0 + r;
        size_t slot = ring ? (size_t)(qonly || !sdev ? 0 : sdev[r]) * ring + p % ring : r;
        const half* src;
        half* dst;
        const bf16* w = nullptr;
        bool rot = true;
        if (x < NH) {
            src = dst = qb + ((size_t)r * NH + x) * SD;
            w = qn;
        } else if (x < NH + SKV) {
            int h = x - NH;
            src = kb + ((size_t)r * SKV + h) * SD;
            dst = Kd + ((size_t)slot * SKV + h) * SD;
            w = kn;
        } else {
            int h = x - NH - SKV;
            src = vb + ((size_t)r * SKV + h) * SD;
            dst = Vd + ((size_t)slot * SKV + h) * SD;
            rot = false;
        }
        V16 sv = ld(src + l * 16);
        float acc = 0;
        for (int i = 0; i < 16; i++) { float v = clamp16(f(sv[i])); acc += v * v; }
        acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
        float rm = 1.0f / sycl::sqrt(acc / SD + EPS);
        V16 out;
        for (int i = 0; i < 16; i++) {
            float n = clamp16(f(sv[i])) * rm;
            if (w) n *= f(w[l * 16 + i]);
            out[i] = half(n);
        }
        if (rot) {
            // pairs (j, j + 128): lane l and lane l ^ 8
            const float* cs = srope + ((size_t)r * (SD / 2) + (l & 7) * 16) * 2;
            for (int i = 0; i < 16; i++) {
                float mine = f(out[i]);
                float other = sycl::permute_group_by_xor(sg, mine, 8);
                float c = cs[i * 2], s = cs[i * 2 + 1];
                out[i] = half(l < 8 ? mine * c - other * s : mine * c + other * s);
            }
        }
        *reinterpret_cast<V16*>(dst + l * 16) = out;
    });
}
}  // namespace

void sliding_post(sycl::queue& q, half* qb, const half* kb, const half* vb, const bf16* qn, const bf16* kn,
                  const float* srope, int R, const int* pdev, int p0, half* Kd, half* Vd, int ring, const int* sdev) {
    sliding_post_(q, qb, kb, vb, qn, kn, srope, R, pdev, p0, Kd, Vd, ring, false, sdev);
}
void sliding_q(sycl::queue& q, half* qb, const bf16* qn, const float* srope, int R) {
    sliding_post_(q, qb, nullptr, nullptr, qn, nullptr, srope, R, nullptr, 0, nullptr, nullptr, 0, true, nullptr);
}

namespace {
void full_post_(sycl::queue& q, half* qb, const half* kb, const bf16* qn, const bf16* kn, const float* rope, int R,
                const int* pdev, int p0, half* Pc, half* Kr, float* Rc, bool qonly, const int* sdev, int TS) {
    constexpr int NU = NH + FKV;
    const size_t units = (size_t)R * (qonly ? NH : NU);
    q.parallel_for(nd_range<1>(units_range(units), WG), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        auto sg = it.get_sub_group();
        size_t u = (size_t)it.get_group(0) * PER + sg.get_group_id()[0];
        if (u >= units) return;
        int l = sg.get_local_id()[0];
        int r = qonly ? u / NH : u / NU, x = qonly ? u % NH : u % NU;
        int p = pdev ? pdev[r] : p0 + r;
        // a lane: dims 16 l .. and 256 + 16 l .., so a rotated pair (j, j + 256) stays in one lane
        const int ia = l * 16, ib = FD / 2 + l * 16;
        if (x < NH) {
            half* row = qb + ((size_t)r * NH + x) * FD;
            V16 a = ld(row + ia), b = ld(row + ib);
            float acc = 0;
            for (int i = 0; i < 16; i++) {
                float va = clamp16(f(a[i])), vb = clamp16(f(b[i]));
                acc += va * va + vb * vb;
            }
            acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
            float rm = 1.0f / sycl::sqrt(acc / FD + EPS);
            V16 oa, ob;
            for (int i = 0; i < 16; i++) {
                float na = f(half(clamp16(f(a[i])) * rm * f(qn[ia + i])));
                float nb = f(half(clamp16(f(b[i])) * rm * f(qn[ib + i])));
                if (l < FROT / 16) {
                    const float* cs = rope + ((size_t)p * FROT + ia + i) * 2;
                    float c = cs[0], s = cs[1];
                    float ra = na * c - nb * s;
                    nb = nb * c + na * s;
                    na = ra;
                }
                oa[i] = half(na);
                ob[i] = half(nb);
            }
            *reinterpret_cast<V16*>(row + ia) = oa;
            *reinterpret_cast<V16*>(row + ib) = ob;
        } else {
            int h = x - NH;
            const half* src = kb + ((size_t)r * FKV + h) * FD;
            const size_t ai = (size_t)(sdev ? sdev[r] : 0) * TS + p;      // the row in this game's cache
            half* dst = Pc + (ai * FKV + h) * FD;
            V16 a = ld(src + ia), b = ld(src + ib);
            float acc = 0;
            for (int i = 0; i < 16; i++) {
                float va = clamp16(f(a[i])), vb = clamp16(f(b[i]));
                acc += va * va + vb * vb;
            }
            acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
            if (l == 0) Rc[ai * FKV + h] = 1.0f / sycl::sqrt(acc / FD + EPS);
            *reinterpret_cast<V16*>(dst + ia) = a;
            *reinterpret_cast<V16*>(dst + ib) = b;
            if (l < FROT / 16) {
                half* kr = Kr + (ai * FKV + h) * (2 * FROT);
                V16 ra, rb;
                for (int i = 0; i < 16; i++) {
                    float ua = clamp16(f(a[i])) * f(kn[ia + i]), ub = clamp16(f(b[i])) * f(kn[ib + i]);
                    const float* cs = rope + ((size_t)p * FROT + ia + i) * 2;
                    float c = cs[0], s = cs[1];
                    ra[i] = half(ua * c - ub * s);
                    rb[i] = half(ub * c + ua * s);
                }
                *reinterpret_cast<V16*>(kr + ia) = ra;
                *reinterpret_cast<V16*>(kr + FROT + ia) = rb;
            }
        }
    });
}
}  // namespace

void full_post(sycl::queue& q, half* qb, const half* kb, const bf16* qn, const bf16* kn, const float* rope, int R,
               const int* pdev, int p0, half* Pc, half* Kr, float* Rc, const int* sdev, int TS) {
    full_post_(q, qb, kb, qn, kn, rope, R, pdev, p0, Pc, Kr, Rc, false, sdev, TS);
}
void full_q(sycl::queue& q, half* qb, const bf16* qn, const float* rope, int R, const int* pdev) {
    full_post_(q, qb, nullptr, qn, nullptr, rope, R, pdev, 0, nullptr, nullptr, nullptr, true, nullptr, 0);
}

void full_kv(sycl::queue& q, const half* Pc, const float* Rc, const bf16* kn, const float* rope, int k0, int n,
             half* K, half* V_) {
    const size_t units = (size_t)n * FKV;
    q.parallel_for(nd_range<1>(units_range(units), WG), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        auto sg = it.get_sub_group();
        size_t u = (size_t)it.get_group(0) * PER + sg.get_group_id()[0];
        if (u >= units) return;
        int l = sg.get_local_id()[0];
        int t = u / FKV, h = u % FKV;
        int p = k0 + t;
        const int ia = l * 16, ib = FD / 2 + l * 16;
        const half* src = Pc + ((size_t)p * FKV + h) * FD;
        float rv = Rc[(size_t)p * FKV + h];
        V16 a = ld(src + ia), b = ld(src + ib);
        V16 ka, kb, va, vb;
        for (int i = 0; i < 16; i++) {
            float pa = clamp16(f(a[i])) * rv, pb = clamp16(f(b[i])) * rv;
            va[i] = half(pa);
            vb[i] = half(pb);
            float ua = pa * f(kn[ia + i]), ub = pb * f(kn[ib + i]);
            if (l < FROT / 16) {
                const float* cs = rope + ((size_t)p * FROT + ia + i) * 2;
                float c = cs[0], s = cs[1];
                float ra = ua * c - ub * s;
                ub = ub * c + ua * s;
                ua = ra;
            }
            ka[i] = half(ua);
            kb[i] = half(ub);
        }
        half* kd = K + (size_t)u * FD;
        half* vd = V_ + (size_t)u * FD;
        *reinterpret_cast<V16*>(kd + ia) = ka;
        *reinterpret_cast<V16*>(kd + ib) = kb;
        *reinterpret_cast<V16*>(vd + ia) = va;
        *reinterpret_cast<V16*>(vd + ib) = vb;
    });
}

// ------------------------------------------------------------------------------------------------ decode attention

// part layout of one call: m [R * NH * NS], l [R * NH * NS], o [R * NH * NS * HD]
size_t attn_part_floats(int R) { return (size_t)R * NH * NS_F * (FD + 2); }

namespace {
// o16 = sum_s 2^(m_s - M) o_s / sum_s 2^(m_s - M) l_s per row and head (m in base 2)
template <int HD, int NS>
void combine(sycl::queue& q, const float* part, int R, half* o16) {
    const size_t nm = (size_t)R * NH * NS;
    const float* pm = part;
    const float* pl = part + nm;
    const float* po = part + 2 * nm;
    constexpr int NW = 64;
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

void attn_sliding_dec(sycl::queue& q, const half* q16, const half* K, const half* V_, int R, const int* pdev, const int* sdev,
                      int p0, float* part, half* o16) {
    constexpr int NS = NS_S, HD = SD;
    const size_t nm = (size_t)R * NH * NS;
    g4fa::sliding_dec(q, q16, K, V_, R, pdev, sdev, p0, part, part + nm, part + 2 * nm, NS, BLK_S);
    combine<HD, NS>(q, part, R, o16);
}

void attn_full_dec(sycl::queue& q, const half* q16, const half* Pc, const half* Kr, const float* Rc, const half* kn, int R,
                   const int* pdev, const int* sdev, int TS, int p0, float* part, half* o16) {
    constexpr int NS = NS_F, HD = FD;
    const size_t nm = (size_t)R * NH * NS;
    g4fa::full_dec(q, q16, Pc, Kr, Rc, kn, R, pdev, sdev, TS, p0, part, part + nm, part + 2 * nm, NS);
    combine<HD, NS>(q, part, R, o16);
}

// ------------------------------------------------------------------------------------------------ ring <-> rows

void ring_gather(sycl::queue& q, const half* ring, int p_first, int n, half* dst) {
    if (n <= 0) return;
    const size_t row = (size_t)SKV * SD;
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
    const size_t row = (size_t)SKV * SD;
    q.parallel_for(range<1>((size_t)n * row / 16), [=](sycl::id<1> id) {
        size_t i = id[0] * 16;
        int t = i / row;
        size_t c = i % row;
        size_t slot = (size_t)(p_first + t) % RING;
        *reinterpret_cast<V16*>(ring + slot * row + c) = ld(src + (size_t)t * row + c);
    });
}

// ------------------------------------------------------------------------------------------------ MLP, head

void gelu_mul(sycl::queue& q, const half* g, const half* u, half* m, size_t n) {
    q.parallel_for(range<1>(n / 16), [=](sycl::id<1> id) {
        size_t i = id[0] * 16;
        V16 gv = ld(g + i), uv = ld(u + i), out;
        for (int k = 0; k < 16; k++) {
            float z = f(gv[k]);
            // gelu_pytorch_tanh: 0.5 z (1 + tanh(sqrt(2 / pi) (z + 0.044715 z^3)))
            float ge = 0.5f * z * (1.0f + sycl::tanh(0.7978845608028654f * (z + 0.044715f * z * z * z)));
            out[k] = half(f(half(ge)) * f(uv[k]));
        }
        *reinterpret_cast<V16*>(m + i) = out;
    });
}

namespace {
inline float cap(float l) { return SOFTCAP * sycl::tanh(l / SOFTCAP); }

template <int QR>
void head_q(sycl::queue& q, const half* x, const half* W, int Q, half* out) {
    // one sub-group per CN columns of the head, read together: W rows are contiguous, each lane 8 values of a column
    constexpr int NW = 256, SG = 16, CN = 4;
    constexpr int SGP = NW / SG, COLS = SGP * CN;
    q.parallel_for(nd_range<1>((size_t)(V / COLS) * NW, NW), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        auto sg = it.get_sub_group();
        int l = sg.get_local_id()[0];
        size_t c0 = (size_t)it.get_group(0) * COLS + (size_t)sg.get_group_id()[0] * CN;
        using V8 = sycl::vec<half, 8>;
        float acc[QR][CN] = {};
        for (int d = l * 8; d < D; d += SG * 8) {
            V8 xv[QR];
            float xf[QR][8];
            for (int r = 0; r < Q; r++) {
                xv[r] = *reinterpret_cast<const V8*>(x + (size_t)r * D + d);
                for (int e = 0; e < 8; e++) xf[r][e] = f(xv[r][e]);
            }
            for (int c = 0; c < CN; c++) {
                V8 wv = *reinterpret_cast<const V8*>(W + (c0 + c) * D + d);
                for (int e = 0; e < 8; e++) {
                    float wf = f(wv[e]);
                    for (int r = 0; r < Q; r++) acc[r][c] += xf[r][e] * wf;
                }
            }
        }
        for (int r = 0; r < Q; r++)
            for (int c = 0; c < CN; c++) {
                float v = sycl::reduce_over_group(sg, acc[r][c], sycl::plus<float>());
                if (l == 0) out[(size_t)r * V + c0 + c] = half(cap(v));
            }
    });
}
}  // namespace

void head_logits(sycl::queue& q, const half* x, const half* W, int Q, half* out) {
    GLM_CHECK(Q <= QMAX, "head_logits: %d rows", Q);
    if (Q == 1) head_q<1>(q, x, W, Q, out);
    else if (Q <= 2) head_q<2>(q, x, W, Q, out);
    else if (Q <= 4) head_q<4>(q, x, W, Q, out);
    else if (Q <= 8) head_q<8>(q, x, W, Q, out);
    else head_q<16>(q, x, W, Q, out);
}

void softcap(sycl::queue& q, half* l, size_t n) {
    q.parallel_for(range<1>(n), [=](sycl::id<1> i) { l[i] = half(cap(f(l[i]))); });
}

void concat(sycl::queue& q, const half* a, const half* b, half* z, int R, int D_) {
    q.parallel_for(range<1>((size_t)R * D_ / 16), [=](sycl::id<1> id) {
        size_t i = id[0] * 16, r = i / D_, c = i % D_;
        *reinterpret_cast<V16*>(z + r * 2 * D_ + c) = ld(a + r * D_ + c);
        *reinterpret_cast<V16*>(z + r * 2 * D_ + D_ + c) = ld(b + r * D_ + c);
    });
}

void f32_to_f16(sycl::queue& q, const float* a, half* b, size_t n) {
    q.parallel_for(range<1>(n), [=](sycl::id<1> i) { b[i] = half(a[i]); });
}

void fill(sycl::queue& q, float* a, float v, size_t n) {
    q.parallel_for(range<1>(n), [=](sycl::id<1> i) { a[i] = v; });
}

}  // namespace glm::g4ops
