#include "q4_ops.h"
#include <cmath>

namespace glm::q4ops {

using namespace q4;
using sycl::nd_item;
using sycl::nd_range;
using sycl::range;

namespace {
template <class T> inline float f(T v) { return static_cast<float>(v); }
inline float sigm(float x) { return 1.0f / (1.0f + sycl::exp(-x)); }
inline float silu(float x) { return x / (1.0f + sycl::exp(-x)); }
constexpr float NEG_INF = -INFINITY;

// NeoX rotate-half over the first 64 dims of a head held in local memory (buf), at position p (theta 1e7)
inline float rope_at(const float* buf, int l, int p) {
    if (l >= ROT) return buf[l];
    int j = l & (ROT / 2 - 1);
    float inv = sycl::pow(1.0e7f, -(float)(2 * j) / (float)ROT);
    float th = (float)p * inv, c = sycl::cos(th), s = sycl::sin(th);
    return l < ROT / 2 ? buf[l] * c - buf[l + ROT / 2] * s : buf[l] * c + buf[l - ROT / 2] * s;
}

// floats ordered as unsigned integers (larger float, larger key)
inline uint32_t okey(float x) {
    uint32_t b = sycl::bit_cast<uint32_t>(x);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
}  // namespace

// ------------------------------------------------------------------------------------------------ sites

void site_norm(sycl::queue& q, const float* X, const float* w1p, half* n16, int rows) {
    q.parallel_for(nd_range<1>((size_t)rows * S * 256, 256), [=](nd_item<1> it) {
        size_t g = it.get_group(0);
        int l = it.get_local_id(0);
        const float* x = X + g * D;
        const float* w = w1p + (g % S) * D;
        half* o = n16 + g * D;
        float ss = 0;
        for (int d = l; d < D; d += 256) ss += x[d] * x[d];
        ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
        float rs = sycl::rsqrt(ss / D + EPS);
        for (int d = l; d < D; d += 256) o[d] = half(x[d] * rs * w[d]);
    });
}

void site_gates(sycl::queue& q, const float* t, int nd, half* a16, float* inj, int rows) {
    q.parallel_for(range<2>(rows, nd), [=](sycl::item<2> it) {
        int r = it[0], c = it[1];
        float v = t[(size_t)r * nd + c] * 0.25f;
        if (c < R) a16[(size_t)r * R + c] = half(silu(v));
        else inj[r * S + c - R] = 2.0f * sigm(v);
    });
}

void site_combine(sycl::queue& q, const float* gpre, const half* n16, half* y, int rows) {
    q.parallel_for(range<1>((size_t)rows * D), [=](sycl::id<1> i) {
        size_t r = i / D, d = i % D;
        const float* gp = gpre + r * SD + d;
        const half* nn = n16 + r * SD + d;
        float acc = 0;
        for (int s = 0; s < S; s++) acc += sigm(gp[s * D]) * f(nn[s * D]);
        y[i] = half(acc * 0.25f);
    });
}

void site_apply(sycl::queue& q, float* X, const float* inj, const float* o, int rows) {
    q.parallel_for(range<1>((size_t)rows * SD), [=](sycl::id<1> i) {
        size_t r = i / SD, s = (i % SD) / D, d = i % D;
        X[i] += inj[r * S + s] * o[r * D + d];
    });
}

void expand(sycl::queue& q, const float* e, float* X, int rows) {
    q.parallel_for(range<1>((size_t)rows * SD), [=](sycl::id<1> i) {
        size_t r = i / SD, d = i % D;
        X[i] = e[r * D + d];
    });
}

// ------------------------------------------------------------------------------------------------ GDN

void gdn_gates(sycl::queue& q, const float* ab, const float* neg_a, const float* dtb, float* beta, float* g, int rows) {
    q.parallel_for(range<2>(rows, GV), [=](sycl::item<2> it) {
        int r = it[0], j = it[1];
        float b = ab[(size_t)r * 2 * GV + j], a = ab[(size_t)r * 2 * GV + GV + j];
        beta[r * GV + j] = f(bf16(sigm(b)));
        float x = a + dtb[j];
        float sp = x > 20.0f ? x : sycl::log1p(sycl::exp(x));
        g[r * GV + j] = neg_a[j] * sp;
    });
}

void gdn_conv(sycl::queue& q, const float* qkv, bf16* state, const bf16* w, bf16* out, int rows) {
    constexpr int CK = GCONV, ST = GCONV - 1;
    q.parallel_for(range<1>(GQKV), [=](sycl::id<1> ci) {
        int c = ci;
        bf16* sc = state + (size_t)c * ST;
        float wk[CK];
        for (int k = 0; k < CK; k++) wk[k] = f(w[c * CK + k]);
        float win[CK];
        for (int k = 0; k < ST; k++) win[k] = f(sc[k]);
        for (int t = 0; t < rows; t++) {
            win[ST] = f(bf16(qkv[(size_t)t * GQKV + c]));
            float acc = 0;
            for (int k = 0; k < CK; k++) acc += wk[k] * win[k];
            out[(size_t)t * GQKV + c] = bf16(silu(acc));
            for (int k = 0; k < ST; k++) win[k] = win[k + 1];
        }
        for (int k = 0; k < ST; k++) sc[k] = bf16(win[k]);
    });
}

void gdn_delta(sycl::queue& q, const bf16* c, const float* beta, const float* g, float* rec, bf16* core, int rows) {
    // one work-group per (v-head j, 32 v columns); thread i owns state row k = i. v-head j reads k-head j / 3
    constexpr int DK = GDK, DV = GDV, SG = 16, NSG = DK / SG, VB = 32;
    const float scale = 1.0f / sycl::sqrt((float)DK);
    q.submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 2> part(sycl::range<2>(NSG, VB), hd);
        sycl::local_accessor<float, 1> red(sycl::range<1>(2 * NSG), hd);
        hd.parallel_for(nd_range<1>((size_t)GV * (DV / VB) * DK, DK), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            int gid = it.get_group(0), i = it.get_local_id(0);
            int j = gid / (DV / VB), vb = gid % (DV / VB), kh = j / (GV / GK);
            auto grp = it.get_group();
            auto sg = it.get_sub_group();
            int sgi = sg.get_group_id()[0], lane = sg.get_local_id()[0];
            float* Sp = rec + (size_t)j * DK * DV;
            float Sr[VB];
            for (int v = 0; v < VB; v++) Sr[v] = Sp[(size_t)i * DV + vb * VB + v];
            for (int t = 0; t < rows; t++) {
                const bf16* row = c + (size_t)t * GQKV;
                float qi = f(row[kh * DK + i]), ki = f(row[GK * DK + kh * DK + i]);
                float qq = sycl::reduce_over_group(sg, qi * qi, sycl::plus<float>());
                float kk = sycl::reduce_over_group(sg, ki * ki, sycl::plus<float>());
                if (lane == 0) { red[sgi] = qq; red[NSG + sgi] = kk; }
                float a = sycl::exp(g[t * GV + j]);
                float bt = beta[t * GV + j];
                for (int v = 0; v < VB; v++) Sr[v] *= a;
                sycl::group_barrier(grp);
                float qn = 0, kn = 0;
                for (int s = 0; s < NSG; s++) { qn += red[s]; kn += red[NSG + s]; }
                qi = qi * sycl::rsqrt(qn + 1e-6f) * scale;
                ki = ki * sycl::rsqrt(kn + 1e-6f);
                for (int v = 0; v < VB; v++) {
                    float p = sycl::reduce_over_group(sg, Sr[v] * ki, sycl::plus<float>());
                    if (lane == 0) part[sgi][v] = p;
                }
                sycl::group_barrier(grp);
                const bf16* vr = row + 2 * GK * DK + j * DV + vb * VB;
                for (int v = 0; v < VB; v++) {
                    float sk = 0;
                    for (int s = 0; s < NSG; s++) sk += part[s][v];
                    Sr[v] += bt * ki * (f(vr[v]) - sk);
                }
                sycl::group_barrier(grp);
                for (int v = 0; v < VB; v++) {
                    float p = sycl::reduce_over_group(sg, Sr[v] * qi, sycl::plus<float>());
                    if (lane == 0) part[sgi][v] = p;
                }
                sycl::group_barrier(grp);
                if (i < VB) {
                    float o = 0;
                    for (int s = 0; s < NSG; s++) o += part[s][i];
                    core[((size_t)t * GV + j) * DV + vb * VB + i] = bf16(o);
                }
                sycl::group_barrier(grp);
            }
            for (int v = 0; v < VB; v++) Sp[(size_t)i * DV + vb * VB + v] = Sr[v];
        });
    });
}

void gdn_norm(sycl::queue& q, const bf16* core, const float* z, const float* w, half* out, int rows) {
    q.parallel_for(nd_range<1>((size_t)rows * GV * GDV, GDV), [=](nd_item<1> it) {
        size_t base = it.get_group(0) * GDV;
        int d = it.get_local_id(0);
        float h = f(core[base + d]);
        float ss = sycl::reduce_over_group(it.get_group(), h * h, sycl::plus<float>());
        h = h * sycl::rsqrt(ss / GDV + EPS) * w[d];
        out[base + d] = half(h * sigm(z[base + d]));
    });
}

// ------------------------------------------------------------------------------------------------ attention

void attn_prep(sycl::queue& q, const half* qg, const half* k, const half* v, const half* ik, const float* qn,
               const float* kn, const float* iqn, int rows, int p0, half* qn16, half* Kc, half* Vc, float* qi, half* rk,
               const int* pdev) {
    constexpr int U = AH + 2 * AKV + IH + 1;     // per row: 24 q heads, 2 k, 2 v, 4 indexer q, the raw indexer key
    q.submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> buf(sycl::range<1>(256), hd);
        hd.parallel_for(nd_range<1>((size_t)rows * U * 256, 256), [=](nd_item<1> it) {
            int gid = it.get_group(0), l = it.get_local_id(0);
            int r = gid / U, u = gid % U, p = (pdev ? *pdev : p0) + r;
            auto grp = it.get_group();
            if (u < AH || u < AH + AKV) {
                bool isq = u < AH;
                int h = isq ? u : u - AH;
                float x = isq ? f(qg[(size_t)r * AH * 2 * AD + h * 2 * AD + l]) : f(k[(size_t)r * AKV * AD + h * AD + l]);
                float ss = sycl::reduce_over_group(grp, x * x, sycl::plus<float>());
                buf[l] = x * sycl::rsqrt(ss / AD + EPS) * (isq ? qn : kn)[l];
                sycl::group_barrier(grp);
                float o = rope_at(&buf[0], l, p);
                if (isq) qn16[((size_t)r * AH + h) * AD + l] = half(o);
                else Kc[((size_t)p * AKV + h) * AD + l] = half(o);
            } else if (u < AH + 2 * AKV) {
                int h = u - AH - AKV;
                Vc[((size_t)p * AKV + h) * AD + l] = v[(size_t)r * AKV * AD + h * AD + l];
            } else if (u < AH + 2 * AKV + IH) {
                int h = u - AH - 2 * AKV;
                float x = l < ID ? f(ik[(size_t)r * (IH + 1) * ID + h * ID + l]) : 0.0f;
                float ss = sycl::reduce_over_group(grp, x * x, sycl::plus<float>());
                if (l < ID) buf[l] = x * sycl::rsqrt(ss / ID + EPS) * iqn[l];
                sycl::group_barrier(grp);
                if (l < ID) qi[((size_t)r * IH + h) * ID + l] = rope_at(&buf[0], l, p);
            } else if (l < ID) {
                rk[(size_t)p * ID + l] = ik[(size_t)r * (IH + 1) * ID + IH * ID + l];
            }
        });
    });
}

void attn_pool(sycl::queue& q, const half* rk, const float* ikn, half* pk, int b0, int b1) {
    if (b1 <= b0) return;
    q.submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> buf(sycl::range<1>(ID), hd);
        hd.parallel_for(nd_range<1>((size_t)(b1 - b0) * ID, ID), [=](nd_item<1> it) {
            int b = b0 + it.get_group(0), l = it.get_local_id(0);
            const half* r0 = rk + (size_t)b * BLK * ID + l;
            float m = (f(r0[0]) + f(r0[ID]) + f(r0[2 * ID]) + f(r0[3 * ID])) * 0.25f;
            m = f(half(m));
            float ss = sycl::reduce_over_group(it.get_group(), m * m, sycl::plus<float>());
            buf[l] = m * sycl::rsqrt(ss / ID + EPS) * ikn[l];
            sycl::group_barrier(it.get_group());
            pk[(size_t)b * ID + l] = half(rope_at(&buf[0], l, b * BLK));
        });
    });
}

void attn_pool_dec(sycl::queue& q, const half* rk, const float* ikn, half* pk, const int* pdev) {
    q.submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> buf(sycl::range<1>(ID), hd);
        hd.parallel_for(nd_range<1>(ID, ID), [=](nd_item<1> it) {
            int p = *pdev, l = it.get_local_id(0);
            if ((p + 1) % BLK) return;
            int b = (p + 1) / BLK - 1;
            const half* r0 = rk + (size_t)b * BLK * ID + l;
            float m = (f(r0[0]) + f(r0[ID]) + f(r0[2 * ID]) + f(r0[3 * ID])) * 0.25f;
            m = f(half(m));
            float ss = sycl::reduce_over_group(it.get_group(), m * m, sycl::plus<float>());
            buf[l] = m * sycl::rsqrt(ss / ID + EPS) * ikn[l];
            sycl::group_barrier(it.get_group());
            pk[(size_t)b * ID + l] = half(rope_at(&buf[0], l, b * BLK));
        });
    });
}

void attn_select(sycl::queue& q, const float* qi, const half* pk, int rows, int p0, int nbcap, float* sc, int* list,
                 int* cnt, const int* pdev) {
    const int nbmax = nbcap;
    if (pdev || (p0 + rows) / BLK > BUDGET) {
        const float rs = 1.0f / sycl::sqrt((float)ID);
        q.parallel_for(range<2>(rows, nbmax), [=](sycl::item<2> it) {
            int r = it[0], b = it[1], p = (pdev ? *pdev : p0) + r, nb = (p + 1) / BLK;
            if (nb <= BUDGET || b >= nb) return;
            const half* kp = pk + (size_t)b * ID;
            const float* qr = qi + (size_t)r * IH * ID;
            float s = 0;
            for (int h = 0; h < IH; h++) {
                float d = 0;
                for (int e = 0; e < ID; e++) d += qr[h * ID + e] * f(kp[e]);
                s += sycl::fmax(d, 0.0f);
            }
            sc[(size_t)r * nbmax + b] = s * rs;
        });
    }
    constexpr int WG = 1024;
    q.parallel_for(nd_range<1>((size_t)rows * WG, WG), [=](nd_item<1> it) {
        int r = it.get_group(0), l = it.get_local_id(0), p = (pdev ? *pdev : p0) + r, nb = (p + 1) / BLK;
        auto grp = it.get_group();
        if (nb <= BUDGET) {
            if (l == 0) cnt[r] = -1;
            return;
        }
        const float* s = sc + (size_t)r * nbmax;
        // the largest T with count(key >= T) >= BUDGET, bit by bit from the top
        uint32_t T = 0;
        for (int bit = 31; bit >= 0; bit--) {
            uint32_t cand = T | (1u << bit);
            int c = 0;
            for (int b = l; b < nb; b += WG) c += okey(s[b]) >= cand;
            c = sycl::reduce_over_group(grp, c, sycl::plus<int>());
            if (c >= BUDGET) T = cand;
        }
        int seg = (nb + WG - 1) / WG, b0 = l * seg, b1 = sycl::min(nb, b0 + seg);
        int ngt = 0, neq = 0;
        for (int b = b0; b < b1; b++) {
            uint32_t k = okey(s[b]);
            ngt += k > T;
            neq += k == T;
        }
        int gt_all = sycl::reduce_over_group(grp, ngt, sycl::plus<int>());
        int need = BUDGET - gt_all;                       // ties at T, lowest blocks first
        int eq_before = sycl::exclusive_scan_over_group(grp, neq, sycl::plus<int>());
        int take_eq = sycl::max(0, sycl::min(neq, need - eq_before));
        int mine = ngt + take_eq;
        int off = sycl::exclusive_scan_over_group(grp, mine, sycl::plus<int>());
        int* lr = list + (size_t)r * NSEL;
        int eq_seen = 0;
        for (int b = b0; b < b1; b++) {
            uint32_t k = okey(s[b]);
            bool take = k > T;
            if (k == T) { take = eq_seen < take_eq; eq_seen++; }
            if (take) {
                for (int e = 0; e < BLK; e++) lr[off * BLK + e] = b * BLK + e;
                off++;
            }
        }
        int tail = p + 1 - nb * BLK;
        if (l < tail) lr[BUDGET * BLK + l] = nb * BLK + l;
        if (l == 0) cnt[r] = BUDGET * BLK + tail;
    });
}

namespace {
constexpr int CH = 128, NCH = (NSEL + CH - 1) / CH, HG = AH / AKV;      // tokens per chunk, chunks, heads per kv head
constexpr int PSZ = 2 * HG + HG * AD;                                 // a chunk's m[12], l[12], o[12][256]
}  // namespace

size_t attn_part_floats(int rows) { return (size_t)rows * AKV * NCH * PSZ; }

void attn_core(sycl::queue& q, const half* qn16, const half* Kc, const half* Vc, const int* list, const int* cnt,
               const half* qg, int rows, int p0, float* part, half* o16, const int* pdev) {
    q.submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> qs(sycl::range<1>(HG * AD), hd);
        sycl::local_accessor<float, 1> ps(sycl::range<1>(HG * CH), hd);
        sycl::local_accessor<int, 1> tk(sycl::range<1>(CH), hd);
        hd.parallel_for(nd_range<1>((size_t)rows * AKV * NCH * 256, 256), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            int gid = it.get_group(0), l = it.get_local_id(0);
            int c = gid % NCH, g = (gid / NCH) % AKV, r = gid / (NCH * AKV), p = (pdev ? *pdev : p0) + r;
            auto grp = it.get_group();
            int cr = cnt[r];
            bool dense = cr < 0;
            int n = dense ? p + 1 : cr, t0 = c * CH;
            float* P = part + ((size_t)(r * AKV + g) * NCH + c) * PSZ;
            if (t0 >= n) {
                if (l < HG) { P[l] = NEG_INF; P[HG + l] = 0.0f; }
                return;
            }
            int nt = sycl::min(CH, n - t0);
            for (int i = l; i < HG * AD; i += 256) qs[i] = f(qn16[((size_t)r * AH + g * HG) * AD + i]);
            if (l < nt) tk[l] = dense ? t0 + l : list[(size_t)r * NSEL + t0 + l];
            sycl::group_barrier(grp);
            {
                int j = l & (CH - 1), hh = l / CH;          // token slot, half of the heads (6 each)
                float acc[HG / 2] = {};
                if (j < nt) {
                    const half* kp = Kc + ((size_t)tk[j] * AKV + g) * AD;
                    for (int d = 0; d < AD; d += 8) {
                        sycl::vec<half, 8> kv = *reinterpret_cast<const sycl::vec<half, 8>*>(kp + d);
                        for (int e = 0; e < 8; e++) {
                            float kf = f(kv[e]);
                            for (int h = 0; h < HG / 2; h++) acc[h] += qs[(hh * (HG / 2) + h) * AD + d + e] * kf;
                        }
                    }
                }
                for (int h = 0; h < HG / 2; h++) ps[(hh * (HG / 2) + h) * CH + j] = j < nt ? acc[h] * 0.0625f : NEG_INF;
            }
            sycl::group_barrier(grp);
            {
                auto sg = it.get_sub_group();
                int h = sg.get_group_id()[0], lane = sg.get_local_id()[0];
                if (h < HG) {
                    float m = NEG_INF;
                    for (int j = lane; j < CH; j += 16) m = sycl::fmax(m, ps[h * CH + j]);
                    m = sycl::reduce_over_group(sg, m, sycl::maximum<float>());
                    float sum = 0;
                    for (int j = lane; j < CH; j += 16) {
                        float e = j < nt ? sycl::exp(ps[h * CH + j] - m) : 0.0f;
                        ps[h * CH + j] = e;
                        sum += e;
                    }
                    sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>());
                    if (lane == 0) { P[h] = m; P[HG + h] = sum; }
                }
            }
            sycl::group_barrier(grp);
            {
                float acc[HG] = {};
                for (int j = 0; j < nt; j++) {
                    float vv = f(Vc[((size_t)tk[j] * AKV + g) * AD + l]);
                    for (int h = 0; h < HG; h++) acc[h] += ps[h * CH + j] * vv;
                }
                for (int h = 0; h < HG; h++) P[2 * HG + h * AD + l] = acc[h];
            }
        });
    });
    q.parallel_for(range<1>((size_t)rows * AH * AD), [=](sycl::id<1> i) {
        int r = i / (AH * AD), h = (i / AD) % AH, d = i % AD, g = h / HG, hh = h % HG;
        const float* P = part + (size_t)(r * AKV + g) * NCH * PSZ;
        float M = NEG_INF;
        for (int c = 0; c < NCH; c++) M = sycl::fmax(M, P[c * PSZ + hh]);
        float num = 0, den = 0;
        for (int c = 0; c < NCH; c++) {
            float m = P[c * PSZ + hh];
            if (m == NEG_INF) continue;
            float e = sycl::exp(m - M);
            den += P[c * PSZ + HG + hh] * e;
            num += P[c * PSZ + 2 * HG + hh * AD + d] * e;
        }
        float gate = f(qg[(size_t)r * AH * 2 * AD + h * 2 * AD + AD + d]);
        o16[(size_t)r * AH * AD + h * AD + d] = half(num / den * sigm(gate));
    });
}

// ------------------------------------------------------------------------------------------------ MoE

void route(sycl::queue& q, const half* rl, int stride, int rows, int* sel, float* w) {
    q.parallel_for(range<1>(rows), [=](sycl::id<1> ri) {
        int r = ri;
        const half* lg = rl + (size_t)r * stride;
        int idx[TOPK];
        float val[TOPK];
        for (int k = 0; k < TOPK; k++) { idx[k] = -1; val[k] = NEG_INF; }
        for (int e = 0; e < E; e++) {
            float v = f(lg[e]);
            if (!(v > val[TOPK - 1])) continue;          // ties keep the lower index (already in)
            int k = TOPK - 1;
            while (k > 0 && v > val[k - 1]) { val[k] = val[k - 1]; idx[k] = idx[k - 1]; k--; }
            val[k] = v;
            idx[k] = e;
        }
        float sum = 0, ex[TOPK];
        for (int k = 0; k < TOPK; k++) { ex[k] = sycl::exp(val[k] - val[0]); sum += ex[k]; }
        for (int k = 0; k < TOPK; k++) {
            sel[r * TOPK + k] = idx[k];
            w[r * TOPK + k] = ex[k] / (sum + 1e-20f);
        }
    });
}

void count_use(sycl::queue& q, const int* sel, int rows, int* use, const uint8_t* in_vram, unsigned long long* stats) {
    q.parallel_for(range<1>((size_t)rows * TOPK), [=](sycl::id<1> i) {
        int e = sel[i];
        sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device> u(use[e]);
        u.fetch_add(1);
        sycl::atomic_ref<unsigned long long, sycl::memory_order::relaxed, sycl::memory_scope::device> a(stats[0]), b(stats[1]);
        a.fetch_add(1ull);
        if (!in_vram[e]) b.fetch_add(1ull);
    });
}

void swiglu(sycl::queue& q, const half* g, const half* u, half* a, size_t n) {
    q.parallel_for(range<1>(n), [=](sycl::id<1> i) { a[i] = half(silu(f(g[i])) * f(u[i])); });
}

void moe_combine(sycl::queue& q, const half* moe, const float* sh, const half* rl, int stride, float* out, int rows) {
    q.parallel_for(range<1>((size_t)rows * D), [=](sycl::id<1> i) {
        size_t r = i / D;
        out[i] = f(moe[i]) + sigm(f(rl[r * stride + E])) * sh[i];
    });
}

// ------------------------------------------------------------------------------------------------ PLE

void ple_gate(sycl::queue& q, const float* kv, const float* X, const float* nk, const float* nq, const float* nc,
              float* gv, half* cn, int rows) {
    constexpr int KV = SD + D;
    q.parallel_for(nd_range<1>((size_t)rows * 256, 256), [=](nd_item<1> it) {
        int r = it.get_group(0), l = it.get_local_id(0);
        auto grp = it.get_group();
        const float* key = kv + (size_t)r * KV;
        const float* val = key + SD;
        const float* x = X + (size_t)r * SD;
        float sk[S], sx[S];
        for (int s = 0; s < S; s++) {
            float a = 0, b = 0;
            for (int d = l; d < D; d += 256) { a += key[s * D + d] * key[s * D + d]; b += x[s * D + d] * x[s * D + d]; }
            sk[s] = sycl::rsqrt(sycl::reduce_over_group(grp, a, sycl::plus<float>()) / D + EPS);
            sx[s] = sycl::rsqrt(sycl::reduce_over_group(grp, b, sycl::plus<float>()) / D + EPS);
        }
        float sg[S];
        for (int s = 0; s < S; s++) {
            float a = 0;
            for (int d = l; d < D; d += 256) {
                int c = s * D + d;
                a += (key[c] * sk[s] * nk[c]) * (x[c] * sx[s] * nq[c]);
            }
            float g = sycl::reduce_over_group(grp, a, sycl::plus<float>()) / sycl::sqrt((float)D);
            float sgn = g > 0 ? 1.0f : (g < 0 ? -1.0f : 0.0f);
            g = sgn * sycl::sqrt(sycl::fmax(sycl::fabs(g), 1e-6f));
            sg[s] = sigm(g);
        }
        float* gr = gv + (size_t)r * SD;
        for (int s = 0; s < S; s++) {
            float a = 0;
            for (int d = l; d < D; d += 256) {
                float v = sg[s] * val[d];
                gr[s * D + d] = v;
                a += v * v;
            }
            float rs = sycl::rsqrt(sycl::reduce_over_group(grp, a, sycl::plus<float>()) / D + EPS);
            for (int d = l; d < D; d += 256) cn[(size_t)r * SD + s * D + d] = half(gr[s * D + d] * rs * nc[s * D + d]);
        }
    });
}

void ple_conv(sycl::queue& q, const half* cn, const float* gv, const float* w, half* state, float* X, int rows) {
    constexpr int ST = 9;
    q.parallel_for(range<1>(SD), [=](sycl::id<1> ci) {
        int c = ci;
        half* sc = state + (size_t)c * ST;
        float wk[4];
        for (int j = 0; j < 4; j++) wk[j] = w[c * 4 + j];
        float win[ST + 1];
        for (int k = 0; k < ST; k++) win[k] = f(sc[k]);
        for (int t = 0; t < rows; t++) {
            win[ST] = f(cn[(size_t)t * SD + c]);
            float acc = wk[0] * win[0] + wk[1] * win[3] + wk[2] * win[6] + wk[3] * win[9];
            X[(size_t)t * SD + c] += gv[(size_t)t * SD + c] + silu(acc);
            for (int k = 0; k < ST; k++) win[k] = win[k + 1];
        }
        for (int k = 0; k < ST; k++) sc[k] = half(win[k]);
    });
}

}  // namespace glm::q4ops
