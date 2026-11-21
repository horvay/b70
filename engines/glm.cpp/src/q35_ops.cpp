#include "q35_ops.h"
#include <cmath>

namespace glm::q35ops {

using namespace q35;
using sycl::nd_item;
using sycl::nd_range;
using sycl::range;

namespace {
template <class T> inline float f(T v) { return static_cast<float>(v); }
inline float sigm(float x) { return 1.0f / (1.0f + sycl::exp(-x)); }
inline float silu(float x) { return x / (1.0f + sycl::exp(-x)); }
constexpr float NEG_INF = -INFINITY;

// NeoX rotate-half over the first 64 dims of a head in local memory: pair j rotates by the position of its axis in the
// interleaved mRoPE layout (j % 3 == 1: H, j % 3 == 2 below 30: W, the rest T), theta 1e7
inline float rope_at(const float* buf, int l, const int* pid) {
    if (l >= ROT) return buf[l];
    int j = l & (ROT / 2 - 1);
    int axis = j % 3 == 1 ? 1 : (j % 3 == 2 && j < 30 ? 2 : 0);
    float inv = sycl::pow(1.0e7f, -(float)(2 * j) / (float)ROT);
    float th = (float)pid[axis] * inv, c = sycl::cos(th), s = sycl::sin(th);
    return l < ROT / 2 ? buf[l] * c - buf[l + ROT / 2] * s : buf[l] * c + buf[l - ROT / 2] * s;
}
}  // namespace

void norm16(sycl::queue& q, const float* x, const float* w1p, half* y, int rows) {
    q.parallel_for(nd_range<1>((size_t)rows * 256, 256), [=](nd_item<1> it) {
        size_t r = it.get_group(0);
        int l = it.get_local_id(0);
        const float* xr = x + r * D;
        half* o = y + r * D;
        float ss = 0;
        for (int d = l; d < D; d += 256) ss += xr[d] * xr[d];
        ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
        float rs = sycl::rsqrt(ss / D + EPS);
        for (int d = l; d < D; d += 256) o[d] = half(xr[d] * rs * w1p[d]);
    });
}

void add_norm16(sycl::queue& q, float* x, const float* o, const float* w1p, half* y, int rows) {
    // add() then norm16(): the norm reads the values the add wrote, the same sums in the same order
    q.parallel_for(nd_range<1>((size_t)rows * 256, 256), [=](nd_item<1> it) {
        size_t r = it.get_group(0);
        int l = it.get_local_id(0);
        float* xr = x + r * D;
        const float* orow = o + r * D;
        half* yo = y + r * D;
        float ss = 0;
        for (int d = l; d < D; d += 256) {
            const float v = xr[d] + orow[d];
            xr[d] = v;
            ss += v * v;
        }
        ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
        float rs = sycl::rsqrt(ss / D + EPS);
        for (int d = l; d < D; d += 256) yo[d] = half(xr[d] * rs * w1p[d]);
    });
}

void add(sycl::queue& q, float* x, const float* o, int rows) {
    q.parallel_for(range<1>((size_t)rows * D), [=](sycl::id<1> i) { x[i] += o[i]; });
}

void gdn_norm(sycl::queue& q, const bf16* core, const float* z, const float* w, half* out, int rows) {
    constexpr int GV = q4::GV, GDV = q4::GDV;
    q.parallel_for(nd_range<1>((size_t)rows * GV * GDV, GDV), [=](nd_item<1> it) {
        size_t base = it.get_group(0) * GDV;
        int d = it.get_local_id(0);
        float h = f(core[base + d]);
        float ss = sycl::reduce_over_group(it.get_group(), h * h, sycl::plus<float>());
        h = h * sycl::rsqrt(ss / GDV + EPS) * w[d];
        out[base + d] = half(h * silu(z[base + d]));
    });
}

void gdn_conv_rows(sycl::queue& q, const float* qkv, bf16* state, const bf16* w, bf16* out, int rows, const int* sdev) {
    constexpr int GQ = q4::GQKV, CK = q4::GCONV, ST = q4::GCONV - 1;
    q.parallel_for(range<2>(rows, GQ), [=](sycl::item<2> it) {
        int r = it[0], c = it[1];
        bf16* sc = state + ((size_t)sdev[r] * GQ + c) * ST;
        float win[CK];
        for (int k = 0; k < ST; k++) win[k] = f(sc[k]);
        win[ST] = f(bf16(qkv[(size_t)r * GQ + c]));
        float acc = 0;
        for (int k = 0; k < CK; k++) acc += f(w[c * CK + k]) * win[k];
        out[(size_t)r * GQ + c] = bf16(silu(acc));
        for (int k = 0; k < ST; k++) sc[k] = bf16(win[k + 1]);
    });
}

void gdn_delta_rows(sycl::queue& q, const bf16* c, const float* beta, const float* g, float* rec, bf16* core, int rows,
                    const int* sdev) {
    // the delta rule of q4ops::gdn_delta, one token a row: one work-group per (row, v-head j, 32 v columns), thread i
    // owns state row k = i
    constexpr int GK = q4::GK, GV = q4::GV, GQ = q4::GQKV, DK = q4::GDK, DV = q4::GDV, SG = 16, NSG = DK / SG, VB = 32;
    const float scale = 1.0f / sycl::sqrt((float)DK);
    q.submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 2> part(sycl::range<2>(NSG, VB), hd);
        sycl::local_accessor<float, 1> red(sycl::range<1>(2 * NSG), hd);
        hd.parallel_for(nd_range<1>((size_t)rows * GV * (DV / VB) * DK, DK), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            int gid = it.get_group(0), i = it.get_local_id(0);
            int vb = gid % (DV / VB), j = (gid / (DV / VB)) % GV, t = gid / ((DV / VB) * GV), kh = j / (GV / GK);
            auto grp = it.get_group();
            auto sg = it.get_sub_group();
            int sgi = sg.get_group_id()[0], lane = sg.get_local_id()[0];
            float* Sp = rec + (((size_t)sdev[t] * GV + j) * DK) * DV;
            float Sr[VB];
            for (int v = 0; v < VB; v++) Sr[v] = Sp[(size_t)i * DV + vb * VB + v];
            const bf16* row = c + (size_t)t * GQ;
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
            for (int v = 0; v < VB; v++) Sp[(size_t)i * DV + vb * VB + v] = Sr[v];
        });
    });
}

namespace {
// a channel's conv over a group's rows (gdn_conv_seq; the replay of every layer at once runs the same code)
inline void conv_seq_one(const float* qkv, bf16* state, const bf16* w, bf16* out, int r, int c, const int* desc) {
    constexpr int GQ = q4::GQKV, CK = q4::GCONV, ST = q4::GCONV - 1;
    int n = desc[r * 3];
    if (n == 0) return;
    const bf16* ss = state + ((size_t)desc[r * 3 + 1] * GQ + c) * ST;
    bf16* sd = state + ((size_t)desc[r * 3 + 2] * GQ + c) * ST;
    float wk[CK], win[CK];
    for (int k = 0; k < CK; k++) wk[k] = f(w[c * CK + k]);
    for (int k = 0; k < ST; k++) win[k] = f(ss[k]);
    for (int t = 0; t < n; t++) {
        win[ST] = f(bf16(qkv[(size_t)(r + t) * GQ + c]));
        float acc = 0;
        for (int k = 0; k < CK; k++) acc += wk[k] * win[k];
        out[(size_t)(r + t) * GQ + c] = bf16(silu(acc));
        for (int k = 0; k < ST; k++) win[k] = win[k + 1];
    }
    for (int k = 0; k < ST; k++) sd[k] = bf16(win[k]);
}

// a work-group's (v-head j, 32 value columns vb) delta rule over a group's rows, the state in registers (gdn_delta_seq;
// the replay of every layer at once runs the same code). core null: outputs not written.
template <class PartAcc, class RedAcc>
inline void delta_seq_one(nd_item<1> it, const PartAcc& part, const RedAcc& red, const bf16* c, const float* beta,
                          const float* g, float* rec, bf16* core, const int* desc, int r, int j, int vb) {
    constexpr int GK = q4::GK, GV = q4::GV, GQ = q4::GQKV, DK = q4::GDK, DV = q4::GDV, SG = 16, NSG = DK / SG, VB = 32;
    const float scale = 1.0f / sycl::sqrt((float)DK);
    const int i = it.get_local_id(0), kh = j / (GV / GK);
    const int n = desc[r * 3];
    if (n == 0) return;
    auto grp = it.get_group();
    auto sg = it.get_sub_group();
    int sgi = sg.get_group_id()[0], lane = sg.get_local_id()[0];
    const float* Ss = rec + (((size_t)desc[r * 3 + 1] * GV + j) * DK) * DV;
    float* Sd = rec + (((size_t)desc[r * 3 + 2] * GV + j) * DK) * DV;
    float Sr[VB];
    for (int v = 0; v < VB; v++) Sr[v] = Ss[(size_t)i * DV + vb * VB + v];
    for (int tt = 0; tt < n; tt++) {
        const int t = r + tt;
        const bf16* row = c + (size_t)t * GQ;
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
        if (core && i < VB) {
            float o = 0;
            for (int s = 0; s < NSG; s++) o += part[s][i];
            core[((size_t)t * GV + j) * DV + vb * VB + i] = bf16(o);
        }
        // the next row's red / part writes follow its first barrier, after every read here
    }
    for (int v = 0; v < VB; v++) Sd[(size_t)i * DV + vb * VB + v] = Sr[v];
}
// The same rule with a lane per value column: a work-group (v-head j, 32 value columns vb) of 128 threads, thread t
// owns column vb * 32 + t % 32 at key rows 32 (t / 32) .. + 31, so state loads run along the columns and the sums over k
// are a thread's 32 rows then the 4 quarters in order (through SLM). Its sums run in another order than delta_seq_one's;
// the step and the replay both take this one (Q35_DELTA), so a replay still gives the step's state bit for bit.
template <class NrmAcc, class PartAcc>
inline void delta_lane_one(nd_item<1> it, const NrmAcc& nrm, const PartAcc& prt, const bf16* c, const float* beta,
                           const float* g, float* rec, bf16* core, const int* desc, int r, int j, int vb) {
    constexpr int GK = q4::GK, GV = q4::GV, GQ = q4::GQKV, DK = q4::GDK, DV = q4::GDV, VB = 32, KR = 32;
    const float scale = 1.0f / sycl::sqrt((float)DK);
    const int t = it.get_local_id(0), v = t % VB, kq = t / VB, kh = j / (GV / GK);
    const int n = desc[r * 3];
    if (n == 0) return;
    auto grp = it.get_group();
    const float* Ss = rec + (((size_t)desc[r * 3 + 1] * GV + j) * DK + kq * KR) * DV + vb * VB + v;
    float* Sd = rec + (((size_t)desc[r * 3 + 2] * GV + j) * DK + kq * KR) * DV + vb * VB + v;
    float S[KR];
    for (int i = 0; i < KR; i++) S[i] = Ss[(size_t)i * DV];
    for (int tt = 0; tt < n; tt++) {
        const int row_i = r + tt;
        const bf16* row = c + (size_t)row_i * GQ;
        float qk[KR], kk[KR];
        float qs = 0, ks = 0;
        for (int i = 0; i < KR; i++) {
            qk[i] = f(row[kh * DK + kq * KR + i]);
            kk[i] = f(row[GK * DK + kh * DK + kq * KR + i]);
            qs += qk[i] * qk[i];
            ks += kk[i] * kk[i];
        }
        if (v == 0) { nrm[kq] = qs; nrm[4 + kq] = ks; }
        const float a = sycl::exp(g[row_i * GV + j]);
        const float bt = beta[row_i * GV + j];
        sycl::group_barrier(grp);
        const float qn = sycl::rsqrt(((nrm[0] + nrm[1]) + (nrm[2] + nrm[3])) + 1e-6f) * scale;
        const float kn = sycl::rsqrt(((nrm[4] + nrm[5]) + (nrm[6] + nrm[7])) + 1e-6f);
        float sp = 0;
        for (int i = 0; i < KR; i++) {
            S[i] *= a;
            sp += S[i] * (kk[i] * kn);
        }
        prt[kq][v] = sp;
        sycl::group_barrier(grp);
        const float sk = (prt[0][v] + prt[1][v]) + (prt[2][v] + prt[3][v]);
        const float vv = f(row[2 * GK * DK + j * DV + vb * VB + v]);
        float op = 0;
        for (int i = 0; i < KR; i++) {
            S[i] += bt * (kk[i] * kn) * (vv - sk);
            op += S[i] * (qk[i] * qn);
        }
        prt[4 + kq][v] = op;
        sycl::group_barrier(grp);
        if (core && kq == 0)
            core[((size_t)row_i * GV + j) * DV + vb * VB + v] = bf16((prt[4][v] + prt[5][v]) + (prt[6][v] + prt[7][v]));
        // the next row writes nrm after this row's last read of it (two barriers back), prt likewise
    }
    for (int i = 0; i < KR; i++) Sd[(size_t)i * DV] = S[i];
}
}  // namespace

int gdn_delta_variant = 1;

void gdn_conv_seq(sycl::queue& q, const float* qkv, bf16* state, const bf16* w, bf16* out, int rows, const int* desc) {
    q.parallel_for(range<2>(rows, q4::GQKV), [=](sycl::item<2> it) { conv_seq_one(qkv, state, w, out, it[0], it[1], desc); });
}

void gdn_delta_seq(sycl::queue& q, const bf16* c, const float* beta, const float* g, float* rec, bf16* core, int rows,
                   const int* desc) {
    // gdn_delta_rows' work-groups, the first row of a group walking its rows with the state in registers
    constexpr int GV = q4::GV, DK = q4::GDK, DV = q4::GDV, SG = 16, NSG = DK / SG, VB = 32;
    if (gdn_delta_variant == 1) {
        q.submit([&](sycl::handler& hd) {
            sycl::local_accessor<float, 1> nrm(sycl::range<1>(8), hd);
            sycl::local_accessor<float, 2> prt(sycl::range<2>(8, VB), hd);
            hd.parallel_for(nd_range<1>((size_t)rows * GV * (DV / VB) * 128, 128), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                int gid = it.get_group(0);
                int vb = gid % (DV / VB), j = (gid / (DV / VB)) % GV, r = gid / ((DV / VB) * GV);
                delta_lane_one(it, nrm, prt, c, beta, g, rec, core, desc, r, j, vb);
            });
        });
        return;
    }
    q.submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 2> part(sycl::range<2>(NSG, VB), hd);
        sycl::local_accessor<float, 1> red(sycl::range<1>(2 * NSG), hd);
        hd.parallel_for(nd_range<1>((size_t)rows * GV * (DV / VB) * DK, DK), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            int gid = it.get_group(0);
            int vb = gid % (DV / VB), j = (gid / (DV / VB)) % GV, r = gid / ((DV / VB) * GV);
            delta_seq_one(it, part, red, c, beta, g, rec, core, desc, r, j, vb);
        });
    });
}

void gdn_replay(sycl::queue& q, const GdnReplayLayer* layers, int nl, bf16* cv, int rows, const int* desc) {
    // every GDN layer's conv in one launch, then every layer's delta rule in one: the code of gdn_conv_seq and
    // gdn_delta_seq, so the state comes out bit for bit as the per-layer launches leave it
    constexpr int GQ = q4::GQKV, GV = q4::GV, DK = q4::GDK, DV = q4::GDV, SG = 16, NSG = DK / SG, VB = 32;
    q.parallel_for(range<3>(nl, rows, GQ), [=](sycl::item<3> it) {
        const GdnReplayLayer L = layers[it[0]];
        conv_seq_one(L.kq, L.conv, L.conv_w, cv + (size_t)it[0] * rows * GQ, it[1], it[2], desc);
    });
    if (gdn_delta_variant == 1) {
        q.submit([&](sycl::handler& hd) {
            sycl::local_accessor<float, 1> nrm(sycl::range<1>(8), hd);
            sycl::local_accessor<float, 2> prt(sycl::range<2>(8, VB), hd);
            const size_t per = (size_t)rows * GV * (DV / VB);
            hd.parallel_for(nd_range<1>((size_t)nl * per * 128, 128), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                const size_t gl = it.get_group(0);
                const int li = gl / per, gid = gl % per;
                int vb = gid % (DV / VB), j = (gid / (DV / VB)) % GV, r = gid / ((DV / VB) * GV);
                const GdnReplayLayer L = layers[li];
                delta_lane_one(it, nrm, prt, cv + (size_t)li * rows * GQ, L.kb, L.kg, L.rec, (bf16*)nullptr, desc, r, j, vb);
            });
        });
        return;
    }
    q.submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 2> part(sycl::range<2>(NSG, VB), hd);
        sycl::local_accessor<float, 1> red(sycl::range<1>(2 * NSG), hd);
        const size_t per = (size_t)rows * GV * (DV / VB);
        hd.parallel_for(nd_range<1>((size_t)nl * per * DK, DK), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const size_t gl = it.get_group(0);
            const int li = gl / per, gid = gl % per;
            int vb = gid % (DV / VB), j = (gid / (DV / VB)) % GV, r = gid / ((DV / VB) * GV);
            const GdnReplayLayer L = layers[li];
            delta_seq_one(it, part, red, cv + (size_t)li * rows * GQ, L.kb, L.kg, L.rec, (bf16*)nullptr, desc, r, j, vb);
        });
    });
}

template <class T> void norm_rows(sycl::queue& q, const T* x, const float* w1p, half* y, int ys, int rows) {
    q.parallel_for(nd_range<1>((size_t)rows * 256, 256), [=](nd_item<1> it) {
        size_t r = it.get_group(0);
        int l = it.get_local_id(0);
        const T* xr = x + r * D;
        half* o = y + r * ys;
        float ss = 0;
        for (int d = l; d < D; d += 256) ss += f(xr[d]) * f(xr[d]);
        ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
        float rs = sycl::rsqrt(ss / D + EPS);
        for (int d = l; d < D; d += 256) o[d] = half(f(xr[d]) * rs * w1p[d]);
    });
}
template void norm_rows<float>(sycl::queue&, const float*, const float*, half*, int, int);
template void norm_rows<half>(sycl::queue&, const half*, const float*, half*, int, int);

void put_rows(sycl::queue& q, const half* src, int rows, half* ring, const int* pdev, const int* sdev, int RING) {
    q.parallel_for(range<1>((size_t)rows * D), [=](sycl::id<1> i) {
        int r = i / D, d = i % D;
        ring[((size_t)sdev[r] * RING + pdev[r] % RING) * D + d] = src[i];
    });
}

void gather_hidden(sycl::queue& q, const int* src, int rows, const half* ring, const half* mh, half* out) {
    q.parallel_for(range<1>((size_t)rows * D), [=](sycl::id<1> i) {
        int r = i / D, d = i % D, s = src[r];
        out[i] = s < 0 ? half(0.0f) : s < MH ? ring[(size_t)s * D + d] : mh[(size_t)(s - MH) * D + d];
    });
}

void gather_col_tiles(sycl::queue& q, const uint16_t* src, uint16_t* dst, int rows_t, int src_tn, int dst_tn, int words,
                      const int* map) {
    q.parallel_for(range<3>(rows_t, dst_tn, words), [=](sycl::item<3> it) {
        const size_t i = it[0], j = it[1], w = it[2];
        dst[(i * dst_tn + j) * words + w] = src[(i * src_tn + map[j]) * words + w];
    });
}

void argmax_rows(sycl::queue& q, const half* logits, int rows, int n, int* out) {
    q.submit([&](sycl::handler& hd) {
        hd.parallel_for(nd_range<1>((size_t)rows * 1024, 1024), [=](nd_item<1> it) {
            auto grp = it.get_group();
            int r = it.get_group(0), l = it.get_local_id(0);
            const half* lg = logits + (size_t)r * n;
            float bv = NEG_INF;
            int bi = n;
            for (int i = l; i < n; i += 1024) {
                float v = f(lg[i]);
                if (v > bv) { bv = v; bi = i; }
            }
            float m = sycl::reduce_over_group(grp, bv, sycl::maximum<float>());
            int idx = sycl::reduce_over_group(grp, bv == m ? bi : n, sycl::minimum<int>());
            float s = 0;
            for (int i = l; i < n; i += 1024) s += sycl::exp(f(lg[i]) - m);
            s = sycl::reduce_over_group(grp, s, sycl::plus<float>());
            if (l == 0) {
                out[2 * r] = idx;
                out[2 * r + 1] = sycl::bit_cast<int>(1.0f / s);
            }
        });
    });
}

void attn_prep(sycl::queue& q, const half* qg, const half* k, const half* v, const float* qn, const float* kn, int rows,
               int p0, const int* pid, half* qn16, half* Kc, half* Vc, const int* pdev, const int* sdev, int TS) {
    constexpr int U = AH + 2 * AKV;     // per row: 24 q heads, 4 k, 4 v
    q.submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> buf(sycl::range<1>(AD), hd);
        hd.parallel_for(nd_range<1>((size_t)rows * U * AD, AD), [=](nd_item<1> it) {
            int gid = it.get_group(0), l = it.get_local_id(0);
            int r = gid / U, u = gid % U;
            size_t p = pdev ? (size_t)sdev[r] * TS + pdev[r] : (size_t)p0 + r;
            auto grp = it.get_group();
            if (u < AH + AKV) {
                bool isq = u < AH;
                int h = isq ? u : u - AH;
                float x = isq ? f(qg[(size_t)r * AH * 2 * AD + h * 2 * AD + l]) : f(k[(size_t)r * AKV * AD + h * AD + l]);
                float ss = sycl::reduce_over_group(grp, x * x, sycl::plus<float>());
                buf[l] = x * sycl::rsqrt(ss / AD + EPS) * (isq ? qn : kn)[l];
                sycl::group_barrier(grp);
                float o = rope_at(&buf[0], l, pid + (size_t)r * 3);
                if (isq) qn16[((size_t)r * AH + h) * AD + l] = half(o);
                else Kc[((size_t)p * AKV + h) * AD + l] = half(o);
            } else {
                int h = u - AH - AKV;
                Vc[((size_t)p * AKV + h) * AD + l] = v[(size_t)r * AKV * AD + h * AD + l];
            }
        });
    });
}

namespace {
constexpr int HG = AH / AKV;                  // heads per kv head
constexpr int PSZ = 2 * HG + HG * AD;         // a piece's m[6], l[6], o[6][256]
}  // namespace

size_t attn_part_floats(int rows, int nch) { return (size_t)rows * AKV * nch * PSZ; }

namespace {
// attn_core's decode kernel for rows grouped by sequence (below)
template <int GR>
void attn_dec_grouped(sycl::queue& q, const half* qn16, const half* Kc, const half* Vc, int rows, int nch, float* part,
                      const int* pdev, const int* sdev, int TS) {
        q.submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> qs(sycl::range<1>(GR * HG * AD), hd);
        sycl::local_accessor<float, 1> ps(sycl::range<1>(GR * HG * CH), hd);
        hd.parallel_for(nd_range<1>((size_t)rows * AKV * nch * 256, 256), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            int gid = it.get_group(0), l = it.get_local_id(0);
            const int r0 = gid % rows, g = (gid / rows) % AKV, c = gid / (rows * AKV);
            const int s = sdev[r0];
            int k = 0;                                   // r0's place among its sequence's rows
            while (k < r0 && sdev[r0 - k - 1] == s) k++;
            if (k % GR) return;                          // a leader (every GR-th row) reads for its followers
            int n = 1;
            while (n < GR && r0 + n < rows && sdev[r0 + n] == s) n++;
            const size_t kbase = (size_t)s * TS;
            auto grp = it.get_group();
            const int t0 = c * CH;
            int ntr[GR], ntmax = 0;
            for (int rr = 0; rr < GR; rr++) {
                ntr[rr] = rr < n ? sycl::clamp(pdev[r0 + rr] + 1 - t0, 0, CH) : 0;
                ntmax = sycl::max(ntmax, ntr[rr]);
            }
            for (int rr = 0; rr < n; rr++)
                if (ntr[rr] == 0 && l < HG) {
                    float* P = part + ((size_t)((r0 + rr) * AKV + g) * nch + c) * PSZ;
                    P[l] = NEG_INF;
                    P[HG + l] = 0.0f;
                }
            if (ntmax == 0) return;
            for (int i = l; i < n * HG * AD; i += 256) {
                int rr = i / (HG * AD), w = i % (HG * AD);
                qs[i] = f(qn16[((size_t)(r0 + rr) * AH + g * HG) * AD + w]);
            }
            sycl::group_barrier(grp);
            {
                int j = l & (CH - 1), hh = l / CH;
                float acc[GR][HG / 2] = {};
                if (j < ntmax) {
                    const half* kp = Kc + ((kbase + t0 + j) * AKV + g) * AD;
                    for (int d = 0; d < AD; d += 8) {
                        sycl::vec<half, 8> kv = *reinterpret_cast<const sycl::vec<half, 8>*>(kp + d);
                        for (int e = 0; e < 8; e++) {
                            float kf = f(kv[e]);
                            for (int rr = 0; rr < GR; rr++) {
                                if (rr >= n) break;
                                for (int h = 0; h < HG / 2; h++)
                                    acc[rr][h] += qs[(rr * HG + hh * (HG / 2) + h) * AD + d + e] * kf;
                            }
                        }
                    }
                }
                for (int rr = 0; rr < n; rr++)
                    for (int h = 0; h < HG / 2; h++)
                        ps[(rr * HG + hh * (HG / 2) + h) * CH + j] = j < ntr[rr] ? acc[rr][h] * 0.0625f : NEG_INF;
            }
            sycl::group_barrier(grp);
            {
                auto sg = it.get_sub_group();
                int sgi = sg.get_group_id()[0], lane = sg.get_local_id()[0];
                for (int u = sgi; u < n * HG; u += 16) {
                    const int rr = u / HG, h = u % HG, nt = ntr[rr];
                    float* pr = &ps[u * CH];
                    float m = NEG_INF;
                    for (int j = lane; j < CH; j += 16) m = sycl::fmax(m, pr[j]);
                    m = sycl::reduce_over_group(sg, m, sycl::maximum<float>());
                    float sum = 0;
                    for (int j = lane; j < CH; j += 16) {
                        float e = j < nt ? sycl::exp(pr[j] - m) : 0.0f;
                        pr[j] = e;
                        sum += e;
                    }
                    sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>());
                    if (lane == 0 && nt > 0) {
                        float* P = part + ((size_t)((r0 + rr) * AKV + g) * nch + c) * PSZ;
                        P[h] = m;
                        P[HG + h] = sum;
                    }
                }
            }
            sycl::group_barrier(grp);
            {
                float acc[GR][HG] = {};
                for (int j = 0; j < ntmax; j++) {
                    float vv = f(Vc[((kbase + t0 + j) * AKV + g) * AD + l]);
                    for (int rr = 0; rr < GR; rr++) {
                        if (rr >= n) break;
                        if (j < ntr[rr])
                            for (int h = 0; h < HG; h++) acc[rr][h] += ps[(rr * HG + h) * CH + j] * vv;
                    }
                }
                for (int rr = 0; rr < n; rr++) {
                    if (ntr[rr] == 0) continue;
                    float* P = part + ((size_t)((r0 + rr) * AKV + g) * nch + c) * PSZ;
                    for (int h = 0; h < HG; h++) P[2 * HG + h * AD + l] = acc[rr][h];
                }
            }
        });
    });
}
}  // namespace

void attn_core(sycl::queue& q, const half* qn16, const half* Kc, const half* Vc, const half* qg, int rows, int p0, int nch,
               float* part, half* o16, const int* pdev, const int* sdev, int TS) {
    static const bool inner_env = !(getenv("Q35_ATTN_ROWS_INNER") && atoi(getenv("Q35_ATTN_ROWS_INNER")) == 0);
    static const bool group_env = !(getenv("Q35_ATTN_GROUP") && atoi(getenv("Q35_ATTN_GROUP")) == 0);
    const bool inner = inner_env;
    if (pdev && group_env && rows > 1) {
        // decode rows: up to 4 rows of one sequence (a verify step) in one work-group, which reads each key and value
        // once for all of them. Each row's sums run in the order of the one-row kernel below, so its results are
        // the same bit for bit. (One row keeps that kernel: 4 rows' local memory costs it occupancy.)
        if (rows == 2) attn_dec_grouped<2>(q, qn16, Kc, Vc, rows, nch, part, pdev, sdev, TS);
        else if (rows == 3) attn_dec_grouped<3>(q, qn16, Kc, Vc, rows, nch, part, pdev, sdev, TS);
        else attn_dec_grouped<4>(q, qn16, Kc, Vc, rows, nch, part, pdev, sdev, TS);
    } else
    q.submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> qs(sycl::range<1>(HG * AD), hd);
        sycl::local_accessor<float, 1> ps(sycl::range<1>(HG * CH), hd);
        hd.parallel_for(nd_range<1>((size_t)rows * AKV * nch * 256, 256), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            int gid = it.get_group(0), l = it.get_local_id(0);
            // rows innermost: the rows of a verify step read each key piece back to back (from L2 after the first)
            int c, g, r;
            if (inner) { r = gid % rows; g = (gid / rows) % AKV; c = gid / (rows * AKV); }
            else { c = gid % nch; g = (gid / nch) % AKV; r = gid / (nch * AKV); }
            const int p = pdev ? pdev[r] : p0 + r;
            const size_t kbase = pdev ? (size_t)sdev[r] * TS : 0;
            auto grp = it.get_group();
            int n = p + 1, t0 = c * CH;
            float* P = part + ((size_t)(r * AKV + g) * nch + c) * PSZ;
            if (t0 >= n) {
                if (l < HG) { P[l] = NEG_INF; P[HG + l] = 0.0f; }
                return;
            }
            int nt = sycl::min(CH, n - t0);
            for (int i = l; i < HG * AD; i += 256) qs[i] = f(qn16[((size_t)r * AH + g * HG) * AD + i]);
            sycl::group_barrier(grp);
            {
                int j = l & (CH - 1), hh = l / CH;          // key slot, half of the heads (3 each)
                float acc[HG / 2] = {};
                if (j < nt) {
                    const half* kp = Kc + ((kbase + t0 + j) * AKV + g) * AD;
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
                    float vv = f(Vc[((kbase + t0 + j) * AKV + g) * AD + l]);
                    for (int h = 0; h < HG; h++) acc[h] += ps[h * CH + j] * vv;
                }
                for (int h = 0; h < HG; h++) P[2 * HG + h * AD + l] = acc[h];
            }
        });
    });
    q.parallel_for(range<1>((size_t)rows * AH * AD), [=](sycl::id<1> i) {
        int r = i / (AH * AD), h = (i / AD) % AH, d = i % AD, g = h / HG, hh = h % HG;
        int p = pdev ? pdev[r] : p0 + r, used = sycl::min(nch, p / CH + 1);
        const float* Pb = part + (size_t)(r * AKV + g) * nch * PSZ;
        float M = NEG_INF;
        for (int c = 0; c < used; c++) M = sycl::fmax(M, Pb[c * PSZ + hh]);
        float num = 0, den = 0;
        for (int c = 0; c < used; c++) {
            float e = sycl::exp(Pb[c * PSZ + hh] - M);
            den += Pb[c * PSZ + HG + hh] * e;
            num += Pb[c * PSZ + 2 * HG + hh * AD + d] * e;
        }
        float gate = f(qg[(size_t)r * AH * 2 * AD + h * 2 * AD + AD + d]);
        o16[(size_t)r * AH * AD + h * AD + d] = half(num / den * sigm(gate));
    });
}

size_t dec_part_floats(int rows, int ns) { return (size_t)rows * AH * ns * (AD + 2); }

void attn_dec_combine(sycl::queue& q, const float* part, int rows, int ns, const half* qg, half* o16) {
    // o = sum_s 2^(m_s - M) o_s / sum_s 2^(m_s - M) l_s per row and head (m in base 2), pieces in order; empty pieces
    // (m = -inf) are skipped, so their unwritten o is never read
    const size_t nm = (size_t)rows * AH * ns;
    const float* pm = part;
    const float* pl = part + nm;
    const float* po = part + 2 * nm;
    constexpr int NW = 64;
    q.submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> es(sycl::range<1>(ns), hd);
        hd.parallel_for(nd_range<1>((size_t)rows * AH * NW, NW), [=](nd_item<1> it) {
            const size_t rh = it.get_group(0);
            const int t = it.get_local_id(0);
            const float* mp = pm + rh * ns;
            float M = NEG_INF;
            for (int s = 0; s < ns; s++) M = sycl::fmax(M, mp[s]);
            for (int s = t; s < ns; s += NW) es[s] = mp[s] == NEG_INF ? 0.0f : sycl::exp2(mp[s] - M);
            sycl::group_barrier(it.get_group());
            float den = 0;
            for (int s = 0; s < ns; s++) den += es[s] * pl[rh * ns + s];
            const float inv = 1.0f / den;
            const int r = rh / AH, h = rh % AH;
            const float* os = po + rh * ns * AD;
            for (int d = t; d < AD; d += NW) {
                float acc = 0;
                for (int s = 0; s < ns; s++)
                    if (es[s] != 0.0f) acc += es[s] * os[(size_t)s * AD + d];
                const float gate = f(qg[(size_t)r * AH * 2 * AD + h * 2 * AD + AD + d]);
                o16[(size_t)r * AH * AD + h * AD + d] = half(acc * inv * sigm(gate));
            }
        });
    });
}

void kt_fill(sycl::queue& q, const half* Kc, half* kt, int TS, int from, int to) {
    if (to <= from) return;
    // a work-item a (key, kv head, d pair), keys fastest: the writes run along a row of kt
    const int n = to - from;
    const uint32_t* src = reinterpret_cast<const uint32_t*>(Kc);
    uint32_t* dst = reinterpret_cast<uint32_t*>(kt);
    q.parallel_for(range<2>((size_t)AKV * (AD / 2), (size_t)n), [=](sycl::item<2> it) {
        const int hd = it[0], t = from + (int)it[1];
        const int h = hd / (AD / 2), dp = hd % (AD / 2);
        dst[(size_t)hd * TS + t] = src[((size_t)t * AKV + h) * (AD / 2) + dp];
    });
}

void vt_fill(sycl::queue& q, const half* Vc, half* vt, int TS, int from, int to) {
    from &= ~1;
    if (to <= from) return;
    // a work-item a (kv head, key pair, d), d fastest: reads along a cache row, writes along a row of vt
    const int np = (to - from + 1) / 2;
    q.parallel_for(range<3>((size_t)AKV, (size_t)np, (size_t)AD), [=](sycl::item<3> it) {
        const int h = it[0], tp = from / 2 + (int)it[1], d = it[2];
        const int t0 = 2 * tp;
        const half a = Vc[((size_t)t0 * AKV + h) * AD + d];
        const half b = t0 + 1 < to ? Vc[((size_t)(t0 + 1) * AKV + h) * AD + d] : half(0.0f);
        half* o = vt + (((size_t)h * (TS / 2) + tp) * AD + d) * 2;
        o[0] = a;
        o[1] = b;
    });
}

void attn_gate(sycl::queue& q, const half* qg, half* o, int rows) {
    q.parallel_for(range<1>((size_t)rows * AH * AD), [=](sycl::id<1> i) {
        size_t r = i / (AH * AD), h = (i / AD) % AH, d = i % AD;
        float gate = f(qg[r * AH * 2 * AD + h * 2 * AD + AD + d]);
        o[i] = half(f(o[i]) * sigm(gate));
    });
}

void gdn_conv_prompt(sycl::queue& q, const float* qkv, bf16* state, const bf16* w, bf16* out, int rows) {
    // every (row, channel) at once: the window of row t is the inputs t - 3 .. t, the ones before row 0 from the state;
    // the same sums in the same order as q4ops::gdn_conv
    constexpr int GQ = q4::GQKV, CK = q4::GCONV, ST = q4::GCONV - 1;
    q.parallel_for(range<2>(rows, GQ), [=](sycl::item<2> it) {
        const int t = it[0], c = it[1];
        const bf16* sc = state + (size_t)c * ST;
        float acc = 0;
        for (int k = 0; k < CK; k++) {
            const int src = t - ST + k;
            const float x = src >= 0 ? f(bf16(qkv[(size_t)src * GQ + c])) : f(sc[ST + src]);
            acc += f(w[c * CK + k]) * x;
        }
        out[(size_t)t * GQ + c] = bf16(silu(acc));
    });
    q.parallel_for(range<1>(GQ), [=](sycl::id<1> ci) {
        const int c = ci;
        bf16* sc = state + (size_t)c * ST;
        float win[ST];
        for (int k = 0; k < ST; k++) {
            const int src = rows - ST + k;
            win[k] = src >= 0 ? f(bf16(qkv[(size_t)src * GQ + c])) : f(sc[ST + src]);
        }
        for (int k = 0; k < ST; k++) sc[k] = bf16(win[k]);
    });
}

// ------------------------------------------------------------------------------------------------ chunked delta rule

namespace {
constexpr int GCK = 64;
size_t al256(size_t b) { return (b + 255) & ~size_t(255); }
}  // namespace

size_t gdn_chunk_bytes(int rows) {
    const size_t nc = (rows + GCK - 1) / GCK, npad = nc * GCK;
    return 2 * al256(npad * q4::GK * q4::GDK * 2) + al256(nc * q4::GV * GCK * GCK * 4) + al256(nc * q4::GV * GCK * GCK * 2) +
           3 * al256(nc * q4::GV * GCK * 4);
}

void gdn_chunked(sycl::queue& q, const bf16* c, const float* beta, const float* g, float* rec, bf16* core, int rows,
                 void* scratch) {
    constexpr int GK = q4::GK, GV = q4::GV, DK = q4::GDK, GQ = q4::GQKV;
    const int nc = (rows + GCK - 1) / GCK, npad = nc * GCK;
    char* p = static_cast<char*>(scratch);
    auto take = [&](size_t b) { char* r = p; p += al256(b); return r; };
    half* qh = reinterpret_cast<half*>(take((size_t)npad * GK * DK * 2));
    half* kh = reinterpret_cast<half*>(take((size_t)npad * GK * DK * 2));
    float* A = reinterpret_cast<float*>(take((size_t)nc * GV * GCK * GCK * 4));
    half* Mq = reinterpret_cast<half*>(take((size_t)nc * GV * GCK * GCK * 2));
    float* bt = reinterpret_cast<float*>(take((size_t)nc * GV * GCK * 4));
    float* gam = reinterpret_cast<float*>(take((size_t)nc * GV * GCK * 4));
    float* wl = reinterpret_cast<float*>(take((size_t)nc * GV * GCK * 4));
    // q, k of every k-head l2-normed (q times 1/sqrt(128)) as gdn_delta computes them, then fp16; pad rows zero
    const float scale = 1.0f / sycl::sqrt((float)DK);
    q.parallel_for(nd_range<1>((size_t)npad * 2 * GK * DK, DK), [=](nd_item<1> it) {
        const int u = it.get_group(0) % (2 * GK), r = it.get_group(0) / (2 * GK), l = it.get_local_id(0);
        const bool isq = u < GK;
        const int h = u % GK;
        float x = r < rows ? f(c[(size_t)r * GQ + (isq ? 0 : GK * DK) + h * DK + l]) : 0.0f;
        float ss = sycl::reduce_over_group(it.get_group(), x * x, sycl::plus<float>());
        x = x * sycl::rsqrt(ss + 1e-6f);
        if (isq) x *= scale;
        (isq ? qh : kh)[((size_t)r * GK + h) * DK + l] = half(x);
    });
    prof_point("GDN chunks: q k norm");
    gdn_prep(q, qh, kh, beta, g, A, Mq, bt, gam, wl, rows, npad, nc);
    prof_point("GDN chunks: prep");
    gdn_scan(q, c, qh, kh, A, Mq, bt, gam, wl, rec, core, rows, npad, nc);
    prof_point("GDN chunks: scan");
}

}  // namespace glm::q35ops
