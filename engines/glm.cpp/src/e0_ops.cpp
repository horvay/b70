#include "e0_ops.h"
#include "q4.h"
#include <oneapi/mkl/blas.hpp>
#include <cstring>
#include <type_traits>

namespace glm::e0ops {

using sycl::nd_item;
using sycl::nd_range;
using sycl::range;

namespace {
template <class T> inline float f(T v) { return static_cast<float>(v); }
inline float sigm(float x) { return 1.0f / (1.0f + sycl::exp(-x)); }
inline float silu(float x) { return x / (1.0f + sycl::exp(-x)); }

// UE4M3 as ggml decodes it (ggml_ue4m3_to_fp32 without its 0.5: the codes below are E2M1, not 2 * E2M1)
inline float ue4m3(uint32_t x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    int e = (x >> 3) & 0xF, m = x & 7;
    return e == 0 ? sycl::ldexp((float)m, -9) : sycl::ldexp(1.0f + (float)m * 0.125f, e - 7);
}
// E2M1 code (bit 3 the sign): 0, 0.5, 1, 1.5, 2, 3, 4, 6
inline float e2m1(uint32_t c) {
    uint32_t n = c & 7;
    float v = n < 2 ? 0.5f * (float)n : sycl::ldexp(1.0f + 0.5f * (float)(n & 1), (int)(n >> 1) - 1);
    return (c & 8) ? -v : v;
}
// 2 x E2M1 as an integer (0, 1, 2, 3, 4, 6, 8, 12, signed): shifts and a select, no ldexp
inline int e2m1x2(uint32_t c) {
    int n = (int)(c & 7);
    int v = n < 2 ? n : (2 + (n & 1)) << ((n >> 1) - 1);
    return (c & 8) ? -v : v;
}
inline float tail(const char* rec, int i) { return reinterpret_cast<const float*>(rec + NV_TAIL)[i]; }

// one NVFP4 sub-block of 16 (sub-block s of a row): sum of value * x, the UE4M3 scale applied
template <class TX>
inline float nv_sub(const uint8_t* row, int s, const TX* x) {
    const uint8_t* blk = row + (s >> 2) * 36;
    int sub = s & 3;
    float sc = ue4m3(blk[sub]);
    const uint8_t* qs = blk + 4 + sub * 8;
    uint32_t w0 = *reinterpret_cast<const uint32_t*>(qs), w1 = *reinterpret_cast<const uint32_t*>(qs + 4);
    const TX* xb = x + (s >> 2) * 64 + sub * 16;
    float acc = 0;
#pragma unroll
    for (int i = 0; i < 8; i++) {
        uint32_t b = ((i < 4 ? w0 : w1) >> ((i & 3) * 8)) & 0xFF;
        acc += e2m1(b & 15) * f(xb[i]) + e2m1(b >> 4) * f(xb[8 + i]);
    }
    return acc * sc;
}
}  // namespace

// ------------------------------------------------------------------------------------------------ tiled records

void nv_repack(const char* raw, char* out) {
    auto mat = [&](size_t off, int R, int C, int rowb) {
        const int NS = C / 16;
        for (int G = 0; G < R / 16; G++)
            for (int s = 0; s < NS; s++) {
                char* t = out + off + ((size_t)G * NS + s) * NV_TILE;
                for (int i = 0; i < 16; i++) {
                    const char* blk = raw + off + (size_t)(G * 16 + i) * rowb + (s >> 2) * 36;
                    t[i] = blk[s & 3];
                    memcpy(t + 16 + i * 8, blk + 4 + (s & 3) * 8, 8);
                }
            }
    };
    mat(0, 2 * EI, ELL, NV_ROW_GU);                         // gate and up: 1280 rows of 1792
    mat(NV_DOWN_OFF, ELL, EI, NV_ROW_D);                    // down: 1792 rows of 640
    memcpy(out + NV_TAIL, raw + NV_TAIL, 16);
}

void nv_repack_dev(sycl::queue& q, const char* src, const int64_t* dst, int n) {
    constexpr int GU_T = (2 * EI / 16) * (ELL / 16), DN_T = (ELL / 16) * (EI / 16);
    q.parallel_for(range<3>(n, GU_T + DN_T, 16), [=](sycl::id<3> ii) {
        const char* raw = src + ii[0] * NV_REC;
        char* out = reinterpret_cast<char*>(dst[ii[0]]);
        int ti = (int)ii[1], i = (int)ii[2];
        const char* blk;
        char* t;
        int s;
        if (ti < GU_T) {
            int G = ti / (ELL / 16);
            s = ti % (ELL / 16);
            blk = raw + (size_t)(G * 16 + i) * NV_ROW_GU + (s >> 2) * 36;
            t = out + (size_t)ti * NV_TILE;
        } else {
            int tj = ti - GU_T, G = tj / (EI / 16);
            s = tj % (EI / 16);
            blk = raw + NV_DOWN_OFF + (size_t)(G * 16 + i) * NV_ROW_D + (s >> 2) * 36;
            t = out + NV_DOWN_OFF + (size_t)tj * NV_TILE;
        }
        t[i] = blk[s & 3];
        const uint32_t* c = reinterpret_cast<const uint32_t*>(blk + 4 + (s & 3) * 8);     // 4-aligned in the file
        *reinterpret_cast<uint64_t*>(t + 16 + i * 8) = (uint64_t)c[0] | ((uint64_t)c[1] << 32);
        if (ti == 0 && i == 0)
            for (int k = 0; k < 16; k++) out[NV_TAIL + k] = raw[NV_TAIL + k];
    });
}

namespace {
// lane i's row of tile t: its 16 values as 2 x E2M1 into w, and 0.5 x its UE4M3 scale (applied to the dot product)
inline float tile_row(const uint8_t* t, int i, float* w) {
    const float lut[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    const uint32_t c0 = *reinterpret_cast<const uint32_t*>(t + 16 + i * 8);
    const uint32_t c1 = *reinterpret_cast<const uint32_t*>(t + 20 + i * 8);
#pragma unroll
    for (int k = 0; k < 8; k++) {
        uint32_t b = ((k < 4 ? c0 : c1) >> ((k & 3) * 8)) & 0xFF;
        w[k] = lut[b & 15];
        w[8 + k] = lut[b >> 4];
    }
    return 0.5f * ue4m3(t[i]);
}
}  // namespace

void nv_gate_up_t(sycl::queue& q, const float* xl, const int* sel, const int* lead, const int* label,
                  const int64_t* ptrs, int M, float* h) {
    constexpr int KS = 8, NG = EI / 16, NS = ELL / 16, PS = NS / KS, DM = 16;
    const int P = M * 10;
    q.submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> red(sycl::range<1>(KS * DM * 2 * 16), hd);
        hd.parallel_for(nd_range<2>({(size_t)P, (size_t)NG * 16 * KS}, {1, 16 * KS}),
                        [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
            const int p = it.get_global_id(0), G = it.get_group(1), t = it.get_local_id(1), k = t / 16, i = t % 16;
            if (lead && lead[p] != p) return;
            int dp[DM], nd = 0;
            if (lead) {
                for (int qq = p; qq < P && nd < DM; qq++)
                    if (lead[qq] == p) dp[nd++] = qq;
            } else {
                dp[nd++] = p;
            }
            const int e = sel[p];
            const char* rec = reinterpret_cast<const char*>(ptrs[e]);
            const uint8_t* gt = reinterpret_cast<const uint8_t*>(rec) + (size_t)G * NS * NV_TILE;
            const uint8_t* ut = gt + NV_DOWN_OFF / 2;
            const int g = label[e];
            float ag[DM], au[DM];
#pragma unroll
            for (int d = 0; d < DM; d++) { ag[d] = 0; au[d] = 0; }
            for (int s = k * PS; s < (k + 1) * PS; s++) {
                float wg[16], wu[16];
                const float scg = tile_row(gt + (size_t)s * NV_TILE, i, wg);
                const float scu = tile_row(ut + (size_t)s * NV_TILE, i, wu);
#pragma unroll
                for (int d = 0; d < DM; d++) {
                    if (d < nd) {
                        const float* x = xl + (size_t)(dp[d] / 10) * GE + (size_t)g * ELL + s * 16;
                        float a = 0, b = 0;
#pragma unroll
                        for (int c = 0; c < 16; c++) { a += wg[c] * x[c]; b += wu[c] * x[c]; }
                        ag[d] += a * scg;
                        au[d] += b * scu;
                    }
                }
            }
#pragma unroll
            for (int d = 0; d < DM; d++)
                if (d < nd) { red[((k * DM + d) * 2) * 16 + i] = ag[d]; red[((k * DM + d) * 2 + 1) * 16 + i] = au[d]; }
            sycl::group_barrier(it.get_group());
            if (k == 0) {
                const float sg = tail(rec, 0), su = tail(rec, 1);
                for (int d = 0; d < nd; d++) {
                    float G2 = 0, U2 = 0;
                    for (int kk = 0; kk < KS; kk++) {
                        G2 += red[((kk * DM + d) * 2) * 16 + i];
                        U2 += red[((kk * DM + d) * 2 + 1) * 16 + i];
                    }
                    h[(size_t)dp[d] * EI + G * 16 + i] = silu(sg * G2) * (su * U2);
                }
            }
        });
    });
}

void nv_down_t(sycl::queue& q, const float* h, const int* sel, const int* lead, const int64_t* ptrs, int M, float* yl) {
    constexpr int KS = 4, NG = ELL / 16, NS = EI / 16, PS = NS / KS, DM = 16;
    const int P = M * 10;
    q.submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> red(sycl::range<1>(KS * DM * 16), hd);
        hd.parallel_for(nd_range<2>({(size_t)P, (size_t)NG * 16 * KS}, {1, 16 * KS}),
                        [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
            const int p = it.get_global_id(0), G = it.get_group(1), t = it.get_local_id(1), k = t / 16, i = t % 16;
            if (lead && lead[p] != p) return;
            int dp[DM], nd = 0;
            if (lead) {
                for (int qq = p; qq < P && nd < DM; qq++)
                    if (lead[qq] == p) dp[nd++] = qq;
            } else {
                dp[nd++] = p;
            }
            const char* rec = reinterpret_cast<const char*>(ptrs[sel[p]]);
            const uint8_t* dt = reinterpret_cast<const uint8_t*>(rec) + NV_DOWN_OFF + (size_t)G * NS * NV_TILE;
            float a[DM];
#pragma unroll
            for (int d = 0; d < DM; d++) a[d] = 0;
            for (int s = k * PS; s < (k + 1) * PS; s++) {
                float w[16];
                const float sc = tile_row(dt + (size_t)s * NV_TILE, i, w);
#pragma unroll
                for (int d = 0; d < DM; d++) {
                    if (d < nd) {
                        const float* x = h + (size_t)dp[d] * EI + s * 16;
                        float v = 0;
#pragma unroll
                        for (int c = 0; c < 16; c++) v += w[c] * x[c];
                        a[d] += v * sc;
                    }
                }
            }
#pragma unroll
            for (int d = 0; d < DM; d++)
                if (d < nd) red[(k * DM + d) * 16 + i] = a[d];
            sycl::group_barrier(it.get_group());
            if (k == 0) {
                const float sd = tail(rec, 2);
                for (int d = 0; d < nd; d++) {
                    float v = 0;
                    for (int kk = 0; kk < KS; kk++) v += red[(kk * DM + d) * 16 + i];
                    yl[(size_t)dp[d] * ELL + G * 16 + i] = sd * v;
                }
            }
        });
    });
}

// ------------------------------------------------------------------------------------------------ Q8_0

// up to 4 input rows: a sub-group takes 4 weight rows, so each x chunk it loads serves all 4 (the x reads, not the
// weights, bounded the one-row-per-sub-group kernel at 4 input rows)
template <class TX, class TY, int MM>
void q8_matvec_r4(sycl::queue& q, const TX* x, int M, const Q8W& w, TY* y, const uint32_t* gmask = nullptr,
                  int gmode = 0) {
    const int8_t* qs = w.qs.ptr<int8_t>();
    const half* d = w.d.ptr<half>();
    const int N = w.N, K = w.K, NB = K / 32;
    constexpr int SG = 16, WG = 128, RW = 4, PER = WG / SG * RW;
    size_t groups = (N + PER - 1) / PER;
    q.parallel_for(nd_range<1>(groups * WG, WG), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        auto sg = it.get_sub_group();
        const int n0 = (it.get_group(0) * (WG / SG) + sg.get_group_id()[0]) * RW;
        const int l = sg.get_local_id()[0];
        if (n0 >= N) return;
        // gmode 1: rows in groups of ELL (D), 2: columns in groups of ELL (U); a group no pick uses is skipped
        const uint32_t gm = gmask ? *gmask : 0xFFFFFFFFu;
        if (gmode == 1 && !((gm >> (n0 / ELL)) & 1)) return;
        float acc[RW][MM];
#pragma unroll
        for (int r = 0; r < RW; r++)
#pragma unroll
            for (int m = 0; m < MM; m++) acc[r][m] = 0;
        for (int b = l; b < NB; b += SG) {
            if (gmode == 2 && !((gm >> (b * 32 / ELL)) & 1)) continue;
            float part[RW][MM];
#pragma unroll
            for (int r = 0; r < RW; r++)
#pragma unroll
                for (int m = 0; m < MM; m++) part[r][m] = 0;
#pragma unroll
            for (int hf = 0; hf < 2; hf++) {             // the block in two halves of 16 (registers)
                float xv[MM][16];
#pragma unroll
                for (int m = 0; m < MM; m++) {
                    if (m < M) {
                        const TX* xr = x + (size_t)m * K + b * 32 + hf * 16;
#pragma unroll
                        for (int j = 0; j < 16; j++) xv[m][j] = f(xr[j]);
                    }
                }
#pragma unroll
                for (int r = 0; r < RW; r++) {
                    const int n = n0 + r < N ? n0 + r : N - 1;
                    sycl::vec<int8_t, 16> v = *reinterpret_cast<const sycl::vec<int8_t, 16>*>(qs + (size_t)n * K + b * 32 + hf * 16);
#pragma unroll
                    for (int m = 0; m < MM; m++) {
                        if (m < M) {
                            float s = 0;
#pragma unroll
                            for (int j = 0; j < 16; j++) s += (float)v[j] * xv[m][j];
                            part[r][m] += s;
                        }
                    }
                }
            }
#pragma unroll
            for (int r = 0; r < RW; r++) {
                const int n = n0 + r < N ? n0 + r : N - 1;
                const float sc = f(d[(size_t)n * NB + b]);
#pragma unroll
                for (int m = 0; m < MM; m++) acc[r][m] += sc * part[r][m];
            }
        }
#pragma unroll
        for (int r = 0; r < RW; r++) {
            const int n = n0 + r;
#pragma unroll
            for (int m = 0; m < MM; m++) {
                if (m < M) {
                    float s = sycl::reduce_over_group(sg, acc[r][m], sycl::plus<float>());
                    if (l == 0 && n < N) y[(size_t)m * N + n] = static_cast<TY>(s);
                }
            }
        }
    });
}

template <class TX, class TY>
void q8_matvec(sycl::queue& q, const TX* x, int M, const Q8W& w, TY* y) {
    GLM_CHECK(M >= 1 && M <= MV_MAX, "q8_matvec: %d rows", M);
    static const int r4 = getenv("E0_MATVEC_R4") ? atoi(getenv("E0_MATVEC_R4")) : 1;
    if (r4 && M == 1) { q8_matvec_r4<TX, TY, 1>(q, x, M, w, y); return; }
    if (r4 && M <= 4 && w.N >= 8192) { q8_matvec_r4<TX, TY, 4>(q, x, M, w, y); return; }   // (few outputs: too few groups)
    const int8_t* qs = w.qs.ptr<int8_t>();
    const half* d = w.d.ptr<half>();
    const int N = w.N, K = w.K, NB = K / 32;
    constexpr int SG = 16, WG = 128, PER = WG / SG;
    size_t groups = (N + PER - 1) / PER;
    q.parallel_for(nd_range<1>(groups * WG, WG), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        auto sg = it.get_sub_group();
        int n = it.get_group(0) * PER + sg.get_group_id()[0];
        int l = sg.get_local_id()[0];
        if (n >= N) return;
        const int8_t* row = qs + (size_t)n * K;
        const half* dr = d + (size_t)n * NB;
        float acc[MV_MAX];
#pragma unroll
        for (int m = 0; m < MV_MAX; m++) acc[m] = 0;
        for (int b = l; b < NB; b += SG) {
            float sc = f(dr[b]);
            sycl::vec<int8_t, 16> v0 = *reinterpret_cast<const sycl::vec<int8_t, 16>*>(row + b * 32);
            sycl::vec<int8_t, 16> v1 = *reinterpret_cast<const sycl::vec<int8_t, 16>*>(row + b * 32 + 16);
            float wv[32];
#pragma unroll
            for (int j = 0; j < 16; j++) { wv[j] = (float)v0[j]; wv[16 + j] = (float)v1[j]; }
#pragma unroll
            for (int m = 0; m < MV_MAX; m++) {
                if (m < M) {
                    const TX* xr = x + (size_t)m * K + b * 32;
                    float s = 0;
#pragma unroll
                    for (int j = 0; j < 32; j++) s += wv[j] * f(xr[j]);
                    acc[m] += sc * s;
                }
            }
        }
#pragma unroll
        for (int m = 0; m < MV_MAX; m++) {
            if (m < M) {
                float s = sycl::reduce_over_group(sg, acc[m], sycl::plus<float>());
                if (l == 0) y[(size_t)m * N + n] = static_cast<TY>(s);
            }
        }
    });
}
template <class TX>
void q8_matvec_grouped(sycl::queue& q, const TX* x, const Q8W& w, float* y, const uint32_t* gmask, int gmode) {
    q8_matvec_r4<TX, float, 1>(q, x, 1, w, y, gmask, gmode);
}
template void q8_matvec_grouped<half>(sycl::queue&, const half*, const Q8W&, float*, const uint32_t*, int);
template void q8_matvec_grouped<float>(sycl::queue&, const float*, const Q8W&, float*, const uint32_t*, int);

void group_mask(sycl::queue& q, const int* sel, const int* label, int P, uint32_t* mask) {
    q.single_task([=] {
        uint32_t m = 0;
        for (int p = 0; p < P; p++) m |= 1u << label[sel[p]];
        *mask = m;
    });
}

template void q8_matvec<half, half>(sycl::queue&, const half*, int, const Q8W&, half*);
template void q8_matvec<half, float>(sycl::queue&, const half*, int, const Q8W&, float*);
template void q8_matvec<float, half>(sycl::queue&, const float*, int, const Q8W&, half*);
template void q8_matvec<float, float>(sycl::queue&, const float*, int, const Q8W&, float*);

void q8_dequant(sycl::queue& q, const Q8W& w, half* out) {
    const int8_t* qs = w.qs.ptr<int8_t>();
    const half* d = w.d.ptr<half>();
    size_t n = (size_t)w.N * w.K;
    q.parallel_for(range<1>(n / 32), [=](sycl::id<1> i) {
        float sc = f(d[i]);
        const int8_t* s = qs + i * 32;
        half* o = out + i * 32;
#pragma unroll
        for (int j = 0; j < 32; j++) o[j] = half(sc * (float)s[j]);
    });
}

template <class TY>
void q8_linear(sycl::queue& q, const half* x, int M, const Q8W& w, TY* y, half* scratch) {
    if (M <= MV_MAX) { q8_matvec<half, TY>(q, x, M, w, y); return; }
    q8_dequant(q, w, scratch);
    if constexpr (std::is_same_v<TY, float>) {
        oneapi::mkl::blas::row_major::gemm(q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::trans, M, w.N,
                                           w.K, 1.0f, x, w.K, scratch, w.K, 0.0f, y, w.N);
    } else {
        oneapi::mkl::blas::row_major::gemm(q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::trans, M, w.N,
                                           w.K, half(1.0f), x, w.K, scratch, w.K, half(0.0f), y, w.N);
    }
}
template void q8_linear<half>(sycl::queue&, const half*, int, const Q8W&, half*, half*);
template void q8_linear<float>(sycl::queue&, const half*, int, const Q8W&, float*, half*);

// ------------------------------------------------------------------------------------------------ latent MoE, decode

void nv_gate_up(sycl::queue& q, const float* xl, const int* sel, const int* label, const int64_t* ptrs, int M,
                float* h) {
    constexpr int SG = 16, WG = 128, PER = WG / SG, NSUB = ELL / 16;
    const size_t items = (size_t)M * 10 * EI;
    q.parallel_for(nd_range<1>((items + PER - 1) / PER * WG, WG), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        auto sg = it.get_sub_group();
        size_t s = it.get_group(0) * PER + sg.get_group_id()[0];
        int l = sg.get_local_id()[0];
        if (s >= items) return;
        int p = (int)(s / EI), j = (int)(s % EI);
        int e = sel[p];
        const char* rec = reinterpret_cast<const char*>(ptrs[e]);
        const float* x = xl + (size_t)(p / 10) * GE + (size_t)label[e] * ELL;
        const uint8_t* gr = reinterpret_cast<const uint8_t*>(rec) + (size_t)j * NV_ROW_GU;
        const uint8_t* ur = gr + (size_t)EI * NV_ROW_GU;
        float ag = 0, au = 0;
        for (int sb = l; sb < NSUB; sb += SG) {
            ag += nv_sub(gr, sb, x);
            au += nv_sub(ur, sb, x);
        }
        ag = sycl::reduce_over_group(sg, ag, sycl::plus<float>());
        au = sycl::reduce_over_group(sg, au, sycl::plus<float>());
        if (l == 0) h[s] = silu(tail(rec, 0) * ag) * (tail(rec, 1) * au);
    });
}

void nv_down(sycl::queue& q, const float* h, const int* sel, const int64_t* ptrs, int M, float* yl) {
    constexpr int SG = 16, WG = 128, PER = WG / SG, NSUB = EI / 16;
    const size_t items = (size_t)M * 10 * ELL;
    q.parallel_for(nd_range<1>((items + PER - 1) / PER * WG, WG), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        auto sg = it.get_sub_group();
        size_t s = it.get_group(0) * PER + sg.get_group_id()[0];
        int l = sg.get_local_id()[0];
        if (s >= items) return;
        int p = (int)(s / ELL), c = (int)(s % ELL);
        const char* rec = reinterpret_cast<const char*>(ptrs[sel[p]]);
        const uint8_t* row = reinterpret_cast<const uint8_t*>(rec) + NV_DOWN_OFF + (size_t)c * NV_ROW_D;
        const float* x = h + (size_t)p * EI;
        float a = 0;
        for (int sb = l; sb < NSUB; sb += SG) a += nv_sub(row, sb, x);
        a = sycl::reduce_over_group(sg, a, sycl::plus<float>());
        if (l == 0) yl[s] = tail(rec, 2) * a;
    });
}

void pick_leaders(sycl::queue& q, const int* sel, int P, int* lead) {
    q.parallel_for(range<1>(P), [=](sycl::id<1> pi) {
        int p = (int)pi, e = sel[p], l = p;
        for (int k = 0; k < p; k++)
            if (sel[k] == e) { l = k; break; }
        lead[p] = l;
    });
}

void nv_gate_up_g(sycl::queue& q, const float* xl, const int* sel, const int* lead, const int* label, const int64_t* ptrs,
                  int M, float* h) {
    constexpr int SG = 16, WG = 128, PER = WG / SG, NSUB = ELL / 16, NL = (NSUB + SG - 1) / SG;   // 7 sub-blocks a lane
    const int P = M * 10;
    const size_t items = (size_t)P * EI;
    q.parallel_for(nd_range<1>((items + PER - 1) / PER * WG, WG), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        auto sg = it.get_sub_group();
        size_t s = it.get_group(0) * PER + sg.get_group_id()[0];
        int l = sg.get_local_id()[0];
        if (s >= items) return;
        int u = (int)(s / EI), j = (int)(s % EI);
        if (lead[u] != u) return;
        int e = sel[u];
        const char* rec = reinterpret_cast<const char*>(ptrs[e]);
        const uint8_t* gr = reinterpret_cast<const uint8_t*>(rec) + (size_t)j * NV_ROW_GU;
        const uint8_t* ur = gr + (size_t)EI * NV_ROW_GU;
        // this lane's sub-blocks of the gate and up rows, read once
        float gsc[NL], usc[NL];
        uint32_t gw[NL][2], uw[NL][2];
#pragma unroll
        for (int i = 0; i < NL; i++) {
            int sb = l + i * SG;
            if (sb < NSUB) {
                const uint8_t* gb = gr + (sb >> 2) * 36;
                const uint8_t* ub = ur + (sb >> 2) * 36;
                int sub = sb & 3;
                gsc[i] = ue4m3(gb[sub]);
                usc[i] = ue4m3(ub[sub]);
                gw[i][0] = *reinterpret_cast<const uint32_t*>(gb + 4 + sub * 8);
                gw[i][1] = *reinterpret_cast<const uint32_t*>(gb + 8 + sub * 8);
                uw[i][0] = *reinterpret_cast<const uint32_t*>(ub + 4 + sub * 8);
                uw[i][1] = *reinterpret_cast<const uint32_t*>(ub + 8 + sub * 8);
            }
        }
        const float sgt = tail(rec, 0), sut = tail(rec, 1);
        const int g = label[e];
        for (int p = u; p < P; p++) {
            if (lead[p] != u) continue;
            const float* x = xl + (size_t)(p / 10) * GE + (size_t)g * ELL;
            float ag = 0, au = 0;
#pragma unroll
            for (int i = 0; i < NL; i++) {
                int sb = l + i * SG;
                if (sb >= NSUB) continue;
                const float* xb = x + (sb >> 2) * 64 + (sb & 3) * 16;
                float pg = 0, pu = 0;
#pragma unroll
                for (int k = 0; k < 8; k++) {
                    uint32_t bg = (gw[i][k >> 2] >> ((k & 3) * 8)) & 0xFF, bu = (uw[i][k >> 2] >> ((k & 3) * 8)) & 0xFF;
                    pg += e2m1(bg & 15) * xb[k] + e2m1(bg >> 4) * xb[8 + k];
                    pu += e2m1(bu & 15) * xb[k] + e2m1(bu >> 4) * xb[8 + k];
                }
                ag += pg * gsc[i];
                au += pu * usc[i];
            }
            ag = sycl::reduce_over_group(sg, ag, sycl::plus<float>());
            au = sycl::reduce_over_group(sg, au, sycl::plus<float>());
            if (l == 0) h[(size_t)p * EI + j] = silu(sgt * ag) * (sut * au);
        }
    });
}

void nv_down_g(sycl::queue& q, const float* h, const int* sel, const int* lead, const int64_t* ptrs, int M, float* yl) {
    constexpr int SG = 16, WG = 128, PER = WG / SG, NSUB = EI / 16, NL = (NSUB + SG - 1) / SG;     // 3 sub-blocks a lane
    const int P = M * 10;
    const size_t items = (size_t)P * ELL;
    q.parallel_for(nd_range<1>((items + PER - 1) / PER * WG, WG), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        auto sg = it.get_sub_group();
        size_t s = it.get_group(0) * PER + sg.get_group_id()[0];
        int l = sg.get_local_id()[0];
        if (s >= items) return;
        int u = (int)(s / ELL), c = (int)(s % ELL);
        if (lead[u] != u) return;
        const char* rec = reinterpret_cast<const char*>(ptrs[sel[u]]);
        const uint8_t* row = reinterpret_cast<const uint8_t*>(rec) + NV_DOWN_OFF + (size_t)c * NV_ROW_D;
        float sc[NL];
        uint32_t w[NL][2];
#pragma unroll
        for (int i = 0; i < NL; i++) {
            int sb = l + i * SG;
            if (sb < NSUB) {
                const uint8_t* b = row + (sb >> 2) * 36;
                int sub = sb & 3;
                sc[i] = ue4m3(b[sub]);
                w[i][0] = *reinterpret_cast<const uint32_t*>(b + 4 + sub * 8);
                w[i][1] = *reinterpret_cast<const uint32_t*>(b + 8 + sub * 8);
            }
        }
        const float sdt = tail(rec, 2);
        for (int p = u; p < P; p++) {
            if (lead[p] != u) continue;
            const float* x = h + (size_t)p * EI;
            float a = 0;
#pragma unroll
            for (int i = 0; i < NL; i++) {
                int sb = l + i * SG;
                if (sb >= NSUB) continue;
                const float* xb = x + (sb >> 2) * 64 + (sb & 3) * 16;
                float pa = 0;
#pragma unroll
                for (int k = 0; k < 8; k++) {
                    uint32_t bb = (w[i][k >> 2] >> ((k & 3) * 8)) & 0xFF;
                    pa += e2m1(bb & 15) * xb[k] + e2m1(bb >> 4) * xb[8 + k];
                }
                a += pa * sc[i];
            }
            a = sycl::reduce_over_group(sg, a, sycl::plus<float>());
            if (l == 0) yl[(size_t)p * ELL + c] = sdt * a;
        }
    });
}

void latent_combine(sycl::queue& q, const float* yl, const int* sel, const float* w, const int* label, int M, float* z) {
    q.parallel_for(range<1>((size_t)M * ELL), [=](sycl::id<1> i) {
        int r = (int)(i / ELL), c = (int)(i % ELL);
        float acc[G];
#pragma unroll
        for (int g = 0; g < G; g++) acc[g] = 0;
        for (int k = 0; k < 10; k++) {
            int p = r * 10 + k;
            int g = label[sel[p]];
            float v = w[p] * yl[(size_t)p * ELL + c];
#pragma unroll
            for (int gg = 0; gg < G; gg++) if (gg == g) acc[gg] += v;
        }
#pragma unroll
        for (int g = 0; g < G; g++) z[(size_t)r * GE + (size_t)g * ELL + c] = acc[g];
    });
}

// ------------------------------------------------------------------------------------------------ latent MoE, prompt

void gather_rows(sycl::queue& q, const half* src, int stride, int col0, const int* rows, int n, int width, half* A) {
    q.parallel_for(range<2>(n, width / 8), [=](sycl::id<2> i) {
        const half* s = src + (size_t)rows[i[0]] * stride + col0 + i[1] * 8;
        half* dst = A + (size_t)i[0] * width + i[1] * 8;
        *reinterpret_cast<sycl::vec<half, 8>*>(dst) = *reinterpret_cast<const sycl::vec<half, 8>*>(s);
    });
}

void combine_rows(sycl::queue& q, const float* O, const int* row_ids, const int* off, const int* pick, const float* w,
                  int nrows, int width, float* out) {
    q.parallel_for(range<2>(nrows, width), [=](sycl::id<2> i) {
        const int r = (int)i[0], c = (int)i[1];
        float a = 0;
        for (int k = off[r]; k < off[r + 1]; k++) a += w[pick[k]] * O[(size_t)pick[k] * width + c];
        out[(size_t)row_ids[r] * width + c] += a;
    });
}

void scatter_rows(sycl::queue& q, const float* O, const int* rows, const float* wk, int n, int width, float* out) {
    q.parallel_for(range<2>(n, width), [=](sycl::id<2> i) {
        out[(size_t)rows[i[0]] * width + i[1]] += wk[i[0]] * O[(size_t)i[0] * width + i[1]];
    });
}

void nv_dequant(sycl::queue& q, const char* rec, half* gu, half* dn) {
    // one item per (tile, row): 1280/16 x 112 gate|up tiles, then 112 x 40 down tiles, 16 rows each
    constexpr int GU_T = (2 * EI / 16) * (ELL / 16), DN_T = (ELL / 16) * (EI / 16);
    q.parallel_for(range<2>(GU_T + DN_T, 16), [=](sycl::id<2> ii) {
        int ti = (int)ii[0], i = (int)ii[1];
        const uint8_t* t;
        half* out;
        if (ti < GU_T) {
            int G = ti / (ELL / 16), s = ti % (ELL / 16);
            t = reinterpret_cast<const uint8_t*>(rec) + (size_t)ti * NV_TILE;
            out = gu + (size_t)(G * 16 + i) * ELL + s * 16;
        } else {
            ti -= GU_T;
            int G = ti / (EI / 16), s = ti % (EI / 16);
            t = reinterpret_cast<const uint8_t*>(rec) + NV_DOWN_OFF + (size_t)ti * NV_TILE;
            out = dn + (size_t)(G * 16 + i) * EI + s * 16;
        }
        float w[16];
        const float sc = tile_row(t, i, w);
        for (int k = 0; k < 16; k++) out[k] = half(w[k] * sc);
    });
}

void nv_act(sycl::queue& q, const float* GU, const char* rec, int n, half* h16) {
    q.parallel_for(range<1>((size_t)n * EI), [=](sycl::id<1> i) {
        size_t r = i / EI, j = i % EI;
        float g = tail(rec, 0) * GU[r * 2 * EI + j], u = tail(rec, 1) * GU[r * 2 * EI + EI + j];
        h16[i] = half(silu(g) * u);
    });
}

void scatter_add(sycl::queue& q, const float* Y, const int* rows, const float* wk, int n, float* z, int col0) {
    q.parallel_for(range<2>(n, ELL), [=](sycl::id<2> i) {
        z[(size_t)rows[i[0]] * GE + col0 + i[1]] += wk[i[0]] * Y[(size_t)i[0] * ELL + i[1]];
    });
}

void gather_picks(sycl::queue& q, const half* xl16, const int* rows, const int* col, int n, half* Xb) {
    q.parallel_for(range<2>(n, ELL / 8), [=](sycl::id<2> i) {
        const half* s = xl16 + (size_t)rows[i[0]] * GE + col[i[0]] + i[1] * 8;
        *reinterpret_cast<sycl::vec<half, 8>*>(Xb + (size_t)i[0] * ELL + i[1] * 8) = *reinterpret_cast<const sycl::vec<half, 8>*>(s);
    });
}

void nv_dequant_batch(sycl::queue& q, const int64_t* recs, int cnt, half* gu, half* dn) {
    constexpr int GU_T = (2 * EI / 16) * (ELL / 16), DN_T = (ELL / 16) * (EI / 16);
    q.parallel_for(range<3>(cnt, GU_T + DN_T, 16), [=](sycl::id<3> ii) {
        const char* rec = reinterpret_cast<const char*>(recs[ii[0]]);
        int ti = (int)ii[1], i = (int)ii[2];
        const uint8_t* t;
        half* out;
        if (ti < GU_T) {
            int G = ti / (ELL / 16), s = ti % (ELL / 16);
            t = reinterpret_cast<const uint8_t*>(rec) + (size_t)ti * NV_TILE;
            out = gu + ii[0] * (size_t)2 * EI * ELL + (size_t)(G * 16 + i) * ELL + s * 16;
        } else {
            ti -= GU_T;
            int G = ti / (EI / 16), s = ti % (EI / 16);
            t = reinterpret_cast<const uint8_t*>(rec) + NV_DOWN_OFF + (size_t)ti * NV_TILE;
            out = dn + ii[0] * (size_t)ELL * EI + (size_t)(G * 16 + i) * EI + s * 16;
        }
        float w[16];
        const float sc = tile_row(t, i, w);
        for (int k = 0; k < 16; k++) out[k] = half(w[k] * sc);
    });
}

void nv_act_rows(sycl::queue& q, const float* GU, const float* sgu, int n, half* h16) {
    q.parallel_for(range<1>((size_t)n * EI), [=](sycl::id<1> i) {
        size_t r = i / EI, j = i % EI;
        float g = sgu[2 * r] * GU[r * 2 * EI + j], u = sgu[2 * r + 1] * GU[r * 2 * EI + EI + j];
        h16[i] = half(silu(g) * u);
    });
}

// ------------------------------------------------------------------------------------------------ output

void add_shared(sycl::queue& q, float* out, const float* sh, const half* rl, int stride, int rows) {
    constexpr int D = 2560;
    q.parallel_for(range<1>((size_t)rows * D), [=](sycl::id<1> i) {
        size_t r = i / D;
        out[i] += sigm(f(rl[r * stride + 512])) * sh[i];
    });
}

void f32_to_f16(sycl::queue& q, const float* x, half* y, size_t n) {
    q.parallel_for(range<1>(n), [=](sycl::id<1> i) { y[i] = half(x[i]); });
}

}  // namespace glm::e0ops

// ------------------------------------------------------------------------------------------------ int8 KV

namespace glm::e0ops {
namespace {
using namespace q4;
constexpr float NEG_INF = -INFINITY;
constexpr int CH = 128, NCH = (NSEL + CH - 1) / CH, HG = AH / AKV;   // as q4_ops.cpp (the partials' layout)
constexpr int PSZ = 2 * HG + HG * AD;

inline float rope_at(const float* buf, int l, int p) {
    if (l >= ROT) return buf[l];
    int j = l & (ROT / 2 - 1);
    float inv = sycl::pow(1.0e7f, -(float)(2 * j) / (float)ROT);
    float th = (float)p * inv, c = sycl::cos(th), s = sycl::sin(th);
    return l < ROT / 2 ? buf[l] * c - buf[l + ROT / 2] * s : buf[l] * c + buf[l - ROT / 2] * s;
}

// a work-group of 256 (sub-groups of 16) holds one head row o, one value a lane: per 64 values, s = fp16(absmax / 127),
// q = round(o / s) against the stored fp16 s (Strata's kv_q8). sm: local scratch of 16 floats
template <class It>
inline int8_t quant8(const It& it, float* sm, float o, half& s) {
    auto sg = it.get_sub_group();
    int l = it.get_local_id(0);
    float a = sycl::reduce_over_group(sg, sycl::fabs(o), sycl::maximum<float>());
    if (sg.get_local_id()[0] == 0) sm[sg.get_group_id()[0]] = a;
    sycl::group_barrier(it.get_group());
    int g4 = (l / 64) * 4;
    a = sycl::fmax(sycl::fmax(sm[g4], sm[g4 + 1]), sycl::fmax(sm[g4 + 2], sm[g4 + 3]));
    s = half(a / 127.0f);
    float sf = f(s);
    return sf > 0 ? (int8_t)sycl::clamp(sycl::rint(o / sf), -127.0f, 127.0f) : (int8_t)0;
}
}  // namespace

void attn_prep_i8(sycl::queue& q, const half* qg, const half* k, const half* v, const half* ik, const float* qn,
                  const float* kn, const float* iqn, int rows, int p0, half* qn16, int8_t* Kq, half* Ks, int8_t* Vq,
                  half* Vs, float* qi, half* rk, const int* pdev) {
    constexpr int U = AH + 2 * AKV + IH + 1;
    q.submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> buf(sycl::range<1>(256), hd);
        sycl::local_accessor<float, 1> sm(sycl::range<1>(16), hd);
        hd.parallel_for(nd_range<1>((size_t)rows * U * 256, 256), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            int gid = it.get_group(0), l = it.get_local_id(0);
            int r = gid / U, u = gid % U, p = (pdev ? *pdev : p0) + r;
            auto grp = it.get_group();
            if (u < AH + AKV) {
                bool isq = u < AH;
                int h = isq ? u : u - AH;
                float x = isq ? f(qg[(size_t)r * AH * 2 * AD + h * 2 * AD + l]) : f(k[(size_t)r * AKV * AD + h * AD + l]);
                float ss = sycl::reduce_over_group(grp, x * x, sycl::plus<float>());
                buf[l] = x * sycl::rsqrt(ss / AD + EPS) * (isq ? qn : kn)[l];
                sycl::group_barrier(grp);
                float o = rope_at(&buf[0], l, p);
                if (isq) {
                    qn16[((size_t)r * AH + h) * AD + l] = half(o);
                } else {
                    half s;
                    // the key as the fp16 path stores it, then quantized
                    int8_t qv = quant8(it, &sm[0], f(half(o)), s);
                    Kq[((size_t)p * AKV + h) * AD + l] = qv;
                    if (l % 64 == 0) Ks[((size_t)p * AKV + h) * 4 + l / 64] = s;
                }
            } else if (u < AH + 2 * AKV) {
                int h = u - AH - AKV;
                half s;
                int8_t qv = quant8(it, &sm[0], f(v[(size_t)r * AKV * AD + h * AD + l]), s);
                Vq[((size_t)p * AKV + h) * AD + l] = qv;
                if (l % 64 == 0) Vs[((size_t)p * AKV + h) * 4 + l / 64] = s;
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

void attn_core_i8(sycl::queue& q, const half* qn16, const int8_t* Kq, const half* Ks, const int8_t* Vq, const half* Vs,
                  const int* list, const int* cnt, const half* qg, int rows, int p0, float* part, half* o16,
                  const int* pdev) {
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
                int j = l & (CH - 1), hh = l / CH;
                float acc[HG / 2] = {};
                if (j < nt) {
                    const size_t at = (size_t)tk[j] * AKV + g;
                    const int8_t* kp = Kq + at * AD;
                    for (int g4 = 0; g4 < 4; g4++) {
                        float part[HG / 2] = {};
                        for (int d = g4 * 64; d < g4 * 64 + 64; d += 16) {
                            sycl::vec<int8_t, 16> kv = *reinterpret_cast<const sycl::vec<int8_t, 16>*>(kp + d);
                            for (int e = 0; e < 16; e++) {
                                float kf = (float)kv[e];
                                for (int h = 0; h < HG / 2; h++) part[h] += qs[(hh * (HG / 2) + h) * AD + d + e] * kf;
                            }
                        }
                        float ks = f(Ks[at * 4 + g4]);
                        for (int h = 0; h < HG / 2; h++) acc[h] += part[h] * ks;
                    }
                }
                for (int h = 0; h < HG / 2; h++)
                    ps[(hh * (HG / 2) + h) * CH + j] = j < nt ? acc[h] * 0.0625f : NEG_INF;
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
                    size_t at = (size_t)tk[j] * AKV + g;
                    float vv = (float)Vq[at * AD + l] * f(Vs[at * 4 + l / 64]);
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

}  // namespace glm::e0ops

// ------------------------------------------------------------------------------------------------ MTP

namespace glm::e0ops {
namespace {
constexpr int HD = 2560, FF = 640;
constexpr size_t Q2_DN_CODES = (size_t)2 * FF * (HD / 64) * 16, Q2_GU_SC = Q2_DN_CODES + (size_t)HD * (FF / 64) * 16,
                 Q2_DN_SC = Q2_GU_SC + (size_t)2 * FF * (HD / 64) * 2;
// one Q2_0 block of 64 (16 code bytes, its scale): sum of value * x
template <class TX>
inline float q2_blk(const uint8_t* codes, half d, const TX* x) {
    float a = 0;
#pragma unroll
    for (int b = 0; b < 16; b++) {
        uint32_t c = codes[b];
#pragma unroll
        for (int j = 0; j < 4; j++) a += (float)((int)((c >> (2 * j)) & 3) - 1) * f(x[b * 4 + j]);
    }
    return a * f(d);
}
}  // namespace

void q2_gate_up(sycl::queue& q, const half* y, const int* sel, const int64_t* ptrs, int M, float* h) {
    constexpr int SG = 16, WG = 128, PER = WG / SG, NB = HD / 64;
    const size_t items = (size_t)M * 10 * FF;
    q.parallel_for(nd_range<1>((items + PER - 1) / PER * WG, WG), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        auto sg = it.get_sub_group();
        size_t s = it.get_group(0) * PER + sg.get_group_id()[0];
        int l = sg.get_local_id()[0];
        if (s >= items) return;
        int p = (int)(s / FF), j = (int)(s % FF);
        const uint8_t* rec = reinterpret_cast<const uint8_t*>(ptrs[sel[p]]);
        const half* x = y + (size_t)(p / 10) * HD;
        const uint8_t* cg = rec + (size_t)(2 * j) * NB * 16;
        const uint8_t* cu = cg + (size_t)NB * 16;
        const half* dg = reinterpret_cast<const half*>(rec + Q2_GU_SC) + (size_t)(2 * j) * NB;
        const half* du = dg + NB;
        float ag = 0, au = 0;
        for (int b = l; b < NB; b += SG) {
            ag += q2_blk(cg + b * 16, dg[b], x + b * 64);
            au += q2_blk(cu + b * 16, du[b], x + b * 64);
        }
        ag = sycl::reduce_over_group(sg, ag, sycl::plus<float>());
        au = sycl::reduce_over_group(sg, au, sycl::plus<float>());
        if (l == 0) h[s] = silu(ag) * au;
    });
}

void q2_down_sum(sycl::queue& q, const float* h, const int* sel, const float* w, const int64_t* ptrs, int M, float* out) {
    constexpr int SG = 16, WG = 128, PER = WG / SG, NB = FF / 64;
    const size_t items = (size_t)M * HD;
    q.parallel_for(nd_range<1>((items + PER - 1) / PER * WG, WG), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        auto sg = it.get_sub_group();
        size_t s = it.get_group(0) * PER + sg.get_group_id()[0];
        int l = sg.get_local_id()[0];
        if (s >= items) return;
        int r = (int)(s / HD), c = (int)(s % HD);
        float acc = 0;
        for (int k = 0; k < 10; k++) {
            const int p = r * 10 + k;
            const uint8_t* rec = reinterpret_cast<const uint8_t*>(ptrs[sel[p]]);
            const uint8_t* cd = rec + Q2_DN_CODES + (size_t)c * NB * 16;
            const half* dd = reinterpret_cast<const half*>(rec + Q2_DN_SC) + (size_t)c * NB;
            float a = 0;
            for (int b = l; b < NB; b += SG) a += q2_blk(cd + b * 16, dd[b], h + (size_t)p * FF + b * 64);
            acc += w[p] * a;
        }
        acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
        if (l == 0) out[s] = acc;
    });
}

void mtp_prep(sycl::queue& q, const half* qg, const half* k, const half* v, const float* qn, const float* kn, int rows,
              int p0, half* qn16, half* Kc, half* Vc) {
    const int U = (qg ? AH : 0) + 2 * AKV;
    q.submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> buf(sycl::range<1>(256), hd);
        hd.parallel_for(nd_range<1>((size_t)rows * U * 256, 256), [=](nd_item<1> it) {
            int gid = it.get_group(0), l = it.get_local_id(0);
            int r = gid / U, u = gid % U, p = p0 + r;
            if (!qg) u += AH;
            auto grp = it.get_group();
            if (u < AH + AKV) {
                bool isq = u < AH;
                int h = isq ? u : u - AH;
                float x = isq ? f(qg[(size_t)r * AH * 2 * AD + h * 2 * AD + l]) : f(k[(size_t)r * AKV * AD + h * AD + l]);
                float ss = sycl::reduce_over_group(grp, x * x, sycl::plus<float>());
                buf[l] = x * sycl::rsqrt(ss / AD + EPS) * (isq ? qn : kn)[l];
                sycl::group_barrier(grp);
                float o = rope_at(&buf[0], l, p);
                if (isq) qn16[((size_t)r * AH + h) * AD + l] = half(o);
                else Kc[((size_t)p * AKV + h) * AD + l] = half(o);
            } else {
                int h = u - AH - AKV;
                Vc[((size_t)p * AKV + h) * AD + l] = v[(size_t)r * AKV * AD + h * AD + l];
            }
        });
    });
}

void route512(sycl::queue& q, const half* rl, int stride, int rows, int* sel, float* w) {
    q.submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> lg(sycl::range<1>(512), hd);
        sycl::local_accessor<float, 1> top(sycl::range<1>(10), hd);
        sycl::local_accessor<int, 1> topi(sycl::range<1>(10), hd);
        hd.parallel_for(nd_range<1>((size_t)rows * 512, 512), [=](nd_item<1> it) {
            int r = it.get_group(0), e = it.get_local_id(0);
            float v = f(rl[(size_t)r * stride + e]);
            lg[e] = v;
            sycl::group_barrier(it.get_group());
            int rank = 0;
            for (int k = 0; k < 512; k++) {
                float o = lg[k];
                rank += o > v || (o == v && k < e);
            }
            if (rank < 10) { top[rank] = v; topi[rank] = e; }
            sycl::group_barrier(it.get_group());
            if (e < 10) {
                float m = top[0], s = 0;
                for (int k = 0; k < 10; k++) s += sycl::exp(top[k] - m);
                sel[r * 10 + e] = topi[e];
                w[r * 10 + e] = sycl::exp(top[e] - m) / (s + 1e-20f);
            }
        });
    });
}

void attn_pool_rows(sycl::queue& q, const half* rk, const float* ikn, half* pk, const int* pdev, int rows) {
    q.submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> buf(sycl::range<1>(ID), hd);
        hd.parallel_for(nd_range<1>((size_t)rows * ID, ID), [=](nd_item<1> it) {
            int p = *pdev + (int)it.get_group(0), l = it.get_local_id(0);
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

size_t mtp_attn_floats(int max_tokens) { return (size_t)AKV * ((max_tokens + CH - 1) / CH) * PSZ; }

void mtp_attn(sycl::queue& q, const half* qn16, const half* Kc, const half* Vc, const half* qg, int p, float* part,
              half* o16) {
    const int n = p + 1, nch = (n + CH - 1) / CH;
    q.submit([&](sycl::handler& hd) {
        sycl::local_accessor<float, 1> qs(sycl::range<1>(HG * AD), hd);
        sycl::local_accessor<float, 1> ps(sycl::range<1>(HG * CH), hd);
        hd.parallel_for(nd_range<1>((size_t)AKV * nch * 256, 256), [=](nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            int gid = it.get_group(0), l = it.get_local_id(0);
            int c = gid % nch, g = gid / nch;
            auto grp = it.get_group();
            int t0 = c * CH, nt = sycl::min(CH, n - t0);
            float* P = part + ((size_t)g * nch + c) * PSZ;
            for (int i = l; i < HG * AD; i += 256) qs[i] = f(qn16[(size_t)g * HG * AD + i]);
            sycl::group_barrier(grp);
            {
                int j = l & (CH - 1), hh = l / CH;
                float acc[HG / 2] = {};
                if (j < nt) {
                    const half* kp = Kc + ((size_t)(t0 + j) * AKV + g) * AD;
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
                    float vv = f(Vc[((size_t)(t0 + j) * AKV + g) * AD + l]);
                    for (int h = 0; h < HG; h++) acc[h] += ps[h * CH + j] * vv;
                }
                for (int h = 0; h < HG; h++) P[2 * HG + h * AD + l] = acc[h];
            }
        });
    });
    q.parallel_for(range<1>((size_t)AH * AD), [=](sycl::id<1> i) {
        int h = i / AD, d = i % AD, g = h / HG, hh = h % HG;
        const float* P = part + (size_t)g * nch * PSZ;
        float M = NEG_INF;
        for (int c = 0; c < nch; c++) M = sycl::fmax(M, P[c * PSZ + hh]);
        float num = 0, den = 0;
        for (int c = 0; c < nch; c++) {
            float m = P[c * PSZ + hh];
            if (m == NEG_INF) continue;
            float e = sycl::exp(m - M);
            den += P[c * PSZ + HG + hh] * e;
            num += P[c * PSZ + 2 * HG + hh * AD + d] * e;
        }
        float gate = f(qg[h * 2 * AD + AD + d]);
        o16[h * AD + d] = half(num / den * sigm(gate));
    });
}

void rms_rows(sycl::queue& q, const float* x, const float* w, int rows, int n, half* y) {
    q.parallel_for(nd_range<1>((size_t)rows * 256, 256), [=](nd_item<1> it) {
        int r = it.get_group(0), l = it.get_local_id(0);
        const float* xr = x + (size_t)r * n;
        float ss = 0;
        for (int i = l; i < n; i += 256) ss += xr[i] * xr[i];
        ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
        float sc = sycl::rsqrt(ss / n + EPS);
        for (int i = l; i < n; i += 256) y[(size_t)r * n + i] = half(xr[i] * sc * w[i]);
    });
}

void add_streams(sycl::queue& q, const float* h, const float* e, int rows, float* R) {
    q.parallel_for(range<1>((size_t)rows * 4 * HD), [=](sycl::id<1> i) {
        size_t t = i / (4 * HD), d = i % HD;
        R[i] = h[i] + e[t * HD + d];
    });
}

void argmax_prob(sycl::queue& q, const half* logits, int n, int* out) {
    q.submit([&](sycl::handler& hd) {
        hd.parallel_for(nd_range<1>(1024, 1024), [=](nd_item<1> it) {
            auto grp = it.get_group();
            int l = it.get_local_id(0);
            float bv = NEG_INF;
            int bi = n;
            for (int i = l; i < n; i += 1024) {
                float v = f(logits[i]);
                if (v > bv) { bv = v; bi = i; }
            }
            float m = sycl::reduce_over_group(grp, bv, sycl::maximum<float>());
            int idx = sycl::reduce_over_group(grp, bv == m ? bi : n, sycl::minimum<int>());
            float s = 0;
            for (int i = l; i < n; i += 1024) s += sycl::exp(f(logits[i]) - m);
            s = sycl::reduce_over_group(grp, s, sycl::plus<float>());
            if (l == 0) {
                out[0] = idx;
                out[1] = sycl::bit_cast<int>(1.0f / s);
            }
        });
    });
}

}  // namespace glm::e0ops

