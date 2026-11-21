#pragma once
// The gated delta rule over a prompt's rows in chunks of 64 (Qwen3.5 / Qwen3.8 GDN layers; ESIMD with DPAS).
//
// Per v-head j, token t: S <- exp(g_t) S;  S += beta_t k_t (v_t - S^T k_t)^T;  o_t = S^T q_t   (S: [128 k][128 v],
// q and k l2-normed, q scaled by 1/sqrt(128)). Over a chunk with c_i = g_0 + .. + g_i (Gamma_i = exp(c_i)), from the
// state S0 at its start (the WY form of FLA's chunk_gated_delta_rule, solved by substitution instead of an inverse):
//   A[i][m]  = beta_i (k_i . k_m) exp(c_i - c_m), m < i
//   Delta_i  = beta_i x_i - sum_{m < i} A[i][m] Delta_m,   x_i = v_i - Gamma_i S0^T k_i     the rows' corrected values
//   o_i      = Gamma_i S0^T q_i + sum_{m <= i} Mq[i][m] Delta_m,   Mq[i][m] = (q_i . k_m) exp(c_i - c_m)
//   S1       = exp(c_63) S0 + sum_m exp(c_63 - c_m) k_m Delta_m^T
// q35ops::gdn_chunked runs: norm (q, k -> fp16, SYCL), GdnPrep (k k^T and q k^T per chunk and k-head by DPAS, then A,
// Mq and the decays of its three v-heads), GdnScan (the chunks in order).
//
// GdnScan: a thread owns one v-head's 8 v columns, its state transposed in registers hT [8 kc][8 v][16 k] fp32 (DPAS
// A tiles once converted, DPAS C tiles of the update). The products run transposed (v is the DPAS M dimension):
//   R^T = S0^T K^T, Q^T = S0^T Q^T                      B = k^T / q^T tiles by transposed dword loads (VNNI)
//   Delta row by row in fp32, [64 i][8 v]               the substitution; the chunk's A staged in SLM by the
//                                                       work-group (one v-head's 8 threads), not loaded row by row
//   O^T = Gamma Q^T + Delta^T Mq^T                      B = Mq^T by a transposed dword load of Mq
//   S1^T = exp(c_63) S0^T + (w Delta)^T K               B = k tiles by a VNNI load
// DPAS inputs are fp16 (q, k normed; S0, Delta, Mq: magnitudes well inside fp16), every sum fp32, the state and the
// substitution fp32.

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>

namespace q35gdn {

using namespace sycl::ext::intel::esimd;
namespace xesimd = sycl::ext::intel::experimental::esimd;
using fp16 = sycl::half;
using xesimd::cache_hint;

constexpr int CK = 64;          // rows a chunk
constexpr int GK = 16, GV = 48, DK = 128, DV = 128, QKV = 10240;
constexpr int VB = 8;           // v columns a scan thread owns

inline simd<float, 16> bf16_to_f32(simd<uint16_t, 16> u) {
    simd<uint32_t, 16> w = convert<uint32_t>(u) << 16;
    return w.template bit_cast_view<float>().read();
}
template <int N> inline simd<uint16_t, N> f32_to_bf16(simd<float, N> f) {
    simd<uint32_t, N> u = f.template bit_cast_view<uint32_t>().read();
    u = u + 0x7FFFu + ((u >> 16) & 1u);
    return convert<uint16_t>(u >> 16);
}

// thread = (chunk, k-head, 8-row block): k k^T and q k^T of its rows (DPAS), then for each of the k-head's three v-heads
// its rows of A (fp32) and Mq (fp16), [nc][48][64][64] with zeros above the diagonal; the row block 0 thread also writes
// the chunk's beta, exp(c_i) and exp(c_63 - c_i), [nc][48][64]. Rows past n count as beta = g = 0, k = q = 0.
struct GdnPrep {
    const fp16* qh; const fp16* kh;     // [npad][16][128]
    const float* beta; const float* g;  // [n][48]
    float* A; fp16* Mq; float* bt; float* gam; float* wl;
    int n, npad, nc;

    void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
        const int id = it.get_global_id(0);
        const int rb = id % 8, khi = (id / 8) % GK, c = id / (8 * GK);
        if (c >= nc) return;
        const unsigned W = DK * 2 - 1, H = npad - 1, P = GK * DK * 2 - 1;
        const fp16* kb = kh + khi * DK;
        const fp16* qb = qh + khi * DK;
        const int r0 = c * CK;
        simd<fp16, 8 * 128> Ak, Aq;      // [kc][8 rows][16]
#pragma unroll
        for (int kc = 0; kc < 8; kc++) {
            Ak.template select<128, 1>(kc * 128) = xesimd::lsc_load_2d<fp16, 16, 8, 1, false, false, cache_hint::cached,
                                                                       cache_hint::cached>(kb, W, H, P, kc * 16, r0 + rb * 8);
            Aq.template select<128, 1>(kc * 128) = xesimd::lsc_load_2d<fp16, 16, 8, 1, false, false, cache_hint::cached,
                                                                       cache_hint::cached>(qb, W, H, P, kc * 16, r0 + rb * 8);
        }
        const uint32_t* kb32 = reinterpret_cast<const uint32_t*>(kb);
        const int tmax = (rb * 8 + 7) / 16;
        simd<float, 8 * 64> KK = 0.0f, QK = 0.0f;   // [8 rows][64 m]
        for (int t = 0; t <= tmax; t++) {
            simd<float, 128> ak = 0.0f, aq = 0.0f;
#pragma unroll
            for (int kc = 0; kc < 8; kc++) {
                simd<uint32_t, 128> b = xesimd::lsc_load_2d<uint32_t, 8, 16, 1, true, false, cache_hint::cached,
                                                            cache_hint::cached>(kb32, W, H, P, kc * 8, r0 + t * 16);
                simd<fp16, 256> bh = b.template bit_cast_view<fp16>().read();
                ak = xmx::dpas<8, 8, float, float, fp16, fp16>(ak, bh, simd<fp16, 128>(Ak.template select<128, 1>(kc * 128)));
                aq = xmx::dpas<8, 8, float, float, fp16, fp16>(aq, bh, simd<fp16, 128>(Aq.template select<128, 1>(kc * 128)));
            }
#pragma unroll
            for (int r = 0; r < 8; r++) {
                KK.template select<16, 1>(r * 64 + t * 16) = ak.template select<16, 1>(r * 16);
                QK.template select<16, 1>(r * 64 + t * 16) = aq.template select<16, 1>(r * 16);
            }
        }
        // rows of the chunk (clamped to the prompt; past it g = beta = 0)
        simd<uint32_t, 64> lane(0, 1);
        simd<uint32_t, 64> row = lane + (uint32_t)r0;
        simd_mask<64> live = row < (uint32_t)n;
        simd<uint32_t, 64> rowc = row;
        rowc.merge(simd<uint32_t, 64>((uint32_t)(n - 1)), !live);
        constexpr float LOG2E = 1.4426950408889634f;
#pragma unroll
        for (int jj = 0; jj < GV / GK; jj++) {
            const int j = khi * (GV / GK) + jj;
            simd<uint32_t, 64> off = (rowc * (uint32_t)GV + (uint32_t)j) * 4u;
            simd<float, 64> gv = gather<float, 64>(g, off);
            simd<float, 64> bv = gather<float, 64>(beta, off);
            gv.merge(simd<float, 64>(0.0f), !live);
            bv.merge(simd<float, 64>(0.0f), !live);
            // c = inclusive prefix sum of g
            simd<float, 64> cs = gv;
#define Q35GDN_SCAN_STEP(S)                                                                              \
            {                                                                                           \
                simd<float, 64> sh = 0.0f;                                                              \
                sh.template select<64 - S, 1>(S) = cs.template select<64 - S, 1>(0);                    \
                cs += sh;                                                                               \
            }
            Q35GDN_SCAN_STEP(1) Q35GDN_SCAN_STEP(2) Q35GDN_SCAN_STEP(4) Q35GDN_SCAN_STEP(8) Q35GDN_SCAN_STEP(16)
            Q35GDN_SCAN_STEP(32)
#undef Q35GDN_SCAN_STEP
            const size_t ch = (size_t)c * GV + j;
#pragma unroll
            for (int r = 0; r < 8; r++) {
                const int i = rb * 8 + r;
                const float ci = cs[i];
                simd<float, 64> arg = ci - cs;
                arg.merge(simd<float, 64>(0.0f), lane > (uint32_t)i);
                simd<float, 64> e = exp2(arg * LOG2E);
                simd<float, 64> a = simd<float, 64>(KK.template select<64, 1>(r * 64)) * e * (float)bv[i];
                a.merge(simd<float, 64>(0.0f), lane >= (uint32_t)i);
                simd<float, 64> mq = simd<float, 64>(QK.template select<64, 1>(r * 64)) * e;
                mq.merge(simd<float, 64>(0.0f), lane > (uint32_t)i);
                block_store<float, 64>(A + (ch * CK + i) * CK, a);
                block_store<fp16, 64>(Mq + (ch * CK + i) * CK, convert<fp16>(mq));
            }
            if (rb == 0) {
                block_store<float, 64>(bt + ch * CK, bv);
                block_store<float, 64>(gam + ch * CK, exp2(cs * LOG2E));
                block_store<float, 64>(wl + ch * CK, exp2(((float)cs[CK - 1] - cs) * LOG2E));
            }
        }
    }
};

// The chunks in order: thread = (v-head j, 8 v columns vb). rec fp32 [48][128 k][128 v] in and out, core bf16
// [n][48][128] out.
struct GdnScan {
    const uint16_t* cv;                 // bf16 [n][10240]: q | k | v
    const fp16* qh; const fp16* kh;     // [npad][16][128]
    const float* A; const fp16* Mq;     // [nc][48][64][64]
    const float* bt;                    // [nc][48][64]: beta,
    const float* gam; const float* wl;  // exp(c_i), exp(c_63 - c_i)
    float* rec;
    uint16_t* core;
    int n, npad, nc;

    void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
        const int id = it.get_global_id(0);
        const int vb = id % (DV / VB), j = id / (DV / VB);
        if (j >= GV) return;                  // never: 768 threads, whole work-groups (barriers below)
        const int khi = j / (GV / GK);
        slm_init<CK * CK * 4>();              // the chunk's A, shared by the work-group's 8 threads (one v-head)
        const int w8 = vb % 8;

        // ---- state in: hT [kc][v][16 k]
        simd<float, 8 * 128> hT;
        float* rp = rec + (size_t)j * DK * DV;
#pragma unroll
        for (int kb = 0; kb < 4; kb++) {
            // rows k 32 kb .. + 31, columns v vb * 8 .. + 7, transposed: [8 v][32 k]
            simd<float, 256> t = xesimd::lsc_load_2d<float, 8, 32, 1, true, false>(rp, DV * 4 - 1, DK - 1, DV * 4 - 1,
                                                                                     vb * VB, kb * 32);
#pragma unroll
            for (int v = 0; v < 8; v++) {
                hT.template select<16, 1>((2 * kb) * 128 + v * 16) = t.template select<16, 1>(v * 32);
                hT.template select<16, 1>((2 * kb + 1) * 128 + v * 16) = t.template select<16, 1>(v * 32 + 16);
            }
        }

        const unsigned kW = DK * 2 - 1, kH = npad - 1, kP = GK * DK * 2 - 1;
        const fp16* kb_ = kh + khi * DK;
        const fp16* qb_ = qh + khi * DK;
        const uint32_t* kb32 = reinterpret_cast<const uint32_t*>(kb_);
        const uint32_t* qb32 = reinterpret_cast<const uint32_t*>(qb_);
        const unsigned cvW = QKV * 2 - 1, cvH = n - 1, cvP = QKV * 2 - 1;
        const unsigned oW = GV * DV * 2 - 1, oH = n - 1, oP = GV * DV * 2 - 1;

        for (int c = 0; c < nc; c++) {
            const int r0 = c * CK;
            const size_t ch = (size_t)c * GV + j;
            // ---- this chunk's A into SLM, 8 rows a thread (read by the substitution below, after a barrier)
            barrier();                        // the last chunk's substitution has read its A
            {
                const float* Ab = A + ch * CK * CK + (size_t)w8 * 8 * CK;
#pragma unroll
                for (int r = 0; r < 8; r++)
                    slm_block_store<float, 64>((uint32_t)((w8 * 8 + r) * CK * 4), block_load<float, 64>(Ab + (size_t)r * CK));
            }
            // ---- R^T = S0^T K^T, Q^T = S0^T Q^T: [t][8 v][16 i]
            simd<fp16, 8 * 128> hA = convert<fp16>(hT);
            simd<float, 4 * 128> RT, QT;
#pragma unroll
            for (int t = 0; t < 4; t++) {
                simd<float, 128> ar = 0.0f, aq = 0.0f;
#pragma unroll
                for (int kc = 0; kc < 8; kc++) {
                    simd<uint32_t, 128> bk = xesimd::lsc_load_2d<uint32_t, 8, 16, 1, true, false, cache_hint::cached,
                                                                 cache_hint::cached>(kb32, kW, kH, kP, kc * 8, r0 + t * 16);
                    simd<uint32_t, 128> bq = xesimd::lsc_load_2d<uint32_t, 8, 16, 1, true, false, cache_hint::cached,
                                                                 cache_hint::cached>(qb32, kW, kH, kP, kc * 8, r0 + t * 16);
                    simd<fp16, 128> a = hA.template select<128, 1>(kc * 128);
                    ar = xmx::dpas<8, 8, float, float, fp16, fp16>(ar, bk.template bit_cast_view<fp16>().read(), a);
                    aq = xmx::dpas<8, 8, float, float, fp16, fp16>(aq, bq.template bit_cast_view<fp16>().read(), a);
                }
                RT.template select<128, 1>(t * 128) = ar;
                QT.template select<128, 1>(t * 128) = aq;
            }
            // ---- x = V - Gamma R, [64 i][8 v]
            simd<float, 64> g = block_load<float, 64>(gam + ch * CK);
            simd<float, 64> w = block_load<float, 64>(wl + ch * CK);
            simd<float, 64> bb = block_load<float, 64>(bt + ch * CK);
            simd<float, 512> X;
#pragma unroll
            for (int h2 = 0; h2 < 2; h2++) {
                // V rows r0 + 32 h2 .. + 31, columns of this thread: [32 i][8 v]
                simd<uint16_t, 256> vv = xesimd::lsc_load_2d<uint16_t, 8, 32, 1, false, false, cache_hint::cached,
                                                             cache_hint::cached>(cv, cvW, cvH, cvP,
                                                                                 2 * GK * DK + j * DV + vb * VB, r0 + 32 * h2);
#pragma unroll
                for (int q16 = 0; q16 < 16; q16++)
                    X.template select<16, 1>(h2 * 256 + q16 * 16) = bf16_to_f32(vv.template select<16, 1>(q16 * 16));
            }
#pragma unroll
            for (int t = 0; t < 4; t++) {
                simd<float, 16> gt = g.template select<16, 1>(t * 16);
#pragma unroll
                for (int v = 0; v < 8; v++) {
                    auto xs = X.template select<16, 8>(t * 128 + v);
                    xs = simd<float, 16>(xs) - gt * simd<float, 16>(RT.template select<16, 1>(t * 128 + v * 16));
                }
            }
            // ---- Delta_i = beta_i x_i - sum_{m < i} A[i][m] Delta_m (fp32), [64 i][8 v]; rows not yet made stay zero
            simd<float, 512> Dl = 0.0f;
            barrier();                        // A is in SLM
#pragma unroll
            for (int i = 0; i < CK; i++) {
                simd<float, 8> acc = simd<float, 8>(X.template select<8, 1>(i * 8)) * (float)bb[i];
                if (i > 0) {
                    const int nb = (i + 7) / 8;
                    simd<float, 64> arow = slm_block_load<float, 64>((uint32_t)(i * CK * 4));
                    simd<float, 64> sum = 0.0f;
#pragma unroll
                    for (int mb = 0; mb < 8; mb++) {
                        if (mb >= nb) break;
                        simd<float, 64> coef = arow.template replicate_vs_w_hs<8, 1, 8, 0>(mb * 8);
                        sum += coef * simd<float, 64>(Dl.template select<64, 1>(mb * 64));
                    }
                    simd<float, 32> s32 = simd<float, 32>(sum.template select<32, 1>(0)) + simd<float, 32>(sum.template select<32, 1>(32));
                    simd<float, 16> s16 = simd<float, 16>(s32.template select<16, 1>(0)) + simd<float, 16>(s32.template select<16, 1>(16));
                    acc -= simd<float, 8>(s16.template select<8, 1>(0)) + simd<float, 8>(s16.template select<8, 1>(8));
                }
                Dl.template select<8, 1>(i * 8) = acc;
            }
            // Delta^T as DPAS A tiles [t][8 v][16 i], plain and weighted by exp(c_63 - c_i)
            simd<fp16, 4 * 128> DA, WA;
#pragma unroll
            for (int t = 0; t < 4; t++) {
                simd<float, 16> wt = w.template select<16, 1>(t * 16);
#pragma unroll
                for (int v = 0; v < 8; v++) {
                    simd<float, 16> d = Dl.template select<16, 8>(t * 128 + v);
                    DA.template select<16, 1>(t * 128 + v * 16) = convert<fp16>(d);
                    WA.template select<16, 1>(t * 128 + v * 16) = convert<fp16>(d * wt);
                }
            }
            const uint32_t* mq32 = reinterpret_cast<const uint32_t*>(Mq + ch * CK * CK);
            // ---- O^T = Gamma Q^T + Delta^T Mq^T, out as bf16 rows
#pragma unroll
            for (int t = 0; t < 4; t++) {
                simd<float, 16> gt = g.template select<16, 1>(t * 16);
                simd<float, 128> acc;
#pragma unroll
                for (int v = 0; v < 8; v++)
                    acc.template select<16, 1>(v * 16) = simd<float, 16>(QT.template select<16, 1>(t * 128 + v * 16)) * gt;
#pragma unroll
                for (int jc = 0; jc < 4; jc++) {
                    if (jc > t) break;
                    simd<uint32_t, 128> b = xesimd::lsc_load_2d<uint32_t, 8, 16, 1, true, false, cache_hint::cached,
                                                                cache_hint::cached>(mq32, CK * 2 - 1, CK - 1, CK * 2 - 1, jc * 8, t * 16);
                    acc = xmx::dpas<8, 8, float, float, fp16, fp16>(acc, b.template bit_cast_view<fp16>().read(),
                                                                    simd<fp16, 128>(DA.template select<128, 1>(jc * 128)));
                }
                // [8 v][16 i] -> rows i of 8 v, two stores of 8 rows
                simd<uint16_t, 128> ob = f32_to_bf16<128>(acc);
#pragma unroll
                for (int hh = 0; hh < 2; hh++) {
                    simd<uint16_t, 64> tile;
#pragma unroll
                    for (int v = 0; v < 8; v++) tile.template select<8, 8>(v) = ob.template select<8, 1>(v * 16 + hh * 8);
                    xesimd::lsc_store_2d<uint16_t, 8, 8>(core, oW, oH, oP, j * DV + vb * VB, r0 + t * 16 + hh * 8, tile);
                }
            }
            // ---- S1^T = exp(c_63) S0^T + (w Delta)^T K
            const float glast = g[CK - 1];
#pragma unroll
            for (int kt = 0; kt < 8; kt++) {
                simd<float, 128> acc = simd<float, 128>(hT.template select<128, 1>(kt * 128)) * glast;
#pragma unroll
                for (int jc = 0; jc < 4; jc++) {
                    simd<fp16, 256> b = xesimd::lsc_load_2d<fp16, 16, 16, 1, false, true, cache_hint::cached,
                                                            cache_hint::cached>(kb_, kW, kH, kP, kt * 16, r0 + jc * 16);
                    acc = xmx::dpas<8, 8, float, float, fp16, fp16>(acc, b, simd<fp16, 128>(WA.template select<128, 1>(jc * 128)));
                }
                hT.template select<128, 1>(kt * 128) = acc;
            }
        }

        // ---- state out: rows k, columns vb * 8 .. + 7
#pragma unroll
        for (int kc = 0; kc < 8; kc++) {
#pragma unroll
            for (int hh = 0; hh < 2; hh++) {
                simd<float, 64> tile;     // [8 k][8 v]
#pragma unroll
                for (int v = 0; v < 8; v++) tile.template select<8, 8>(v) = hT.template select<8, 1>(kc * 128 + v * 16 + hh * 8);
                xesimd::lsc_store_2d<float, 8, 8>(rp, DV * 4 - 1, DK - 1, DV * 4 - 1, vb * VB, kc * 16 + hh * 8, tile);
            }
        }
    }
};

}  // namespace q35gdn
