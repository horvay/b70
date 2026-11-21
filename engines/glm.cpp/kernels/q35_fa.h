#pragma once
// Flash attention for Qwen3.5 / Qwen3.8 prompt rows (ESIMD, DPAS; head dim 256, GQA 24 / 4). kernels/g4_fa.h's
// structure: a thread = 8 query rows x 1 head, 256 registers, Q and O in registers, base-2 online softmax over blocks of
// 64 keys, P in fp16 into the P V DPAS. Changes for this shape and card:
//   - K^T tiles. A transposed 2D dword load (g4_fa.h's DPAS B for Q K^T) costs ~2.5x a VNNI load per byte here: at
//     30K it took 23 of a layer's 36 ms (q35_gen fabench, dbg 1). With KT2 they come by plain loads from kt, a copy of
//     the keys with each key's d pairs side by side, [Hk][D / 2][TS][2] (q35ops::kt_fill), four d-chunks a load.
//     Plain loads transposed in registers instead were slower still (strided register moves).
//   - V tiles. VT2: plain loads from vt, the values with key pairs interleaved, [Hk][TS / 2][D][2] (q35ops::vt_fill);
//     otherwise VNNI loads of 32 keys x 16 dims.
//   - lazy rescale: a row's reference max moves only when a block's max passes it by more than 8 (log2 units), so O
//     (128 registers) is rescaled a few times a row instead of every block; P stays below 2^8, inside fp16. The same
//     function: a row's P and l always share one reference.
// Tried and dropped: two threads a head (D split, scores through SLM), 32-key blocks, prefetching the next block.
// Query row i sits at position qbase + i and sees keys 0 .. its position. Output fp16 [Lq, 24, 256], before the gate.

#include <cmath>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>

namespace q35fa {

using namespace sycl::ext::intel::esimd;
namespace xesimd = sycl::ext::intel::experimental::esimd;
using fp16 = sycl::half;
using xesimd::cache_hint;

constexpr int D = 256, NC = D / 16, QR = 8, KT = 16, KBT = 4, KB = KT * KBT;
constexpr float LOG2E = 1.4426950408889634f;
constexpr float SM = LOG2E * 0.0625f;          // log2 e / sqrt(256)
constexpr float LAZY = 8.0f;                   // rescale only past 2^8

// DBG (timing experiments only, wrong results): 1 no K loads, 2 no V loads, 4 no softmax, 8 no P V, 16 no Q K^T
template <bool KT2, bool VT2, int DBG = 0>
struct Q35Fa {
    const fp16* q; const fp16* k; const fp16* v;   // q [Lq, Hq, D], k / v [Lk, Hk, D]
    fp16* o;
    int Lq, Lk, Hq, Hk, G, qbase, nqt;
    const fp16* kt = nullptr;                      // KT2: [Hk][D / 2][TS][2]
    const fp16* vt = nullptr;                      // VT2: [Hk][TS / 2][D][2]
    int TS = 0;

    void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
        const int id = it.get_global_id(0);
        const int g = id % G, rest = id / G;
        const int qt = rest % nqt, kvh = rest / nqt;
        if (kvh >= Hk) return;
        const int q0 = qt * QR, h = kvh * G + g;
        const int nrows = Lq - q0 < QR ? Lq - q0 : QR;

        simd<fp16, NC * QR * 16> Qc = 0;          // [chunk][row][16]
        const fp16* qp = q + ((size_t)q0 * Hq + h) * D;
#pragma unroll
        for (int r = 0; r < QR; ++r)
            if (r < nrows) {
#pragma unroll
                for (int c = 0; c < NC; ++c)
                    Qc.template select<16, 1>((c * QR + r) * 16) = block_load<fp16, 16>(qp + (size_t)r * Hq * D + c * 16);
            }

        simd<float, NC * QR * 16> O = 0.0f;
        simd<float, QR> m = -INFINITY, l = 0.0f;

        const int qmin = qbase + q0, qmax = qmin + nrows - 1;
        const int kend = qmax + 1 < Lk ? qmax + 1 : Lk;
        const uint32_t* kb32 = reinterpret_cast<const uint32_t*>(k + (size_t)kvh * D);
        const fp16* vbase = v + (size_t)kvh * D;
        const unsigned surfW = D * 2 - 1, surfH = (unsigned)Lk - 1, pitch = (unsigned)Hk * D * 2 - 1;
        const fp16* ktb = KT2 ? kt + (size_t)kvh * (D / 2) * TS * 2 : nullptr;
        const unsigned tW = (unsigned)TS * 4 - 1, tH = D / 2 - 1;
        const fp16* vtb = VT2 ? vt + (size_t)kvh * (TS / 2) * D * 2 : nullptr;
        // rows past the last key pair read as zeros (stale bits there could be NaN, and 0 * NaN is NaN)
        const unsigned vW = D * 4 - 1, vH = (unsigned)(Lk + 1) / 2 - 1;
        simd<uint32_t, 8 * KT> kfix;
        simd<fp16, 32 * 16> vfix;
        if constexpr ((DBG & 3) != 0) {
            kfix = xesimd::lsc_load_2d<uint32_t, 8, KT, 1, true, false>(kb32, surfW, surfH, pitch, 0, 0);
            vfix = xesimd::lsc_load_2d<fp16, 16, 32, 1, false, true>(vbase, surfW, surfH, pitch, 0, 0);
        }

        for (int kb0 = 0; kb0 < kend; kb0 += KB) {
            // ---- S = Q K^T * scale, [t][8 rows][16 keys]
            simd<float, KBT * QR * KT> S;
            if constexpr (KT2) {
                S = 0.0f;
#pragma unroll
                for (int c4 = 0; c4 < NC / 4; ++c4) {
#pragma unroll
                    for (int t = 0; t < KBT; ++t) {
                        // [32 d pairs][16 keys][2]: four DPAS B tiles (d-chunks 4 c4 .. 4 c4 + 3) of key tile t
                        simd<fp16, 32 * 32> b4 = xesimd::lsc_load_2d<fp16, 32, 32, 1, false, false, cache_hint::cached,
                                                                     cache_hint::cached>(ktb, tW, tH, tW, (kb0 + t * KT) * 2, c4 * 32);
                        auto st = S.template select<QR * KT, 1>(t * QR * KT);
                        simd<float, QR * KT> acc = st;
#pragma unroll
                        for (int cc = 0; cc < 4; ++cc)
                            acc = xmx::dpas<8, QR, float, float, fp16, fp16>(acc, simd<fp16, 256>(b4.template select<256, 1>(cc * 256)),
                                                                             simd<fp16, QR * 16>(Qc.template select<QR * 16, 1>((4 * c4 + cc) * QR * 16)));
                        st = acc;
                    }
                }
                S *= SM;
            } else {
#pragma unroll
                for (int t = 0; t < KBT; ++t) {
                    simd<float, QR * KT> acc = 0.0f;
                    if constexpr (!(DBG & 16)) {
#pragma unroll
                        for (int c = 0; c < NC; ++c) {
                            simd<uint32_t, 8 * KT> kt2;
                            if constexpr ((DBG & 1) != 0) kt2 = kfix + (uint32_t)c;
                            else kt2 = xesimd::lsc_load_2d<uint32_t, 8, KT, 1, true, false, cache_hint::cached,
                                                           cache_hint::cached>(kb32, surfW, surfH, pitch, c * 8, kb0 + t * KT);
                            acc = xmx::dpas<8, QR, float, float, fp16, fp16>(acc, kt2.template bit_cast_view<fp16>().read(),
                                                                             simd<fp16, QR * 16>(Qc.template select<QR * 16, 1>(c * QR * 16)));
                        }
                    } else {
                        acc = (float)kb0;
                    }
                    S.template select<QR * KT, 1>(t * QR * KT) = acc * SM;
                }
            }
            // ---- P = exp2(S - reference), fp16
            simd<fp16, KBT * QR * KT> P;
            if constexpr ((DBG & 4) != 0) {
                P = convert<fp16>(S);
                l += 1.0f;
            } else {
                // keys past Lk or after a row
                if (kb0 + KB > Lk || kb0 + KB - 1 > qmin) {
                    simd<int, KT> lane(0, 1);
#pragma unroll
                    for (int t = 0; t < KBT; ++t)
#pragma unroll
                        for (int r = 0; r < QR; ++r) {
                            simd<int, KT> kj = lane + (kb0 + t * KT);
                            simd<float, KT> sv = S.template select<KT, 1>((t * QR + r) * KT);
                            sv.merge(simd<float, KT>(-INFINITY), kj >= Lk);
                            sv.merge(simd<float, KT>(-INFINITY), kj > qmin + r);
                            S.template select<KT, 1>((t * QR + r) * KT) = sv;
                        }
                }
                simd<float, QR> mb;
#pragma unroll
                for (int r = 0; r < QR; ++r) {
                    simd<float, KT> mx = S.template select<KT, 1>(r * KT);
#pragma unroll
                    for (int t = 1; t < KBT; ++t) mx = max(mx, simd<float, KT>(S.template select<KT, 1>((t * QR + r) * KT)));
                    mb[r] = hmax<float>(mx);
                }
                // lazy reference: move it (and rescale) only where the block's max passes it by more than LAZY
                simd_mask<QR> up = mb > m + LAZY;
                if (up.any()) {
                    simd<float, QR> m_new = m;
                    m_new.merge(mb, up);
                    simd<float, QR> alpha = exp2(m - m_new);
                    alpha.merge(simd<float, QR>(0.0f), m == simd<float, QR>(-INFINITY));
                    alpha.merge(simd<float, QR>(1.0f), !up);
                    l = l * alpha;
                    m = m_new;
                    simd<float, QR * 16> arep = alpha.template replicate_vs_w_hs<QR, 1, 16, 0>(0);
#pragma unroll
                    for (int c = 0; c < NC; ++c) {
                        auto oc = O.template select<QR * 16, 1>(c * QR * 16);
                        oc = simd<float, QR * 16>(oc) * arep;
                    }
                }
                simd<float, QR> m_use = m;
                m_use.merge(simd<float, QR>(0.0f), m == simd<float, QR>(-INFINITY));
                simd<float, QR * KT> mrep = m_use.template replicate_vs_w_hs<QR, 1, KT, 0>(0);
                simd<float, QR> rs = 0.0f;
#pragma unroll
                for (int t = 0; t < KBT; ++t) {
                    simd<float, QR * KT> sv = S.template select<QR * KT, 1>(t * QR * KT);
                    simd<float, QR * KT> p = exp2(sv - mrep);
                    p.merge(simd<float, QR * KT>(0.0f), sv == simd<float, QR * KT>(-INFINITY));
#pragma unroll
                    for (int r = 0; r < QR; ++r) rs[r] += reduce<float>(simd<float, KT>(p.template select<KT, 1>(r * KT)), std::plus<>());
                    P.template select<QR * KT, 1>(t * QR * KT) = convert<fp16>(p);
                }
                l += rs;
            }
            // ---- O += P V
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                simd<float, QR * 16> acc = O.template select<QR * 16, 1>(c * QR * 16);
#pragma unroll
                for (int t2 = 0; t2 < KBT / 2; ++t2) {
                    // two key tiles of d-chunk c: [16 key pairs][16 d][2]
                    simd<fp16, 32 * 16> vv;
                    if constexpr ((DBG & 2) != 0) vv = vfix;
                    else if constexpr (VT2) vv = xesimd::lsc_load_2d<fp16, 32, 16, 1, false, false, cache_hint::cached,
                                                                     cache_hint::cached>(vtb, vW, vH, vW, c * 32, kb0 / 2 + t2 * 16);
                    else vv = xesimd::lsc_load_2d<fp16, 16, 32, 1, false, true, cache_hint::cached,
                                                  cache_hint::cached>(vbase, surfW, surfH, pitch, c * 16, kb0 + t2 * 32);
                    if constexpr ((DBG & 8) != 0) {
                        acc += simd<float, QR * 16>(vv.template select<128, 1>(0));
                    } else {
#pragma unroll
                        for (int th = 0; th < 2; ++th)
                            acc = xmx::dpas<8, QR, float, float, fp16, fp16>(acc, simd<fp16, 256>(vv.template select<256, 1>(th * 256)),
                                                                             simd<fp16, QR * KT>(P.template select<QR * KT, 1>((2 * t2 + th) * QR * KT)));
                    }
                }
                O.template select<QR * 16, 1>(c * QR * 16) = acc;
            }
        }

        simd<float, QR> inv = 1.0f / l;
        inv.merge(simd<float, QR>(0.0f), l <= simd<float, QR>(0.0f));
#pragma unroll
        for (int r = 0; r < QR; ++r) {
            if (r >= nrows) break;
            fp16* op = o + ((size_t)(q0 + r) * Hq + h) * D;
            const float ir = inv[r];
#pragma unroll
            for (int c = 0; c < NC; ++c)
                block_store<fp16, 16>(op + c * 16, convert<fp16>(simd<float, 16>(O.template select<16, 1>((c * QR + r) * 16)) * ir));
        }
    }
};

}  // namespace q35fa
