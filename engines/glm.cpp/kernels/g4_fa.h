#pragma once
// Flash attention for a prompt's rows on Xe2 (ESIMD, DPAS), after fa_esimd.h: fp16 Q / K / V, GQA, any head dim the
// registers hold. Query row i sits at position qbase + i and key j at kbase + j; a row sees keys at or before its
// position, and within the last win positions when win > 0, so the same kernel serves the sliding layers (keys:
// the window behind the rows, then the rows) and the full layers (a block of keys at a time, merged by log-sum).
//
// Thread = QR query rows x 1 query head, 256-register mode:
//   Qc  [NC d-chunks][QR rows][16]  fp16  (DPAS A operands are free register slices)
//   O   [NC d-chunks][QR rows][16]  fp32  accumulator
// Per 32-key block:
//   S = Q K^T : 2 key tiles x NC d-chunks DPAS, B = K^T chunk from a transposed 2D dword load (= VNNI)
//   online softmax in base 2, rescale O once per block
//   O += P V  : NC d-chunks x 2 key tiles DPAS, B = V chunk from a VNNI-transformed 2D load
// D = 256 runs one thread per head. D = 512 runs PAIR: O for 8 rows would need all 256 registers, so two threads share a
// head, each owning half the dims: its half of Q, O and every K / V load (each key byte is read once), its partial
// scores exchanged through SLM once per block. Two threads per 8 rows beat one per 4: +17% whole-prompt at 32K.
// Work-groups hold neighbouring query tiles of one KV head, so K / V hit in L1.

#include <cmath>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>

namespace g4fa {

using namespace sycl::ext::intel::esimd;
namespace xesimd = sycl::ext::intel::experimental::esimd;
using fp16 = sycl::half;

constexpr int KT = 16;          // keys per DPAS tile
constexpr float LOG2E = 1.4426950408889634f;

// MERGE: o32 / lse hold the running result of the earlier key blocks (fp32, natural-log LSE); else fp16 o, normalised.
// SM: log2 e times the attention scale
template <int D, int QR, bool MERGE, bool PAIR, int BT, float SM = LOG2E>
struct FaKernel {
    static constexpr int KBT = BT;                  // key tiles per softmax block
    static constexpr int KB = KT * KBT;             // keys per softmax block
    static constexpr int NC = PAIR ? D / 32 : D / 16;   // d-chunks a thread owns
    static constexpr int SSZ = KBT * QR * KT * 4;   // bytes of a thread's partial scores in SLM
    const fp16* q; const fp16* k; const fp16* v;    // q [Lq, Hq, D], k / v [Lk, Hk, D]
    fp16* o;                                        // MERGE false: [Lq, Hq, D] (may alias q)
    float* o32; float* lse;                         // MERGE true: [Lq, Hq, D], [Lq, Hq]
    int Lq, Lk, Hq, Hk, G;                          // G = Hq / Hk
    int qbase, kbase, win, nqt;                     // nqt = query tiles

    void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
        int side = 0, lid = 0;
        int id = it.get_global_id(0);
        if constexpr (PAIR) {
            slm_init<8 * 2 * SSZ>();                 // 8 threads x 2 slots: a slot is reused every other block, by then read
            lid = it.get_local_id(0);
            side = lid & 1;
            id /= 2;
        }
        const int c0 = side * NC;                    // the first d-chunk this thread owns
        int g = id % G;
        int rest = id / G;
        int qt = rest % nqt;
        int kvh = rest / nqt;
        if (kvh >= Hk) return;                       // rounded-up range
        int q0 = qt * QR;
        int h = kvh * G + g;
        int nrows = Lq - q0 < QR ? Lq - q0 : QR;

        // ---- Q tile, chunk-major; pad rows are zero
        simd<fp16, NC * QR * 16> Qc = 0;
        const fp16* qp = q + ((size_t)q0 * Hq + h) * D + c0 * 16;
#pragma unroll
        for (int r = 0; r < QR; ++r) {
            if (r < nrows) {
#pragma unroll
                for (int c = 0; c < NC; ++c)
                    Qc.template select<16, 1>((c * QR + r) * 16) =
                        block_load<fp16, 16>(qp + (size_t)r * Hq * D + c * 16);
            }
        }

        simd<float, NC * QR * 16> O = 0.0f;
        simd<float, QR> m = -INFINITY, l = 0.0f;

        // keys of the tile: rows' positions qmin .. qmax
        int qmin = qbase + q0, qmax = qmin + nrows - 1;
        int kst = 0;
        if (win > 0 && qmin - win + 1 > kbase) kst = (qmin - win + 1 - kbase) / KB * KB;
        int kend = qmax - kbase + 1 < Lk ? qmax - kbase + 1 : Lk;

        // 2D surfaces for this KV head: rows = keys, row pitch = Hk * D halves
        const uint32_t* kb32 = reinterpret_cast<const uint32_t*>(k + (size_t)kvh * D);
        const fp16* vbase = v + (size_t)kvh * D;
        unsigned surfW = D * 2 - 1;                  // bytes - 1
        unsigned surfH = (unsigned)Lk - 1;           // rows - 1
        unsigned pitch = (unsigned)Hk * D * 2 - 1;   // bytes - 1

        int slot = 0;
        for (int kb0 = kst; kb0 < kend; kb0 += KB) {
            simd<float, KBT * QR * KT> S;
#pragma unroll
            for (int t = 0; t < KBT; ++t) {
                simd<float, QR * KT> acc = 0.0f;
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    // K^T chunk: 8 dword columns (16 d) x 16 keys, transposed -> [8 dpair][16 key] = VNNI B
                    simd<uint32_t, 8 * KT> kt = xesimd::lsc_load_2d<uint32_t, 8, KT, 1, true, false,
                        xesimd::cache_hint::cached, xesimd::cache_hint::cached>(
                        kb32, surfW, surfH, pitch, (c + c0) * 8, kb0 + t * KT);
                    acc = xmx::dpas<8, QR, float, float, fp16, fp16>(
                        acc, kt.template bit_cast_view<fp16>().read(),
                        simd<fp16, QR * 16>(Qc.template select<QR * 16, 1>(c * QR * 16)));
                }
                S.template select<QR * KT, 1>(t * QR * KT) = acc * SM;
            }
            if constexpr (PAIR) {
                // scores = mine + the other half's
                uint32_t mine = (uint32_t)(lid * 2 + slot) * SSZ, other = (uint32_t)((lid ^ 1) * 2 + slot) * SSZ;
#pragma unroll
                for (int t = 0; t < KBT; ++t)
                    slm_block_store<float, QR * KT>(mine + t * QR * KT * 4, S.template select<QR * KT, 1>(t * QR * KT));
                barrier();
#pragma unroll
                for (int t = 0; t < KBT; ++t) {
                    simd<float, QR * KT> theirs = slm_block_load<float, QR * KT>(other + t * QR * KT * 4);
                    S.template select<QR * KT, 1>(t * QR * KT) = simd<float, QR * KT>(S.template select<QR * KT, 1>(t * QR * KT)) + theirs;
                }
                slot ^= 1;
            }
            // masking: keys past Lk, keys after a row, keys before a row's window
            int kp0 = kbase + kb0;                    // the block's first key position
            bool need_mask = (kb0 + KB > Lk) || (kp0 + KB - 1 > qmin) || (win > 0 && kp0 < qmax - win + 1);
            if (need_mask) {
                simd<int, KT> lane(0, 1);
#pragma unroll
                for (int t = 0; t < KBT; ++t)
#pragma unroll
                    for (int r = 0; r < QR; ++r) {
                        simd<int, KT> kj = lane + (kb0 + t * KT);
                        simd<int, KT> kpos = kj + kbase;
                        int qp_r = qmin + r;
                        simd<float, KT> sv = S.template select<KT, 1>((t * QR + r) * KT);
                        sv.merge(simd<float, KT>(-INFINITY), kj >= Lk);
                        sv.merge(simd<float, KT>(-INFINITY), kpos > qp_r);
                        if (win > 0) sv.merge(simd<float, KT>(-INFINITY), kpos < qp_r - win + 1);
                        S.template select<KT, 1>((t * QR + r) * KT) = sv;
                    }
            }
            // online softmax (base 2). A row with nothing visible yet keeps m = -inf and adds nothing.
            simd<float, QR> mb;
#pragma unroll
            for (int r = 0; r < QR; ++r) {
                simd<float, KT> mx = S.template select<KT, 1>(r * KT);
#pragma unroll
                for (int t = 1; t < KBT; ++t) mx = max(mx, simd<float, KT>(S.template select<KT, 1>((t * QR + r) * KT)));
                mb[r] = hmax<float>(mx);
            }
            simd<float, QR> m_new = max(m, mb);
            simd<float, QR> m_use = m_new;
            m_use.merge(simd<float, QR>(0.0f), m_new == simd<float, QR>(-INFINITY));
            simd<float, QR> alpha = exp2(m - m_use);
            alpha.merge(simd<float, QR>(0.0f), m == simd<float, QR>(-INFINITY));
            simd<float, QR * KT> mrep = m_use.template replicate_vs_w_hs<QR, 1, KT, 0>(0);
            simd<float, QR> rs = 0.0f;
            simd<fp16, KBT * QR * KT> P;
#pragma unroll
            for (int t = 0; t < KBT; ++t) {
                simd<float, QR * KT> sv = S.template select<QR * KT, 1>(t * QR * KT);
                simd<float, QR * KT> p = exp2(sv - mrep);
                p.merge(simd<float, QR * KT>(0.0f), sv == simd<float, QR * KT>(-INFINITY));
#pragma unroll
                for (int r = 0; r < QR; ++r) rs[r] += reduce<float>(simd<float, KT>(p.template select<KT, 1>(r * KT)), std::plus<>());
                P.template select<QR * KT, 1>(t * QR * KT) = convert<fp16>(p);
            }
            l = l * alpha + rs;
            m = m_new;
            simd<float, QR * 16> arep = alpha.template replicate_vs_w_hs<QR, 1, 16, 0>(0);
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                auto oc = O.template select<QR * 16, 1>(c * QR * 16);
                simd<float, QR * 16> acc = simd<float, QR * 16>(oc) * arep;
#pragma unroll
                for (int t = 0; t < KBT; ++t) {
                    // V chunk: 16 keys x 16 d, VNNI-transformed -> [8 keypair][16 d][2]
                    simd<fp16, KT * 16> vt = xesimd::lsc_load_2d<fp16, 16, KT, 1, false, true,
                        xesimd::cache_hint::cached, xesimd::cache_hint::cached>(
                        vbase, surfW, surfH, pitch, (c + c0) * 16, kb0 + t * KT);
                    acc = xmx::dpas<8, QR, float, float, fp16, fp16>(
                        acc, vt, simd<fp16, QR * 16>(P.template select<QR * KT, 1>(t * QR * KT)));
                }
                oc = acc;
            }
        }

        // ---- normalise + store
        simd<float, QR> inv = 1.0f / l;
        inv.merge(simd<float, QR>(0.0f), l <= simd<float, QR>(0.0f));
        if constexpr (MERGE) {
            // this block's result o_b = O inv, log-sum (m + log2 l) / log2 e; merged with the running (o32, lse)
            simd<float, QR> lse_b = (m + log2(l)) * (1.0f / LOG2E);
            lse_b.merge(simd<float, QR>(-INFINITY), l <= simd<float, QR>(0.0f));
#pragma unroll
            for (int r = 0; r < QR; ++r) {
                if (r >= nrows) break;
                size_t rh = (size_t)(q0 + r) * Hq + h;
                float la = lse[rh], lb = lse_b[r];
                if constexpr (PAIR) barrier();                     // both read the running lse before side 0 writes it
                float mx = la > lb ? la : lb;
                if (lb == -INFINITY) continue;                     // nothing visible in this block
                float wa = la == -INFINITY ? 0.0f : exp2(simd<float, 1>((la - mx) * LOG2E))[0];
                float wb = exp2(simd<float, 1>((lb - mx) * LOG2E))[0];
                float den = 1.0f / (wa + wb);
                float ib = inv[r] * wb * den, ia = wa * den;
                float* op = o32 + rh * D + c0 * 16;
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    simd<float, 16> cur = simd<float, 16>(O.template select<16, 1>((c * QR + r) * 16)) * ib;
                    if (la != -INFINITY) cur += block_load<float, 16>(op + c * 16) * ia;
                    block_store<float, 16>(op + c * 16, cur);
                }
                // log-sum of the two
                if (side == 0) lse[rh] = mx + log2(simd<float, 1>(wa + wb))[0] * (1.0f / LOG2E);
            }
        } else {
#pragma unroll
            for (int r = 0; r < QR; ++r) {
                if (r >= nrows) break;
                fp16* op = o + ((size_t)(q0 + r) * Hq + h) * D + c0 * 16;
                float ir = inv[r];
#pragma unroll
                for (int c = 0; c < NC; ++c)
                    block_store<fp16, 16>(op + c * 16, convert<fp16>(simd<float, 16>(O.template select<16, 1>((c * QR + r) * 16)) * ir));
            }
        }
    }
};

}  // namespace g4fa
