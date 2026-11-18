#pragma once
// Decode / verify attention for Gemma 4 (ESIMD, DPAS): a row's query heads over its keys, split into pieces of keys whose
// partial results (m, l, O: online softmax in base 2, unnormalised) a plain kernel combines. Memory is the limit here:
// 2D block loads read each key row whole, and query heads sharing a KV head run in one thread's DPAS rows.
//
// Sliding: a thread = 1 row x the 2 query heads of a KV head x 3 blocks of 32 ring slots. A slot holds the last position
// written to it; it is visible when that is at or after position 0 and within the window behind the row's position.
//
// Full: a thread pair = 1 row x the 8 query heads of a KV head x a piece of the keys. O for 8 heads over all 512 dims would
// not fit the 256 registers, so each thread owns half the dims and reads only that half of the keys (no key byte
// twice), the partial scores exchanged through SLM once per block. The cache has P (raw k_proj) and Kr = rope(k_norm *
// P), the 128 rotated dims. So q . K = r (q . Kr over the rotated dims + (q * k_norm) . P over the others), V = r P,
// with r = 1 / rms(P).

#include <cmath>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>

namespace g4dec {

using namespace sycl::ext::intel::esimd;
namespace xesimd = sycl::ext::intel::experimental::esimd;
using fp16 = sycl::half;

constexpr int KT = 16, KBT = 2, KB = KT * KBT;
constexpr float LOG2E = 1.4426950408889634f;
constexpr int S_NH = 32, S_KV = 16, S_D = 256, S_G = 2;      // sliding
constexpr int F_NH = 32, F_KV = 4, F_D = 512, F_G = 8;       // full
constexpr int F_ROT = 64;                                    // rotated pairs (j, j + 256), j < 64

// One softmax block: S [KBT * QR * KT] (scores in base 2, masked -inf) -> m, l updated, P fp16 probabilities times the
// keys' r (rm[t * KT + key], when given); returns alpha, the scale of O so far
template <int QR>
ESIMD_INLINE simd<float, QR> softmax_block(simd<float, KBT * QR * KT>& S, simd<float, QR>& m, simd<float, QR>& l,
                                           simd<fp16, KBT * QR * KT>& P, const simd<float, KB>* rm) {
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
#pragma unroll
    for (int t = 0; t < KBT; ++t) {
        simd<float, QR * KT> sv = S.template select<QR * KT, 1>(t * QR * KT);
        simd<float, QR * KT> p = exp2(sv - mrep);
        p.merge(simd<float, QR * KT>(0.0f), sv == simd<float, QR * KT>(-INFINITY));
#pragma unroll
        for (int r = 0; r < QR; ++r) {
            simd<float, KT> pr = p.template select<KT, 1>(r * KT);
            rs[r] += reduce<float>(pr, std::plus<>());
            if (rm) { simd<float, KB> rc = *rm; pr *= simd<float, KT>(rc.template select<KT, 1>(t * KT)); }
            P.template select<KT, 1>((t * QR + r) * KT) = convert<fp16>(pr);
        }
    }
    l = l * alpha + rs;
    m = m_new;
    return alpha;
}

struct SlidingDec {
    static constexpr int QR = S_G, NC = S_D / 16;
    const fp16* q; const fp16* k; const fp16* v;      // q [R, 32, 256], rings [RING, 16, 256]
    float* pm; float* pl; float* po;                  // [R * 32 * NS], [same], [R * 32 * NS * 256]
    const int* pdev; const int* sdev;                 // each row's position, game (null: game 0; a game's ring: ring rows)
    int p0, R, ring, win, ns, blocks;                 // ns pieces of `blocks` blocks of slots

    void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
        int id = it.get_global_id(0);
        int s = id % ns;
        int rest = id / ns;
        int kvh = rest % S_KV;
        int r = rest / S_KV;
        if (r >= R) return;
        int p = p0 + r;
        if (pdev) { simd<int, 1> pv = block_load<int, 1>(pdev + r); p = pv[0]; }
        int ps = p % ring;
        size_t gbase = 0;
        if (sdev) { simd<int, 1> sv = block_load<int, 1>(sdev + r); gbase = (size_t)sv[0] * ring * S_KV * S_D; }

        simd<fp16, NC * QR * 16> Qc;
        const fp16* qp = q + ((size_t)r * S_NH + kvh * S_G) * S_D;
#pragma unroll
        for (int h = 0; h < QR; ++h)
#pragma unroll
            for (int c = 0; c < NC; ++c)
                Qc.template select<16, 1>((c * QR + h) * 16) = block_load<fp16, 16>(qp + h * S_D + c * 16);

        simd<float, NC * QR * 16> O = 0.0f;
        simd<float, QR> m = -INFINITY, l = 0.0f;

        const uint32_t* kb32 = reinterpret_cast<const uint32_t*>(k + gbase + (size_t)kvh * S_D);
        const fp16* vbase = v + gbase + (size_t)kvh * S_D;
        unsigned surfW = S_D * 2 - 1, surfH = (unsigned)ring - 1, pitch = (unsigned)S_KV * S_D * 2 - 1;

        for (int b = s * blocks; b < (s + 1) * blocks && b * KB < ring; ++b) {
            int s0 = b * KB;
            // distance behind the row of each slot in the block; beyond the window: not visible
            simd<int, KB> slots(s0, 1);
            simd<int, KB> dist = ps - slots;
            dist.merge(dist + ring, dist < 0);
            // (before the window fills, or after a reset: a slot further back than the row's position holds nothing of
            // this sequence)
            simd_mask<KB> hide = (dist >= win) | (dist > p);
            if (hide.all()) continue;
            simd<float, KBT * QR * KT> S;
#pragma unroll
            for (int t = 0; t < KBT; ++t) {
                simd<float, QR * KT> acc = 0.0f;
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    simd<uint32_t, 8 * KT> kt = xesimd::lsc_load_2d<uint32_t, 8, KT, 1, true, false,
                        xesimd::cache_hint::cached, xesimd::cache_hint::cached>(kb32, surfW, surfH, pitch, c * 8, s0 + t * KT);
                    acc = xmx::dpas<8, QR, float, float, fp16, fp16>(
                        acc, kt.template bit_cast_view<fp16>().read(),
                        simd<fp16, QR * 16>(Qc.template select<QR * 16, 1>(c * QR * 16)));
                }
                acc *= LOG2E;
#pragma unroll
                for (int h = 0; h < QR; ++h) {
                    simd<float, KT> sv = acc.template select<KT, 1>(h * KT);
                    simd<int, KT> dt = dist.template select<KT, 1>(t * KT);
                    sv.merge(simd<float, KT>(-INFINITY), (dt >= win) | (dt > p));
                    S.template select<KT, 1>((t * QR + h) * KT) = sv;
                }
            }
            simd<fp16, KBT * QR * KT> P;
            simd<float, QR> alpha = softmax_block<QR>(S, m, l, P, nullptr);
            simd<float, QR * 16> arep = alpha.template replicate_vs_w_hs<QR, 1, 16, 0>(0);
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                auto oc = O.template select<QR * 16, 1>(c * QR * 16);
                simd<float, QR * 16> acc = simd<float, QR * 16>(oc) * arep;
#pragma unroll
                for (int t = 0; t < KBT; ++t) {
                    simd<fp16, KT * 16> vt = xesimd::lsc_load_2d<fp16, 16, KT, 1, false, true,
                        xesimd::cache_hint::cached, xesimd::cache_hint::cached>(vbase, surfW, surfH, pitch, c * 16, s0 + t * KT);
                    acc = xmx::dpas<8, QR, float, float, fp16, fp16>(
                        acc, vt, simd<fp16, QR * 16>(P.template select<QR * KT, 1>(t * QR * KT)));
                }
                oc = acc;
            }
        }
#pragma unroll
        for (int h = 0; h < QR; ++h) {
            size_t rhs = ((size_t)r * S_NH + kvh * S_G + h) * ns + s;
            simd<float, 1> mv = m[h], lv = l[h];
            block_store<float, 1>(pm + rhs, mv);
            block_store<float, 1>(pl + rhs, lv);
            float* op = po + rhs * S_D;
#pragma unroll
            for (int c = 0; c < NC; ++c)
                block_store<float, 16>(op + c * 16, simd<float, 16>(O.template select<16, 1>((c * QR + h) * 16)));
        }
    }
};

struct FullDec {
    static constexpr int QR = F_G, NC = F_D / 32;     // a thread: all 8 query heads of a KV head, half the dims
    static constexpr int NROT = F_ROT / 16;           // rotated chunks at each half's start
    static constexpr int SSZ = KBT * QR * KT * 4;     // bytes of a thread's partial scores in SLM
    const fp16* q; const fp16* Pc; const fp16* Kr; const float* Rc;   // q [R, 32, 512]; P [T, 4, 512], Kr [T, 4, 128], r [T, 4]
    const fp16* kn;                                   // k_norm fp16 [512]
    float* pm; float* pl; float* po;                  // [R * 32 * NS], [same], [R * 32 * NS * 512]
    const int* pdev; const int* sdev;                 // each row's position, game (null: game 0; a game's cache: TS rows)
    int p0, R, ns, TS;

    void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
        slm_init<2 * 2 * SSZ>();                      // 2 threads x 2 slots
        const int lid = it.get_local_id(0), side = lid & 1;
        int id = it.get_global_id(0) / 2;
        int s = id % ns;
        int rest = id / ns;
        int kvh = rest % F_KV;
        int r = rest / F_KV;
        if (r >= R) return;
        int p = p0 + r;
        if (pdev) { simd<int, 1> pv = block_load<int, 1>(pdev + r); p = pv[0]; }
        size_t grow = 0;
        if (sdev) { simd<int, 1> sv = block_load<int, 1>(sdev + r); grow = (size_t)sv[0] * TS; }
        int nk = p + 1;
        // pieces of whole blocks, as few as the keys need
        int blocks = ((nk + KB - 1) / KB + ns - 1) / ns;
        int k0 = s * blocks * KB, k1 = k0 + blocks * KB;
        if (k1 > nk) k1 = nk;
        int h0 = kvh * F_G;
        const int c0 = side * NC;                     // the first d-chunk this thread owns

        simd<float, QR> m = -INFINITY, l = 0.0f;
        simd<float, NC * QR * 16> O;
        if (k0 < nk) {
            // Qc: chunk c of the rotated dims as q, of the others as fp16(q * k_norm), to dot with Kr / P
            simd<fp16, NC * QR * 16> Qc;
            const fp16* qp = q + ((size_t)r * F_NH + h0) * F_D + c0 * 16;
            const fp16* knp = kn + c0 * 16;
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                const bool rot = c < NROT;
                simd<float, 16> w;
                if (!rot) w = convert<float>(block_load<fp16, 16>(knp + c * 16));
#pragma unroll
                for (int h = 0; h < QR; ++h) {
                    simd<fp16, 16> qv = block_load<fp16, 16>(qp + (size_t)h * F_D + c * 16);
                    if (!rot) qv = convert<fp16>(convert<float>(qv) * w);
                    Qc.template select<16, 1>((c * QR + h) * 16) = qv;
                }
            }
            O = 0.0f;
            const uint32_t* pb32 = reinterpret_cast<const uint32_t*>(Pc + (grow * F_KV + kvh) * F_D);
            const uint32_t* kr32 = reinterpret_cast<const uint32_t*>(Kr + (grow * F_KV + kvh) * 2 * F_ROT);
            const fp16* vbase = Pc + (grow * F_KV + kvh) * F_D;
            unsigned surfW = F_D * 2 - 1, surfH = (unsigned)nk + KB - 1, pitch = (unsigned)F_KV * F_D * 2 - 1;
            unsigned krW = 2 * F_ROT * 2 - 1, krP = (unsigned)F_KV * 2 * F_ROT * 2 - 1;
            const int crb = side * NROT;                  // this half's rotated chunks in Kr

            int slot = 0;
            for (int kb0 = k0; kb0 < k1; kb0 += KB) {
                // r of the block's keys: [KB keys][4 heads] -> one head
                simd<float, KB> rv;
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    simd<float, 64> rr = block_load<float, 64>(Rc + (grow + kb0 + e * 16) * F_KV);
                    rv.template select<16, 1>(e * 16) = rr.template select<16, 4>(kvh);
                }
                simd<float, KBT * QR * KT> S;
#pragma unroll
                for (int t = 0; t < KBT; ++t) {
                    simd<float, QR * KT> acc = 0.0f;
#pragma unroll
                    for (int c = 0; c < NC; ++c) {
                        simd<uint32_t, 8 * KT> kt;
                        if (c < NROT) {
                            kt = xesimd::lsc_load_2d<uint32_t, 8, KT, 1, true, false,
                                xesimd::cache_hint::cached, xesimd::cache_hint::cached>(kr32, krW, surfH, krP, (crb + c) * 8, kb0 + t * KT);
                        } else {
                            kt = xesimd::lsc_load_2d<uint32_t, 8, KT, 1, true, false,
                                xesimd::cache_hint::cached, xesimd::cache_hint::cached>(pb32, surfW, surfH, pitch, (c0 + c) * 8, kb0 + t * KT);
                        }
                        acc = xmx::dpas<8, QR, float, float, fp16, fp16>(
                            acc, kt.template bit_cast_view<fp16>().read(),
                            simd<fp16, QR * 16>(Qc.template select<QR * 16, 1>(c * QR * 16)));
                    }
                    S.template select<QR * KT, 1>(t * QR * KT) = acc;
                }
                // scores = mine + the other half's. Two slots: the next block writes the other one, so one barrier does
                uint32_t mine = (uint32_t)(lid * 2 + slot) * SSZ, other = (uint32_t)((lid ^ 1) * 2 + slot) * SSZ;
#pragma unroll
                for (int t = 0; t < KBT; ++t)
                    slm_block_store<float, QR * KT>(mine + t * QR * KT * 4, S.template select<QR * KT, 1>(t * QR * KT));
                barrier();
#pragma unroll
                for (int t = 0; t < KBT; ++t) {
                    simd<float, QR * KT> theirs = slm_block_load<float, QR * KT>(other + t * QR * KT * 4);
                    simd<float, QR * KT> sum = simd<float, QR * KT>(S.template select<QR * KT, 1>(t * QR * KT)) + theirs;
                    simd<float, KT> rt = rv.template select<KT, 1>(t * KT);
                    simd<int, KT> kj(kb0 + t * KT, 1);
#pragma unroll
                    for (int h = 0; h < QR; ++h) {
                        simd<float, KT> sv = sum.template select<KT, 1>(h * KT);
                        sv = sv * rt * LOG2E;
                        sv.merge(simd<float, KT>(-INFINITY), kj >= nk);
                        S.template select<KT, 1>((t * QR + h) * KT) = sv;
                    }
                }
                slot ^= 1;
                simd<fp16, KBT * QR * KT> P;
                simd<float, QR> alpha = softmax_block<QR>(S, m, l, P, &rv);
                simd<float, QR * 16> arep = alpha.template replicate_vs_w_hs<QR, 1, 16, 0>(0);
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    auto oc = O.template select<QR * 16, 1>(c * QR * 16);
                    simd<float, QR * 16> acc = simd<float, QR * 16>(oc) * arep;
#pragma unroll
                    for (int t = 0; t < KBT; ++t) {
                        simd<fp16, KT * 16> vt = xesimd::lsc_load_2d<fp16, 16, KT, 1, false, true,
                            xesimd::cache_hint::cached, xesimd::cache_hint::cached>(vbase, surfW, surfH, pitch, (c0 + c) * 16, kb0 + t * KT);
                        acc = xmx::dpas<8, QR, float, float, fp16, fp16>(
                            acc, vt, simd<fp16, QR * 16>(P.template select<QR * KT, 1>(t * QR * KT)));
                    }
                    oc = acc;
                }
            }
        }
#pragma unroll
        for (int h = 0; h < QR; ++h) {
            size_t rhs = ((size_t)r * F_NH + h0 + h) * ns + s;
            if (side == 0) {
                simd<float, 1> mv = m[h], lv = l[h];
                block_store<float, 1>(pm + rhs, mv);
                block_store<float, 1>(pl + rhs, lv);
            }
            if (k0 < nk) {
                float* op = po + rhs * F_D + c0 * 16;
#pragma unroll
                for (int c = 0; c < NC; ++c)
                    block_store<float, 16>(op + c * 16, simd<float, 16>(O.template select<16, 1>((c * QR + h) * 16)));
            }
        }
    }
};

// y[r, n] = sum_k x[r, k] W[n, k] for an fp16 weight [N, K] (K a multiple of 16), QR <= 8 rows: a thread = 16 output
// columns, read as K^T tiles straight into DPAS. CAP: softcap 30 tanh(y / 30), the target's logits.
template <int QR, class TO, bool CAP>
struct DenseKernel {
    const fp16* x; const fp16* w; TO* y;              // x [QR, K] (row stride xs), y [QR, N]
    int K, N, xs;

    void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
        int n0 = (int)it.get_global_id(0) * 16;
        if (n0 >= N) return;
        simd<float, QR * 16> acc = 0.0f;
        const uint32_t* wb = reinterpret_cast<const uint32_t*>(w);
        unsigned surfW = (unsigned)K * 2 - 1, surfH = (unsigned)N - 1, pitch = surfW;
        const int nc = K / 16;
        for (int c = 0; c < nc; ++c) {
            simd<fp16, QR * 16> a;
#pragma unroll
            for (int r = 0; r < QR; ++r)
                a.template select<16, 1>(r * 16) = block_load<fp16, 16>(x + (size_t)r * xs + c * 16);
            simd<uint32_t, 8 * 16> kt = xesimd::lsc_load_2d<uint32_t, 8, 16, 1, true, false,
                xesimd::cache_hint::cached, xesimd::cache_hint::cached>(wb, surfW, surfH, pitch, c * 8, n0);
            acc = xmx::dpas<8, QR, float, float, fp16, fp16>(acc, kt.template bit_cast_view<fp16>().read(), a);
        }
        if constexpr (CAP) {
            // tanh(z) = (e^2z - 1) / (e^2z + 1), with z clamped where it saturates anyway
            simd<float, QR * 16> z = acc * (1.0f / 30.0f);
            z = max(min(z, simd<float, QR * 16>(10.0f)), simd<float, QR * 16>(-10.0f));
            simd<float, QR * 16> e = exp2(z * (2.0f * LOG2E));
            acc = 30.0f * (e - 1.0f) / (e + 1.0f);
        }
#pragma unroll
        for (int r = 0; r < QR; ++r) {
            simd<float, 16> v = acc.template select<16, 1>(r * 16);
            TO* yr = y + (size_t)r * N + n0;
            if constexpr (std::is_same_v<TO, float>) block_store<float, 16>(yr, v);
            else block_store<fp16, 16>(yr, convert<fp16>(v));
        }
    }
};

}  // namespace g4dec
