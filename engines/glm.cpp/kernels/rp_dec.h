#pragma once
// Decode / verify attention for Ripple-65B (ESIMD, DPAS): a row's query heads over its keys, split into pieces of keys whose
// partial results (m, l, O: online softmax in base 2, unnormalised) a plain kernel combines, as kernels/g4_dec.h. Query
// heads sharing a KV head run in one thread, so each key byte is read once: 10 heads take two DPAS per tile of keys.
//
// SLIDE: the cache is a ring of `span` slots per game, a slot holds the last position written to it, visible when within
// the window behind the row's position and at or after position 0. Otherwise (global layers) keys 0 .. p of the game's
// span rows. Scores are q . k / sqrt(128). A draft block (last >= 0, a block's last row index) also sees the block's later rows.

#include "g4_dec.h"

namespace rpdec {

using namespace sycl::ext::intel::esimd;
namespace xesimd = sycl::ext::intel::experimental::esimd;
using fp16 = sycl::half;
using g4dec::KT;
using g4dec::KBT;
using g4dec::KB;

constexpr int NH = 80, NKV = 8, HD = 128, GRP = 10, GA = 5;
constexpr float SCORE = 1.4426950408889634f * 0.08838834764831845f;    // log2 e / sqrt(128)

template <bool SLIDE>
struct Dec {
    static constexpr int QR = GRP, NC = HD / 16;
    const fp16* q; const fp16* k; const fp16* v;      // q [R, 80, 128]; K, V [S * span, 8, 128]
    float* pm; float* pl; float* po;                  // [R * 80 * NS], [same], [R * 80 * NS * 128]
    const int* pdev; const int* sdev;                 // each row's position, game
    int p0, R, span, win, ns, blocks;                 // SLIDE: ns pieces of `blocks` blocks of slots
    int last = -1;

    void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
        int id = it.get_global_id(0);
        int s = id % ns;
        int rest = id / ns;
        int kvh = rest % NKV;
        int r = rest / NKV;
        if (r >= R) return;
        int p = p0 + r;
        if (pdev) { simd<int, 1> pv = block_load<int, 1>(pdev + r); p = pv[0]; }
        size_t gbase = 0;
        if (sdev) { simd<int, 1> sv = block_load<int, 1>(sdev + r); gbase = (size_t)sv[0] * span * NKV * HD; }

        simd<fp16, NC * QR * 16> Qc;
        const fp16* qp = q + ((size_t)r * NH + kvh * GRP) * HD;
#pragma unroll
        for (int h = 0; h < QR; ++h)
#pragma unroll
            for (int c = 0; c < NC; ++c)
                Qc.template select<16, 1>((c * QR + h) * 16) = block_load<fp16, 16>(qp + h * HD + c * 16);

        simd<float, NC * QR * 16> O = 0.0f;
        simd<float, QR> m = -INFINITY, l = 0.0f;

        const uint32_t* kb32 = reinterpret_cast<const uint32_t*>(k + gbase + (size_t)kvh * HD);
        const fp16* vbase = v + gbase + (size_t)kvh * HD;
        const int nk = p + 1;
        int b0, b1, ps = 0;
        const int fwd = last >= 0 ? last - r : 0;     // rows visible ahead of this one
        unsigned surfW = HD * 2 - 1, surfH, pitch = (unsigned)NKV * HD * 2 - 1;
        if constexpr (SLIDE) {
            ps = (p + fwd) % span;
            b0 = s * blocks;
            b1 = (s + 1) * blocks;
            surfH = (unsigned)span - 1;
        } else {
            // pieces of whole blocks, as few as the keys need
            int bl = ((nk + KB - 1) / KB + ns - 1) / ns;
            b0 = s * bl;
            b1 = b0 + bl;
            surfH = (unsigned)nk + KB - 1;
        }

        for (int b = b0; b < b1 && b * KB < (SLIDE ? span : nk); ++b) {
            int s0 = b * KB;
            simd<int, KB> dist;
            if constexpr (SLIDE) {
                // distance behind the row of each slot in the block; beyond the window or further back than the row's
                // position: not visible
                simd<int, KB> slots(s0, 1);
                dist = ps - slots;
                dist.merge(dist + span, dist < 0);
                simd_mask<KB> hide = (dist - fwd >= win) | (dist > p + fwd);
                if (hide.all()) continue;
            }
            simd<float, KBT * QR * KT> S;
#pragma unroll
            for (int t = 0; t < KBT; ++t) {
                simd<float, GA * KT> a0 = 0.0f, a1 = 0.0f;
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    simd<uint32_t, 8 * KT> kt = xesimd::lsc_load_2d<uint32_t, 8, KT, 1, true, false,
                        xesimd::cache_hint::cached, xesimd::cache_hint::cached>(kb32, surfW, surfH, pitch, c * 8, s0 + t * KT);
                    simd<fp16, 256> B = kt.template bit_cast_view<fp16>().read();
                    a0 = xmx::dpas<8, GA, float, float, fp16, fp16>(
                        a0, B, simd<fp16, GA * 16>(Qc.template select<GA * 16, 1>(c * QR * 16)));
                    a1 = xmx::dpas<8, GA, float, float, fp16, fp16>(
                        a1, B, simd<fp16, GA * 16>(Qc.template select<GA * 16, 1>((c * QR + GA) * 16)));
                }
                S.template select<GA * KT, 1>(t * QR * KT) = a0 * SCORE;
                S.template select<GA * KT, 1>((t * QR + GA) * KT) = a1 * SCORE;
#pragma unroll
                for (int h = 0; h < QR; ++h) {
                    simd<float, KT> sv = S.template select<KT, 1>((t * QR + h) * KT);
                    if constexpr (SLIDE) {
                        simd<int, KT> dt = dist.template select<KT, 1>(t * KT);
                        sv.merge(simd<float, KT>(-INFINITY), (dt - fwd >= win) | (dt > p + fwd));
                    } else {
                        simd<int, KT> kj(s0 + t * KT, 1);
                        sv.merge(simd<float, KT>(-INFINITY), kj >= nk);
                    }
                    S.template select<KT, 1>((t * QR + h) * KT) = sv;
                }
            }
            simd<fp16, KBT * QR * KT> P;
            simd<float, QR> alpha = g4dec::softmax_block<QR>(S, m, l, P, nullptr);
            simd<float, QR * 16> arep = alpha.template replicate_vs_w_hs<QR, 1, 16, 0>(0);
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                auto oc = O.template select<QR * 16, 1>(c * QR * 16);
                simd<float, QR * 16> acc = simd<float, QR * 16>(oc) * arep;
                simd<float, GA * 16> a0 = acc.template select<GA * 16, 1>(0), a1 = acc.template select<GA * 16, 1>(GA * 16);
#pragma unroll
                for (int t = 0; t < KBT; ++t) {
                    simd<fp16, 256> vt = xesimd::lsc_load_2d<fp16, 16, KT, 1, false, true,
                        xesimd::cache_hint::cached, xesimd::cache_hint::cached>(vbase, surfW, surfH, pitch, c * 16, s0 + t * KT);
                    a0 = xmx::dpas<8, GA, float, float, fp16, fp16>(
                        a0, vt, simd<fp16, GA * 16>(P.template select<GA * KT, 1>(t * QR * KT)));
                    a1 = xmx::dpas<8, GA, float, float, fp16, fp16>(
                        a1, vt, simd<fp16, GA * 16>(P.template select<GA * KT, 1>((t * QR + GA) * KT)));
                }
                oc.template select<GA * 16, 1>(0) = a0;
                oc.template select<GA * 16, 1>(GA * 16) = a1;
            }
        }
#pragma unroll
        for (int h = 0; h < QR; ++h) {
            size_t rhs = ((size_t)r * NH + kvh * GRP + h) * ns + s;
            simd<float, 1> mv = m[h], lv = l[h];
            block_store<float, 1>(pm + rhs, mv);
            block_store<float, 1>(pl + rhs, lv);
            float* op = po + rhs * HD;
#pragma unroll
            for (int c = 0; c < NC; ++c)
                block_store<float, 16>(op + c * 16, simd<float, 16>(O.template select<16, 1>((c * QR + h) * 16)));
        }
    }
};

}  // namespace rpdec
