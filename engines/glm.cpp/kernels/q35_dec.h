#pragma once
// Decode / verify attention for Qwen3.5 / Qwen3.8 (ESIMD, DPAS), after kernels/g4_dec.h: a thread is one row x the 6 query
// heads of a KV head x a piece of that row's keys; the pieces' partial results (m, l, O: online softmax in base 2,
// unnormalised) are combined by q35ops::attn_dec_combine. K^T tiles come from transposed 2D block loads, V tiles from
// VNNI loads, so each key row is read whole and once per row.
//
// Exactness (MTP, docs/qwen35-spec.md section 7): a thread reads nothing of the other rows of its step. Its pieces come
// from its own key count (blocks per piece = ceil(blocks / ns)), its surface ends at its own last key (2D loads past it
// read zeros), so a verify row computes bit for bit what a one-row step at its position does.

#include "g4_dec.h"

namespace q35dec {

using namespace sycl::ext::intel::esimd;
namespace xesimd = sycl::ext::intel::experimental::esimd;
using fp16 = sycl::half;
using g4dec::KT;
using g4dec::KBT;
using g4dec::KB;

constexpr int NH = 24, NKV = 4, HD = 256, G = NH / NKV;
constexpr float SCORE = 1.4426950408889634f * 0.0625f;      // log2 e / sqrt(256)

struct Dec {
    static constexpr int QR = G, NC = HD / 16;
    const fp16* q; const fp16* k; const fp16* v;      // q [R, 24, 256]; K, V [seqs * TS, 4, 256]
    float* pm; float* pl; float* po;                  // [R * 24 * NS], [same], [R * 24 * NS * 256]
    const int* pdev; const int* sdev;                 // each row's position and sequence
    int R, ns, TS;

    void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
        // rows fastest: the rows of a verify step at the same piece share a work-group (and so L1)
        const int id = it.get_global_id(0);
        const int r = id % R, rest = id / R;
        const int s = rest % ns, kvh = rest / ns;
        if (kvh >= NKV) return;
        simd<int, 1> pv = block_load<int, 1>(pdev + r);
        simd<int, 1> sv = block_load<int, 1>(sdev + r);
        const int nk = pv[0] + 1;
        const size_t grow = (size_t)sv[0] * TS;
        // pieces of whole blocks, as few as this row's keys need
        const int bpp = ((nk + KB - 1) / KB + ns - 1) / ns;
        const int k0 = s * bpp * KB;
        int k1 = k0 + bpp * KB;
        if (k1 > nk) k1 = nk;
        const size_t rh0 = ((size_t)r * NH + kvh * G) * ns + s;     // head h of the group: rh0 + h * ns

        simd<float, QR> m = -INFINITY, l = 0.0f;
        if (k0 >= nk) {
#pragma unroll
            for (int h = 0; h < QR; ++h) {
                simd<float, 1> mv = -INFINITY, lv = 0.0f;
                block_store<float, 1>(pm + rh0 + (size_t)h * ns, mv);
                block_store<float, 1>(pl + rh0 + (size_t)h * ns, lv);
            }
            return;
        }
        simd<fp16, NC * QR * 16> Qc;
        const fp16* qp = q + ((size_t)r * NH + kvh * G) * HD;
#pragma unroll
        for (int h = 0; h < QR; ++h) {
            simd<fp16, HD> qh = block_load<fp16, HD>(qp + (size_t)h * HD);
#pragma unroll
            for (int c = 0; c < NC; ++c) Qc.template select<16, 1>((c * QR + h) * 16) = qh.template select<16, 1>(c * 16);
        }
        simd<float, NC * QR * 16> O = 0.0f;
        const uint32_t* kb32 = reinterpret_cast<const uint32_t*>(k + (grow * NKV + kvh) * HD);
        const fp16* vbase = v + (grow * NKV + kvh) * HD;
        const unsigned surfW = HD * 2 - 1, surfH = (unsigned)nk - 1, pitch = (unsigned)NKV * HD * 2 - 1;

        for (int kb0 = k0; kb0 < k1; kb0 += KB) {
            simd<float, KBT * QR * KT> S;
#pragma unroll
            for (int t = 0; t < KBT; ++t) {
                simd<float, QR * KT> acc = 0.0f;
#pragma unroll
                for (int c = 0; c < NC; ++c) {
                    simd<uint32_t, 8 * KT> kt = xesimd::lsc_load_2d<uint32_t, 8, KT, 1, true, false,
                        xesimd::cache_hint::cached, xesimd::cache_hint::cached>(kb32, surfW, surfH, pitch, c * 8, kb0 + t * KT);
                    acc = xmx::dpas<8, QR, float, float, fp16, fp16>(
                        acc, kt.template bit_cast_view<fp16>().read(),
                        simd<fp16, QR * 16>(Qc.template select<QR * 16, 1>(c * QR * 16)));
                }
                acc *= SCORE;
                simd<int, KT> kj(kb0 + t * KT, 1);
#pragma unroll
                for (int h = 0; h < QR; ++h) {
                    simd<float, KT> sv2 = acc.template select<KT, 1>(h * KT);
                    sv2.merge(simd<float, KT>(-INFINITY), kj >= nk);
                    S.template select<KT, 1>((t * QR + h) * KT) = sv2;
                }
            }
            simd<fp16, KBT * QR * KT> P;
            simd<float, QR> alpha = g4dec::softmax_block<QR>(S, m, l, P, nullptr);
            simd<float, QR * 16> arep = alpha.template replicate_vs_w_hs<QR, 1, 16, 0>(0);
#pragma unroll
            for (int c = 0; c < NC; ++c) {
                auto oc = O.template select<QR * 16, 1>(c * QR * 16);
                simd<float, QR * 16> acc = simd<float, QR * 16>(oc) * arep;
#pragma unroll
                for (int t = 0; t < KBT; ++t) {
                    simd<fp16, KT * 16> vt = xesimd::lsc_load_2d<fp16, 16, KT, 1, false, true,
                        xesimd::cache_hint::cached, xesimd::cache_hint::cached>(vbase, surfW, surfH, pitch, c * 16, kb0 + t * KT);
                    acc = xmx::dpas<8, QR, float, float, fp16, fp16>(
                        acc, vt, simd<fp16, QR * 16>(P.template select<QR * KT, 1>(t * QR * KT)));
                }
                oc = acc;
            }
        }
#pragma unroll
        for (int h = 0; h < QR; ++h) {
            const size_t rhs = rh0 + (size_t)h * ns;
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

}  // namespace q35dec
