// exl3conv: the trellis encoder on the B70 (SYCL). Bit-for-bit the algorithm of viterbi_ref.hpp and of exllamav3's
// quantize_tiles_kernel (notes/exl3-algorithm.md §4.3); tests/test_gpu.cpp checks it against the CPU reference.
//
// One work-group per 256-weight tile, WG = min(1024, E) work-items, E = 2^(16-K) DP nodes, so each work-item owns
// NPL = E / WG nodes e = j * WG + lid. Per ring position every node takes the best of its 2^K branches:
//     state = (k << (16-K)) | e,  pred = state >> K = (k << (16-2K)) | (e >> K)
//     err   = fma16(d, d, cost_prev[pred]),  d = f16(V[state] - w)
// A sub-group's 16 consecutive nodes share e >> K for K >= 4, so cost_prev[pred] is one SLM broadcast per branch.
// Codebook values depend only on the state, not the position: for NPL * 2^K <= 64 they stay in registers for the
// whole tile; above that (K = 7, 8) they are decoded per use (integer multiply, byte sum, one fp16 FMA).
// Costs: two E-entry fp16 rows in SLM (16 KB at K = 4); K = 1 (128 KB) keeps them in global memory.
// History: one byte per node and position (the winning branch k) in global memory, 256 E bytes per tile, sized per
// launch; the backward trace is one work-item reading 256 of them.
// fp16 math is native: subtract, multiply and fused multiply-add round once each, as CUDA's __hsub/__hmul/__hfma.
#pragma once

#include "rt/dbuf.hpp"

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <string>

namespace exl3 {

using half = sycl::half;

namespace dev {

inline half bits_to_half(uint16_t b) { return sycl::bit_cast<half>(b); }
inline uint16_t half_to_bits(half h) { return sycl::bit_cast<uint16_t>(h); }

// Codebook value of a 16-bit state (see codebook.hpp for the formulas).
template <int CB>
inline half decode(uint32_t s)
{
    if constexpr (CB == 2)
    {
        const uint32_t x = s * 0x83DCD12Du;
        const uint32_t p = (x & 0x00FF00FFu) + ((x >> 8) & 0x00FF00FFu);
        const uint32_t b = (p & 0xFFFFu) + (p >> 16);                          // byte sum, 0..1020
        const half h = bits_to_half((uint16_t) (0x6400u + b));                // = 1024 + b exactly
        return sycl::fma(h, bits_to_half(0x1EEE), bits_to_half(0xC931));
    }
    else
    {
        uint32_t x = (CB == 0) ? s * 89226354u + 64248484u : s * 0xCBAC1FEDu;
        x = (x & 0x8FFF8FFFu) ^ 0x3B603B60u;
        return bits_to_half((uint16_t) (x & 0xFFFF)) + bits_to_half((uint16_t) (x >> 16));
    }
}

inline uint32_t brev5(uint32_t v)                                           // reverse the low 5 bits
{
    return ((v & 1) << 4) | ((v & 2) << 2) | (v & 4) | ((v & 8) >> 2) | ((v & 16) >> 4);
}

} // namespace dev

template <int K> constexpr int vit_edges() { return 65536 >> K; }
template <int K> constexpr bool vit_slm_costs() { return K >= 2; }
inline size_t vit_history_bytes(int K, int L = 256) { return (size_t) L * (65536 >> K); }

// Kernel shape. WG = work-items per tile (capped at E); CACHE = keep codebook values in registers when a work-item
// needs at most this many; BIG = 256-register mode (an Xe2 Xe-core then holds 32 hardware threads, so WG <= 512);
// UNROLL = unroll factor of the branch loop when values are decoded per use.
struct VitCfg { int wg, cache, big, unroll; };

// in: ntiles x L floats (ring order). idx: ntiles x L states. out: ntiles x L decoded values. L = 256 for linear tiles
// (tensor-core order), 160 for n-gram table rows (natural order).
// hist: ntiles x vit_history_bytes(K, L) bytes. gcost: ntiles x 2 x E halves (only used when costs don't fit SLM).
// OPT bits (schedule only; every combination computes the same states):
//   1 PEEL     the first ring position runs outside the loop, so the loop body has no position-0 branch
//   2 PACKMIN  each branch's cost and index fold into one u32 key ((half bits << 8) | k): one integer min per branch
//              instead of a compare and two selects; ties still go to the lowest k
//   4 CONTIG   a work-item owns NPL consecutive nodes, so its history bytes leave as one store (u32 for NPL = 4)
//   8 VECSTORE with CONTIG, the NPL new costs go to SLM as one vector store
//  16 SPLIT    for E < WG_ (K >= 7): S = WG_ / E work-items share a node, each taking NB / S of its branches, and
//              their packed keys meet in a sub-group xor butterfly (needs PACKMIN); lane 0 of each node writes
template <int K, int CB, int WG_, int CACHE_MAX, bool BIG, int UNROLL, int OPT = 0, int L = 256>
sycl::event viterbi_launch(sycl::queue& q, const float* in, uint16_t* idx, float* out, uint8_t* hist, half* gcost,
                           int64_t ntiles)
{
    constexpr int E = vit_edges<K>();
    constexpr bool SPLIT = (OPT & 16) && WG_ > E;
    constexpr int S = SPLIT ? WG_ / E : 1;                 // work-items per node
    constexpr int WG = SPLIT ? WG_ : (WG_ < E ? WG_ : E);
    static_assert(!BIG || WG <= 512, "256-register mode holds at most 32 threads of 16 per Xe-core");
    static_assert(!SPLIT || ((OPT & 2) && S <= 16), "SPLIT needs PACKMIN and at most 16 lanes per node");
    constexpr int NPL = SPLIT ? 1 : E / WG;
    constexpr int NB = 1 << K;
    constexpr int NBL = NB / S;                            // branches per work-item
    constexpr int Kr = 16 - K;
    constexpr bool CACHE = NPL * NBL <= CACHE_MAX;
    constexpr bool SLMC = vit_slm_costs<K>();
    constexpr bool PEEL = OPT & 1, PACKMIN = OPT & 2, CONTIG = OPT & 4;
    constexpr bool VECSTORE = CONTIG && (OPT & 8) && (NPL == 2 || NPL == 4 || NPL == 8 || NPL == 16);
    static_assert(L % 2 == 0, "ring length must be even (pass 1 starts at L / 2)");

    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<half, 1> sh_w(sycl::range<1>(L), h);
        sycl::local_accessor<half, 1> sh_cost(sycl::range<1>(SLMC ? 2 * E : 1), h);
        sycl::local_accessor<int, 1> sh_edge(sycl::range<1>(1), h);
        auto body = [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                const int64_t tile = it.get_group(0);
                const int lid = (int) it.get_local_id(0);
                auto grp = it.get_group();
                const float* tin = in + tile * L;
                uint8_t* th = hist + tile * (int64_t) (L * E);
                half* costs = SLMC ? sh_cost.template get_multi_ptr<sycl::access::decorated::no>().get()
                                   : gcost + tile * (int64_t) (2 * E);
                auto node = [&](int j) { return SPLIT ? lid / S : CONTIG ? lid * NPL + j : j * WG + lid; };
                const int part = SPLIT ? lid % S : 0;              // which slice of the branches this work-item takes
                auto sg = it.get_sub_group();

                for (int i = lid; i < L; i += WG) sh_w[i] = half(tin[i]);

                half vals[CACHE ? NPL : 1][CACHE ? NBL : 1];
                if constexpr (CACHE)
                {
                    #pragma unroll
                    for (int j = 0; j < NPL; ++j)
                        #pragma unroll
                        for (int kk = 0; kk < NBL; ++kk)
                            vals[j][kk] = dev::decode<CB>((uint32_t) (((part * NBL + kk) << Kr) | node(j)));
                }
                sycl::group_barrier(grp);

                int cur = 0;          // costs[cur * E ..] = the position just computed
                // One ring position. FIRST: no incoming cost; pre >= 0 forces the predecessor node.
                auto step = [&](auto first_c, int ri, int pre, bool keep) {
                    constexpr bool FIRST = decltype(first_c)::value;
                    cur ^= 1;
                    half* cc = costs + cur * E;
                    const half* cp = costs + (cur ^ 1) * E;
                    const half w = sh_w[ri];
                    uint8_t args[NPL];
                    half bests[NPL];
                    #pragma unroll
                    for (int j = 0; j < NPL; ++j)
                    {
                        const int e = node(j);
                        half best = half(0.0f); int arg = 0;
                        uint32_t bkey = 0xFFFFFFFFu;
                        #pragma unroll (CACHE ? NBL : UNROLL)
                        for (int kk = 0; kk < NBL; ++kk)
                        {
                            const int k = part * NBL + kk;
                            const int state = (k << Kr) | e;
                            const int pred = state >> K;
                            half v;
                            if constexpr (CACHE) v = vals[j][kk]; else v = dev::decode<CB>((uint32_t) state);
                            const half d = v - w;
                            half err;
                            if constexpr (FIRST)
                            {
                                err = d * d;
                                if (pre >= 0 && pred != pre) err = dev::bits_to_half(0x7C00);
                            }
                            else err = sycl::fma(d, d, cp[pred]);
                            if constexpr (PACKMIN)
                                bkey = sycl::min(bkey, ((uint32_t) dev::half_to_bits(err) << 8) | (uint32_t) k);
                            else if (k == 0 || err < best) { best = err; arg = k; }
                        }
                        if constexpr (SPLIT)
                        {
                            #pragma unroll
                            for (int m = 1; m < S; m <<= 1) bkey = sycl::min(bkey, sycl::permute_group_by_xor(sg, bkey, m));
                        }
                        if constexpr (PACKMIN) { best = dev::bits_to_half((uint16_t) (bkey >> 8)); arg = (int) (bkey & 0xFF); }
                        if constexpr (VECSTORE) bests[j] = best; else if (part == 0) cc[e] = best;
                        args[j] = (uint8_t) arg;
                    }
                    if constexpr (VECSTORE)
                    {
                        sycl::vec<half, NPL> v;
                        #pragma unroll
                        for (int j = 0; j < NPL; ++j) v[j] = bests[j];
                        *reinterpret_cast<sycl::vec<half, NPL>*>(cc + node(0)) = v;
                    }
                    if (keep && part == 0)
                    {
                        if constexpr (CONTIG && NPL == 4)
                        {
                            uint32_t packed = (uint32_t) args[0] | ((uint32_t) args[1] << 8) | ((uint32_t) args[2] << 16) | ((uint32_t) args[3] << 24);
                            *reinterpret_cast<uint32_t*>(th + ri * E + node(0)) = packed;
                        }
                        else if constexpr (CONTIG && NPL == 2)
                            *reinterpret_cast<uint16_t*>(th + ri * E + node(0)) = (uint16_t) (args[0] | (args[1] << 8));
                        else
                        {
                            #pragma unroll
                            for (int j = 0; j < NPL; ++j) th[ri * E + node(j)] = args[j];
                        }
                    }
                    sycl::group_barrier(grp);
                };
                auto forward = [&](int roll, int pre) {
                    auto ring = [&](int i) { int ri = i + roll; return ri >= L ? ri - L : ri; };
                    auto keep = [&](int ri) { return pre >= 0 || ri < L / 2; };
                    if constexpr (PEEL)
                    {
                        step(std::true_type{}, ring(0), pre, keep(ring(0)));
                        for (int i = 1; i < L; ++i) step(std::false_type{}, ring(i), pre, keep(ring(i)));
                    }
                    else
                        for (int i = 0; i < L; ++i)
                        {
                            if (i == 0) step(std::true_type{}, ring(i), pre, keep(ring(i)));
                            else step(std::false_type{}, ring(i), pre, keep(ring(i)));
                        }
                };

                auto argmin_end = [&]() -> int {
                    const half* cc = costs + cur * E;
                    uint32_t best = 0x7C00FFFFu;
                    #pragma unroll
                    for (int j = 0; j < NPL; ++j)
                    {
                        const uint32_t e = (uint32_t) node(j), v = e & 1023;
                        const uint32_t rank = (dev::brev5(v >> 5) << 10) | (dev::brev5(v & 31) << 5) | (e >> 10);
                        const uint32_t key = part == 0 ? (((uint32_t) dev::half_to_bits(cc[e]) << 16) | rank) : 0xFFFFFFFFu;
                        best = sycl::min(best, key);
                    }
                    best = sycl::reduce_over_group(grp, best, sycl::minimum<uint32_t>());
                    if (best >= 0x7C000000u) return 0;
                    const uint32_t rank = best & 0xFFFF;
                    return (int) (((rank & 31) << 10) | (dev::brev5(rank >> 10) << 5) | dev::brev5((rank >> 5) & 31));
                };

                auto backward = [&](int roll, bool write, int edge) -> int {
                    if (lid == 0)
                    {
                        for (int i = L - 1; i >= 0; --i)
                        {
                            int ri = i + roll; if (ri >= L) ri -= L;
                            const int k = th[ri * E + edge];
                            const int pe = (k << (16 - 2 * K)) | (edge >> K);
                            const int enc = (pe << K) | edge;
                            edge = pe;
                            if (write)
                            {
                                idx[tile * L + ri] = (uint16_t) enc;
                                out[tile * L + ri] = (float) dev::decode<CB>((uint32_t) enc);
                            }
                            else if (ri == 0) break;
                        }
                        sh_edge[0] = edge;
                    }
                    sycl::group_barrier(grp);
                    const int r = sh_edge[0];
                    sycl::group_barrier(grp);
                    return r;
                };

                forward(L / 2, -1);
                const int end_state = backward(L / 2, false, argmin_end());
                forward(0, end_state);
                backward(0, true, end_state);
            };
        const sycl::nd_range<1> nr(sycl::range<1>((size_t) ntiles * WG), sycl::range<1>(WG));
        if constexpr (BIG)
            h.parallel_for(nr, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, body);
        else
            h.parallel_for(nr, body);
    });
}

// Kernel shape per K: the fastest one that compiles without register spills, each kernel in its own device module
// (-fsycl-device-code-split=per_kernel: in a shared module the register allocation of a kernel depends on its
// neighbours). Measured 2026-10-24 on the B70 with tests/vit_variants.cpp, Gaussian tiles, mul1, tiles/s:
// K1 6.2k, K2 21.3k, K3 44.9k, K4 49.8k, K5 36.7k, K6 28.4k, K7 29.4k, K8 22.4k. (The first version, 1024/c64
// without the OPT bits, spilled 20-65 registers and ran K4 at 25.9k, K8 at 13.3k.)
#define EXL3_VIT_SHAPE_1 1024, 64, false, 8, 3
#define EXL3_VIT_SHAPE_2 1024, 64, false, 8, 15
#define EXL3_VIT_SHAPE_3 1024, 64, false, 8, 15
#define EXL3_VIT_SHAPE_4 1024, 64, false, 8, 15
#define EXL3_VIT_SHAPE_5 1024, 64, false, 8, 15
#define EXL3_VIT_SHAPE_6 1024, 64, false, 8, 3
#define EXL3_VIT_SHAPE_7 512, 256, true, 8, 2
#define EXL3_VIT_SHAPE_8 1024, 64, false, 8, 18
#define EXL3_VIT_SHAPE(KK) EXL3_VIT_SHAPE_##KK

// Runtime dispatch over K = 1..8 and the three codebooks.
inline sycl::event viterbi(sycl::queue& q, int K, int cb, const float* in, uint16_t* idx, float* out, uint8_t* hist,
                           half* gcost, int64_t ntiles, int L = 256)
{
    if (L == 160)
    {
        if (cb != 2) throw std::runtime_error("viterbi: length-160 rings are mul1 only");
#define EXL3_VIT_CASE160(KK) case KK: return viterbi_launch<KK, 2, EXL3_VIT_SHAPE(KK), 160>(q, in, idx, out, hist, gcost, ntiles);
        switch (K)
        {
            EXL3_VIT_CASE160(1) EXL3_VIT_CASE160(2) EXL3_VIT_CASE160(3) EXL3_VIT_CASE160(4)
            EXL3_VIT_CASE160(5) EXL3_VIT_CASE160(6) EXL3_VIT_CASE160(7) EXL3_VIT_CASE160(8)
            default: throw std::runtime_error("viterbi: K must be 1..8");
        }
#undef EXL3_VIT_CASE160
    }
    if (L != 256) throw std::runtime_error("viterbi: ring length must be 256 or 160");
#define EXL3_VIT_CASE(KK)                                                                                    \
    case KK:                                                                                                 \
        if (cb == 2) return viterbi_launch<KK, 2, EXL3_VIT_SHAPE(KK)>(q, in, idx, out, hist, gcost, ntiles);                     \
        if (cb == 1) return viterbi_launch<KK, 1, EXL3_VIT_SHAPE(KK)>(q, in, idx, out, hist, gcost, ntiles);                     \
        return viterbi_launch<KK, 0, EXL3_VIT_SHAPE(KK)>(q, in, idx, out, hist, gcost, ntiles);
    switch (K)
    {
        EXL3_VIT_CASE(1) EXL3_VIT_CASE(2) EXL3_VIT_CASE(3) EXL3_VIT_CASE(4)
        EXL3_VIT_CASE(5) EXL3_VIT_CASE(6) EXL3_VIT_CASE(7) EXL3_VIT_CASE(8)
        default: throw std::runtime_error("viterbi: K must be 1..8, got " + std::to_string(K));
    }
#undef EXL3_VIT_CASE
}

inline bool viterbi_needs_gcost(int K) { return K < 2; }
inline size_t viterbi_gcost_halves(int K) { return (size_t) 2 * (65536 >> K); }

} // namespace exl3

namespace exl3 {

// Encoder workspace: traceback history for up to max_tiles tiles per launch (512 MB budget), reused across calls.
struct VitWork
{
    sycl::queue& q;
    int K;
    int64_t max_tiles;
    uint8_t* hist = nullptr;
    half* gcost = nullptr;
    int L;
    size_t ledger = 0;
    VitWork(sycl::queue& q_, int K_, size_t budget = 512ull << 20, int L_ = 256) : q(q_), K(K_), L(L_)
    {
        max_tiles = std::max<int64_t>(32, (int64_t) (budget / vit_history_bytes(K, L)));
        ledger = max_tiles * vit_history_bytes(K, L) + (viterbi_needs_gcost(K) ? max_tiles * viterbi_gcost_halves(K) * sizeof(half) : 0);
        Vram::take(ledger);
        hist = sycl::malloc_device<uint8_t>(max_tiles * vit_history_bytes(K, L), q);
        if (viterbi_needs_gcost(K)) gcost = sycl::malloc_device<half>(max_tiles * viterbi_gcost_halves(K), q);
        if (!hist || (viterbi_needs_gcost(K) && !gcost)) throw std::runtime_error("VitWork: device allocation failed");
    }
    ~VitWork() { if (hist) sycl::free(hist, q); if (gcost) sycl::free(gcost, q); Vram::give(ledger); }
    VitWork(const VitWork&) = delete;
    VitWork& operator=(const VitWork&) = delete;

    // Encode n rings (ring order) in launches of at most max_tiles. Queued; returns the last event.
    sycl::event encode(int cb, const float* in, uint16_t* idx, float* out, int64_t n)
    {
        sycl::event ev;
        for (int64_t s = 0; s < n; s += max_tiles)
            ev = viterbi(q, K, cb, in + s * L, idx + s * L, out + s * L, hist, gcost, std::min(max_tiles, n - s), L);
        return ev;
    }
};

} // namespace exl3
