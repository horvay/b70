// exl3conv: scalar CPU trellis encoder, bit-faithful to exllamav3's CUDA quantize_tiles_kernel (integer K).
// Slow (about a second per tile); it exists to check the SYCL encoder bit for bit. notes/exl3-algorithm.md §4.3.
//
// One tile = 256 weights in ring (tile_perm) order. State s_i is 16 bits, s_i = (s_{i-1} << K | new) & 0xFFFF; a
// DP node is the low 16 - K bits of a state (E = 2^(16-K) nodes), and node e with branch k is state (k << (16-K)) | e,
// whose predecessor node is state >> K. Cost = sum of fp16 squared errors with fp16 FMA accumulation.
// Two passes close the ring (tail-biting): pass 1 starts mid-ring unconstrained and traces back the node at position
// 255; pass 2 starts at position 0 forced to follow that node and traces the final path.
// Ties: the lowest branch k wins (strict <); the end node is the lowest cost, ties broken by the bit-reversed rank.
#pragma once

#include "codebook.hpp"
#include "f16.hpp"
#include "pack.hpp"

#include <cstdint>
#include <vector>

namespace exl3 {

struct ViterbiRef
{
    int K, cb;
    const CodebookTable& V;
    int L;      // ring length: 256 (linear tiles) or 160 (n-gram rows)
    ViterbiRef(int K_, int cb_, const CodebookTable& table, int L_ = 256) : K(K_), cb(cb_), V(table), L(L_) {}

    static uint32_t brev5(uint32_t v)
    {
        uint32_t r = 0;
        for (int b = 0; b < 5; ++b) r |= ((v >> b) & 1) << (4 - b);
        return r;
    }

    // w: L floats in ring order. Writes the L states (ring order) and their decoded values.
    void encode(const float* w, uint16_t* idx, float* out) const
    {
        const int Kr = 16 - K, E = 1 << Kr, NB = 1 << K;
        uint16_t wh[TILE];
        for (int i = 0; i < L; ++i) wh[i] = f16_from_float(w[i]);
        std::vector<uint16_t> cost_a(E), cost_b(E), hist((size_t) L * E);
        uint16_t* cost = cost_a.data();          // costs of the position just computed
        uint16_t* prev = cost_b.data();

        auto ring = [&](int i, int roll) { int r = i + roll; return r >= L ? r - L : r; };

        auto forward = [&](int roll, int pre)
        {
            for (int i = 0; i < L; ++i)
            {
                const int ri = ring(i, roll);
                std::swap(cost, prev);
                const uint16_t wv = wh[ri];
                for (int e = 0; e < E; ++e)
                {
                    uint16_t best = 0; int arg = 0;
                    for (int k = 0; k < NB; ++k)
                    {
                        const int state = (k << Kr) | e;
                        const int pred = state >> K;
                        const uint16_t d = f16_sub(V[state], wv);
                        uint16_t err;
                        if (i == 0)
                        {
                            err = f16_mul(d, d);
                            if (pre >= 0 && pred != pre) err = 0x7C00;
                        }
                        else err = f16_fma(d, d, prev[pred]);
                        if (k == 0 || f16_lt_nonneg(err, best)) { best = err; arg = pred; }
                    }
                    cost[e] = best;
                    if (pre >= 0 || ri < L / 2) hist[(size_t) ri * E + e] = (uint16_t) arg;
                }
            }
        };

        auto argmin_end = [&]() -> int
        {
            uint32_t best = 0x7C00FFFFu; int best_e = 0;
            for (int e = 0; e < E; ++e)
            {
                const uint32_t v = e & 1023;
                const uint32_t rank = (brev5(v >> 5) << 10) | (brev5(v & 31) << 5) | ((uint32_t) e >> 10);
                const uint32_t key = ((uint32_t) cost[e] << 16) | rank;
                if (key < best) { best = key; best_e = e; }
            }
            return best >= 0x7C000000u ? 0 : best_e;
        };

        auto backward = [&](int roll, bool write, int edge) -> int
        {
            for (int i = L - 1; i >= 0; --i)
            {
                const int ri = ring(i, roll);
                const int pe = hist[(size_t) ri * E + edge];
                const int enc = (pe << K) | edge;
                edge = pe;
                if (write) { idx[ri] = (uint16_t) enc; out[ri] = f16_to_float(V[enc]); }
                else if (ri == 0) break;
            }
            return edge;
        };

        forward(L / 2, -1);
        const int end_state = backward(L / 2, false, argmin_end());
        forward(0, end_state);
        backward(0, true, end_state);
    }
};

} // namespace exl3
