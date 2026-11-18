// exl3conv: CPU dequantization of one EXL3 linear, the way inference reads it (notes/exl3-algorithm.md §5.5):
//   W[k, n] = diag(suh) Hk W_hat Hn diag(svh),  W_hat[16a + r, 16b + c] = decode(state at ring perm^-1(16r + c))
// with fp16 suh / svh and blockwise H128 = Sylvester / sqrt(128). Used by tests to check what the quantizer wrote.
#pragma once

#include "exl3/codebook.hpp"
#include "exl3/f16.hpp"
#include "exl3/pack.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

namespace exl3 {

// In-place unnormalized Walsh-Hadamard transform of 128 values (natural order), then * 1/sqrt(128).
inline void fwht128(double* v, int64_t stride)
{
    for (int h = 1; h < 128; h <<= 1)
        for (int i = 0; i < 128; i += 2 * h)
            for (int j = i; j < i + h; ++j)
            {
                const double a = v[j * stride], b = v[(j + h) * stride];
                v[j * stride] = a + b; v[(j + h) * stride] = a - b;
            }
    const double s = 1.0 / std::sqrt(128.0);
    for (int i = 0; i < 128; ++i) v[i * stride] *= s;
}

// trellis: [k/16][n/16][16K] words; returns W (k x n, row-major, double).
inline std::vector<double> dequant_host(const uint16_t* trellis, const uint16_t* suh, const uint16_t* svh, int64_t k,
                                        int64_t n, int K, int cb)
{
    const int64_t tk = k / 16, tn = n / 16;
    const Rate rate = Rate::integer(K);
    const int words = rate.words16();
    const CodebookTable table(cb);
    const auto& perm = tile_perm();
    std::vector<double> W((size_t) k * n);
    uint16_t states[256];
    for (int64_t a = 0; a < tk; ++a)
        for (int64_t b = 0; b < tn; ++b)
        {
            unpack_tile(trellis + (a * tn + b) * words, states, rate);
            for (int i = 0; i < 256; ++i)
            {
                const int e = perm[i];
                W[(a * 16 + e / 16) * n + b * 16 + e % 16] = f16_to_double(table[states[i]]);
            }
        }
    for (int64_t r = 0; r < k; ++r) for (int64_t c = 0; c < n; c += 128) fwht128(&W[r * n + c], 1);   // Hn (rows' blocks)
    for (int64_t r = 0; r < k; r += 128) for (int64_t c = 0; c < n; ++c) fwht128(&W[r * n + c], n);   // Hk (columns' blocks)
    for (int64_t r = 0; r < k; ++r)
    {
        const double su = f16_to_double(suh[r]);
        for (int64_t c = 0; c < n; ++c) W[r * n + c] *= su * f16_to_double(svh[c]);
    }
    return W;
}

} // namespace exl3
