// exl3conv: EXL3 linear -> dense fp16 weight on the card, W[in, out] = diag(suh) Hk W_hat Hn diag(svh)
// (notes/exl3-algorithm.md §5.5). The calibration state advances through this, i.e. through the stored model with
// its fp16 scales, as inference will see it. The Hadamards and scales run in fp32; W is rounded to fp16 once.
#pragma once

#include "exl3/kernels.hpp"
#include "exl3/viterbi.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>

namespace exl3 {

// trellis: device [k/16][n/16][16K] u16 words; suh [k], svh [n] fp16 bits (device); tmp: k x n fp32 scratch.
inline void dequant_linear(sycl::queue& q, const uint16_t* trellis, const uint16_t* suh, const uint16_t* svh, int64_t k,
                           int64_t n, int K, int cb, const int* d_perm, float* tmp, sycl::half* W)
{
    const int64_t tk = k / 16, tn = n / 16;
    const int words = 16 * K, R = 256 * K, W32 = R / 32;
    q.parallel_for(sycl::range<2>(tk * tn, 256), [=](sycl::id<2> id) {
        const int64_t t = id[0];
        const int i = (int) id[1];
        const uint16_t* tw = trellis + t * words;
        auto word = [&](int j) { return (uint32_t) tw[2 * j] | ((uint32_t) tw[2 * j + 1] << 16); };
        int start = ((i + 1) * K - 16) % R; if (start < 0) start += R;
        const int j0 = start / 32, off = start % 32;
        const uint64_t win = ((uint64_t) word(j0) << 32) | word((j0 + 1) % W32);
        const uint32_t s = (uint32_t) (win >> (64 - 16 - off)) & 0xFFFFu;
        float v;
        if (cb == 2) v = (float) dev::decode<2>(s); else if (cb == 1) v = (float) dev::decode<1>(s); else v = (float) dev::decode<0>(s);
        const int e = d_perm[i];
        const int64_t a = t / tn, b = t % tn;
        tmp[(a * 16 + e / 16) * n + b * 16 + e % 16] = v;
    });
    k::had_rows(q, tmp, k, n, n);
    k::had_cols(q, tmp, k, n, n);
    q.parallel_for(sycl::range<2>(k, n), [=](sycl::id<2> id) {
        const float su = (float) sycl::bit_cast<sycl::half>(suh[id[0]]), sv = (float) sycl::bit_cast<sycl::half>(svh[id[1]]);
        W[id[0] * n + id[1]] = sycl::half(tmp[id[0] * n + id[1]] * su * sv);
    }).wait();
}

} // namespace exl3
