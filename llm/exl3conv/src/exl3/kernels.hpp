// exl3conv: small device kernels for the quantizer. Matrices are row-major fp32 with an explicit leading dimension
// (ld, in elements) so a tensor can be a column slice of a wider matrix (q/k/v quantized side by side).
//
//   had_rows   X[r, 128b .. 128b+127] <- H128 x (the output-side / "had_r" transform), every row, every block
//   had_cols   X[128b .. 128b+127, c] <- H128 x (the input-side / "had_l" transform), every column, every block
//              H128 = Sylvester(128) / sqrt(128), natural order: H[i][j] = (-1)^popcount(i & j) / sqrt(128).
//              Both run a radix-2 butterfly in SLM on 16 vectors per work-group, in fp32.
//   scale      X[r, c] *= row[r] (optional) * col[c] (optional), or divides
//   sumsq      column or row sums of squares (fp32 accumulation)
//   tiles      gather 16 x 16 tiles (row-major inside the tile) into ring order, and scatter decoded ones back
#pragma once

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdint>

namespace exl3::k {

constexpr float INV_SQRT128 = 0.088388347648318440550f;

// 16 vectors of 128 elements per work-group; vector v starts at base(v), element i at base(v) + i * stride.
template <typename BaseFn>
inline sycl::event had128(sycl::queue& q, float* x, int64_t nvec, int64_t stride, BaseFn base)
{
    constexpr int NV = 16, WG = 256;
    const int64_t groups = (nvec + NV - 1) / NV;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> s(sycl::range<1>(NV * 128), h);
        h.parallel_for(sycl::nd_range<1>(groups * WG, WG), [=](sycl::nd_item<1> it) {
            const int64_t g = it.get_group(0);
            const int t = (int) it.get_local_id(0);
            // Load: element e = v * 128 + i.
            for (int e = t; e < NV * 128; e += WG)
            {
                const int v = e / 128, i = e % 128;
                const int64_t vi = g * NV + v;
                s[e] = vi < nvec ? x[base(vi) + i * stride] : 0.0f;
            }
            sycl::group_barrier(it.get_group());
            for (int hh = 1; hh < 128; hh <<= 1)
            {
                for (int p = t; p < NV * 64; p += WG)
                {
                    const int v = p / 64, j = p % 64;
                    const int lo = (j / hh) * (2 * hh) + (j % hh);
                    const float a = s[v * 128 + lo], b = s[v * 128 + lo + hh];
                    s[v * 128 + lo] = a + b;
                    s[v * 128 + lo + hh] = a - b;
                }
                sycl::group_barrier(it.get_group());
            }
            for (int e = t; e < NV * 128; e += WG)
            {
                const int v = e / 128, i = e % 128;
                const int64_t vi = g * NV + v;
                if (vi < nvec) x[base(vi) + i * stride] = s[e] * INV_SQRT128;
            }
        });
    });
}

// Past this many elements a Hadamard is split into several launches, each well under the xe driver's 5 s job timeout.
constexpr int64_t HAD_SPLIT = 1ll << 28;

// Output-side transform of an (rows x cols) block: every row's 128-column blocks. cols % 128 == 0.
inline sycl::event had_rows(sycl::queue& q, float* x, int64_t rows, int64_t cols, int64_t ld)
{
    const int64_t nb = cols / 128;
    const int64_t step = std::max<int64_t>(1, HAD_SPLIT / cols);
    sycl::event ev;
    for (int64_t r = 0; r < rows; r += step)
    {
        float* xr = x + r * ld;
        ev = had128(q, xr, std::min(step, rows - r) * nb, 1, [=](int64_t v) { return (v / nb) * ld + (v % nb) * 128; });
    }
    return ev;
}

// Input-side transform: every column's 128-row blocks. rows % 128 == 0. Consecutive vectors are consecutive
// columns, so a work-group's loads of one element index are contiguous.
inline sycl::event had_cols(sycl::queue& q, float* x, int64_t rows, int64_t cols, int64_t ld)
{
    const int64_t step = std::max<int64_t>(128, (HAD_SPLIT / cols) / 128 * 128);
    sycl::event ev;
    for (int64_t r = 0; r < rows; r += step)
    {
        float* xr = x + r * ld;
        const int64_t nb = std::min(step, rows - r) / 128;
        ev = had128(q, xr, nb * cols, ld, [=](int64_t v) { return (v / cols) * 128 * ld + (v % cols); });
    }
    return ev;
}

// X[r, c] = X[r, c] * rs[r] * cs[c] (null = 1); with div, divides instead.
inline sycl::event scale(sycl::queue& q, float* x, int64_t rows, int64_t cols, int64_t ld, const float* rs,
                         const float* cs, bool div = false)
{
    return q.parallel_for(sycl::range<2>(rows, cols), [=](sycl::id<2> id) {
        const int64_t r = id[0], c = id[1];
        float f = 1.0f;
        if (rs) f *= rs[r];
        if (cs) f *= cs[c];
        float& v = x[r * ld + c];
        v = div ? v / f : v * f;
    });
}

// out[c] = sum_r X[r, c]^2 (one work-item per column, rows in order).
inline sycl::event col_sumsq(sycl::queue& q, const float* x, int64_t rows, int64_t cols, int64_t ld, float* out)
{
    return q.parallel_for(sycl::range<1>(cols), [=](sycl::id<1> c) {
        float s = 0.0f;
        for (int64_t r = 0; r < rows; ++r) { const float v = x[r * ld + c]; s += v * v; }
        out[c] = s;
    });
}

// out[r] = sum_c X[r, c]^2 (one work-group per row).
inline sycl::event row_sumsq(sycl::queue& q, const float* x, int64_t rows, int64_t cols, int64_t ld, float* out)
{
    constexpr int WG = 256;
    return q.parallel_for(sycl::nd_range<1>(rows * WG, WG), [=](sycl::nd_item<1> it) {
        const int64_t r = it.get_group(0);
        float s = 0.0f;
        for (int64_t c = it.get_local_id(0); c < cols; c += WG) { const float v = x[r * ld + c]; s += v * v; }
        s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
        if (it.get_local_id(0) == 0) out[r] = s;
    });
}

// out[i] = sum over the whole of a .* b (rows x cols blocks with their own ld), split into per-row partials.
inline sycl::event row_dot(sycl::queue& q, const float* a, int64_t lda, const float* b, int64_t ldb, int64_t rows,
                           int64_t cols, float* out)
{
    constexpr int WG = 256;
    return q.parallel_for(sycl::nd_range<1>(rows * WG, WG), [=](sycl::nd_item<1> it) {
        const int64_t r = it.get_group(0);
        float s = 0.0f;
        for (int64_t c = it.get_local_id(0); c < cols; c += WG) s += a[r * lda + c] * b[r * ldb + c];
        s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
        if (it.get_local_id(0) == 0) out[r] = s;
    });
}

// out[c] = sum_r a[r, c] * b[r, c]
inline sycl::event col_dot(sycl::queue& q, const float* a, int64_t lda, const float* b, int64_t ldb, int64_t rows,
                           int64_t cols, float* out)
{
    return q.parallel_for(sycl::range<1>(cols), [=](sycl::id<1> c) {
        float s = 0.0f;
        for (int64_t r = 0; r < rows; ++r) s += a[r * lda + c] * b[r * ldb + c];
        out[c] = s;
    });
}

// Tile gather: tiles of the 16-row block starting at row r0, columns [c0, c0 + 16 * nt), into out[nt][256] in ring
// order (perm[i] = row-major element of ring position i), each value * mul.
inline sycl::event gather_tiles(sycl::queue& q, const float* x, int64_t ld, int64_t r0, int64_t c0, int64_t nt,
                                const int* perm, float* out, float mul = 1.0f)
{
    return q.parallel_for(sycl::range<2>(nt, 256), [=](sycl::id<2> id) {
        const int64_t t = id[0];
        const int i = (int) id[1], e = perm[i], r = e / 16, c = e % 16;
        out[t * 256 + i] = x[(r0 + r) * ld + c0 + t * 16 + c] * mul;
    });
}

// Scatter decoded ring-order tiles back into rows r0 .. r0 + 15.
inline sycl::event scatter_tiles(sycl::queue& q, const float* tiles, const int* perm, int64_t nt, float* x, int64_t ld,
                                 int64_t r0, int64_t c0)
{
    return q.parallel_for(sycl::range<2>(nt, 256), [=](sycl::id<2> id) {
        const int64_t t = id[0];
        const int i = (int) id[1], e = perm[i], r = e / 16, c = e % 16;
        x[(r0 + r) * ld + c0 + t * 16 + c] = tiles[t * 256 + i];
    });
}

// Gather arbitrary tiles (tile row ta[j], tile column tb[j]) of a matrix into ring order.
inline sycl::event gather_tile_list(sycl::queue& q, const float* x, int64_t ld, const int* ta, const int* tb, int64_t nt,
                                    const int* perm, float* out, float mul)
{
    return q.parallel_for(sycl::range<2>(nt, 256), [=](sycl::id<2> id) {
        const int64_t j = id[0];
        const int i = (int) id[1], e = perm[i], r = e / 16, c = e % 16;
        out[j * 256 + i] = x[((int64_t) ta[j] * 16 + r) * ld + (int64_t) tb[j] * 16 + c] * mul;
    });
}

// Mean square of every 16 x 16 tile: out[a * tn + b].
inline sycl::event tile_meansq(sycl::queue& q, const float* x, int64_t ld, int64_t tk, int64_t tn, float* out)
{
    return q.parallel_for(sycl::range<2>(tk, tn), [=](sycl::id<2> id) {
        const int64_t a = id[0], b = id[1];
        float s = 0.0f;
        for (int r = 0; r < 16; ++r)
            for (int c = 0; c < 16; ++c) { const float v = x[(a * 16 + r) * ld + b * 16 + c]; s += v * v; }
        out[a * tn + b] = s / 256.0f;
    });
}

// y = a - b, elementwise over a (rows x cols) block.
inline sycl::event sub(sycl::queue& q, const float* a, int64_t lda, const float* b, int64_t ldb, float* y, int64_t ldy,
                       int64_t rows, int64_t cols)
{
    return q.parallel_for(sycl::range<2>(rows, cols), [=](sycl::id<2> id) {
        y[id[0] * ldy + id[1]] = a[id[0] * lda + id[1]] - b[id[0] * ldb + id[1]];
    });
}

} // namespace exl3::k
