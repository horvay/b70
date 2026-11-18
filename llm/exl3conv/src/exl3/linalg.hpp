// exl3conv: dense linear algebra on the B70 for the quantizer. Row-major fp32 with explicit leading dimensions.
//
//   gemm        C = alpha op(A) op(B) + beta C (oneMKL SYCL BLAS, fp32, standard compute mode: no TF32)
//   cholesky    in-place lower Cholesky A = L L^T, right-looking and blocked (nb = 256): the small diagonal blocks are
//               factored on the CPU in fp64 (host LAPACKE_dpotrf), the panel (trsm) and trailing update (syrk) run
//               on the card. No device LAPACK is needed (oneMKL ships it here only as a 934 MB static library).
//               Returns false when A is not positive definite; the caller adds damping and retries.
//   chol_solve  x = A^-1 b from the Cholesky factor (two trsm)
//   block_ldl   turn a Cholesky factor L into the strictly block-lower LDLQ matrix (exllamav3 block_ldl, b = 16):
//               every block column times the inverse of its 16 x 16 diagonal block, then the diagonal blocks zeroed.
#pragma once

#include <oneapi/mkl/blas.hpp>
#include <mkl_lapacke.h>
#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace exl3::la {

namespace blas = oneapi::mkl::blas::row_major;
using oneapi::mkl::transpose;
using oneapi::mkl::uplo;
using oneapi::mkl::side;
using oneapi::mkl::diag;

inline sycl::event gemm(sycl::queue& q, bool ta, bool tb, int64_t m, int64_t n, int64_t k, float alpha, const float* a,
                        int64_t lda, const float* b, int64_t ldb, float beta, float* c, int64_t ldc)
{
    if (m == 0 || n == 0) return {};
    return blas::gemm(q, ta ? transpose::trans : transpose::nontrans, tb ? transpose::trans : transpose::nontrans, m, n,
                      k, alpha, a, lda, b, ldb, beta, c, ldc);
}

// Past this many multiply-adds a GEMM is split, each launch well under the xe driver's 5 s job timeout. Below it (every
// Gemma tensor) it is one plain gemm.
constexpr int64_t GEMM_SPLIT = 1000000000000ll;

// C = A B + beta C, no transposes, split over C's columns when it is large.
inline sycl::event gemm_big(sycl::queue& q, int64_t m, int64_t n, int64_t k, const float* a, int64_t lda, const float* b,
                            int64_t ldb, float beta, float* c, int64_t ldc)
{
    if (m == 0 || n == 0) return {};
    const int64_t step = std::max<int64_t>(1, GEMM_SPLIT / (m * k));
    sycl::event ev;
    for (int64_t j = 0; j < n; j += step)
        ev = gemm(q, false, false, m, std::min(step, n - j), k, 1.0f, a, lda, b + j, ldb, beta, c + j, ldc);
    return ev;
}

// C = A A^T, lower triangle only (whole block columns up to the diagonal: the upper part of each diagonal block is set too,
// the rest of the upper triangle is left alone), for a large A; the Cholesky below reads the lower triangle only.
inline sycl::event gemm_aat_lower(sycl::queue& q, int64_t m, int64_t k, const float* a, int64_t lda, float* c, int64_t ldc, int blocks)
{
    const int64_t bs = m / blocks;
    sycl::event ev;
    for (int i = 0; i < blocks; ++i)
        for (int j = 0; j <= i; ++j)
            ev = gemm(q, false, true, bs, bs, k, 1.0f, a + i * bs * lda, lda, a + j * bs * lda, lda, 0.0f, c + i * bs * ldc + j * bs, ldc);
    return ev;
}

// Zero the strict upper triangle of an n x n matrix.
inline sycl::event zero_upper(sycl::queue& q, float* a, int64_t n, int64_t lda)
{
    return q.parallel_for(sycl::range<2>(n, n), [=](sycl::id<2> id) {
        if (id[1] > id[0]) a[id[0] * lda + id[1]] = 0.0f;
    });
}

inline bool cholesky(sycl::queue& q, float* a, int64_t n, int64_t lda, int64_t nb = 256)
{
    float* dblk = sycl::malloc_device<float>(nb * nb, q);
    std::vector<float> hf(nb * nb);
    std::vector<double> hd(nb * nb);
    bool ok = true;
    for (int64_t j0 = 0; j0 < n && ok; j0 += nb)
    {
        const int64_t jb = std::min(nb, n - j0);
        float* d = a + j0 * lda + j0;
        // Diagonal block to the host, factor in fp64, back.
        q.parallel_for(sycl::range<2>(jb, jb), [=](sycl::id<2> id) { dblk[id[0] * jb + id[1]] = d[id[0] * lda + id[1]]; });
        q.memcpy(hf.data(), dblk, jb * jb * sizeof(float)).wait();
        for (int64_t i = 0; i < jb * jb; ++i) hd[i] = hf[i];
        const lapack_int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', (lapack_int) jb, hd.data(), (lapack_int) jb);
        if (info != 0) { ok = false; break; }
        for (int64_t r = 0; r < jb; ++r)
            for (int64_t c = 0; c < jb; ++c) hf[r * jb + c] = c <= r ? (float) hd[r * jb + c] : 0.0f;
        q.memcpy(dblk, hf.data(), jb * jb * sizeof(float)).wait();
        q.parallel_for(sycl::range<2>(jb, jb), [=](sycl::id<2> id) { d[id[0] * lda + id[1]] = dblk[id[0] * jb + id[1]]; });
        const int64_t m = n - j0 - jb;
        if (m > 0)
        {
            float* a21 = a + (j0 + jb) * lda + j0;
            float* a22 = a + (j0 + jb) * lda + j0 + jb;
            // A21 <- A21 L11^-T
            blas::trsm(q, side::right, uplo::lower, transpose::trans, diag::nonunit, m, jb, 1.0f, d, lda, a21, lda);
            // A22 <- A22 - A21 A21^T (lower triangle)
            blas::syrk(q, uplo::lower, transpose::nontrans, m, jb, -1.0f, a21, lda, 1.0f, a22, lda);
        }
    }
    q.wait();
    sycl::free(dblk, q);
    if (ok) zero_upper(q, a, n, lda).wait();
    return ok;
}

// b (n) <- A^-1 b, with L the Cholesky factor of A.
inline void chol_solve(sycl::queue& q, const float* L, int64_t n, int64_t ldl, float* b)
{
    blas::trsm(q, side::left, uplo::lower, transpose::nontrans, diag::nonunit, n, 1, 1.0f, L, ldl, b, 1);
    blas::trsm(q, side::left, uplo::lower, transpose::trans, diag::nonunit, n, 1, 1.0f, L, ldl, b, 1).wait();
}

// L (k x k, lower Cholesky factor) -> strictly block-lower-triangular LDLQ factor, in place.
inline void block_ldl(sycl::queue& q, float* L, int64_t k, int64_t ld)
{
    const int64_t m = k / 16;
    float* dg = sycl::malloc_device<float>(m * 256, q);
    q.parallel_for(sycl::range<2>(m, 256), [=](sycl::id<2> id) {
        const int64_t i = id[0]; const int r = (int) id[1] / 16, c = (int) id[1] % 16;
        dg[i * 256 + id[1]] = L[(i * 16 + r) * ld + i * 16 + c];
    }).wait();
    std::vector<float> h(m * 256);
    q.memcpy(h.data(), dg, m * 256 * sizeof(float)).wait();
    // Invert each lower-triangular 16 x 16 block in fp64 (forward substitution on the identity).
    for (int64_t i = 0; i < m; ++i)
    {
        double D[16][16], X[16][16] = {};
        for (int r = 0; r < 16; ++r) for (int c = 0; c < 16; ++c) D[r][c] = h[i * 256 + r * 16 + c];
        for (int col = 0; col < 16; ++col)
            for (int r = 0; r < 16; ++r)
            {
                double s = (r == col) ? 1.0 : 0.0;
                for (int t = 0; t < r; ++t) s -= D[r][t] * X[t][col];
                X[r][col] = s / D[r][r];
            }
        for (int r = 0; r < 16; ++r) for (int c = 0; c < 16; ++c) h[i * 256 + r * 16 + c] = (float) X[r][c];
    }
    q.memcpy(dg, h.data(), m * 256 * sizeof(float)).wait();
    // Every row r, every block column i with rows in or below block i: L[r, blk i] <- L[r, blk i] @ Dinv_i. In pieces of
    // rows: a big k would outlive the xe driver's job timeout in one launch.
    constexpr int64_t ROWS = 8192;
    for (int64_t r0 = 0; r0 < k; r0 += ROWS)
    {
        const int64_t nr = std::min(ROWS, k - r0);
        q.parallel_for(sycl::range<2>(nr, m), [=](sycl::id<2> id) {
            const int64_t r = r0 + id[0], i = id[1];
            if (r < i * 16) return;
            float* row = L + r * ld + i * 16;
            float in[16], out[16] = {};
            for (int t = 0; t < 16; ++t) in[t] = row[t];
            for (int t = 0; t < 16; ++t)
                for (int c = 0; c < 16; ++c) out[c] += in[t] * dg[i * 256 + t * 16 + c];
            if (r < (i + 1) * 16)
                for (int c = 0; c < 16; ++c) row[c] = 0.0f;       // diagonal block: identity, then L[r, r] = 0
            else
                for (int c = 0; c < 16; ++c) row[c] = out[c];
        });
    }
    q.wait();
    sycl::free(dg, q);
}

} // namespace exl3::la
