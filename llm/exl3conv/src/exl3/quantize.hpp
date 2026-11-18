// exl3conv: per-tensor EXL3 quantization on the B70, after exllamav3 1.5.3 (`quantize_exl3`, `finalize_capture_H`,
// `regularize`, `g_scale_search_batch`, `ldlq`, `refit_scales`; notes/exl3-algorithm.md §3).
//
// Orientation: every weight here is (in = k, out = n), row-major, the transpose of the HF nn.Linear weight, with k and
// n multiples of 128. Tensors that share an input (and so a Hessian: q/k/v, gate/up) form a Group: they get the same
// input signs, their regularized weights sit side by side in one k x N matrix, and LDLQ runs once over all N columns
// (columns are independent given L, so this equals quantizing each one alone, in fewer, larger encoder launches).
//
//   finalize   H <- H / count + 0.025 mean(diag) I (Hd, kept for the refit); Hr = Hk S Hd S Hk with S the input
//              signs; L = block_ldl(cholesky(Hr)). A failed Cholesky adds 2 x 0.025 x mean(diag) and retries (10x).
//   regularize per tensor: output scales (column RMS / mean, signed), output Hadamard, input scales (row RMS divided
//              by -1.24371088, signed), input Hadamard, then the global scale from the sampled-tile search
//   ldlq       bottom-up 16-row blocks in 128-row spans; each block's tiles get the error feedback of the rows below
//              through L, then go through the trellis encoder
//   restore    Q0 = diag(su) Hk Q Hn diag(sv)
//   refit      two rounds of closed-form output scales and a k x k solve for input scales in the Hd metric
//   pack       states -> trellis words; suh, svh rounded to fp16
#pragma once

#include "exl3/codebook.hpp"
#include "exl3/f16.hpp"
#include "exl3/kernels.hpp"
#include "exl3/linalg.hpp"
#include "exl3/pack.hpp"
#include "exl3/viterbi.hpp"
#include "rt/dbuf.hpp"
#include "rt/rt.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

namespace exl3 {

constexpr float SIGMA_REG = 0.025f;

// Counter-based +-1 signs: the same (seed, stream, i) always gives the same sign. exllamav3 uses torch's Philox
// stream; any fixed +-1 vectors make an equally good model.
inline uint64_t mix64(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}
inline uint64_t hash_str(const std::string& s) { uint64_t h = 1469598103934665603ull; for (unsigned char c : s) h = (h ^ c) * 1099511628211ull; return h; }
inline std::vector<float> random_signs(int64_t n, uint64_t seed)
{
    std::vector<float> v(n);
    for (int64_t i = 0; i < n; ++i) v[i] = (mix64(seed ^ mix64((uint64_t) i)) & 1) ? -1.0f : 1.0f;
    return v;
}

// A finalized Hessian group.
struct Hessian
{
    int64_t k = 0;
    DBuf<float> Hd;          // damped H, original basis (refit)
    DBuf<float> L;           // block-LDL factor of the rotated H (LDLQ)
    std::vector<float> su0;  // input signs
    int retries = 0;
};

struct QuantArgs
{
    int K = 4;
    int cb = CB_MUL1;
    bool out_scales = true;  // "always"
    bool out_scales_auto = false;   // "auto" (exllamav3 1.6.1): off for tensors whose QuantIn::out_sens predicts a higher loss with them
    bool refit = true;
    bool feedback = true;    // LDLQ error feedback (off: every block quantized alone; for tests)
    bool keep_L = false;     // keep the Hessian's LDL factor for more groups sharing it (else freed before the refit)
    bool scaled_hessian = false;    // exllamav3's default since 1.6.6: one factorization per tensor with its input scales folded in (scaled_ldl)
    bool verbose = false;
};

// Input: the weight of one tensor, (k x n) fp32, already rounded through fp16 and zero-padded.
// out_sens: optional host array of n per-output-channel sensitivities (loss per unit of error energy), for out_scales_auto.
struct QuantIn { std::string key; int64_t n; const float* W; const float* out_sens = nullptr; };   // W: device pointer, ld = n

struct QuantOut
{
    std::string key;
    int64_t k = 0, n = 0;
    int K = 0;
    std::vector<uint16_t> trellis;     // [k/16][n/16][16K]
    std::vector<uint16_t> suh, svh;    // fp16 bits
    float g_scale = 0, err_before = 0, err_after = 0;
    float out_ratio = -1;    // out_scales_auto: predicted loss with output scales relative to without (-1: not computed)
    bool out_scaled = true;
    double seconds = 0;
};

// L = block_ldl(cholesky(Hk S Hd S Hk)), S = diag(s): the input signs, or (scaled) a tensor's whole input scales. A failed Cholesky
// retries with more damping: signs, on Hd (which the refit then reads too); scales, on the rotated matrix alone, because S would blow
// up a damping on Hd unevenly, and Hd is the refit's metric, as exllamav3's block_ldl.
inline void factor_hessian(sycl::queue& q, Hessian& hs, const std::vector<float>& s, bool scaled = false)
{
    const int64_t k = hs.k;
    float* hd = hs.Hd.get();
    std::vector<float> diag(k);
    DBuf<float> sd(q, k), d(q, k);
    sd.upload(s.data(), k);
    if (!hs.L.get()) hs.L = DBuf<float>(q, (size_t) k * k);
    float* L = hs.L.get();
    float* dp = d.get();
    auto mean_diag = [&](const float* m) {
        q.parallel_for(sycl::range<1>(k), [=](sycl::id<1> i) { dp[i] = m[i * k + i]; }).wait();
        d.download(diag.data(), k);
        double v = 0; for (float x : diag) v += x;
        return v / k;
    };
    auto add_diag = [&](float* m, float v) { q.parallel_for(sycl::range<1>(k), [=](sycl::id<1> i) { m[i * k + i] += v; }).wait(); };
    for (int tries = 0;; ++tries)
    {
        // Hr = Hk S Hd S Hk (S Hd S, then the 128-blockwise Hadamard on both sides; Hd stays as it is).
        q.memcpy(L, hd, sizeof(float) * k * k).wait();
        k::scale(q, L, k, k, k, sd.get(), sd.get()).wait();
        k::had_rows(q, L, k, k, k).wait();
        k::had_cols(q, L, k, k, k).wait();
        if (scaled && tries > 0) add_diag(L, (float) (2.0 * SIGMA_REG * tries * mean_diag(L)));
        if (la::cholesky(q, L, k, k)) break;
        if (tries >= 10) throw std::runtime_error("Cholesky failed after 10 damping retries");
        ++hs.retries;
        std::fprintf(stderr, "     !! Cholesky failed, increasing diagonal damping (attempt %d/10)\n", tries + 1);
        if (!scaled) add_diag(hd, (float) (2.0 * SIGMA_REG * mean_diag(hd)));
    }
    la::block_ldl(q, L, k, k);
}

// H: device k x k fp32 sum of x^T x over `count` rows (consumed: it becomes Hd). seed: per-group signs. factor false: leave L to
// quantize_group (scaled_hessian).
inline Hessian finalize_hessian(sycl::queue& q, DBuf<float>&& H, int64_t k, int64_t count, uint64_t seed, bool factor = true)
{
    Hessian hs;
    hs.k = k;
    hs.Hd = std::move(H);
    float* hd = hs.Hd.get();
    std::vector<float> diag(k);
    auto mean_diag = [&](const float* m) {
        DBuf<float> d(q, k);
        float* dp = d.get();
        q.parallel_for(sycl::range<1>(k), [=](sycl::id<1> i) { dp[i] = m[i * k + i]; }).wait();
        d.download(diag.data(), k);
        double s = 0; for (float v : diag) s += v;
        return s / k;
    };
    const float inv = 1.0f / (float) std::max<int64_t>(count, 1);
    q.parallel_for(sycl::range<1>(k * k), [=](sycl::id<1> i) { hd[i] *= inv; }).wait();
    const double dm = mean_diag(hd);
    if (!(dm > 1e-20) || !std::isfinite(dm)) throw std::runtime_error("degenerate Hessian (mean diagonal " + std::to_string(dm) + ")");
    auto add_diag = [&](float* m, float v) { q.parallel_for(sycl::range<1>(k), [=](sycl::id<1> i) { m[i * k + i] += v; }).wait(); };
    add_diag(hd, (float) (SIGMA_REG * dm));

    hs.su0 = random_signs(k, seed);
    if (factor) factor_hessian(q, hs, hs.su0);
    return hs;
}

// Global scale search on sampled tiles of the regularized weight W4 (k x n slice, ld). Returns g.
inline float scale_search(sycl::queue& q, VitWork& vw, const float* W4, int64_t k, int64_t n, int64_t ld,
                          const QuantArgs& qa, const int* d_perm)
{
    const int64_t tk = k / 16, tn = n / 16;
    // Diagonal tiles: (d % tk, (d + w) % tn), d-major, w = 0..2.
    const int64_t D = std::max(tk, tn);
    std::vector<int> ta, tb;
    for (int64_t d = 0; d < D; ++d) for (int w = 0; w < 3; ++w) { ta.push_back((int) (d % tk)); tb.push_back((int) ((d + w) % tn)); }
    // Extreme tiles by mean square.
    DBuf<float> ms(q, tk * tn);
    k::tile_meansq(q, W4, ld, tk, tn, ms.get()).wait();
    std::vector<float> hms(tk * tn);
    ms.download(hms.data(), tk * tn);
    const int64_t m = std::min<int64_t>(std::max<int64_t>(8, (3 * D) / 16), (tk * tn + 1) / 2);
    std::vector<int> order(tk * tn);
    std::iota(order.begin(), order.end(), 0);
    std::vector<int> hi(order), lo(order);
    std::partial_sort(hi.begin(), hi.begin() + m, hi.end(), [&](int a, int b) { return hms[a] > hms[b]; });
    std::partial_sort(lo.begin(), lo.begin() + m, lo.end(), [&](int a, int b) { return hms[a] < hms[b]; });
    for (int64_t i = 0; i < m; ++i) { ta.push_back((int) (hi[i] / tn)); tb.push_back((int) (hi[i] % tn)); }
    for (int64_t i = 0; i < m; ++i) { ta.push_back((int) (lo[i] / tn)); tb.push_back((int) (lo[i] % tn)); }
    const int64_t nt = (int64_t) ta.size();

    static const double DRIFT[9] = {1, 1.08, 1.018, 1.004, 1, 1, 1, 1, 1};
    DBuf<int> dta(q, nt), dtb(q, nt);
    dta.upload(ta.data(), nt); dtb.upload(tb.data(), nt);
    DBuf<float> tiles(q, nt * 256), tin(q, nt * 256), tout(q, nt * 256);
    DBuf<uint16_t> tidx(q, nt * 256);
    k::gather_tile_list(q, W4, ld, dta.get(), dtb.get(), nt, d_perm, tiles.get(), (float) DRIFT[qa.K]).wait();
    std::vector<float> htiles(nt * 256), hout(nt * 256);
    tiles.download(htiles.data(), nt * 256);

    // Every third tile for the coarse grid: copy them to the front of a second buffer.
    std::vector<int64_t> sub; for (int64_t i = 0; i < nt; i += 3) sub.push_back(i);
    std::vector<float> hsub(sub.size() * 256);
    for (size_t j = 0; j < sub.size(); ++j) std::copy_n(&htiles[sub[j] * 256], 256, &hsub[j * 256]);
    DBuf<float> dsub(q, hsub.size());
    dsub.upload(hsub.data(), hsub.size());

    auto mse = [&](const DBuf<float>& src, const std::vector<float>& hsrc, int64_t cnt, double s) {
        const float sf = (float) s;
        float* ti = tin.get(); const float* sp = src.get();
        q.parallel_for(sycl::range<1>(cnt * 256), [=](sycl::id<1> i) { ti[i] = sp[i] * sf; });
        vw.encode(qa.cb, tin.get(), tidx.get(), tout.get(), cnt).wait();
        tout.download(hout.data(), cnt * 256);
        double e = 0;
        for (int64_t i = 0; i < cnt * 256; ++i) { const double d = hout[i] / s - hsrc[i]; e += d * d; }
        return e / (double) (cnt * 256);
    };
    double best_c = 0.1, best_e = 1e300;
    for (int i = 0; i < 10; ++i)
    {
        const double s = 0.1 + 0.2 * i, e = mse(dsub, hsub, (int64_t) sub.size(), s);
        if (e < best_e) { best_e = e; best_c = s; }
    }
    double fine[5], y[5];
    int b = 0;
    for (int i = 0; i < 5; ++i)
    {
        fine[i] = best_c + 0.075 * (i - 2);
        y[i] = mse(tiles, htiles, nt, fine[i]);
        if (y[i] < y[b]) b = i;
    }
    double off = 0;
    if (b > 0 && b < 4)
    {
        const double den = y[b - 1] - 2 * y[b] + y[b + 1];
        off = den > 0 ? std::clamp(0.5 * (y[b - 1] - y[b + 1]) / den, -0.5, 0.5) : 0.0;
    }
    return (float) std::max(fine[b] + off * 0.075, 0.01);
}

// Quantize a group of tensors that share one Hessian. Every W has k rows; the outputs come back in input order.
inline std::vector<QuantOut> quantize_group(sycl::queue& q, Hessian& hs, const std::vector<QuantIn>& ins,
                                            const QuantArgs& qa, uint64_t seed)
{
    const int64_t k = hs.k, tk = k / 16;
    int64_t N = 0;
    for (auto& t : ins) N += t.n;
    const int64_t tnN = N / 16;
    const auto& perm = tile_perm();
    DBuf<int> d_perm(q, 256);
    d_perm.upload(perm.data(), 256);
    VitWork vw(q, qa.K);

    std::vector<QuantOut> outs(ins.size());
    DBuf<float> W5(q, (size_t) k * N), Qm(q, (size_t) k * N);
    DBuf<float> d_su0(q, k);
    d_su0.upload(hs.su0.data(), k);
    std::vector<std::vector<float>> su(ins.size()), sv(ins.size());
    std::vector<int64_t> off(ins.size());

    // Regularize each tensor into its column slice of W5.
    rt::Timer treg;
    {
        int64_t c0 = 0;
        for (size_t t = 0; t < ins.size(); ++t)
        {
            rt::Timer tm;
            const int64_t n = ins[t].n;
            off[t] = c0;
            outs[t].key = ins[t].key; outs[t].k = k; outs[t].n = n; outs[t].K = qa.K;
            float* w = W5.get() + c0;
            q.ext_oneapi_memcpy2d(w, N * sizeof(float), ins[t].W, n * sizeof(float), n * sizeof(float), k).wait();
            // Output scales.
            DBuf<float> cs(q, n), rs(q, k);
            k::col_sumsq(q, w, k, n, N, cs.get()).wait();
            std::vector<float> r(n);
            cs.download(r.data(), n);
            for (auto& v : r) v = std::sqrt(v / (float) k);
            double mu = 0; for (float v : r) mu += v; mu /= n;
            std::vector<char> zero(n);
            for (int64_t j = 0; j < n; ++j)
            {
                if (mu > 1e-30) r[j] = (float) (r[j] / mu);
                zero[j] = std::fabs(r[j]) < 1e-30f;
            }
            const auto sv0 = random_signs(n, seed ^ mix64(hash_str(ins[t].key)) ^ 0x5356u);
            sv[t].resize(n);
            // Auto: scales give every channel the same relative error, no scales the same absolute error. For
            // sensitivity s and channel energy e = r^2 the loss ratio is E[s e] / (E[s] E[e]) (out_scales_ratio).
            bool use_scales = qa.out_scales || qa.out_scales_auto;
            if (qa.out_scales_auto && ins[t].out_sens && mu > 1e-30)
            {
                double se = 0, s_ = 0, e_ = 0;
                bool finite = true;
                for (int64_t j = 0; j < n; ++j)
                {
                    const double sj = ins[t].out_sens[j], ej = (double) r[j] * r[j];
                    finite &= std::isfinite(sj);
                    se += sj * ej; s_ += sj; e_ += ej;
                }
                const double den = (s_ / n) * (e_ / n);
                if (finite && den > 0)
                {
                    outs[t].out_ratio = (float) ((se / n) / den);
                    use_scales = outs[t].out_ratio <= 1.05f;
                }
            }
            outs[t].out_scaled = use_scales;
            for (int64_t j = 0; j < n; ++j)
            {
                if (use_scales) { if (zero[j]) r[j] = 0.1f; sv[t][j] = sv0[j] * r[j] + 1e-10f; }
                else sv[t][j] = sv0[j];
            }
            cs.upload(sv[t].data(), n);
            k::scale(q, w, k, n, N, nullptr, cs.get(), true).wait();
            for (int64_t j = 0; j < n; ++j) if (zero[j]) sv[t][j] = 0.0f;
            k::had_rows(q, w, k, n, N).wait();
            // Input scales.
            k::row_sumsq(q, w, k, n, N, rs.get()).wait();
            std::vector<float> qv(k);
            rs.download(qv.data(), k);
            su[t].resize(k);
            for (int64_t i = 0; i < k; ++i)
            {
                float s = std::sqrt(qv[i] / (float) n);
                if (s < 1e-30f) s = 0.1f;
                su[t][i] = hs.su0[i] * s / (-CODEBOOK_SCALE) + 1e-10f;
            }
            rs.upload(su[t].data(), k);
            k::scale(q, w, k, n, N, rs.get(), nullptr, true).wait();
            k::had_cols(q, w, k, n, N).wait();
            // Global scale.
            const float g = scale_search(q, vw, w, k, n, N, qa, d_perm.get());
            outs[t].g_scale = g;
            q.parallel_for(sycl::range<2>(k, n), [=](sycl::id<2> id) { w[id[0] * N + id[1]] *= g; }).wait();
            for (auto& v : su[t]) v /= g;
            outs[t].seconds += tm.s();
            c0 += n;
        }
    }

    const double reg_s = treg.s();

    // LDLQ over all N columns, or (scaled_hessian) over each tensor's own with its own factorization.
    rt::Timer tl;
    DBuf<uint16_t> enc(q, (size_t) tk * tnN * 256);
    {
        DBuf<float> prod(q, (size_t) k * N), Espan(q, (size_t) 128 * N), X(q, (size_t) 16 * N);
        DBuf<float> tin(q, (size_t) tnN * 256), tout(q, (size_t) tnN * 256);
        float* w5 = W5.get(); float* qm = Qm.get(); float* pc = prod.get(); float* es = Espan.get(); float* x = X.get();
        auto ldlq = [&](int64_t c0, int64_t C) {
        prod.zero();
        const float* L = hs.L.get();
        const int64_t tnC = C / 16;
        for (int64_t j = k; j > 0; j -= 128)
        {
            const int64_t i = j - 128;
            for (int64_t bj = 128; bj > 0; bj -= 16)
            {
                const int64_t bi = bj - 16;
                // prod[i+bi .. i+bj) += L[i+bj .. j, i+bi .. i+bj)^T @ E[bj .. 128)
                if (bj < 128 && qa.feedback)
                    la::gemm(q, true, false, 16, C, 128 - bj, 1.0f, L + (i + bj) * k + i + bi, k, es + bj * N, N, 1.0f,
                             pc + (i + bi) * N + c0, N);
                // X = W5 rows + feedback
                const int64_t r0 = i + bi;
                q.parallel_for(sycl::range<2>(16, C), [=](sycl::id<2> id) {
                    x[id[0] * N + id[1]] = w5[(r0 + id[0]) * N + c0 + id[1]] + pc[(r0 + id[0]) * N + c0 + id[1]];
                });
                k::gather_tiles(q, x, N, 0, 0, tnC, d_perm.get(), tin.get());
                vw.encode(qa.cb, tin.get(), enc.get() + ((r0 / 16) * tnN + c0 / 16) * 256, tout.get(), tnC);
                k::scatter_tiles(q, tout.get(), d_perm.get(), tnC, qm, N, r0, c0);
                k::sub(q, w5 + r0 * N + c0, N, qm + r0 * N + c0, N, es + bi * N, N, 16, C);
            }
            // prod[0 .. i) += L[i .. j, 0 .. i)^T @ E_span
            if (i > 0 && qa.feedback) la::gemm(q, true, false, i, C, 128, 1.0f, L + i * k, k, es, N, 1.0f, pc + c0, N);
        }
        q.wait();
        };
        if (qa.scaled_hessian)
            for (size_t t = 0; t < ins.size(); ++t)
            {
                factor_hessian(q, hs, su[t], true);
                ldlq(off[t], ins[t].n);
            }
        else
            ldlq(0, N);
    }
    const double ldlq_s = tl.s();
    if (!qa.keep_L) hs.L.reset();          // the k x k buffer is free again for the refit's A
    W5.reset();
    if (qa.verbose) std::printf("      regularize %.1f s, LDLQ %.1f s\n", reg_s, ldlq_s);

    // Restore, refit and pack each tensor.
    for (size_t t = 0; t < ins.size(); ++t)
    {
        rt::Timer tm;
        const int64_t n = ins[t].n, tn = n / 16;
        DBuf<float> Q0(q, (size_t) k * n), dsu(q, k), dsv(q, n);
        q.ext_oneapi_memcpy2d(Q0.get(), n * sizeof(float), Qm.get() + off[t], N * sizeof(float), n * sizeof(float), k).wait();
        float* qq = Q0.get();
        dsu.upload(su[t].data(), k); dsv.upload(sv[t].data(), n);
        k::had_cols(q, qq, k, n, n);
        k::scale(q, qq, k, n, n, dsu.get(), nullptr);
        k::had_rows(q, qq, k, n, n);
        k::scale(q, qq, k, n, n, nullptr, dsv.get()).wait();

        if (qa.refit)
        {
            const float* Hd = hs.Hd.get();
            const float* W0 = ins[t].W;
            DBuf<float> HW(q, (size_t) k * n), HQ(q, (size_t) k * n), cn(q, n), cd(q, n), rk(q, k);
            la::gemm_big(q, k, n, k, Hd, k, W0, n, 0.0f, HW.get(), n);
            k::row_dot(q, W0, n, HW.get(), n, k, n, rk.get()).wait();
            std::vector<float> hk(k), hn(n), hn2(n);
            rk.download(hk.data(), k);
            double den = 0; for (float v : hk) den += v;
            auto err = [&]() {
                k::sub(q, W0, n, qq, n, HQ.get(), n, k, n);              // HQ <- E
                DBuf<float> HE(q, (size_t) k * n);
                la::gemm_big(q, k, n, k, Hd, k, HQ.get(), n, 0.0f, HE.get(), n);
                k::row_dot(q, HQ.get(), n, HE.get(), n, k, n, rk.get()).wait();
                rk.download(hk.data(), k);
                double e = 0; for (float v : hk) e += v;
                return e / std::max(den, 1e-30);
            };
            outs[t].err_before = (float) err();
            DBuf<float> A(q, (size_t) k * k);
            for (int round = 0; round < 2; ++round)
            {
                // Output side.
                la::gemm_big(q, k, n, k, Hd, k, qq, n, 0.0f, HQ.get(), n);
                k::col_dot(q, qq, n, HW.get(), n, k, n, cn.get());
                k::col_dot(q, qq, n, HQ.get(), n, k, n, cd.get()).wait();
                cn.download(hn.data(), n); cd.download(hn2.data(), n);
                for (int64_t j = 0; j < n; ++j) { const float c = hn2[j] > 1e-30f ? hn[j] / hn2[j] : 1.0f; hn[j] = c; sv[t][j] *= c; }
                cn.upload(hn.data(), n);
                k::scale(q, qq, k, n, n, nullptr, cn.get()).wait();
                // Input side: ((Q Q^T) o Hd + 1e-6 mean(diag) I) r = rowsum(Q o HW)
                float* a = A.get();
                if (k * k * n > la::GEMM_SPLIT) la::gemm_aat_lower(q, k, n, qq, n, a, k, 8);
                else la::gemm(q, false, true, k, k, n, 1.0f, qq, n, qq, n, 0.0f, a, k);
                q.parallel_for(sycl::range<1>(k * k), [=](sycl::id<1> i) { a[i] *= Hd[i]; }).wait();
                DBuf<float> dg(q, k);
                float* dgp = dg.get();
                q.parallel_for(sycl::range<1>(k), [=](sycl::id<1> i) { dgp[i] = a[i * k + i]; }).wait();
                dg.download(hk.data(), k);
                double md = 0; for (float v : hk) md += v; md /= k;
                const float eps = (float) (1e-6 * md);
                q.parallel_for(sycl::range<1>(k), [=](sycl::id<1> i) { a[i * k + i] += eps; }).wait();
                k::row_dot(q, qq, n, HW.get(), n, k, n, rk.get()).wait();
                std::vector<float> r(k, 1.0f);
                if (la::cholesky(q, a, k, k))
                {
                    la::chol_solve(q, a, k, k, rk.get());
                    rk.download(r.data(), k);
                }
                else std::fprintf(stderr, "     !! refit: input-scale system not positive definite, skipped\n");
                for (int64_t i2 = 0; i2 < k; ++i2) { if (!std::isfinite(r[i2]) || r[i2] <= 0) r[i2] = 1.0f; su[t][i2] *= r[i2]; }
                rk.upload(r.data(), k);
                k::scale(q, qq, k, n, n, rk.get(), nullptr).wait();
            }
            A.reset();
            outs[t].err_after = (float) err();
        }

        // Pack.
        std::vector<uint16_t> states((size_t) tk * tn * 256);
        {
            std::vector<uint16_t> row((size_t) tnN * 256);
            for (int64_t a = 0; a < tk; ++a)
            {
                q.memcpy(row.data(), enc.get() + a * tnN * 256, row.size() * sizeof(uint16_t)).wait();
                std::copy_n(&row[(off[t] / 16) * 256], tn * 256, &states[a * tn * 256]);
            }
        }
        const Rate rate = Rate::integer(qa.K);
        const int words = rate.words16();
        outs[t].trellis.resize((size_t) tk * tn * words);
        for (int64_t i2 = 0; i2 < tk * tn; ++i2) pack_tile(&states[i2 * 256], &outs[t].trellis[i2 * words], rate);
        outs[t].suh.resize(k); outs[t].svh.resize(n);
        for (int64_t i2 = 0; i2 < k; ++i2) outs[t].suh[i2] = f16_from_float(su[t][i2]);
        for (int64_t j = 0; j < n; ++j) outs[t].svh[j] = f16_from_float(sv[t][j]);
        outs[t].seconds += tm.s() + ldlq_s * (double) n / (double) N;
    }
    return outs;
}

} // namespace exl3
