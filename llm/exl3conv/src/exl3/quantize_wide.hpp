// exl3conv: one very wide tensor (an lm_head of 248,320 outputs) quantized in column chunks, so the card holds one chunk's
// working set (~2 GB) instead of quantize_group's six or seven copies of the whole weight (~30 GB for a 5120 x 248,320 head).
//
// The result is one EXL3 tensor, the same as quantize_group's for it alone with output scales on: every step that couples
// columns is done over all of them. Output channel scales need the mean of all columns' rms (pass 1); the input scales sum
// over all columns (pass 2); LDLQ is column-separable given the shared LDL factor (pass 4); the refit's output side is per
// column and its input side's system ((Q Q^T) o H) r = rowsum(Q o H W) is a sum over columns (pass 5). The one difference: the
// global scale is the median of a scale search per chunk instead of one search over tiles sampled from the whole matrix.
// The pre-refit Q (k x n fp32) waits in host memory between passes.
#pragma once

#include "exl3/quantize.hpp"

#include <algorithm>
#include <functional>

namespace exl3 {

// src(c0, C, W): columns [c0, c0 + C) of the weight, (in, out) layout, into W (k x C fp32, ld C). n % chunk == 0 is not needed,
// but every chunk is a multiple of 128 columns except possibly the last, which must still be a multiple of 16 (n % 128 == 0 here).
inline QuantOut quantize_wide(sycl::queue& q, Hessian& hs, const std::string& key, int64_t n,
                              const std::function<void(int64_t, int64_t, float*)>& src, const QuantArgs& qa, uint64_t seed, int64_t chunk = 16384)
{
    const int64_t k = hs.k, tk = k / 16, tn = n / 16;
    if (n % 128 || chunk % 128) throw std::runtime_error(key + ": quantize_wide needs multiples of 128");
    rt::Timer total;
    QuantOut out;
    out.key = key; out.k = k; out.n = n; out.K = qa.K;
    const auto& perm = tile_perm();
    DBuf<int> d_perm(q, 256);
    d_perm.upload(perm.data(), 256);
    VitWork vw(q, qa.K);
    const int64_t CM = std::min(chunk, n);
    DBuf<float> W(q, (size_t) k * CM), dv(q, CM), dk(q, k);
    std::vector<std::pair<int64_t, int64_t>> chunks;
    for (int64_t c0 = 0; c0 < n; c0 += CM) chunks.push_back({c0, std::min(CM, n - c0)});

    // Pass 1: output channel scales over all columns.
    std::vector<float> r(n);
    for (auto [c0, C] : chunks)
    {
        src(c0, C, W.get());
        k::col_sumsq(q, W.get(), k, C, C, dv.get()).wait();
        dv.download(r.data() + c0, C);
    }
    for (auto& v : r) v = std::sqrt(v / (float) k);
    double mu = 0; for (float v : r) mu += v; mu /= n;
    const auto sv0 = random_signs(n, seed ^ mix64(hash_str(key)) ^ 0x5356u);
    std::vector<float> sv(n);
    std::vector<char> zero(n);
    for (int64_t j = 0; j < n; ++j)
    {
        if (mu > 1e-30) r[j] = (float) (r[j] / mu);
        zero[j] = std::fabs(r[j]) < 1e-30f;
        if (zero[j]) r[j] = 0.1f;
        sv[j] = sv0[j] * r[j] + 1e-10f;
    }
    out.out_scaled = true;

    // Pass 2: input scales over all columns.
    std::vector<double> qv(k, 0.0);
    {
        std::vector<float> part(k);
        for (auto [c0, C] : chunks)
        {
            src(c0, C, W.get());
            dv.upload(sv.data() + c0, C);
            k::scale(q, W.get(), k, C, C, nullptr, dv.get(), true).wait();
            k::had_rows(q, W.get(), k, C, C).wait();
            k::row_sumsq(q, W.get(), k, C, C, dk.get()).wait();
            dk.download(part.data(), k);
            for (int64_t i = 0; i < k; ++i) qv[i] += part[i];
        }
    }
    std::vector<float> su(k);
    for (int64_t i = 0; i < k; ++i)
    {
        float s = (float) std::sqrt(qv[i] / (double) n);
        if (s < 1e-30f) s = 0.1f;
        su[i] = hs.su0[i] * s / (-CODEBOOK_SCALE) + 1e-10f;
    }
    // A chunk regularized: output scales, output Hadamard, input scales, input Hadamard (then * g when g is known). su_reg: the input
    // scales before the global scale is divided out of su (quantize_group regularizes with those too).
    const std::vector<float> su_reg = su;
    auto regularize = [&](int64_t c0, int64_t C, float g) {
        src(c0, C, W.get());
        dv.upload(sv.data() + c0, C);
        k::scale(q, W.get(), k, C, C, nullptr, dv.get(), true).wait();
        k::had_rows(q, W.get(), k, C, C).wait();
        dk.upload(su_reg.data(), k);
        k::scale(q, W.get(), k, C, C, dk.get(), nullptr, true).wait();
        k::had_cols(q, W.get(), k, C, C).wait();
        if (g != 1.0f) { float* w = W.get(); q.parallel_for(sycl::range<1>(k * C), [=](sycl::id<1> i) { w[i] *= g; }).wait(); }
    };

    // Pass 3: the global scale, the median over the chunks' searches.
    {
        std::vector<float> gs;
        for (auto [c0, C] : chunks)
        {
            regularize(c0, C, 1.0f);
            gs.push_back(scale_search(q, vw, W.get(), k, C, C, qa, d_perm.get()));
        }
        std::vector<float> s = gs;
        std::nth_element(s.begin(), s.begin() + s.size() / 2, s.end());
        out.g_scale = s[s.size() / 2];
        const auto [mn, mx] = std::minmax_element(gs.begin(), gs.end());
        if (qa.verbose) std::printf("      %s: global scale %.4f (chunks %.4f .. %.4f)\n", key.c_str(), out.g_scale, *mn, *mx);
    }
    const float g = out.g_scale;
    for (auto& v : su) v /= g;

    // Pass 4: LDLQ per chunk, the encoder's states kept on the host, the restored Q (pre-refit) too. All-zero columns keep the
    // scale 0.1 to regularize with and get 0 in the stored scales (svz), as quantize_group.
    std::vector<float> svz(sv);
    for (int64_t j = 0; j < n; ++j) if (zero[j]) svz[j] = 0.0f;
    std::vector<uint16_t> states((size_t) tk * tn * 256);
    std::vector<float> Q0h((size_t) k * n);
    {
        rt::Timer tl;
        DBuf<float> Qm(q, (size_t) k * CM), prod(q, (size_t) k * CM), Espan(q, (size_t) 128 * CM), X(q, (size_t) 16 * CM);
        DBuf<float> tin(q, (size_t) (CM / 16) * 256), tout(q, (size_t) (CM / 16) * 256);
        DBuf<uint16_t> enc(q, (size_t) tk * (CM / 16) * 256);
        std::vector<uint16_t> encd((size_t) tk * (CM / 16) * 256);
        DBuf<float> dsu(q, k);
        dsu.upload(su.data(), k);
        for (auto [c0, C] : chunks)
        {
            regularize(c0, C, g);
            const int64_t N = C, tnC = C / 16;
            float* w5 = W.get(); float* qm = Qm.get(); float* pc = prod.get(); float* es = Espan.get(); float* x = X.get();
            prod.zero();
            const float* L = hs.L.get();
            for (int64_t j = k; j > 0; j -= 128)
            {
                const int64_t i = j - 128;
                for (int64_t bj = 128; bj > 0; bj -= 16)
                {
                    const int64_t bi = bj - 16;
                    if (bj < 128 && qa.feedback)
                        la::gemm(q, true, false, 16, C, 128 - bj, 1.0f, L + (i + bj) * k + i + bi, k, es + bj * N, N, 1.0f, pc + (i + bi) * N, N);
                    const int64_t r0 = i + bi;
                    q.parallel_for(sycl::range<2>(16, C), [=](sycl::id<2> id) {
                        x[id[0] * N + id[1]] = w5[(r0 + id[0]) * N + id[1]] + pc[(r0 + id[0]) * N + id[1]];
                    });
                    k::gather_tiles(q, x, N, 0, 0, tnC, d_perm.get(), tin.get());
                    vw.encode(qa.cb, tin.get(), enc.get() + (r0 / 16) * tnC * 256, tout.get(), tnC);
                    k::scatter_tiles(q, tout.get(), d_perm.get(), tnC, qm, N, r0, 0);
                    k::sub(q, w5 + r0 * N, N, qm + r0 * N, N, es + bi * N, N, 16, C);
                }
                if (i > 0 && qa.feedback) la::gemm(q, true, false, i, C, 128, 1.0f, L + i * k, k, es, N, 1.0f, pc, N);
            }
            q.wait();
            enc.download(encd.data(), (size_t) tk * tnC * 256);
            for (int64_t a = 0; a < tk; ++a)
                std::copy_n(&encd[(size_t) a * tnC * 256], tnC * 256, &states[((size_t) a * tn + c0 / 16) * 256]);
            // Restore: Q0 = Sv Hr Su Hc (Qm), as quantize_group.
            dv.upload(svz.data() + c0, C);
            k::had_cols(q, qm, k, C, C);
            k::scale(q, qm, k, C, C, dsu.get(), nullptr);
            k::had_rows(q, qm, k, C, C);
            k::scale(q, qm, k, C, C, nullptr, dv.get()).wait();
            for (int64_t a = 0; a < k; ++a) q.memcpy(&Q0h[(size_t) a * n + c0], qm + a * C, sizeof(float) * C);
            q.wait();
        }
        if (qa.verbose) std::printf("      %s: LDLQ over %zu chunks in %.1f s\n", key.c_str(), chunks.size(), tl.s());
    }
    if (!qa.keep_L) hs.L.reset();
    sv = svz;

    // Pass 5: refit. Q = diag(rt) Q0 diag(ct): rt, ct the refit's input and output factors so far.
    if (qa.refit)
    {
        rt::Timer tr;
        const float* Hd = hs.Hd.get();
        std::vector<float> rtot(k, 1.0f), ctot(n, 1.0f);
        DBuf<float> Q(q, (size_t) k * CM), HW(q, (size_t) k * CM), HQ(q, (size_t) k * CM), A(q, (size_t) k * k), dr(q, k), dc(q, CM);
        std::vector<float> hk(k);
        auto load_Q = [&](int64_t c0, int64_t C) {
            for (int64_t a = 0; a < k; ++a) q.memcpy(Q.get() + a * C, &Q0h[(size_t) a * n + c0], sizeof(float) * C);
            dr.upload(rtot.data(), k); dc.upload(ctot.data() + c0, C);
            k::scale(q, Q.get(), k, C, C, dr.get(), dc.get()).wait();
        };
        auto sum_rows = [&]() { dk.download(hk.data(), k); double s = 0; for (float v : hk) s += v; return s; };
        // tr(W^T H W) and tr(E^T H E), E = W - Q, over all chunks.
        auto err = [&]() {
            double num = 0, den = 0;
            for (auto [c0, C] : chunks)
            {
                src(c0, C, W.get());
                la::gemm_big(q, k, C, k, Hd, k, W.get(), C, 0.0f, HW.get(), C);
                k::row_dot(q, W.get(), C, HW.get(), C, k, C, dk.get()).wait();
                den += sum_rows();
                load_Q(c0, C);
                k::sub(q, W.get(), C, Q.get(), C, HQ.get(), C, k, C);          // HQ <- E
                la::gemm_big(q, k, C, k, Hd, k, HQ.get(), C, 0.0f, HW.get(), C); // HW <- H E
                k::row_dot(q, HQ.get(), C, HW.get(), C, k, C, dk.get()).wait();
                num += sum_rows();
            }
            return num / std::max(den, 1e-30);
        };
        out.err_before = (float) err();
        for (int round = 0; round < 2; ++round)
        {
            A.zero();
            std::vector<double> b(k, 0.0);
            std::vector<float> hn(CM), hd2(CM);
            for (auto [c0, C] : chunks)
            {
                src(c0, C, W.get());
                la::gemm_big(q, k, C, k, Hd, k, W.get(), C, 0.0f, HW.get(), C);
                load_Q(c0, C);
                // Output side: c = <q_j, H w_j> / <q_j, H q_j> per column.
                la::gemm_big(q, k, C, k, Hd, k, Q.get(), C, 0.0f, HQ.get(), C);
                k::col_dot(q, Q.get(), C, HW.get(), C, k, C, dc.get()).wait();
                dc.download(hn.data(), C);
                k::col_dot(q, Q.get(), C, HQ.get(), C, k, C, dc.get()).wait();
                dc.download(hd2.data(), C);
                for (int64_t j = 0; j < C; ++j)
                {
                    const float c = hd2[j] > 1e-30f ? hn[j] / hd2[j] : 1.0f;
                    hn[j] = c; sv[c0 + j] *= c; ctot[c0 + j] *= c;
                }
                dc.upload(hn.data(), C);
                k::scale(q, Q.get(), k, C, C, nullptr, dc.get()).wait();
                // Input side: A += Q Q^T, b += rowsum(Q o H W).
                la::gemm(q, false, true, k, k, C, 1.0f, Q.get(), C, Q.get(), C, 1.0f, A.get(), k).wait();
                k::row_dot(q, Q.get(), C, HW.get(), C, k, C, dk.get()).wait();
                dk.download(hk.data(), k);
                for (int64_t i = 0; i < k; ++i) b[i] += hk[i];
            }
            float* a = A.get();
            q.parallel_for(sycl::range<1>(k * k), [=](sycl::id<1> i) { a[i] *= Hd[i]; }).wait();
            float* dp = dk.get();
            q.parallel_for(sycl::range<1>(k), [=](sycl::id<1> i) { dp[i] = a[i * k + i]; }).wait();
            dk.download(hk.data(), k);
            double md = 0; for (float v : hk) md += v; md /= k;
            const float eps = (float) (1e-6 * md);
            q.parallel_for(sycl::range<1>(k), [=](sycl::id<1> i) { a[i * k + i] += eps; }).wait();
            for (int64_t i = 0; i < k; ++i) hk[i] = (float) b[i];
            dk.upload(hk.data(), k);
            std::vector<float> rr(k, 1.0f);
            if (la::cholesky(q, a, k, k))
            {
                la::chol_solve(q, a, k, k, dk.get());
                dk.download(rr.data(), k);
            }
            else std::fprintf(stderr, "     !! refit: input-scale system not positive definite, skipped\n");
            for (int64_t i = 0; i < k; ++i) { if (!std::isfinite(rr[i]) || rr[i] <= 0) rr[i] = 1.0f; su[i] *= rr[i]; rtot[i] *= rr[i]; }
        }
        out.err_after = (float) err();
        if (qa.verbose) std::printf("      %s: refit in %.1f s\n", key.c_str(), tr.s());
    }

    // Pack.
    const Rate rate = Rate::integer(qa.K);
    const int words = rate.words16();
    out.trellis.resize((size_t) tk * tn * words);
    for (int64_t i = 0; i < tk * tn; ++i) pack_tile(&states[(size_t) i * 256], &out.trellis[(size_t) i * words], rate);
    out.suh.resize(k); out.svh.resize(n);
    for (int64_t i = 0; i < k; ++i) out.suh[i] = f16_from_float(su[i]);
    for (int64_t j = 0; j < n; ++j) out.svh[j] = f16_from_float(sv[j]);
    out.seconds = total.s();
    return out;
}

} // namespace exl3
