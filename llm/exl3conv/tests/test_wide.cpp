// exl3conv: exl3/quantize_wide.hpp against quantize_group on a slice of a real lm_head small enough for both, with a real Hessian.
//
//   test_wide <model dir> <head_H.bin> <tokens in it> [n = 32768] [K = 6] [chunk = 8192]
//
// head_H.bin: the converter's lm_head Hessian (<work>/head_H.bin, hidden x hidden fp32, summed over the tokens). Both quantizers
// get the same Hessian and seed. For each result the packed tensor is decoded the way inference reads it (fp16 scales,
// exl3::dequant_linear) and its error tr(E^T H E) / tr(W^T H W) recomputed: it must equal what the quantizer reported, and
// the chunked result must be as good as the one-pass one (they differ only in the global scale: a median of per-chunk searches).
#include "arch/qwen35.hpp"
#include "exl3/quantize_wide.hpp"

#include <cstdio>
#include <fstream>

using namespace q35;
namespace fs = std::filesystem;
namespace la = exl3::la;

int main(int argc, char** argv)
{
    if (argc < 4) { std::fprintf(stderr, "usage: test_wide <model dir> <head_H.bin> <tokens> [n] [K] [chunk]\n"); return 2; }
    const fs::path dir = argv[1];
    const int64_t count = std::atoll(argv[3]);
    const int64_t n = argc > 4 ? std::atoll(argv[4]) : 32768;
    const int K = argc > 5 ? std::atoi(argv[5]) : 6;
    const int64_t chunk = argc > 6 ? std::atoll(argv[6]) : 8192;
    st::Model m(dir);
    const auto& lh = m.get("lm_head.weight");
    const int64_t k = lh.shape[1];
    std::vector<float> Hh((size_t) k * k);
    { std::ifstream f(argv[2], std::ios::binary); f.read((char*) Hh.data(), (std::streamsize) (Hh.size() * 4)); if (!f) { std::fprintf(stderr, "short H\n"); return 1; } }
    auto q = rt::make_queue();
    rt::print_device(q.get_device());
    Upload up(q);
    // W: the first n rows of lm_head, (in, out) fp32 = k x n.
    DBuf<half> wh(q, (size_t) n * k);
    {
        std::vector<uint16_t> h((size_t) n * k);
        for (size_t i = 0; i < h.size(); ++i) h[i] = exl3::f16_from_float(elem_f32(lh, i));
        up(wh.get(), h.data(), h.size() * 2);
    }
    DBuf<float> W(q, (size_t) k * n);
    nn::transpose_h2f(q, wh.get(), n, k, W.get()).wait();
    auto hessian = [&]() {
        DBuf<float> H(q, (size_t) k * k);
        H.upload(Hh.data(), Hh.size());
        return exl3::finalize_hessian(q, std::move(H), k, count, 1234);
    };
    DBuf<int> perm(q, 256);
    perm.upload(exl3::tile_perm().data(), 256);
    // tr(E^T Hd E) / tr(W^T Hd W) of a decoded result, Hd the finalized (damped) Hessian.
    auto decoded_err = [&](const exl3::QuantOut& o, const float* Hd) {
        DBuf<uint16_t> tr(q, o.trellis.size()), su(q, o.suh.size()), sv(q, o.svh.size());
        tr.upload(o.trellis.data(), o.trellis.size()); su.upload(o.suh.data(), o.suh.size()); sv.upload(o.svh.data(), o.svh.size());
        DBuf<float> tmp(q, (size_t) k * n), E(q, (size_t) k * n), HE(q, (size_t) k * n), rk(q, k);
        DBuf<half> wq(q, (size_t) k * n);
        exl3::dequant_linear(q, tr.get(), su.get(), sv.get(), k, n, o.K, exl3::CB_MUL1, perm.get(), tmp.get(), wq.get());
        float* e = E.get(); const float* w = W.get(); const half* d = wq.get();
        q.parallel_for(sycl::range<1>(k * n), [=](sycl::id<1> i) { e[i] = w[i] - (float) d[i]; }).wait();
        auto trace = [&](const float* X) {
            la::gemm_big(q, k, n, k, Hd, k, X, n, 0.0f, HE.get(), n);
            exl3::k::row_dot(q, X, n, HE.get(), n, k, n, rk.get()).wait();
            std::vector<float> v(k); rk.download(v.data(), k);
            double s = 0; for (float x : v) s += x; return s;
        };
        return trace(E.get()) / trace(W.get());
    };
    exl3::QuantArgs qa; qa.K = K; qa.verbose = true;
    int bad = 0;
    double e1, e2, d1, d2;
    {
        auto hs = hessian();
        rt::Timer t;
        auto o = exl3::quantize_group(q, hs, {{"lm_head", n, W.get()}}, qa, 99)[0];
        const double s = t.s();
        d1 = decoded_err(o, hs.Hd.get()); e1 = o.err_after;
        std::printf("quantize_group: g %.4f  err %.6f -> %.6f  decoded %.6f  (%.1f s)\n", o.g_scale, o.err_before, o.err_after, d1, s);
    }
    {
        auto hs = hessian();
        const half* w = wh.get();
        auto src = [&](int64_t c0, int64_t C, float* dst) { nn::transpose_h2f(q, w + c0 * k, C, k, dst).wait(); };
        rt::Timer t;
        auto o = exl3::quantize_wide(q, hs, "lm_head", n, src, qa, 99, chunk);
        const double s = t.s();
        d2 = decoded_err(o, hs.Hd.get()); e2 = o.err_after;
        std::printf("quantize_wide (%lld-column chunks): g %.4f  err %.6f -> %.6f  decoded %.6f  (%.1f s)\n", (long long) chunk, o.g_scale, o.err_before, o.err_after, d2, s);
    }
    auto close = [](double a, double b, double tol) { return std::fabs(a - b) <= tol * std::max(a, b); };
    if (!close(d1, e1, 0.05)) { std::printf("FAIL: quantize_group's decoded error differs from its report\n"); ++bad; }
    if (!close(d2, e2, 0.05)) { std::printf("FAIL: quantize_wide's decoded error differs from its report\n"); ++bad; }
    if (d2 > d1 * 1.05) { std::printf("FAIL: quantize_wide is worse than quantize_group by more than 5%%\n"); ++bad; }
    std::printf("%s\n", bad ? "FAILED" : "ok");
    return bad ? 1 : 0;
}
