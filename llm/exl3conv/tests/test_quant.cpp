// exl3conv: per-tensor quantizer test on the B70. test_quant <bf16 model dir> [tensor] [K list]
//
// Quantizes one real weight (default: Gemma 4 layer 0 q_proj, [8192, 5376]) against a synthetic Hessian built from
// correlated random activations with a few strong channels (a stand-in until the calibration forward exists), then
// checks what was written:
//   round trip   dequantize the packed trellis on the CPU with the fp16 suh / svh, as inference reads it; its error
//                in the Hessian metric, tr(E^T H E) / tr(W^T H W), must match what the quantizer reported
//   feedback     LDLQ must beat the same run with the error feedback switched off
//   rates        the error must fall as K rises
// Plus a two-tensor group (q_proj and k_proj side by side, sharing the Hessian) against the same tensors alone.
#include "exl3/dequant_host.hpp"
#include "exl3/quantize.hpp"
#include "io/safetensors.hpp"
#include "rt/rt.hpp"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace exl3;

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++failures; std::printf("  FAIL: "); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

// HF [out, in] BF16 -> (in, out) fp32 through fp16, as exllamav3 loads it.
static std::vector<float> load_weight(const st::Model& m, const std::string& name, int64_t& k, int64_t& n)
{
    const auto& t = m.get(name);
    n = t.shape[0]; k = t.shape[1];
    const uint16_t* h = (const uint16_t*) t.data;
    std::vector<float> w((size_t) k * n);
    for (int64_t o = 0; o < n; ++o)
        for (int64_t i = 0; i < k; ++i)
            w[i * n + o] = f16_to_float(f16_from_float(st::bf16_to_float(h[o * k + i])));
    return w;
}

// H = X^T X over T rows of correlated activations: X = Z M + outlier channels, Z ~ N(0, 1).
static DBuf<float> make_hessian(sycl::queue& q, int64_t k, int64_t T, int64_t& count)
{
    std::mt19937 rng(5);
    std::normal_distribution<float> nd(0, 1);
    const int64_t r = 256;                       // rank of the shared structure
    std::vector<float> Z((size_t) T * r), M((size_t) r * k), X((size_t) T * k);
    for (auto& v : Z) v = nd(rng);
    for (auto& v : M) v = nd(rng) * 0.1f;
    std::vector<float> chan(k);
    for (int64_t i = 0; i < k; ++i) chan[i] = std::exp(0.5f * nd(rng)) * (i % 97 == 0 ? 20.0f : 1.0f);
    DBuf<float> dZ(q, Z.size()), dM(q, M.size()), dX(q, X.size()), H(q, (size_t) k * k);
    dZ.upload(Z.data(), Z.size()); dM.upload(M.data(), M.size());
    la::gemm(q, false, false, T, k, r, 1.0f, dZ.get(), r, dM.get(), k, 0.0f, dX.get(), k).wait();
    dX.download(X.data(), X.size());
    for (int64_t t = 0; t < T; ++t) for (int64_t i = 0; i < k; ++i) X[t * k + i] = f16_to_float(f16_from_float((X[t * k + i] + 0.3f * nd(rng)) * chan[i]));
    dX.upload(X.data(), X.size());
    la::gemm(q, true, false, k, k, T, 1.0f, dX.get(), k, dX.get(), k, 0.0f, H.get(), k).wait();
    count = T;
    return H;
}

// tr(E^T H E) / tr(W^T H W) with E = W - Wq, on the card. H: the damped Hessian (Hd).
static double hessian_error(sycl::queue& q, const float* Hd, int64_t k, const std::vector<float>& W, const std::vector<double>& Wq, int64_t n)
{
    std::vector<float> e(W.size());
    for (size_t i = 0; i < W.size(); ++i) e[i] = (float) (W[i] - Wq[i]);
    DBuf<float> dW(q, W.size()), dE(q, W.size()), HX(q, W.size()), rk(q, k);
    dW.upload(W.data(), W.size()); dE.upload(e.data(), e.size());
    std::vector<float> hk(k);
    auto tr = [&](const float* X) {
        la::gemm(q, false, false, k, n, k, 1.0f, Hd, k, X, n, 0.0f, HX.get(), n);
        k::row_dot(q, X, n, HX.get(), n, k, n, rk.get()).wait();
        rk.download(hk.data(), k);
        double s = 0; for (float v : hk) s += v; return s;
    };
    return tr(dE.get()) / tr(dW.get());
}

int main(int argc, char** argv)
{
    if (argc < 2) { std::fprintf(stderr, "usage: test_quant <bf16 model dir> [tensor] [K,K,...]\n"); return 2; }
    const std::string name = argc > 2 ? argv[2] : "model.language_model.layers.0.self_attn.q_proj.weight";
    std::vector<int> Ks = {2, 3, 4, 5, 6};
    if (argc > 3) { Ks.clear(); std::stringstream ss(argv[3]); std::string t; while (std::getline(ss, t, ',')) Ks.push_back(std::stoi(t)); }
    try
    {
        st::Model m(argv[1]);
        int64_t k, n;
        const auto W = load_weight(m, name, k, n);
        std::printf("%s: in %lld, out %lld\n", name.c_str(), (long long) k, (long long) n);
        auto q = rt::make_queue();
        rt::print_device(q.get_device());
        DBuf<float> dW(q, W.size());
        dW.upload(W.data(), W.size());
        int64_t count;
        rt::Timer th;
        auto H = make_hessian(q, k, 8192, count);
        std::printf("synthetic Hessian: %.1f s\n", th.s());

        auto run = [&](int K, bool fb, const std::vector<QuantIn>& ins) {
            DBuf<float> Hc(q, (size_t) k * k);
            q.memcpy(Hc.get(), H.get(), sizeof(float) * k * k).wait();
            rt::Timer t;
            Hessian hs = finalize_hessian(q, std::move(Hc), k, count, 1234);
            const double tf = t.s();
            QuantArgs qa; qa.K = K; qa.feedback = fb;
            auto outs = quantize_group(q, hs, ins, qa, 99);
            std::vector<double> errs;
            for (size_t i = 0; i < outs.size(); ++i)
            {
                const auto& o = outs[i];
                std::vector<float> Wi((size_t) k * o.n);
                q.memcpy(Wi.data(), ins[i].W, Wi.size() * sizeof(float)).wait();
                const auto Wq = dequant_host(o.trellis.data(), o.suh.data(), o.svh.data(), k, o.n, K, CB_MUL1);
                const double e = hessian_error(q, hs.Hd.get(), k, Wi, Wq, o.n);
                double num = 0, den = 0;
                for (size_t j = 0; j < Wi.size(); ++j) { num += (Wi[j] - Wq[j]) * (Wi[j] - Wq[j]); den += (double) Wi[j] * Wi[j]; }
                std::printf("  K %d%s %-48s g %.4f  H-error %.6f -> refit %.6f, decoded %.6f  rel. RMS %.4f  (finalize %.1f s, quantize %.1f s)\n",
                            K, fb ? "" : " no-feedback", o.key.c_str(), o.g_scale, o.err_before, o.err_after, e, std::sqrt(num / den), tf, o.seconds);
                CHECK(std::fabs(e - o.err_after) <= 0.02 * o.err_after + 1e-7, "%s K %d: decoded error %.6f vs reported %.6f", o.key.c_str(), K, e, o.err_after);
                CHECK(o.err_after <= o.err_before * 1.0001, "%s K %d: refit made it worse", o.key.c_str(), K);
                errs.push_back(e);
            }
            return errs;
        };

        std::printf("rates\n");
        double prev = 1e9, e4 = 0;
        for (int K : Ks)
        {
            const double e = run(K, true, {{name, n, dW.get()}})[0];
            CHECK(e < prev, "K %d: error did not fall with K", K);
            prev = e;
            if (K == 4) e4 = e;
        }
        std::printf("feedback\n");
        const double e_nofb = run(4, false, {{name, n, dW.get()}})[0];
        if (e4 > 0) CHECK(e4 < e_nofb, "LDLQ (%.6f) did not beat no feedback (%.6f)", e4, e_nofb);

        // Group: this tensor plus k_proj of the same layer, side by side, against each alone.
        std::string name2 = name;
        const auto pos = name2.find("q_proj");
        if (pos != std::string::npos && (name2.replace(pos, 6, "k_proj"), m.has(name2)))
        {
            std::printf("group\n");
            int64_t k2, n2;
            const auto W2 = load_weight(m, name2, k2, n2);
            DBuf<float> dW2(q, W2.size());
            dW2.upload(W2.data(), W2.size());
            const auto g = run(4, true, {{name, n, dW.get()}, {name2, n2, dW2.get()}});
            const double a2 = run(4, true, {{name2, n2, dW2.get()}})[0];
            CHECK(std::fabs(g[0] - e4) <= 0.02 * e4, "group q error %.6f vs alone %.6f", g[0], e4);
            CHECK(std::fabs(g[1] - a2) <= 0.02 * a2, "group k error %.6f vs alone %.6f", g[1], a2);
        }
    }
    catch (const std::exception& e) { std::fprintf(stderr, "error: %s\n", e.what()); return 2; }
    std::printf(failures ? "\n%d FAILED\n" : "\nall passed\n", failures);
    return failures ? 1 : 0;
}
