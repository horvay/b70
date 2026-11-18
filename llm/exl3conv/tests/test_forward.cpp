// exl3conv: the Gemma 4 calibration forward on the B70 against a plain double-precision CPU version of the same
// layer, written separately from notes/gemma4-forward.md (no shared code with the GPU path but the weight reader).
//   test_forward <bf16 model dir> [layers, default 0,5] [T, default 64] [window, default 16]
// A short window makes the sliding mask matter at a short T. Checks per layer: the output stream (relative
// Frobenius error; the GPU rounds activations to fp16 where exllamav3 does, the CPU never does, so ~1e-3 is
// expected) and the attention-input Hessian h^T h.
#include "arch/gemma4.hpp"
#include "io/tokenizer.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <sstream>
#include <thread>

using namespace g4;
namespace fs = std::filesystem;

static int failures = 0;

// y[T x out] = x[T x in] W^T, W bf16 [out, in], in double, threaded over outputs.
static std::vector<double> mm(const std::vector<double>& x, int64_t T, const st::TensorView& w)
{
    const int64_t out = w.shape[0], in = w.shape[1];
    const uint16_t* W = (const uint16_t*) w.data;
    std::vector<double> y((size_t) T * out);
    const int nth = 12;
    std::vector<std::thread> th;
    for (int p = 0; p < nth; ++p)
        th.emplace_back([&, p] {
            std::vector<double> wr(in);
            for (int64_t o = p; o < out; o += nth)
            {
                for (int64_t i = 0; i < in; ++i) wr[i] = st::bf16_to_float(W[o * in + i]);
                for (int64_t t = 0; t < T; ++t)
                {
                    double s = 0; const double* xr = &x[t * in];
                    for (int64_t i = 0; i < in; ++i) s += xr[i] * wr[i];
                    y[t * out + o] = s;
                }
            }
        });
    for (auto& t : th) t.join();
    return y;
}

static std::vector<double> vecw(const st::Model& m, const std::string& n)
{
    const auto& t = m.get(n);
    std::vector<double> v(t.numel());
    for (int64_t i = 0; i < t.numel(); ++i) v[i] = st::bf16_to_float(((const uint16_t*) t.data)[i]);
    return v;
}

// rmsnorm over rows of width n, weight w (empty = none)
static void rmsnorm(std::vector<double>& x, int64_t rows, int64_t n, const std::vector<double>& w, double eps)
{
    for (int64_t r = 0; r < rows; ++r)
    {
        double s = 0; for (int64_t i = 0; i < n; ++i) s += x[r * n + i] * x[r * n + i];
        const double f = 1.0 / std::sqrt(s / n + eps);
        for (int64_t i = 0; i < n; ++i) x[r * n + i] *= f * (w.empty() ? 1.0 : w[i]);
    }
}

static std::vector<double> cpu_layer(const st::Model& m, const G4& g, int64_t li, std::vector<double> x, int64_t T, std::vector<double>* h_out)
{
    const auto p = g.lp(li);
    const bool full = g.full[li];
    const int64_t H = g.H, NH = g.NH, D = full ? g.DG : g.D, NKV = full && g.k_eq_v ? g.NKVG : g.NKV;
    const double eps = g.eps;
    auto h = x; rmsnorm(h, T, H, vecw(m, p + "input_layernorm.weight"), eps);
    if (h_out) *h_out = h;
    auto qv = mm(h, T, m.get(p + "self_attn.q_proj.weight"));
    auto kraw = mm(h, T, m.get(p + "self_attn.k_proj.weight"));
    auto vv = full && g.k_eq_v ? kraw : mm(h, T, m.get(p + "self_attn.v_proj.weight"));
    auto kv = kraw;
    rmsnorm(qv, T * NH, D, vecw(m, p + "self_attn.q_norm.weight"), eps);
    rmsnorm(kv, T * NKV, D, vecw(m, p + "self_attn.k_norm.weight"), eps);
    rmsnorm(vv, T * NKV, D, {}, eps);
    // RoPE, rotate-half pairs (j, j + D/2).
    auto rope = [&](std::vector<double>& a, int64_t nh) {
        for (int64_t t = 0; t < T; ++t)
            for (int64_t hh = 0; hh < nh; ++hh)
                for (int64_t j = 0; j < D / 2; ++j)
                {
                    double f;
                    if (!full) f = std::pow(g.theta_s, -2.0 * j / D);
                    else f = j < (int64_t) (g.partial * D / 2) ? std::pow(g.theta_f, -2.0 * j / D) : 0.0;
                    const double ang = t * f, c = std::cos(ang), s = std::sin(ang);
                    double& u = a[(t * nh + hh) * D + j]; double& v = a[(t * nh + hh) * D + j + D / 2];
                    const double nu = u * c - v * s, nv = v * c + u * s; u = nu; v = nv;
                }
    };
    rope(qv, NH); rope(kv, NKV);
    std::vector<double> o((size_t) T * NH * D, 0.0);
    for (int64_t hq = 0; hq < NH; ++hq)
    {
        const int64_t kh = hq / (NH / NKV);
        for (int64_t t = 0; t < T; ++t)
        {
            std::vector<double> sc(T, -INFINITY);
            double mx = -INFINITY;
            for (int64_t u = 0; u <= t; ++u)
            {
                if (!full && u <= t - g.window) continue;
                double s = 0;
                for (int64_t d = 0; d < D; ++d) s += qv[(t * NH + hq) * D + d] * kv[(u * NKV + kh) * D + d];
                sc[u] = s; mx = std::max(mx, s);
            }
            double z = 0; for (int64_t u = 0; u <= t; ++u) if (sc[u] > -INFINITY) z += std::exp(sc[u] - mx);
            for (int64_t u = 0; u <= t; ++u)
                if (sc[u] > -INFINITY)
                {
                    const double pr = std::exp(sc[u] - mx) / z;
                    for (int64_t d = 0; d < D; ++d) o[(t * NH + hq) * D + d] += pr * vv[(u * NKV + kh) * D + d];
                }
        }
    }
    auto a = mm(o, T, m.get(p + "self_attn.o_proj.weight"));
    rmsnorm(a, T, H, vecw(m, p + "post_attention_layernorm.weight"), eps);
    for (size_t i = 0; i < x.size(); ++i) x[i] += a[i];
    auto h2 = x; rmsnorm(h2, T, H, vecw(m, p + "pre_feedforward_layernorm.weight"), eps);
    auto gg = mm(h2, T, m.get(p + "mlp.gate_proj.weight"));
    auto uu = mm(h2, T, m.get(p + "mlp.up_proj.weight"));
    for (size_t i = 0; i < gg.size(); ++i)
    {
        const double z = gg[i];
        gg[i] = 0.5 * z * (1.0 + std::tanh(std::sqrt(2.0 / M_PI) * (z + 0.044715 * z * z * z))) * uu[i];
    }
    {
        double mx = 0, mg = 0, mu = 0; int64_t over = 0;
        for (size_t i = 0; i < gg.size(); ++i) { mx = std::max(mx, std::fabs(gg[i])); mu = std::max(mu, std::fabs(uu[i])); over += std::fabs(gg[i]) > 65504; }
        (void) mg;
        if (std::getenv("DUMP")) std::printf("  CPU: max |gelu(g) u| %.1f (above fp16 max: %lld), max |u| %.1f\n", mx, (long long) over, mu);
    }
    auto d = mm(gg, T, m.get(p + "mlp.down_proj.weight"));
    rmsnorm(d, T, H, vecw(m, p + "post_feedforward_layernorm.weight"), eps);
    const double ls = st::bf16_to_float(*(const uint16_t*) m.get(p + "layer_scalar").data);
    for (size_t i = 0; i < x.size(); ++i) x[i] = (x[i] + d[i]) * ls;
    return x;
}

int main(int argc, char** argv)
{
    if (argc < 2) { std::fprintf(stderr, "usage: test_forward <bf16 model dir> [layers] [T] [window]\n"); return 2; }
    std::vector<int> layers = {0, 5};
    if (argc > 2) { layers.clear(); std::stringstream ss(argv[2]); std::string t; while (std::getline(ss, t, ',')) layers.push_back(std::stoi(t)); }
    const int64_t T = argc > 3 ? std::atoll(argv[3]) : 64;
    try
    {
        std::ifstream cf(fs::path(argv[1]) / "config.json");
        G4 g(json::parse(cf));
        if (argc > 4) g.window = std::atoll(argv[4]); else g.window = 16;
        st::Model m(argv[1]);
        // Random tokens, embedded as the converter does.
        std::mt19937 rng(3);
        std::vector<int> ids(T);
        for (auto& v : ids) v = (int) (rng() % 200000);
        ids[0] = g.bos;
        if (argc > 5)
        {   // real text instead of random tokens
            std::ifstream tf(argv[5]); std::stringstream ss; ss << tf.rdbuf();
            tok::BPE tk((fs::path(argv[1]) / "tokenizer.json").string());
            const auto t = tk.encode(ss.str());
            for (int64_t i = 1; i < T; ++i) ids[i] = t[(size_t) (i - 1) % t.size()];
        }
        const auto& e = m.get(g.prefix + "embed_tokens.weight");
        const float mult = st::bf16_to_float(nn::f2bf16(std::sqrt((float) g.H)));
        std::vector<double> x0((size_t) T * g.H);
        for (int64_t t = 0; t < T; ++t)
            for (int64_t i = 0; i < g.H; ++i)
            {
                const float v = st::bf16_to_float(((const uint16_t*) e.data)[(int64_t) ids[t] * g.H + i]) * mult;
                x0[t * g.H + i] = st::bf16_to_float(nn::f2bf16(v));
            }

        auto q = rt::make_queue();
        rt::print_device(q.get_device());
        Upload up(q);
        Scratch s(q, g, T);
        DBuf<uint16_t> ws(q, 1);
        // Run layers 0 .. max on the card in sequence; each tested layer gets the card's real input for it (the output
        // of the layers before), on the CPU too, so its input distribution is the one calibration sees.
        const int last = *std::max_element(layers.begin(), layers.end());
        std::vector<float> xf(x0.begin(), x0.end());
        DBuf<float> dx(q, xf.size()); dx.upload(xf.data(), xf.size());
        for (int li = 0; li <= last; ++li)
        {
            const bool tested = std::find(layers.begin(), layers.end(), li) != layers.end();
            Layer L = load_layer(q, m, g, li, up, ws);
            const auto inv = g.inv_freq(L.full);
            DBuf<float> dinv(q, inv.size()); dinv.upload(inv.data(), inv.size());
            std::vector<float> xin(xf.size());
            dx.download(xin.data(), xin.size());
            Capture cap;
            if (tested)
            {
                cap.qkv = DBuf<float>(q, (size_t) g.H * g.H); cap.qkv.zero();
                cap.o = DBuf<float>(q, (size_t) (g.NH * L.D) * (g.NH * L.D)); cap.o.zero();
                cap.gu = DBuf<float>(q, (size_t) g.H * g.H); cap.gu.zero();
                cap.d = DBuf<float>(q, (size_t) g.I * g.I); cap.d.zero();
            }
            forward_layer(q, g, L, dx.get(), T, s, dinv.get(), tested ? &cap : nullptr);
            dx.download(xf.data(), xf.size());
            if (!tested) continue;
            std::vector<double> h;
            rt::Timer tc;
            const auto ref = cpu_layer(m, g, li, std::vector<double>(xin.begin(), xin.end()), T, &h);
            double num = 0, den = 0;
            size_t bad_g = 0;
            for (size_t i = 0; i < ref.size(); ++i) { num += (xf[i] - ref[i]) * (xf[i] - ref[i]); den += ref[i] * ref[i]; bad_g += !std::isfinite(xf[i]); }
            const double rel = std::sqrt(num / den);
            std::vector<float> Hg((size_t) g.H * g.H);
            cap.qkv.download(Hg.data(), Hg.size());
            double hn = 0, hd = 0;
            for (int64_t i = 0; i < 256; ++i)
                for (int64_t j = 0; j < 256; ++j)
                {
                    double r = 0; for (int64_t t = 0; t < T; ++t) r += h[t * g.H + i] * h[t * g.H + j];
                    hn += (Hg[i * g.H + j] - r) * (Hg[i * g.H + j] - r); hd += r * r;
                }
            const double hrel = std::sqrt(hn / hd);
            std::printf("layer %d (%s): output rel. error %.2e, Hessian rel. error %.2e, rows counted %lld/%lld, non-finite %zu  (CPU %.1f s)\n",
                        li, L.full ? "full" : "sliding", rel, hrel, (long long) cap.n_qkv, (long long) T, bad_g, tc.s());
            if (!(rel < 1e-2) || !(hrel < 1e-2) || cap.n_qkv != T) ++failures;
        }
    }
    catch (const std::exception& ex) { std::fprintf(stderr, "error: %s\n", ex.what()); return 2; }
    std::printf(failures ? "\n%d FAILED\n" : "\nall passed\n", failures);
    return failures ? 1 : 0;
}
