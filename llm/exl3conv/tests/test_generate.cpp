// exl3conv: end-to-end sanity of the Gemma 4 forward: run a BF16 (or EXL3) model over a short prompt and print the
// top next tokens at the last position. test_generate <model dir> [--exl3] ["prompt"]
#include "arch/gemma4.hpp"
#include "io/tokenizer.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>

using namespace g4;
namespace fs = std::filesystem;

int main(int argc, char** argv)
{
    if (argc < 2) return 2;
    const fs::path dir = argv[1];
    bool exl3 = false; std::string prompt = "The capital of France is";
    size_t maxtok = 2048;
    for (int i = 2; i < argc; ++i)
    {
        const std::string s = argv[i];
        if (s == "--exl3") exl3 = true; else if (s.rfind("--max=", 0) == 0) maxtok = std::stoul(s.substr(6)); else prompt = s;
    }
    if (!prompt.empty() && prompt[0] == '@') { std::ifstream pf(prompt.substr(1)); std::stringstream ss; ss << pf.rdbuf(); prompt = ss.str(); }
    std::ifstream cf(dir / "config.json");
    const G4 g(json::parse(cf));
    st::Model m(dir);
    tok::BPE tk((dir / "tokenizer.json").string());
    std::vector<int32_t> ids = {g.bos};
    for (int t : tk.encode(prompt)) ids.push_back(t);
    if (ids.size() > maxtok) ids.resize(maxtok);
    const int64_t T = (int64_t) ids.size();
    // id -> string for printing
    std::ifstream tf(dir / "tokenizer.json");
    const auto tj = json::parse(tf);
    std::map<int, std::string> names;
    for (auto& [k, v] : tj["model"]["vocab"].items()) names[v.get<int>()] = k;
    auto q = rt::make_queue();
    Upload up(q);
    const uint16_t* E = (const uint16_t*) m.get(g.prefix + "embed_tokens.weight").data;
    const float mult = st::bf16_to_float(nn::f2bf16(std::sqrt((float) g.H)));
    std::vector<uint16_t> gath(T * g.H);
    for (int64_t t = 0; t < T; ++t) std::memcpy(&gath[t * g.H], E + (int64_t) ids[t] * g.H, g.H * 2);
    DBuf<uint16_t> dg(q, gath.size()); dg.upload(gath.data(), gath.size());
    DBuf<float> x(q, T * g.H);
    nn::embed(q, dg.get(), T, g.H, mult, x.get()).wait();
    Scratch s(q, g, T);
    DBuf<uint16_t> ws(q, 1); DBuf<int> perm(q, 256); perm.upload(exl3::tile_perm().data(), 256); DBuf<float> tmp(q, 1);
    const auto inv_s = g.inv_freq(false), inv_f = g.inv_freq(true);
    DBuf<float> di_s(q, inv_s.size()), di_f(q, inv_f.size()); di_s.upload(inv_s.data(), inv_s.size()); di_f.upload(inv_f.data(), inv_f.size());
    std::vector<float> h(g.H);
    for (int64_t li = 0; li < g.L; ++li)
    {
        Layer L = exl3 ? load_layer_exl3(q, m, g, li, up, perm, tmp) : load_layer(q, m, g, li, up, ws);
        forward_layer(q, g, L, x.get(), T, s, L.full ? di_f.get() : di_s.get(), nullptr);
        if (li < 3 || li % 10 == 9 || li == g.L - 1)
        {
            x.download(h.data(), g.H);   // token 0
            std::vector<float> last(g.H); q.memcpy(last.data(), x.get() + (T - 1) * g.H, g.H * 4).wait();
            double r0 = 0, r1 = 0; for (int i = 0; i < g.H; ++i) { r0 += h[i] * h[i]; r1 += last[i] * last[i]; }
            std::printf("  after layer %2lld: rms token0 %.4g, last %.4g\n", (long long) li, std::sqrt(r0 / g.H), std::sqrt(r1 / g.H));
        }
    }
    const auto normw = load_bf16_vec(q, m, g.prefix + "norm.weight", up);
    if (T > 16)
    {
        // Mean next-token loss over the prompt, in chunks, with the eval's own scoring kernel.
        const auto& e2 = m.get(g.prefix + "embed_tokens.weight");
        DBuf<uint16_t> eb2(q, (size_t) e2.numel()); up(eb2.get(), e2.data, e2.nbytes);
        DBuf<half> Eh2(q, (size_t) e2.numel()); nn::bf16_to_half(q, eb2.get(), e2.numel(), Eh2.get()).wait(); eb2.reset();
        const int64_t Tc = 128; DBuf<half> yc(q, Tc * g.H); DBuf<float> L2(q, Tc * g.vocab), st4(q, Tc * 4); DBuf<int32_t> tg(q, Tc);
        double nll = 0, nll_first = 0, nll_late = 0; int64_t n = 0, n1 = 0, n2 = 0;
        double qnll[4] = {0, 0, 0, 0}; int64_t qn[4] = {0, 0, 0, 0};
        for (int64_t c0 = 0; c0 < T - 1; c0 += Tc)
        {
            const int64_t tc = std::min(Tc, T - 1 - c0);
            std::vector<int32_t> ht(tc); for (int64_t t = 0; t < tc; ++t) ht[t] = ids[c0 + t + 1];
            tg.upload(ht.data(), tc);
            nn::rmsnorm_h(q, x.get() + c0 * g.H, normw.get(), tc, g.H, (float) g.eps, yc.get());
            oneapi::mkl::blas::row_major::gemm(q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::trans, tc, g.vocab, g.H, 1.0f, yc.get(), g.H, Eh2.get(), g.H, 0.0f, L2.get(), g.vocab);
            nn::logit_stats(q, L2.get(), L2.get(), tc, g.vocab, 30.0f, tg.get(), st4.get()).wait();
            std::vector<float> h4(tc * 4); st4.download(h4.data(), tc * 4);
            if (c0 == 0)
            {
                std::vector<float> lh((size_t) std::min<int64_t>(tc, 6) * g.vocab);
                q.memcpy(lh.data(), L2.get(), lh.size() * 4).wait();
                for (int64_t t = 0; t < std::min<int64_t>(tc, 6); ++t)
                {
                    const float* r = &lh[t * g.vocab];
                    double mx = -1e30; int am = 0;
                    for (int64_t i = 0; i < g.vocab; ++i) { const double v = 30.0 * std::tanh(r[i] / 30.0); if (v > mx) { mx = v; am = (int) i; } }
                    double z = 0; for (int64_t i = 0; i < g.vocab; ++i) z += std::exp(30.0 * std::tanh(r[i] / 30.0) - mx);
                    const int y = ht[t];
                    const double nl = -(30.0 * std::tanh(r[y] / 30.0) - mx - std::log(z));
                    std::printf("  pos %lld: target %-14s logit %7.3f, top %-14s %7.3f; host nll %.3f, kernel nll %.3f\n", (long long) t,
                                names.count(y) ? names[y].c_str() : "?", 30.0 * std::tanh(r[y] / 30.0), names.count(am) ? names[am].c_str() : "?", mx, nl, h4[t * 4 + 3]);
                }
            }
            for (int64_t t = 0; t < tc; ++t) { const int qq = (int) std::min<int64_t>(3, (c0 + t) * 4 / (T - 1)); qnll[qq] += h4[t * 4 + 3]; ++qn[qq]; }
            for (int64_t t = 0; t < tc; ++t) { nll += h4[t * 4 + 3]; ++n; if (c0 + t < 1024) { nll_first += h4[t * 4 + 3]; ++n1; } else { nll_late += h4[t * 4 + 3]; ++n2; } }
        }
        std::printf("%lld tokens: perplexity %.3f (positions < 1024: %.3f, >= 1024: %.3f)\n", (long long) T, std::exp(nll / n),
                    std::exp(nll_first / std::max<int64_t>(1, n1)), n2 ? std::exp(nll_late / n2) : 0.0);
        std::printf("  by quarter:");
        for (int qq = 0; qq < 4; ++qq) std::printf(" %.3f", std::exp(qnll[qq] / std::max<int64_t>(1, qn[qq])));
        std::printf("\n");
    }
    DBuf<half> y(q, g.H);
    nn::rmsnorm_h(q, x.get() + (T - 1) * g.H, normw.get(), 1, g.H, (float) g.eps, y.get());
    const auto& e = m.get(g.prefix + "embed_tokens.weight");
    DBuf<uint16_t> eb(q, (size_t) e.numel()); up(eb.get(), e.data, e.nbytes);
    DBuf<half> Eh(q, (size_t) e.numel()); nn::bf16_to_half(q, eb.get(), e.numel(), Eh.get()).wait(); eb.reset();
    DBuf<float> lg(q, g.vocab);
    oneapi::mkl::blas::row_major::gemm(q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::trans, 1, g.vocab, g.H, 1.0f, y.get(), g.H, Eh.get(), g.H, 0.0f, lg.get(), g.vocab).wait();
    std::vector<float> l(g.vocab); lg.download(l.data(), g.vocab);
    std::vector<int> idx(g.vocab); for (int i = 0; i < (int) g.vocab; ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + 8, idx.end(), [&](int a, int b) { return l[a] > l[b]; });
    std::printf("prompt \"%s\" (%lld tokens). Top next tokens:\n", prompt.c_str(), (long long) T);
    for (int k = 0; k < 8; ++k) std::printf("  %8.3f  %s\n", 30.0 * std::tanh(l[idx[k]] / 30.0), names.count(idx[k]) ? names[idx[k]].c_str() : "?");
}
