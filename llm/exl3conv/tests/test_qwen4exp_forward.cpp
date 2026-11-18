// exl3conv: the Qwen3.8-Flash-Next calibration forward on the B70 against a double-precision CPU version written from
// notes/qwen4exp-forward.md §2 (it shares only the weight reader and the n-gram hashing with the GPU path).
//   test_qwen4exp_forward <bf16 model dir> [layers, default 0,1,3] [T, default 64]
// Layers 0 .. max run on the card in order; each tested layer's CPU version gets the card's real input. The PLE reads
// the BF16 table rows here (the converter uses the quantized ones). Checks: output streams (relative Frobenius error)
// and the attention-site Hessian.
#include "arch/qwen4exp.hpp"
#include "arch/qwen4exp_ngram.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <thread>

using namespace qwen4exp;
namespace fs = std::filesystem;
static int failures = 0;

using V = std::vector<double>;
static double bfv(const uint16_t* p, int64_t i) { return st::bf16_to_float(p[i]); }
static const uint16_t* W(const st::Model& m, const std::string& n) { return (const uint16_t*) m.get(n).data; }

// y[T x out] = x[T x in] W^T, W bf16 [out, in] (at element offset off), threaded over outputs.
static V mm(const V& x, int64_t T, int64_t in, const uint16_t* w, int64_t out)
{
    V y((size_t) T * out);
    std::vector<std::thread> th;
    const int nth = 12;
    for (int p = 0; p < nth; ++p)
        th.emplace_back([&, p] {
            for (int64_t o = p; o < out; o += nth)
                for (int64_t t = 0; t < T; ++t)
                {
                    double s = 0; for (int64_t i = 0; i < in; ++i) s += x[t * in + i] * bfv(w, o * in + i);
                    y[t * out + o] = s;
                }
        });
    for (auto& t : th) t.join();
    return y;
}
static double sig(double x) { return 1.0 / (1.0 + std::exp(-x)); }
static double silu(double x) { return x * sig(x); }

// rms norm of rows of width n; w: weight pointer (bf16) or null; one_plus: (1 + w); group: weight index offset per row
static void rmsn(V& x, int64_t rows, int64_t n, const uint16_t* w, bool one_plus, double eps, int64_t wstride = 0)
{
    for (int64_t r = 0; r < rows; ++r)
    {
        double s = 0; for (int64_t i = 0; i < n; ++i) s += x[r * n + i] * x[r * n + i];
        const double f = 1.0 / std::sqrt(s / n + eps);
        for (int64_t i = 0; i < n; ++i)
        {
            double wv = w ? bfv(w, (wstride ? (r % wstride) * n : 0) + i) : 0.0;
            x[r * n + i] *= f * (w ? (one_plus ? 1.0 + wv : wv) : 1.0);
        }
    }
}

struct SiteOut { V y, inj; };
static SiteOut site(const st::Model& m, const std::string& p, const V& X, int64_t T, int64_t S, int64_t D, int64_t r, double eps, bool inj)
{
    V N = X; rmsn(N, T * S, D, W(m, p + "hc_norm.weight"), true, eps, S);
    // tables are F16 in the checkpoint? (BF16 in the source) -> read via bf16 helper on BF16 sources
    V a = mm(N, T, S * D, W(m, p + "input_mix_weight_down.weight"), r);
    for (auto& v : a) v = silu(v / (double) S);
    V G = mm(a, T, r, W(m, p + "input_mix_weight_up.weight"), S * D);
    SiteOut o; o.y.assign(T * D, 0.0);
    for (int64_t t = 0; t < T; ++t) for (int64_t s = 0; s < S; ++s) for (int64_t d = 0; d < D; ++d)
        o.y[t * D + d] += sig(G[(t * S + s) * D + d]) * N[(t * S + s) * D + d] / (double) S;
    if (inj) { o.inj = mm(N, T, S * D, W(m, p + "block_inject_weight.weight"), S); for (auto& v : o.inj) v = 2.0 * sig(v / (double) S); }
    return o;
}

static V cpu_layer(const st::Model& m, const Config& c, int64_t li, V X, int64_t T, const V& ple_rows, V* y_attn)
{
    const auto p = c.lp(li);
    const int64_t S = c.S, D = c.D;
    const double eps = c.eps;
    if (li == c.ple_layer)
    {
        const auto pp = p + "ple.";
        V K = mm(ple_rows, T, D, W(m, pp + "key_proj.weight"), S * D);
        V Vv = mm(ple_rows, T, D, W(m, pp + "value_proj.weight"), D);
        rmsn(K, T * S, D, W(m, pp + "norm_key.weight"), true, eps, S);
        V Q = X; rmsn(Q, T * S, D, W(m, pp + "norm_query.weight"), true, eps, S);
        V Gv(T * S * D);
        for (int64_t t = 0; t < T; ++t) for (int64_t s = 0; s < S; ++s)
        {
            double g = 0; for (int64_t d = 0; d < D; ++d) g += K[(t * S + s) * D + d] * Q[(t * S + s) * D + d];
            g /= std::sqrt((double) D);
            const double sg = (g > 0) - (g < 0);
            const double gv = sig(sg * std::sqrt(std::max(std::fabs(g), 1e-6)));
            for (int64_t d = 0; d < D; ++d) Gv[(t * S + s) * D + d] = gv * Vv[t * D + d];
        }
        V Cn = Gv; rmsn(Cn, T * S, D, W(m, pp + "norm_conv.weight"), true, eps, S);
        const uint16_t* wc = W(m, pp + "conv1d.weight");
        for (int64_t t = 0; t < T; ++t) for (int64_t ch = 0; ch < S * D; ++ch)
        {
            double acc = 0;
            for (int j = 0; j < 4; ++j) { const int64_t tt = t - 3 * (3 - j); if (tt >= 0) acc += bfv(wc, ch * 4 + j) * Cn[tt * S * D + ch]; }
            X[t * S * D + ch] += Gv[t * S * D + ch] + silu(acc);
        }
    }
    auto sa = site(m, p + "attn_hyper_connection.", X, T, S, D, c.rank, eps, true);
    if (y_attn) *y_attn = sa.y;
    V out;
    if (!c.full[li])
    {
        const int64_t KD = c.HK * c.DK, C = 2 * KD + c.HV * c.DV, HV = c.HV, DK = c.DK, DV = c.DV;
        V qkv = mm(sa.y, T, D, W(m, p + "linear_attn.in_proj_qkv.weight"), C);
        V z = mm(sa.y, T, D, W(m, p + "linear_attn.in_proj_z.weight"), HV * DV);
        V a = mm(sa.y, T, D, W(m, p + "linear_attn.in_proj_a.weight"), HV);
        V b = mm(sa.y, T, D, W(m, p + "linear_attn.in_proj_b.weight"), HV);
        const uint16_t* wc = W(m, p + "linear_attn.conv1d.weight");
        V cv(T * C);
        for (int64_t t = 0; t < T; ++t) for (int64_t ch = 0; ch < C; ++ch)
        {
            double acc = 0; for (int j = 0; j < 4; ++j) { const int64_t tt = t - 3 + j; if (tt >= 0) acc += bfv(wc, ch * 4 + j) * qkv[tt * C + ch]; }
            cv[t * C + ch] = silu(acc);
        }
        const uint16_t* Al = W(m, p + "linear_attn.A_log"); const uint16_t* dtb = W(m, p + "linear_attn.dt_bias");
        V o(T * HV * DV);
        for (int64_t hv = 0; hv < HV; ++hv)
        {
            const int64_t hk = hv / (HV / c.HK);
            std::vector<double> St(DK * DV, 0.0);
            for (int64_t t = 0; t < T; ++t)
            {
                double qn[128], kn[128], sq = 0, sk = 0;
                for (int i = 0; i < DK; ++i) { qn[i] = cv[t * C + hk * DK + i]; kn[i] = cv[t * C + KD + hk * DK + i]; sq += qn[i] * qn[i]; sk += kn[i] * kn[i]; }
                for (int i = 0; i < DK; ++i) { qn[i] /= std::sqrt(sq + 1e-6) * std::sqrt((double) DK); kn[i] /= std::sqrt(sk + 1e-6); }
                const double x = a[t * HV + hv] + bfv(dtb, hv);
                const double g = -std::exp(bfv(Al, hv)) * (x > 20 ? x : std::log1p(std::exp(x)));
                const double beta = sig(b[t * HV + hv]), dec = std::exp(g);
                for (auto& s : St) s *= dec;
                for (int j = 0; j < DV; ++j)
                {
                    double kv = 0; for (int i = 0; i < DK; ++i) kv += St[i * DV + j] * kn[i];
                    const double dl = beta * (cv[t * C + 2 * KD + hv * DV + j] - kv);
                    for (int i = 0; i < DK; ++i) St[i * DV + j] += kn[i] * dl;
                }
                for (int j = 0; j < DV; ++j) { double s = 0; for (int i = 0; i < DK; ++i) s += St[i * DV + j] * qn[i]; o[(t * HV + hv) * DV + j] = s; }
            }
        }
        const uint16_t* gn = W(m, p + "linear_attn.norm.weight");
        for (int64_t r = 0; r < T * HV; ++r)
        {
            double s = 0; for (int i = 0; i < DV; ++i) s += o[r * DV + i] * o[r * DV + i];
            const double f = 1.0 / std::sqrt(s / DV + eps);
            for (int i = 0; i < DV; ++i) o[r * DV + i] = bfv(gn, i) * o[r * DV + i] * f * sig(z[r * DV + i]);
        }
        out = mm(o, T, HV * DV, W(m, p + "linear_attn.out_proj.weight"), D);
    }
    else
    {
        const int64_t NH = c.NH, NKV = c.NKV, Dh = c.Dh, R = c.R;
        V qg = mm(sa.y, T, D, W(m, p + "self_attn.q_proj.weight"), NH * 2 * Dh);
        V k = mm(sa.y, T, D, W(m, p + "self_attn.k_proj.weight"), NKV * Dh);
        V v = mm(sa.y, T, D, W(m, p + "self_attn.v_proj.weight"), NKV * Dh);
        V q(T * NH * Dh), gate(T * NH * Dh);
        for (int64_t t = 0; t < T; ++t) for (int64_t h = 0; h < NH; ++h) for (int64_t d = 0; d < Dh; ++d)
        { q[(t * NH + h) * Dh + d] = qg[(t * NH + h) * 2 * Dh + d]; gate[(t * NH + h) * Dh + d] = qg[(t * NH + h) * 2 * Dh + Dh + d]; }
        rmsn(q, T * NH, Dh, W(m, p + "self_attn.q_norm.weight"), true, eps);
        rmsn(k, T * NKV, Dh, W(m, p + "self_attn.k_norm.weight"), true, eps);
        auto rope = [&](V& x, int64_t nh) {
            for (int64_t t = 0; t < T; ++t) for (int64_t h = 0; h < nh; ++h) for (int64_t j = 0; j < R / 2; ++j)
            {
                const double f = std::pow(c.theta, -2.0 * j / (double) R), th = t * f;
                double& a = x[(t * nh + h) * Dh + j]; double& bb = x[(t * nh + h) * Dh + j + R / 2];
                const double na = a * std::cos(th) - bb * std::sin(th), nb = bb * std::cos(th) + a * std::sin(th); a = na; bb = nb;
            }
        };
        rope(q, NH); rope(k, NKV);
        V o(T * NH * Dh, 0.0);
        for (int64_t h = 0; h < NH; ++h)
        {
            const int64_t kh = h / (NH / NKV);
            for (int64_t t = 0; t < T; ++t)
            {
                std::vector<double> sc(t + 1); double mx = -1e300;
                for (int64_t u = 0; u <= t; ++u) { double s = 0; for (int64_t d = 0; d < Dh; ++d) s += q[(t * NH + h) * Dh + d] * k[(u * NKV + kh) * Dh + d]; sc[u] = s / std::sqrt((double) Dh); mx = std::max(mx, sc[u]); }
                double zs = 0; for (auto& s : sc) zs += std::exp(s - mx);
                for (int64_t u = 0; u <= t; ++u) { const double pr = std::exp(sc[u] - mx) / zs; for (int64_t d = 0; d < Dh; ++d) o[(t * NH + h) * Dh + d] += pr * v[(u * NKV + kh) * Dh + d]; }
            }
        }
        for (size_t i = 0; i < o.size(); ++i) o[i] *= sig(gate[i]);
        out = mm(o, T, NH * Dh, W(m, p + "self_attn.o_proj.weight"), D);
    }
    for (int64_t t = 0; t < T; ++t) for (int64_t s = 0; s < S; ++s) for (int64_t d = 0; d < D; ++d) X[(t * S + s) * D + d] += sa.inj[t * S + s] * out[t * D + d];
    // MoE
    auto sm = site(m, p + "mlp_hyper_connection.", X, T, S, D, c.rank, eps, true);
    V logits = mm(sm.y, T, D, W(m, p + "mlp.gate.weight"), c.E);
    const uint16_t* gu = W(m, p + "mlp.experts.gate_up_proj"); const uint16_t* dn = W(m, p + "mlp.experts.down_proj");
    const int64_t F = c.F;
    V moe(T * D, 0.0);
    for (int64_t t = 0; t < T; ++t)
    {
        std::vector<int> idx(c.E); for (int e = 0; e < c.E; ++e) idx[e] = e;
        std::stable_sort(idx.begin(), idx.end(), [&](int a2, int b2) { return logits[t * c.E + a2] > logits[t * c.E + b2]; });
        double mx = logits[t * c.E + idx[0]], zs = 0;
        for (int s = 0; s < c.topk; ++s) zs += std::exp(logits[t * c.E + idx[s]] - mx);
        for (int s = 0; s < c.topk; ++s)
        {
            const int e = idx[s];
            const double w = std::exp(logits[t * c.E + e] - mx) / zs;
            std::vector<double> h(F);
            for (int64_t f = 0; f < F; ++f)
            {
                double g = 0, u = 0;
                for (int64_t i = 0; i < D; ++i) { g += sm.y[t * D + i] * bfv(gu, ((int64_t) e * 2 * F + f) * D + i); u += sm.y[t * D + i] * bfv(gu, ((int64_t) e * 2 * F + F + f) * D + i); }
                h[f] = silu(g) * u;
            }
            for (int64_t d = 0; d < D; ++d) { double acc = 0; for (int64_t f = 0; f < F; ++f) acc += h[f] * bfv(dn, ((int64_t) e * D + d) * F + f); moe[t * D + d] += w * acc; }
        }
    }
    V sgv = mm(sm.y, T, D, W(m, p + "mlp.shared_expert.gate_proj.weight"), c.SF);
    V suv = mm(sm.y, T, D, W(m, p + "mlp.shared_expert.up_proj.weight"), c.SF);
    for (size_t i = 0; i < sgv.size(); ++i) sgv[i] = silu(sgv[i]) * suv[i];
    V sd = mm(sgv, T, c.SF, W(m, p + "mlp.shared_expert.down_proj.weight"), D);
    V gz = mm(sm.y, T, D, W(m, p + "mlp.shared_expert_gate.weight"), 1);
    for (int64_t t = 0; t < T; ++t) for (int64_t d = 0; d < D; ++d) moe[t * D + d] += sig(gz[t]) * sd[t * D + d];
    for (int64_t t = 0; t < T; ++t) for (int64_t s = 0; s < S; ++s) for (int64_t d = 0; d < D; ++d) X[(t * S + s) * D + d] += sm.inj[t * S + s] * moe[t * D + d];
    return X;
}

int main(int argc, char** argv)
{
    if (argc < 2) { std::fprintf(stderr, "usage: test_qwen4exp_forward <bf16 dir> [layers] [T]\n"); return 2; }
    std::vector<int> layers = {0, 1, 3};
    if (argc > 2) { layers.clear(); std::stringstream ss(argv[2]); std::string t; while (std::getline(ss, t, ',')) layers.push_back(std::stoi(t)); }
    const int64_t T = argc > 3 ? std::atoll(argv[3]) : 64;
    try
    {
        std::ifstream cf(fs::path(argv[1]) / "config.json");
        const Config c(json::parse(cf));
        st::Model m(argv[1]);
        const int64_t S = c.S, D = c.D;
        // Tokens: BOS-free text-ish ids with one EOS in the middle (n-gram reset).
        std::mt19937 rng(4);
        std::vector<int32_t> ids(T);
        for (auto& v : ids) v = (int32_t) (rng() % 150000);
        ids[T / 2] = (int32_t) c.eos;
        // Embedding, expanded to S streams.
        const uint16_t* E = W(m, c.prefix + "embed_tokens.weight");
        std::vector<float> X0((size_t) T * S * D);
        for (int64_t t = 0; t < T; ++t) for (int64_t s = 0; s < S; ++s) for (int64_t d = 0; d < D; ++d) X0[(t * S + s) * D + d] = st::bf16_to_float(E[(int64_t) ids[t] * D + d]);
        // PLE rows from the BF16 table (fp16-rounded), for the GPU and the CPU alike.
        const std::string P = c.lp(c.ple_layer) + "ple.ple_embedding.";
        int64_t mult[3], sizes[16], offs[16];
        std::memcpy(mult, m.get(P + "layer_multipliers").data, 24);
        std::memcpy(sizes, m.get(P + "ngram_heads_vocab_sizes").data, 128);
        std::memcpy(offs, m.get(P + "ngram_heads_offsets").data, 128);
        std::vector<int64_t> nid(T * 16);
        ngram_ids(ids.data(), T, c.eos, mult, sizes, offs, nid.data());
        const int64_t shard_rows = m.get(P + "ngram_embedding.shard_0.weight").shape[0];
        std::vector<uint16_t> rows_h((size_t) T * D);
        V rows_d((size_t) T * D);
        for (int64_t t = 0; t < T; ++t) for (int h = 0; h < 16; ++h)
        {
            const int64_t r = nid[t * 16 + h];
            const uint16_t* src = W(m, P + "ngram_embedding.shard_" + std::to_string(r / shard_rows) + ".weight") + (r % shard_rows) * 160;
            for (int i = 0; i < 160; ++i)
            {
                const uint16_t hb = exl3::f16_from_float(st::bf16_to_float(src[i]));
                rows_h[t * D + h * 160 + i] = hb; rows_d[t * D + h * 160 + i] = exl3::f16_to_double(hb);
            }
        }
        auto q = rt::make_queue();
        rt::print_device(q.get_device());
        Upload up(q);
        Scratch sc(q, c, T);
        DBuf<uint16_t> ws(q, 1);
        const auto inv = c.inv_freq();
        DBuf<float> dinv(q, inv.size()); dinv.upload(inv.data(), inv.size());
        DBuf<half> dple(q, rows_h.size()); q.memcpy(dple.get(), rows_h.data(), rows_h.size() * 2).wait();
        DBuf<float> dx(q, X0.size()); dx.upload(X0.data(), X0.size());
        std::vector<float> xf(X0);
        const int last = *std::max_element(layers.begin(), layers.end());
        for (int li = 0; li <= last; ++li)
        {
            const bool tested = std::find(layers.begin(), layers.end(), li) != layers.end();
            rt::Timer tl;
            Layer L = load_layer(q, m, c, li, up, ws);
            std::vector<float> xin(xf);
            Capture cap;
            if (tested)
            {
                cap.alloc(q, c, L.full);
                DBuf<float> xc(q, X0.size()); xc.upload(xin.data(), xin.size());
                forward_layer(q, c, L, xc.get(), T, sc, dinv.get(), &cap, dple.get());       // capture pass (Hessians)
            }
            forward_layer(q, c, L, dx.get(), T, sc, dinv.get(), nullptr, dple.get());
            dx.download(xf.data(), xf.size());
            if (!tested) { std::printf("layer %d run (%.1f s)\n", li, tl.s()); continue; }
            V yref;
            rt::Timer tc;
            const V ref = cpu_layer(m, c, li, V(xin.begin(), xin.end()), T, rows_d, &yref);
            double num = 0, den = 0; size_t bad = 0;
            for (size_t i = 0; i < ref.size(); ++i) { num += (xf[i] - ref[i]) * (xf[i] - ref[i]); den += ref[i] * ref[i]; bad += !std::isfinite(xf[i]); }
            // Attention-site Hessian vs y^T y (a 256 x 256 corner).
            std::vector<float> Hg((size_t) D * D);
            cap.attn_in.download(Hg.data(), Hg.size());
            double hn = 0, hd = 0;
            for (int64_t i = 0; i < 256; ++i) for (int64_t j = 0; j < 256; ++j)
            {
                double r = 0; for (int64_t t = 0; t < T; ++t) r += yref[t * D + i] * yref[t * D + j];
                hn += (Hg[i * D + j] - r) * (Hg[i * D + j] - r); hd += r * r;
            }
            const double rel = std::sqrt(num / den), hrel = std::sqrt(hn / hd);
            std::printf("layer %d (%s%s): output rel. error %.2e, attention-site Hessian rel. error %.2e, non-finite %zu  (GPU %.1f s, CPU %.1f s)\n",
                        li, L.full ? "full attention" : "Gated DeltaNet", L.ple ? " + PLE" : "", rel, hrel, bad, tl.s() - tc.s(), tc.s());
            if (!(rel < 1e-2) || !(hrel < 1e-2) || bad) ++failures;
        }
    }
    catch (const std::exception& e) { std::fprintf(stderr, "error: %s\n", e.what()); return 2; }
    std::printf(failures ? "\n%d FAILED\n" : "\nall passed\n", failures);
    return failures ? 1 : 0;
}
