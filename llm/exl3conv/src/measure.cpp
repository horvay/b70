// exl3measure: how much each tensor group of a Gemma 4 model's decoder layers matters, by shaped noise injection
// (exllamav3's sc_measure.py --shaped, notes/optimize-recipes.md §2), streamed one layer at a time on the B70.
//
//   exl3measure -i <bf16 dir> --trace cal.safetensors --anchors rfn.json -o noise.json [-w <work dir>]
//               [--rows 10] [--length 1024] [--h-rows 64] [--levels-k 3,4] [--layers-per-pass 8] [--max-minutes M]
//
// Rows: the trace's rows 0..rows-1 are scored, the next h-rows give the Hessians that shape the noise, each cut to
// --length tokens. Targets per layer, as the B70 server fuses them: self_attn.qkv (q, k, v), self_attn.o_proj,
// mlp.gate_up (gate, up), mlp.down_proj. For each target and level the target's weights get noise
//
//   dW = diag(su) H128 L^-T eta diag(col_scale),   eta ~ N(0, 1),   scaled so ||dW|| / ||W|| = level
//
// (L, su: the target group's block-LDL factor and input signs, computed exactly as the quantizer does from the
// clean capture rows; col_scale: the output channels' RMS, mean square one). The level is the anchor error of
// that tensor (--anchors: exl3eval --rfn of a real quant, error at its K) moved to each K of --levels-k by the
// amplitude law of 1.96 per bit. One noise draw per tensor serves every level (sc_measure draws again per level;
// the level only scales it). The perturbed layer runs on the clean input of the scored rows, and the result goes
// on through the clean rest of the model; the score is KL(clean || perturbed) of the softcapped logits, averaged
// over every position of a row, then over rows.
//
// Streaming: a first pass runs the clean model once, writes the reference logits (fp16, raw) and the clean states
// at the start of every pass to <work>. Each pass then spawns the perturbed states of --layers-per-pass layers
// (8 per layer) and carries them, on the card, through every later layer to the head. Exit 99 after a pass that
// ends past --max-minutes (b70q submit --chunked runs it again); the output (noise.json, sc_optimize's format) is
// rewritten after every pass.
#include "arch/gemma4.hpp"
#include "exl3/quantize.hpp"
#include "io/safetensors.hpp"
#include "nn/ops.hpp"
#include "rt/rt.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;
using exl3::DBuf;
using half = sycl::half;
using namespace g4;

namespace {

struct Args
{
    fs::path in, trace, anchors, out, work;
    int rows = 10, h_rows = 64, length = 1024, per_pass = 8;
    std::vector<int> levels_k = {3, 4};
    double max_minutes = 0;
};

// Rows of a state buffer: row r at r * T * H, the first len[r] positions real.
struct RowSet
{
    std::vector<std::vector<int32_t>> ids;
    std::vector<int64_t> len;
    int64_t T = 0;
    int64_t n() const { return (int64_t) len.size(); }
};

inline uint64_t smix(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// x[i] ~ N(0, 1), Box-Muller on a counter hash of (seed, i).
void gauss(sycl::queue& q, float* x, int64_t n, uint64_t seed)
{
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
        const uint64_t i = id[0];
        const uint64_t a = smix(seed ^ (2 * i)), b = smix(seed ^ (2 * i + 1) ^ 0xA5A5A5A5ull);
        const float u1 = ((float) (a >> 40) + 1.0f) * (1.0f / 16777216.0f);
        const float u2 = (float) (b >> 40) * (1.0f / 16777216.0f);
        x[i] = sycl::sqrt(-2.0f * sycl::log(u1)) * sycl::cos(6.283185307179586f * u2);
    }).wait();
}

double sum_sq(sycl::queue& q, const float* x, int64_t rows, int64_t cols)
{
    DBuf<float> part(q, rows);
    float* p = part.get();
    q.parallel_for(sycl::range<1>(rows), [=](sycl::id<1> r) {
        float s = 0; const float* xr = x + r[0] * cols;
        for (int64_t c = 0; c < cols; ++c) s += xr[c] * xr[c];
        p[r] = s;
    }).wait();
    std::vector<float> h(rows);
    part.download(h.data(), rows);
    double s = 0; for (float v : h) s += v;
    return s;
}

// Shaped noise for one weight (hf layout: out x in fp16), as dW (in x out fp32); returns ||W||^2 and ||dW||^2.
struct Noise { DBuf<float> X; double w_sq = 0, x_sq = 0; int64_t k = 0, n = 0; };

Noise shaped_noise(sycl::queue& q, const Lin& l, const exl3::Hessian& hs, const DBuf<float>& d_su, uint64_t seed)
{
    namespace blas = oneapi::mkl::blas::row_major;
    Noise z;
    z.k = l.in; z.n = l.out;
    const int64_t k = z.k, n = z.n;
    z.X = DBuf<float>(q, (size_t) k * n);
    float* X = z.X.get();
    gauss(q, X, k * n, seed);
    // L^T x = eta, L strictly block-lower with an implicit unit diagonal.
    blas::trsm(q, oneapi::mkl::side::left, oneapi::mkl::uplo::lower, oneapi::mkl::transpose::trans, oneapi::mkl::diag::unit,
               k, n, 1.0f, hs.L.get(), k, X, n).wait();
    exl3::k::had_cols(q, X, k, n, n).wait();
    // Output channel weighting: RMS of each output channel of W, mean square one, at least 1e-4.
    DBuf<float> cs(q, n);
    {
        const half* w = l.w.get();
        float* c = cs.get();
        q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> j) {
            float s = 0; const half* wr = w + j[0] * k;
            for (int64_t i = 0; i < k; ++i) { const float v = (float) wr[i]; s += v * v; }
            c[j] = s;
        }).wait();
    }
    std::vector<float> col(n);
    cs.download(col.data(), n);
    double w_sq = 0; for (float v : col) w_sq += v;
    const double ms = w_sq / n;
    for (auto& v : col) v = (float) std::max(1e-4, std::sqrt(v / std::max(ms, 1e-30)));
    cs.upload(col.data(), n);
    exl3::k::scale(q, X, k, n, n, d_su.get(), cs.get()).wait();
    z.w_sq = w_sq;
    z.x_sq = sum_sq(q, X, k, n);
    return z;
}

// W (hf, out x in) += f * dW^T, rounded to fp16; returns ||W_new - W_old||^2.
double apply_noise(sycl::queue& q, Lin& l, const Noise& z, float f)
{
    const int64_t k = z.k, n = z.n;
    half* w = l.w.get();
    const float* X = z.X.get();
    DBuf<float> part(q, n);
    float* p = part.get();
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> jj) {
        const int64_t j = jj[0];
        float s = 0;
        for (int64_t i = 0; i < k; ++i)
        {
            const float o = (float) w[j * k + i];
            const half nw = half(o + f * X[i * n + j]);
            const float d = (float) nw - o;
            s += d * d;
            w[j * k + i] = nw;
        }
        p[j] = s;
    }).wait();
    std::vector<float> h(n);
    part.download(h.data(), n);
    double s = 0; for (float v : h) s += v;
    return s;
}

void fwd_rows(sycl::queue& q, const G4& g, const Layer& L, float* state, const RowSet& rs, Scratch& s, const float* inv, Capture* cap)
{
    for (int64_t r = 0; r < rs.n(); ++r) forward_layer(q, g, L, state + r * rs.T * g.H, rs.len[r], s, inv, cap);
}

void save_state(const fs::path& path, const DBuf<float>& x, size_t n)
{
    std::vector<float> h(n);
    x.download(h.data(), n);
    std::ofstream f(path.string() + ".tmp", std::ios::binary);
    f.write((const char*) h.data(), (std::streamsize) (n * 4));
    f.close();
    fs::rename(path.string() + ".tmp", path);
}

void load_state(const fs::path& path, DBuf<float>& x, size_t n)
{
    std::vector<float> h(n);
    std::ifstream f(path, std::ios::binary);
    if (!f.read((char*) h.data(), (std::streamsize) (n * 4))) throw std::runtime_error("cannot read " + path.string());
    x.upload(h.data(), n);
}

struct Target
{
    std::string key, tag;                    // "...self_attn.qkv"; the Hessian group's tag (same su as the converter)
    std::vector<std::string> members;
    std::vector<Lin*> lins;
    DBuf<float>* H = nullptr;
    int64_t count = 0, k = 0;
};

} // namespace

int main(int argc, char** argv)
{
    Args a;
    for (int i = 1; i < argc; ++i)
    {
        const std::string s = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", s.c_str()); std::exit(2); } return argv[++i]; };
        if (s == "-i") a.in = next(); else if (s == "--trace") a.trace = next(); else if (s == "--anchors") a.anchors = next();
        else if (s == "-o") a.out = next(); else if (s == "-w") a.work = next();
        else if (s == "--rows") a.rows = std::stoi(next()); else if (s == "--h-rows") a.h_rows = std::stoi(next());
        else if (s == "--length") a.length = std::stoi(next()); else if (s == "--layers-per-pass") a.per_pass = std::stoi(next());
        else if (s == "--max-minutes") a.max_minutes = std::stod(next());
        else if (s == "--levels-k") { a.levels_k.clear(); std::stringstream ss(next()); std::string t; while (std::getline(ss, t, ',')) a.levels_k.push_back(std::stoi(t)); }
        else
        {
            std::fprintf(stderr, "usage: exl3measure -i <bf16 dir> --trace cal.safetensors --anchors rfn.json -o noise.json [-w work] [--rows 10]\n"
                                 "       [--length 1024] [--h-rows 64] [--levels-k 3,4] [--layers-per-pass 8] [--max-minutes M]\n");
            return 2;
        }
    }
    if (a.in.empty() || a.trace.empty() || a.anchors.empty() || a.out.empty()) { std::fprintf(stderr, "need -i, --trace, --anchors and -o\n"); return 2; }
    if (a.work.empty()) a.work = a.out.string() + ".work";
    try
    {
        rt::Timer total;
        std::ifstream cf(a.in / "config.json");
        const json cfg = json::parse(cf);
        const G4 g(cfg);
        const json& tc = cfg.contains("text_config") ? cfg["text_config"] : cfg;
        const float cap = tc.value("final_logit_softcapping", 0.0f);
        st::Model m(a.in);
        fs::create_directories(a.work);

        // Rows from the trace.
        RowSet ev, cp;
        {
            st::File tf(a.trace);
            const auto& ids = tf.tensors().at("input_ids");
            const int64_t n = ids.shape[0], c = ids.shape[1];
            const int64_t* p = (const int64_t*) ids.data;
            const int64_t* lp = tf.tensors().count("lengths") ? (const int64_t*) tf.tensors().at("lengths").data : nullptr;
            if (n < a.rows + a.h_rows) throw std::runtime_error("the trace has " + std::to_string(n) + " rows; need --rows + --h-rows");
            for (int64_t r = 0; r < a.rows + a.h_rows; ++r)
            {
                const int64_t len = std::min<int64_t>({lp ? lp[r] : c, c, (int64_t) a.length});
                std::vector<int32_t> row(len);
                for (int64_t t = 0; t < len; ++t) row[t] = (int32_t) p[r * c + t];
                RowSet& rs = r < a.rows ? ev : cp;
                rs.ids.push_back(std::move(row)); rs.len.push_back(len);
            }
            ev.T = cp.T = a.length;
        }
        int64_t ev_tokens = 0; for (auto l : ev.len) ev_tokens += l;

        // Anchors: per tensor (K, rfn).
        std::map<std::string, std::pair<int, double>> anchor;
        {
            std::ifstream af(a.anchors);
            const json aj = json::parse(af);
            for (const auto& r : aj.at("results")) anchor[r.at("key").get<std::string>()] = {r.at("K").get<int>(), r.at("rfn").get<double>()};
        }
        auto level_rfn = [&](const std::string& key, int K) {
            auto it = anchor.find(key);
            if (it == anchor.end()) throw std::runtime_error("no anchor for " + key);
            return it->second.second * std::pow(1.96, it->second.first - K);
        };

        const int64_t P = a.per_pass, npass = (g.L + P - 1) / P;
        const size_t ev_n = (size_t) ev.n() * ev.T * g.H, cp_n = (size_t) cp.n() * cp.T * g.H;
        const auto prog = a.work / "progress.json";
        json pj = fs::exists(prog) ? json::parse(std::ifstream(prog)) : json{{"ref", false}, {"next_pass", 0}};
        json outj = fs::exists(a.out) ? json::parse(std::ifstream(a.out)) : json{{"results", json::array()}};
        outj["model"] = a.in.string(); outj["rows"] = a.rows; outj["length"] = a.length; outj["draws"] = 1; outj["mode"] = "shaped";
        outj["trace"] = a.trace.string(); outj["h_rows"] = a.h_rows; outj["rfn_ref"] = a.anchors.string(); outj["levels_k"] = a.levels_k;
        auto save_json = [&](const fs::path& p, const json& j) { std::ofstream(p.string() + ".tmp") << j.dump(1); fs::rename(p.string() + ".tmp", p); };
        std::printf("exl3measure: %s, %lld layers; %lld scored rows (%lld tokens), %lld Hessian rows; levels at K = ", a.in.filename().c_str(),
                    (long long) g.L, (long long) ev.n(), (long long) ev_tokens, (long long) cp.n());
        for (int K : a.levels_k) std::printf("%d ", K);
        std::printf("; %lld passes of %lld layers\n", (long long) npass, (long long) P);

        auto q = rt::make_queue();
        rt::print_device(q.get_device());
        Upload up(q);
        DBuf<uint16_t> wscratch(q, 1);
        Scratch s(q, g, a.length);
        const auto inv_s = g.inv_freq(false), inv_f = g.inv_freq(true);
        DBuf<float> d_inv_s(q, inv_s.size()), d_inv_f(q, inv_f.size());
        d_inv_s.upload(inv_s.data(), inv_s.size()); d_inv_f.upload(inv_f.data(), inv_f.size());
        const int64_t V = g.vocab, Tc = 256;
        const auto ref_path = a.work / "ref_logits.f16";   // [rows][length][V] fp16 raw logits

        auto embed_rows = [&](const RowSet& rs, DBuf<float>& x) {
            const uint16_t* E = (const uint16_t*) m.get(g.prefix + "embed_tokens.weight").data;
            const float mult = st::bf16_to_float(nn::f2bf16(std::sqrt((float) g.H)));
            std::vector<uint16_t> gath((size_t) rs.T * g.H);
            DBuf<uint16_t> dg(q, gath.size());
            for (int64_t r = 0; r < rs.n(); ++r)
            {
                for (int64_t t = 0; t < rs.len[r]; ++t) std::memcpy(&gath[t * g.H], E + (int64_t) rs.ids[r][t] * g.H, g.H * 2);
                up(dg.get(), gath.data(), (size_t) rs.len[r] * g.H * 2);
                nn::embed(q, dg.get(), rs.len[r], g.H, mult, x.get() + r * rs.T * g.H).wait();
            }
        };

        // The head: final norm and the tied embedding as fp16, loaded only while the layers are off the card.
        struct Head { DBuf<half> Eh; DBuf<uint16_t> normw; };
        auto load_head = [&]() {
            Head h;
            const auto& e = m.get(g.prefix + "embed_tokens.weight");
            DBuf<uint16_t> eb(q, (size_t) e.numel());
            up(eb.get(), e.data, e.nbytes);
            h.Eh = DBuf<half>(q, (size_t) e.numel());
            nn::bf16_to_half(q, eb.get(), e.numel(), h.Eh.get()).wait();
            h.normw = load_bf16_vec(q, m, g.prefix + "norm.weight", up);
            return h;
        };
        DBuf<half> y(q, (size_t) Tc * g.H);
        DBuf<float> Lg(q, (size_t) Tc * V), Lr(q, (size_t) Tc * V), st4(q, Tc * 4);
        DBuf<half> rh(q, (size_t) Tc * V);
        DBuf<int32_t> notgt(q, Tc);
        { std::vector<int32_t> neg(Tc, -1); notgt.upload(neg.data(), Tc); }
        half* pin = sycl::malloc_host<half>((size_t) Tc * V, q);
        auto logits = [&](const Head& h, const float* xr, int64_t c0, int64_t tc) {
            nn::rmsnorm_h(q, xr + c0 * g.H, h.normw.get(), tc, g.H, (float) g.eps, y.get());
            oneapi::mkl::blas::row_major::gemm(q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::trans, tc, V, g.H, 1.0f,
                                               y.get(), g.H, h.Eh.get(), g.H, 0.0f, Lg.get(), V).wait();
        };
        // Mean over rows of the mean over each row's positions of KL(reference || state).
        auto kl_of = [&](const Head& h, const DBuf<float>& x) {
            const int fd = ::open(ref_path.c_str(), O_RDONLY);
            if (fd < 0) throw std::runtime_error("cannot open " + ref_path.string());
            double acc = 0;
            std::vector<float> h4(Tc * 4);
            for (int64_t r = 0; r < ev.n(); ++r)
            {
                double kr = 0;
                for (int64_t c0 = 0; c0 < ev.len[r]; c0 += Tc)
                {
                    const int64_t tcn = std::min(Tc, ev.len[r] - c0);
                    const size_t bytes = (size_t) tcn * V * 2;
                    if (::pread(fd, pin, bytes, (off_t) (((size_t) r * ev.T + c0) * V * 2)) != (ssize_t) bytes) throw std::runtime_error("short read of the reference logits");
                    q.memcpy(rh.get(), pin, bytes).wait();
                    const half* rp = rh.get(); float* lr = Lr.get();
                    q.parallel_for(sycl::range<1>(tcn * V), [=](sycl::id<1> i) { lr[i] = (float) rp[i]; });
                    logits(h, x.get() + r * ev.T * g.H, c0, tcn);
                    nn::logit_stats(q, Lr.get(), Lg.get(), tcn, V, cap, notgt.get(), st4.get()).wait();
                    st4.download(h4.data(), tcn * 4);
                    for (int64_t t = 0; t < tcn; ++t) kr += h4[t * 4];
                }
                acc += kr / ev.len[r];
            }
            ::close(fd);
            return acc / ev.n();
        };

        // Pass 0: the clean model; reference logits and the clean states at every pass start.
        if (!pj.value("ref", false))
        {
            rt::Timer t;
            DBuf<float> E(q, ev_n), C(q, cp_n);
            embed_rows(ev, E); embed_rows(cp, C);
            for (int64_t li = 0; li < g.L; ++li)
            {
                if (li % P == 0)
                {
                    save_state(a.work / ("eval_" + std::to_string(li) + ".f32"), E, ev_n);
                    save_state(a.work / ("cap_" + std::to_string(li) + ".f32"), C, cp_n);
                }
                Layer L = load_layer(q, m, g, li, up, wscratch);
                const float* inv = L.full ? d_inv_f.get() : d_inv_s.get();
                fwd_rows(q, g, L, E.get(), ev, s, inv, nullptr);
                fwd_rows(q, g, L, C.get(), cp, s, inv, nullptr);
                if (li % 10 == 9) { std::printf("  reference: layer %lld (%.0f s)\n", (long long) li + 1, t.s()); std::fflush(stdout); }
            }
            C.reset();
            Head h = load_head();
            std::ofstream rf(ref_path.string() + ".tmp", std::ios::binary);
            for (int64_t r = 0; r < ev.n(); ++r)
                for (int64_t c0 = 0; c0 < ev.T; c0 += Tc)
                {
                    const int64_t tcn = std::min(Tc, ev.T - c0);
                    if (c0 < ev.len[r])
                    {
                        const int64_t tv = std::min(tcn, ev.len[r] - c0);
                        logits(h, E.get() + r * ev.T * g.H, c0, tv);
                        const float* lg = Lg.get(); half* rp = rh.get();
                        q.parallel_for(sycl::range<1>(tv * V), [=](sycl::id<1> i) { rp[i] = half(lg[i]); }).wait();
                        q.memset(rh.get() + tv * V, 0, (size_t) (tcn - tv) * V * 2).wait();
                    }
                    else q.memset(rh.get(), 0, (size_t) tcn * V * 2).wait();
                    q.memcpy(pin, rh.get(), (size_t) tcn * V * 2).wait();
                    rf.write((const char*) pin, (std::streamsize) ((size_t) tcn * V * 2));
                }
            rf.close();
            fs::rename(ref_path.string() + ".tmp", ref_path);
            // Control: the clean state against its own fp16 reference (rounding only).
            std::printf("reference pass done in %.0f s; control KL %.3g (fp16 rounding of the reference)\n", t.s(), kl_of(h, E));
            pj["ref"] = true;
            save_json(prog, pj);
            if (a.max_minutes > 0 && total.s() > a.max_minutes * 60) { std::printf("turn over; run again to continue\n"); sycl::free(pin, q); return 99; }
        }

        for (int64_t p = pj.value("next_pass", 0); p < npass; ++p)
        {
            rt::Timer tp;
            const int64_t a0 = p * P, b0 = std::min<int64_t>(g.L, a0 + P);
            DBuf<float> E(q, ev_n), C(q, cp_n);
            load_state(a.work / ("eval_" + std::to_string(a0) + ".f32"), E, ev_n);
            load_state(a.work / ("cap_" + std::to_string(a0) + ".f32"), C, cp_n);
            struct Pend { size_t target; size_t level; DBuf<float> x; double rfn_actual = 0, rfn_nominal = 0; };
            std::vector<Pend> pend;
            struct Rec { std::string key; std::vector<std::string> members; int64_t numel = 0, idx = 0; double w_norm = 0; };
            std::vector<Rec> recs;
            for (int64_t li = a0; li < g.L; ++li)
            {
                Layer L = load_layer(q, m, g, li, up, wscratch);
                const float* inv = L.full ? d_inv_f.get() : d_inv_s.get();
                // Older perturbed states go through the clean layer.
                for (auto& pe : pend) fwd_rows(q, g, L, pe.x.get(), ev, s, inv, nullptr);
                if (li < b0)
                {
                    rt::Timer tl;
                    const auto pfx = g.lp(li);
                    Capture capt;
                    const int64_t QD = g.NH * L.D;
                    capt.qkv = DBuf<float>(q, (size_t) g.H * g.H); capt.qkv.zero();
                    capt.o = DBuf<float>(q, (size_t) QD * QD); capt.o.zero();
                    capt.gu = DBuf<float>(q, (size_t) g.H * g.H); capt.gu.zero();
                    capt.d = DBuf<float>(q, (size_t) g.I * g.I); capt.d.zero();
                    fwd_rows(q, g, L, C.get(), cp, s, inv, &capt);   // capture rows advance to the next layer here
                    std::vector<Target> tg(4);
                    tg[0] = {pfx + "self_attn.qkv", "attn.input", {pfx + "self_attn.q_proj", pfx + "self_attn.k_proj"}, {&L.q, &L.k}, &capt.qkv, capt.n_qkv, g.H};
                    if (!(L.full && g.k_eq_v)) { tg[0].members.push_back(pfx + "self_attn.v_proj"); tg[0].lins.push_back(&L.v); }
                    tg[1] = {pfx + "self_attn.o_proj", "attn.o", {pfx + "self_attn.o_proj"}, {&L.o}, &capt.o, capt.n_o, QD};
                    tg[2] = {pfx + "mlp.gate_up", "mlp.input", {pfx + "mlp.gate_proj", pfx + "mlp.up_proj"}, {&L.g, &L.u}, &capt.gu, capt.n_gu, g.H};
                    tg[3] = {pfx + "mlp.down_proj", "mlp.down", {pfx + "mlp.down_proj"}, {&L.d}, &capt.d, capt.n_d, g.I};
                    for (auto& t : tg)
                    {
                        const uint64_t seed = exl3::mix64(exl3::hash_str(pfx + t.tag));
                        exl3::Hessian hs = exl3::finalize_hessian(q, std::move(*t.H), t.k, t.count, seed);
                        DBuf<float> d_su(q, t.k);
                        d_su.upload(hs.su0.data(), t.k);
                        std::vector<Noise> nz;
                        std::vector<DBuf<half>> backup;
                        Rec rec; rec.key = t.key; rec.members = t.members; rec.idx = li + 1;
                        double wsq = 0;
                        for (size_t mi = 0; mi < t.lins.size(); ++mi)
                        {
                            nz.push_back(shaped_noise(q, *t.lins[mi], hs, d_su, exl3::mix64(exl3::hash_str(t.members[mi] + "|noise"))));
                            backup.emplace_back(q, (size_t) t.lins[mi]->in * t.lins[mi]->out);
                            q.memcpy(backup.back().get(), t.lins[mi]->w.get(), backup.back().n * 2).wait();
                            rec.numel += t.lins[mi]->in * t.lins[mi]->out;
                            wsq += nz.back().w_sq;
                        }
                        rec.w_norm = std::sqrt(wsq);
                        hs.L.reset(); hs.Hd.reset();
                        for (size_t lv = 0; lv < a.levels_k.size(); ++lv)
                        {
                            double err = 0, nom = 0;
                            for (size_t mi = 0; mi < nz.size(); ++mi)
                            {
                                const double rfn = level_rfn(t.members[mi], a.levels_k[lv]);
                                const float f = (float) (rfn * std::sqrt(nz[mi].w_sq / nz[mi].x_sq));
                                err += apply_noise(q, *t.lins[mi], nz[mi], f);
                                nom += rfn * rfn * (double) (t.lins[mi]->in * t.lins[mi]->out);
                            }
                            Pend pe{recs.size(), lv, DBuf<float>(q, ev_n), std::sqrt(err / wsq), std::sqrt(nom / rec.numel)};
                            q.memcpy(pe.x.get(), E.get(), ev_n * 4).wait();
                            fwd_rows(q, g, L, pe.x.get(), ev, s, inv, nullptr);
                            for (size_t mi = 0; mi < nz.size(); ++mi) q.memcpy(t.lins[mi]->w.get(), backup[mi].get(), backup[mi].n * 2).wait();
                            pend.push_back(std::move(pe));
                        }
                        recs.push_back(std::move(rec));
                    }
                    // The clean scored rows go on to the next layer.
                    fwd_rows(q, g, L, E.get(), ev, s, inv, nullptr);
                    std::printf("  pass %lld/%lld: layer %lld spawned (%.1f s), %zu states on the card\n", (long long) p + 1,
                                (long long) npass, (long long) li, tl.s(), pend.size());
                    std::fflush(stdout);
                    if (li == b0 - 1) { E.reset(); C.reset(); }
                }
                else if (li % 10 == 9) { std::printf("  pass %lld/%lld: layer %lld (%.0f s)\n", (long long) p + 1, (long long) npass, (long long) li + 1, tp.s()); std::fflush(stdout); }
            }
            // Scores.
            Head h = load_head();
            std::vector<json> res(recs.size());
            for (size_t r = 0; r < recs.size(); ++r)
            {
                res[r] = {{"idx", recs[r].idx}, {"key", recs[r].key}, {"qbits_key", "bits"}, {"numel", recs[r].numel}, {"shaped", true},
                          {"w_norm", recs[r].w_norm}, {"levels", json::array()}};
                if (recs[r].members.size() > 1) res[r]["members"] = recs[r].members;
            }
            for (auto& pe : pend)
            {
                const double kld = kl_of(h, pe.x);
                res[pe.target]["levels"].push_back({{"rfn", pe.rfn_nominal}, {"rfn_actual", pe.rfn_actual}, {"draw", 0}, {"kld", kld},
                                                     {"level_k", a.levels_k[pe.level]}});
                pe.x.reset();
            }
            // Replace this pass's keys in the output, keep the rest.
            json all = json::array();
            std::map<std::string, bool> mine;
            for (auto& r : res) mine[r["key"].get<std::string>()] = true;
            for (auto& r : outj["results"]) if (!mine.count(r["key"].get<std::string>())) all.push_back(r);
            for (auto& r : res) all.push_back(r);
            std::sort(all.begin(), all.end(), [](const json& x, const json& y) { return x["idx"].get<int64_t>() < y["idx"].get<int64_t>(); });
            outj["results"] = all;
            save_json(a.out, outj);
            pj["next_pass"] = p + 1;
            save_json(prog, pj);
            std::printf("pass %lld/%lld (layers %lld-%lld) done in %.0f s; %zu targets in %s\n", (long long) p + 1, (long long) npass,
                        (long long) a0, (long long) b0 - 1, tp.s(), outj["results"].size(), a.out.c_str());
            std::fflush(stdout);
            if (p + 1 < npass && a.max_minutes > 0 && total.s() > a.max_minutes * 60) { std::printf("turn over; run again to continue\n"); sycl::free(pin, q); return 99; }
        }
        sycl::free(pin, q);
        std::printf("done in %.1f min\n", total.s() / 60);
        return 0;
    }
    catch (const std::exception& ex) { std::fprintf(stderr, "error: %s\n", ex.what()); return 1; }
}
