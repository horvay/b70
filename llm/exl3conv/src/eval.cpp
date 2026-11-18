// exl3eval: quality of EXL3 models against their BF16 source, on the B70.
//
//   exl3eval --ref <bf16 dir> --test <exl3 dir> [--test <exl3 dir> ...] [--chat F.jsonl] [--cols 2048]
//   exl3eval ... --wiki [--rows 32] [--cal-dir D]
//   exl3eval ... --trace heldout.json      (the model's own replies: tools/make_trace.py pack --heldout)
//   exl3eval --ref <bf16 dir> --test <exl3 dir> --rfn out.json
//   exl3eval ... --layers N                (the first N layers only, a test on part of a checkpoint)
//   exl3eval ... --dump F                  (each scored token's top-64 log-probabilities, JSON lines: the first --test to F,
//                                           BF16 to F.ref; what another engine's teacher-forced run is compared with)
//
// --rfn: no forward; per decoder linear of the (first) test model, its relative error ||W_q - W|| / ||W|| against
// the BF16 weight (rounded to fp16, as the converter quantizes it), with its K: the anchors exl3measure scales its
// noise by (exllamav3's sc_rfn_probe.py, notes/optimize-recipes.md §3).
//
// Text, by default: conversations (data/oasst2-eval.jsonl: 200 human-written OpenAssistant conversations, built by
// tools/chat_eval_set.py) rendered with the model's chat template (io/chat.hpp), one conversation a row; only the
// assistant replies and their end-of-turn tokens are scored, so the model is measured in the mode it is used in.
// --wiki: Wikipedia articles from the end of exllamav3's wiki.utf8 instead, rows of cols tokens, every token scored
// (fine-tuned merges are far from their training distribution on raw text, which inflates every number).
// Every model runs through the same forward as the converter (arch/gemma4.hpp, arch/ripple.hpp), layer by layer in lockstep;
// EXL3 linears are dequantized to the fp16 weight inference sees. Per scored token: KL(BF16 || EXL3) over the full
// vocabulary of the logits (softcapped on Gemma 4), whether the top-1 token matches BF16, and the log-likelihood of the
// real next token under each model. Intervals are 95% bootstrap intervals that resample whole rows (conversations).
#include "arch/gemma4.hpp"
#include "arch/ripple.hpp"
#include "io/caldata.hpp"
#include "io/chat.hpp"
#include "io/tokenizer.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace g4;

// What the scoring loop needs of an architecture. Model 0 is the BF16 reference, then the quants; every model's state is [tokens, H] fp32.
struct Arch
{
    int64_t H = 0, V = 0, L = 0;
    int bos = 0;
    float cap = 0;     // softcap of the logits, 0: none
    std::string prefix;
    virtual ~Arch() = default;
    virtual std::string lp(int64_t li) const = 0;
    virtual std::vector<std::string> linears() const = 0;   // of a layer, for --rfn
    virtual void setup(sycl::queue& q, Upload& up, const std::vector<const st::Model*>& models, const DBuf<int>& d_perm, int64_t Tmax, const std::vector<chat::Row>& rows) = 0;
    virtual void embed(sycl::queue& q, Upload& up, int64_t r, float* x) = 0;
    virtual void layer(sycl::queue& q, Upload& up, const std::vector<const st::Model*>& models, size_t mi, int64_t li, const std::vector<float*>& row_x, const std::vector<chat::Row>& rows) = 0;
    virtual void setup_heads(sycl::queue& q, Upload& up, const std::vector<const st::Model*>& models, const DBuf<int>& d_perm) = 0;
    // Raw logits of tc tokens, Lg [tc, V] fp32, from their state x.
    virtual void logits(sycl::queue& q, size_t mi, const float* x, int64_t tc, float* Lg) = 0;
};

struct G4Arch : Arch
{
    G4 g;
    const uint16_t* Eh_src = nullptr;
    std::vector<int32_t> ids;
    std::vector<uint16_t> gath;
    std::unique_ptr<DBuf<uint16_t>> dg, ws;
    std::unique_ptr<DBuf<float>> tmp, d_inv_s, d_inv_f;
    std::unique_ptr<Scratch> s;
    DBuf<half> Eh, y;
    DBuf<uint16_t> normw;
    const DBuf<int>* perm = nullptr;
    const std::vector<chat::Row>* rows_ = nullptr;
    explicit G4Arch(const json& cfg) : g(cfg) { H = g.H; V = g.vocab; L = g.L; bos = g.bos; cap = 30.0f; prefix = g.prefix; }
    std::string lp(int64_t li) const override { return g.lp(li); }
    std::vector<std::string> linears() const override
    {
        return {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"};
    }
    void setup(sycl::queue& q, Upload&, const std::vector<const st::Model*>& models, const DBuf<int>& d_perm, int64_t Tmax, const std::vector<chat::Row>& rows) override
    {
        Eh_src = (const uint16_t*) models[0]->get(g.prefix + "embed_tokens.weight").data;
        perm = &d_perm; rows_ = &rows;
        gath.resize((size_t) Tmax * g.H);
        dg = std::make_unique<DBuf<uint16_t>>(q, (size_t) Tmax * g.H);
        s = std::make_unique<Scratch>(q, g, Tmax);
        ws = std::make_unique<DBuf<uint16_t>>(q, 1);
        tmp = std::make_unique<DBuf<float>>(q, 1);
        const auto inv_s = g.inv_freq(false), inv_f = g.inv_freq(true);
        d_inv_s = std::make_unique<DBuf<float>>(q, inv_s.size());
        d_inv_f = std::make_unique<DBuf<float>>(q, inv_f.size());
        d_inv_s->upload(inv_s.data(), inv_s.size()); d_inv_f->upload(inv_f.data(), inv_f.size());
    }
    void embed(sycl::queue& q, Upload& up, int64_t r, float* x) override
    {
        const float mult = st::bf16_to_float(nn::f2bf16(std::sqrt((float) g.H)));
        const auto& row = (*rows_)[r];
        const int64_t n = (int64_t) row.ids.size();
        for (int64_t t = 0; t < n; ++t) std::memcpy(&gath[t * g.H], Eh_src + (int64_t) row.ids[t] * g.H, g.H * 2);
        up(dg->get(), gath.data(), (size_t) n * g.H * 2);
        nn::embed(q, dg->get(), n, g.H, mult, x).wait();
    }
    void layer(sycl::queue& q, Upload& up, const std::vector<const st::Model*>& models, size_t mi, int64_t li, const std::vector<float*>& row_x, const std::vector<chat::Row>& rows) override
    {
        Layer L1 = mi == 0 ? load_layer(q, *models[0], g, li, up, *ws) : load_layer_exl3(q, *models[mi], g, li, up, *perm, *tmp);
        const float* inv = L1.full ? d_inv_f->get() : d_inv_s->get();
        for (size_t r = 0; r < rows.size(); ++r) forward_layer(q, g, L1, row_x[r], (int64_t) rows[r].ids.size(), *s, inv, nullptr);
    }
    void setup_heads(sycl::queue& q, Upload& up, const std::vector<const st::Model*>& models, const DBuf<int>&) override
    {
        // Tied embedding as fp16; final norm from the reference.
        const auto& e = models[0]->get(g.prefix + "embed_tokens.weight");
        DBuf<uint16_t> eb(q, (size_t) e.numel());
        up(eb.get(), e.data, e.nbytes);
        Eh = DBuf<half>(q, (size_t) e.numel());
        nn::bf16_to_half(q, eb.get(), e.numel(), Eh.get()).wait();
        normw = load_bf16_vec(q, *models[0], g.prefix + "norm.weight", up);
        y = DBuf<half>(q, (size_t) 128 * g.H);
    }
    void logits(sycl::queue& q, size_t, const float* x, int64_t tc, float* Lg) override
    {
        nn::rmsnorm_h(q, x, normw.get(), tc, g.H, (float) g.eps, y.get());
        blas::gemm(q, transpose::nontrans, transpose::trans, tc, V, g.H, 1.0f, y.get(), g.H, Eh.get(), g.H, 0.0f, Lg, V);
    }
};

struct RpArch : Arch
{
    rp::Rp g;
    std::unique_ptr<rp::Scratch> s;
    std::unique_ptr<DBuf<uint16_t>> ws;
    std::unique_ptr<DBuf<float>> tmp, d_inv;
    DBuf<uint16_t> E, normw;
    DBuf<int32_t> dids;
    DBuf<half> y;
    std::vector<Lin> heads;
    const DBuf<int>* perm = nullptr;
    std::vector<int64_t> id_off;
    explicit RpArch(const json& cfg, int64_t layers) : g(cfg) { H = g.H; V = g.vocab; L = layers > 0 ? std::min(g.L, layers) : g.L; bos = g.bos; cap = 0; prefix = g.prefix; }
    std::string lp(int64_t li) const override { return g.lp(li); }
    std::vector<std::string> linears() const override
    {
        return {"self_attn.q_proj", "self_attn.k_proj", "self_attn.g_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"};
    }
    void setup(sycl::queue& q, Upload& up, const std::vector<const st::Model*>& models, const DBuf<int>& d_perm, int64_t Tmax, const std::vector<chat::Row>& rows) override
    {
        const auto& et = models[0]->get(g.prefix + "embed_tokens.weight");
        E = DBuf<uint16_t>(q, (size_t) et.numel());
        up(E.get(), et.data, et.nbytes);
        // Every row's tokens on the card: v_proj reads their embedding in every layer.
        id_off.assign(rows.size() + 1, 0);
        for (size_t r = 0; r < rows.size(); ++r) id_off[r + 1] = id_off[r] + (int64_t) rows[r].ids.size();
        std::vector<int32_t> all;
        all.reserve(id_off.back());
        for (const auto& r : rows) all.insert(all.end(), r.ids.begin(), r.ids.end());
        dids = DBuf<int32_t>(q, all.size());
        dids.upload(all.data(), all.size());
        perm = &d_perm;
        s = std::make_unique<rp::Scratch>(q, g, Tmax);
        ws = std::make_unique<DBuf<uint16_t>>(q, 1);
        tmp = std::make_unique<DBuf<float>>(q, 1);
        const auto inv = g.inv_freq();
        d_inv = std::make_unique<DBuf<float>>(q, inv.size());
        d_inv->upload(inv.data(), inv.size());
    }
    void embed(sycl::queue& q, Upload&, int64_t r, float* x) override
    {
        nn::embed_row(q, E.get(), dids.get() + id_off[r], g.H, g.mult, id_off[r + 1] - id_off[r], x).wait();
    }
    void layer(sycl::queue& q, Upload& up, const std::vector<const st::Model*>& models, size_t mi, int64_t li, const std::vector<float*>& row_x, const std::vector<chat::Row>& rows) override
    {
        rp::Layer L1 = mi == 0 ? rp::load_layer(q, *models[0], g, li, up, *ws) : rp::load_layer_exl3(q, *models[mi], g, li, up, *perm, *tmp);
        for (size_t r = 0; r < rows.size(); ++r) rp::forward_layer(q, g, L1, row_x[r], E.get(), dids.get() + id_off[r], (int64_t) rows[r].ids.size(), *s, d_inv->get(), nullptr);
    }
    void setup_heads(sycl::queue& q, Upload& up, const std::vector<const st::Model*>& models, const DBuf<int>& d_perm) override
    {
        for (const auto* m : models) heads.push_back(rp::load_head(q, *m, up, d_perm, *tmp, *ws));
        *tmp = DBuf<float>(q, 1);
        *ws = DBuf<uint16_t>(q, 1);
        normw = load_bf16_vec(q, *models[0], g.prefix + "norm.weight", up);
        y = DBuf<half>(q, (size_t) 128 * g.H);
    }
    void logits(sycl::queue& q, size_t mi, const float* x, int64_t tc, float* Lg) override
    {
        nn::rmsnorm_h(q, x, normw.get(), tc, g.H, g.eps, y.get());
        gemm_lin(q, y.get(), tc, heads[mi], Lg);
    }
};

int main(int argc, char** argv)
{
    fs::path ref, cal_dir, chat_file, rfn_out, trace_file, dump_file;
    std::vector<fs::path> tests;
    int64_t R = 32, T = 2048, layers = -1;
    bool wiki = false;
    for (int i = 1; i < argc; ++i)
    {
        const std::string s = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", s.c_str()); std::exit(2); } return argv[++i]; };
        if (s == "--ref") ref = next(); else if (s == "--test") tests.push_back(next());
        else if (s == "--rows") R = std::stoll(next()); else if (s == "--cols") T = std::stoll(next());
        else if (s == "--cal-dir") cal_dir = next(); else if (s == "--chat") chat_file = next(); else if (s == "--wiki") wiki = true;
        else if (s == "--rfn") rfn_out = next(); else if (s == "--trace") trace_file = next();
        else if (s == "--dump") dump_file = next(); else if (s == "--layers") layers = std::stoll(next());
        else { std::fprintf(stderr, "usage: exl3eval --ref <bf16 dir> --test <exl3 dir> [--test ...] [--chat F.jsonl] [--cols T] | [--wiki --rows N --cal-dir D]\n"); return 2; }
    }
    if (ref.empty() || tests.empty()) { std::fprintf(stderr, "need --ref and at least one --test\n"); return 2; }
    if (cal_dir.empty()) cal_dir = fs::path(argv[0]).parent_path() / "../../dsv41/exllamav3/exllamav3/conversion/standard_cal_data";
    if (chat_file.empty()) chat_file = fs::path(argv[0]).parent_path() / "../data/oasst2-eval.jsonl";
    try
    {
        std::ifstream cf(ref / "config.json");
        const json cfg = json::parse(cf);
        const std::string model_type = cfg.value("model_type", std::string());
        std::unique_ptr<Arch> arch;
        if (model_type == "ricursive") arch = std::make_unique<RpArch>(cfg, layers);
        else { arch = std::make_unique<G4Arch>(cfg); if (layers > 0) arch->L = std::min(arch->L, layers); }
        Arch& g = *arch;
        st::Model mref(ref);
        std::vector<std::unique_ptr<st::Model>> mt;
        for (auto& t : tests) mt.push_back(std::make_unique<st::Model>(t));
        const size_t NM = 1 + tests.size();
        std::vector<const st::Model*> models = {&mref};
        for (auto& m1 : mt) models.push_back(m1.get());

        // Rows: token ids, which targets are scored, and where each row starts in the stacked state.
        tok::BPE tk((ref / "tokenizer.json").string());
        std::vector<chat::Row> rows;
        if (wiki)
        {
            // Held-out articles from the end of wiki.utf8, each row BOS + text, every target scored.
            auto arts = cal::split(cal::read_file(cal_dir / "wiki.utf8"), "</doc>\n");
            chat::Row row; row.ids = {g.bos};
            for (int64_t a = (int64_t) arts.size() - 1; a >= 0 && (int64_t) rows.size() < R; --a)
            {
                const auto nl = arts[a].find('\n');
                const std::string body = nl == std::string::npos ? arts[a] : arts[a].substr(nl + 1);
                if (body.size() < 200) continue;
                for (int t : tk.encode(body))
                {
                    row.ids.push_back(t);
                    if ((int64_t) row.ids.size() == T) { row.scored.assign(T, 1); row.scored[0] = 0; rows.push_back(row); row.ids = {g.bos}; break; }
                }
            }
            if ((int64_t) rows.size() < R) throw std::runtime_error("not enough held-out text");
        }
        else if (!trace_file.empty())
        {
            // Held-out self-sampled rows (tools/make_trace.py pack --heldout): context + the model's reply, the reply
            // scored, cut to cols tokens.
            std::ifstream tf(trace_file);
            const json tj = json::parse(tf);
            for (const auto& r : tj.at("rows"))
            {
                chat::Row row;
                for (int t : r.at("input_ids").get<std::vector<int>>()) { row.ids.push_back(t); row.scored.push_back(0); }
                for (int t : r.at("response_ids").get<std::vector<int>>()) { row.ids.push_back(t); row.scored.push_back(1); }
                if ((int64_t) row.ids.size() > T) { row.ids.resize(T); row.scored.resize(T); }
                if (row.n_scored() >= 32) rows.push_back(std::move(row));
            }
            R = (int64_t) rows.size();
        }
        else
        {
            rows = chat::load(tk, chat_file.string(), model_type, g.bos, (size_t) T);
            R = (int64_t) rows.size();
        }
        std::vector<int64_t> off(R + 1, 0);
        int64_t Tmax = 0, nscored = 0;
        for (int64_t r = 0; r < R; ++r)
        {
            off[r + 1] = off[r] + (int64_t) rows[r].ids.size();
            Tmax = std::max<int64_t>(Tmax, (int64_t) rows[r].ids.size());
            nscored += rows[r].n_scored();
        }
        if (wiki) std::printf("exl3eval: %lld rows x %lld tokens of held-out Wikipedia text\n", (long long) R, (long long) T);
        else std::printf("exl3eval: %lld conversations from %s, %lld tokens, %lld scored (assistant replies)\n", (long long) R,
                         (trace_file.empty() ? chat_file : trace_file).filename().c_str(), (long long) off[R], (long long) nscored);

        auto q = rt::make_queue();
        rt::print_device(q.get_device());
        Upload up(q);
        DBuf<int> d_perm(q, 256);
        d_perm.upload(exl3::tile_perm().data(), 256);
        if (!rfn_out.empty())
        {
            json res = json::array();
            DBuf<float> tmp(q, 1);
            for (int64_t li = 0; li < g.L; ++li)
            {
                const auto p = g.lp(li);
                for (const auto& n : g.linears())
                {
                    const std::string key = p + n;
                    if (!mt[0]->has(key + ".trellis")) continue;
                    const auto& tr = mt[0]->get(key + ".trellis");
                    const int K = (int) (tr.shape[2] / 16);
                    Lin t = load_exl3_linear(q, *mt[0], key, up, d_perm, tmp);    // (in, out)
                    std::vector<half> wq((size_t) t.in * t.out);
                    q.memcpy(wq.data(), t.w.get(), wq.size() * 2).wait();
                    const auto& w = mref.get(key + ".weight");                     // BF16 (out, in)
                    const uint16_t* wb = (const uint16_t*) w.data;
                    double num = 0, den = 0;
                    for (int64_t o = 0; o < t.out; ++o)
                        for (int64_t i = 0; i < t.in; ++i)
                        {
                            const double r = (double) (float) half(st::bf16_to_float(wb[o * t.in + i]));
                            const double d = (double) (float) wq[i * t.out + o] - r;
                            num += d * d; den += r * r;
                        }
                    res.push_back({{"key", key}, {"K", K}, {"qbits_key", "bits"}, {"numel", t.in * t.out}, {"rfn", std::sqrt(num / den)},
                                   {"ref_norm", std::sqrt(den)}});
                }
                if (li % 10 == 9) { std::printf("  layer %lld\n", (long long) li + 1); std::fflush(stdout); }
            }
            std::ofstream(rfn_out) << json{{"model_q", tests[0].string()}, {"model_r", ref.string()}, {"results", res}}.dump(1);
            std::printf("%zu tensors -> %s\n", res.size(), rfn_out.c_str());
            return 0;
        }
        std::vector<DBuf<float>> state;
        for (size_t mi = 0; mi < NM; ++mi) state.emplace_back(q, (size_t) off[R] * g.H);
        auto row_x = [&](size_t mi, int64_t r) { return state[mi].get() + off[r] * g.H; };
        g.setup(q, up, models, d_perm, Tmax, rows);
        for (int64_t r = 0; r < R; ++r) g.embed(q, up, r, row_x(0, r));
        for (size_t mi = 1; mi < NM; ++mi) q.memcpy(state[mi].get(), state[0].get(), (size_t) off[R] * g.H * 4).wait();
        rt::Timer tt;
        for (int64_t li = 0; li < g.L; ++li)
        {
            for (size_t mi = 0; mi < NM; ++mi)
            {
                std::vector<float*> xs(R);
                for (int64_t r = 0; r < R; ++r) xs[r] = row_x(mi, r);
                g.layer(q, up, models, mi, li, xs, rows);
            }
            if (li % 10 == 9) { std::printf("  layer %lld (%.0f s)\n", (long long) li + 1, tt.s()); std::fflush(stdout); }
        }
        // Head: final norm, then the architecture's lm_head (tied embedding as fp16 and a softcap on Gemma 4; Ripple's own).
        g.setup_heads(q, up, models, d_perm);
        const float cap = g.cap;
        const int64_t Tc = 128, V = g.V;
        std::vector<DBuf<float>> Lg;
        for (size_t mi = 0; mi < NM; ++mi) Lg.emplace_back(q, (size_t) Tc * V);
        DBuf<float> st4(q, Tc * 4);
        DBuf<int32_t> tgt(q, Tc);
        // Per row and model: sums of KL, top-1 matches and NLL over the scored targets; and every scored token's KL.
        struct Sums { double kl = 0, top1 = 0, nll = 0; };
        std::vector<std::vector<Sums>> rs(NM, std::vector<Sums>(R));
        std::vector<std::vector<float>> tok_kl(NM);
        std::vector<int64_t> rn(R, 0);
        std::vector<float> h4(Tc * 4);
        for (int64_t r = 0; r < R; ++r)
        {
            const auto& row = rows[r];
            const int64_t n = (int64_t) row.ids.size();
            for (int64_t c0 = 0; c0 < n - 1; c0 += Tc)
            {
                const int64_t tc = std::min(Tc, n - 1 - c0);
                bool any = false;
                for (int64_t t = 0; t < tc; ++t) any |= row.scored[c0 + t + 1] != 0;
                if (!any) continue;
                std::vector<int32_t> ht(tc);
                for (int64_t t = 0; t < tc; ++t) ht[t] = row.ids[c0 + t + 1];
                tgt.upload(ht.data(), tc);
                for (size_t mi = 0; mi < NM; ++mi) g.logits(q, mi, row_x(mi, r) + c0 * g.H, tc, Lg[mi].get());
                if (!dump_file.empty())
                    for (size_t mi = 0; mi < 2; ++mi)
                    {
                        static FILE* fo[2] = {std::fopen((dump_file.string() + ".ref").c_str(), "w"), std::fopen(dump_file.c_str(), "w")};
                        std::vector<float> lh((size_t) tc * V);
                        q.memcpy(lh.data(), Lg[mi].get(), lh.size() * 4).wait();
                        for (int64_t t = 0; t < tc; ++t)
                        {
                            if (!row.scored[c0 + t + 1]) continue;
                            float* l = lh.data() + (size_t) t * V;
                            double mx = -1e30;
                            for (int64_t v = 0; v < V; ++v) { if (cap > 0) l[v] = cap * std::tanh(l[v] / cap); mx = std::max(mx, (double) l[v]); }
                            double se = 0;
                            for (int64_t v = 0; v < V; ++v) se += std::exp(l[v] - mx);
                            const double lse = mx + std::log(se);
                            std::vector<int> idx(V);
                            for (int64_t v = 0; v < V; ++v) idx[v] = (int) v;
                            std::partial_sort(idx.begin(), idx.begin() + 64, idx.end(), [&](int a, int b) { return l[a] > l[b]; });
                            std::fprintf(fo[mi], "{\"pos\": %lld, \"next\": %d, \"next_lp\": %.6f, \"top\": [", (long long) (c0 + t), row.ids[c0 + t + 1], l[row.ids[c0 + t + 1]] - lse);
                            for (int k = 0; k < 64; ++k) std::fprintf(fo[mi], "%s[%d, %.6f]", k ? ", " : "", idx[k], l[idx[k]] - lse);
                            std::fprintf(fo[mi], "]}\n");
                        }
                        std::fflush(fo[mi]);
                    }
                for (size_t mi = 0; mi < NM; ++mi)
                {
                    nn::logit_stats(q, Lg[0].get(), Lg[mi].get(), tc, V, cap, tgt.get(), st4.get()).wait();
                    st4.download(h4.data(), tc * 4);
                    for (int64_t t = 0; t < tc; ++t)
                    {
                        if (!row.scored[c0 + t + 1]) continue;
                        // h4: KL, top-1 match, NLL under the tested model, NLL under BF16.
                        rs[mi][r].kl += h4[t * 4]; rs[mi][r].top1 += h4[t * 4 + 1];
                        rs[mi][r].nll += mi == 0 ? h4[t * 4 + 3] : h4[t * 4 + 2];
                        if (mi > 0) tok_kl[mi].push_back(h4[t * 4]);
                        if (mi == 0) ++rn[r];
                    }
                }
            }
        }
        int64_t ntok = 0;
        for (auto c : rn) ntok += c;
        // Bootstrap over rows: the same resamples for every model, so differences between models are paired.
        const int B = 2000;
        std::mt19937_64 rng(1);
        std::uniform_int_distribution<int64_t> pick(0, R - 1);
        std::vector<std::vector<int64_t>> draws(B, std::vector<int64_t>(R));
        for (auto& d : draws) for (auto& x : d) x = pick(rng);
        auto ci = [&](auto stat) {
            std::vector<double> v(B);
            for (int b = 0; b < B; ++b) v[b] = stat(draws[b]);
            std::sort(v.begin(), v.end());
            return std::pair<double, double>(v[B / 40], v[B - 1 - B / 40]);
        };
        auto mean_of = [&](size_t mi, double Sums::* f, const std::vector<int64_t>& d) {
            double a = 0; int64_t c = 0;
            for (auto r : d) { a += rs[mi][r].*f; c += rn[r]; }
            return a / std::max<int64_t>(1, c);
        };
        std::vector<int64_t> all(R);
        for (int64_t r = 0; r < R; ++r) all[r] = r;
        auto pct = [](std::vector<float> v, double p) { if (v.empty()) return 0.0f; std::sort(v.begin(), v.end()); return v[(size_t) (p * (v.size() - 1))]; };

        std::printf("\n%lld scored tokens in %lld %s. Against %s (BF16):\n", (long long) ntok, (long long) R, wiki ? "rows" : "conversations", ref.filename().c_str());
        std::printf("  %-44s %-26s %-24s %8s %8s %7s\n", "model", "KL(bf16||m) [95% CI]", "top-1 [95% CI]", "median", "p99 KL", "ppl");
        std::printf("  %-44s %-26s %-24s %8s %8s %7.3f\n", (ref.filename().string() + " (BF16)").c_str(), "-", "-", "-", "-", std::exp(mean_of(0, &Sums::nll, all)));
        for (size_t mi = 1; mi < NM; ++mi)
        {
            const auto [kl0, kl1] = ci([&](auto& d) { return mean_of(mi, &Sums::kl, d); });
            const auto [t0, t1] = ci([&](auto& d) { return mean_of(mi, &Sums::top1, d); });
            char a[64], b[64];
            std::snprintf(a, sizeof a, "%.4f [%.4f, %.4f]", mean_of(mi, &Sums::kl, all), kl0, kl1);
            std::snprintf(b, sizeof b, "%.2f%% [%.2f, %.2f]", 100 * mean_of(mi, &Sums::top1, all), 100 * t0, 100 * t1);
            std::printf("  %-44s %-26s %-24s %8.5f %8.4f %7.3f\n", tests[mi - 1].filename().c_str(), a, b, pct(tok_kl[mi], 0.5), pct(tok_kl[mi], 0.99),
                        std::exp(mean_of(mi, &Sums::nll, all)));
        }
        for (size_t mi = 2; mi < NM; ++mi)
        {
            auto dkl = [&](auto& d) { return mean_of(mi, &Sums::kl, d) - mean_of(1, &Sums::kl, d); };
            auto dt1 = [&](auto& d) { return 100 * (mean_of(mi, &Sums::top1, d) - mean_of(1, &Sums::top1, d)); };
            const auto [k0, k1] = ci(dkl);
            const auto [t0, t1] = ci(dt1);
            std::printf("  paired, %s minus %s: KL %+.4f [%+.4f, %+.4f], top-1 %+.2f points [%+.2f, %+.2f]\n", tests[mi - 1].filename().c_str(),
                        tests[0].filename().c_str(), dkl(all), k0, k1, dt1(all), t0, t1);
        }
        return 0;
    }
    catch (const std::exception& ex) { std::fprintf(stderr, "error: %s\n", ex.what()); return 1; }
}
