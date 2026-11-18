// Gemma 4 on the C++ engine, without the server (token ids in and out; no tokenizer needed):
//   g4_gen <model dir> gen <ids.txt> [max_new] [ctx]     greedy continuation; prints the new ids and the timings
//   g4_gen <model dir> score <ids.txt> <out.jsonl> [topk] teacher-forced: each position's top-k logprobs (one token at a
//                                                         time through forward(), the first G4_SCORE_PREFILL ids
//                                                         prefilled first; G4_SCORE_CTX: the context, default ids + 64)
//   g4_gen <model dir> bench [prompt] [decode] [ctx]      prompt tokens of a repeating id, then free-running decode
//   g4_gen <model dir> batch <games> <max_new> <ctx> <ids.txt>...  the prompts together on G4Engine (cycled over games): ids, tok/s
//   g4_gen <model dir> spec <ids.txt> <max_new> [ctx]     the Engine's loop with G4_DRAFTER (greedy): reply ids, acceptance
// G4_DRAFTER: the assistant model's folder; G4_DRAFT: drafts per step (default 2). G4_BENCH_Q: rows per bench step.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include "../src/engine.h"
#include "../src/g4.h"
#include "../src/g4_engine.h"

using namespace glm;

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static std::vector<int> read_ids(const std::string& path) {
    std::vector<int> ids;
    std::ifstream f(path);
    int x;
    while (f >> x) ids.push_back(x);
    return ids;
}

static std::string drafter() { return getenv("G4_DRAFTER") ? getenv("G4_DRAFTER") : ""; }

static std::vector<float> row(const half* lg) {
    std::vector<half> h(g4::V);
    Dev::get().q.memcpy(h.data(), lg, h.size() * 2).wait();
    return std::vector<float>(h.begin(), h.end());
}

int main(int argc, char** argv) {
    GLM_CHECK(argc >= 3, "usage: g4_gen <model dir> gen|score|bench ...");
    std::string dir = argv[1], mode = argv[2];
    double t0 = now();
    if (mode == "gen") {
        std::vector<int> ids = read_ids(argv[3]);
        int max_new = argc > 4 ? atoi(argv[4]) : 64;
        int ctx = argc > 5 ? atoi(argv[5]) : (int)ids.size() + max_new + 64;
        G4Model m(dir, ctx);
        printf("load %.1f s\n", now() - t0);
        double tp = now();
        if (ids.size() > 1) m.prefill(ids.data(), (int)ids.size() - 1);
        Dev::get().q.wait();
        double tp1 = now();
        int t = ids.back();
        double td = 0;
        std::string out;
        int made = 0;
        for (int i = 0; i < max_new; i++) {
            if (i == 1) td = now();
            auto lg = row(m.forward(&t, 1));
            int best = 0;
            for (int v = 1; v < g4::V; v++)
                if (lg[v] > lg[best]) best = v;
            if (i == 0) printf("prefill %.2f s (%zu tokens), first token %.2f s\n", tp1 - tp, ids.size() - 1, now() - tp);
            made++;
            out += std::to_string(best) + " ";
            t = best;
            if (best == 106 || best == 1) break;
        }
        double t1 = now();
        printf("ids: %s\n", out.c_str());
        if (g_prof_points) printf("%s", prof_points_report(made - 1).c_str());
        printf("%d tokens, decode %.2f tok/s\n", made, (made - 1) / std::max(1e-9, t1 - td));
        return 0;
    }
    if (mode == "batch") {
        int games = atoi(argv[3]), max_new = atoi(argv[4]), ctx = atoi(argv[5]);
        std::vector<std::vector<int>> prompts;
        for (int i = 6; i < argc; i++) prompts.push_back(read_ids(argv[i]));
        G4Model m(dir, ctx, drafter(), games);
        G4Engine e(m);
        std::vector<std::string> out(games);
        std::vector<GenResult> res(games);
        int tot = 0;
        double t1 = now();
        for (int g = 0; g < games; g++) {
            GenRequest r;
            r.prompt = prompts[g % prompts.size()];
            r.max_new = max_new;
            r.stop_ids = {1, 106, 50};
            r.sampling.temperature = getenv("G4_TEMP") ? (float)atof(getenv("G4_TEMP")) : 0.0f;
            r.on_token = [&out, &tot, g](int id) { out[g] += std::to_string(id) + " "; tot++; return true; };
            e.submit(std::move(r), [&res, g](const GenResult& x) { res[g] = x; });
        }
        double td = 0;
        int tot0 = 0;
        while (e.busy()) {
            e.step();
            if (!td && tot > 0) { td = now(); tot0 = tot; }
        }
        double t2 = now();
        for (int g = 0; g < games; g++) {
            printf("ids: %s\n", out[g].c_str());
            printf("game %d: %d tokens (%s), first token %.2f s, %.2f tok/s, drafted %d, accepted %d\n", g, res[g].completion_tokens,
                   res[g].finish.c_str(), res[g].prompt_s, res[g].decode_tok_s, res[g].drafted, res[g].accepted);
        }
        printf("total: %d tokens, %.2f tok/s decoding (%.1f s in all)\n", tot, (tot - tot0) / std::max(1e-9, t2 - td), t2 - t1);
        return 0;
    }
    if (mode == "lat") {     // g4_gen <dir> lat n...: seconds to read a prompt of n tokens (the second try of each)
        G4Model m(dir, 4096);
        for (int i = 3; i < argc; i++) {
            int n = atoi(argv[i]);
            std::vector<int> ids(n);
            for (int k = 0; k < n; k++) ids[k] = 1000 + (k * 7919) % 50000;
            double dt = 0;
            for (int rep = 0; rep < 2; rep++) {
                m.reset();
                double tp = now();
                m.prefill(ids.data(), n);
                Dev::get().q.wait();
                dt = now() - tp;
            }
            printf("%5d tokens: %6.0f ms (%.0f tok/s)\n", n, dt * 1000, n / dt);
        }
        return 0;
    }
    if (mode == "spec") {
        std::vector<int> ids = read_ids(argv[3]);
        int max_new = argc > 4 ? atoi(argv[4]) : 64;
        int ctx = argc > 5 ? atoi(argv[5]) : (int)ids.size() + max_new + 64;
        G4Model m(dir, ctx, drafter());
        Engine e(m);
        e.ckpt_every = 1;
        GenRequest r;
        r.prompt = ids;
        r.max_new = max_new;
        r.stop_ids = {1, 106, 50};
        r.sampling.temperature = getenv("G4_TEMP") ? (float)atof(getenv("G4_TEMP")) : 0.0f;
        std::string out;
        r.on_token = [&](int id) { out += std::to_string(id) + " "; return true; };
        GenResult g = e.run(r);
        printf("ids: %s\n", out.c_str());
        printf("%d tokens (%s), prompt %.2f s, decode %.2f tok/s; drafted %d, accepted %d (%.1f%%)\n", g.completion_tokens,
               g.finish.c_str(), g.prompt_s, g.decode_tok_s, g.drafted, g.accepted, 100.0 * g.accepted / std::max(1, g.drafted));
        return 0;
    }
    if (mode == "scoretrace") {      // g4_gen <dir> scoretrace <trace.json> <out.jsonl> [cols]: exl3eval --trace --dump's rows and format
        Json tj = Json::parse(read_file(argv[3]));
        int cols = argc > 5 ? atoi(argv[5]) : 2048;
        G4Model m(dir, cols + 64);
        FILE* fo = fopen(argv[4], "w");
        GLM_CHECK(fo, "cannot write %s", argv[4]);
        double ts = now();
        size_t tot = 0;
        for (auto& r : tj["rows"].arr) {
            std::vector<int> ids;
            for (auto& x : r["input_ids"].arr) ids.push_back((int)x.i());
            int n0 = (int)ids.size();
            for (auto& x : r["response_ids"].arr) ids.push_back((int)x.i());
            if ((int)ids.size() > cols) ids.resize(cols);
            if ((int)ids.size() - n0 < 32) continue;
            m.reset();
            if (n0 > 1) m.prefill(ids.data(), n0 - 1);
            for (int p = n0 - 1; p + 1 < (int)ids.size(); p++) {
                auto lg = row(m.forward(&ids[p], 1));
                double mx = -1e30;
                for (float v : lg) mx = std::max(mx, (double)v);
                double se = 0;
                for (float v : lg) se += std::exp(v - mx);
                double lse = mx + std::log(se);
                std::vector<int> idx(g4::V);
                std::iota(idx.begin(), idx.end(), 0);
                std::partial_sort(idx.begin(), idx.begin() + 64, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
                fprintf(fo, "{\"pos\": %d, \"next\": %d, \"next_lp\": %.6f, \"top\": [", p, ids[p + 1], lg[ids[p + 1]] - lse);
                for (int k = 0; k < 64; k++) fprintf(fo, "%s[%d, %.6f]", k ? ", " : "", idx[k], lg[idx[k]] - lse);
                fprintf(fo, "]}\n");
                tot++;
            }
        }
        fclose(fo);
        printf("scored %zu tokens in %.1f s\n", tot, now() - ts);
        return 0;
    }
    if (mode == "score") {
        std::vector<int> ids = read_ids(argv[3]);
        std::string outp = argv[4];
        int topk = argc > 5 ? atoi(argv[5]) : 64;
        int npre = getenv("G4_SCORE_PREFILL") ? atoi(getenv("G4_SCORE_PREFILL")) : 0;
        int ctx = getenv("G4_SCORE_CTX") ? atoi(getenv("G4_SCORE_CTX")) : (int)ids.size() + 64;
        G4Model m(dir, ctx);
        if (npre > 0) {
            double tp = now();
            m.prefill(ids.data(), npre);
            Dev::get().q.wait();
            printf("prefill %d tokens: %.2f s (%.0f tok/s)\n", npre, now() - tp, npre / (now() - tp));
            if (g_prof_points) { printf("%s", prof_points_report(npre).c_str()); return 0; }
        }
        FILE* fo = fopen(outp.c_str(), "w");
        GLM_CHECK(fo, "cannot write %s", outp.c_str());
        double ts = now();
        for (int p = npre; p < (int)ids.size(); p++) {
            auto lg = row(m.forward(&ids[p], 1));
            double mx = -1e30;
            for (float v : lg) mx = std::max(mx, (double)v);
            double se = 0;
            for (float v : lg) se += std::exp(v - mx);
            double lse = mx + std::log(se);
            std::vector<int> idx(g4::V);
            std::iota(idx.begin(), idx.end(), 0);
            std::partial_sort(idx.begin(), idx.begin() + topk, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
            fprintf(fo, "{\"pos\": %d, \"next\": %d, \"next_lp\": %.6f, \"top\": [", p,
                    p + 1 < (int)ids.size() ? ids[p + 1] : -1, p + 1 < (int)ids.size() ? lg[ids[p + 1]] - lse : 0.0);
            for (int k = 0; k < topk; k++) fprintf(fo, "%s[%d, %.6f]", k ? ", " : "", idx[k], lg[idx[k]] - lse);
            fprintf(fo, "]}\n");
        }
        fclose(fo);
        printf("scored %zu positions in %.1f s\n", ids.size() - npre, now() - ts);
        return 0;
    }
    if (mode == "bench") {
        int npre = argc > 3 ? atoi(argv[3]) : 2048;
        int ndec = argc > 4 ? atoi(argv[4]) : 64;
        int ctx = argc > 5 ? atoi(argv[5]) : npre + ndec + 64;
        G4Model m(dir, ctx, drafter());
        printf("load %.1f s, VRAM %.2f GiB\n", now() - t0, Dev::get().vram_used / 1073741824.0);
        std::vector<int> ids(npre);
        for (int i = 0; i < npre; i++) ids[i] = 1000 + (i * 7919) % 50000;
        double tp = now();
        if (npre) m.prefill(ids.data(), npre);
        Dev::get().q.wait();
        double tp1 = now();
        printf("prefill %d tokens: %.2f s (%.0f tok/s)\n", npre, tp1 - tp, npre / std::max(1e-9, tp1 - tp));
        if (g_prof_points) printf("%s", prof_points_report(npre).c_str());
        const int Q = getenv("G4_BENCH_Q") ? atoi(getenv("G4_BENCH_Q")) : 1;
        std::vector<int> t(Q, 1000);
        auto& q = Dev::get().q;
        std::vector<half> h(g4::V);
        double td = 0;
        double tdraft = 0;
        for (int i = 0; i < ndec; i++) {
            if (i == 8) { q.wait(); td = now(); tdraft = 0; if (g_prof_points) prof_points_report(-1); }
            if (m.draft_tokens && i > 0) {
                double t3 = now();
                std::vector<int> d(m.draft_tokens);
                m.draft(t[0], 0, m.draft_tokens, d.data());
                tdraft += now() - t3;
            }
            const half* lg = m.forward(t.data(), Q);
            q.memcpy(h.data(), lg, 2 * 16).wait();      // the kernels' end, without a full logits copy
            for (int r = 0; r < Q; r++) t[r] = 1000 + (i * 31 + r) % 50000;
            m.restore(m.position() - (Q - 1));          // one row kept: every step sits at the same kind of position
        }
        q.wait();
        double t1 = now();
        printf("decode: %.2f ms a step of %d rows (%.2f ms of it drafting), at %d tokens of context\n", (t1 - td) * 1000 / (ndec - 8), Q,
               tdraft * 1000 / (ndec - 8), m.position());
        if (g_prof_points) printf("%s", prof_points_report(ndec - 8).c_str());
        return 0;
    }
    GLM_CHECK(false, "mode %s", mode.c_str());
}
