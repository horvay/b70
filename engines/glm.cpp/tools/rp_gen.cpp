// Ripple-65B on the C++ engine, without the server (token ids in and out; no tokenizer needed):
//   rp_gen <model dir> gen <ids.txt> [max_new] [ctx]        greedy continuation; prints the new ids and the timings
//   rp_gen <model dir> score <ids.txt> <out.jsonl> [topk]   teacher-forced: each position's top-k logprobs, one token at a time
//                                                           through forward(), the first RP_SCORE_PREFILL ids prefilled first
//   rp_gen <model dir> bench [prompt] [decode] [ctx]        prompt tokens, then free-running decode (RP_BENCH_Q: rows a step)
//   rp_gen <model dir> batch <games> <max_new> <ctx> <ids.txt>...  the prompts together on G4Engine: ids, tok/s
//   rp_gen <model dir> spec <ids.txt> <max_new> [ctx]       the Engine's loop with RP_DRAFTER (greedy): reply ids, acceptance
//   rp_gen <model dir> draft <ids.txt> [decoded]            the drafter's chain after all ids but the last (the last `decoded` of them
//                                                           one at a time through forward(), the rest as a prompt)
// RP_DRAFTER: the DFlash2 model's folder.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <numeric>
#include "../src/engine.h"
#include "../src/g4_engine.h"
#include "../src/rp.h"

using namespace glm;

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static std::vector<int> read_ids(const std::string& path) {
    std::vector<int> ids;
    std::ifstream f(path);
    int x;
    while (f >> x) ids.push_back(x);
    return ids;
}

static std::string drafter() { return getenv("RP_DRAFTER") ? getenv("RP_DRAFTER") : ""; }

static std::vector<float> row(const half* lg) {
    std::vector<half> h(rp::V);
    Dev::get().q.memcpy(h.data(), lg, h.size() * 2).wait();
    return std::vector<float>(h.begin(), h.end());
}

int main(int argc, char** argv) {
    GLM_CHECK(argc >= 3, "usage: rp_gen <model dir> gen|score|bench|batch ...");
    std::string dir = argv[1], mode = argv[2];
    double t0 = now();
    const std::vector<int> stops = {100265, 100257};
    if (mode == "gen") {
        std::vector<int> ids = read_ids(argv[3]);
        int max_new = argc > 4 ? atoi(argv[4]) : 64;
        int ctx = argc > 5 ? atoi(argv[5]) : (int)ids.size() + max_new + 64;
        RpModel m(dir, ctx);
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
            for (int v = 1; v < rp::V; v++)
                if (lg[v] > lg[best]) best = v;
            if (i == 0) printf("prefill %.2f s (%zu tokens), first token %.2f s\n", tp1 - tp, ids.size() - 1, now() - tp);
            made++;
            out += std::to_string(best) + " ";
            t = best;
            if (std::find(stops.begin(), stops.end(), best) != stops.end()) break;
        }
        double t1 = now();
        printf("ids: %s\n", out.c_str());
        if (g_prof_points) printf("%s", prof_points_report(made - 1).c_str());
        printf("%d tokens, decode %.2f tok/s\n", made, (made - 1) / std::max(1e-9, t1 - td));
        return 0;
    }
    if (mode == "batch") {
        int games = atoi(argv[3]), max_new = atoi(argv[4]), ctx = atoi(argv[5]), e_rows = 8;
        std::vector<std::vector<int>> prompts;
        for (int i = 6; i < argc; i++) prompts.push_back(read_ids(argv[i]));
        RpModel m(dir, ctx, drafter(), games);
        if (m.draft_tokens) e_rows = 16;
        G4Engine e(m);
        e.max_rows = e_rows;
        std::vector<std::string> out(games);
        std::vector<GenResult> res(games);
        int tot = 0;
        double t1 = now();
        for (int g = 0; g < games; g++) {
            GenRequest r;
            r.prompt = prompts[g % prompts.size()];
            r.max_new = max_new;
            r.stop_ids = stops;
            r.sampling.temperature = 0.0f;
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
            printf("game %d: %d tokens (%s), first token %.2f s, %.2f tok/s\n", g, res[g].completion_tokens, res[g].finish.c_str(), res[g].prompt_s, res[g].decode_tok_s);
        }
        printf("total: %d tokens, %.2f tok/s decoding (%.1f s in all)\n", tot, (tot - tot0) / std::max(1e-9, t2 - td), t2 - t1);
        return 0;
    }
    if (mode == "draft") {
        std::vector<int> ids = read_ids(argv[3]);
        int nd = argc > 4 ? atoi(argv[4]) : 0;
        RpModel m(dir, (int)ids.size() + 64, drafter());
        const int body = (int)ids.size() - 1;
        if (body - nd > 0) m.prefill(ids.data(), body - nd);
        for (int i = body - nd; i < body; i++) m.forward(&ids[i], 1);
        Dev::get().q.wait();
        if (getenv("RP_DUMP")) {
            for (int nl = 1; nl <= 5; nl++) {
                std::string dp = std::string(getenv("RP_DUMP")) + "/L" + std::to_string(nl);
                m.dump_draft(dp, ids.back(), nl);
            }
            return 0;
        }
        std::vector<int> o(rp::DBLOCK);
        double td = now();
        int made = m.draft(ids.back(), 0, rp::DBLOCK - 1, o.data());
        printf("chain:");
        for (int i = 0; i < made; i++) printf(" %d", o[i]);
        printf("\nfirst block %.1f ms\n", (now() - td) * 1000);
        td = now();
        made = m.draft(ids.back(), 0, rp::DBLOCK - 1, o.data());
        printf("next block %.1f ms\n", (now() - td) * 1000);
        return 0;
    }
    if (mode == "spec") {
        std::vector<int> ids = read_ids(argv[3]);
        int max_new = argc > 4 ? atoi(argv[4]) : 64;
        int ctx = argc > 5 ? atoi(argv[5]) : (int)ids.size() + max_new + 64;
        RpModel m(dir, ctx, drafter());
        Engine e(m);
        e.ckpt_every = 1;
        GenRequest r;
        r.prompt = ids;
        r.max_new = max_new;
        r.stop_ids = stops;
        r.sampling.temperature = 0.0f;
        std::string out;
        r.on_token = [&](int id) { out += std::to_string(id) + " "; return true; };
        GenResult g = e.run(r);
        printf("ids: %s\n", out.c_str());
        printf("%d tokens (%s), prompt %.2f s, decode %.2f tok/s; drafted %d, accepted %d (%.1f%%)\n", g.completion_tokens,
               g.finish.c_str(), g.prompt_s, g.decode_tok_s, g.drafted, g.accepted, 100.0 * g.accepted / std::max(1, g.drafted));
        return 0;
    }
    if (mode == "score") {
        std::vector<int> ids = read_ids(argv[3]);
        std::string outp = argv[4];
        int topk = argc > 5 ? atoi(argv[5]) : 64;
        int npre = getenv("RP_SCORE_PREFILL") ? atoi(getenv("RP_SCORE_PREFILL")) : 0;
        RpModel m(dir, (int)ids.size() + 64);
        if (npre > 0) {
            double tp = now();
            m.prefill(ids.data(), npre);
            Dev::get().q.wait();
            printf("prefill %d tokens: %.2f s\n", npre, now() - tp);
        }
        FILE* fo = fopen(outp.c_str(), "w");
        GLM_CHECK(fo, "cannot write %s", outp.c_str());
        for (int p = npre; p < (int)ids.size(); p++) {
            auto lg = row(m.forward(&ids[p], 1));
            double mx = -1e30;
            for (float v : lg) mx = std::max(mx, (double)v);
            double se = 0;
            for (float v : lg) se += std::exp(v - mx);
            double lse = mx + std::log(se);
            std::vector<int> idx(rp::V);
            std::iota(idx.begin(), idx.end(), 0);
            std::partial_sort(idx.begin(), idx.begin() + topk, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
            fprintf(fo, "{\"pos\": %d, \"top\": [", p);
            for (int k = 0; k < topk; k++) fprintf(fo, "%s[%d, %.6f]", k ? ", " : "", idx[k], lg[idx[k]] - lse);
            fprintf(fo, "]}\n");
        }
        fclose(fo);
        printf("scored %zu positions\n", ids.size() - npre);
        return 0;
    }
    if (mode == "bench") {
        int npre = argc > 3 ? atoi(argv[3]) : 2048;
        int ndec = argc > 4 ? atoi(argv[4]) : 64;
        int ctx = argc > 5 ? atoi(argv[5]) : npre + ndec + 64;
        RpModel m(dir, ctx, drafter());
        printf("load %.1f s, VRAM %.2f GiB, context %d\n", now() - t0, Dev::get().vram_used / 1073741824.0, m.max_context());
        std::vector<int> ids(npre);
        for (int i = 0; i < npre; i++) ids[i] = 1000 + (i * 7919) % 50000;
        double tp = now();
        if (npre) m.prefill(ids.data(), npre);
        Dev::get().q.wait();
        double tp1 = now();
        printf("prefill %d tokens: %.2f s (%.0f tok/s)\n", npre, tp1 - tp, npre / std::max(1e-9, tp1 - tp));
        if (g_prof_points) printf("%s", prof_points_report(npre).c_str());
        const int Q = getenv("RP_BENCH_Q") ? atoi(getenv("RP_BENCH_Q")) : 1;
        std::vector<int> t(Q, 1000);
        auto& q = Dev::get().q;
        std::vector<half> h(rp::V);
        double td = 0;
        for (int i = 0; i < ndec; i++) {
            if (i == 8) { q.wait(); td = now(); if (g_prof_points) prof_points_report(-1); }
            const half* lg = m.forward(t.data(), Q);
            q.memcpy(h.data(), lg, 2 * 16).wait();
            for (int r = 0; r < Q; r++) t[r] = 1000 + (i * 31 + r) % 50000;
            m.restore(m.position() - (Q - 1));
        }
        q.wait();
        double t1 = now();
        printf("decode: %.2f ms a step of %d rows, at %d tokens of context\n", (t1 - td) * 1000 / (ndec - 8), Q, m.position());
        if (g_prof_points) printf("%s", prof_points_report(ndec - 8).c_str());
        return 0;
    }
    GLM_CHECK(false, "mode %s", mode.c_str());
}
