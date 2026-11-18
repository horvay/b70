// Qwen3.8-Flash-Next on the C++ engine, without the server:
//   q4_gen <model dir> gen <text | @file> [max_new] [ctx]     greedy continuation of raw text (no chat template)
//   q4_gen <model dir> score <ids.txt> <out.jsonl> [topk]     teacher-forced: each position's top-k logprobs
//                                                             (one token at a time through forward(), and the
//                                                             prompt prefilled first when Q4_SCORE_PREFILL=n;
//                                                             Q4_SCORE_CTX: the context, default the ids + 64)
// Q4_HOT: expert ranking files for the VRAM slots (default flashnext/hot/gm-7500.json, gm-ram-4500.json).
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include "../src/q4.h"
#include "../src/tokenizer.h"

using namespace glm;

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static std::vector<float> row(const half* lg) {
    std::vector<half> h(q4::V);
    Dev::get().q.memcpy(h.data(), lg, h.size() * 2).wait();
    return std::vector<float>(h.begin(), h.end());
}

int main(int argc, char** argv) {
    GLM_CHECK(argc >= 4, "usage: q4_gen <model dir> gen <text|@file> [max_new] [ctx] | score <ids.txt> <out.jsonl> [topk]");
    std::string dir = argv[1], mode = argv[2];
    std::string hot = getenv("Q4_HOT") ? getenv("Q4_HOT")
                                       : std::string(getenv("HOME")) + "/b70/llm/flashnext/hot/gm-7500.json," +
                                             getenv("HOME") + "/b70/llm/flashnext/hot/gm-ram-4500.json";
    double t0 = now();
    if (mode == "tok") {
        std::ifstream f(argv[3]);
        std::stringstream ss;
        ss << f.rdbuf();
        Tokenizer tok = Tokenizer::load(dir);
        for (int id : tok.encode(ss.str(), true)) printf("%d ", id);
        printf("\n");
        return 0;
    }
    if (mode == "gen") {
        std::string text = argv[3];
        if (text[0] == '@') {
            std::ifstream f(text.substr(1));
            std::stringstream ss;
            ss << f.rdbuf();
            text = ss.str();
        }
        int max_new = argc > 4 ? atoi(argv[4]) : 64;
        int ctx = argc > 5 ? atoi(argv[5]) : 16384;
        Tokenizer tok = Tokenizer::load(dir);
        std::vector<int> ids = tok.encode(text, true);
        printf("prompt: %zu tokens\n", ids.size());
        Q4Model m(dir, ctx, 2.0, hot);
        printf("load %.1f s\n", now() - t0);
        double tp = now();
        if (ids.size() > 1) m.prefill(ids.data(), (int)ids.size() - 1);
        Dev::get().q.wait();
        double tp1 = now();
        int t = ids.back();
        StreamDecoder sd(tok);
        double td = 0;
        std::string out;
        int made = 0;
        for (int i = 0; i < max_new; i++) {
            if (i == 1) { td = now(); prof_points_report(-1); m.expert_stats(); }
            auto lg = row(m.forward(&t, 1));
            int best = 0;
            for (int v = 1; v < q4::V; v++)
                if (lg[v] > lg[best]) best = v;
            if (i == 0) printf("prefill %.2f s (%zu tokens), first token %.2f s\n", tp1 - tp, ids.size() - 1, now() - tp);
            if (best == 248046 || best == 248044) break;
            made++;
            std::string s = sd.push(best);
            out += s;
            printf("%s", s.c_str());
            fflush(stdout);
            t = best;
        }
        double t1 = now();
        printf("\n---\n%d tokens, decode %.2f tok/s, %zu expert swaps\n", made, (made - 1) / std::max(1e-9, t1 - td), m.swaps());
        if (g_prof_points) printf("%s", prof_points_report(max_new - 1).c_str());
        printf("%s", m.expert_stats().c_str());
        return 0;
    }
    if (mode == "score") {
        std::vector<int> ids;
        {
            std::ifstream f(argv[3]);
            int x;
            while (f >> x) ids.push_back(x);
        }
        std::string outp = argv[4];
        int topk = argc > 5 ? atoi(argv[5]) : 64;
        int npre = getenv("Q4_SCORE_PREFILL") ? atoi(getenv("Q4_SCORE_PREFILL")) : 0;
        int ctx = getenv("Q4_SCORE_CTX") ? atoi(getenv("Q4_SCORE_CTX")) : (int)ids.size() + 64;
    Q4Model m(dir, ctx, 2.0, hot);
        if (npre > 0) {
            double tp = now();
            int step = getenv("Q4_SCORE_STEP") ? atoi(getenv("Q4_SCORE_STEP")) : npre;      // prefill in calls of this
            for (int p = 0; p < npre; p += step) m.prefill(ids.data() + p, std::min(step, npre - p));
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
            std::vector<int> idx(q4::V);
            for (int v = 0; v < q4::V; v++) idx[v] = v;
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
    if (mode == "pfstep") {          // q4_gen <dir> pfstep <ids.txt> <step> [ctx]: prefill step by step, decode 4 after each
        std::vector<int> ids;
        {
            std::ifstream f(argv[3]);
            int x;
            while (f >> x) ids.push_back(x);
        }
        int step = argc > 4 ? atoi(argv[4]) : 2048;
        int ctx = argc > 5 ? atoi(argv[5]) : (int)ids.size() + 64;
        Q4Model m(dir, ctx, 2.0, hot);
        auto& q = Dev::get().q;
        for (int p = 0; p + step + 4 <= (int)ids.size(); p += step + 4) {
            double tp = now();
            m.prefill(ids.data() + p, step);
            q.wait();
            double tf = now();
            for (int k = 0; k < 4; k++) row(m.forward(&ids[p + step + k], 1));
            printf("pos %d ok: prefill %.2f s, 4 decode %.3f s\n", m.position(), tf - tp, now() - tf);
            fflush(stdout);
        }
        return 0;
    }
    GLM_CHECK(false, "mode %s", mode.c_str());
}
