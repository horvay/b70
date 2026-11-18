// Edge0-145B on the C++ engine, without the server:
//   e0_gen <model dir> gen <text | @file> [max_new] [ctx]    greedy continuation of raw text
//   e0_gen <model dir> chat <text | @file> [max_new] [ctx]   one user turn through Qwen's chat template (thinking on)
//   e0_gen <model dir> ppl <@file> [max_tokens]              perplexity of a text, 8 rows a forward() after a
//                                                            prefill of its first 64 tokens
//   e0_gen <model dir> score <ids.txt> <out.jsonl> [topk]    teacher-forced: each position's top-k logprobs
// E0_SCORE_PREFILL=n: prefill the first n ids in one call before scoring (the long-prompt path).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include "../src/e0.h"
#include "../src/tokenizer.h"

using namespace glm;

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static std::vector<float> rows_f(const half* lg, int n) {
    std::vector<half> h((size_t)n * q4::V);
    Dev::get().q.memcpy(h.data(), lg, h.size() * 2).wait();
    return std::vector<float>(h.begin(), h.end());
}

static std::string text_arg(const std::string& a) {
    if (a.empty() || a[0] != '@') return a;
    std::ifstream f(a.substr(1));
    GLM_CHECK(f, "cannot read %s", a.c_str() + 1);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static double logsumexp(const float* v, int n) {
    double mx = -1e30;
    for (int i = 0; i < n; i++) mx = std::max(mx, (double)v[i]);
    double se = 0;
    for (int i = 0; i < n; i++) se += std::exp(v[i] - mx);
    return mx + std::log(se);
}

int main(int argc, char** argv) {
    GLM_CHECK(argc >= 4, "usage: e0_gen <model dir> gen|chat <text|@file> [max_new] [ctx] | ppl <@file> [n] | "
                         "score <ids.txt> <out.jsonl> [topk]");
    std::string dir = argv[1], mode = argv[2];
    double t0 = now();
    if (mode == "gen" || mode == "chat") {
        std::string text = text_arg(argv[3]);
        if (mode == "chat") text = "<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n<think>\n";
        int max_new = argc > 4 ? atoi(argv[4]) : 64;
        int ctx = argc > 5 ? atoi(argv[5]) : 16384;
        Tokenizer tok = Tokenizer::load(dir);
        std::vector<int> ids = tok.encode(text, true);
        printf("prompt: %zu tokens\n", ids.size());
        E0Model m(dir, ctx, getenv("E0_RESERVE_GIB") ? atof(getenv("E0_RESERVE_GIB")) : 2.0);
        printf("load %.1f s\n", now() - t0);
        double tp = now();
        if (ids.size() > 1) m.prefill(ids.data(), (int)ids.size() - 1);
        Dev::get().q.wait();
        double tp1 = now();
        int t = ids.back();
        StreamDecoder sd(tok);
        double td = 0;
        int made = 0;
        if (m.draft_tokens > 0 && !getenv("E0_NO_DRAFT")) {
            // greedy with MTP drafts, as the server's Engine runs them: verify tok + drafts in one forward()
            std::vector<int> batch(1 + m.draft_tokens);
            int seed = -1, drafted = 0, accepted = 0, steps = 0;
            double t_draft = 0, t_fwd[16] = {};
            int n_fwd[16] = {};
            bool done = false;
            while (!done && made < max_new) {
                if (made == 1 && !td) { td = now(); prof_points_report(-1); m.expert_stats(); }
                batch[0] = t;
                int Q = 1;
                double ta = now();
                if (seed >= 0) Q += m.draft(t, seed, std::min(m.draft_tokens, max_new - made - 1), batch.data() + 1);
                double tb = now();
                auto lg = rows_f(m.forward(batch.data(), Q), Q);
                if (td) { t_draft += tb - ta; t_fwd[Q] += now() - tb; n_fwd[Q]++; }
                steps++;
                drafted += Q - 1;
                if (made == 0) printf("prefill %.2f s (%zu tokens), first token %.2f s\n", tp1 - tp, ids.size() - 1, now() - tp);
                for (int i = 0; i < Q; i++) {
                    const float* l = lg.data() + (size_t)i * q4::V;
                    int best = (int)(std::max_element(l, l + q4::V) - l);
                    if (best == 248046 || best == 248044) { done = true; break; }
                    made++;
                    printf("%s", sd.push(best).c_str());
                    fflush(stdout);
                    t = best;
                    seed = i;
                    if (i + 1 < Q && best == batch[i + 1]) { accepted++; continue; }
                    if (i + 1 < Q) m.restore(m.position() - (Q - i - 1));
                    break;
                }
            }
            double t1 = now();
            printf("\n---\n%d tokens in %d steps, decode %.2f tok/s, drafts %d accepted %d (%.0f%%), %zu expert swaps\n",
                   made, steps, (made - 1) / std::max(1e-9, t1 - td), drafted, accepted,
                   drafted ? 100.0 * accepted / drafted : 0.0, m.swaps());
            printf("drafting %.1f ms a step;", 1000 * t_draft / std::max(1, steps - 1));
            for (int qq = 1; qq < 16; qq++)
                if (n_fwd[qq]) printf(" forward(%d) %.1f ms x%d;", qq, 1000 * t_fwd[qq] / n_fwd[qq], n_fwd[qq]);
            printf("\n");
            if (g_prof_points) printf("%s", prof_points_report(made - 1).c_str());
            printf("%s", m.expert_stats().c_str());
            return 0;
        }
        for (int i = 0; i < max_new; i++) {
            if (i == 1) { td = now(); prof_points_report(-1); m.expert_stats(); }
            auto lg = rows_f(m.forward(&t, 1), 1);
            int best = (int)(std::max_element(lg.begin(), lg.end()) - lg.begin());
            if (i == 0) printf("prefill %.2f s (%zu tokens), first token %.2f s\n", tp1 - tp, ids.size() - 1, now() - tp);
            if (best == 248046 || best == 248044) break;
            made++;
            printf("%s", sd.push(best).c_str());
            fflush(stdout);
            t = best;
        }
        double t1 = now();
        printf("\n---\n%d tokens, decode %.2f tok/s, %zu expert swaps\n", made, (made - 1) / std::max(1e-9, t1 - td),
               m.swaps());
        if (g_prof_points) printf("%s", prof_points_report(max_new - 1).c_str());
        printf("%s", m.expert_stats().c_str());
        return 0;
    }
    if (mode == "ppl") {
        std::string text = text_arg(argv[3]);
        int maxn = argc > 4 ? atoi(argv[4]) : 2048;
        Tokenizer tok = Tokenizer::load(dir);
        std::vector<int> ids = tok.encode(text, true);
        if ((int)ids.size() > maxn) ids.resize(maxn);
        // E0_PPL_PREFILL: tokens prefilled in one call first (default 64); E0_PPL_FROM: also report the positions from
        // there on (the same window for a run that prefilled less)
        const int pre = getenv("E0_PPL_PREFILL") ? atoi(getenv("E0_PPL_PREFILL")) : std::min<int>(64, (int)ids.size() / 2);
        const int from = getenv("E0_PPL_FROM") ? atoi(getenv("E0_PPL_FROM")) : pre;
        double wnll = 0;
        int wn = 0;
        E0Model m(dir, (int)ids.size() + 64, getenv("E0_RESERVE_GIB") ? atof(getenv("E0_RESERVE_GIB")) : 2.0);
        double tp = now();
        m.prefill(ids.data(), pre);
        Dev::get().q.wait();
        if (g_prof_points) {          // GLM_PROF=5: where the prefill's time went, then stop
            printf("prefill %d tokens %.2f s\n%s", pre, now() - tp, prof_points_report(pre).c_str());
            return 0;
        }
        double nll = 0, ts = now();
        int n = 0, top1 = 0;
        for (int p = pre; p + 1 < (int)ids.size(); p += 8) {
            int Q = std::min(8, (int)ids.size() - 1 - p);
            auto lg = rows_f(m.forward(&ids[p], Q), Q);
            for (int r = 0; r < Q; r++) {
                const float* l = lg.data() + (size_t)r * q4::V;
                double lse = logsumexp(l, q4::V);
                int nxt = ids[p + r + 1];
                nll += lse - l[nxt];
                if (p + r >= from) { wnll += lse - l[nxt]; wn++; }
                top1 += (int)(std::max_element(l, l + q4::V) - l) == nxt;
                n++;
            }
        }
        printf("prefill %d tokens %.2f s; %d positions in %.1f s\n", pre, ts - tp, n, now() - ts);
        printf("ppl %.3f (nll %.4f), top-1 %.1f%%\n", std::exp(nll / n), nll / n, 100.0 * top1 / n);
        if (from != pre) printf("positions %d..: nll %.4f over %d\n", from, wnll / std::max(1, wn), wn);
        printf("prefill rate %.0f tok/s\n", pre / std::max(1e-9, ts - tp));
        printf("%s", m.expert_stats().c_str());
        return 0;
    }
    if (mode == "rollback") {
        // e0_gen <dir> rollback <text|@file>: a verify window of 4 rolled back to 2 rows, then one more token, against
        // the same tokens one at a time from a fresh start (the recurrent state's replay)
        std::string text = text_arg(argv[3]);
        Tokenizer tok = Tokenizer::load(dir);
        std::vector<int> ids = tok.encode(text, true);
        GLM_CHECK(ids.size() >= 12, "rollback: a longer text");
        const int n = (int)ids.size() - 6;
        const int* x = ids.data() + n;                // x[0..3]: the window, x[4]: the token after the rollback
        E0Model m(dir, (int)ids.size() + 64, getenv("E0_RESERVE_GIB") ? atof(getenv("E0_RESERVE_GIB")) : 2.0);
        auto run = [&](bool window) {
            m.reset();
            m.prefill(ids.data(), n - 1);
            m.forward(ids.data() + n - 1, 1);
            if (window) {
                m.forward(x, 4);
                GLM_CHECK(m.restore(m.position() - 2), "restore inside the window");
            } else {
                m.forward(x, 1);
                m.forward(x + 1, 1);
            }
            return rows_f(m.forward(x + 4, 1), 1);
        };
        const bool same = getenv("E0_RB_SAME") && atoi(getenv("E0_RB_SAME"));   // both runs without the window
        auto a = run(!same), b = run(false);
        double lsa = logsumexp(a.data(), q4::V), lsb = logsumexp(b.data(), q4::V), kl = 0, mx = 0;
        for (int v = 0; v < q4::V; v++) {
            double pa = std::exp(a[v] - lsa);
            kl += pa * ((a[v] - lsa) - (b[v] - lsb));
            mx = std::max(mx, (double)std::fabs(a[v] - b[v]));
        }
        int ta = (int)(std::max_element(a.begin(), a.end()) - a.begin()), tb = (int)(std::max_element(b.begin(), b.end()) - b.begin());
        printf("rollback vs sequential: KL %.6f, max |dlogit| %.4f, top %d vs %d\n", kl, mx, ta, tb);
        return 0;
    }
    if (mode == "score") {
        std::vector<int> ids;
        {
            std::ifstream f(argv[3]);
            int x;
            while (f >> x) ids.push_back(x);
        }
        GLM_CHECK(argc > 4, "score: output path");
        std::string outp = argv[4];
        int topk = argc > 5 ? atoi(argv[5]) : 64;
        int npre = getenv("E0_SCORE_PREFILL") ? atoi(getenv("E0_SCORE_PREFILL")) : 0;
        E0Model m(dir, (int)ids.size() + 64, getenv("E0_RESERVE_GIB") ? atof(getenv("E0_RESERVE_GIB")) : 2.0);
        if (npre > 0) {
            double tp = now();
            m.prefill(ids.data(), npre);
            Dev::get().q.wait();
            printf("prefill %d tokens: %.2f s (%.0f tok/s)\n", npre, now() - tp, npre / (now() - tp));
        }
        FILE* fo = fopen(outp.c_str(), "w");
        GLM_CHECK(fo, "cannot write %s", outp.c_str());
        double ts = now();
        for (int p = npre; p < (int)ids.size(); p++) {
            auto lg = rows_f(m.forward(&ids[p], 1), 1);
            double lse = logsumexp(lg.data(), q4::V);
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
    GLM_CHECK(false, "mode %s", mode.c_str());
}
