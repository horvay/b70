// Qwen3.8-27B on the C++ engine, without the server:
//   q35_gen <model dir> gen <text | @file> [max_new] [ctx]      greedy continuation of raw text
//   q35_gen <model dir> nll <@file> [ctx]                       teacher-forced mean NLL, as llama-perplexity scores it
//   q35_gen <model dir> bench <text | @file> [ctx] [iters]    after the prompt: a verify step of 1..4 rows and its
//       replay, and MTP draft chains of 1..3, timed (GLM_PROF=5: per stage)
//   q35_gen <model dir> spec <@prompts> [max_new] [ctx] [drafts] [temp] [seed]
//       MTP drafting through G4Engine (the server's path): each prompt (a file of prompts split by lines "=====",
//       raw text with chat tokens) is answered once per draft count in drafts ("0,1,2,3"; 0 = no drafts), then a
//       follow-up turn (prompt + reply + Q35_GEN_FOLLOW) that reuses the cached prefix. Every run's tokens are
//       compared with the first count's. Q35_GEN_PAR=N: N sequences, and the prompts answered at once with the last
//       count as well. temp 0: greedy; else seeded sampling (top_p 0.95).
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include "../src/g4_engine.h"
#include "../src/q35.h"
#include "../src/tokenizer.h"

using namespace glm;

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static std::vector<float> row(const half* lg) {
    std::vector<half> h(q35::V);
    Dev::get().q.memcpy(h.data(), lg, h.size() * 2).wait();
    return std::vector<float>(h.begin(), h.end());
}

static std::string read_arg(std::string t) {
    if (t[0] != '@') return t;
    std::ifstream f(t.substr(1));
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

int main(int argc, char** argv) {
    GLM_CHECK(argc >= 4, "usage: q35_gen <model dir> gen <text|@file> [max_new] [ctx] | nll <@file> [ctx]");
    std::string dir = argv[1], mode = argv[2];
    Tokenizer tok = Tokenizer::load(dir);
    std::vector<int> ids = tok.encode(read_arg(argv[3]), true);
    double t0 = now();
    if (mode == "gen") {
        int max_new = argc > 4 ? atoi(argv[4]) : 64;
        int ctx = argc > 5 ? atoi(argv[5]) : 16384;
        printf("prompt: %zu tokens\n", ids.size());
        Q35Model m(dir, ctx);
        printf("load %.1f s\n", now() - t0);
        double tp = now();
        if (ids.size() > 1) m.prefill(ids.data(), (int)ids.size() - 1);
        Dev::get().q.wait();
        double tp1 = now();
        int t = ids.back();
        StreamDecoder sd(tok);
        double td = 0;
        int made = 0;
        for (int i = 0; i < max_new; i++) {
            if (i == 1) { td = now(); if (g_prof_points) prof_points_report(-1); }
            auto lg = row(m.forward(&t, 1));
            int best = 0;
            for (int v = 1; v < q35::V; v++) if (lg[v] > lg[best]) best = v;
            t = best;
            made++;
            printf("%s", sd.push(t).c_str());
            fflush(stdout);
        }
        if (g_prof_points) printf("\n%s", prof_points_report(made - 1).c_str());
        printf("\nprompt %.2f s (%.0f tok/s), decode %.1f tok/s\n", tp1 - tp, (ids.size() - 1) / std::max(1e-3, tp1 - tp),
               (made - 1) / std::max(1e-3, now() - td));
        return 0;
    }
    if (mode == "nll") {
        // as llama-perplexity: chunks of ctx tokens, the first half read as a prompt, the second half scored
        int ctx = argc > 4 ? atoi(argv[4]) : 1024;
        Q35Model m(dir, ctx + 8);
        double nll = 0, tp = 0, td = 0;
        int n = 0, top1 = 0, np = 0;
        for (size_t c0 = 0; c0 + ctx <= ids.size(); c0 += ctx) {
            m.reset();
            double t = now();
            m.prefill(ids.data() + c0, ctx / 2);
            Dev::get().q.wait();
            tp += now() - t;
            np += ctx / 2;
            t = now();
            for (int i = ctx / 2; i + 1 < ctx; i++) {
                auto lg = row(m.forward(&ids[c0 + i], 1));
                double mx = lg[0];
                int best = 0;
                for (int v = 1; v < q35::V; v++) if (lg[v] > mx) { mx = lg[v]; best = v; }
                double s = 0;
                for (int v = 0; v < q35::V; v++) s += std::exp((double)lg[v] - mx);
                nll += -(lg[ids[c0 + i + 1]] - mx - std::log(s));
                top1 += best == ids[c0 + i + 1];
                n++;
            }
            td += now() - t;
        }
        printf("nll %.4f (PPL %.4f) over %d tokens, top-1 %.1f%%; prompt %.0f tok/s, decode %.1f tok/s\n", nll / n, std::exp(nll / n), n,
               100.0 * top1 / n, np / std::max(1e-3, tp), n / std::max(1e-3, td));
        return 0;
    }
    if (mode == "bench") {
        // the parts of an MTP step, timed: a verify step of Q rows (Q = 1 .. 4) and its replay, draft chains of k
        int ctx = argc > 4 ? atoi(argv[4]) : 16384;
        int iters = argc > 5 ? atoi(argv[5]) : 20;
        Q35Model m(dir, ctx);
        auto& q = Dev::get().q;
        if (ids.size() > 1) m.prefill(ids.data(), (int)ids.size() - 1);
        int t = ids.back();
        m.forward(&t, 1);
        q.wait();
        printf("context %d\n", m.position());
        for (int Q = 1; Q <= 4; Q++) {
            std::vector<int> sq(Q, 0), tt(Q, t);
            double tf = 0, tr = 0;
            for (int i = 0; i < iters + 2; i++) {
                double a = now();
                std::vector<half> h((size_t)Q * q35::V);
                q.memcpy(h.data(), m.forward_rows(sq.data(), tt.data(), Q), h.size() * 2).wait();
                double b = now();
                if (Q > 1) m.restore(m.position() - Q + 1);     // keep the first row: a replay of one row
                q.wait();
                double c = now();
                if (i >= 2) { tf += b - a; tr += c - b; }
            }
            printf("verify %d rows: %.2f ms, then restore %.2f ms\n", Q, 1e3 * tf / iters, 1e3 * tr / iters);
            if (g_prof_points) printf("%s\n", prof_points_report(iters + 2).c_str());
        }
        for (int k = 1; k <= 3; k++) {
            double td = 0;
            for (int i = 0; i < iters + 2; i++) {
                std::vector<std::vector<int>> d;
                int tk = t, nn = k, rw = 0, sq = 0;
                double a = now();
                m.draft_rows(&sq, &tk, &rw, &nn, 1, d);
                double b = now();
                if (i >= 2) td += b - a;
            }
            printf("draft chain of %d: %.2f ms\n", k, 1e3 * td / iters);
        }
        return 0;
    }
    if (mode == "spec") {
        int max_new = argc > 4 ? atoi(argv[4]) : 256;
        int ctx = argc > 5 ? atoi(argv[5]) : 16384;
        std::vector<int> ks;
        {
            std::stringstream ss(argc > 6 ? argv[6] : "0,2");
            std::string t;
            while (std::getline(ss, t, ',')) ks.push_back(atoi(t.c_str()));
        }
        float temp = argc > 7 ? (float)atof(argv[7]) : 0.0f;
        uint64_t seed = argc > 8 ? strtoull(argv[8], nullptr, 10) : 1234;
        int par = getenv("Q35_GEN_PAR") ? atoi(getenv("Q35_GEN_PAR")) : 1;
        std::string follow = getenv("Q35_GEN_FOLLOW") ? getenv("Q35_GEN_FOLLOW")
                                                      : "<|im_end|>\n<|im_start|>user\nI draw my blade and step closer.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        // prompts
        std::string all = read_arg(argv[3]);
        std::vector<std::string> texts;
        for (size_t a = 0;;) {
            size_t b = all.find("\n=====\n", a);
            texts.push_back(all.substr(a, b == std::string::npos ? std::string::npos : b - a));
            if (b == std::string::npos) break;
            a = b + 7;
        }
        Q35Model m(dir, ctx, par);
        printf("load %.1f s, %zu prompts, MTP %s\n", now() - t0, texts.size(), m.has_mtp() ? "loaded" : "off");
        G4Engine eng(m);
        std::vector<int> stops;
        for (const char* t : {"<|im_end|>", "<|endoftext|>"}) if (auto id = tok.id_of(t)) stops.push_back(*id);
        auto run = [&](const std::vector<std::vector<int>>& prompts, int k, bool fresh, std::vector<std::vector<int>>& outs,
                       std::vector<GenResult>& res) {
            m.draft_tokens = k;
            outs.assign(prompts.size(), {});
            res.assign(prompts.size(), {});
            int left = (int)prompts.size();
            for (size_t i = 0; i < prompts.size(); i++) {
                GenRequest r;
                r.prompt = prompts[i];
                r.max_new = max_new;
                r.fresh = fresh;
                r.stop_ids = stops;
                r.sampling.temperature = temp;
                if (temp > 0) {
                    r.sampling.top_p = 0.95f;
                    r.sampling.top_k = getenv("Q35_GEN_TOPK") ? atoi(getenv("Q35_GEN_TOPK")) : 0;
                    r.sampling.has_seed = true;
                    r.sampling.seed = seed + i;
                }
                r.on_token = [&outs, i](int id) { outs[i].push_back(id); return true; };
                while (!eng.has_free()) eng.step();
                eng.submit(std::move(r), [&res, &left, i](const GenResult& g) { res[i] = g; left--; });
                if (par == 1) while (left > (int)(prompts.size() - i - 1)) eng.step();
            }
            while (left > 0) eng.step();
        };
        auto diff = [](const std::vector<int>& a, const std::vector<int>& b) {
            size_t i = 0;
            while (i < a.size() && i < b.size() && a[i] == b[i]) i++;
            return i == a.size() && i == b.size() ? -1 : (int)i;
        };
        std::vector<std::vector<int>> prompts;
        for (auto& t : texts) prompts.push_back(tok.encode(t, true));
        std::vector<std::vector<int>> ref1, ref2;
        int bad = 0;
        for (size_t pi = 0; pi < prompts.size(); pi++) {
            for (size_t ki = 0; ki < ks.size(); ki++) {
                std::vector<std::vector<int>> o1, o2;
                std::vector<GenResult> r1, r2;
                long long mr0 = m.mtp_rows;
                // turn 1 read fresh for the first count, from the cache after that; turn 2 continues the reply
                run({prompts[pi]}, ks[ki], ki == 0, o1, r1);
                std::vector<int> p2 = prompts[pi];
                p2.insert(p2.end(), o1[0].begin(), o1[0].end());
                auto f = tok.encode(follow, true);
                p2.insert(p2.end(), f.begin(), f.end());
                run({p2}, ks[ki], false, o2, r2);
                if (ki == 0) { ref1.push_back(o1[0]); ref2.push_back(o2[0]); }
                int d1 = diff(o1[0], ref1[pi]), d2 = diff(o2[0], ref2[pi]);
                if (ki > 0 && (d1 >= 0 || d2 >= 0)) bad++;
                for (int t = 0; t < 2; t++) {
                    const GenResult& g = t ? r2[0] : r1[0];
                    printf("prompt %zu turn %d drafts %d: %d prompt (%d cached, %.0f tok/s), %d tokens, %.2f tok/s, accepted %d/%d (%.0f%%), "
                           "%.2f tokens/step%s\n",
                           pi, t + 1, ks[ki], g.prompt_tokens, g.cached_tokens,
                           (g.prompt_tokens - g.cached_tokens) / std::max(1e-3, g.prompt_s), g.completion_tokens, g.decode_tok_s, g.accepted,
                           g.drafted, g.drafted ? 100.0 * g.accepted / g.drafted : 0.0,
                           g.completion_tokens / std::max(1.0, (double)(g.completion_tokens - g.accepted)),
                           ki == 0 ? "" : ((t ? d2 : d1) < 0 ? ", same tokens" : (" DIFFERS at token " + std::to_string(t ? d2 : d1)).c_str()));
                }
                printf("  (MTP rows %lld)\n", m.mtp_rows - mr0);
                if (ki == 0 && pi == 0) printf("  reply: %s\n", tok.decode(o1[0]).substr(0, 600).c_str());
                fflush(stdout);
            }
        }
        if (par > 1) {
            std::vector<std::vector<int>> o;
            std::vector<GenResult> r;
            run(prompts, ks.back(), true, o, r);
            for (size_t pi = 0; pi < prompts.size(); pi++) {
                int d = diff(o[pi], ref1[pi]);
                if (d >= 0) bad++;
                printf("together, prompt %zu (game %d): %d tokens, %.2f tok/s, accepted %d/%d, %s\n", pi, r[pi].game,
                       r[pi].completion_tokens, r[pi].decode_tok_s, r[pi].accepted, r[pi].drafted,
                       d < 0 ? "same tokens" : ("DIFFERS at token " + std::to_string(d)).c_str());
            }
        }
        printf("%s\n", bad ? "MISMATCHES" : "all runs gave the first count's tokens");
        return bad ? 2 : 0;
    }
    GLM_CHECK(false, "unknown mode %s", mode.c_str());
    return 1;
}
