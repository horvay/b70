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
//   q35_gen <model dir> prefill <text | @file> <lengths> [ctx] [iters]     prompt reading speed (GLM_PROF=5: by stage)
//   q35_gen <model dir> kl <text | @file> <lengths> [scored] [ctx] [parts] the old prompt kernels against the new
//   q35_gen <model dir> gdncheck x <rows "1,63,64,700,2048">   the chunked delta rule against gdn_delta, random inputs
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include "../src/g4_engine.h"
#include "../src/q35.h"
#include "../src/q35_ops.h"
#include "../src/q4_ops.h"
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
        printf("after the prompt: %.2f GiB VRAM\n", Dev::get().vram_used / 1073741824.0);
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
        // as llama-perplexity: chunks of ctx tokens, the first half read as a prompt, the second half scored (or the
        // last `scored` tokens of each chunk: q35_gen nll <@file> [ctx] [scored] [chunks]); Q35_GEN_DUMP=file: each
        // scored position's target, its logprob, and the top 64 (id, logprob) as a line, for comparing two builds
        int ctx = argc > 4 ? atoi(argv[4]) : 1024;
        const int scored = argc > 5 ? atoi(argv[5]) : ctx / 2;
        const int chunks = argc > 6 ? atoi(argv[6]) : 1 << 30;
        Q35Model m(dir, ctx + 8);
        FILE* dump_f = getenv("Q35_GEN_DUMP") ? fopen(getenv("Q35_GEN_DUMP"), "w") : nullptr;
        double nll = 0, tp = 0, td = 0;
        int n = 0, top1 = 0, np = 0, nc = 0;
        for (size_t c0 = 0; c0 + ctx <= ids.size() && nc < chunks; c0 += ctx, nc++) {
            m.reset();
            double t = now();
            m.prefill(ids.data() + c0, ctx - scored);
            Dev::get().q.wait();
            tp += now() - t;
            np += ctx - scored;
            t = now();
            for (int i = ctx - scored; i + 1 < ctx; i++) {
                auto lg = row(m.forward(&ids[c0 + i], 1));
                double mx = lg[0];
                int best = 0;
                for (int v = 1; v < q35::V; v++) if (lg[v] > mx) { mx = lg[v]; best = v; }
                double s = 0;
                for (int v = 0; v < q35::V; v++) s += std::exp((double)lg[v] - mx);
                const double lse = mx + std::log(s);
                nll += -(lg[ids[c0 + i + 1]] - lse);
                top1 += best == ids[c0 + i + 1];
                n++;
                if (dump_f) {
                    std::vector<int> idx(q35::V);
                    for (int v = 0; v < q35::V; v++) idx[v] = v;
                    std::partial_sort(idx.begin(), idx.begin() + 64, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
                    fprintf(dump_f, "%d %.6f", ids[c0 + i + 1], lg[ids[c0 + i + 1]] - lse);
                    for (int k = 0; k < 64; k++) fprintf(dump_f, " %d %.6f", idx[k], lg[idx[k]] - lse);
                    fprintf(dump_f, "\n");
                }
            }
            td += now() - t;
        }
        if (dump_f) fclose(dump_f);
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
        const int ck = (int)ids.size() - 1;                     // a checkpoint at the prompt's end (the replay check)
        if (ids.size() > 1) m.prefill(ids.data(), ck, ck);
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
        // the fused replay against the per-layer one: the same step, rolled back to 1 .. Q - 1 rows, state hashes
        for (int Q = 2; Q <= 4; Q++)
            for (int keep = 1; keep < Q; keep++) {
                std::vector<int> sq(Q, 0), tt(Q, t);
                uint64_t h[2];
                for (int mode = 1; mode >= 0; mode--) {
                    m.replay_mode = mode;
                    GLM_CHECK(m.restore(ck) && m.position() == ck, "bench: no checkpoint at %d", ck);
                    m.forward_rows(sq.data(), tt.data(), Q);
                    m.restore(ck + keep);
                    h[mode] = m.state_hash();
                }
                m.replay_mode = 1;
                printf("replay %d of %d rows: fused %016llx, per layer %016llx: %s\n", keep, Q, (unsigned long long)h[1],
                       (unsigned long long)h[0], h[0] == h[1] ? "same" : "DIFFERENT");
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
        printf("sampling: %lld rows from GPU pools (%lld with the GPU's top-p rest), %lld whole rows; check %lld rows, %lld differ\n",
               Sampler::n_pool, Sampler::n_rest, Sampler::n_full, eng.check_rows_, eng.check_diff_);
        printf("%s\n", bad ? "MISMATCHES" : "all runs gave the first count's tokens");
        return bad ? 2 : 0;
    }
    if (mode == "mergecheck") {
        // q35_gen <dir> mergecheck x: real linears of layers 0 and 3 merged (Exl3Linear::merge) against the separate ones on
        // random inputs, rows 1, 3 (decode kernels) and 300 (the prompt path): relative error of each part's output
        SafeTensors st(dir);
        auto& q = Dev::get().q;
        const std::string P = "model.language_model.layers.";
        exl3_same_split = true;
        exl3_vec_max = 0;
        bool ok = true;
        std::vector<std::vector<std::string>> sets;
        const bool all = getenv("Q35_GEN_ALL") != nullptr;    // every layer's GDN and attention merges
        if (!all) sets = {{P + "0.mlp.gate_proj", P + "0.mlp.up_proj"},
                          {P + "0.linear_attn.in_proj_qkv", P + "0.linear_attn.in_proj_z"},
                          {P + "3.self_attn.q_proj", P + "3.self_attn.k_proj", P + "3.self_attn.v_proj"}};
        for (int li = 0; all && st.has(P + std::to_string(li) + ".mlp.gate_proj.trellis"); li++) {
            const std::string lp = P + std::to_string(li) + ".";
            if (st.has(lp + "linear_attn.in_proj_qkv.trellis")) sets.push_back({lp + "linear_attn.in_proj_qkv", lp + "linear_attn.in_proj_z"});
            else sets.push_back({lp + "self_attn.q_proj", lp + "self_attn.k_proj", lp + "self_attn.v_proj"});
        }
        if (all) sets.push_back({"mtp.layers.0.self_attn.q_proj", "mtp.layers.0.self_attn.k_proj", "mtp.layers.0.self_attn.v_proj"});
        for (auto names : sets) {
            std::vector<Exl3Linear> sep, parts;
            for (auto& n : names) { sep.push_back(Exl3Linear::load(st, n)); parts.push_back(Exl3Linear::load(st, n)); }
            std::vector<Exl3Linear*> pp;
            for (auto& p : parts) pp.push_back(&p);
            if (!Exl3Linear::can_merge(std::vector<const Exl3Linear*>(pp.begin(), pp.end()))) { printf("%s: cannot merge\n", names[0].c_str()); continue; }
            printf("%s: K", names[0].c_str());
            for (auto& l : sep) printf(" %d", l.K);
            printf("\n");
            Exl3Linear m = Exl3Linear::merge(pp);
            const int in = m.in;
            Workspace ws;
            size_t wb = m.ws_bytes(300);
            for (auto& l : sep) wb = std::max(wb, l.ws_bytes(300));
            ws.init(wb + (1 << 20));
            for (int M : {1, 3, 8, 300})
            for (DType dt : {DType::F16, DType::F32}) {
                const size_t es = dt == DType::F16 ? 2 : 4;
                std::vector<half> hx((size_t)M * in);
                uint64_t r = 99;
                for (auto& v : hx) { r ^= r << 13; r ^= r >> 7; r ^= r << 17; v = half((float)((r >> 11) % 2000) / 1000.0f - 1.0f); }
                half* x = sycl::malloc_device<half>(hx.size(), q);
                q.memcpy(x, hx.data(), hx.size() * 2).wait();
                std::vector<void*> ya, yb;
                for (auto& l : sep) { ya.push_back(sycl::malloc_device((size_t)M * l.out * es, q)); yb.push_back(sycl::malloc_device((size_t)M * l.out * es, q)); }
                for (size_t i = 0; i < sep.size(); i++) sep[i].forward(q, ws, x, M, ya[i], dt);
                m.forward_split(q, ws, x, M, yb.data(), dt);
                q.wait();
                printf("%s + %zu, %d rows, %s:", names[0].c_str(), names.size() - 1, M, dt == DType::F16 ? "fp16" : "fp32");
                for (size_t i = 0; i < sep.size(); i++) {
                    std::vector<float> a((size_t)M * sep[i].out), b(a.size());
                    if (dt == DType::F32) {
                        q.memcpy(a.data(), ya[i], a.size() * 4);
                        q.memcpy(b.data(), yb[i], b.size() * 4).wait();
                    } else {
                        std::vector<half> ha(a.size()), hb(a.size());
                        q.memcpy(ha.data(), ya[i], ha.size() * 2);
                        q.memcpy(hb.data(), yb[i], hb.size() * 2).wait();
                        for (size_t k = 0; k < a.size(); k++) { a[k] = ha[k]; b[k] = hb[k]; }
                    }
                    double num = 0, den = 0;
                    for (size_t k = 0; k < a.size(); k++) {
                        double d = (double)(float)a[k] - (double)(float)b[k];
                        num += d * d;
                        den += (double)(float)a[k] * (double)(float)a[k];
                    }
                    const double re = std::sqrt(num / std::max(den, 1e-30));
                    ok = ok && re < 2e-3 && std::isfinite(re);
                    printf(" part %zu rel %.2e", i, re);
                    sycl::free(ya[i], q);
                    sycl::free(yb[i], q);
                }
                printf("\n");
                sycl::free(x, q);
            }
        }
        printf("%s\n", ok ? "mergecheck ok" : "mergecheck FAILED");
        return ok ? 0 : 2;
    }
    if (mode == "linbench") {
        // q35_gen <dir> linbench x [iters]: each kind of linear at decode rows M = 1..4, the DPAS kernel (MTP's) and the
        // vector GEMV: time a call and GB/s of trellis read
        const int iters = argc > 4 ? atoi(argv[4]) : 200;
        SafeTensors st(dir);
        auto& q = Dev::get().q;
        const std::string P = "model.language_model.layers.";
        exl3_same_split = true;
        for (std::string name : {P + "0.linear_attn.in_proj_qkv", P + "0.linear_attn.in_proj_z", P + "0.linear_attn.out_proj",
                                 P + "0.mlp.gate_proj", P + "0.mlp.down_proj", P + "3.self_attn.q_proj", P + "3.self_attn.k_proj",
                                 P + "3.self_attn.o_proj", std::string("mtp.fc"), std::string("lm_head")}) {
            if (!Exl3Linear::present(st, name)) { printf("%-46s absent\n", name.c_str()); continue; }
            Exl3Linear L = Exl3Linear::load(st, name);
            Workspace ws;
            ws.init(L.ws_bytes(8) + (1 << 20));
            half* x = sycl::malloc_device<half>((size_t)8 * L.in, q);
            half* y = sycl::malloc_device<half>((size_t)8 * L.out, q);
            q.memset(x, 0, (size_t)8 * L.in * 2).wait();
            printf("%-46s %5d -> %6d K=%d %6.1f MB:", name.c_str(), L.in, L.out, L.K, L.trellis.bytes() / 1048576.0);
            for (int vm : {0, 8})
                for (int M = 1; M <= 4; M++) {
                    exl3_vec_max = vm;
                    for (int i = 0; i < 10; i++) L.forward(q, ws, x, M, y, DType::F16);
                    q.wait();
                    double a = now();
                    for (int i = 0; i < iters; i++) L.forward(q, ws, x, M, y, DType::F16);
                    q.wait();
                    double t = (now() - a) / iters;
                    printf(" %s%d %.0fus %.0f", M == 1 ? (vm ? "| vec " : "dpas ") : "", M, t * 1e6, L.trellis.bytes() / t * 1e-9);
                }
            printf("\n");
            fflush(stdout);
            sycl::free(x, q);
            sycl::free(y, q);
        }
        return 0;
    }
    if (mode == "many") {
        // q35_gen <dir> many <@prompts> <out ids file> [max_new] [ctx] [games] [temp] [seed]: every prompt (split by
        // "=====" lines) answered through G4Engine with `games` sequences at once (MTP as loaded), sampled at temp
        // (top_p 0.95); each reply's token ids written as one line of the out file
        GLM_CHECK(argc >= 5, "many: <@prompts> <out file>");
        const std::string outp = argv[4];
        int max_new = argc > 5 ? atoi(argv[5]) : 400;
        int ctx = argc > 6 ? atoi(argv[6]) : 4096;
        int games = argc > 7 ? atoi(argv[7]) : 8;
        float temp = argc > 8 ? (float)atof(argv[8]) : 0.8f;
        uint64_t seed = argc > 9 ? strtoull(argv[9], nullptr, 10) : 1;
        std::string all = read_arg(argv[3]);
        std::vector<std::vector<int>> prompts;
        for (size_t a = 0;;) {
            size_t b = all.find("\n=====\n", a);
            prompts.push_back(tok.encode(all.substr(a, b == std::string::npos ? std::string::npos : b - a), true));
            if (b == std::string::npos) break;
            a = b + 7;
        }
        Q35Model m(dir, ctx, games);
        G4Engine eng(m);
        std::vector<int> stops;
        for (const char* t : {"<|im_end|>", "<|endoftext|>"}) if (auto id = tok.id_of(t)) stops.push_back(*id);
        std::vector<std::vector<int>> outs(prompts.size());
        long long made = 0, acc = 0, dr = 0;
        double ta = now();
        size_t next = 0;
        int left = (int)prompts.size();
        while (left > 0) {
            while (next < prompts.size() && eng.has_free()) {
                GenRequest r;
                r.prompt = prompts[next];
                r.max_new = max_new;
                r.fresh = true;
                r.stop_ids = stops;
                r.sampling.temperature = temp;
                r.sampling.top_p = 0.95f;
                r.sampling.has_seed = true;
                r.sampling.seed = seed + next;
                const size_t i = next;
                r.on_token = [&outs, i](int id) { outs[i].push_back(id); return true; };
                eng.submit(std::move(r), [&, i](const GenResult& g) {
                    left--;
                    made += g.completion_tokens;
                    acc += g.accepted;
                    dr += g.drafted;
                    if (g.stop_id >= 0) outs[i].push_back(g.stop_id);
                });
                next++;
            }
            eng.step();
        }
        FILE* fo = fopen(outp.c_str(), "w");
        for (auto& o : outs) {
            for (size_t k = 0; k < o.size(); k++) fprintf(fo, k ? " %d" : "%d", o[k]);
            fprintf(fo, "\n");
        }
        fclose(fo);
        printf("many: %zu replies, %lld tokens in %.1f s (%.1f tok/s in total), drafts accepted %lld/%lld (%.0f%%)\n", prompts.size(),
               made, now() - ta, made / std::max(1e-3, now() - ta), acc, dr, dr ? 100.0 * acc / dr : 0.0);
        return 0;
    }
    if (mode == "fabench") {
        // q35_gen <dir> fabench x [p0] [rows] [variants "1,5,13,15"] [dbg "1,2,4,8,16"]: prompt attention alone on random
        // data, rows at positions p0 .. p0 + rows - 1 of one sequence (variants: attn_prompt's; dbg: attn_bench's)
        const int p0 = argc > 4 ? atoi(argv[4]) : 28672, rows = argc > 5 ? atoi(argv[5]) : 2048, T = p0 + rows;
        auto list = [](const char* a) {
            std::vector<int> v;
            std::stringstream ss(a);
            std::string t;
            while (std::getline(ss, t, ',')) v.push_back(atoi(t.c_str()));
            return v;
        };
        std::vector<int> vars = list(argc > 6 ? argv[6] : "1,5,13,15"), dbgs = list(argc > 7 ? argv[7] : "");
        auto& q = Dev::get().q;
        std::vector<half> hq((size_t)rows * q35::AH * q35::AD), hk((size_t)T * q35::AKV * q35::AD);
        uint64_t st = 1234567;
        auto rnd = [&]() { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return (float)((st >> 11) % 2000) / 1000.0f - 1.0f; };
        for (auto& x : hq) x = half(rnd() * 2.0f);
        for (auto& x : hk) x = half(rnd());
        Tensor dq = Tensor::empty({(int64_t)hq.size()}, DType::F16), dk = Tensor::empty({(int64_t)hk.size()}, DType::F16),
               dv = Tensor::empty({(int64_t)hk.size()}, DType::F16), dout = Tensor::empty({(int64_t)hq.size()}, DType::F16);
        q.memcpy(dq.p, hq.data(), hq.size() * 2);
        q.memcpy(dk.p, hk.data(), hk.size() * 2);
        q.memcpy(dv.p, hk.data(), hk.size() * 2).wait();
        double flop = 0;
        for (int r = 0; r < rows; r++) flop += 4.0 * q35::AH * q35::AD * (p0 + r + 1);
        auto timeit = [&](auto fn) {
            fn();
            q.wait();
            std::vector<double> ts;
            for (int i = 0; i < 5; i++) {
                double a = now();
                fn();
                q.wait();
                ts.push_back(now() - a);
            }
            std::sort(ts.begin(), ts.end());
            return ts[2];
        };
        const int TS = (T + 15) / 16 * 16;
        Tensor dkt = Tensor::empty({(int64_t)TS * q35::AKV * q35::AD}, DType::F16),
               dvt = Tensor::empty({(int64_t)TS * q35::AKV * q35::AD}, DType::F16);
        q35ops::kt_fill(q, dk.ptr<half>(), dkt.ptr<half>(), TS, 0, T);
        q35ops::vt_fill(q, dv.ptr<half>(), dvt.ptr<half>(), TS, 0, T);
        q.wait();
        std::vector<half> ref(hq.size()), got(hq.size());
        q35ops::attn_prompt(q, dq.ptr<half>(), dk.ptr<half>(), dv.ptr<half>(), rows, p0, dout.ptr<half>(), 1);
        q.memcpy(ref.data(), dout.p, ref.size() * 2).wait();
        for (int v : vars) {
            double t = timeit([&] { q35ops::attn_prompt(q, dq.ptr<half>(), dk.ptr<half>(), dv.ptr<half>(), rows, p0, dout.ptr<half>(), v,
                                                        dkt.ptr<half>(), TS, dvt.ptr<half>()); });
            q.memcpy(got.data(), dout.p, got.size() * 2).wait();
            double num = 0, den = 0;
            for (size_t i = 0; i < got.size(); i++) {
                double d = (double)(float)got[i] - (double)(float)ref[i];
                num += d * d;
                den += (double)(float)ref[i] * (double)(float)ref[i];
            }
            printf("variant %d: %.2f ms, %.1f TFLOPS; output vs variant 1: rel %.2e\n", v, t * 1e3, flop / t * 1e-12,
                   std::sqrt(num / std::max(den, 1e-30)));
        }
        for (int d : dbgs) {
            double t = timeit([&] { q35ops::attn_bench(q, dq.ptr<half>(), dk.ptr<half>(), dv.ptr<half>(), rows, p0, dout.ptr<half>(), d); });
            printf("q35_fa dbg %d: %.2f ms (%.1f TFLOPS-equivalent)\n", d, t * 1e3, flop / t * 1e-12);
        }
        return 0;
    }
    if (mode == "decbench") {
        // q35_gen <dir> decbench x [contexts "241,7800,30600"] [ns "32,64,128"]: decode attention alone on random data,
        // a verify step of R = 1..4 rows (positions T - R .. T - 1 of one sequence): the plain kernel (attn_core) against
        // the ESIMD one (attn_dec + combine) for each piece count: time, error, and whether each row of the R-row step is
        // bit-identical to a one-row step at its position (MTP's exactness)
        auto list = [](const char* a) {
            std::vector<int> v;
            std::stringstream ss(a);
            std::string t;
            while (std::getline(ss, t, ',')) v.push_back(atoi(t.c_str()));
            return v;
        };
        std::vector<int> Ts = list(argc > 4 ? argv[4] : "241,7800,30600"), nss = list(argc > 5 ? argv[5] : "32,64,128");
        auto& q = Dev::get().q;
        using namespace q35;
        uint64_t st = 1234567;
        auto rnd = [&]() { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return (float)((st >> 11) % 2000) / 1000.0f - 1.0f; };
        bool exact_all = true;
        for (int T : Ts) {
            const int TS = T + 16;
            std::vector<half> hq((size_t)4 * AH * AD), hg((size_t)4 * AH * 2 * AD), hk((size_t)TS * AKV * AD), hv(hk.size());
            for (auto& x : hq) x = half(rnd() * 3.0f);
            for (auto& x : hg) x = half(rnd() * 2.0f);
            for (auto& x : hk) x = half(rnd());
            for (auto& x : hv) x = half(rnd());
            Tensor dq = Tensor::empty({(int64_t)hq.size()}, DType::F16), dg = Tensor::empty({(int64_t)hg.size()}, DType::F16),
                   dk = Tensor::empty({(int64_t)hk.size()}, DType::F16), dv = Tensor::empty({(int64_t)hv.size()}, DType::F16),
                   o1 = Tensor::empty({(int64_t)hq.size()}, DType::F16), o2 = Tensor::empty({(int64_t)hq.size()}, DType::F16),
                   pd = Tensor::empty({8}, DType::I32), sd = Tensor::zeros({8}, DType::I32);
            size_t pf = q35ops::attn_part_floats(4, TS / CH + 1);
            for (int ns : nss) pf = std::max(pf, q35ops::dec_part_floats(4, ns));
            Tensor part = Tensor::empty({(int64_t)pf}, DType::F32);
            q.memcpy(dq.p, hq.data(), hq.size() * 2);
            q.memcpy(dg.p, hg.data(), hg.size() * 2);
            q.memcpy(dk.p, hk.data(), hk.size() * 2);
            q.memcpy(dv.p, hv.data(), hv.size() * 2).wait();
            auto timeit = [&](auto fn) {
                fn();
                q.wait();
                std::vector<double> ts;
                for (int i = 0; i < 7; i++) {
                    double a = now();
                    for (int j = 0; j < 10; j++) fn();
                    q.wait();
                    ts.push_back((now() - a) / 10);
                }
                std::sort(ts.begin(), ts.end());
                return ts[3];
            };
            for (int R = 1; R <= 4; R++) {
                std::vector<int> pos(R);
                for (int i = 0; i < R; i++) pos[i] = T - R + i;
                q.memcpy(pd.p, pos.data(), R * 4).wait();
                const int nch = TS / CH + 1;
                double t_old = timeit([&] {
                    q35ops::attn_core(q, dq.ptr<half>(), dk.ptr<half>(), dv.ptr<half>(), dg.ptr<half>(), R, 0, nch, part.ptr<float>(),
                                      o1.ptr<half>(), pd.ptr<int>(), sd.ptr<int>(), TS);
                });
                std::vector<half> ref((size_t)R * AH * AD), got(ref.size()), one(ref.size());
                q.memcpy(ref.data(), o1.p, ref.size() * 2).wait();
                printf("context %d, %d rows: plain kernel %.3f ms (%.0f GB/s of K/V)\n", T, R, t_old * 1e3,
                       (double)T * AKV * AD * 4 / t_old * 1e-9);
                for (int ns : nss) {
                    auto run = [&](int rows) {
                        q35ops::attn_dec(q, dq.ptr<half>(), dk.ptr<half>(), dv.ptr<half>(), rows, pd.ptr<int>(), sd.ptr<int>(), TS,
                                         part.ptr<float>(), ns);
                        q35ops::attn_dec_combine(q, part.ptr<float>(), rows, ns, dg.ptr<half>(), o2.ptr<half>());
                    };
                    double t_new = timeit([&] { run(R); });
                    q.memcpy(got.data(), o2.p, got.size() * 2).wait();
                    double num = 0, den = 0;
                    for (size_t i = 0; i < got.size(); i++) {
                        double d = (double)(float)got[i] - (double)(float)ref[i];
                        num += d * d;
                        den += (double)(float)ref[i] * (double)(float)ref[i];
                    }
                    // each row alone, as a one-row step at its position (the q / gate rows moved to row 0)
                    int differ = 0;
                    for (int i = 0; i < R; i++) {
                        q.memcpy(pd.p, &pos[i], 4);
                        q.memcpy(dq.p, hq.data() + (size_t)i * AH * AD, (size_t)AH * AD * 2);
                        q.memcpy(dg.p, hg.data() + (size_t)i * AH * 2 * AD, (size_t)AH * 2 * AD * 2);
                        run(1);
                        q.memcpy(one.data() + (size_t)i * AH * AD, o2.p, (size_t)AH * AD * 2).wait();
                    }
                    q.memcpy(dq.p, hq.data(), hq.size() * 2);
                    q.memcpy(dg.p, hg.data(), hg.size() * 2);
                    q.memcpy(pd.p, pos.data(), R * 4).wait();
                    for (size_t i = 0; i < got.size(); i++)
                        differ += std::memcmp(&got[i], &one[i], 2) != 0;
                    exact_all = exact_all && differ == 0;
                    printf("  ESIMD ns %3d: %.3f ms (%.0f GB/s), rel err vs plain %.2e, rows vs one-row steps: %s\n", ns, t_new * 1e3,
                           (double)T * AKV * AD * 4 / t_new * 1e-9, std::sqrt(num / std::max(den, 1e-30)),
                           differ ? ("DIFFER (" + std::to_string(differ) + " values)").c_str() : "bit-identical");
                }
                fflush(stdout);
            }
        }
        printf("%s\n", exact_all ? "decbench: every verify row bit-identical" : "decbench: MISMATCH");
        return exact_all ? 0 : 2;
    }
    if (mode == "gdncheck") {
        std::vector<int> lens;
        {
            std::stringstream ss(argc > 4 ? argv[4] : "1,63,64,65,700,2048");
            std::string t;
            while (std::getline(ss, t, ',')) lens.push_back(atoi(t.c_str()));
        }
        auto& q = Dev::get().q;
        constexpr int GQ = q4::GQKV, GV = q4::GV, ST = GV * q4::GDK * q4::GDV;
        uint64_t st = 88172645463325252ull;
        auto unif = [&]() { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return (double)(st >> 11) / 9007199254740992.0; };
        auto gauss = [&]() { return std::sqrt(-2 * std::log(unif() + 1e-300)) * std::cos(6.283185307179586 * unif()); };
        bool ok = true;
        for (int n : lens) {
            std::vector<bf16> c((size_t)n * GQ);
            std::vector<float> beta((size_t)n * GV), g((size_t)n * GV), rec(ST);
            for (size_t i = 0; i < c.size(); i++) c[i] = bf16((float)(gauss() * ((i % GQ) < 4096 ? 0.6 : 1.0)));
            for (auto& b : beta) b = (float)bf16((float)(1.0 / (1.0 + std::exp(-gauss()))));
            for (auto& x : g) x = (float)(-std::exp(gauss() - 1.5));
            for (auto& x : rec) x = (float)(gauss() * 0.05);
            Tensor dc = Tensor::empty({(int64_t)c.size()}, DType::BF16), db = Tensor::empty({(int64_t)beta.size()}, DType::F32),
                   dg = Tensor::empty({(int64_t)g.size()}, DType::F32), r1 = Tensor::empty({ST}, DType::F32),
                   r2 = Tensor::empty({ST}, DType::F32), o1 = Tensor::empty({(int64_t)n * GV * 128}, DType::BF16),
                   o2 = Tensor::empty({(int64_t)n * GV * 128}, DType::BF16),
                   scr = Tensor::empty({(int64_t)q35ops::gdn_chunk_bytes(n)}, DType::U8);
            q.memcpy(dc.p, c.data(), c.size() * 2);
            q.memcpy(db.p, beta.data(), beta.size() * 4);
            q.memcpy(dg.p, g.data(), g.size() * 4);
            q.memcpy(r1.p, rec.data(), ST * 4);
            q.memcpy(r2.p, rec.data(), ST * 4).wait();
            double a = now();
            q4ops::gdn_delta(q, dc.ptr<bf16>(), db.ptr<float>(), dg.ptr<float>(), r1.ptr<float>(), o1.ptr<bf16>(), n);
            q.wait();
            double b = now();
            q35ops::gdn_chunked(q, dc.ptr<bf16>(), db.ptr<float>(), dg.ptr<float>(), r2.ptr<float>(), o2.ptr<bf16>(), n, scr.p);
            q.wait();
            double e = now();
            std::vector<bf16> h1((size_t)n * GV * 128), h2(h1.size());
            std::vector<float> s1(ST), s2(ST);
            q.memcpy(h1.data(), o1.p, h1.size() * 2);
            q.memcpy(h2.data(), o2.p, h2.size() * 2);
            q.memcpy(s1.data(), r1.p, ST * 4);
            q.memcpy(s2.data(), r2.p, ST * 4).wait();
            auto cmp = [](auto& x, auto& y, double& mx) {
                double num = 0, den = 0;
                mx = 0;
                for (size_t i = 0; i < x.size(); i++) {
                    double d = (double)(float)x[i] - (double)(float)y[i];
                    num += d * d;
                    den += (double)(float)x[i] * (double)(float)x[i];
                    mx = std::max(mx, std::fabs(d));
                }
                return std::sqrt(num / std::max(den, 1e-30));
            };
            double mo, ms;
            double ro = cmp(h1, h2, mo), rs = cmp(s1, s2, ms);
            // the conv: parallel prompt kernel against the serial one, bit for bit
            int conv_diff = 0;
            {
                constexpr int CS = q4::GCONV - 1;
                std::vector<float> x((size_t)n * GQ);
                std::vector<bf16> wv((size_t)GQ * q4::GCONV), sv((size_t)GQ * CS);
                for (auto& v : x) v = (float)gauss();
                for (auto& v : wv) v = bf16((float)(gauss() * 0.5));
                for (auto& v : sv) v = bf16((float)gauss());
                Tensor dx = Tensor::empty({(int64_t)x.size()}, DType::F32), dw = Tensor::empty({(int64_t)wv.size()}, DType::BF16),
                       s1 = Tensor::empty({(int64_t)sv.size()}, DType::BF16), s2 = Tensor::empty({(int64_t)sv.size()}, DType::BF16),
                       c1 = Tensor::empty({(int64_t)n * GQ}, DType::BF16), c2 = Tensor::empty({(int64_t)n * GQ}, DType::BF16);
                q.memcpy(dx.p, x.data(), x.size() * 4);
                q.memcpy(dw.p, wv.data(), wv.size() * 2);
                q.memcpy(s1.p, sv.data(), sv.size() * 2);
                q.memcpy(s2.p, sv.data(), sv.size() * 2).wait();
                q4ops::gdn_conv(q, dx.ptr<float>(), s1.ptr<bf16>(), dw.ptr<bf16>(), c1.ptr<bf16>(), n);
                q35ops::gdn_conv_prompt(q, dx.ptr<float>(), s2.ptr<bf16>(), dw.ptr<bf16>(), c2.ptr<bf16>(), n);
                q.wait();
                std::vector<uint16_t> a((size_t)n * GQ), b2(a.size()), sa(sv.size()), sb(sv.size());
                q.memcpy(a.data(), c1.p, a.size() * 2);
                q.memcpy(b2.data(), c2.p, b2.size() * 2);
                q.memcpy(sa.data(), s1.p, sa.size() * 2);
                q.memcpy(sb.data(), s2.p, sb.size() * 2).wait();
                for (size_t i = 0; i < a.size(); i++) conv_diff += a[i] != b2[i];
                for (size_t i = 0; i < sa.size(); i++) conv_diff += sa[i] != sb[i];
            }
            printf("rows %5d: conv prompt kernel vs serial: %d values differ\n", n, conv_diff);
            bool good = ro < 2e-2 && rs < 2e-2 && std::isfinite(ro) && std::isfinite(rs) && conv_diff == 0;
            ok = ok && good;
            printf("rows %5d: out rel %.2e (max %.3g), state rel %.2e (max %.3g); serial %.2f ms, chunked %.2f ms%s\n", n, ro, mo, rs,
                   ms, 1e3 * (b - a), 1e3 * (e - b), good ? "" : "  BAD");
        }
        if (g_prof_points) {
            // the chunked kernels alone, a few times over the last size
            const int n = lens.back(), reps = 5;
            Tensor dc = Tensor::zeros({(int64_t)n * GQ}, DType::BF16), db = Tensor::zeros({(int64_t)n * GV}, DType::F32),
                   dg = Tensor::zeros({(int64_t)n * GV}, DType::F32), r2 = Tensor::zeros({ST}, DType::F32),
                   o2 = Tensor::empty({(int64_t)n * GV * 128}, DType::BF16),
                   scr = Tensor::empty({(int64_t)q35ops::gdn_chunk_bytes(n)}, DType::U8);
            q.wait();
            prof_points_report(-1);
            prof_point("start");
            for (int i = 0; i < reps; i++)
                q35ops::gdn_chunked(q, dc.ptr<bf16>(), db.ptr<float>(), dg.ptr<float>(), r2.ptr<float>(), o2.ptr<bf16>(), n, scr.p);
            printf("chunked kernels, %d rows (ms a call):\n%s", n, prof_points_report(reps).c_str());
        }
        // the decode delta rule's two kernels (q35ops::gdn_delta_variant 0 / 1) on one group of n rows
        for (int n = 1; n <= 4; n++) {
            std::vector<bf16> c((size_t)n * GQ);
            std::vector<float> beta((size_t)n * GV), g((size_t)n * GV), rec(2 * (size_t)ST);
            for (size_t i = 0; i < c.size(); i++) c[i] = bf16((float)(gauss() * ((i % GQ) < 4096 ? 0.6 : 1.0)));
            for (auto& b : beta) b = (float)(1.0 / (1.0 + std::exp(-gauss())));
            for (auto& x : g) x = (float)(-std::exp(gauss() - 1.5));
            for (auto& x : rec) x = (float)(gauss() * 0.05);
            std::vector<int> desc((size_t)n * 3, 0);
            desc[0] = n; desc[1] = 0; desc[2] = 1;              // read slot 0, write slot 1
            Tensor dc = Tensor::empty({(int64_t)c.size()}, DType::BF16), db = Tensor::empty({(int64_t)beta.size()}, DType::F32),
                   dg = Tensor::empty({(int64_t)g.size()}, DType::F32), dd = Tensor::empty({(int64_t)desc.size()}, DType::I32);
            q.memcpy(dc.p, c.data(), c.size() * 2);
            q.memcpy(db.p, beta.data(), beta.size() * 4);
            q.memcpy(dg.p, g.data(), g.size() * 4);
            q.memcpy(dd.p, desc.data(), desc.size() * 4).wait();
            std::vector<bf16> o[2];
            std::vector<float> st[2];
            const int keep = q35ops::gdn_delta_variant;
            for (int var = 0; var < 2; var++) {
                Tensor r = Tensor::empty({2 * (int64_t)ST}, DType::F32), oc = Tensor::empty({(int64_t)n * GV * 128}, DType::BF16);
                q.memcpy(r.p, rec.data(), rec.size() * 4).wait();
                q35ops::gdn_delta_variant = var;
                q35ops::gdn_delta_seq(q, dc.ptr<bf16>(), db.ptr<float>(), dg.ptr<float>(), r.ptr<float>(), oc.ptr<bf16>(), n, dd.ptr<int>());
                q.wait();
                o[var].resize((size_t)n * GV * 128);
                st[var].resize(ST);
                q.memcpy(o[var].data(), oc.p, o[var].size() * 2);
                q.memcpy(st[var].data(), r.ptr<float>() + ST, (size_t)ST * 4).wait();
            }
            q35ops::gdn_delta_variant = keep;
            auto rel = [](auto& x, auto& y) {
                double num = 0, den = 0;
                for (size_t i = 0; i < x.size(); i++) {
                    double d = (double)(float)x[i] - (double)(float)y[i];
                    num += d * d;
                    den += (double)(float)x[i] * (double)(float)x[i];
                }
                return std::sqrt(num / std::max(den, 1e-30));
            };
            const double ro = rel(o[0], o[1]), rs = rel(st[0], st[1]);
            const bool good = ro < 1e-2 && rs < 1e-4 && std::isfinite(ro) && std::isfinite(rs);
            ok = ok && good;
            printf("decode delta rule, %d rows: kernel 1 vs 0: out rel %.2e, state rel %.2e%s\n", n, ro, rs, good ? "" : "  BAD");
        }
        printf("%s\n", ok ? "gdncheck ok" : "gdncheck FAILED");
        return ok ? 0 : 2;
    }
    if (mode == "prefill" || mode == "kl") {
        // prefill: q35_gen <dir> prefill <text|@file> <lengths "8192,30720"> [ctx] [iters]
        //   each length read iters times from an empty sequence (the text repeated as needed): median time and tok/s;
        //   GLM_PROF=5 adds the last read's time per stage
        // kl: q35_gen <dir> kl <text|@file> <lengths> [scored] [ctx] [parts]
        //   the first length tokens read as a prompt by the old prompt kernels and by the new ones (parts: attn, gdn or
        //   all), then the next scored tokens teacher-forced one at a time; NLL of each, KL(old || new), top-1 agreement
        std::vector<int> lens;
        {
            std::stringstream ss(argc > 4 ? argv[4] : "8192");
            std::string t;
            while (std::getline(ss, t, ',')) lens.push_back(atoi(t.c_str()));
        }
        const bool kl = mode == "kl";
        const int scored = kl ? (argc > 5 ? atoi(argv[5]) : 64) : 0;
        int maxlen = 0;
        for (int n : lens) maxlen = std::max(maxlen, n);
        const int cargc = kl ? 6 : 5;
        int ctx = argc > cargc ? atoi(argv[cargc]) : maxlen + scored + 64;
        const int iters = !kl && argc > 6 ? atoi(argv[6]) : 3;
        const std::string parts = kl && argc > 7 ? argv[7] : "all";
        std::vector<int> base = ids;
        while ((int)ids.size() < maxlen + scored + 1) ids.insert(ids.end(), base.begin() + 1, base.end());
        Q35Model m(dir, ctx);
        auto& q = Dev::get().q;
        printf("load %.1f s, MTP %s\n", now() - t0, m.has_mtp() ? "loaded" : "off");
        if (!kl) {
            // Q35_GEN_PATHS="00,11": the prompt kernels to time, attention then GDN (0 old, 1 new); default as loaded
            std::vector<std::pair<int, int>> paths;
            if (getenv("Q35_GEN_PATHS")) {
                std::stringstream ss(getenv("Q35_GEN_PATHS"));
                std::string t;
                while (std::getline(ss, t, ',')) if (t.size() == 2) paths.push_back({t[0] - '0', t[1] - '0'});
            }
            if (paths.empty()) paths.push_back({m.pf_attn, m.pf_gdn});
            for (auto [pa, pg] : paths)
            for (int n : lens) {
                m.pf_attn = pa;
                m.pf_gdn = pg;
                std::vector<double> ts;
                for (int it = 0; it < iters; it++) {
                    m.reset();
                    q.wait();
                    if (g_prof_points) prof_points_report(-1);
                    double a = now();
                    m.prefill(ids.data(), n);
                    q.wait();
                    ts.push_back(now() - a);
                }
                std::vector<double> s = ts;
                std::sort(s.begin(), s.end());
                printf("attn %d gdn %d, prompt %d tokens: median %.2f s (%.0f tok/s), runs", pa, pg, n, s[s.size() / 2],
                       n / s[s.size() / 2]);
                for (double t : ts) printf(" %.2f", t);
                if (getenv("Q35_GEN_HASH")) {
                    // the next token's logits after the last read, hashed (bit-identical prompt paths agree)
                    std::vector<half> h(q35::V);
                    q.memcpy(h.data(), m.forward(&ids[n], 1), h.size() * 2).wait();
                    uint64_t x = 1469598103934665603ull;
                    for (auto v : h) { uint16_t b; std::memcpy(&b, &v, 2); x = (x ^ b) * 1099511628211ull; }
                    printf(", logits hash %016llx", (unsigned long long)x);
                }
                printf("\n");
                if (g_prof_points) printf("  last run by stage (ms in total):\n%s", prof_points_report(1).c_str());
                fflush(stdout);
            }
            return 0;
        }
        const int pa = m.pf_attn, pg = m.pf_gdn;
        for (int n : lens) {
            std::vector<std::vector<float>> ls[2];
            double nll[2] = {0, 0}, tp[2] = {0, 0};
            for (int path = 0; path < 2; path++) {
                m.pf_attn = path == 0 || parts == "gdn" ? 0 : pa;
                m.pf_gdn = path == 0 || parts == "attn" ? 0 : pg;
                m.reset();
                q.wait();
                double a = now();
                m.prefill(ids.data(), n);
                q.wait();
                tp[path] = now() - a;
                for (int i = 0; i < scored; i++) {
                    auto lg = row(m.forward(&ids[n + i], 1));
                    double mx = lg[0];
                    for (int v = 1; v < q35::V; v++) mx = std::max(mx, (double)lg[v]);
                    double s = 0;
                    for (int v = 0; v < q35::V; v++) s += std::exp((double)lg[v] - mx);
                    const double lz = mx + std::log(s);
                    std::vector<float> l(q35::V);
                    for (int v = 0; v < q35::V; v++) l[v] = (float)(lg[v] - lz);
                    nll[path] -= l[ids[n + i + 1]];
                    ls[path].push_back(std::move(l));
                }
            }
            double kls = 0, klmax = 0;
            int agree = 0;
            for (int i = 0; i < scored; i++) {
                const auto &a = ls[0][i], &b = ls[1][i];
                double k = 0;
                int ia = 0, ib = 0;
                for (int v = 0; v < q35::V; v++) {
                    k += std::exp((double)a[v]) * ((double)a[v] - b[v]);
                    if (a[v] > a[ia]) ia = v;
                    if (b[v] > b[ib]) ib = v;
                }
                kls += k;
                klmax = std::max(klmax, k);
                agree += ia == ib;
            }
            printf("prompt %d (%s): NLL old %.5f new %.5f over %d tokens, KL mean %.6f max %.6f, top-1 same %d/%d; "
                   "read %.2f s old, %.2f s new\n",
                   n, parts.c_str(), nll[0] / scored, nll[1] / scored, scored, kls / scored, klmax, agree, scored, tp[0], tp[1]);
            fflush(stdout);
        }
        m.pf_attn = pa;
        m.pf_gdn = pg;
        return 0;
    }
    GLM_CHECK(false, "unknown mode %s", mode.c_str());
    return 1;
}
