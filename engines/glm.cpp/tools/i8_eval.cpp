// What the EXL3 int8 prompt path (GLM_EXL3_INT8) costs and gains on a whole model (Qwen3.5/3.8 dense or Gemma 4), the
// two paths toggled in one process:
//   i8_eval <model dir> text <file> <prompt P> <scored S> <pieces>
//       pieces of the text: P tokens read as a prompt (fp16 path, then int8), the next S tokens scored one at a time
//       (decode runs fp16 either way); KL(fp16 || int8) over the whole vocabulary, top-1 agreement, NLL of each
//   i8_eval <model dir> trace <rows.json> [cols 2048]
//       exl3eval --trace rows (input_ids read as a prompt, response_ids scored), as g4_gen scoretrace
//   i8_eval <model dir> tracecat <rows.json> <P> <S> <pieces>
//       long chat prompts from exl3eval --trace rows: whole rows (prompt + reply) joined until P tokens, then the next
//       row's prompt; its reply's first S tokens scored
//   i8_eval <model dir> speed <n> [reps 3] [paths 01]
//       a prompt of n tokens read reps times per path (0 fp16, 1 int8): median seconds and tok/s
// GLM_EXL3_INT8 has no effect here: each run sets the path itself. I8_EVAL_A / I8_EVAL_B (text, trace, tracecat): the
// two prompt paths compared, f16 (default A), i8 (default B) or dpas (fp16, every row on the DPAS kernel: a second
// fp16 implementation, for the noise floor)
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include "../src/exl3.h"
#include "../src/g4.h"
#include "../src/json.h"
#include "../src/q35.h"
#include "../src/tokenizer.h"

using namespace glm;

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// log-softmax of fp16 logits on the device, in double
static void logsm(const half* lg, int V, std::vector<double>& out) {
    std::vector<half> h(V);
    Dev::get().q.memcpy(h.data(), lg, (size_t)V * 2).wait();
    out.resize(V);
    double mx = -1e300;
    for (int v = 0; v < V; v++) mx = std::max(mx, (double)(float)h[v]);
    double s = 0;
    for (int v = 0; v < V; v++) s += std::exp((double)(float)h[v] - mx);
    double lse = mx + std::log(s);
    for (int v = 0; v < V; v++) out[v] = (double)(float)h[v] - lse;
}

struct Stats {
    std::vector<double> kl, d_nll;     // per scored token: KL(fp16 || int8), nll(int8) - nll(fp16)
    double nll16 = 0, nll8 = 0;
    int agree = 0, top16 = 0, top8 = 0;
    void add(const std::vector<double>& a, const std::vector<double>& b, int next) {
        double k = 0;
        int ia = 0, ib = 0;
        for (size_t v = 0; v < a.size(); v++) {
            double pa = std::exp(a[v]);
            if (pa > 0) k += pa * (a[v] - b[v]);
            if (a[v] > a[ia]) ia = (int)v;
            if (b[v] > b[ib]) ib = (int)v;
        }
        kl.push_back(std::max(0.0, k));
        nll16 += -a[next];
        nll8 += -b[next];
        d_nll.push_back(-b[next] + a[next]);
        agree += ia == ib;
        top16 += ia == next;
        top8 += ib == next;
    }
    static void ci(const std::vector<double>& x, double& mean, double& half) {
        double s = 0, s2 = 0;
        for (double v : x) s += v;
        mean = s / x.size();
        for (double v : x) s2 += (v - mean) * (v - mean);
        half = 1.96 * std::sqrt(s2 / std::max<size_t>(1, x.size() - 1) / x.size());
    }
    void print() const {
        int n = (int)kl.size();
        if (!n) return;
        double km, kh, dm, dh;
        ci(kl, km, kh);
        ci(d_nll, dm, dh);
        std::vector<double> s = kl;
        std::sort(s.begin(), s.end());
        printf("%d scored tokens: KL(fp16||int8) %.5f [+-%.5f], median %.6f, p99 %.4f, max %.3f; top-1 agreement %.2f%%\n", n, km,
               kh, s[n / 2], s[(size_t)(n * 0.99)], s.back(), 100.0 * agree / n);
        printf("  NLL fp16 %.5f  int8 %.5f  diff %+.5f [+-%.5f] (%+.3f%%); top-1 vs the text: fp16 %.2f%%, int8 %.2f%%\n",
               nll16 / n, nll8 / n, dm, dh, 100.0 * dm / (nll16 / n), 100.0 * top16 / n, 100.0 * top8 / n);
        fflush(stdout);
    }
};

int main(int argc, char** argv) {
    GLM_CHECK(argc >= 4, "usage: i8_eval <model dir> text|trace|speed ...");
    std::string dir = argv[1], mode = argv[2];
    Json cfg = Json::parse(read_file(dir + "/config.json"));
    const bool gemma = cfg.get_s("model_type", "") == "gemma4";
    auto load = [&](int ctx) -> std::unique_ptr<LM> {
        if (gemma) return std::make_unique<G4Model>(dir, ctx);
        return std::make_unique<Q35Model>(dir, ctx);
    };
    auto& q = Dev::get().q;
    Stats st;
    const std::string pa = getenv("I8_EVAL_A") ? getenv("I8_EVAL_A") : "f16", pb = getenv("I8_EVAL_B") ? getenv("I8_EVAL_B") : "i8";
    printf("paths: A %s, B %s\n", pa.c_str(), pb.c_str());
    const int large0 = exl3_large_above;
    auto set_path = [&](const std::string& p) {
        GLM_CHECK(p == "f16" || p == "i8" || p == "dpas", "path %s: f16, i8 or dpas", p.c_str());
        exl3_int8 = p == "i8";
        exl3_large_above = p == "dpas" ? (1 << 30) : large0;
    };
    auto score = [&](LM& m, const std::vector<int>& ids, int n0, int S) {
        // ids[0, n0) prompt (the last one read by the first forward), then S predictions
        std::vector<std::vector<double>> ref(S);
        std::vector<double> lp;
        for (int path = 0; path < 2; path++) {
            m.reset();
            set_path(path ? pb : pa);
            if (n0 > 1) m.prefill(ids.data(), n0 - 1);
            set_path("f16");
            for (int j = 0; j < S; j++) {
                int p = n0 - 1 + j;
                logsm(m.forward(&ids[p], 1), m.vocab(), path ? lp : ref[j]);
                if (path) st.add(ref[j], lp, ids[p + 1]);
            }
        }
    };
    if (mode == "text") {
        GLM_CHECK(argc >= 7, "usage: i8_eval <dir> text <file> <P> <S> <pieces>");
        Tokenizer tok = Tokenizer::load(dir);
        std::vector<int> all = tok.encode(read_file(argv[3]), true);
        int P = atoi(argv[4]), S = atoi(argv[5]), pieces = atoi(argv[6]);
        auto m = load(P + S + 64);
        printf("%zu tokens of text; %d pieces of %d prompt + %d scored\n", all.size(), pieces, P, S);
        for (int i = 0; i < pieces; i++) {
            size_t off = (size_t)i * (P + S + 1);
            if (off + P + S + 1 > all.size()) break;
            std::vector<int> ids;
            if (gemma) ids.push_back(2);    // <bos>
            ids.insert(ids.end(), all.begin() + off, all.begin() + off + P + S + 1);
            double t = now();
            score(*m, ids, (int)ids.size() - S, S);
            printf("piece %d (%.0f s): ", i, now() - t);
            st.print();
        }
        printf("FINAL ");
        st.print();
        return 0;
    }
    if (mode == "trace") {
        Json tj = Json::parse(read_file(argv[3]));
        int cols = argc > 4 ? atoi(argv[4]) : 2048;
        auto m = load(cols + 64);
        for (auto& r : tj["rows"].arr) {
            std::vector<int> ids;
            for (auto& x : r["input_ids"].arr) ids.push_back((int)x.i());
            int n0 = (int)ids.size();
            for (auto& x : r["response_ids"].arr) ids.push_back((int)x.i());
            if ((int)ids.size() > cols) ids.resize(cols);
            if ((int)ids.size() - n0 < 32) continue;
            score(*m, ids, n0, (int)ids.size() - n0);
            printf("row (prompt %d): ", n0);
            st.print();
        }
        printf("FINAL ");
        st.print();
        return 0;
    }
    if (mode == "tracecat") {
        GLM_CHECK(argc >= 7, "usage: i8_eval <dir> tracecat <rows.json> <P> <S> <pieces>");
        Json tj = Json::parse(read_file(argv[3]));
        int P = atoi(argv[4]), S = atoi(argv[5]), pieces = atoi(argv[6]);
        auto m = load(P + 8192 + S + 64);
        std::vector<int> ctx;
        int done = 0;
        for (auto& r : tj["rows"].arr) {
            if (done >= pieces) break;
            std::vector<int> in, out;
            for (auto& x : r["input_ids"].arr) in.push_back((int)x.i());
            for (auto& x : r["response_ids"].arr) out.push_back((int)x.i());
            if (!ctx.empty() && !in.empty() && in[0] == 2) in.erase(in.begin());     // one <bos>, at the start
            if ((int)ctx.size() < P || (int)out.size() < S + 1) {
                ctx.insert(ctx.end(), in.begin(), in.end());
                ctx.insert(ctx.end(), out.begin(), out.end());
                continue;
            }
            std::vector<int> ids = ctx;
            ids.insert(ids.end(), in.begin(), in.end());
            int n0 = (int)ids.size();
            ids.insert(ids.end(), out.begin(), out.begin() + S);
            double t = now();
            score(*m, ids, n0, S);
            printf("piece %d (prompt %d, %.0f s): ", done, n0, now() - t);
            st.print();
            done++;
            ctx.clear();
        }
        printf("FINAL ");
        st.print();
        return 0;
    }
    if (mode == "speed") {
        int n = atoi(argv[3]);
        int reps = argc > 4 ? atoi(argv[4]) : 3;
        std::string paths = argc > 5 ? argv[5] : "01";
        auto m = load(n + 64);
        std::vector<int> ids(n);
        for (int i = 0; i < n; i++) ids[i] = 1000 + (i * 7919) % 50000;
        for (char pc : paths) {
            exl3_int8 = pc == '1';
            std::vector<double> ts;
            for (int r = 0; r < reps; r++) {
                m->reset();
                q.wait();
                double t = now();
                m->prefill(ids.data(), n);
                q.wait();
                ts.push_back(now() - t);
                printf("  %s %d tokens: %.2f s (%.0f tok/s)\n", exl3_int8 ? "int8" : "fp16", n, ts.back(), n / ts.back());
                if (g_prof_points) printf("%s\n", prof_points_report(n).c_str());
                fflush(stdout);
            }
            std::sort(ts.begin(), ts.end());
            double med = ts[ts.size() / 2];
            printf("SPEED %s %d tokens: median %.2f s, %.0f tok/s (%d runs)\n", exl3_int8 ? "int8" : "fp16", n, med, n / med, reps);
        }
        return 0;
    }
    GLM_CHECK(false, "unknown mode %s", mode.c_str());
}
