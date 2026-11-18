// Fine-tune or evaluate Gemma 4's assistant drafter against a target's own replies (src/g4_draft.h, llm/g4draft/README.md).
//   g4_draft train --target <dir> --drafter <dir> --data a.jsonl[,b.jsonl] --out <dir> [--state <file>] [--minutes 15]
//                  [--epochs 1] [--lr 3e-5] [--warmup 20] [--batch 4] [--anchors 256] [--steps 4] [--ctx 40960]
//   g4_draft eval  --target <dir> --drafter <dir> --data a.jsonl[,...] [--anchors 256] [--steps 4] [--margin 3]
// Data: the server's GLM_DUMP_TOKENS lines ({"prompt": [...], "reply": [...]}); requests whose prompt carries the data
// run's player marker are dropped. Sequences (prompt + reply) are read in sorted order, so a prompt shared with the one
// before is read once (the target restores to the branch point; checkpoints are placed where later sequences branch).
// Every reply position is an anchor. A train run stops after --minutes at a sequence boundary and saves --state (the
// optimizer and where it got to); run it again with the same arguments to go on. --out gets the drafter at the end.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include "../src/g4.h"
#include "../src/g4_draft.h"
#include "../src/json.h"
#include "../src/st.h"
#include "../src/tokenizer.h"

using namespace glm;

namespace {
double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

std::vector<std::string> split(const std::string& s, char c) {
    std::vector<std::string> out;
    size_t a = 0;
    while (a <= s.size()) {
        size_t b = s.find(c, a);
        if (b == std::string::npos) b = s.size();
        if (b > a) out.push_back(s.substr(a, b - a));
        a = b + 1;
    }
    return out;
}

struct Seq {
    std::vector<int> t;
    int n0 = 0;          // the reply's first position
};

// "[[g4draft player]]" heads the player's system prompt in llm/g4draft/gen_batch.py
bool is_player(const std::vector<int>& prompt, const std::vector<int>& mark) {
    if (mark.empty()) return false;
    const size_t lim = std::min<size_t>(prompt.size(), 600);
    for (size_t i = 0; i + mark.size() <= lim; i++)
        if (std::equal(mark.begin(), mark.end(), prompt.begin() + i)) return true;
    return false;
}

int lcp(const std::vector<int>& a, const std::vector<int>& b) {
    const size_t n = std::min(a.size(), b.size());
    size_t i = 0;
    while (i < n && a[i] == b[i]) i++;
    return (int)i;
}

void print_stats(const char* tag, const G4DraftStats& s, double secs) {
    printf("%s: %d anchors, KL %.4f, accepted/step %.3f (margin rule %.3f of %.3f drafted), tokens/step %.3f, hits",
           tag, s.anchors, s.rows ? s.kl / s.rows : 0.0, s.anchors ? s.accepted / s.anchors : 0.0,
           s.anchors ? s.accepted_m / s.anchors : 0.0, s.anchors ? s.drafted_m / s.anchors : 0.0,
           s.anchors ? 1.0 + s.accepted_m / s.anchors : 0.0);
    for (size_t k = 0; k < s.hit.size(); k++) printf(" %.3f", s.hit_n[k] ? s.hit[k] / s.hit_n[k] : 0.0);
    printf(" | ceiling: accepted/step %.3f, hits", s.anchors ? s.oracle / s.anchors : 0.0);
    for (size_t k = 0; k < s.thit.size(); k++) printf(" %.3f", s.hit_n[k] ? s.thit[k] / s.hit_n[k] : 0.0);
    printf(" (%.0f anchors/s)\n", secs > 0 ? s.anchors / secs : 0.0);
    fflush(stdout);
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: g4_draft train|eval --target DIR --drafter DIR --data A.jsonl[,B.jsonl] [options]\n");
        return 2;
    }
    const std::string mode = argv[1];
    std::string target, drafter, data, out, state;
    int epochs = 1, warmup = 20, batch = 4, anchors = 256, steps = 4, ctx = 40960, limit = 0;
    double minutes = 0, lr = 3e-5, margin = 3.0, min_lr_frac = 0.1;
    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        auto val = [&] { GLM_CHECK(i + 1 < argc, "%s needs a value", a.c_str()); return std::string(argv[++i]); };
        if (a == "--target") target = val();
        else if (a == "--drafter") drafter = val();
        else if (a == "--data") data = val();
        else if (a == "--out") out = val();
        else if (a == "--state") state = val();
        else if (a == "--minutes") minutes = std::stod(val());
        else if (a == "--epochs") epochs = std::stoi(val());
        else if (a == "--lr") lr = std::stod(val());
        else if (a == "--warmup") warmup = std::stoi(val());
        else if (a == "--batch") batch = std::stoi(val());
        else if (a == "--anchors") anchors = std::stoi(val());
        else if (a == "--steps") steps = std::stoi(val());
        else if (a == "--ctx") ctx = std::stoi(val());
        else if (a == "--margin") margin = std::stod(val());
        else if (a == "--limit") limit = std::stoi(val());
        else GLM_CHECK(false, "unknown argument %s", a.c_str());
    }
    GLM_CHECK(mode == "train" || mode == "eval" || mode == "gradcheck", "mode: train, eval or gradcheck");
    GLM_CHECK(!target.empty() && !drafter.empty() && !data.empty(), "--target, --drafter and --data are required");
    GLM_CHECK(mode != "train" || !out.empty(), "train needs --out");
    const bool train = mode == "train";
    const double t0 = now_s();

    // data
    Tokenizer tok = Tokenizer::load(target);
    const std::vector<int> mark = tok.encode("[[g4draft player]]", false);
    std::vector<Seq> seqs;
    int players = 0, too_long = 0, short_reply = 0;
    for (auto& path : split(data, ',')) {
        std::ifstream f(path);
        GLM_CHECK(f, "cannot read %s", path.c_str());
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            Json j = Json::parse(line);
            Seq s;
            for (auto& x : j["prompt"].arr) s.t.push_back((int)x.i());
            s.n0 = (int)s.t.size();
            if (is_player(s.t, mark)) { players++; continue; }
            for (auto& x : j["reply"].arr) s.t.push_back((int)x.i());
            if ((int)s.t.size() - s.n0 < 3) { short_reply++; continue; }
            if ((int)s.t.size() > ctx) { too_long++; continue; }
            seqs.push_back(std::move(s));
        }
    }
    std::sort(seqs.begin(), seqs.end(), [](const Seq& a, const Seq& b) { return a.t < b.t; });
    seqs.erase(std::unique(seqs.begin(), seqs.end(), [](const Seq& a, const Seq& b) { return a.t == b.t; }), seqs.end());
    if (limit > 0 && (int)seqs.size() > limit) seqs.resize(limit);
    std::vector<int> L(seqs.size(), 0);     // L[i]: prefix shared with sequence i - 1
    size_t tokens = 0, fresh = 0, reply = 0;
    for (size_t i = 0; i < seqs.size(); i++) {
        if (i) L[i] = lcp(seqs[i - 1].t, seqs[i].t);
        tokens += seqs[i].t.size();
        fresh += seqs[i].t.size() - L[i];
        reply += seqs[i].t.size() - seqs[i].n0;
    }
    printf("data: %zu sequences (%d player requests, %d too long, %d short replies dropped), %zu tokens, %zu to read, "
           "%zu reply tokens\n",
           seqs.size(), players, too_long, short_reply, tokens, fresh, reply);
    GLM_CHECK(!seqs.empty(), "no sequences");
    fflush(stdout);

    // the cursor of a resumed run
    int epoch0 = 0;
    size_t idx0 = 0;
    const std::string cursor = state.empty() ? "" : state + ".json";
    // the trainer's VRAM is kept out of the target's context sizing
    if (!getenv("G4_VRAM_RESERVE_GIB")) setenv("G4_VRAM_RESERVE_GIB", "10.5", 1);
    G4Model m(target, ctx + 64, "", 1);
    m.set_turn_checkpoints(6);
    G4DraftTrainer tr(m, drafter, anchors, steps, ctx);
    m.set_capture(tr.capture());
    if (train && !state.empty() && tr.load_state(state)) {
        Json c = Json::parse(read_file(cursor));
        epoch0 = (int)c.get_i("epoch", 0);
        idx0 = (size_t)c.get_i("index", 0);
        printf("resumed %s: optimizer step %d, epoch %d, sequence %zu\n", state.c_str(), tr.opt_steps(), epoch0, idx0);
    }
    printf("ready in %.1f s\n", now_s() - t0);
    fflush(stdout);

    if (mode == "gradcheck") {
        // the first sequence with a reply long enough: the loss along a random direction u, by central differences,
        // against the backward pass's gradient . u (fp32: expect agreement to a few parts in a thousand)
        size_t i = 0;
        while (i < seqs.size() && (int)seqs[i].t.size() - seqs[i].n0 < anchors + steps + 2) i++;
        GLM_CHECK(i < seqs.size(), "no reply of %d tokens", anchors + steps + 2);
        const Seq& s = seqs[i];
        m.reset();
        m.prefill(s.t.data(), (int)s.t.size());
        tr.begin_sequence((int)s.t.size());
        std::vector<int> an;
        for (int x = s.n0; x < s.n0 + anchors; x++) an.push_back(x);
        auto loss = [&](bool back) {
            G4DraftStats st = tr.chunk(s.t, an, back, (float)margin);
            return st.kl / st.rows;
        };
        tr.zero_grad();
        const double l0 = loss(true);
        const uint32_t seed = 12345;
        const double gu = tr.grad_dot(seed);
        tr.zero_grad();
        printf("gradcheck: %d anchors at %d of %zu tokens, loss %.6f, grad . u %.6e\n", anchors, s.n0, s.t.size(), l0, gu);
        for (float eps : {1e-3f, 3e-4f, 1e-4f}) {
            tr.perturb(eps, seed);
            const double lp = loss(false);
            tr.perturb(-2 * eps, seed);
            const double lm = loss(false);
            tr.perturb(eps, seed);
            const double fd = (lp - lm) / (2 * eps);
            printf("  eps %.0e: finite difference %.6e, ratio %.4f\n", eps, fd, fd / gu);
        }
        return 0;
    }

    // optimizer steps planned (for the schedule): chunks / batch per epoch
    size_t chunks_epoch = 0;
    for (auto& s : seqs) {
        const int a = std::max(s.n0, 1), b = (int)s.t.size() - 2;
        if (b >= a) chunks_epoch += (b - a + anchors) / anchors;
    }
    const int total_steps = std::max<int>(1, (int)(chunks_epoch * epochs / batch));
    auto lr_at = [&](int t) {
        if (t < warmup) return lr * (t + 1) / warmup;
        const double x = std::min(1.0, (double)(t - warmup) / std::max(1, total_steps - warmup));
        return lr * (min_lr_frac + (1 - min_lr_frac) * 0.5 * (1 + std::cos(M_PI * x)));
    };

    G4DraftStats total, window;
    double t_win = now_s(), t_read = 0, t_train = 0;
    int pending = 0;
    bool stopped = false;
    size_t prev = SIZE_MAX;            // the sequence the target last read (this process)
    for (int ep = epoch0; ep < epochs && !stopped; ep++) {
        for (size_t i = ep == epoch0 ? idx0 : 0; i < seqs.size(); i++) {
            if (train && minutes > 0 && now_s() - t0 > minutes * 60) {
                stopped = true;
                if (!state.empty()) {
                    tr.save_state(state);
                    std::ofstream(cursor) << "{\"epoch\": " << ep << ", \"index\": " << i << "}\n";
                    printf("stopped at epoch %d, sequence %zu of %zu; state saved to %s\n", ep, i, seqs.size(), state.c_str());
                }
                break;
            }
            const Seq& s = seqs[i];
            const int len = (int)s.t.size();
            // the target: back to the prefix shared with the sequence before (a checkpoint or the ring), then read the rest,
            // with a checkpoint wherever a later sequence branches from this one's new part
            double tr0 = now_s();
            const int share = prev != SIZE_MAX && prev + 1 == i ? L[i] : 0;
            if (!m.restore(std::min(share, len - 1))) m.reset();
            int pos = m.position();
            std::vector<int> marks;
            int mn = INT32_MAX;
            for (size_t j = i + 1; j < seqs.size() && j < i + 256; j++) {
                mn = std::min(mn, L[j]);
                if (mn <= pos) break;
                if (mn < len && (marks.empty() || marks.back() != mn)) marks.push_back(mn);
            }
            std::sort(marks.begin(), marks.end());
            for (int mk : marks) {
                if (mk - pos > 0) m.prefill_turn(s.t.data() + pos, mk - pos, mk);
                pos = mk;
            }
            if (len > pos) m.prefill(s.t.data() + pos, len - pos);
            tr.begin_sequence(len);
            Dev::get().q.wait();
            prev = i;
            t_read += now_s() - tr0;
            // anchors: every reply position with a next token
            const double tt0 = now_s();
            const int a0 = std::max(s.n0, 1), a1 = len - 2;
            for (int p = a0; p <= a1; p += anchors) {
                std::vector<int> an;
                for (int x = p; x <= std::min(a1, p + anchors - 1); x++) an.push_back(x);
                G4DraftStats st = tr.chunk(s.t, an, train, (float)margin);
                total.add(st);
                window.add(st);
                if (train && ++pending == batch) {
                    const double rate = lr_at(tr.opt_steps());
                    const double gn = tr.step((float)rate, pending);
                    pending = 0;
                    if (tr.opt_steps() % 10 == 0) {
                        printf("step %d/%d lr %.2e grad %.3f | ", tr.opt_steps(), total_steps, rate, gn);
                        print_stats("train", window, now_s() - t_win);
                        window = G4DraftStats();
                        t_win = now_s();
                    }
                }
            }
            t_train += now_s() - tt0;
        }
        if (!stopped) {
            idx0 = 0;
            printf("epoch %d done\n", ep);
            print_stats(train ? "epoch" : "eval", total, t_train);
            fflush(stdout);
        }
    }
    printf("time: reading %.0f s, drafter %.0f s\n", t_read, t_train);
    if (train && !stopped) {
        if (pending) tr.step((float)lr_at(tr.opt_steps()), pending);
        tr.save_model(out);
        printf("saved %s (%d optimizer steps)\n", out.c_str(), tr.opt_steps());
        if (!state.empty()) {
            tr.save_state(state);
            std::ofstream(cursor) << "{\"epoch\": " << epochs << ", \"index\": 0, \"done\": true}\n";
        }
    }
    print_stats(train ? "total" : "eval", total, t_train);
    return 0;
}
