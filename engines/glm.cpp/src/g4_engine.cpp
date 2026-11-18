#include "g4_engine.h"
#include <algorithm>
#include <chrono>

namespace glm {

namespace {
double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
}  // namespace

G4Engine::G4Engine(MultiLM& m) : m_(m), games_(m.games()) {}

bool G4Engine::has_free() const {
    for (auto& g : games_) if (!g.active) return true;
    return false;
}

bool G4Engine::busy() const {
    for (auto& g : games_) if (g.active) return true;
    return false;
}

void G4Engine::submit(GenRequest r, std::function<void(const GenResult&)> done) {
    const int n = (int)r.prompt.size();
    GLM_CHECK(n >= 1, "empty prompt");
    // the free game that saves the most reading: what it can bring back of the tokens it shares with the prompt. A
    // shared beginning it cannot go back to saves nothing, and a conversation of its own that starts like the story
    // would take the story's game and wipe it. Among games that bring back the same (all of them, for a new
    // conversation): the one continued least, then the one idle longest. An empty game is both.
    const double now = now_s();
    int best = -1, best_l = -1, best_keep = -1, best_cont = 0;
    for (int g = 0; g < (int)games_.size(); g++) {
        Game& G = games_[g];
        if (G.active) continue;
        int L = 0;
        if (!r.fresh) while (L < n && L < (int)G.seq.size() && r.prompt[L] == G.seq[L]) L++;
        const int keep = r.fresh ? 0 : std::max(0, m_.reach(g, std::min(L, n - 1)));
        const int cont = now - G.t_used > stale_s ? 0 : G.continued;
        if (best < 0 || keep > best_keep ||
            (keep == best_keep && (cont < best_cont || (cont == best_cont && G.used < games_[best].used)))) {
            best = g;
            best_l = L;
            best_keep = keep;
            best_cont = cont;
        }
    }
    GLM_CHECK(best >= 0, "no free game");
    Game& G = games_[best];
    m_.select(best);
    const int L = best_l;
    if (r.before) r.before();                    // image encoding: its embeddings go to the selected sequence
    if (r.fresh) G.seq.clear();
    if (r.fresh) { m_.reset(); G.start = 0; }
    else if (L == (int)G.seq.size() && L == m_.position() && L <= n - 1) G.start = L;
    else if (m_.restore(std::min(L, n - 1))) G.start = m_.position();
    else { m_.reset(); G.start = 0; }
    G.seq.resize(G.start);
    G.continued = G.start * 2 >= n ? G.continued + 1 : 0;
    G.active = true;
    G.reading = true;
    G.r = std::move(r);
    G.done = std::move(done);
    G.next = G.start;
    G.res = GenResult();
    G.res.prompt_tokens = n;
    G.res.cached_tokens = G.start;
    G.res.game = best;
    G.res.shared_tokens = L;
    G.t0 = now_s();
    G.t_first = 0;
    G.seed_row = -1;
    G.sm = std::make_unique<Sampler>(G.r.sampling, m_.vocab());
    for (int id : G.r.prompt) G.sm->seen(id);
    G.tok = G.r.prompt[n - 1];
    // the checkpoints of this read: the turn starts still ahead, then the prompt's end, each only when it is more than
    // a short read past the one before (reading mark_gap tokens costs about what a checkpoint's copy does)
    G.marks.clear();
    int last = -1;
    for (int m : G.r.turn_marks) {
        if (m <= G.start || m > n - 1 || (last >= 0 && m - last < mark_gap)) continue;
        G.marks.push_back(m);
        last = m;
    }
    G.end_mark = last < 0 || n - 1 - last >= mark_gap;
    if (G.r.on_progress) G.r.on_progress(G.start, n, G.start);
}

void G4Engine::finish(int g) {
    Game& G = games_[g];
    double t1 = now_s();
    G.res.prompt_s = (G.t_first ? G.t_first : t1) - G.t0;
    G.res.total_s = t1 - G.t0;
    double dec = t1 - (G.t_first ? G.t_first : t1);
    G.res.decode_tok_s = dec > 0 && G.res.completion_tokens > 1 ? (G.res.completion_tokens - 1) / dec : 0;
    G.active = false;
    G.used = ++clock_;
    G.t_used = t1;
    G.sm.reset();
    auto done = std::move(G.done);
    G.done = nullptr;
    done(G.res);
}

int G4Engine::shared_prefix(const std::vector<int>& prompt, bool running) const {
    int best = 0;
    for (auto& G : games_) {
        if (G.active != running) continue;
        // a game still reading: what it holds once its prompt is read
        const std::vector<int>& have = G.active && G.reading ? G.r.prompt : G.seq;
        const int n = (int)std::min(prompt.size(), have.size());
        int L = 0;
        while (L < n && prompt[L] == have[L]) L++;
        best = std::max(best, L);
    }
    return best;
}

void G4Engine::reap() {
    for (int g = 0; g < (int)games_.size(); g++)
        if (games_[g].active && games_[g].r.cancelled && games_[g].r.cancelled->load()) {
            games_[g].res.finish = "cancelled";
            finish(g);
        }
}

void G4Engine::step() {
    auto& q = Dev::get().q;
    const int V = m_.vocab();
    if (!logits_h_) logits_h_ = sycl::malloc_host<half>((size_t)V * g4::QMAX, q);
    reap();

    std::vector<int> act;
    for (int g = 0; g < (int)games_.size(); g++)
        if (games_[g].active && !games_[g].reading) act.push_back(g);

    // a piece of one prompt (the longest-waiting reader first); all of it when nobody is decoding
    for (int g = 0; g < (int)games_.size(); g++) {
        Game& G = games_[g];
        if (!G.active || !G.reading) continue;
        const int n = (int)G.r.prompt.size(), body_end = n - 1;
        int nr = body_end - G.next;
        if (!act.empty()) nr = std::min(nr, piece);
        // a read ends at the next turn start that gets a checkpoint
        int mark = -1;
        for (int m : G.marks)
            if (m > G.next && m <= G.next + nr) { mark = m; nr = m - G.next; break; }
        if (nr > 0) {
            if (G.r.on_progress) G.r.on_progress(G.next, n, G.start);     // this prompt is the one read now
            m_.select(g);
            // checkpoints where the next prompt may branch: at a turn's start (the reply's thought, a prefilled one and
            // a hidden pass's instruction are dropped from it), and at the prompt's very end (a tool call's result
            // continues from there)
            if (mark >= 0) m_.prefill_turn(G.r.prompt.data() + G.next, nr, mark);
            else m_.prefill(G.r.prompt.data() + G.next, nr, G.end_mark && G.next + nr == body_end ? body_end : -1);
            G.seq.insert(G.seq.end(), G.r.prompt.begin() + G.next, G.r.prompt.begin() + G.next + nr);
            G.next += nr;
            if (G.r.on_progress) G.r.on_progress(G.next, n, G.start);
        }
        if (G.next >= body_end) {
            G.reading = false;
            if (G.r.on_progress) G.r.on_progress(n, n, G.start);
            act.push_back(g);
        }
        break;
    }
    if (act.empty()) return;

    // the step's rows: each game's token, then its drafts, sharing the rows left over
    const int R = (int)act.size();
    GLM_CHECK(R <= max_rows, "%d games decoding, %d rows a step", R, max_rows);
    const int extra = max_rows - R;
    std::vector<int> dg, dt, dr, dn;
    std::vector<int> map(R, -1);
    for (int a = 0; a < R; a++) {
        Game& G = games_[act[a]];
        const int room = m_.max_context() - m_.position_of(act[a]) - 1;
        int nd = std::min({m_.draft_tokens, extra / R + (a < extra % R ? 1 : 0), room,
                           G.r.max_new - G.res.completion_tokens - 1});
        if (G.seed_row >= 0 && nd > 0) {
            map[a] = (int)dg.size();
            dg.push_back(act[a]);
            dt.push_back(G.tok);
            dr.push_back(G.seed_row);
            dn.push_back(nd);
        }
    }
    if (!dg.empty()) m_.draft_rows(dg.data(), dt.data(), dr.data(), dn.data(), (int)dg.size(), drafts_);

    std::vector<int> rg, rt, first(R);
    for (int a = 0; a < R; a++) {
        Game& G = games_[act[a]];
        first[a] = (int)rt.size();
        rg.push_back(act[a]);
        rt.push_back(G.tok);
        if (map[a] >= 0)
            for (int d : drafts_[map[a]]) { rg.push_back(act[a]); rt.push_back(d); }
    }
    const int Q = (int)rt.size();
    std::vector<int> p0(R);
    for (int a = 0; a < R; a++) p0[a] = m_.position_of(act[a]);
    const half* lg = m_.forward_rows(rg.data(), rt.data(), Q);
    q.memcpy(logits_h_, lg, (size_t)Q * V * sizeof(half)).wait();

    const double tn = now_s();
    for (int a = 0; a < R; a++) {
        const int g = act[a];
        Game& G = games_[g];
        if (!G.t_first) G.t_first = tn;
        const int rows = (a + 1 < R ? first[a + 1] : Q) - first[a];
        G.res.drafted += rows - 1;
        bool done = false;
        for (int i = 0; i < rows; i++) {
            const int row = first[a] + i;
            G.seq.push_back(rt[row]);
            int next = G.sm->sample(logits_h_ + (size_t)row * V);
            G.sm->seen(next);
            if (std::find(G.r.stop_ids.begin(), G.r.stop_ids.end(), next) != G.r.stop_ids.end()) {
                G.res.finish = "stop";
                G.res.stop_id = next;
                done = true;
            } else {
                G.res.completion_tokens++;
                if (G.r.on_token && !G.r.on_token(next)) { G.res.finish = "stop"; done = true; }
                else if (G.res.completion_tokens >= G.r.max_new) { G.res.finish = "length"; done = true; }
            }
            G.tok = next;
            G.seed_row = row;
            if (!done && i + 1 < rows && next == rt[row + 1]) { G.res.accepted++; continue; }
            if (i + 1 < rows) {                  // the rows behind this one were read for nothing
                m_.select(g);
                m_.restore(p0[a] + i + 1);
            }
            break;
        }
        if (!done && m_.position_of(g) >= m_.max_context()) { G.res.finish = "length"; done = true; }
        if (done) finish(g);
    }
}

}  // namespace glm
