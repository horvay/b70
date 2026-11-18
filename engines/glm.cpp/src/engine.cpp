#include "engine.h"
#include <algorithm>
#include <chrono>

namespace glm {

namespace {
double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
}  // namespace

GenResult Engine::run(const GenRequest& r) {
    auto& q = Dev::get().q;
    const int V = m_.vocab();
    if (!logits_h_) logits_h_ = sycl::malloc_host<half>(V, q);
    GenResult res;
    const int n = (int)r.prompt.size();
    GLM_CHECK(n >= 1, "empty prompt");
    res.prompt_tokens = n;
    double t0 = now_s();

    // where to start: continue, restore a checkpoint inside the shared prefix, or start over. The prompt's last token
    // is always read by forward() (its logits start the reply).
    if (r.before) r.before();
    if (r.fresh) {
        m_.reset();
        seq_.clear();
    }
    int L = 0;
    while (L < n && L < (int)seq_.size() && r.prompt[L] == seq_[L]) L++;
    int start;
    if (L == (int)seq_.size() && L == m_.position() && L <= n - 1) {
        start = L;
    } else if (m_.restore(std::min(L, n - 1))) {
        start = m_.position();
    } else {
        m_.reset();
        start = 0;
    }
    seq_.resize(start);
    res.cached_tokens = start;
    int body = n - 1 - start;
    if (body > 0) {
        int ck = (n - 1) / ckpt_every * ckpt_every;
        if (r.on_progress) r.on_progress(start, n, start);
        m_.prefill(r.prompt.data() + start, body, ck > start ? ck : -1);
        seq_.insert(seq_.end(), r.prompt.begin() + start, r.prompt.end() - 1);
    }
    if (r.on_progress) r.on_progress(n, n, start);

    Sampler sm(r.sampling, V);
    for (int id : r.prompt) sm.seen(id);
    int tok = r.prompt[n - 1];
    double t_first = 0;
    // Each verify step reads tok and the drafts behind it; every reply token is sampled from the target's own logits at
    // its position, so the reply has the target's distribution (a draft only saves work when it equals the sample).
    if ((int)logits_cap_ < 1 + m_.draft_tokens) {
        logits_cap_ = 1 + m_.draft_tokens;
        q.wait();
        sycl::free(logits_h_, q);
        logits_h_ = sycl::malloc_host<half>((size_t)V * logits_cap_, q);
    }
    std::vector<int> batch(1 + m_.draft_tokens);
    int seed_row = -1;                                   // the row of the last forward() that predicted tok
    bool done = false;
    while (!done) {
        if (r.cancelled && r.cancelled->load()) { res.finish = "cancelled"; break; }
        const int p0 = m_.position();
        if (p0 >= m_.max_context()) { res.finish = "length"; break; }
        batch[0] = tok;
        int Q = 1;
        // no point drafting past the context or the reply's end
        int nd = std::min({m_.draft_tokens, m_.max_context() - p0 - 1, r.max_new - res.completion_tokens - 1});
        if (seed_row >= 0 && nd > 0) Q += m_.draft(tok, seed_row, nd, batch.data() + 1);
        const half* lg = m_.forward(batch.data(), Q);
        q.memcpy(logits_h_, lg, (size_t)Q * V * sizeof(half)).wait();
        if (!t_first) t_first = now_s();
        res.drafted += Q - 1;
        for (int i = 0; i < Q; i++) {
            seq_.push_back(batch[i]);
            int next = sm.sample(logits_h_ + (size_t)i * V);
            sm.seen(next);
            bool stop = std::find(r.stop_ids.begin(), r.stop_ids.end(), next) != r.stop_ids.end();
            if (stop) { res.finish = "stop"; res.stop_id = next; done = true; }
            else {
                res.completion_tokens++;
                if (r.on_token && !r.on_token(next)) { res.finish = "stop"; done = true; }
                else if (res.completion_tokens >= r.max_new) { res.finish = "length"; done = true; }
            }
            tok = next;
            seed_row = i;
            bool go_on = !done && i + 1 < Q && next == batch[i + 1];
            if (go_on) { res.accepted++; continue; }
            // the rows behind this one were read for nothing: back to right after it
            if (i + 1 < Q) m_.restore(p0 + i + 1);
            break;
        }
    }
    double t1 = now_s();
    res.prompt_s = (t_first ? t_first : t1) - t0;
    res.total_s = t1 - t0;
    double dec = t1 - (t_first ? t_first : t1);
    res.decode_tok_s = dec > 0 && res.completion_tokens > 1 ? (res.completion_tokens - 1) / dec : 0;
    return res;
}

}  // namespace glm
