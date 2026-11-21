#include "g4.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <sys/mman.h>
#include <thread>
#include "g4_ops.h"
#include "json.h"
#include "ops.h"
#include "progress.h"
#include "sampler.h"

namespace glm {

using namespace g4;

namespace {
const std::string P = "model.language_model.";
std::string lp(int li) { return P + "layers." + std::to_string(li) + "."; }
double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int env_i(const char* n, int d) { return getenv(n) ? atoi(getenv(n)) : d; }
}  // namespace

void G4Cache::init(int TS, int S) {
    for (int li = 0; li < L; li++) {
        if (is_full(li)) {
            p.push_back(Tensor::empty({(int64_t)S * TS, FKV, FD}, DType::F16));
            kr.push_back(Tensor::empty({(int64_t)S * TS, FKV, 2 * FROT}, DType::F16));
            r.push_back(Tensor::empty({(int64_t)S * TS, FKV}, DType::F32));
        } else {
            k.push_back(Tensor::empty({(int64_t)S * RING, SKV, SD}, DType::F16));
            v.push_back(Tensor::empty({(int64_t)S * RING, SKV, SD}, DType::F16));
        }
    }
}

// ------------------------------------------------------------------------------------------------ load

G4Model::G4Model(const std::string& dir, int max_tok, const std::string& drafter, int games) : max_tokens(max_tok), games_(games) {
    GLM_CHECK(games >= 1, "games: %d", games);
    // a whole Gemma layer pass at 5 bpw, by rows: 8: 50 ms, 32: 75, 64: 287, 128: 542 (DPAS); 256: 322 (GEMM, flat from 33)
    exl3_large_above = env_i("GLM_EXL3_LARGE", 32);
    pos_.assign(games, 0);
    ckpts_.resize(games);
    turn_ckpts = std::max(0, env_i("G4_TURN_CKPTS", 2));
    TS = max_tokens + 32;                       // a block of keys past the context: the kernels' block loads may read it
    Json c = Json::parse(read_file(dir + "/config.json"));
    GLM_CHECK(c.get_s("model_type", "") == "gemma4", "%s: not a gemma4 model", dir.c_str());
    const Json& tc = c["text_config"];
    GLM_CHECK(tc.get_i("hidden_size", 0) == D && tc.get_i("num_hidden_layers", 0) == L &&
                  tc.get_i("intermediate_size", 0) == I && tc.get_i("vocab_size", 0) == V &&
                  tc.get_i("sliding_window", 0) == WIN,
              "%s: not the Gemma 4 31B shape this engine is built for", dir.c_str());
    for (int li = 0; li < L; li++)
        GLM_CHECK((tc["layer_types"][li].s() == "full_attention") == is_full(li), "layer %d: type", li);
    st = std::make_unique<SafeTensors>(dir);
    auto& q = Dev::get().q;
    // what fits: the weights are known from the files; each game's context and the prompt scratch share the rest (a bigger
    // super-chunk reads prompts a little faster, the context comes first)
    {
        const double GiB = 1073741824.0;
        double weights = 0;
        for (auto& n : st->names_with_prefix(P + "layers.")) weights += st->at(n).bytes;
        for (auto& n : st->names_with_prefix("lm_head")) weights += st->at(n).bytes;
        const double total = (double)Dev::get().dev.get_info<sycl::info::device::global_mem_size>();
        const double reserve = (getenv("G4_VRAM_RESERVE_GIB") ? atof(getenv("G4_VRAM_RESERVE_GIB")) : 1.0) * GiB;   // driver, graphs, oneMKL
        const double draft_b = drafter.empty() ? 0 : 1.05 * GiB, rows_b = 0.25 * GiB;
        const double ring_b = 100.0 * RING * SKV * SD * 2 * games_;       // K and V of 50 layers
        const double tok_b = 10.0 * FKV * (FD * 2 + 2 * FROT * 2 + 4) * games_ + FROT * 8;   // a context token of all games, its RoPE
        double avail = total - weights - reserve - draft_b - rows_b - ring_b;
        auto scratch = [](int sc) { return (sc * 266e3 + 0.05 * 1073741824.0); };   // ~2.2 GB at 8192
        int want_sc = env_i("GLM_PREFILL_SC", 0);
        double need = (double)max_tokens * tok_b;
        if (want_sc > 0) super_chunk = want_sc;
        else {
            super_chunk = 2048;
            for (int sc : {8192, 4096})
                if (avail - scratch(sc) >= need) { super_chunk = sc; break; }
        }
        int fit = (int)((avail - scratch(super_chunk)) / tok_b);
        GLM_CHECK(fit >= 1024, "%d games do not fit: %.1f GiB of weights, %.1f GiB left for the caches", games_, weights / GiB, avail / GiB);
        if (fit < max_tokens) {
            fprintf(stderr, "context: %d tokens do not fit %d game(s) on the card; %d a game\n", max_tokens, games_, fit);
            max_tokens = fit;
            TS = max_tokens + 32;
        }
        fprintf(stderr, "context %d x %d game(s), prompt scratch for %d rows\n", max_tokens, games_, super_chunk);
    }
    q_chunk = std::min(env_i("GLM_PREFILL_QC", q_chunk), super_chunk);
    key_block = env_i("GLM_PREFILL_KB", key_block);
    graphs_on = env_i("G4_GRAPHS", 1) != 0 && !g_prof_points && !g_sync_points;

    const StTensor& et = st->at(P + "embed_tokens.weight");
    GLM_CHECK(et.dt == DType::BF16 && et.shape[0] == V && et.shape[1] == D, "embedding: expected bf16 [V, D]");
    embed = static_cast<const bf16*>(et.data);
    // its pages in now (it is read a few rows a token, each new row a page fault in the decode loop: ~2 ms an 8-game
    // step), beside the weights' reading; the weights are read in one parallel pass after the layers (SafeTensors::flush)
    std::thread embed_in([p = et.data, n = et.bytes] { SafeTensors::populate(p, n, 4); });
    st->defer(true);
    head_exl3 = Exl3Linear::present(*st, "lm_head");
    if (head_exl3) head_q = Exl3Linear::load(*st, "lm_head");
    else {
        head16 = st->load("lm_head.weight");
        GLM_CHECK(head16.dt == DType::F16 && head16.size(0) == V && head16.size(1) == D, "lm_head: expected fp16 [V, D]");
    }
    w_norm = st->load(P + "norm.weight");

    layers.resize(L);
    size_t lw = 0;
    auto need = [&](const Exl3Linear& l) { lw = std::max({lw, l.ws_bytes(super_chunk), l.ws_bytes(QMAX), l.ws_bytes(1)}); };
    for (int li = 0; li < L; li++) {
        G4Layer& b = layers[li];
        std::string p = lp(li), ap = p + "self_attn.";
        b.full = is_full(li);
        b.q = Exl3Linear::load(*st, ap + "q_proj");
        b.k = Exl3Linear::load(*st, ap + "k_proj");
        if (!b.full) b.v = Exl3Linear::load(*st, ap + "v_proj");
        b.o = Exl3Linear::load(*st, ap + "o_proj");
        b.g = Exl3Linear::load(*st, p + "mlp.gate_proj");
        b.u = Exl3Linear::load(*st, p + "mlp.up_proj");
        b.d = Exl3Linear::load(*st, p + "mlp.down_proj");
        int hd = b.full ? FD : SD, kv = b.full ? FKV : SKV;
        GLM_CHECK(b.q.out == NH * hd && b.k.out == kv * hd && b.o.in == NH * hd && b.g.out == I && b.d.in == I,
                  "%s: shapes", p.c_str());
        b.w_in = st->load(p + "input_layernorm.weight");
        b.w_pa = st->load(p + "post_attention_layernorm.weight");
        b.w_pf = st->load(p + "pre_feedforward_layernorm.weight");
        b.w_ff = st->load(p + "post_feedforward_layernorm.weight");
        b.qn = st->load(ap + "q_norm.weight");
        b.kn = st->load(ap + "k_norm.weight");
        if (b.full) {
            auto& t = st->at(ap + "k_norm.weight");
            std::vector<half> h(FD);
            for (int d = 0; d < FD; d++) h[d] = half(static_cast<float>(static_cast<const bf16*>(t.data)[d]));
            b.kn16 = Tensor::empty({FD}, DType::F16);
            upload(b.kn16.p, h.data(), FD * 2);
        }
        const StTensor& ls = st->at(p + "layer_scalar");
        b.scalar = static_cast<float>(*static_cast<const bf16*>(ls.data));
        need(b.q); need(b.k); need(b.o); need(b.g); need(b.u); need(b.d);
        if (!b.full) need(b.v);
        load_progress("loaded layer %d/%d (%.2f GiB VRAM)", li + 1, L, Dev::get().vram_used / 1073741824.0);
    }
    load_progress_end();
    st->flush();
    st->defer(false);
    embed_in.join();
    if (head_exl3) need(head_q);
    // experiment: G4_RAM_GIB of trellis in pinned host memory, the kernels reading it in place (from the last layer down)
    if (getenv("G4_RAM_GIB")) {
        double left = atof(getenv("G4_RAM_GIB")) * 1073741824.0;
        size_t moved = 0;
        auto move = [&](Exl3Linear& l) {
            if (left <= 0) return;
            size_t n = l.trellis.bytes();
            void* h = sycl::malloc_host(n, q);
            GLM_CHECK(h, "G4_RAM_GIB: pinned allocation of %zu bytes failed", n);
            q.memcpy(h, l.trellis.p, n).wait();
            Tensor t = Tensor::view(h, l.trellis.shape, l.trellis.dt);
            l.trellis = std::move(t);
            left -= n;
            moved += n;
        };
        for (int li = L - 1; li >= 0 && left > 0; li--) {
            G4Layer& b = layers[li];
            move(b.d); move(b.g); move(b.u); move(b.o); move(b.q); move(b.k);
            if (!b.full) move(b.v);
        }
        fprintf(stderr, "G4_RAM_GIB: %.2f GiB of trellis in RAM\n", moved / 1073741824.0);
    }
    lin.init(lw + ((size_t)1 << 20));

    // the full layers' RoPE at every position: angle in fp32 (pos * inv_freq, as HF and exllamav3 compute it), cos / sin
    {
        rope_full = Tensor::empty({max_tokens, FROT, 2}, DType::F32);
        std::vector<float> inv(FROT), h((size_t)max_tokens * FROT * 2);
        for (int j = 0; j < FROT; j++) inv[j] = (float)std::pow(1000000.0, -2.0 * j / (double)FD);
        for (int t = 0; t < max_tokens; t++)
            for (int j = 0; j < FROT; j++) {
                float th = (float)t * inv[j];
                h[((size_t)t * FROT + j) * 2] = (float)std::cos((double)th);
                h[((size_t)t * FROT + j) * 2 + 1] = (float)std::sin((double)th);
            }
        upload(rope_full.p, h.data(), h.size() * 4);
    }
    cache.init(TS, games_);

    // decode rows
    X = Tensor::empty({QMAX, D}, DType::F32);
    h16 = Tensor::empty({QMAX, D}, DType::F16);
    q16 = Tensor::empty({QMAX, NH * FD}, DType::F16);
    k16 = Tensor::empty({QMAX, SKV * SD}, DType::F16);
    v16 = Tensor::empty({QMAX, SKV * SD}, DType::F16);
    o16 = Tensor::empty({QMAX, NH * FD}, DType::F16);
    a16 = Tensor::empty({QMAX, D}, DType::F16);
    g16 = Tensor::empty({QMAX, I}, DType::F16);
    u16 = Tensor::empty({QMAX, I}, DType::F16);
    d32 = Tensor::empty({QMAX, D}, DType::F32);
    fin16 = Tensor::empty({QMAX, D}, DType::F16);
    logits = Tensor::empty({QMAX, V}, DType::F16);
    part = Tensor::empty({(int64_t)g4ops::attn_part_floats(QMAX)}, DType::F32);
    srope_d = Tensor::empty({QMAX, SD / 2, 2}, DType::F32);
    pos_dev = Tensor::zeros({2 * QMAX}, DType::I32);
    pos_h = sycl::malloc_host<int>(2 * QMAX, q);

    // a prompt's super-chunk: kp / vp hold the window's WIN - 1 rows in front of the rows, or a full layer's key block
    const int64_t SC = super_chunk;
    Xp = Tensor::empty({SC, D}, DType::F32);
    srope_p = Tensor::empty({SC, SD / 2, 2}, DType::F32);
    hp = Tensor::empty({SC, D}, DType::F16);                // also o_proj's output
    qp = Tensor::empty({SC, NH * FD}, DType::F16);          // also attention's output and down_proj's (fp32 [SC, D])
    int64_t kvrows = std::max<int64_t>(WIN - 1 + SC, (int64_t)key_block * FKV * FD / (SKV * SD));
    kp = Tensor::empty({kvrows, SKV * SD}, DType::F16);
    vp = Tensor::empty({kvrows, SKV * SD}, DType::F16);
    gp = Tensor::empty({SC, I}, DType::F16);
    up = Tensor::empty({SC, I}, DType::F16);
    fo32 = Tensor::empty({q_chunk, NH * FD}, DType::F32);
    flse = Tensor::empty({q_chunk, NH}, DType::F32);
    x_host = sycl::malloc_host<float>((size_t)std::max(super_chunk, QMAX) * D, q);
    GLM_CHECK(x_host && pos_h, "pinned upload buffers");
    q.wait();
    fprintf(stderr, "buffers: %.2f GiB VRAM, Exl3 workspace %.2f GiB\n", Dev::get().vram_used / 1073741824.0,
            lw / 1073741824.0);

    if (!drafter.empty()) load_drafter(drafter);

    // every kernel once (modules load on first use): a short prompt and a decode step
    if (env_i("G4_WARMUP", 1) != 0) {
        double tw = now_s();
        std::vector<int> t(std::min(300, max_tokens - 1), 236761);
        prefill(t.data(), (int)t.size());
        forward(t.data(), 1);
        q.wait();
        // every row count's graph and kernels now, not on a request's time (20-30 ms each at first use)
        for (int Q = 2; Q <= 8; Q++) {
            restore(position() - 1);
            std::vector<int> vr(Q, t[0]);
            forward(vr.data(), Q);
            q.wait();
            restore(position() - Q + 1);
        }
        if (!dlayers.empty()) {
            std::vector<std::vector<int>> o;
            for (int R = 1; R <= std::min(games_, 8); R++) {
                std::vector<int> dg(R, 0), dt(R, t[0]), dr(R, 0), dn(R, 1);
                draft_rows(dg.data(), dt.data(), dr.data(), dn.data(), R, o);
            }
        }
        reset();
        fprintf(stderr, "warm-up %.1f s\n", now_s() - tw);
    }
    fprintf(stderr, "model ready: %.2f GiB VRAM\n", Dev::get().vram_used / 1073741824.0);
}

G4Model::~G4Model() {
    auto& q = Dev::get().q;
    q.wait();
    for (auto& g : ckpts_)
        for (auto& c : g) {
            for (void* p : c.k) if (p) sycl::free(p, q);
            for (void* p : c.v) if (p) sycl::free(p, q);
        }
    graphs.clear();
    dgraphs.clear();
    if (dz_h) sycl::free(dz_h, q);
    if (dpos_h) sycl::free(dpos_h, q);
    if (dlog_h) sycl::free(dlog_h, q);
    if (pos_h) sycl::free(pos_h, q);
    if (x_host) sycl::free(x_host, q);
}

// ------------------------------------------------------------------------------------------------ one layer

// E[id] * bf16(sqrt(5376)) rounded to bf16, widened to fp32
void G4Model::embed_host(const int* tokens, int n, float* dst) {
    const bf16 mul(EMBED_MUL);
    for (int t = 0; t < n; t++) {
        const bf16* e = embed + (size_t)tokens[t] * D;
        float* o = dst + (size_t)t * D;
        for (int d = 0; d < D; d++) o[d] = static_cast<float>(bf16(static_cast<float>(e[d]) * static_cast<float>(mul)));
    }
}

// x: rows' stream; h: their input-norm output (written with the next layer's on the way out; the last layer's norm is
// the final norm, in fin when set)
void G4Model::decode_layer(int li, float* x, half* h, int Q, const int* pdev, const int* sdev, half* fin) {
    auto& q = Dev::get().q;
    G4Layer& b = layers[li];
    half* q_ = q16.ptr<half>();
    half* o_ = o16.ptr<half>();
    b.q.forward(q, lin, h, Q, q_, DType::F16);
    b.k.forward(q, lin, h, Q, k16.p, DType::F16);
    if (b.full) {
        int fi = li / 6;
        g4ops::full_post(q, q_, k16.ptr<half>(), b.qn.ptr<bf16>(), b.kn.ptr<bf16>(), rope_full.ptr<float>(), Q, pdev, 0,
                         cache.p[fi].ptr<half>(), cache.kr[fi].ptr<half>(), cache.r[fi].ptr<float>(), sdev, TS);
        g4ops::attn_full_dec(q, q_, cache.p[fi].ptr<half>(), cache.kr[fi].ptr<half>(), cache.r[fi].ptr<float>(),
                             b.kn16.ptr<half>(), Q, pdev, sdev, TS, 0, part.ptr<float>(), o_);
    } else {
        b.v.forward(q, lin, h, Q, v16.p, DType::F16);
        int si = li - li / 6 - (li % 6 == 5);
        g4ops::sliding_post(q, q_, k16.ptr<half>(), v16.ptr<half>(), b.qn.ptr<bf16>(), b.kn.ptr<bf16>(),
                            srope_d.ptr<float>(), Q, pdev, 0, cache.k[si].ptr<half>(), cache.v[si].ptr<half>(), RING, sdev);
        g4ops::attn_sliding_dec(q, q_, cache.k[si].ptr<half>(), cache.v[si].ptr<half>(), Q, pdev, sdev, 0, part.ptr<float>(), o_);
    }
    prof_point("attention");
    b.o.forward(q, lin, o_, Q, a16.p, DType::F16);
    g4ops::add_norm<half>(q, x, a16.ptr<half>(), b.w_pa.ptr<bf16>(), 1.0f, b.w_pf.ptr<bf16>(), h, Q);
    prof_point("o_proj + norms");
    b.g.forward(q, lin, h, Q, g16.p, DType::F16);
    b.u.forward(q, lin, h, Q, u16.p, DType::F16);
    g4ops::gelu_mul(q, g16.ptr<half>(), u16.ptr<half>(), g16.ptr<half>(), (size_t)Q * I);
    prof_point("gate, up");
    b.d.forward(q, lin, g16.ptr<half>(), Q, d32.p, DType::F32);
    const bf16* wn = li + 1 < L ? layers[li + 1].w_in.ptr<bf16>() : w_norm.ptr<bf16>();
    g4ops::add_norm<float>(q, x, d32.ptr<float>(), b.w_ff.ptr<bf16>(), b.scalar, wn, li + 1 < L ? h : fin, Q);
    prof_point("down + norms");
}

void G4Model::forward_dev(int Q) {
    auto& q = Dev::get().q;
    float* x = X.ptr<float>();
    const int* pdev = pos_dev.ptr<int>();
    const int* sdev = pdev + QMAX;
    q.memcpy(x, x_host, (size_t)Q * D * 4);
    q.memcpy(pos_dev.p, pos_h, (size_t)2 * QMAX * sizeof(int));
    g4ops::rope_rows(q, Q, pdev, 0, srope_d.ptr<float>());
    g4ops::norm16(q, x, layers[0].w_in.ptr<bf16>(), h16.ptr<half>(), Q);
    for (int li = 0; li < L; li++) decode_layer(li, x, h16.ptr<half>(), Q, pdev, sdev, fin16.ptr<half>());
    if (head_exl3) {
        head_q.forward(q, lin, fin16.ptr<half>(), Q, logits.p, DType::F16);
        g4ops::softcap(q, logits.ptr<half>(), (size_t)Q * V);
    } else {
        if (Q <= 8) g4fa::dense16(q, fin16.ptr<half>(), Q, D, head16.ptr<half>(), D, V, logits.ptr<half>(), true);
        else g4ops::head_logits(q, fin16.ptr<half>(), head16.ptr<half>(), Q, logits.ptr<half>());
    }
    prof_point("head");
}

void G4Model::select(int game) {
    GLM_CHECK(game >= 0 && game < games_, "game %d of %d", game, games_);
    pos_[cur_] = pos;
    cur_ = game;
    pos = pos_[game];
}

const half* G4Model::forward(const int* tokens, int Q) {
    std::vector<int> g(Q, cur_);
    return forward_rows(g.data(), tokens, Q);
}

const half* G4Model::forward_rows(const int* game, const int* tokens, int Q) {
    GLM_CHECK(Q >= 1 && Q <= QMAX, "forward: %d rows (at most %d)", Q, QMAX);
    auto& q = Dev::get().q;
    dec_ev.wait();
    x_ev.wait();
    embed_host(tokens, Q, x_host);
    pos_[cur_] = pos;
    for (int r = 0; r < Q; r++) {
        int& pg = pos_[game[r]];
        GLM_CHECK(pg < max_tokens, "context full (%d tokens)", max_tokens);
        pos_h[r] = pg++;
        pos_h[QMAX + r] = game[r];
    }
    pos = pos_[cur_];
    if (graphs_on) {
        namespace ex = sycl::ext::oneapi::experimental;
        if ((int)graphs.size() <= Q) graphs.resize(Q + 1);
        if (!graphs[Q]) {
            ex::command_graph<ex::graph_state::modifiable> g{q.get_context(), q.get_device()};
            g.begin_recording(q);
            forward_dev(Q);
            g.end_recording(q);
            graphs[Q] = std::make_unique<ex::command_graph<ex::graph_state::executable>>(g.finalize());
        }
        dec_ev = q.ext_oneapi_graph(*graphs[Q]);
    } else {
        forward_dev(Q);
    }
    return logits.ptr<half>();
}

// ------------------------------------------------------------------------------------------------ prompts

void G4Model::prefill_sc(const int* tokens, int n) {
    auto& q = Dev::get().q;
    const int p0 = pos;
    x_ev.wait();
    embed_host(tokens, n, x_host);
    float* x = Xp.ptr<float>();
    half* h = hp.ptr<half>();
    half* qr = qp.ptr<half>();
    x_ev = q.memcpy(x, x_host, (size_t)n * D * 4);
    g4ops::rope_rows(q, n, nullptr, p0, srope_p.ptr<float>());
    g4ops::norm16(q, x, layers[0].w_in.ptr<bf16>(), h, n);
    const size_t skv = (size_t)SKV * SD;
    const size_t grow = (size_t)cur_ * TS, gring = (size_t)cur_ * RING * skv;     // this game's part of the caches
    for (int li = 0; li < L; li++) {
        G4Layer& b = layers[li];
        prof_point("(entry)");
        b.q.forward(q, lin, h, n, qr, DType::F16);
        if (b.full) {
            int fi = li / 6;
            half* Pc = cache.p[fi].ptr<half>() + grow * FKV * FD;
            float* Rc = cache.r[fi].ptr<float>() + grow * FKV;
            b.k.forward(q, lin, h, n, kp.p, DType::F16);
            prof_point("q, k");
            g4ops::full_post(q, qr, kp.ptr<half>(), b.qn.ptr<bf16>(), b.kn.ptr<bf16>(), rope_full.ptr<float>(), n, nullptr, p0,
                             Pc, cache.kr[fi].ptr<half>() + grow * FKV * 2 * FROT, Rc);
            // query rows in pieces (the fp32 accumulators), each over the keys it sees, a block of rebuilt keys at a time
            for (int q0 = 0; q0 < n; q0 += q_chunk) {
                int qn = std::min(q_chunk, n - q0);
                g4ops::fill(q, fo32.ptr<float>(), 0.0f, (size_t)qn * NH * FD);
                g4ops::fill(q, flse.ptr<float>(), -INFINITY, (size_t)qn * NH);
                int kend = p0 + q0 + qn;
                for (int k0 = 0; k0 < kend; k0 += key_block) {
                    int nk = std::min(key_block, kend - k0);
                    g4ops::full_kv(q, Pc, Rc, b.kn.ptr<bf16>(), rope_full.ptr<float>(), k0, nk, kp.ptr<half>(), vp.ptr<half>());
                    g4fa::full_block(q, qr + (size_t)q0 * NH * FD, kp.ptr<half>(), vp.ptr<half>(), qn, nk, p0 + q0, k0,
                                     fo32.ptr<float>(), flse.ptr<float>());
                }
                g4ops::f32_to_f16(q, fo32.ptr<float>(), qr + (size_t)q0 * NH * FD, (size_t)qn * NH * FD);
            }
        } else {
            int si = li - li / 6 - (li % 6 == 5);
            // the rows' K / V in place behind the window's rows
            half* kr = kp.ptr<half>() + (size_t)(WIN - 1) * skv;
            half* vr = vp.ptr<half>() + (size_t)(WIN - 1) * skv;
            b.k.forward(q, lin, h, n, kr, DType::F16);
            b.v.forward(q, lin, h, n, vr, DType::F16);
            prof_point("q, k, v");
            g4ops::sliding_post(q, qr, kr, vr, b.qn.ptr<bf16>(), b.kn.ptr<bf16>(), srope_p.ptr<float>(), n, nullptr, p0, kr,
                                vr, 0);
            if (capture_ && si + 1 == (int)cache.k.size()) {
                GLM_CHECK(p0 + n <= capture_->rows, "capture: %d rows", capture_->rows);
                q.memcpy(capture_->sk + (size_t)p0 * skv, kr, (size_t)n * skv * 2);
                q.memcpy(capture_->sv + (size_t)p0 * skv, vr, (size_t)n * skv * 2);
            }
            int prior = std::min(p0, WIN - 1);
            half* kw = kr - (size_t)prior * skv;
            half* vw = vr - (size_t)prior * skv;
            half* Kg = cache.k[si].ptr<half>() + gring;
            half* Vg = cache.v[si].ptr<half>() + gring;
            g4ops::ring_gather(q, Kg, p0 - prior, prior, kw);
            g4ops::ring_gather(q, Vg, p0 - prior, prior, vw);
            for (int q0 = 0; q0 < n; q0 += q_chunk) {
                int qn = std::min(q_chunk, n - q0);
                // keys up to the last row of this piece; the piece's output replaces its queries
                g4fa::sliding(q, qr + (size_t)q0 * NH * SD, kw, vw, qn, prior + q0 + qn, p0 + q0, p0 - prior, WIN,
                              qr + (size_t)q0 * NH * SD);
            }
            // the window for what follows
            int keep = std::min(n, RING);
            g4ops::ring_scatter(q, kr + (size_t)(n - keep) * skv, p0 + n - keep, keep, Kg);
            g4ops::ring_scatter(q, vr + (size_t)(n - keep) * skv, p0 + n - keep, keep, Vg);
        }
        prof_point("attention");
        b.o.forward(q, lin, qr, n, h, DType::F16);
        g4ops::add_norm<half>(q, x, h, b.w_pa.ptr<bf16>(), 1.0f, b.w_pf.ptr<bf16>(), h, n);
        prof_point("o_proj + norms");
        b.g.forward(q, lin, h, n, gp.p, DType::F16);
        b.u.forward(q, lin, h, n, up.p, DType::F16);
        g4ops::gelu_mul(q, gp.ptr<half>(), up.ptr<half>(), gp.ptr<half>(), (size_t)n * I);
        prof_point("gate, up");
        float* d = qp.ptr<float>();     // [SC, D] fp32: the query rows are done with
        b.d.forward(q, lin, gp.ptr<half>(), n, d, DType::F32);
        const bf16* wn = li + 1 < L ? layers[li + 1].w_in.ptr<bf16>() : capture_ ? w_norm.ptr<bf16>() : nullptr;
        g4ops::add_norm<float>(q, x, d, b.w_ff.ptr<bf16>(), b.scalar, wn, h, n);
        if (capture_ && li + 1 == L) q.memcpy(capture_->fin + (size_t)p0 * D, h, (size_t)n * D * 2);
        prof_point("down + norms");
        if (n >= 256) read_marks().mark((double)n / L);     // (a short pass: one mark, below)
    }
    if (n < 256) read_marks().mark(n);
    pos += n;
}

void G4Model::prefill(const int* tokens, int n, int ckpt_at) {
    if (n <= 0) return;
    int cut = ckpt_at > pos && ckpt_at <= pos + n ? ckpt_at - pos : -1;
    if (cut > 0 && cut < n) {
        prefill(tokens, cut, ckpt_at);
        prefill(tokens + cut, n - cut, -1);
        return;
    }
    GLM_CHECK(pos + n <= max_tokens, "context full (%d tokens)", max_tokens);
    auto& q = Dev::get().q;
    dec_ev.wait();
    // up to 96 rows are cheapest as DPAS pieces of 32 (64 rows: 150 ms against 287), more on one GEMM pass per super-chunk
    const int chunk = n <= 96 ? 32 : super_chunk;
    for (int s0 = 0; s0 < n; s0 += chunk) prefill_sc(tokens + s0, std::min(chunk, n - s0));
    q.wait();
    if (cut == n) save_checkpoint(pos, ckpt_turn_);
}

void G4Model::prefill_turn(const int* tokens, int n, int ckpt_at) {
    ckpt_turn_ = true;
    try {
        prefill(tokens, n, ckpt_at);
    } catch (...) {
        ckpt_turn_ = false;
        throw;
    }
    ckpt_turn_ = false;
}

void G4Model::last_full_kv(int k0, int n, half* K, half* V_) {
    const int fi = (int)cache.p.size() - 1;
    const size_t grow = (size_t)cur_ * TS;
    g4ops::full_kv(Dev::get().q, cache.p[fi].ptr<half>() + grow * FKV * FD, cache.r[fi].ptr<float>() + grow * FKV,
                   layers[L - 1].kn.ptr<bf16>(), rope_full.ptr<float>(), k0, n, K, V_);
}

void G4Model::head_rows(const half* fin, int R, half* out) {
    auto& q = Dev::get().q;
    for (int r0 = 0; r0 < R; r0 += QMAX) {
        const int n = std::min(QMAX, R - r0);
        if (head_exl3) {
            head_q.forward(q, lin, fin + (size_t)r0 * D, n, out + (size_t)r0 * V, DType::F16);
            g4ops::softcap(q, out + (size_t)r0 * V, (size_t)n * V);
        } else {
            g4ops::head_logits(q, fin + (size_t)r0 * D, head16.ptr<half>(), n, out + (size_t)r0 * V);
        }
    }
}

// ------------------------------------------------------------------------------------------------ drafter

namespace {
Tensor bf16_to_f16(const SafeTensors& s, const std::string& name) {
    const StTensor& t = s.at(name);
    GLM_CHECK(t.dt == DType::BF16, "%s: expected bf16", name.c_str());
    size_t n = t.bytes / 2;
    std::vector<half> h(n);
    const bf16* b = static_cast<const bf16*>(t.data);
    for (size_t i = 0; i < n; i++) h[i] = half(static_cast<float>(b[i]));
    Tensor o = Tensor::empty(t.shape, DType::F16);
    upload(o.p, h.data(), n * 2);
    return o;
}
constexpr int DH = 1024, DI = 8192, DQ = 8;     // the drafter's hidden, MLP; chains at once
}  // namespace

void G4Model::load_drafter(const std::string& dir) {
    Json c = Json::parse(read_file(dir + "/config.json"));
    GLM_CHECK(c.get_s("model_type", "") == "gemma4_assistant" && c.get_i("backbone_hidden_size", 0) == D,
              "%s: not the assistant of this model", dir.c_str());
    GLM_CHECK(!c.get_b("use_ordered_embeddings", false), "%s: ordered embeddings are not supported", dir.c_str());
    const Json& tc = c["text_config"];
    const int nl = (int)tc.get_i("num_hidden_layers", 0);
    GLM_CHECK(tc.get_i("hidden_size", 0) == DH && tc.get_i("intermediate_size", 0) == DI, "%s: shape", dir.c_str());
    dst = std::make_unique<SafeTensors>(dir);
    dpre = bf16_to_f16(*dst, "pre_projection.weight");
    dpost = bf16_to_f16(*dst, "post_projection.weight");
    dhead = bf16_to_f16(*dst, "model.embed_tokens.weight");
    dw_norm = dst->load("model.norm.weight");
    dlayers.resize(nl);
    for (int li = 0; li < nl; li++) {
        G4DraftLayer& b = dlayers[li];
        std::string p = "model.layers." + std::to_string(li) + ".", ap = p + "self_attn.";
        b.full = tc["layer_types"][li].s() == "full_attention";
        b.q = bf16_to_f16(*dst, ap + "q_proj.weight");
        b.o = bf16_to_f16(*dst, ap + "o_proj.weight");
        b.g = bf16_to_f16(*dst, p + "mlp.gate_proj.weight");
        b.u = bf16_to_f16(*dst, p + "mlp.up_proj.weight");
        b.d = bf16_to_f16(*dst, p + "mlp.down_proj.weight");
        GLM_CHECK(b.q.size(0) == NH * (b.full ? FD : SD), "%s: q_proj shape", ap.c_str());
        b.w_in = dst->load(p + "input_layernorm.weight");
        b.w_pa = dst->load(p + "post_attention_layernorm.weight");
        b.w_pf = dst->load(p + "pre_feedforward_layernorm.weight");
        b.w_ff = dst->load(p + "post_feedforward_layernorm.weight");
        b.qn = dst->load(ap + "q_norm.weight");
        b.scalar = static_cast<float>(*static_cast<const bf16*>(dst->at(p + "layer_scalar").data));
    }
    auto& q = Dev::get().q;
    dz = Tensor::empty({DQ, 2 * D}, DType::F16);
    de = Tensor::empty({DQ, D}, DType::F16);
    dx = Tensor::empty({DQ, DH}, DType::F32);
    dh = Tensor::empty({DQ, DH}, DType::F16);
    dq = Tensor::empty({DQ, NH * FD}, DType::F16);
    dok = Tensor::empty({DQ, NH * FD}, DType::F16);
    dg = Tensor::empty({DQ, DI}, DType::F16);
    du = Tensor::empty({DQ, DI}, DType::F16);
    dd = Tensor::empty({DQ, DH}, DType::F32);
    dy = Tensor::empty({DQ, DH}, DType::F16);
    dprev = Tensor::empty({DQ, D}, DType::F16);
    dlog = Tensor::empty({DQ, V}, DType::F16);
    dpos = Tensor::zeros({3, DQ}, DType::I32);
    dsrope = Tensor::empty({DQ, SD / 2, 2}, DType::F32);
    dz_h = sycl::malloc_host<half>((size_t)DQ * D, q);
    dpos_h = sycl::malloc_host<int>(3 * DQ, q);
    dlog_h = sycl::malloc_host<half>((size_t)DQ * V, q);
    GLM_CHECK(dz_h && dpos_h && dlog_h, "drafter: pinned buffers");
    draft_tokens = env_i("G4_DRAFT", 4);
    if (getenv("G4_DRAFT_MARGIN")) draft_margin = (float)atof(getenv("G4_DRAFT_MARGIN"));
    GLM_CHECK(draft_tokens + 1 <= 8, "G4_DRAFT: at most 7");
    fprintf(stderr, "drafter %s: %d layers, %d tokens a step (%.2f GiB VRAM)\n", dir.c_str(), nl, draft_tokens,
            Dev::get().vram_used / 1073741824.0);
}

// one step's kernels for R chains, recorded once per R: the tokens' embeddings and the rows' positions come from the pinned
// buffers, h from dprev
void G4Model::draft_dev(int R) {
    auto& q = Dev::get().q;
    const int* pq = dpos.ptr<int>();       // the query heads' RoPE
    const int* pk = pq + DQ;               // the last key each row sees
    const int* sd = pq + 2 * DQ;           // the row's game
    half* z = dz.ptr<half>();
    q.memcpy(dpos.p, dpos_h, (size_t)3 * DQ * sizeof(int));
    q.memcpy(de.p, dz_h, (size_t)R * D * 2);
    g4ops::concat(q, de.ptr<half>(), dprev.ptr<half>(), z, R, D);
    g4ops::rope_rows(q, R, pq, 0, dsrope.ptr<float>());
    float* x = dx.ptr<float>();
    g4fa::dense32(q, z, R, 2 * D, dpre.ptr<half>(), 2 * D, DH, x);
    // the target's last layer of each type
    const int fi = cache.p.size() - 1, si = cache.k.size() - 1;
    const half* kn = layers[L - 1].kn16.ptr<half>();
    float* part_f = part.ptr<float>();
    half* o_ = dok.ptr<half>();
    for (size_t li = 0; li < dlayers.size(); li++) {
        G4DraftLayer& b = dlayers[li];
        const int hd = b.full ? FD : SD;
        g4ops::norm16(q, x, b.w_in.ptr<bf16>(), dh.ptr<half>(), R, DH);
        g4fa::dense16(q, dh.ptr<half>(), R, DH, b.q.ptr<half>(), DH, NH * hd, dq.ptr<half>());
        if (b.full) {
            g4ops::full_q(q, dq.ptr<half>(), b.qn.ptr<bf16>(), rope_full.ptr<float>(), R, pq);
            g4ops::attn_full_dec(q, dq.ptr<half>(), cache.p[fi].ptr<half>(), cache.kr[fi].ptr<half>(), cache.r[fi].ptr<float>(),
                                 kn, R, pk, sd, TS, 0, part_f, o_);
        } else {
            g4ops::sliding_q(q, dq.ptr<half>(), b.qn.ptr<bf16>(), dsrope.ptr<float>(), R);
            g4ops::attn_sliding_dec(q, dq.ptr<half>(), cache.k[si].ptr<half>(), cache.v[si].ptr<half>(), R, pk, sd, 0, part_f, o_);
        }
        g4fa::dense16(q, o_, R, NH * hd, b.o.ptr<half>(), NH * hd, DH, dh.ptr<half>());
        g4ops::add_norm<half>(q, x, dh.ptr<half>(), b.w_pa.ptr<bf16>(), 1.0f, b.w_pf.ptr<bf16>(), dh.ptr<half>(), R, DH);
        g4fa::dense16(q, dh.ptr<half>(), R, DH, b.g.ptr<half>(), DH, DI, dg.ptr<half>());
        g4fa::dense16(q, dh.ptr<half>(), R, DH, b.u.ptr<half>(), DH, DI, du.ptr<half>());
        g4ops::gelu_mul(q, dg.ptr<half>(), du.ptr<half>(), dg.ptr<half>(), (size_t)R * DI);
        g4fa::dense32(q, dg.ptr<half>(), R, DI, b.d.ptr<half>(), DI, DH, dd.ptr<float>());
        const bf16* wn = li + 1 < dlayers.size() ? nullptr : dw_norm.ptr<bf16>();
        g4ops::add_norm<float>(q, x, dd.ptr<float>(), b.w_ff.ptr<bf16>(), b.scalar, wn, dy.ptr<half>(), R, DH);
    }
    g4fa::dense16(q, dy.ptr<half>(), R, DH, dhead.ptr<half>(), DH, V, dlog.ptr<half>());
    g4fa::dense16(q, dy.ptr<half>(), R, DH, dpost.ptr<half>(), DH, D, dprev.ptr<half>());
    q.memcpy(dlog_h, dlog.p, (size_t)R * V * 2);
}

void G4Model::draft_rows(const int* game, const int* tok, const int* row, const int* n, int R, std::vector<std::vector<int>>& out) {
    out.assign(R, {});
    if (dlayers.empty() || R <= 0) return;
    auto& q = Dev::get().q;
    namespace ex = sycl::ext::oneapi::experimental;
    dec_ev.wait();
    pos_[cur_] = pos;
    // the live chains; each starts from its target row's normed hidden state
    std::vector<int> live, t(R);
    half* dp = dprev.ptr<half>();
    for (int i = 0; i < R; i++) {
        if (n[i] <= 0 || pos_[game[i]] < 1) continue;
        GLM_CHECK((int)live.size() < DQ, "draft: more than %d chains", DQ);
        q.memcpy(dp + live.size() * D, fin16.ptr<half>() + (size_t)row[i] * D, (size_t)D * 2);
        t[i] = tok[i];
        live.push_back(i);
    }
    std::vector<float> em(D);
    while (!live.empty()) {
        const int Rl = (int)live.size();
        for (int j = 0; j < Rl; j++) {
            int i = live[j], pg = pos_[game[i]];
            dpos_h[j] = pg;                 // the chain's token's position: every step keeps it
            dpos_h[DQ + j] = pg - 1;        // the keys written so far
            dpos_h[2 * DQ + j] = game[i];
            embed_host(&t[i], 1, em.data());
            half* z = dz_h + (size_t)j * D;
            for (int d = 0; d < D; d++) z[d] = half(em[d]);
        }
        if (graphs_on) {
            if ((int)dgraphs.size() <= Rl) dgraphs.resize(Rl + 1);
            if (!dgraphs[Rl]) {
                ex::command_graph<ex::graph_state::modifiable> g{q.get_context(), q.get_device()};
                g.begin_recording(q);
                draft_dev(Rl);
                g.end_recording(q);
                dgraphs[Rl] = std::make_unique<ex::command_graph<ex::graph_state::executable>>(g.finalize());
            }
            q.ext_oneapi_graph(*dgraphs[Rl]).wait();
        } else {
            draft_dev(Rl);
            q.wait();
        }
        // greedy; a doubtful token is rarely followed by a right one: that chain ends here (a wasted step costs ~5 ms, and
        // only accepted tokens pay)
        std::vector<int> next;
        static thread_local std::vector<float> lf(V);
        for (int j = 0; j < Rl; j++) {
            int i = live[j];
            f16_to_f32(dlog_h + (size_t)j * V, lf.data(), V);
            float bv, sv;
            int best = top2(lf.data(), V, bv, sv);
            out[i].push_back(best);
            t[i] = best;
            bool more = (int)out[i].size() < n[i] && bv - sv >= draft_margin;
            if (more) {
                if ((int)next.size() != j) q.memcpy(dp + next.size() * D, dp + (size_t)j * D, (size_t)D * 2);
                next.push_back(i);
            }
        }
        live.swap(next);
    }
}

int G4Model::draft(int tok, int hidden_row, int n, int* out) {
    if (dlayers.empty() || pos < 1) return 0;
    std::vector<std::vector<int>> o;
    draft_rows(&cur_, &tok, &hidden_row, &n, 1, o);
    std::copy(o[0].begin(), o[0].end(), out);
    return (int)o[0].size();
}

// ------------------------------------------------------------------------------------------------ state

void G4Model::reset() {
    Dev::get().q.wait();
    pos = 0;
    for (auto& c : ckpts_[cur_]) c.pos = -1;
}

// the game's sliding caches at `at` in pinned host memory (the full layers' caches are not overwritten going back): a
// checkpoint is ~0.9 GB. A game keeps up to turn_ckpts at turn starts and one at a prompt's end. They are told apart
// because the next prompt rarely continues from a prompt's end: its last tokens open the reply's thought (and, from the client,
// a prefilled one), which the next prompt drops, so it branches at the turn's start; and a prompt that follows a hidden
// pass branches a turn earlier, where that pass's instruction began. A tool loop's prompts each end somewhere new, and
// would push the turn starts out if they shared their places.
void G4Model::save_checkpoint(int at, bool turn) {
    auto& q = Dev::get().q;
    auto& cks = ckpts_[cur_];
    const size_t ring_bytes = (size_t)RING * SKV * SD * 2;
    const int cap_turn = turn_ckpts, cap_end = 1;
    if (turn && cap_turn == 0) return;
    for (auto& e : cks)
        if (e.pos == at) {                      // the same tokens up to here: the same caches
            e.turn = e.turn || turn;
            return;
        }
    Checkpoint* c = nullptr;
    for (auto& e : cks) if (e.pos < 0) { c = &e; break; }
    if (!c && (int)cks.size() < cap_turn + cap_end) {
        cks.emplace_back();
        c = &cks.back();
        for (size_t s = 0; s < cache.k.size(); s++) {
            c->k.push_back(sycl::malloc_host(ring_bytes, q));
            c->v.push_back(sycl::malloc_host(ring_bytes, q));
            GLM_CHECK(c->k.back() && c->v.back(), "checkpoint: pinned allocation failed");
        }
    }
    if (!c) {
        // every place taken: the earliest of this kind goes, or, while this kind has fewer than its share (an unused
        // place went to the other kind), the earliest of the other
        int same = 0;
        for (auto& e : cks) same += e.turn == turn;
        const bool from_same = same >= (turn ? cap_turn : cap_end);
        for (auto& e : cks)
            if ((e.turn == turn) == from_same && (!c || e.pos < c->pos)) c = &e;
        GLM_CHECK(c, "checkpoint: no place to reuse");
    }
    for (size_t s = 0; s < cache.k.size(); s++) {
        q.memcpy(c->k[s], cache.k[s].ptr<char>() + cur_ * ring_bytes, ring_bytes);
        q.memcpy(c->v[s], cache.v[s].ptr<char>() + cur_ * ring_bytes, ring_bytes);
    }
    q.wait();
    c->pos = at;
    c->turn = turn;
}

// where restore(upto) would leave `game`: its two ways back, without taking either
int G4Model::reach(int game, int upto) const {
    const int at = position_of(game);
    if (upto <= at && upto >= at - (RING - WIN + 1) && upto >= 0) return upto;
    int best = -1;
    for (auto& c : ckpts_[game])
        if (c.pos <= upto) best = std::max(best, c.pos);
    return best;
}

bool G4Model::restore(int upto) {
    auto& q = Dev::get().q;
    q.wait();
    auto& cks = ckpts_[cur_];
    const size_t ring_bytes = (size_t)RING * SKV * SD * 2;
    // the ring still holds the window of a position this close behind
    if (upto <= pos && upto >= pos - (RING - WIN + 1) && upto >= 0) {
        pos = upto;
        for (auto& c : cks) if (c.pos > upto) c.pos = -1;
        return true;
    }
    Checkpoint* best = nullptr;
    for (auto& c : cks)
        if (c.pos >= 0 && c.pos <= upto && (!best || c.pos > best->pos)) best = &c;
    if (!best) return false;
    for (size_t s = 0; s < cache.k.size(); s++) {
        q.memcpy(cache.k[s].ptr<char>() + cur_ * ring_bytes, best->k[s], ring_bytes);
        q.memcpy(cache.v[s].ptr<char>() + cur_ * ring_bytes, best->v[s], ring_bytes);
    }
    q.wait();
    pos = best->pos;
    for (auto& c : cks) if (c.pos > pos) c.pos = -1;
    return true;
}

}  // namespace glm
