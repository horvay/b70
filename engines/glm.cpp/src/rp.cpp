#include "rp.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <sys/mman.h>
#include "g4_ops.h"
#include "json.h"
#include "ops.h"
#include "rp_ops.h"
#include "sampler.h"
#include "progress.h"
#include <oneapi/mkl/blas.hpp>

namespace glm {

using namespace rp;

namespace {
std::string lp(int li) { return "model.layers." + std::to_string(li) + "."; }
double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int env_i(const char* n, int d) { return getenv(n) ? atoi(getenv(n)) : d; }
Tensor load_bf16(const SafeTensors& st, const std::string& name) {
    Tensor t = st.load(name);
    GLM_CHECK(t.dt == DType::BF16, "%s: expected bf16", name.c_str());
    return t;
}
}  // namespace

RpLin RpLin::load(const SafeTensors& st, const std::string& prefix) {
    RpLin l;
    if (Exl3Linear::present(st, prefix)) {
        l.e = Exl3Linear::load(st, prefix);
        l.in = l.e.in;
        l.out = l.e.out;
        return l;
    }
    const StTensor& t = st.at(prefix + ".weight");
    l.dense = true;
    l.out = (int)t.shape[0];
    l.in = (int)t.shape[1];
    if (t.dt == DType::F16) {
        l.w = st.load(prefix + ".weight");
    } else {
        GLM_CHECK(t.dt == DType::BF16, "%s: expected trellis, fp16 or bf16", prefix.c_str());
        size_t n = t.bytes / 2;
        std::vector<half> h(n);
        const bf16* b = static_cast<const bf16*>(t.data);
        for (size_t i = 0; i < n; i++) h[i] = half(static_cast<float>(b[i]));
        l.w = Tensor::empty(t.shape, DType::F16);
        upload(l.w.p, h.data(), n * 2);
    }
    return l;
}

void RpLin::forward(sycl::queue& q, Workspace& ws, const half* x, int M, void* y, DType ydt, int xs) const {
    if (xs < 0) xs = in;
    if (!dense) return e.forward(q, ws, x, M, y, ydt, xs);
    // past a few blocks of rows, one fp16 GEMM (fp32 accumulation) beats rows of DPAS threads
    if (M > 64 && ydt == DType::F16) {
        oneapi::mkl::blas::row_major::gemm(q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::trans, M, out, in, half(1.0f), x, xs,
                                           w.ptr<half>(), in, half(0.0f), static_cast<half*>(y), out);
        return;
    }
    for (int r = 0; r < M; r += 8) {
        int n = std::min(8, M - r);
        if (ydt == DType::F16) g4fa::dense16(q, x + (size_t)r * xs, n, xs, w.ptr<half>(), in, out, static_cast<half*>(y) + (size_t)r * out);
        else g4fa::dense32(q, x + (size_t)r * xs, n, xs, w.ptr<half>(), in, out, static_cast<float*>(y) + (size_t)r * out);
    }
}

// ------------------------------------------------------------------------------------------------ load

RpModel::RpModel(const std::string& dir, int max_tok, const std::string& drafter, int games) : max_tokens(max_tok), games_(games) {
    GLM_CHECK(games >= 1, "games: %d", games);
    exl3_large_above = env_i("GLM_EXL3_LARGE", 32);
    pos_.assign(games, 0);
    ckpts_.resize(games);
    TS = max_tokens + 32;                       // a block of keys past the context: the kernels' block loads may read it
    Json c = Json::parse(read_file(dir + "/config.json"));
    GLM_CHECK(c.get_s("model_type", "") == "ricursive", "%s: not a Ripple model", dir.c_str());
    GLM_CHECK(c.get_i("hidden_size", 0) == D && c.get_i("num_hidden_layers", 0) == L && c.get_i("intermediate_size", 0) == I &&
                  c.get_i("vocab_size", 0) == V && c.get_i("sliding_window", 0) == WIN && c.get_i("num_attention_heads", 0) == NH &&
                  c.get_i("num_key_value_heads", 0) == NKV && c.get_i("head_dim", 0) == HD,
              "%s: not the Ripple-65B shape this engine is built for", dir.c_str());
    GLM_CHECK(c.get_b("use_gated_attention", false) && c.get_b("use_valres", false), "%s: gated attention and value residuals expected", dir.c_str());
    for (int li = 0; li < L; li++)
        GLM_CHECK((c["layer_types"][li].s() == "full_attention") == is_global(li), "layer %d: type", li);
    GLM_CHECK(c.get_f("ffn_clamp_threshold", 0) == CLAMP, "%s: FFN clamp", dir.c_str());
    // folded checkpoints bake the scale into the embedding and have none here
    if (const Json* es = c.get("embed_scale"); es && es->kind != Json::Null) embed_scale = (float)es->f();
    nl = env_i("RP_TEST_LAYERS", L);
    st = std::make_unique<SafeTensors>(dir);
    auto& q = Dev::get().q;
    // what fits: the weights are known from the files; each game's context and the prompt scratch share the rest
    {
        const double GiB = 1073741824.0;
        double weights = 0;
        for (auto& n : st->names_with_prefix("model.layers.")) weights += st->at(n).bytes;
        for (auto& n : st->names_with_prefix("lm_head")) weights += st->at(n).bytes;
        const double total = (double)Dev::get().dev.get_info<sycl::info::device::global_mem_size>();
        const double reserve = (getenv("G4_VRAM_RESERVE_GIB") ? atof(getenv("G4_VRAM_RESERVE_GIB")) : 1.0) * GiB;   // driver, graphs, oneMKL
        const double rows_b = 0.1 * GiB;
        const int ng = 16, ns = L - ng;
        const double ring_b = 2.0 * ns * RING * KD * 2 * games_;
        const double tok_b = 2.0 * ng * KD * 2 * games_;        // a context token of all games
        double avail = total - weights - reserve - rows_b - ring_b - (drafter.empty() ? 0.0 : (double)draft_bytes(drafter));
        // with a drafter a prompt's rows also hold their taps and the K, V they make
        const double row_b = drafter.empty() ? 363e3 : 363e3 + 2.0 * (DFC + D + 2 * KD);
        auto scratch = [row_b](int sc) { return sc * row_b + 0.45 * 1073741824.0; };
        int want_sc = env_i("GLM_PREFILL_SC", 0);
        double need = (double)max_tokens * tok_b;
        if (want_sc > 0) super_chunk = want_sc;
        else {
            super_chunk = 1024;
            for (int sc : {8192, 4096, 2048})
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
    graphs_on = env_i("G4_GRAPHS", 1) != 0 && !g_prof_points && !g_sync_points;

    const StTensor& et = st->at("model.embed_tokens.weight");
    GLM_CHECK(et.dt == DType::BF16 && et.shape[0] == V && et.shape[1] == D, "embedding: expected bf16 [V, D]");
    embed = static_cast<const bf16*>(et.data);
    {
        const size_t page = 4096;
        uintptr_t b0 = reinterpret_cast<uintptr_t>(et.data) & ~(page - 1);
        madvise(reinterpret_cast<void*>(b0), et.bytes + (reinterpret_cast<uintptr_t>(et.data) - b0), MADV_POPULATE_READ);
    }
    head = RpLin::load(*st, "lm_head");
    GLM_CHECK(head.in == D && head.out == V, "lm_head: shape");
    w_norm = load_bf16(*st, "model.norm.weight");

    layers.resize(nl);
    size_t lw = 0;
    auto need = [&](const RpLin& l) { lw = std::max({lw, l.ws_bytes(super_chunk), l.ws_bytes(QMAX), l.ws_bytes(1)}); };
    for (int li = 0; li < nl; li++) {
        RpLayer& b = layers[li];
        std::string p = lp(li), ap = p + "self_attn.";
        b.global = is_global(li);
        b.q = RpLin::load(*st, ap + "q_proj");
        b.k = RpLin::load(*st, ap + "k_proj");
        b.v = RpLin::load(*st, ap + "v_proj");
        b.g = RpLin::load(*st, ap + "g_proj");
        b.o = RpLin::load(*st, ap + "o_proj");
        b.gate = RpLin::load(*st, p + "mlp.gate_proj");
        b.up = RpLin::load(*st, p + "mlp.up_proj");
        b.down = RpLin::load(*st, p + "mlp.down_proj");
        GLM_CHECK(b.q.out == QD && b.k.out == KD && b.v.out == KD && b.g.out == QD && b.o.in == QD && b.gate.out == I && b.down.in == I, "%s: shapes", p.c_str());
        b.w_in = load_bf16(*st, p + "input_layernorm.weight");
        b.w_pa = load_bf16(*st, p + "post_attention_layernorm.weight");
        b.w_pf = load_bf16(*st, p + "pre_feedforward_layernorm.weight");
        b.w_ff = load_bf16(*st, p + "post_feedforward_layernorm.weight");
        b.qn = load_bf16(*st, ap + "q_norm.weight");
        b.kn = load_bf16(*st, ap + "k_norm.weight");
        const StTensor& lam = st->at(ap + "value_res_lambda");
        GLM_CHECK(lam.dt == DType::BF16, "%svalue_res_lambda: expected bf16", ap.c_str());
        float lv = static_cast<float>(*static_cast<const bf16*>(lam.data));
        b.blend = static_cast<float>(bf16(1.0f / (1.0f + std::exp(-lv))));
        for (const RpLin* l : {&b.q, &b.k, &b.v, &b.g, &b.o, &b.gate, &b.up, &b.down}) need(*l);
        load_progress("loaded layer %d/%d (%.2f GiB VRAM)", li + 1, nl, Dev::get().vram_used / 1073741824.0);
    }
    load_progress_end();
    need(head);
    // (a drafter's EXL3 linears: a block's rows, less than this)
    lin.init(std::max(lw, drafter.empty() ? (size_t)0 : (size_t)64 << 20) + ((size_t)1 << 20));

    for (int li = 0; li < nl; li++) {
        if (is_global(li)) {
            gk.push_back(Tensor::empty({(int64_t)games_ * TS, NKV, HD}, DType::F16));
            gv.push_back(Tensor::empty({(int64_t)games_ * TS, NKV, HD}, DType::F16));
        } else {
            ck.push_back(Tensor::empty({(int64_t)games_ * RING, NKV, HD}, DType::F16));
            cv.push_back(Tensor::empty({(int64_t)games_ * RING, NKV, HD}, DType::F16));
        }
    }

    // decode rows
    X = Tensor::empty({QMAX, D}, DType::F32);
    E0 = Tensor::empty({QMAX, D}, DType::F32);
    h16 = Tensor::empty({QMAX, D}, DType::F16);
    vin = Tensor::empty({QMAX, D}, DType::F16);
    q16 = Tensor::empty({QMAX, QD}, DType::F16);
    k16 = Tensor::empty({QMAX, KD}, DType::F16);
    v16 = Tensor::empty({QMAX, KD}, DType::F16);
    gt16 = Tensor::empty({QMAX, QD}, DType::F16);
    o16 = Tensor::empty({QMAX, QD}, DType::F16);
    a16 = Tensor::empty({QMAX, D}, DType::F16);
    g16 = Tensor::empty({QMAX, I}, DType::F16);
    u16 = Tensor::empty({QMAX, I}, DType::F16);
    d32 = Tensor::empty({QMAX, D}, DType::F32);
    fin16 = Tensor::empty({QMAX, D}, DType::F16);
    logits = Tensor::empty({QMAX, V}, DType::F16);
    part = Tensor::empty({(int64_t)rpops::attn_part_floats(QMAX)}, DType::F32);
    srope_d = Tensor::empty({QMAX, HD / 2, 2}, DType::F32);
    pos_dev = Tensor::zeros({2 * QMAX}, DType::I32);
    pos_h = sycl::malloc_host<int>(2 * QMAX, q);
    rpops::rope_rows(q, 1, nullptr, 0, srope_d.ptr<float>());     // its table now: a recorded graph cannot wait on the upload

    // a prompt's super-chunk: kp / vp hold the window's WIN - 1 rows in front of the rows
    const int64_t SC = super_chunk;
    Xp = Tensor::empty({SC, D}, DType::F32);
    E0p = Tensor::empty({SC, D}, DType::F32);
    srope_p = Tensor::empty({SC, HD / 2, 2}, DType::F32);
    hp = Tensor::empty({SC, D}, DType::F16);                // also o_proj's output
    vinp = Tensor::empty({SC, D}, DType::F16);
    qp = Tensor::empty({SC, QD}, DType::F16);               // also attention's output and down_proj's (fp32 [SC, D])
    gtp = Tensor::empty({SC, QD}, DType::F16);
    kp = Tensor::empty({WIN - 1 + SC, KD}, DType::F16);
    vp = Tensor::empty({WIN - 1 + SC, KD}, DType::F16);
    gp = Tensor::empty({SC, I}, DType::F16);
    up = Tensor::empty({SC, I}, DType::F16);
    x_host = sycl::malloc_host<float>((size_t)std::max(super_chunk, QMAX) * D, q);
    GLM_CHECK(x_host && pos_h, "pinned upload buffers");
    q.wait();
    fprintf(stderr, "buffers: %.2f GiB VRAM, Exl3 workspace %.2f GiB\n", Dev::get().vram_used / 1073741824.0, lw / 1073741824.0);

    if (!drafter.empty()) load_drafter(drafter);

    // every kernel once (modules load on first use): a short prompt and a decode step
    if (env_i("G4_WARMUP", 1) != 0) {
        double tw = now_s();
        std::vector<int> t(std::min(300, max_tokens - 2 * QMAX), 1000);
        prefill(t.data(), (int)t.size());
        forward(t.data(), 1);
        q.wait();
        for (int Q = 2; Q <= (dlayers.empty() ? 8 : DBLOCK); Q++) {
            restore(position() - 1);
            std::vector<int> vr(Q, t[0]);
            forward(vr.data(), Q);
            q.wait();
            restore(position() - Q + 1);
        }
        if (!dlayers.empty()) {
            std::vector<int> o(DBLOCK);
            draft(t[0], 0, DBLOCK - 1, o.data());
        }
        reset();
        fprintf(stderr, "warm-up %.1f s\n", now_s() - tw);
    }
    fprintf(stderr, "model ready: %.2f GiB VRAM\n", Dev::get().vram_used / 1073741824.0);
}

RpModel::~RpModel() {
    auto& q = Dev::get().q;
    q.wait();
    for (auto& g : ckpts_)
        for (auto& c : g) {
            for (void* p : c.k) if (p) sycl::free(p, q);
            for (void* p : c.v) if (p) sycl::free(p, q);
        }
    for (auto& g : ckpts_)
        for (auto& c : g) {
            for (void* p : c.dk) if (p) sycl::free(p, q);
            for (void* p : c.dv) if (p) sycl::free(p, q);
        }
    graphs.clear();
    dgraph.clear();
    for (void* p : {(void*)dx_h, (void*)dpos_h, (void*)dlog_h, (void*)dgate_h}) if (p) sycl::free(p, q);
    if (pos_h) sycl::free(pos_h, q);
    if (x_host) sycl::free(x_host, q);
}

// ------------------------------------------------------------------------------------------------ one layer

// E[id] * bf16(embed_scale), rounded to bf16 as the bf16 checkpoint's forward does, widened to fp32
void RpModel::embed_host(const int* tokens, int n, float* dst) {
    const bf16 mul(embed_scale);
    for (int t = 0; t < n; t++) {
        const bf16* e = embed + (size_t)tokens[t] * D;
        float* o = dst + (size_t)t * D;
        for (int d = 0; d < D; d++) o[d] = static_cast<float>(bf16(static_cast<float>(e[d]) * static_cast<float>(mul)));
    }
}

// x: rows' stream; h: their input-norm output (written with the next layer's on the way out; the last layer's norm is the
// final norm, in fin)
void RpModel::decode_layer(int li, float* x, const float* e0, half* h, int Q, const int* pdev, const int* sdev, half* fin) {
    auto& q = Dev::get().q;
    RpLayer& b = layers[li];
    half* q_ = q16.ptr<half>();
    half* o_ = o16.ptr<half>();
    b.q.forward(q, lin, h, Q, q_, DType::F16);
    b.k.forward(q, lin, h, Q, k16.p, DType::F16);
    b.g.forward(q, lin, h, Q, gt16.p, DType::F16);
    rpops::lerp(q, e0, h, b.blend, vin.ptr<half>(), Q);
    b.v.forward(q, lin, vin.ptr<half>(), Q, v16.p, DType::F16);
    prof_point("q, k, g, v");
    if (b.global) {
        int gi = global_index(li);
        rpops::post(q, q_, k16.ptr<half>(), v16.ptr<half>(), b.qn.ptr<bf16>(), b.kn.ptr<bf16>(), nullptr, Q, pdev, 0,
                    gk[gi].ptr<half>(), gv[gi].ptr<half>(), TS, false, sdev);
        rpops::attn_global(q, q_, gk[gi].ptr<half>(), gv[gi].ptr<half>(), Q, pdev, sdev, TS, part.ptr<float>(), o_);
    } else {
        int si = sliding_index(li);
        rpops::post(q, q_, k16.ptr<half>(), v16.ptr<half>(), b.qn.ptr<bf16>(), b.kn.ptr<bf16>(), srope_d.ptr<float>(), Q, pdev, 0,
                    ck[si].ptr<half>(), cv[si].ptr<half>(), RING, true, sdev);
        rpops::attn_sliding(q, q_, ck[si].ptr<half>(), cv[si].ptr<half>(), Q, pdev, sdev, part.ptr<float>(), o_);
    }
    prof_point("attention");
    rpops::gate_mul(q, o_, gt16.ptr<half>(), (size_t)Q * QD);
    b.o.forward(q, lin, o_, Q, a16.p, DType::F16);
    g4ops::add_norm<half>(q, x, a16.ptr<half>(), b.w_pa.ptr<bf16>(), 1.0f, b.w_pf.ptr<bf16>(), h, Q, D);
    prof_point("o_proj + norms");
    b.gate.forward(q, lin, h, Q, g16.p, DType::F16);
    b.up.forward(q, lin, h, Q, u16.p, DType::F16);
    rpops::silu_mul(q, g16.ptr<half>(), u16.ptr<half>(), g16.ptr<half>(), (size_t)Q * I);
    prof_point("gate, up");
    b.down.forward(q, lin, g16.ptr<half>(), Q, d32.p, DType::F32);
    const bf16* wn = li + 1 < nl ? layers[li + 1].w_in.ptr<bf16>() : w_norm.ptr<bf16>();
    g4ops::add_norm<float>(q, x, d32.ptr<float>(), b.w_ff.ptr<bf16>(), 1.0f, wn, li + 1 < nl ? h : fin, Q, D);
    if (!dlayers.empty() && tap_of[li] >= 0) rpops::to16(q, x, tap16.ptr<half>() + (size_t)tap_of[li] * D, Q, DFC);
    prof_point("down + norms");
}

void RpModel::forward_dev(int Q) {
    auto& q = Dev::get().q;
    float* x = X.ptr<float>();
    const int* pdev = pos_dev.ptr<int>();
    const int* sdev = pdev + QMAX;
    q.memcpy(x, x_host, (size_t)Q * D * 4);
    q.memcpy(E0.p, x_host, (size_t)Q * D * 4);
    q.memcpy(pos_dev.p, pos_h, (size_t)2 * QMAX * sizeof(int));
    rpops::rope_rows(q, Q, pdev, 0, srope_d.ptr<float>());
    g4ops::norm16(q, x, layers[0].w_in.ptr<bf16>(), h16.ptr<half>(), Q, D);
    for (int li = 0; li < nl; li++) decode_layer(li, x, E0.ptr<float>(), h16.ptr<half>(), Q, pdev, sdev, fin16.ptr<half>());
    head.forward(q, lin, fin16.ptr<half>(), Q, logits.p, DType::F16);
    prof_point("head");
    if (!dlayers.empty()) {
        draft_kv_rows(tap16.ptr<half>(), tapn.ptr<half>(), dk16.ptr<half>(), dv16.ptr<half>(), Q, pdev, 0, sdev, srope_d.ptr<float>(), false);
        prof_point("drafter's K, V");
    }
}

void RpModel::select(int game) {
    GLM_CHECK(game >= 0 && game < games_, "game %d of %d", game, games_);
    pos_[cur_] = pos;
    cur_ = game;
    pos = pos_[game];
}

const half* RpModel::forward(const int* tokens, int Q) {
    std::vector<int> g(Q, cur_);
    return forward_rows(g.data(), tokens, Q);
}

const half* RpModel::forward_rows(const int* game, const int* tokens, int Q) {
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

void RpModel::prefill_sc(const int* tokens, int n) {
    auto& q = Dev::get().q;
    const int p0 = pos;
    x_ev.wait();
    embed_host(tokens, n, x_host);
    float* x = Xp.ptr<float>();
    half* h = hp.ptr<half>();
    half* qr = qp.ptr<half>();
    q.memcpy(x, x_host, (size_t)n * D * 4);
    x_ev = q.memcpy(E0p.p, x_host, (size_t)n * D * 4);
    rpops::rope_rows(q, n, nullptr, p0, srope_p.ptr<float>());
    g4ops::norm16(q, x, layers[0].w_in.ptr<bf16>(), h, n, D);
    const size_t grow = (size_t)cur_ * TS * KD, gring = (size_t)cur_ * RING * KD;     // this game's part of the caches
    for (int li = 0; li < nl; li++) {
        RpLayer& b = layers[li];
        prof_point("(entry)");
        b.q.forward(q, lin, h, n, qr, DType::F16);
        b.g.forward(q, lin, h, n, gtp.p, DType::F16);
        rpops::lerp(q, E0p.ptr<float>(), h, b.blend, vinp.ptr<half>(), n);
        if (b.global) {
            int gi = global_index(li);
            half* Kg = gk[gi].ptr<half>() + grow;
            half* Vg = gv[gi].ptr<half>() + grow;
            b.k.forward(q, lin, h, n, kp.p, DType::F16);
            b.v.forward(q, lin, vinp.ptr<half>(), n, vp.p, DType::F16);
            prof_point("q, g, k, v");
            rpops::post(q, qr, kp.ptr<half>(), vp.ptr<half>(), b.qn.ptr<bf16>(), b.kn.ptr<bf16>(), nullptr, n, nullptr, p0, Kg, Vg, TS, false, nullptr);
            for (int q0 = 0; q0 < n; q0 += q_chunk) {
                int qn = std::min(q_chunk, n - q0);
                rpfa::prompt(q, qr + (size_t)q0 * QD, Kg, Vg, qn, p0 + q0 + qn, p0 + q0, 0, 0, qr + (size_t)q0 * QD);
            }
        } else {
            int si = sliding_index(li);
            // the rows' K / V in place behind the window's rows
            half* kr = kp.ptr<half>() + (size_t)(WIN - 1) * KD;
            half* vr = vp.ptr<half>() + (size_t)(WIN - 1) * KD;
            b.k.forward(q, lin, h, n, kr, DType::F16);
            b.v.forward(q, lin, vinp.ptr<half>(), n, vr, DType::F16);
            prof_point("q, g, k, v");
            rpops::post(q, qr, kr, vr, b.qn.ptr<bf16>(), b.kn.ptr<bf16>(), srope_p.ptr<float>(), n, nullptr, p0, kr, vr, 0, false, nullptr);
            int prior = std::min(p0, WIN - 1);
            half* kw = kr - (size_t)prior * KD;
            half* vw = vr - (size_t)prior * KD;
            half* Kg = ck[si].ptr<half>() + gring;
            half* Vg = cv[si].ptr<half>() + gring;
            rpops::ring_gather(q, Kg, p0 - prior, prior, kw);
            rpops::ring_gather(q, Vg, p0 - prior, prior, vw);
            for (int q0 = 0; q0 < n; q0 += q_chunk) {
                int qn = std::min(q_chunk, n - q0);
                rpfa::prompt(q, qr + (size_t)q0 * QD, kw, vw, qn, prior + q0 + qn, p0 + q0, p0 - prior, WIN, qr + (size_t)q0 * QD);
            }
            int keep = std::min(n, RING);
            rpops::ring_scatter(q, kr + (size_t)(n - keep) * KD, p0 + n - keep, keep, Kg);
            rpops::ring_scatter(q, vr + (size_t)(n - keep) * KD, p0 + n - keep, keep, Vg);
        }
        prof_point("attention");
        rpops::gate_mul(q, qr, gtp.ptr<half>(), (size_t)n * QD);
        b.o.forward(q, lin, qr, n, h, DType::F16);
        g4ops::add_norm<half>(q, x, h, b.w_pa.ptr<bf16>(), 1.0f, b.w_pf.ptr<bf16>(), h, n, D);
        prof_point("o_proj + norms");
        b.gate.forward(q, lin, h, n, gp.p, DType::F16);
        b.up.forward(q, lin, h, n, up.p, DType::F16);
        rpops::silu_mul(q, gp.ptr<half>(), up.ptr<half>(), gp.ptr<half>(), (size_t)n * I);
        prof_point("gate, up");
        float* d = qp.ptr<float>();     // [SC, D] fp32: the query rows are done with
        b.down.forward(q, lin, gp.ptr<half>(), n, d, DType::F32);
        g4ops::add_norm<float>(q, x, d, b.w_ff.ptr<bf16>(), 1.0f, li + 1 < nl ? layers[li + 1].w_in.ptr<bf16>() : nullptr, h, n, D);
        if (!dlayers.empty() && tap_of[li] >= 0) rpops::to16(q, x, tapp.ptr<half>() + (size_t)tap_of[li] * D, n, DFC);
        prof_point("down + norms");
        if (n >= 256) read_marks().mark((double)n / nl);    // (a short pass: one mark, below)
    }
    if (n < 256) read_marks().mark(n);
    if (!dlayers.empty()) {
        draft_kv_rows(tapp.ptr<half>(), tapnp.ptr<half>(), dkp.ptr<half>(), dvp.ptr<half>(), n, nullptr, p0, nullptr, srope_p.ptr<float>(), true);
        prof_point("drafter's K, V");
    }
    pos += n;
}

void RpModel::prefill(const int* tokens, int n, int ckpt_at) {
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
    const int chunk = n <= 96 ? 32 : super_chunk;
    for (int s0 = 0; s0 < n; s0 += chunk) prefill_sc(tokens + s0, std::min(chunk, n - s0));
    q.wait();
    if (cut == n) save_checkpoint(pos);
}

// ------------------------------------------------------------------------------------------------ drafter (DFlash2)

// what load_drafter puts on the card: weights fp16, a game's rings, a block's rows
size_t RpModel::draft_bytes(const std::string& dir) const {
    SafeTensors d(dir);
    size_t b = 0;
    for (auto& n : d.names_with_prefix("layers.")) b += d.at(n).bytes;
    for (auto& n : d.names_with_prefix("fc.")) b += d.at(n).bytes;
    b += d.at("candidate_selector.hidden_projection.weight").bytes;
    b += (size_t)games_ * (4 * DRING_S + DRING_G) * (size_t)KD * 4;
    b += (size_t)DBLOCK * (QD * 12 + I * 4 + D * 20) + (size_t)QMAX * (DFC + D + 2 * KD) * 2 + DBLOCK * (size_t)V * 2;
    return b;
}

void RpModel::load_drafter(const std::string& dir) {
    Json c = Json::parse(read_file(dir + "/config.json"));
    const Json& dc = c["dflash_config"];
    GLM_CHECK(dc.get_i("block_size", 0) == DBLOCK && dc.get_i("conv_kernel_size", 0) == 4 && dc.get_i("conv_group_size", 0) == 16 &&
                  dc.get_i("selector_rank", 0) == DRANK && dc.get_i("selector_top_k", 0) == DTOPK && dc.get_i("mask_token_id", 0) == DMASK &&
                  c.get_i("num_hidden_layers", 0) == DL && c.get_i("hidden_size", 0) == D && c.get_i("intermediate_size", 0) == I &&
                  c.get_i("head_dim", 0) == HD && c.get_i("num_attention_heads", 0) == NH && c.get_i("num_key_value_heads", 0) == NKV,
              "%s: not Ripple-65B's DFlash2", dir.c_str());
    for (int i = 0; i < 5; i++) GLM_CHECK(dc["target_layer_ids"][i].i() == DTAPS[i], "%s: taps", dir.c_str());
    dst = std::make_unique<SafeTensors>(dir);
    std::fill(tap_of, tap_of + L, -1);
    // a test on the first few layers taps those instead
    for (int i = 0; i < 5; i++) tap_of[nl == L ? DTAPS[i] : DTAPS[i] % nl] = i;
    dfc = RpLin::load(*dst, "fc");
    dsel = RpLin::load(*dst, "candidate_selector.hidden_projection");
    dw_hn = load_bf16(*dst, "hidden_norm.weight");
    dw_norm = load_bf16(*dst, "norm.weight");
    const StTensor& a = dst->at("candidate_selector.predecessor_codebook");
    const StTensor& b = dst->at("candidate_selector.successor_codebook");
    GLM_CHECK(a.dt == DType::BF16 && b.dt == DType::BF16 && a.shape[0] == V && a.shape[1] == DRANK && b.shape[0] == V && b.shape[1] == DRANK, "selector codebooks");
    dpred = static_cast<const bf16*>(a.data);
    dsucc = static_cast<const bf16*>(b.data);
    dlayers.resize(DL);
    for (int li = 0; li < DL; li++) {
        RpDraftLayer& d = dlayers[li];
        std::string p = "layers." + std::to_string(li) + ".", ap = p + "self_attn.";
        d.global = c["layer_types"][li].s() == "full_attention";
        GLM_CHECK(d.global == (li == DL - 1), "%s: layer %d type", dir.c_str(), li);
        d.ring = d.global ? DRING_G : DRING_S;
        d.win = d.global ? DWIN_G : DWIN_S;
        d.q = RpLin::load(*dst, ap + "q_proj");
        d.k = RpLin::load(*dst, ap + "k_proj");
        d.v = RpLin::load(*dst, ap + "v_proj");
        d.o = RpLin::load(*dst, ap + "o_proj");
        d.gate = RpLin::load(*dst, p + "mlp.gate_proj");
        d.up = RpLin::load(*dst, p + "mlp.up_proj");
        d.down = RpLin::load(*dst, p + "mlp.down_proj");
        d.ac = RpLin::load(*dst, p + "attention_conv.kernel_projection");
        d.mc = RpLin::load(*dst, p + "mlp_conv.kernel_projection");
        d.ac_base = load_bf16(*dst, p + "attention_conv.base_kernel");
        d.mc_base = load_bf16(*dst, p + "mlp_conv.base_kernel");
        d.w_in = load_bf16(*dst, p + "input_layernorm.weight");
        d.w_pa = load_bf16(*dst, p + "post_attention_layernorm.weight");
        d.qn = load_bf16(*dst, ap + "q_norm.weight");
        d.kn = load_bf16(*dst, ap + "k_norm.weight");
        d.K = Tensor::empty({(int64_t)games_ * d.ring, NKV, HD}, DType::F16);
        d.V = Tensor::empty({(int64_t)games_ * d.ring, NKV, HD}, DType::F16);
        for (const RpLin* l : {&d.q, &d.k, &d.v, &d.o, &d.gate, &d.up, &d.down, &d.ac, &d.mc})
            GLM_CHECK(l->ws_bytes(DBLOCK) <= lin.cap, "drafter: Exl3 workspace");
    }
    auto& q = Dev::get().q;
    tap16 = Tensor::empty({QMAX, DFC}, DType::F16);
    tapn = Tensor::empty({QMAX, D}, DType::F16);
    dk16 = Tensor::empty({QMAX, KD}, DType::F16);
    dv16 = Tensor::empty({QMAX, KD}, DType::F16);
    const int64_t SC = super_chunk;
    tapp = Tensor::empty({SC, DFC}, DType::F16);
    tapnp = Tensor::empty({SC, D}, DType::F16);
    dkp = Tensor::empty({SC, KD}, DType::F16);
    dvp = Tensor::empty({SC, KD}, DType::F16);
    const int64_t R = DBLOCK;
    dX = Tensor::empty({R, D}, DType::F32);
    dy = Tensor::empty({R, D}, DType::F16);
    dy2 = Tensor::empty({R, D}, DType::F16);
    dq = Tensor::empty({R, QD}, DType::F16);
    dok = Tensor::empty({R, QD}, DType::F16);
    dgt = Tensor::empty({R, I}, DType::F16);
    dup = Tensor::empty({R, I}, DType::F16);
    dd = Tensor::empty({R, D}, DType::F32);
    dcv = Tensor::empty({R, 2560}, DType::F16);
    dstate = Tensor::empty({R, D}, DType::F16);
    dgate = Tensor::empty({R, DRANK}, DType::F16);
    dlog = Tensor::empty({R, V}, DType::F16);
    dpos = Tensor::zeros({2 * R}, DType::I32);
    dscale = Tensor::empty({R}, DType::F32);
    dsrope = Tensor::empty({R, HD / 2, 2}, DType::F32);
    dx_h = sycl::malloc_host<float>((size_t)R * D, q);
    dpos_h = sycl::malloc_host<int>(2 * R, q);
    dlog_h = sycl::malloc_host<half>((size_t)R * V, q);
    dgate_h = sycl::malloc_host<half>((size_t)R * DRANK, q);
    GLM_CHECK(dx_h && dpos_h && dlog_h && dgate_h, "drafter: pinned buffers");
    // the mask token's embedding, row 1 on: the same for every block
    embed_host(std::vector<int>(1, DMASK).data(), 1, dx_h + D);
    for (int r = 2; r < DBLOCK; r++) memcpy(dx_h + (size_t)r * D, dx_h + D, D * 4);
    {
        const size_t page = 4096;
        for (const StTensor* t : {&a, &b}) {
            uintptr_t b0 = reinterpret_cast<uintptr_t>(t->data) & ~(page - 1);
            madvise(reinterpret_cast<void*>(b0), t->bytes + (reinterpret_cast<uintptr_t>(t->data) - b0), MADV_POPULATE_READ);
        }
    }
    draft_tokens = std::min(env_i("RP_DRAFT", DBLOCK - 1), DBLOCK - 1);
    if (getenv("RP_DRAFT_MARGIN")) draft_margin = (float)atof(getenv("RP_DRAFT_MARGIN"));
    fprintf(stderr, "drafter %s: %d tokens a step (%.2f GiB VRAM)\n", dir.c_str(), draft_tokens, Dev::get().vram_used / 1073741824.0);
}

// n rows of the target's taps -> the drafter's K, V at the rows' positions. A prompt keeps only what each ring can hold.
void RpModel::draft_kv_rows(const half* tap, half* tn, half* k, half* v, int n, const int* pdev, int p0, const int* sdev, const float* rope, bool prompt) {
    auto& q = Dev::get().q;
    int first = prompt ? std::max(0, n - DRING_G) : 0;
    const int m = n - first;
    dfc.forward(q, lin, tap + (size_t)first * DFC, m, tn, DType::F16, DFC);
    g4ops::add_norm_rows(q, tn, dw_hn.ptr<bf16>(), tn, m, D);
    for (auto& d : dlayers) {
        int f2 = prompt ? std::max(0, m - d.ring) : 0;      // rows the ring keeps
        const int mk = m - f2;
        const half* tr = tn + (size_t)f2 * D;
        d.k.forward(q, lin, tr, mk, k, DType::F16);
        d.v.forward(q, lin, tr, mk, v, DType::F16);
        const size_t base = prompt ? (size_t)cur_ * d.ring * KD : 0;
        rpops::post(q, nullptr, k, v, nullptr, d.kn.ptr<bf16>(), d.global ? nullptr : rope + (size_t)(first + f2) * HD,
                    mk, pdev, p0 + first + f2, d.K.ptr<half>() + base, d.V.ptr<half>() + base, d.ring, true, sdev);
    }
}

// one block's kernels, recorded once: the anchor's embedding and the rows' positions come from the pinned buffers
void RpModel::draft_dev() {
    auto& q = Dev::get().q;
    const int R = DBLOCK;
    const int* pd = dpos.ptr<int>();
    const int* sd = pd + R;
    float* x = dX.ptr<float>();
    q.memcpy(x, dx_h, (size_t)R * D * 4);
    q.memcpy(dpos.p, dpos_h, (size_t)2 * R * sizeof(int));
    rpops::rope_rows(q, R, pd, 0, dsrope.ptr<float>());
    half* y = dy.ptr<half>();
    half* yc = dy2.ptr<half>();
    int dbg = 0;
    for (auto& b : dlayers) {
        if (dbg++ >= draft_debug_layers) break;
        const int dyn2 = 2560;
        g4ops::norm16(q, x, b.w_in.ptr<bf16>(), y, R, D);
        b.ac.forward(q, lin, y, R, dcv.p, DType::F16);
        rpops::dynconv_prepare(q, y, dcv.ptr<half>(), dyn2, b.ac_base.ptr<bf16>(), yc, R);
        b.q.forward(q, lin, yc, R, dq.p, DType::F16);
        b.k.forward(q, lin, yc, R, dk16.p, DType::F16);
        b.v.forward(q, lin, yc, R, dv16.p, DType::F16);
        rpops::post(q, dq.ptr<half>(), dk16.ptr<half>(), dv16.ptr<half>(), b.qn.ptr<bf16>(), b.kn.ptr<bf16>(), b.global ? nullptr : dsrope.ptr<float>(), R, pd, 0,
                    b.K.ptr<half>(), b.V.ptr<half>(), b.ring, true, sd);
        rpops::attn_draft(q, dq.ptr<half>(), b.K.ptr<half>(), b.V.ptr<half>(), R, pd, sd, b.ring, b.win, b.global ? 8 : 2, part.ptr<float>(), dok.ptr<half>());
        b.o.forward(q, lin, dok.ptr<half>(), R, dd.p, DType::F32);
        rpops::dynconv_finish(q, dd.ptr<float>(), dcv.ptr<half>() + 1280, dyn2, b.ac_base.ptr<bf16>() + (size_t)4 * D, x, R);
        g4ops::norm16(q, x, b.w_pa.ptr<bf16>(), y, R, D);
        b.mc.forward(q, lin, y, R, dcv.p, DType::F16);
        rpops::dynconv_prepare(q, y, dcv.ptr<half>(), dyn2, b.mc_base.ptr<bf16>(), yc, R);
        b.gate.forward(q, lin, yc, R, dgt.p, DType::F16);
        b.up.forward(q, lin, yc, R, dup.p, DType::F16);
        rpops::silu_scaled(q, dgt.ptr<half>(), dup.ptr<half>(), dgt.ptr<half>(), dscale.ptr<float>(), R);
        b.down.forward(q, lin, dgt.ptr<half>(), R, dd.p, DType::F32);
        rpops::mul_rows(q, dd.ptr<float>(), dscale.ptr<float>(), R);
        rpops::dynconv_finish(q, dd.ptr<float>(), dcv.ptr<half>() + 1280, dyn2, b.mc_base.ptr<bf16>() + (size_t)4 * D, x, R);
    }
    g4ops::norm16(q, x, dw_norm.ptr<bf16>(), dstate.ptr<half>(), R, D);
    // the target's head over the rows after the anchor, the selector's gate
    head.forward(q, lin, dstate.ptr<half>() + D, R - 1, dlog.p, DType::F16);
    dsel.forward(q, lin, dstate.ptr<half>() + D, R - 1, dgate.p, DType::F16);
    q.memcpy(dlog_h, dlog.p, (size_t)(R - 1) * V * 2);
    q.memcpy(dgate_h, dgate.p, (size_t)(R - 1) * DRANK * 2);
}

// the block at the game's position, then the selector's greedy walk: S_t(a, b) = U_t(b) + <A(a) * H_t, B(b)> over each row's top 32
int RpModel::draft_chain(int game, int tok, int n, int* out) {
    auto& q = Dev::get().q;
    namespace ex = sycl::ext::oneapi::experimental;
    const int p = pos_[game];
    if (p < 1 || n <= 0) return 0;
    embed_host(&tok, 1, dx_h);
    for (int r = 0; r < DBLOCK; r++) {
        dpos_h[r] = p + r;
        dpos_h[DBLOCK + r] = game;
    }
    if (graphs_on) {
        if (dgraph.empty()) {
            ex::command_graph<ex::graph_state::modifiable> g{q.get_context(), q.get_device()};
            g.begin_recording(q);
            draft_dev();
            g.end_recording(q);
            dgraph.push_back(std::make_unique<ex::command_graph<ex::graph_state::executable>>(g.finalize()));
        }
        q.ext_oneapi_graph(*dgraph[0]);
        q.wait();
    } else {
        draft_dev();
        q.wait();
    }
    static thread_local std::vector<float> lf(V), gatef(DRANK), a(DRANK);
    int pred = tok, made = 0;
    int idx[DTOPK];
    float val[DTOPK];
    for (int r = 0; r < std::min(n, DBLOCK - 1); r++) {
        f16_to_f32(dlog_h + (size_t)r * V, lf.data(), V);
        top_k(lf.data(), V, DTOPK, idx, val);
        f16_to_f32(dgate_h + (size_t)r * DRANK, gatef.data(), DRANK);
        const bf16* ap = dpred + (size_t)pred * DRANK;
        for (int k = 0; k < DRANK; k++) a[k] = static_cast<float>(ap[k]) * gatef[k];
        float best = -INFINITY, second = -INFINITY;
        int bi = 0;
        for (int c = 0; c < DTOPK; c++) {
            const bf16* bp = dsucc + (size_t)idx[c] * DRANK;
            float s = 0;
            for (int k = 0; k < DRANK; k++) s += a[k] * static_cast<float>(bp[k]);
            s += val[c];
            if (s > best) { second = best; best = s; bi = idx[c]; }
            else if (s > second) second = s;
        }
        out[made++] = bi;
        pred = bi;
        if (draft_margin > 0 && best - second < draft_margin) break;
    }
    return made;
}

void RpModel::dump_draft(const std::string& dir, int tok, int layers) {
    auto& q = Dev::get().q;
    draft_debug_layers = layers;
    embed_host(&tok, 1, dx_h);
    pos_[cur_] = pos;
    for (int r = 0; r < DBLOCK; r++) { dpos_h[r] = pos + r; dpos_h[DBLOCK + r] = cur_; }
    draft_dev();
    q.wait();
    draft_debug_layers = 99;
    auto out = [&](const char* name, const void* p, size_t bytes) {
        void* h = sycl::malloc_host(bytes, q);
        q.memcpy(h, p, bytes).wait();
        FILE* f = fopen((dir + "/" + name).c_str(), "wb");
        fwrite(h, 1, bytes, f);
        fclose(f);
        sycl::free(h, q);
    };
    out("x.f32", dX.p, DBLOCK * D * 4);
    out("y2.f16", dy2.p, DBLOCK * D * 2);
    out("q.f16", dq.p, (size_t)DBLOCK * QD * 2);
    out("o.f16", dok.p, (size_t)DBLOCK * QD * 2);
    out("cv.f16", dcv.p, (size_t)DBLOCK * 2560 * 2);
    out("state.f16", dstate.p, DBLOCK * D * 2);
}

int RpModel::draft(int tok, int hidden_row, int n, int* out) {
    (void)hidden_row;
    if (dlayers.empty()) return 0;
    dec_ev.wait();
    pos_[cur_] = pos;
    return draft_chain(cur_, tok, n, out);
}

void RpModel::draft_rows(const int* game, const int* tok, const int* row, const int* n, int R, std::vector<std::vector<int>>& out) {
    (void)row;
    out.assign(R, {});
    // a block costs 12 ms a chain: past 4 games together it costs more than the rows it saves (8 games 54 tok/s, without it 59)
    if (dlayers.empty() || R > 4) return;
    dec_ev.wait();
    pos_[cur_] = pos;
    int buf[DBLOCK];
    for (int i = 0; i < R; i++) {
        int m = draft_chain(game[i], tok[i], n[i], buf);
        out[i].assign(buf, buf + m);
    }
}

// ------------------------------------------------------------------------------------------------ state

void RpModel::reset() {
    Dev::get().q.wait();
    pos = 0;
    for (auto& c : ckpts_[cur_]) c.pos = -1;
}

// the game's sliding caches at `at` in pinned host memory (the global layers' caches are not overwritten going back): ~0.3 GB
void RpModel::save_checkpoint(int at) {
    auto& q = Dev::get().q;
    auto& cks = ckpts_[cur_];
    const size_t ring_bytes = (size_t)RING * KD * 2;
    Checkpoint* c = nullptr;
    for (auto& e : cks) if (e.pos < 0) { c = &e; break; }
    if (!c && (int)cks.size() < (games_ > 1 ? 1 : 3)) {
        cks.emplace_back();
        c = &cks.back();
        for (size_t s = 0; s < ck.size(); s++) {
            c->k.push_back(sycl::malloc_host(ring_bytes, q));
            c->v.push_back(sycl::malloc_host(ring_bytes, q));
            GLM_CHECK(c->k.back() && c->v.back(), "checkpoint: pinned allocation failed");
        }
        for (auto& d : dlayers) {
            const size_t rb = (size_t)d.ring * KD * 2;
            c->dk.push_back(sycl::malloc_host(rb, q));
            c->dv.push_back(sycl::malloc_host(rb, q));
            GLM_CHECK(c->dk.back() && c->dv.back(), "checkpoint: pinned allocation failed");
        }
    }
    if (!c) {
        c = &cks[0];
        for (auto& e : cks) if (e.pos < c->pos) c = &e;
    }
    for (size_t s = 0; s < ck.size(); s++) {
        q.memcpy(c->k[s], ck[s].ptr<char>() + cur_ * ring_bytes, ring_bytes);
        q.memcpy(c->v[s], cv[s].ptr<char>() + cur_ * ring_bytes, ring_bytes);
    }
    for (size_t s = 0; s < dlayers.size(); s++) {
        const size_t rb = (size_t)dlayers[s].ring * KD * 2;
        q.memcpy(c->dk[s], dlayers[s].K.ptr<char>() + cur_ * rb, rb);
        q.memcpy(c->dv[s], dlayers[s].V.ptr<char>() + cur_ * rb, rb);
    }
    q.wait();
    c->pos = at;
}

bool RpModel::restore(int upto) {
    auto& q = Dev::get().q;
    q.wait();
    auto& cks = ckpts_[cur_];
    const size_t ring_bytes = (size_t)RING * KD * 2;
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
    for (size_t s = 0; s < ck.size(); s++) {
        q.memcpy(ck[s].ptr<char>() + cur_ * ring_bytes, best->k[s], ring_bytes);
        q.memcpy(cv[s].ptr<char>() + cur_ * ring_bytes, best->v[s], ring_bytes);
    }
    for (size_t s = 0; s < dlayers.size(); s++) {
        const size_t rb = (size_t)dlayers[s].ring * KD * 2;
        q.memcpy(dlayers[s].K.ptr<char>() + cur_ * rb, best->dk[s], rb);
        q.memcpy(dlayers[s].V.ptr<char>() + cur_ * rb, best->dv[s], rb);
    }
    q.wait();
    pos = best->pos;
    for (auto& c : cks) if (c.pos > pos) c.pos = -1;
    return true;
}

}  // namespace glm
