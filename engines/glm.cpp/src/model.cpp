#include "model.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <thread>
#include <tuple>
#include "moe.h"
#include "progress.h"
#include <sycl/ext/oneapi/experimental/graph.hpp>

namespace glm {

namespace {
const std::string P = "model.language_model.";
enum Phase { P_MIX, P_KDA, P_MLA, P_MLP, P_ROUTE, P_TIER, P_MOE, P_HEAD, P_N };
const char* kPhase[P_N] = {"mHC + norms", "KDA", "MLA", "dense / shared MLP", "router + picks copy", "tier (host)",
                           "expert kernels", "embed / lm_head"};


Tensor f32_to_f16_dev(const SafeTensors& st, const std::string& name) {
    Tensor f = st.load(name);
    GLM_CHECK(f.dt == DType::F32, "%s: expected F32", name.c_str());
    Tensor h = Tensor::empty(f.shape, DType::F16);
    ops::f32_to_f16(Dev::get().q, f.ptr<float>(), h.ptr<half>(), f.numel());
    Dev::get().q.wait();
    return h;
}
}  // namespace

Model::Model(const std::string& dir, const std::string& store_bin, int max_tok, TierConfig tier_cfg,
             const std::string& warm_stats, double reserve_gib, const std::string& lora, float lora_scale)
    : max_tokens(max_tok) {
    Json c = Json::parse(read_file(dir + "/config.json"));
    const Json& tc = c.has("text_config") ? c["text_config"] : c;
    cfg.hidden = tc.get_i("hidden_size", 4096);
    cfg.layers = tc.get_i("num_hidden_layers", 45);
    cfg.vocab = tc.get_i("vocab_size", 154880);
    cfg.experts = tc.get_i("n_routed_experts", 288);
    cfg.topk = tc.get_i("num_experts_per_tok", 8);
    cfg.moe_inter = tc.get_i("moe_intermediate_size", 2048);
    cfg.rsf = tc.get_f("routed_scaling_factor", 2.5);
    cfg.eps = tc.get_f("rms_norm_eps", 1e-5);
    int first_dense = tc.get_i("first_k_dense_replace", 3);
    for (int L = 0; L < cfg.layers; L++) {
        cfg.mla.push_back(tc["layer_types"][L].s() != "linear_attention");
        cfg.moe.push_back(L >= first_dense);
    }
    GLM_CHECK(cfg.hidden == 4096, "hidden size %d", cfg.hidden);

    st = std::make_unique<SafeTensors>(dir);
    auto& q = Dev::get().q;
    if (getenv("GLM_PREFILL_CHUNK")) prefill_chunk = std::max(64, std::min(2048, atoi(getenv("GLM_PREFILL_CHUNK"))));
    // workspaces for prefill_chunk rows (linears: the largest of decode-in-GEMM / dequantize + GEMM; activations:
    // KDA ~300 KB and MLA ~230 KB per row; experts: 32-row blocks and fp16 pair outputs)
    act.init((size_t)prefill_chunk * 310 * 1024 + ((size_t)64 << 20));
    mws.init((size_t)prefill_chunk * 8 * (4096 + 2048) * 2 + ((size_t)128 << 20));

    const StTensor& et = st->at(P + "embed_tokens.weight");
    GLM_CHECK(et.dt == DType::BF16 && et.shape[1] == 4096, "embedding: expected bf16 [vocab, 4096]");
    embed = static_cast<const bf16*>(et.data);
    final_norm = st->load(P + "norm.weight");
    lm_head = Exl3Linear::load(*st, "lm_head");

    blocks.resize(cfg.layers);
    kda_st.resize(cfg.layers);
    mla_c.resize(cfg.layers);
    for (int L = 0; L < cfg.layers; L++) {
        Block& b = blocks[L];
        std::string p = P + "layers." + std::to_string(L) + ".";
        b.layer = L;
        b.is_mla = cfg.mla[L];
        b.is_moe = cfg.moe[L];
        const char* site[2] = {"hc_attn_", "hc_ffn_"};
        for (int s = 0; s < 2; s++) {
            b.hc_fn[s] = f32_to_f16_dev(*st, p + site[s] + "fn");
            b.hc_fn32[s] = st->load(p + site[s] + "fn");
            b.hc_base[s] = st->load(p + site[s] + "base");
            b.hc_scale[s] = st->load(p + site[s] + "scale");
        }
        b.in_norm = st->load(p + "input_layernorm.weight");
        b.post_norm = st->load(p + "post_attention_layernorm.weight");
        if (b.is_mla) {
            b.mla = std::make_unique<MlaLayer>(MlaLayer::load(*st, p + "self_attn"));
            mla_c[L].init(max_tokens);
        } else {
            b.kda = std::make_unique<KdaLayer>(KdaLayer::load(*st, p + "self_attn"));
            kda_st[L].init(0);      // history slots come with MTP
        }
        std::string mp = b.is_moe ? p + "mlp.shared_experts." : p + "mlp.";
        b.gate = Exl3Linear::load(*st, mp + "gate_proj");
        b.up = Exl3Linear::load(*st, mp + "up_proj");
        b.down = Exl3Linear::load(*st, mp + "down_proj");
        if (b.is_moe) {
            b.router = st->load(p + "mlp.gate.weight");
            GLM_CHECK(b.router.dt == DType::F16 && b.router.size(0) == cfg.experts, "router: fp16 [E, 4096]");
            const StTensor& esb = st->at(p + "mlp.gate.e_score_correction_bias");
            const float* e = static_cast<const float*>(esb.data);
            double mean = 0;
            for (int i = 0; i < cfg.experts; i++) mean += e[i];
            float m = (float)(mean / cfg.experts);
            std::vector<half> bc(cfg.experts);
            for (int i = 0; i < cfg.experts; i++) bc[i] = half(e[i] - m);
            b.bias_c = Tensor::empty({cfg.experts}, DType::F16);
            upload(b.bias_c.p, bc.data(), bc.size() * 2);
            b.esb = st->load(p + "mlp.gate.e_score_correction_bias");
            b.tier_li = L - first_dense;
            if (L + 1 < cfg.layers) b.next_moe = L + 1;
        }
        load_progress("loaded layer %d/%d (%.2f GiB VRAM)", L + 1, cfg.layers, Dev::get().vram_used / 1073741824.0);
    }
    load_progress_end();
    if (!lora.empty()) {
        // the GM LoRA: runtime deltas on the dense linears, KDA q/k/v fused, MLA kv_b merged into its weight
        loras_ = std::make_unique<LoraSet>(lora, lora_scale);
        int n = 0;
        auto at = [&](Exl3Linear& l, const std::string& m) { if ((l.lora = loras_->get(m))) n++; };
        for (auto& b : blocks) {
            std::string p = P + "layers." + std::to_string(b.layer) + ".";
            std::string mp = b.is_moe ? p + "mlp.shared_experts." : p + "mlp.";
            at(b.gate, mp + "gate_proj"); at(b.up, mp + "up_proj"); at(b.down, mp + "down_proj");
            if (b.kda) {
                KdaLayer& K = *b.kda;
                if ((K.qkv.lora = loras_->fused_qkv(p + "self_attn"))) n++;
                at(K.o, p + "self_attn.o_proj");
                // the five fp16 projections: merged into the weights (W = fp16(W + fp16(sB A)), as kv_b), so they
                // cost nothing at run time; GLM_LORA_MERGE_FP16=0 applies them at run time as the PyTorch stack does
                static const bool merge = !(getenv("GLM_LORA_MERGE_FP16") && atoi(getenv("GLM_LORA_MERGE_FP16")) == 0);
                std::pair<const Lora**, Tensor*> fp[] = {{&K.lb, &K.b_proj}, {&K.lfa, &K.f_a}, {&K.lfb, &K.f_b},
                                                         {&K.lga, &K.g_a}, {&K.lgb, &K.g_b}};
                const char* nm[] = {"b_proj", "f_a_proj", "f_b_proj", "g_a_proj", "g_b_proj"};
                for (int i = 0; i < 5; i++) {
                    std::string mod = p + "self_attn." + nm[i];
                    if (merge) n += loras_->merge_into(mod, *fp[i].second);
                    else if ((*fp[i].first = loras_->get(mod))) n++;
                }
            }
            if (b.mla) {
                MlaLayer& A = *b.mla;
                at(A.q_a, p + "self_attn.q_a_proj"); at(A.q_b, p + "self_attn.q_b_proj");
                at(A.kv_a, p + "self_attn.kv_a_proj_with_mqa"); at(A.o, p + "self_attn.o_proj");
                n += loras_->merge_into(p + "self_attn.kv_b_proj", A.kv_b);
            }
        }
        // first projections shared by linears on the same input: gate + up (y), q_a + kv_a (x); in call order
        for (auto& b : blocks) {
            loras_->group({b.gate.lora, b.up.lora}, qmax);
            if (b.mla) loras_->group({b.mla->q_a.lora, b.mla->kv_a.lora}, qmax);
        }
        fprintf(stderr, "LoRA: %d modules (%zu in the adapter)\n", n, loras_->count());
    }
    for (auto& b : blocks)
        if (b.kda) b.kda->fuse();      // after the LoRA: runtime-LoRA'd projections stay separate
    // the linears' workspace: the most any of them needs for a prompt piece or a decode step
    size_t lw = lm_head.ws_bytes(qmax);
    auto need = [&](const Exl3Linear& l) { lw = std::max({lw, l.ws_bytes(prefill_chunk), l.ws_bytes(qmax)}); };
    for (auto& b : blocks) {
        need(b.gate); need(b.up); need(b.down);
        if (b.kda) { need(b.kda->qkv); need(b.kda->o); }
        if (b.mla) { need(b.mla->q_a); need(b.mla->q_b); need(b.mla->kv_a); need(b.mla->o); need(b.mla->wq_b); }
    }
    lin.init(lw + ((size_t)1 << 20));
    // buffers the prompt pieces use get prefill_chunk rows; the decode-only ones (streams, picks, head, logits) qmax
    int Q = std::max(qmax, prefill_chunk), H = 4096;
    x_in = Tensor::empty({Q, H}, DType::F32);
    R = Tensor::empty({qmax, 4, H}, DType::F32);
    collapsed = Tensor::empty({Q, H}, DType::F16);
    y = Tensor::empty({Q, H}, DType::F16);
    attn_out = Tensor::empty({Q, H}, DType::F32);
    mlp_out = Tensor::empty({Q, H}, DType::F32);
    g = Tensor::empty({Q, 12288}, DType::F16);
    u = Tensor::empty({Q, 12288}, DType::F16);
    a = Tensor::empty({Q, 12288}, DType::F16);
    sh = Tensor::empty({Q, H}, DType::F32);
    moe_out = Tensor::empty({Q, H}, DType::F16);
    post = Tensor::empty({Q, 4}, DType::F32);
    comb = Tensor::empty({Q, 16}, DType::F32);
    hc_part = Tensor::empty({std::max(Q * ops::hc_mix_chunks(Q), qmax * ops::hc_mix_chunks(1)) * 25}, DType::F32);
    sel = Tensor::empty({qmax, cfg.topk}, DType::I32);
    w = Tensor::empty({qmax, cfg.topk}, DType::F32);
    rlogits = Tensor::empty({Q, cfg.experts}, DType::F16);
    fin = Tensor::empty({qmax, H}, DType::F32);
    fin16 = Tensor::empty({qmax, H}, DType::F16);
    logits = Tensor::empty({qmax, cfg.vocab}, DType::F16);
    nlogits = Tensor::empty({qmax, cfg.experts}, DType::F16);
    pos_dev = Tensor::zeros({1}, DType::I32);
    pfw = getenv("GLM_PF_WIDTH") ? std::max(0, std::min(32, atoi(getenv("GLM_PF_WIDTH")))) : 12;
    split_buf = Tensor::empty({48}, DType::I32);       // permuted sel | w | pair_of for moe_split_begin
    int np = qmax * (2 * cfg.topk + pfw);
    picks = Tensor::empty({np}, DType::I64);
    picks_h = sycl::malloc_host<int64_t>(np, q);
    x_host.resize((size_t)Q * H);
    prof_on = getenv("GLM_PROF") && atoi(getenv("GLM_PROF")) > 0;
    prof_gpu = prof_on && atoi(getenv("GLM_PROF")) >= 2;
    prof_moe = prof_on && atoi(getenv("GLM_PROF")) == 3;
    // decode graphs: GLM_GRAPHS=0 turns them off; GLM_PROF=1 (host waits per phase) cannot run inside them
    graphs_on = !(getenv("GLM_GRAPHS") && atoi(getenv("GLM_GRAPHS")) == 0) && !(prof_on && !prof_gpu) && !prof_moe &&
                !g_prof_points;
    split_on = !(getenv("GLM_MOE_SPLIT") && atoi(getenv("GLM_MOE_SPLIT")) == 0);
    prof_s.assign(P_N, 0.0);
    q.wait();

    // expert tier: the VRAM left after the weights and activations, less a reserve
    TierConfig& tc_ = tier_cfg;
    if (!tc_.vram_bytes) {
        size_t total = Dev::get().dev.get_info<sycl::info::device::global_mem_size>();
        size_t used = Dev::get().vram_used + (size_t)(reserve_gib * 1073741824.0);
        GLM_CHECK(total > used + (1ull << 30), "no VRAM left for experts (%.1f GiB used)", used / 1073741824.0);
        tc_.vram_bytes = total - used;
    }
    tier_ = std::make_unique<Tier>(store_bin, tc_);
    for (auto& b : blocks)
        if (b.is_moe) b.kc = tier_->kc_of(b.tier_li);
    if (!warm_stats.empty()) {
        // hottest first: the routing profile's per-expert scores, keyed by the store's layer keys
        Json sj = Json::parse(read_file(warm_stats));
        Json mj = Json::parse(read_file(store_bin.substr(0, store_bin.rfind('.')) + ".json"));
        std::vector<std::tuple<double, int, int>> sc;
        for (auto& l : mj["layers"].arr) {
            std::string key = l["key"].s();
            if (!sj.has(key)) continue;
            int li = l["li"].i();
            auto& v = sj[key].arr;
            for (size_t e = 0; e < v.size(); e++) sc.emplace_back(v[e].f(), li, (int)e);
        }
        std::sort(sc.begin(), sc.end(), [](auto& a, auto& b) { return std::get<0>(a) > std::get<0>(b); });
        std::vector<std::pair<int, int>> order;
        for (auto& [s_, li, e] : sc) order.push_back({li, e});
        tier_->warm(order);
    }
    // the prompts' read-ahead ring now, not on the first prompt (GLM_RING_AT_LOAD=0: on the first prompt)
    if (!getenv("GLM_RING_AT_LOAD") || atoi(getenv("GLM_RING_AT_LOAD")) != 0) tier_->prepare_prompts();
    fprintf(stderr, "model ready: %.2f GiB VRAM\n", Dev::get().vram_used / 1073741824.0);
}

void Model::reset() {
    pos = 0;
    for (auto& c : ckpts_) c.pos = -1;
    for (int L = 0; L < cfg.layers; L++)
        if (!cfg.mla[L]) kda_st[L].zero();
    Dev::get().q.wait();
}

namespace {
void grab(std::vector<float>* d, const void* src, int n, bool f16);
}  // namespace

// --- the decode step's pieces. With graphs (decode, one token) each layer is replayed as recorded segments:
//   MoE layer:   A = mixes + attention + router + picks copy;  S = shared expert;  (host: tier);  C = experts + apply
//   dense layer: one segment;  the head: one segment.
void Model::attn_part(Block& b, int Q) {
    auto& q = Dev::get().q;
    prof_point("(between layers)");
    ops::HcSite sa{b.hc_fn[0].p, true, b.hc_base[0].ptr<float>(), b.hc_scale[0].ptr<float>()};
    ops::hc_mix(q, R.ptr<float>(), sa, post.ptr<float>(), comb.ptr<float>(), collapsed.ptr<half>(), hc_part.ptr<float>(), Q);
    ops::rms_norm<half, half>(q, collapsed.ptr<half>(), b.in_norm.ptr<bf16>(), y.ptr<half>(), Q, 4096, cfg.eps);
    grab(dump, y.p, 4096, true);
    mark(P_MIX);
    prof_point("mHC mix + norm (attn)");
    if (b.is_mla) mla_forward(q, lin, act, *b.mla, mla_c[b.layer], y.ptr<half>(), Q, pos, pos_dev.ptr<int>(),
                              attn_out.ptr<float>());
    else kda_forward(q, lin, act, *b.kda, kda_st[b.layer], y.ptr<half>(), Q, attn_out.ptr<float>(), false);
    mark(b.is_mla ? P_MLA : P_KDA);
    grab(dump, attn_out.p, 4096, false);
    ops::hc_apply(q, R.ptr<float>(), attn_out.ptr<float>(), post.ptr<float>(), comb.ptr<float>(), Q);
    ops::HcSite sf{b.hc_fn[1].p, true, b.hc_base[1].ptr<float>(), b.hc_scale[1].ptr<float>()};
    ops::hc_mix(q, R.ptr<float>(), sf, post.ptr<float>(), comb.ptr<float>(), collapsed.ptr<half>(), hc_part.ptr<float>(), Q);
    ops::rms_norm<half, half>(q, collapsed.ptr<half>(), b.post_norm.ptr<bf16>(), y.ptr<half>(), Q, 4096, cfg.eps);
    grab(dump, y.p, 4096, true);
    mark(P_MIX);
    prof_point("mHC apply + mix + norm (mlp)");
}

void Model::route_part(Block& b, int Q) {
    // router (with the residency bonus) and the next layer's prediction; the picks go to the host while the GPU
    // runs the shared expert
    auto& q = Dev::get().q;
    int k = cfg.topk, E = cfg.experts;
    const Block* nb = b.next_moe >= 0 ? &blocks[b.next_moe] : nullptr;
    ops::gemv_f16<half>(q, y.ptr<half>(), Q, 4096, b.router.ptr<half>(), E, rlogits.ptr<half>());
    if (nb) ops::gemv_f16<half>(q, y.ptr<half>(), Q, 4096, nb->router.ptr<half>(), E, nlogits.ptr<half>());
    if (tier_->bonus_on())
        ops::route_tier(q, rlogits.ptr<half>(), nb ? nlogits.ptr<half>() : nullptr, Q, b.esb.ptr<float>(),
                        nb ? nb->esb.ptr<float>() : nullptr, tier_->bonus(b.tier_li),
                        nb ? tier_->bonus(nb->tier_li) : nullptr, E, k, pfw, cfg.rsf, sel.ptr<int>(),
                        w.ptr<float>(), picks.ptr<int64_t>());
    else   // exact routing: the bsz-1 router's numerics (fp16 centred bias, fp16 weights), spec §6.4a
        ops::route_exact2(q, rlogits.ptr<half>(), nb ? nlogits.ptr<half>() : nullptr, Q, b.bias_c.ptr<half>(),
                          nb ? nb->esb.ptr<float>() : nullptr, E, k, pfw, cfg.rsf, sel.ptr<int>(), w.ptr<float>(),
                          picks.ptr<int64_t>());
    picks_ev = q.memcpy(picks_h, picks.p, (size_t)Q * (2 * k + pfw) * sizeof(int64_t));
    prof_point("router + picks copy");
}

void Model::shared_part(Block& b, int Q) {
    // dense MLP / shared expert: g, u fp16; a = clamped SwiGLU; down fp32
    auto& q = Dev::get().q;
    int I = b.gate.out;
    b.gate.forward(q, lin, y.ptr<half>(), Q, g.p, DType::F16);
    b.up.forward(q, lin, y.ptr<half>(), Q, u.p, DType::F16);
    ops::swiglu(q, g.ptr<half>(), u.ptr<half>(), a.ptr<half>(), (size_t)Q * I);
    b.down.forward(q, lin, a.ptr<half>(), Q, b.is_moe ? sh.ptr<float>() : mlp_out.ptr<float>(), DType::F32);
    prof_point(b.is_moe ? "shared expert (+LoRA)" : "dense MLP (+LoRA)");
}

void Model::moe_part(Block& b, int Q) {
    auto& q = Dev::get().q;
    moe_forward(q, mws, y.ptr<half>(), Q, sel.ptr<int>(), w.ptr<float>(), cfg.topk, tier_->ptrs(b.tier_li), 4096,
                cfg.moe_inter, b.kc, cfg.experts, moe_out.ptr<half>());
    // y = fp32(routed) + shared
    const half* mo = moe_out.ptr<half>();
    const float* s = sh.ptr<float>();
    float* o = mlp_out.ptr<float>();
    q.parallel_for(sycl::range<1>((size_t)Q * 4096), [=](sycl::id<1> i) { o[i] = static_cast<float>(mo[i]) + s[i]; });
    prof_point("expert kernels + add");
}

void Model::apply_part(int Q) {
    auto& q = Dev::get().q;
    grab(dump, mlp_out.p, 4096, false);
    ops::hc_apply(q, R.ptr<float>(), mlp_out.ptr<float>(), post.ptr<float>(), comb.ptr<float>(), Q);
    grab(dump, R.p, 4 * 4096, false);
}

// run f() as a recorded segment (replayed from the second call), or directly; the event ends the segment
template <class F>
sycl::event Model::seg(int key, bool graph, F&& f) {
    auto& q = Dev::get().q;
    if (!graph) {
        f();
        return sycl::event{};
    }
    namespace ex = sycl::ext::oneapi::experimental;
    auto it = graphs_.find(key);
    if (it == graphs_.end()) {
        ex::command_graph<ex::graph_state::modifiable> g{q.get_context(), q.get_device()};
        recording_ = true;
        g.begin_recording(q);
        f();
        g.end_recording(q);
        recording_ = false;
        it = graphs_.emplace(key, g.finalize()).first;
    }
    return q.ext_oneapi_graph(it->second);
}

void Model::mlp_forward(Block& b, int Q) {
    // the eager path (prompt pieces of the decode path, probes): the same pieces in order
    auto& q = Dev::get().q;
    if (b.is_moe) {
        tier_->sync_bonus(q);
        route_part(b, Q);
        mark(P_ROUTE);
    }
    shared_part(b, Q);
    mark(P_MLP);
    if (!b.is_moe) return;
    picks_ev.wait();
    int n = Q * cfg.topk;
    tier_->after_sync(b.tier_li, b.next_moe >= 0 ? blocks[b.next_moe].tier_li : -1, picks_h, picks_h + n, n,
                      picks_h + 2 * n, b.next_moe >= 0 ? Q * pfw : 0);
    size_t zc0 = tier_->stats.zero_copy, vh0 = tier_->stats.vram_hit;
    tier_->ensure(q, b.tier_li, picks_h, n);
    mark(P_TIER);
    prof_point("tier (host + admissions)");
    if (prof_moe) {
        sycl::event e0 = q.ext_oneapi_submit_barrier();
        moe_part(b, Q);
        moe_ev.emplace_back((int)(tier_->stats.zero_copy - zc0), (int)(tier_->stats.vram_hit - vh0), e0,
                            q.ext_oneapi_submit_barrier());
    } else {
        moe_part(b, Q);
    }
    mark(P_MOE);
}

std::string Model::moe_report() {
    Dev::get().q.wait();
    for (auto& [r, v, a, b] : moe_ev) {
        double ms = (b.get_profiling_info<sycl::info::event_profiling::command_end>() -
                     a.get_profiling_info<sycl::info::event_profiling::command_end>()) * 1e-6;
        moe_samples.emplace_back(r, v, ms);
    }
    moe_ev.clear();
    // least squares ms = a + b * ram, and the mean per RAM-pick count
    double n = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
    std::map<int, std::pair<double, int>> by;
    for (auto& [r, v, ms] : moe_samples) {
        n++; sx += r; sy += ms; sxx += (double)r * r; sxy += r * ms;
        by[r].first += ms; by[r].second++;
    }
    double b = (n * sxy - sx * sy) / std::max(n * sxx - sx * sx, 1e-9), a = (sy - b * sx) / std::max(n, 1.0);
    char buf[256];
    std::string out;
    snprintf(buf, sizeof buf, "expert kernels per MoE call: %.0f calls, %.3f ms + %.3f ms per RAM pick (mean %.3f ms)\n",
             n, a, b, sy / std::max(n, 1.0));
    out += buf;
    for (auto& [r, p] : by) {
        snprintf(buf, sizeof buf, "  %d RAM picks: %d calls, %.3f ms\n", r, p.second, p.first / p.second);
        out += buf;
    }
    return out;
}

namespace {
}  // namespace

void Model::prof_collect() {
    // phase time = GPU time between consecutive barriers (the GPU may idle inside a phase while the host works)
    uint64_t prev = 0;
    for (auto& [ph, e] : prof_ev) {
        uint64_t t = e.get_profiling_info<sycl::info::event_profiling::command_end>();
        if (prev && ph >= 0) prof_s[ph] += (t - prev) * 1e-9;
        prev = t;
    }
    prof_ev.clear();
}

void Model::mark(int phase) {
    if (!prof_on || recording_) return;
    if (prof_gpu) { prof_ev.push_back({phase, Dev::get().q.ext_oneapi_submit_barrier()}); return; }
    Dev::get().q.wait();
    auto t = std::chrono::steady_clock::now();
    prof_s[phase] += std::chrono::duration<double>(t - prof_t).count();
    prof_t = t;
}

std::string Model::prof_report(int tokens) {
    std::string r;
    double tot = 0;
    for (double v : prof_s) tot += v;
    char b[160];
    for (int i = 0; i < P_N; i++) {
        snprintf(b, sizeof b, "  %-22s %7.2f ms/token (%4.1f%%)\n", kPhase[i], prof_s[i] * 1e3 / tokens,
                 100 * prof_s[i] / std::max(tot, 1e-9));
        r += b;
    }
    snprintf(b, sizeof b, "  %-22s %7.2f ms/token\n", "total", tot * 1e3 / tokens);
    return r + b;
}

namespace {
void grab(std::vector<float>* d, const void* src, int n, bool f16) {
    if (!d) return;
    auto& q = Dev::get().q;
    q.wait();
    size_t o = d->size();
    d->resize(o + n);
    if (f16) {
        std::vector<sycl::half> h(n);
        q.memcpy(h.data(), src, n * 2).wait();
        for (int i = 0; i < n; i++) (*d)[o + i] = h[i];
    } else {
        q.memcpy(d->data() + o, src, n * 4).wait();
    }
}
}  // namespace

void Model::mix(Block& b, int site, float* Rr, int rows, float* post_, float* comb_, half* y_) {
    auto& q = Dev::get().q;
    bool f16 = rows <= 32;
    ops::HcSite s{(f16 ? b.hc_fn[site] : b.hc_fn32[site]).p, f16, b.hc_base[site].ptr<float>(),
                  b.hc_scale[site].ptr<float>()};
    ops::hc_mix(q, Rr, s, post_, comb_, collapsed.ptr<half>(), hc_part.ptr<float>(), rows);
    ops::rms_norm<half, half>(q, collapsed.ptr<half>(), (site ? b.post_norm : b.in_norm).ptr<bf16>(), y_, rows, 4096,
                              cfg.eps);
}

Model::Checkpoint& Model::new_checkpoint(int at) {
    auto& q = Dev::get().q;
    if (ckpts_.size() >= 2) {                      // reuse the oldest one's buffers
        Checkpoint c = std::move(ckpts_.front());
        ckpts_.erase(ckpts_.begin());
        ckpts_.push_back(std::move(c));
    } else {
        Checkpoint c;
        for (int L = 0; L < cfg.layers; L++) {
            if (cfg.mla[L]) { c.conv.push_back(nullptr); c.rec.push_back(nullptr); continue; }
            c.conv.push_back(sycl::malloc_host(kda_st[L].conv.bytes(), q));
            c.rec.push_back(sycl::malloc_host(kda_st[L].rec.bytes(), q));
            GLM_CHECK(c.conv.back() && c.rec.back(), "checkpoint: pinned allocation failed");
        }
        ckpts_.push_back(std::move(c));
    }
    ckpts_.back().pos = at;
    return ckpts_.back();
}

bool Model::restore(int upto) {
    auto& q = Dev::get().q;
    for (int i = (int)ckpts_.size() - 1; i >= 0; i--) {
        Checkpoint& c = ckpts_[i];
        if (c.pos < 0 || c.pos > upto) continue;
        for (int L = 0; L < cfg.layers; L++) {
            if (cfg.mla[L]) continue;
            q.memcpy(kda_st[L].conv.p, c.conv[L], kda_st[L].conv.bytes());
            q.memcpy(kda_st[L].rec.p, c.rec[L], kda_st[L].rec.bytes());
            kda_st[L].last_history = 0;
        }
        q.wait();
        pos = c.pos;
        // checkpoints past the restored position describe another sequence now
        ckpts_.erase(ckpts_.begin() + i + 1, ckpts_.end());
        return true;
    }
    return false;
}

void Model::prefill(const int* tokens, int n, int ckpt_at) {
    if (n <= 0) return;
    GLM_CHECK(pos + n <= max_tokens, "context full (%d tokens)", max_tokens);
    auto& q = Dev::get().q;
    const int C = prefill_chunk, H = 4096, k = cfg.topk, E = cfg.experts;
    // GLM_PREFILL_TRACE=1: wait after each step and print where the prompt is (1 per layer, 2 per step, 3 per point)
    static const int trace = getenv("GLM_PREFILL_TRACE") ? atoi(getenv("GLM_PREFILL_TRACE")) : 0;
    g_sync_points = trace >= 3;
    auto t_pf = std::chrono::steady_clock::now();
    auto tr = [&](int lvl, int layer, const char* what, int c0) {
        if (trace < lvl) return;
        q.wait();
        fprintf(stderr, "prefill %d rows: layer %d %s (rows %d+) at %.2f s\n", n, layer, what, c0,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t_pf).count());
    };
    if (trace)
        fprintf(stderr, "prefill %d rows at position %d, checkpoint at %d; %.2f GiB VRAM in use\n", n, pos, ckpt_at,
                Dev::get().vram_used / 1073741824.0);
    // streaming (staging + read-ahead of every layer's non-resident experts) pays off for prompt-sized calls only;
    // fewer than 2048 picks read their experts on demand through the decode path's tier (as the PyTorch stack)
    const bool stream = (size_t)n * cfg.topk >= 2048;
    // every row's residual streams and MLP input, 72 KB a row (1.4 GB for 20K tokens): a streamed prompt borrows them
    // from the expert pool, as its staging slots (the VRAM beside the pool is only the reserve)
    const size_t rbytes = (size_t)n * 4 * H * sizeof(float), ybytes = (size_t)n * H * sizeof(half);
    Tensor Rall, yall;
    float* Rp;
    half* yp;
    if (stream) {
        char* lent = static_cast<char*>(tier_->lend(rbytes + ybytes));
        Rp = reinterpret_cast<float*>(lent);
        yp = reinterpret_cast<half*>(lent + rbytes);
    } else {
        Rall = Tensor::empty({n, 4, H}, DType::F32);
        yall = Tensor::empty({n, H}, DType::F16);
        Rp = Rall.ptr<float>();
        yp = yall.ptr<half>();
    }
    Tensor pall = Tensor::empty({n, 4}, DType::F32), call = Tensor::empty({n, 16}, DType::F32);
    Tensor sall = Tensor::empty({n, k}, DType::I32), wall = Tensor::empty({n, k}, DType::F32);
    std::vector<int> sel_h((size_t)n * k);
    for (int c0 = 0; c0 < n; c0 += C) {
        int rows = std::min(C, n - c0);
        std::vector<float> xh((size_t)rows * H);
        for (int t = 0; t < rows; t++) {
            const bf16* e = embed + (size_t)tokens[c0 + t] * H;
            for (int d = 0; d < H; d++) xh[(size_t)t * H + d] = static_cast<float>(e[d]);
        }
        q.memcpy(x_in.p, xh.data(), xh.size() * sizeof(float)).wait();
        ops::hc_expand(q, x_in.ptr<float>(), Rp + (size_t)c0 * 4 * H, rows);
    }
    tr(1, -1, "embedding", 0);
    // long prompts use nearly every expert of every layer: double-buffered staging, the next MoE layer's experts
    // copied while this one computes (GLM_PROMPT_DBL=0: one set, copied at each layer's routing)
    static const bool dbl_on = !getenv("GLM_PROMPT_DBL") || atoi(getenv("GLM_PROMPT_DBL")) != 0;
    if (stream) {
        tier_->begin_prompt(0, dbl_on && (size_t)n * k >= (size_t)32 * E);
        tier_->prestage(0);
    }
    tr(1, -1, "begin_prompt", 0);
    // in-pass checkpoint: each KDA layer's piece holding row `cut` is split there and the state copied out
    int cut = ckpt_at > pos && ckpt_at <= pos + n ? ckpt_at - pos : -1;
    Checkpoint* ck = cut > 0 ? &new_checkpoint(ckpt_at) : nullptr;
    tr(1, -1, "checkpoint", 0);
    for (auto& b : blocks) {
        // attention, piece by piece (KDA state and MLA cache advance in order)
        for (int c0 = 0; c0 < n; c0 += C) {
            int rows = std::min(C, n - c0);
            float* Rc = Rp + (size_t)c0 * 4 * H;
            mix(b, 0, Rc, rows, post.ptr<float>(), comb.ptr<float>(), y.ptr<half>());
            prof_point("prefill: attention mix");
            if (b.is_mla) {
                q.fill(pos_dev.ptr<int>(), pos + c0, 1);
                mla_forward(q, lin, act, *b.mla, mla_c[b.layer], y.ptr<half>(), rows, pos + c0, pos_dev.ptr<int>(),
                            attn_out.ptr<float>());
            } else {
                KdaState& ks = kda_st[b.layer];
                int split = ck && cut > c0 && cut <= c0 + rows ? cut - c0 : rows;
                kda_forward(q, lin, act, *b.kda, ks, y.ptr<half>(), split, attn_out.ptr<float>(), false);
                if (split < rows || (ck && cut == c0 + rows)) {
                    q.memcpy(ck->conv[b.layer], ks.conv.p, ks.conv.bytes());
                    q.memcpy(ck->rec[b.layer], ks.rec.p, ks.rec.bytes());
                }
                if (split < rows)
                    kda_forward(q, lin, act, *b.kda, ks, y.ptr<half>() + (size_t)split * H, rows - split,
                                attn_out.ptr<float>() + (size_t)split * H, false);
            }
            ops::hc_apply(q, Rc, attn_out.ptr<float>(), post.ptr<float>(), comb.ptr<float>(), rows);
            tr(2, b.layer, b.is_mla ? "mla" : "kda", c0);
        }
        if (b.layer == cfg.layers - 1) break;          // the last block's MLP only feeds logits
        for (int c0 = 0; c0 < n; c0 += C) {
            int rows = std::min(C, n - c0);
            mix(b, 1, Rp + (size_t)c0 * 4 * H, rows, pall.ptr<float>() + c0 * 4, call.ptr<float>() + c0 * 16,
                yp + (size_t)c0 * H);
        }
        prof_point("prefill: mlp mix");
        const int64_t* ptrs_l = nullptr;
        if (b.is_moe) {
            // exact router over all rows, then the layer's experts staged once
            for (int c0 = 0; c0 < n; c0 += C) {
                int rows = std::min(C, n - c0);
                ops::route_exact(q, yp + (size_t)c0 * H, rows, b.router.ptr<half>(), b.bias_c.ptr<half>(),
                                 E, k, cfg.rsf, rlogits.ptr<half>(), sall.ptr<int>() + c0 * k, wall.ptr<float>() + c0 * k);
            }
            sycl::event rev = q.memcpy(sel_h.data(), sall.p, sel_h.size() * sizeof(int));
            // while the attention and router run: copy this layer's experts as their reads land
            if (stream)
                while (rev.get_info<sycl::info::event::command_execution_status>() !=
                       sycl::info::event_command_status::complete) {
                    tier_->prestage(b.tier_li);
                    std::this_thread::sleep_for(std::chrono::microseconds(200));
                }
            rev.wait();
            prof_point("prefill: route");
            tr(2, b.layer, "route", 0);
            std::vector<char> used(E, 0);
            std::vector<int> ex;
            for (int v : sel_h) if (!used[v]) { used[v] = 1; ex.push_back(v); }
            std::sort(ex.begin(), ex.end());
            if (stream) {
                ptrs_l = tier_->stage(q, b.tier_li, ex);
            } else {
                std::vector<int64_t> e64(ex.begin(), ex.end());
                tier_->ensure(q, b.tier_li, e64.data(), (int)e64.size());
                ptrs_l = tier_->ptrs(b.tier_li);
            }
            prof_point("prefill: stage (host waits)");
            tr(2, b.layer, "stage", 0);
        }
        for (int c0 = 0; c0 < n; c0 += C) {
            int rows = std::min(C, n - c0);
            const half* yc = yp + (size_t)c0 * H;
            int I = b.gate.out;
            b.gate.forward(q, lin, yc, rows, g.p, DType::F16);
            b.up.forward(q, lin, yc, rows, u.p, DType::F16);
            ops::swiglu(q, g.ptr<half>(), u.ptr<half>(), a.ptr<half>(), (size_t)rows * I);
            b.down.forward(q, lin, a.ptr<half>(), rows, b.is_moe ? sh.p : mlp_out.p, DType::F32);
            prof_point(b.is_moe ? "prefill: shared expert" : "prefill: dense MLP");
            if (b.is_moe) {
                moe_forward(q, mws, yc, rows, sall.ptr<int>() + c0 * k, wall.ptr<float>() + c0 * k, k, ptrs_l, H,
                            cfg.moe_inter, b.kc, E, moe_out.ptr<half>());
                const half* mo = moe_out.ptr<half>();
                const float* s_ = sh.ptr<float>();
                float* o = mlp_out.ptr<float>();
                q.parallel_for(sycl::range<1>((size_t)rows * H), [=](sycl::id<1> i) { o[i] = static_cast<float>(mo[i]) + s_[i]; });
                prof_point("prefill: routed experts");
            }
            ops::hc_apply(q, Rp + (size_t)c0 * 4 * H, mlp_out.ptr<float>(), pall.ptr<float>() + c0 * 4,
                          call.ptr<float>() + c0 * 16, rows);
            tr(2, b.layer, "mlp", c0);
        }
        if (stream && b.is_moe) {
            tier_->moe_done(q, b.tier_li);
            if (b.next_moe >= 0) tier_->prestage(blocks[b.next_moe].tier_li);
        }
        tr(1, b.layer, "done", 0);
    }
    if (stream) tier_->end_prompt();
    q.wait();
    g_sync_points = false;
    pos += n;
}

void Model::block_forward(Block& b, int Q) {
    auto& q = Dev::get().q;
    const bool G = graphs_on && Q == 1 && !dump;
    if (!G) {
        attn_part(b, Q);
        mlp_forward(b, Q);
        apply_part(Q);
        return;
    }
    const int key = b.layer * 4;
    if (!b.is_moe) {
        seg(key, true, [&] { attn_part(b, Q); shared_part(b, Q); apply_part(Q); });
        mark(P_MLP);
        return;
    }
    tier_->sync_bonus(q);
    sycl::event pe = seg(key, true, [&] { attn_part(b, Q); route_part(b, Q); });
    mark(P_ROUTE);
    seg(key + 1, true, [&] { shared_part(b, Q); });
    mark(P_MLP);
    pe.wait();
    auto th = std::chrono::steady_clock::now();
    int n = Q * cfg.topk;
    tier_->after_sync(b.tier_li, b.next_moe >= 0 ? blocks[b.next_moe].tier_li : -1, picks_h, picks_h + n, n,
                      picks_h + 2 * n, b.next_moe >= 0 ? Q * pfw : 0);
    // an NVMe wait: the ready picks' experts run first, the late ones after (bit-identical, see moe_split_begin)
    bool split = false;
    MoeSplit ms;
    auto early = [&](const std::vector<int64_t>& late) {
        int perm[16], j = 0;
        for (int t = 0; t < n; t++) if (!std::count(late.begin(), late.end(), picks_h[t])) perm[j++] = t;
        int n0 = j;
        if (!n0) return;
        for (int t = 0; t < n; t++) if (std::count(late.begin(), late.end(), picks_h[t])) perm[j++] = t;
        int* sp = split_buf.ptr<int>();
        ops::permute_picks(q, sel.ptr<int>(), w.ptr<float>(), perm, n, sp, reinterpret_cast<float*>(sp + 16), sp + 32);
        ms = moe_split_begin(q, mws, y.ptr<half>(), sp, reinterpret_cast<const float*>(sp + 16), sp + 32, cfg.topk, n0,
                             tier_->ptrs(b.tier_li), 4096, cfg.moe_inter, b.kc, cfg.experts);
        split = true;
    };
    if (split_on && Q == 1) tier_->ensure(q, b.tier_li, picks_h, n, early);
    else tier_->ensure(q, b.tier_li, picks_h, n);
    tier_->stats.t_host += std::chrono::duration<double>(std::chrono::steady_clock::now() - th).count();
    mark(P_TIER);
    if (split) {
        moe_split_end(q, ms, y.ptr<half>(), tier_->ptrs(b.tier_li), 4096, cfg.moe_inter, b.kc, cfg.experts,
                      moe_out.ptr<half>());
        const half* mo = moe_out.ptr<half>();
        const float* s_ = sh.ptr<float>();
        float* o = mlp_out.ptr<float>();
        q.parallel_for(sycl::range<1>(4096), [=](sycl::id<1> i) { o[i] = static_cast<float>(mo[i]) + s_[i]; });
        apply_part(Q);
    } else {
        seg(key + 2, true, [&] { moe_part(b, Q); apply_part(Q); });
    }
    mark(P_MOE);
}

const half* Model::forward(const int* tokens, int Q) {
    GLM_CHECK(Q >= 1 && Q <= qmax, "forward: %d tokens (1..%d per call)", Q, qmax);
    GLM_CHECK(pos + Q <= max_tokens, "context full (%d tokens)", max_tokens);
    auto& q = Dev::get().q;
    for (int t = 0; t < Q; t++) {
        GLM_CHECK(tokens[t] >= 0 && tokens[t] < cfg.vocab, "token id %d", tokens[t]);
        const bf16* e = embed + (size_t)tokens[t] * 4096;
        for (int d = 0; d < 4096; d++) x_host[(size_t)t * 4096 + d] = static_cast<float>(e[d]);
    }
    if (prof_on && !prof_gpu) { q.wait(); prof_t = std::chrono::steady_clock::now(); }
    if (prof_gpu) prof_ev.push_back({-1, q.ext_oneapi_submit_barrier()});
    q.memcpy(x_in.p, x_host.data(), (size_t)Q * 4096 * sizeof(float));
    q.fill(pos_dev.ptr<int>(), pos, 1);
    ops::hc_expand(q, x_in.ptr<float>(), R.ptr<float>(), Q);
    mark(P_HEAD);
    for (auto& b : blocks) block_forward(b, Q);
    seg(100000, graphs_on && Q == 1 && !dump, [&] {
        ops::hc_mean(q, R.ptr<float>(), fin.ptr<float>(), Q);
        ops::rms_norm<float, half>(q, fin.ptr<float>(), final_norm.ptr<bf16>(), fin16.ptr<half>(), Q, 4096, cfg.eps);
        lm_head.forward(q, lin, fin16.ptr<half>(), Q, logits.p, DType::F16);
    });
    mark(P_HEAD);
    if (prof_gpu) { q.wait(); prof_collect(); }
    pos += Q;
    return logits.ptr<half>();
}

}  // namespace glm

namespace glm {

void Model::probe_attn(int L, const std::vector<half>& yin, std::vector<float>& out) {
    auto& q = Dev::get().q;
    Block& b = blocks[L];
    upload(y.p, yin.data(), 4096 * 2);
    if (b.is_mla) {
        q.fill(pos_dev.ptr<int>(), 0, 1);
        mla_forward(q, lin, act, *b.mla, mla_c[L], y.ptr<half>(), 1, 0, pos_dev.ptr<int>(), attn_out.ptr<float>());
    }
    else { kda_st[L].zero(); kda_forward(q, lin, act, *b.kda, kda_st[L], y.ptr<half>(), 1, attn_out.ptr<float>(), false); }
    out.resize(4096);
    download(out.data(), attn_out.p, 4096 * 4);
}

void Model::probe_mlp(int L, const std::vector<half>& yin, std::vector<float>& out, std::vector<int>& picks) {
    Block& b = blocks[L];
    upload(y.p, yin.data(), 4096 * 2);
    mlp_forward(b, 1);
    out.resize(4096);
    download(out.data(), mlp_out.p, 4096 * 4);
    picks.assign(picks_h, picks_h + (b.is_moe ? cfg.topk : 0));
}

void Model::probe_mix(int L, int site, const std::vector<float>& Rin, std::vector<half>& yout) {
    auto& q = Dev::get().q;
    Block& b = blocks[L];
    upload(R.p, Rin.data(), 4 * 4096 * 4);
    ops::HcSite s{b.hc_fn[site].p, true, b.hc_base[site].ptr<float>(), b.hc_scale[site].ptr<float>()};
    ops::hc_mix(q, R.ptr<float>(), s, post.ptr<float>(), comb.ptr<float>(), collapsed.ptr<half>(), hc_part.ptr<float>(), 1);
    ops::rms_norm<half, half>(q, collapsed.ptr<half>(), (site ? b.post_norm : b.in_norm).ptr<bf16>(), y.ptr<half>(), 1,
                              4096, cfg.eps);
    yout.resize(4096);
    download(yout.data(), y.p, 4096 * 2);
}

}  // namespace glm
