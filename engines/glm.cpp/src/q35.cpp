#include "q35.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <sstream>
#include <sys/mman.h>
#include <thread>
#include "json.h"
#include "ops.h"
#include "q35_blocks.h"
#include "q35_ops.h"
#include "q4_ops.h"
#include "progress.h"

namespace glm {

using namespace q35;

namespace {
const std::string P = "model.language_model.";
template <class T> float f(T v) { return static_cast<float>(v); }

// host copy of a small tensor as fp32
std::vector<float> host_f32(const SafeTensors& st, const std::string& name) {
    const StTensor& t = st.at(name);
    size_t n = 1;
    for (auto s : t.shape) n *= s;
    std::vector<float> v(n);
    for (size_t i = 0; i < n; i++) {
        if (t.dt == DType::F32) v[i] = static_cast<const float*>(t.data)[i];
        else if (t.dt == DType::F16) v[i] = f(static_cast<const half*>(t.data)[i]);
        else if (t.dt == DType::BF16) v[i] = f(static_cast<const bf16*>(t.data)[i]);
        else GLM_CHECK(false, "%s: dtype", name.c_str());
    }
    return v;
}

Tensor dev_f32(const std::vector<float>& v) {
    Tensor t = Tensor::empty({(int64_t)v.size()}, DType::F32);
    upload(t.p, v.data(), v.size() * 4);
    return t;
}

Tensor norm1p(const SafeTensors& st, const std::string& name) {
    auto v = host_f32(st, name);
    for (auto& x : v) x += 1.0f;
    return dev_f32(v);
}

// two matrices stacked by rows as fp16 [rows, cols] on the device, whatever they are stored as
Tensor stack_f16(const SafeTensors& st, const std::vector<std::string>& names) {
    std::vector<half> all;
    int64_t rows = 0, cols = -1;
    for (auto& n : names) {
        const StTensor& t = st.at(n);
        GLM_CHECK(t.shape.size() == 2 && (cols < 0 || cols == t.shape[1]), "%s: shape", n.c_str());
        cols = t.shape[1];
        rows += t.shape[0];
        auto v = host_f32(st, n);
        all.insert(all.end(), v.begin(), v.end());
    }
    Tensor out = Tensor::empty({rows, cols}, DType::F16);
    upload(out.p, all.data(), all.size() * 2);
    return out;
}
}  // namespace

void Q35Cache::init(int seqs, int T) {
    k = Tensor::empty({seqs, T, AKV, AD}, DType::F16);
    v = Tensor::empty({seqs, T, AKV, AD}, DType::F16);
}

void Q35GdnState::init(int slots, int qmax) {
    conv = Tensor::zeros({slots, q4::GQKV, q4::GCONV - 1}, DType::BF16);
    rec = Tensor::zeros({slots, q4::GV, q4::GDK, q4::GDV}, DType::F32);
    kq = Tensor::empty({qmax, q4::GQKV}, DType::F32);
    kb = Tensor::empty({qmax, q4::GV}, DType::F32);
    kg = Tensor::empty({qmax, q4::GV}, DType::F32);
}

Q35Model::Q35Model(const std::string& dir, int max_tok, int seqs) : max_tokens(max_tok), seqs_(std::max(1, seqs)) {
    GLM_CHECK(seqs_.size() <= 8, "at most 8 sequences (%zu asked)", seqs_.size());
    Json c = Json::parse(read_file(dir + "/config.json"));
    const Json& tc = c["text_config"];
    GLM_CHECK(tc.get_s("model_type", "") == "qwen3_5_text" && tc.get_i("hidden_size", 0) == D &&
                  tc.get_i("num_hidden_layers", 0) > 0 && tc.get_i("intermediate_size", 0) == I &&
                  tc.get_i("num_attention_heads", 0) == AH && tc.get_i("num_key_value_heads", 0) == AKV &&
                  tc.get_i("head_dim", 0) == AD && tc.get_i("linear_num_value_heads", 0) == q4::GV &&
                  tc.get_i("linear_num_key_heads", 0) == q4::GK && tc.get_s("output_gate_type", "") == "swish",
              "%s: not the Qwen3.8-27B shape this engine is built for", dir.c_str());
    L = (int)tc.get_i("num_hidden_layers", 0);
    GLM_CHECK((int)tc["layer_types"].size() >= L, "%s: layer_types has fewer than %d entries", dir.c_str(), L);
    st = std::make_unique<SafeTensors>(dir);
    st->defer(true);                    // weights are read below in one parallel pass (SafeTensors::flush)
    auto& q = Dev::get().q;
    if (getenv("Q35_PF_ATTN")) pf_attn = atoi(getenv("Q35_PF_ATTN"));
    if (getenv("Q35_PF_GDN")) pf_gdn = atoi(getenv("Q35_PF_GDN"));
    if (getenv("Q35_DEC_ATTN")) dec_attn = atoi(getenv("Q35_DEC_ATTN"));
    if (getenv("Q35_REPLAY")) replay_mode = atoi(getenv("Q35_REPLAY"));
    if (getenv("Q35_DELTA")) q35ops::gdn_delta_variant = atoi(getenv("Q35_DELTA"));
    if (getenv("Q35_DEC_NS")) dec_ns = std::max(1, std::min(256, atoi(getenv("Q35_DEC_NS"))));
    if (getenv("GLM_PREFILL_CHUNK")) prefill_chunk = std::max(64, std::min(2048, atoi(getenv("GLM_PREFILL_CHUNK"))));

    const StTensor& et = st->at(P + "embed_tokens.weight");
    GLM_CHECK(et.dt == DType::BF16 && et.shape[0] == V && et.shape[1] == D, "embedding: expected bf16 [V, D]");
    embed = static_cast<const bf16*>(et.data);
    // its pages in now (as G4Model does), beside the weights' reading: a token's row is read on the host, and a row not
    // yet touched is a page fault (a file read on a cold page cache) inside the decode loop
    std::thread embed_in([p = et.data, n = et.bytes] { SafeTensors::populate(p, n, 4); });
    lm_head = Exl3Linear::load(*st, "lm_head");
    norm_w = norm1p(*st, P + "norm.weight");

    // MTP (Q35_MTP: drafts a step, 0 = off and not loaded)
    const int want = getenv("Q35_MTP") ? atoi(getenv("Q35_MTP")) : 2;
    const bool use_mtp = want > 0 && st->has("mtp.fc.trellis");
    if (want > 0 && !use_mtp) fprintf(stderr, "no MTP layer in %s: no drafts\n", dir.c_str());
    nslot = use_mtp ? 2 : 1;
    if (getenv("Q35_MTP_MIN_P")) mtp_min_p = (float)atof(getenv("Q35_MTP_MIN_P"));
    mtp_blocks = q35_draft_blocks();
    if (getenv("Q35_MTP_BLOCKS")) {
        mtp_blocks.clear();
        std::stringstream ss(read_file(getenv("Q35_MTP_BLOCKS")));
        for (int b; ss >> b;) if (b >= 0 && b < V / 128) mtp_blocks.push_back(b);
        GLM_CHECK(!mtp_blocks.empty(), "Q35_MTP_BLOCKS: no blocks in %s", getenv("Q35_MTP_BLOCKS"));
    }
    std::sort(mtp_blocks.begin(), mtp_blocks.end());
    mtp_blocks.erase(std::unique(mtp_blocks.begin(), mtp_blocks.end()), mtp_blocks.end());
    if (getenv("Q35_MTP_HEAD")) mtp_head = atoi(getenv("Q35_MTP_HEAD"));
    else if (!mtp_blocks.empty()) mtp_head = 0;
    if (mtp_head != 0) mtp_head = mtp_head < 0 || mtp_head > V ? V : mtp_head / 128 * 128;

    blocks.resize(L);
    gst.resize(L);
    kv.resize(L + (use_mtp ? 1 : 0));
    for (int li = 0; li < L; li++) {
        Q35Block& b = blocks[li];
        std::string p = P + "layers." + std::to_string(li) + ".";
        b.layer = li;
        b.attn = tc["layer_types"][li].s() == "full_attention";
        b.w1 = norm1p(*st, p + "input_layernorm.weight");
        b.w2 = norm1p(*st, p + "post_attention_layernorm.weight");
        if (b.attn) {
            auto a = std::make_unique<Q35Attn>();
            std::string ap = p + "self_attn.";
            a->q = Exl3Linear::load(*st, ap + "q_proj");
            a->k = Exl3Linear::load(*st, ap + "k_proj");
            a->v = Exl3Linear::load(*st, ap + "v_proj");
            a->o = Exl3Linear::load(*st, ap + "o_proj");
            GLM_CHECK(a->q.out == AH * 2 * AD && a->k.out == AKV * AD && a->v.out == AKV * AD, "%s: shapes", ap.c_str());
            a->qn = norm1p(*st, ap + "q_norm.weight");
            a->kn = norm1p(*st, ap + "k_norm.weight");
            b.att = std::move(a);
            kv[li].init(games(), max_tokens);
        } else {
            auto g = std::make_unique<Q4Gdn>();
            std::string gp = p + "linear_attn.";
            g->qkv = Exl3Linear::load(*st, gp + "in_proj_qkv");
            g->z = Exl3Linear::load(*st, gp + "in_proj_z");
            g->o = Exl3Linear::load(*st, gp + "out_proj");
            GLM_CHECK(g->qkv.out == q4::GQKV && g->z.out == q4::GV * q4::GDV && g->o.in == q4::GV * q4::GDV, "%s: shapes", gp.c_str());
            g->ab = stack_f16(*st, {gp + "in_proj_b.weight", gp + "in_proj_a.weight"});
            g->conv_w = st->load(gp + "conv1d.weight");
            GLM_CHECK(g->conv_w.dt == DType::BF16 && g->conv_w.numel() == q4::GQKV * q4::GCONV, "%s: conv1d", gp.c_str());
            auto al = host_f32(*st, gp + "A_log");
            for (auto& x : al) x = -std::exp(x);
            g->neg_a = dev_f32(al);
            g->dt_bias = dev_f32(host_f32(*st, gp + "dt_bias"));
            g->norm_w = dev_f32(host_f32(*st, gp + "norm.weight"));
            b.gdn = std::move(g);
            gst[li].init(games() * nslot, qmax);
        }
        b.g = Exl3Linear::load(*st, p + "mlp.gate_proj");
        b.u = Exl3Linear::load(*st, p + "mlp.up_proj");
        b.d = Exl3Linear::load(*st, p + "mlp.down_proj");
        load_progress("loaded layer %d/%d (%.2f GiB VRAM)", li + 1, L, Dev::get().vram_used / 1073741824.0);
    }
    if (use_mtp) load_mtp();
    st->flush();
    st->defer(false);
    embed_in.join();
    merge_mask = getenv("Q35_MERGE") ? atoi(getenv("Q35_MERGE")) : 7;
    for (auto& b : blocks) merge_linears(b);
    if (mtp) merge_linears(mtp->blk);
    if (use_mtp) {
        if (mtp_head == 0) build_draft_head();
        draft_tokens = std::min(want, qmax - 1);
        fprintf(stderr, "MTP: %d drafts a step (Q35_MTP; 0 = off)\n", draft_tokens);
    }
    load_progress_end();

    const int C = std::max(qmax, prefill_chunk);
    size_t lw = lm_head.ws_bytes(qmax);
    auto need = [&](const Exl3Linear& l) { lw = std::max({lw, l.ws_bytes(C), l.ws_bytes(qmax)}); };
    for (auto& b : blocks) {
        need(b.g); need(b.u); need(b.d); need(b.gu);
        if (b.att) { need(b.att->q); need(b.att->k); need(b.att->v); need(b.att->o); need(b.att->qkv); }
        if (b.gdn) { need(b.gdn->qkv); need(b.gdn->z); need(b.gdn->o); need(b.gdn_in); }
    }
    if (mtp) {
        const Q35Block& b = mtp->blk;
        need(mtp->fc); need(b.g); need(b.u); need(b.d); need(b.gu);
        need(b.att->q); need(b.att->k); need(b.att->v); need(b.att->o); need(b.att->qkv);
    }
    lin.init(lw + ((size_t)1 << 20));
    act.init((size_t)C * 120 * 1024 + ((size_t)64 << 20));
    X = Tensor::empty({qmax, D}, DType::F32);
    y16 = Tensor::empty({C, D}, DType::F16);
    sub = Tensor::empty({C, D}, DType::F32);
    g16 = Tensor::empty({C, I}, DType::F16);
    u16 = Tensor::empty({C, I}, DType::F16);
    a16 = Tensor::empty({C, I}, DType::F16);
    fin16 = Tensor::empty({qmax, D}, DType::F16);
    logits = Tensor::empty({qmax, V}, DType::F16);
    // softmax partials of decode steps (Q35_DEC_ATTN 0: the plain kernel's, every row at full context); prompts read with
    // the plain kernel (Q35_PF_ATTN=0 at load) get 768 MB so their pieces of rows run in few calls (a later switch to it,
    // as q35_gen kl does, cuts its rows to fit what is there)
    size_t part_floats = dec_attn ? q35ops::dec_part_floats(qmax, dec_ns) : q35ops::attn_part_floats(qmax, max_tokens / CH + 1);
    if (pf_attn == 0) part_floats = std::max<size_t>(part_floats, (size_t)192 << 20);
    part = Tensor::empty({(int64_t)part_floats}, DType::F32);
    pid_dev = Tensor::empty({(int64_t)8 * C, 3}, DType::I32);
    pos_dev = Tensor::zeros({qmax}, DType::I32);
    seq_dev = Tensor::zeros({qmax}, DType::I32);
    dec_graphs.resize(qmax + 1);
    graphs_on = !(getenv("Q35_GRAPHS") && atoi(getenv("Q35_GRAPHS")) == 0) && !g_prof_points && !g_sync_points;
    x_host = sycl::malloc_host<float>((size_t)C * D, q);
    pid_h = sycl::malloc_host<int>((size_t)8 * C * 3, q);
    pos_h = sycl::malloc_host<int>(qmax, q);
    seq_h = sycl::malloc_host<int>(qmax, q);
    GLM_CHECK(x_host && pid_h && pos_h && seq_h, "pinned upload buffers");
    desc_dev = Tensor::zeros({qmax * 3}, DType::I32);
    rdesc_dev = Tensor::zeros({qmax * 3}, DType::I32);
    desc_h = sycl::malloc_host<int>(qmax * 3, q);
    rdesc_h = sycl::malloc_host<int>(qmax * 3, q);
    GLM_CHECK(desc_h && rdesc_h, "pinned upload buffers");
    rep_graphs.resize(2 * (qmax + 1));
    if (mtp) {
        // gdn_replay's table of every GDN layer's state and saved step inputs
        std::vector<q35ops::GdnReplayLayer> t;
        for (int li = 0; li < L; li++) {
            if (blocks[li].attn) continue;
            Q35GdnState& g = gst[li];
            t.push_back({g.kq.ptr<float>(), g.kb.ptr<float>(), g.kg.ptr<float>(), g.conv.ptr<bf16>(),
                         blocks[li].gdn->conv_w.ptr<bf16>(), g.rec.ptr<float>()});
        }
        rep_nl = (int)t.size();
        rep_layers = Tensor::empty({(int64_t)(t.size() * sizeof(t[0]))}, DType::U8);
        upload(rep_layers.p, t.data(), t.size() * sizeof(t[0]));
        rep_cv = Tensor::empty({(int64_t)rep_nl * qmax * q4::GQKV}, DType::BF16);
    }
    if (mtp) {
        MQ = qmax;
        hring = Tensor::zeros({games(), RING, D}, DType::F16);
        hfin = Tensor::empty({C + 1, D}, DType::F16);
        mx = Tensor::empty({MQ, D}, DType::F32);
        mhin = Tensor::empty({MQ, D}, DType::F16);
        mh16 = Tensor::empty({MQ, D}, DType::F16);
        mlog = Tensor::empty({qmax, V}, DType::F16);
        mbest = Tensor::empty({2 * qmax}, DType::I32);
        mpos_dev = Tensor::zeros({MQ}, DType::I32);
        mseq_dev = Tensor::zeros({MQ}, DType::I32);
        mpid_dev = Tensor::zeros({MQ * 3}, DType::I32);
        msrc_dev = Tensor::zeros({MQ}, DType::I32);
        mtp_graphs.resize((size_t)(MQ + 1) * (MQ + 1));
        memb_h = sycl::malloc_host<float>((size_t)MQ * D, q);
        mpos_h = sycl::malloc_host<int>((size_t)MQ * 6, q);
        mbest_h = sycl::malloc_host<int>(2 * qmax, q);
        GLM_CHECK(memb_h && mpos_h && mbest_h, "pinned MTP buffers");
    }
    // every row count of the vector GEMV cuts K the same way: a verify step's rows sum as one-row steps do, so MTP on
    // gives MTP off's tokens; one row also takes the narrow tiles (47.8 ms a step against 50.1)
    exl3_same_split = true;
    // with MTP, decode rows take the DPAS kernel at every row count (one 4-row block for 1 .. 4 rows: the same sums
    // for each row whatever the count): 50.1 / 51.4 / 52.9 / 53.3 ms for 1 / 2 / 3 / 4 rows, against the vector
    // kernel's 47.4 / 51.3 / 63.5 / 66.5 (GLM_EXL3_VECMAX overrides)
    if (mtp) exl3_vec_max = 0;
    // a layer's decoded int8 weights stay for the rest of its prompt pieces (the largest set: the MTP layer's 8 linears,
    // ~424 MB; a GDN layer's 6 ~383 MB, an attention layer's 7 ~372 MB)
    exl3_i8_cache_mb = 448;
    q.wait();
    if (!getenv("Q35_WARMUP") || atoi(getenv("Q35_WARMUP")) != 0) {
        auto tw = std::chrono::steady_clock::now();
        std::vector<int> t(std::min(300, max_tokens - 1), 220);
        prefill(t.data(), (int)t.size());
        forward(t.data(), 1);
        if (mtp && max_tokens > 320) {
            // the verify steps' graphs (1 + drafts rows) and a replay, recorded now rather than in the first reply
            for (int k = 1; k <= draft_tokens; k++) {
                std::vector<std::vector<int>> d;
                int tk = 220, nn = k, rw = 0;
                draft_rows(&cur_, &tk, &rw, &nn, 1, d);
                std::vector<int> sq(k + 1, cur_), tt(k + 1, 220);
                forward_rows(sq.data(), tt.data(), k + 1);
                restore(position() - k);
            }
        }
        q.wait();
        reset();
        fprintf(stderr, "warm-up %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - tw).count());
    }
    fprintf(stderr, "model ready: %.2f GiB VRAM\n", Dev::get().vram_used / 1073741824.0);
}

void Q35Model::merge_linears(Q35Block& b) {
    if ((merge_mask & 1) && Exl3Linear::can_merge({&b.g, &b.u})) b.gu = Exl3Linear::merge({&b.g, &b.u});
    if ((merge_mask & 4) && b.att && Exl3Linear::can_merge({&b.att->q, &b.att->k, &b.att->v}))
        b.att->qkv = Exl3Linear::merge({&b.att->q, &b.att->k, &b.att->v});
    if ((merge_mask & 2) && b.gdn && Exl3Linear::can_merge({&b.gdn->qkv, &b.gdn->z}))
        b.gdn_in = Exl3Linear::merge({&b.gdn->qkv, &b.gdn->z});
}

Q35Model::~Q35Model() {
    auto& q = Dev::get().q;
    q.wait();
    for (auto& se : seqs_)
        for (auto& c : se.ckpts) {
            for (void* p : c.conv) if (p) sycl::free(p, q);
            for (void* p : c.rec) if (p) sycl::free(p, q);
        }
    for (auto& se : seqs_)
        for (auto& c : se.ckpts) if (c.hid) sycl::free(c.hid, q);
    dec_graphs.clear();
    rep_graphs.clear();
    mtp_graphs.clear();
    for (void* p : {(void*)desc_h, (void*)rdesc_h, (void*)memb_h, (void*)mpos_h, (void*)mbest_h}) if (p) sycl::free(p, q);
    if (x_host) sycl::free(x_host, q);
    if (pid_h) sycl::free(pid_h, q);
    if (pos_h) sycl::free(pos_h, q);
    if (seq_h) sycl::free(seq_h, q);
}

void Q35Model::set_images(std::vector<Q35Image> imgs) { seqs_[cur_].images = std::move(imgs); }

void Q35Model::grab(const float* p, int n) {
    if (!dump) return;
    auto& q = Dev::get().q;
    q.wait();
    size_t o = dump->size();
    dump->resize(o + n);
    q.memcpy(dump->data() + o, p, n * 4).wait();
}

void Q35Model::rows_host(Seq& se, const int* tokens, int n, float* dst, int* pid) {
    for (int t = 0; t < n; t++) {
        int* pr = pid + (size_t)t * 3;
        float* e = dst + (size_t)t * D;
        if (tokens[t] == IMAGE_PAD) {
            GLM_CHECK(se.img_i < (int)se.images.size(), "the prompt has more image tokens than its images hold");
            const Q35Image& im = se.images[se.img_i];
            pr[0] = se.rope_next;
            pr[1] = se.rope_next + se.img_row / im.cols;
            pr[2] = se.rope_next + se.img_row % im.cols;
            std::memcpy(e, im.emb.data() + (size_t)se.img_row * D, D * sizeof(float));
            if (++se.img_row == im.rows * im.cols) {
                se.rope_next += std::max(im.rows, im.cols);
                se.img_i++;
                se.img_row = 0;
            }
        } else {
            GLM_CHECK(se.img_row == 0, "an image's tokens were cut short");
            pr[0] = pr[1] = pr[2] = se.rope_next++;
            const bf16* w = embed + (size_t)tokens[t] * D;
            for (int d = 0; d < D; d++) e[d] = f(w[d]);
        }
    }
}

void Q35Model::gdn_forward(const Q4Gdn& g, const Exl3Linear& gin, Q35GdnState& s, const half* y, int rows, float* out,
                           const int* desc) {
    using namespace q4;
    auto& q = Dev::get().q;
    act.reset();
    // decode rows leave their inputs where a replay (restore_window) finds them
    float* qkv = desc ? s.kq.ptr<float>() : act.take<float>((size_t)rows * GQKV);
    float* z = act.take<float>((size_t)rows * GV * GDV);
    float* ab = act.take<float>((size_t)rows * 2 * GV);
    float* beta = desc ? s.kb.ptr<float>() : act.take<float>((size_t)rows * GV);
    float* gg = desc ? s.kg.ptr<float>() : act.take<float>((size_t)rows * GV);
    bf16* cv = act.take<bf16>((size_t)rows * GQKV);
    bf16* core = act.take<bf16>((size_t)rows * GV * GDV);
    half* nrm = act.take<half>((size_t)rows * GV * GDV);
    if (gin.S > 1) {
        void* ys[2] = {qkv, z};
        gin.forward_split(q, lin, y, rows, ys, DType::F32);
    } else {
        g.qkv.forward(q, lin, y, rows, qkv, DType::F32);
        g.z.forward(q, lin, y, rows, z, DType::F32);
    }
    prof_point("GDN: qkv, z");
    ops::gemv_f16<float>(q, y, rows, q35::D, g.ab.ptr<half>(), 2 * GV, ab);
    q4ops::gdn_gates(q, ab, g.neg_a.ptr<float>(), g.dt_bias.ptr<float>(), beta, gg, rows);
    if (desc) {
        q35ops::gdn_conv_seq(q, qkv, s.conv.ptr<bf16>(), g.conv_w.ptr<bf16>(), cv, rows, desc);
        prof_point("GDN: a b gates conv");
        q35ops::gdn_delta_seq(q, cv, beta, gg, s.rec.ptr<float>(), core, rows, desc);
    } else {
        const size_t sl = cslot(cur_);
        bf16* cst = s.conv.ptr<bf16>() + sl * GQKV * (GCONV - 1);
        if (pf_gdn) q35ops::gdn_conv_prompt(q, qkv, cst, g.conv_w.ptr<bf16>(), cv, rows);
        else q4ops::gdn_conv(q, qkv, cst, g.conv_w.ptr<bf16>(), cv, rows);
        prof_point("GDN: a b gates conv");
        float* rec = s.rec.ptr<float>() + sl * GV * GDK * GDV;
        if (pf_gdn) q35ops::gdn_chunked(q, cv, beta, gg, rec, core, rows, act.take<char>(q35ops::gdn_chunk_bytes(rows)));
        else q4ops::gdn_delta(q, cv, beta, gg, rec, core, rows);
    }
    prof_point("GDN: delta rule");
    q35ops::gdn_norm(q, core, z, g.norm_w.ptr<float>(), nrm, rows);
    g.o.forward(q, lin, nrm, rows, out, DType::F32);
    prof_point("GDN: norm + out_proj");
}

void Q35Model::attn_forward(const Q35Attn& a, Q35Cache& c, const half* y, int rows, int p0, const int* pid, float* out,
                            const int* pdev, const int* sdev) {
    auto& q = Dev::get().q;
    act.reset();
    half* qg = act.take<half>((size_t)rows * AH * 2 * AD);
    half* k = act.take<half>((size_t)rows * AKV * AD);
    half* v = act.take<half>((size_t)rows * AKV * AD);
    half* qn16 = act.take<half>((size_t)rows * AH * AD);
    half* o16 = act.take<half>((size_t)rows * AH * AD);
    if (a.qkv.S > 1) {
        void* ys[3] = {qg, k, v};
        a.qkv.forward_split(q, lin, y, rows, ys, DType::F16);
    } else {
        a.q.forward(q, lin, y, rows, qg, DType::F16);
        a.k.forward(q, lin, y, rows, k, DType::F16);
        a.v.forward(q, lin, y, rows, v, DType::F16);
    }
    prof_point("attn: projections");
    // a prompt's rows go to the selected sequence's cache; decode rows find theirs through sdev
    const size_t base = pdev ? 0 : (size_t)cur_ * max_tokens * AKV * AD;
    half* Kc = c.k.ptr<half>() + base;
    half* Vc = c.v.ptr<half>() + base;
    q35ops::attn_prep(q, qg, k, v, a.qn.ptr<float>(), a.kn.ptr<float>(), rows, p0, pid, qn16, Kc, Vc, pdev, sdev, max_tokens);
    float* pt = part.ptr<float>();
    size_t cap = (size_t)part.numel();
    if (pdev && dec_attn) {
        q35ops::attn_dec(q, qn16, Kc, Vc, rows, pdev, sdev, max_tokens, pt, dec_ns);
        q35ops::attn_dec_combine(q, pt, rows, dec_ns, qg, o16);
    } else if (pdev) {
        q35ops::attn_core(q, qn16, Kc, Vc, qg, rows, p0, max_tokens / CH + 1, pt, o16, pdev, sdev, max_tokens);
    } else if (pf_attn) {
        half* kt = nullptr;
        half* vt = nullptr;
        const int TS = (max_tokens + 15) / 16 * 16;          // a 2D surface's pitch: a multiple of 64 bytes
        if (pf_attn == 13 || pf_attn == 15) {
            const bool v2 = pf_attn == 15;
            if (!ktp.p) ktp = Tensor::empty({(int64_t)AKV * AD * TS}, DType::F16);
            if (v2 && !vtp.p) vtp = Tensor::empty({(int64_t)AKV * AD * TS}, DType::F16);
            kt = ktp.ptr<half>();
            vt = v2 ? vtp.ptr<half>() : nullptr;
            if (kt_cache != &c || kt_seq != cur_ || kt_valid != p0) {
                q35ops::kt_fill(q, Kc, kt, TS, 0, p0);
                if (v2) q35ops::vt_fill(q, Vc, vt, TS, 0, p0);
            }
            q35ops::kt_fill(q, Kc, kt, TS, p0, p0 + rows);
            if (v2) q35ops::vt_fill(q, Vc, vt, TS, p0, p0 + rows);
            kt_cache = &c;
            kt_seq = cur_;
            kt_valid = p0 + rows;
        }
        q35ops::attn_prompt(q, qn16, Kc, Vc, rows, p0, o16, pf_attn, kt, TS, vt);
        q35ops::attn_gate(q, qg, o16, rows);
    } else {
        for (int r0 = 0; r0 < rows;) {
            int nch = (p0 + rows - 1) / CH + 1, nr = rows - r0;
            nr = std::min<size_t>(nr, std::max<size_t>(1, cap / q35ops::attn_part_floats(1, nch)));
            q35ops::attn_core(q, qn16 + (size_t)r0 * AH * AD, Kc, Vc, qg + (size_t)r0 * AH * 2 * AD, nr, p0 + r0, nch, pt,
                              o16 + (size_t)r0 * AH * AD);
            r0 += nr;
        }
    }
    prof_point("attn: softmax");
    a.o.forward(q, lin, o16, rows, out, DType::F32);
    prof_point("attn: o_proj");
}

void Q35Model::block_forward(Q35Block& b, float* Xd, int rows, int p0, const int* pid, const int* pdev, const int* sdev,
                             bool add_first, bool defer_last) {
    auto& q = Dev::get().q;
    // add_first: sub holds the block before's MLP output, not yet added (decode steps fold that add into this norm)
    if (add_first) q35ops::add_norm16(q, Xd, sub.ptr<float>(), b.w1.ptr<float>(), y16.ptr<half>(), rows);
    else q35ops::norm16(q, Xd, b.w1.ptr<float>(), y16.ptr<half>(), rows);
    if (b.att) attn_forward(*b.att, kv[b.layer], y16.ptr<half>(), rows, p0, pid, sub.ptr<float>(), pdev, sdev);
    else gdn_forward(*b.gdn, b.gdn_in, gst[b.layer], y16.ptr<half>(), rows, sub.ptr<float>(), pdev ? desc_dev.ptr<int>() : nullptr);
    q35ops::add_norm16(q, Xd, sub.ptr<float>(), b.w2.ptr<float>(), y16.ptr<half>(), rows);
    if (b.gu.S > 1) {
        void* ys[2] = {g16.p, u16.p};
        b.gu.forward_split(q, lin, y16.ptr<half>(), rows, ys, DType::F16);
    } else {
        b.g.forward(q, lin, y16.ptr<half>(), rows, g16.p, DType::F16);
        b.u.forward(q, lin, y16.ptr<half>(), rows, u16.p, DType::F16);
    }
    q4ops::swiglu(q, g16.ptr<half>(), u16.ptr<half>(), a16.ptr<half>(), (size_t)rows * I);
    b.d.forward(q, lin, a16.ptr<half>(), rows, sub.p, DType::F32);
    // defer_last: the residual add is left to the next norm (add_norm16), the output stays in sub
    if (!defer_last || dump) q35ops::add(q, Xd, sub.ptr<float>(), rows);
    prof_point("MLP");
    if (b.layer < L) grab(Xd, D);
}

void Q35Model::forward_dev(int Q) {
    auto& q = Dev::get().q;
    float* Xd = X.ptr<float>();
    q.memcpy(Xd, x_host, (size_t)Q * D * 4);
    q.memcpy(pid_dev.p, pid_h, (size_t)Q * 3 * 4);
    q.memcpy(pos_dev.p, pos_h, (size_t)Q * sizeof(int));
    q.memcpy(seq_dev.p, seq_h, (size_t)Q * sizeof(int));
    q.memcpy(desc_dev.p, desc_h, (size_t)Q * 3 * sizeof(int));
    // each block's MLP output is added by the next norm (one launch for add + norm)
    const bool fold = !dump;
    for (auto& b : blocks)
        block_forward(b, Xd, Q, 0, pid_dev.ptr<int>(), pos_dev.ptr<int>(), seq_dev.ptr<int>(), fold && b.layer > 0, fold);
    if (fold) q35ops::add_norm16(q, Xd, sub.ptr<float>(), norm_w.ptr<float>(), fin16.ptr<half>(), Q);
    else q35ops::norm16(q, Xd, norm_w.ptr<float>(), fin16.ptr<half>(), Q);
    lm_head.forward(q, lin, fin16.ptr<half>(), Q, logits.p, DType::F16);
    if (mtp) q35ops::put_rows(q, fin16.ptr<half>(), Q, hring.ptr<half>(), pos_dev.ptr<int>(), seq_dev.ptr<int>(), RING);
    prof_point("norm + lm_head");
}

const half* Q35Model::forward_rows(const int* seq, const int* tokens, int Q) {
    GLM_CHECK(Q >= 1 && Q <= qmax, "forward: %d rows (at most %d)", Q, qmax);
    auto& q = Dev::get().q;
    dec_ev.wait();
    rep_ev.wait();
    if (!graphs_on || dump) q.wait();   // a step outside a graph may still read the upload buffers
    x_ev.wait();
    for (auto& se : seqs_) se.win_p0 = -1;
    win_Q = Q;
    for (int r = 0; r < Q; r++) {
        const int s = seq[r];
        Seq& se = seqs_[s];
        GLM_CHECK(se.pos < max_tokens, "context full (%d tokens)", max_tokens);
        int* dh = desc_h + (size_t)r * 3;
        if (r == 0 || seq[r - 1] != s) {
            // a sequence's rows run in order from its live state slot; with two slots the state after them goes to
            // the other one, and the one read stays for a replay
            for (int o = 0; o < r; o++) GLM_CHECK(seq[o] != s, "forward_rows: a sequence's rows together");
            int n = 1;
            while (r + n < Q && seq[r + n] == s) n++;
            GLM_CHECK(n == 1 || nslot == 2, "forward_rows: %d rows of one sequence need MTP's state slots", n);
            dh[0] = n;
            dh[1] = cslot(s);
            if (nslot == 2) {
                se.slot ^= 1;
                se.win_p0 = se.pos;
                se.win_first = r;
                se.win_n = n;
                se.win_rope = se.rope_next;
            }
            dh[2] = cslot(s);
        } else {
            dh[0] = dh[1] = dh[2] = 0;
        }
        rows_host(se, tokens + r, 1, x_host + (size_t)r * D, pid_h + (size_t)r * 3);
        pos_h[r] = se.pos;
        seq_h[r] = s;
        if (mtp) {
            const int sl = se.pos % RING;
            se.ring_pos[sl] = se.pos;
            se.ring_tok[sl] = tokens[r];
            for (int a = 0; a < 3; a++) se.ring_pid[sl][a] = pid_h[(size_t)r * 3 + a];
        }
        se.pos++;
    }
    if (graphs_on && !dump) {
        namespace ex = sycl::ext::oneapi::experimental;
        if (!dec_graphs[Q]) {
            ex::command_graph<ex::graph_state::modifiable> g{q.get_context(), q.get_device()};
            g.begin_recording(q);
            forward_dev(Q);
            g.end_recording(q);
            dec_graphs[Q] = std::make_unique<ExecGraph>(g.finalize());
        }
        dec_ev = q.ext_oneapi_graph(*dec_graphs[Q]);
    } else {
        forward_dev(Q);
    }
    return logits.ptr<half>();
}

const half* Q35Model::forward(const int* tokens, int Q) {
    GLM_CHECK(Q == 1, "forward: one row a step (no drafter)");
    return forward_rows(&cur_, tokens, 1);
}

void Q35Model::prefill(const int* tokens, int n, int ckpt_at) {
    if (n <= 0) return;
    Seq& se = seqs_[cur_];
    int cut = ckpt_at > se.pos && ckpt_at <= se.pos + n ? ckpt_at - se.pos : -1;
    if (cut > 0 && cut < n) {
        prefill(tokens, cut, ckpt_at);
        prefill(tokens + cut, n - cut, -1);
        return;
    }
    GLM_CHECK(se.pos + n <= max_tokens, "context full (%d tokens)", max_tokens);
    auto& q = Dev::get().q;
    dec_ev.wait();
    rep_ev.wait();
    se.win_p0 = -1;
    kt_cache = nullptr;                 // decode steps may have written keys since: the pair-transposed copy starts over
    if (mtp) {
        // MTP K/V of decoded positions no draft caught up (the last step's rows), then the prompt's own below
        if (se.mtp_done < se.pos) {
            std::vector<MRow> rows;
            mtp_catchup(cur_, se.pos, rows);
            if (!rows.empty()) mtp_decode(rows, 0);
        }
        se.mtp_done = se.pos;
    }
    const int C = prefill_chunk, SC = 8 * C;
    Tensor Xall = Tensor::empty({std::min(n, SC), D}, DType::F32);
    for (int s0 = 0; s0 < n; s0 += SC) {
        int rows = std::min(SC, n - s0);
        float* Xp = Xall.ptr<float>();
        for (int c0 = 0; c0 < rows; c0 += C) {
            int nr = std::min(C, rows - c0);
            x_ev.wait();
            rows_host(se, tokens + s0 + c0, nr, x_host, pid_h + (size_t)c0 * 3);
            x_ev = q.memcpy(Xp + (size_t)c0 * D, x_host, (size_t)nr * D * 4);
        }
        x_ev.wait();
        q.memcpy(pid_dev.p, pid_h, (size_t)rows * 3 * 4).wait();
        for (auto& b : blocks) {
            for (int c0 = 0; c0 < rows; c0 += C) {
                int nr = std::min(C, rows - c0);
                block_forward(b, Xp + (size_t)c0 * D, nr, se.pos + c0, pid_dev.ptr<int>() + (size_t)c0 * 3, nullptr, nullptr);
            }
            q.wait();                   // keep the backlog of a long prompt short
        }
        if (mtp) {
            // MTP rows: position p takes token p and the hidden state of p - 1 (hfin row 0: the one before the piece)
            half* hf = hfin.ptr<half>();
            if (s0 == 0) {
                const int pp = se.pos - 1;
                if (pp >= 0 && se.ring_pos[pp % RING] == pp)
                    q.memcpy(hf, hring.ptr<half>() + ((size_t)cur_ * RING + pp % RING) * D, (size_t)D * 2);
                else
                    q.memset(hf, 0, (size_t)D * 2);
            }
            for (int c0 = 0; c0 < rows; c0 += C) {
                int nr = std::min(C, rows - c0);
                float* Xm = Xp + (size_t)c0 * D;
                q35ops::norm16(q, Xm, norm_w.ptr<float>(), hf + D, nr);   // hidden of the piece's rows, hfin 1 .. nr
                mtp_prefill_rows(tokens + s0 + c0, nr, se.pos + c0, pid_dev.ptr<int>() + (size_t)c0 * 3, Xm);
                for (int i = 0; i < nr; i++) {
                    const int t = s0 + c0 + i;
                    if (t >= n - RING) ring_put(cur_, se.pos + c0 + i, tokens[t], pid_h + (size_t)(c0 + i) * 3, hf + (size_t)(i + 1) * D);
                }
                q.memcpy(hf, hf + (size_t)nr * D, (size_t)D * 2);
            }
            q.wait();
        }
        se.pos += rows;
    }
    if (mtp) se.mtp_done = se.pos;
    q.wait();
    if (cut == n) save_checkpoint(se, new_checkpoint(se, se.pos));
}

void Q35Model::reset() {
    auto& q = Dev::get().q;
    dec_ev.wait();
    rep_ev.wait();
    Seq& se = seqs_[cur_];
    se.pos = 0;
    se.rope_next = se.img_i = se.img_row = 0;
    se.win_p0 = -1;
    se.mtp_done = 0;
    for (auto& p : se.ring_pos) p = -1;
    for (auto& c : se.ckpts) c.pos = -1;
    const size_t cb = conv_slot_bytes(), rb = rec_slot_bytes(), sl = cslot(cur_);
    for (int li = 0; li < L; li++) {
        if (blocks[li].attn) continue;
        q.memset(static_cast<char*>(gst[li].conv.p) + cb * sl, 0, cb);
        q.memset(static_cast<char*>(gst[li].rec.p) + rb * sl, 0, rb);
    }
    q.wait();
}

Q35Model::Checkpoint& Q35Model::new_checkpoint(Seq& se, int at) {
    auto& q = Dev::get().q;
    if (se.ckpts.size() >= 2) {
        Checkpoint c = std::move(se.ckpts.front());
        se.ckpts.erase(se.ckpts.begin());
        se.ckpts.push_back(std::move(c));
    } else {
        Checkpoint c;
        for (int li = 0; li < L; li++) {
            if (blocks[li].attn) { c.conv.push_back(nullptr); c.rec.push_back(nullptr); continue; }
            c.conv.push_back(sycl::malloc_host(conv_slot_bytes(), q));
            c.rec.push_back(sycl::malloc_host(rec_slot_bytes(), q));
            GLM_CHECK(c.conv.back() && c.rec.back(), "checkpoint: pinned allocation failed");
        }
        if (mtp) {
            c.hid = sycl::malloc_host((size_t)D * 2, q);
            GLM_CHECK(c.hid, "checkpoint: pinned allocation failed");
        }
        se.ckpts.push_back(std::move(c));
    }
    se.ckpts.back().pos = at;
    return se.ckpts.back();
}

void Q35Model::save_checkpoint(Seq& se, Checkpoint& c) {
    auto& q = Dev::get().q;
    const size_t cb = conv_slot_bytes(), rb = rec_slot_bytes(), sl = cslot(cur_);
    for (int li = 0; li < L; li++) {
        if (blocks[li].attn) continue;
        q.memcpy(c.conv[li], static_cast<char*>(gst[li].conv.p) + cb * sl, cb);
        q.memcpy(c.rec[li], static_cast<char*>(gst[li].rec.p) + rb * sl, rb);
    }
    c.hid_ok = false;
    if (mtp && se.pos >= 1 && se.ring_pos[(se.pos - 1) % RING] == se.pos - 1) {
        // the hidden state the next prompt's first MTP row needs
        const int r = (se.pos - 1) % RING;
        q.memcpy(c.hid, hring.ptr<half>() + ((size_t)cur_ * RING + r) * D, (size_t)D * 2);
        c.hid_ok = true;
        c.htok = se.ring_tok[r];
        for (int a = 0; a < 3; a++) c.hpid[a] = se.ring_pid[r][a];
    }
    c.rope = se.rope_next;
    c.img = se.img_i;
    c.img_row = se.img_row;
    q.wait();
}

bool Q35Model::restore(int upto) {
    auto& q = Dev::get().q;
    dec_ev.wait();
    rep_ev.wait();
    Seq& se = seqs_[cur_];
    if (restore_window(se, upto)) return true;
    se.win_p0 = -1;
    for (int i = (int)se.ckpts.size() - 1; i >= 0; i--) {
        Checkpoint& c = se.ckpts[i];
        if (c.pos < 0 || c.pos > upto) continue;
        const size_t cb = conv_slot_bytes(), rb = rec_slot_bytes(), sl = cslot(cur_);
        for (int li = 0; li < L; li++) {
            if (blocks[li].attn) continue;
            q.memcpy(static_cast<char*>(gst[li].conv.p) + cb * sl, c.conv[li], cb);
            q.memcpy(static_cast<char*>(gst[li].rec.p) + rb * sl, c.rec[li], rb);
        }
        if (mtp) {
            // the ring past the checkpoint describes a future this sequence left; MTP K/V below it is still right
            for (int r = 0; r < RING; r++) if (se.ring_pos[r] >= c.pos) se.ring_pos[r] = -1;
            se.mtp_done = std::min(se.mtp_done, c.pos);
            if (c.hid_ok) {
                const int r = (c.pos - 1) % RING;
                q.memcpy(hring.ptr<half>() + ((size_t)cur_ * RING + r) * D, c.hid, (size_t)D * 2);
                se.ring_pos[r] = c.pos - 1;
                se.ring_tok[r] = c.htok;
                for (int a = 0; a < 3; a++) se.ring_pid[r][a] = c.hpid[a];
            }
        }
        se.rope_next = c.rope;
        se.img_i = c.img;
        se.img_row = c.img_row;
        q.wait();
        se.pos = c.pos;
        // checkpoints above this one no longer describe this sequence's past; their buffers stay for reuse
        for (size_t k = i + 1; k < se.ckpts.size(); k++) se.ckpts[k].pos = -1;
        return true;
    }
    return false;
}

// ------------------------------------------------------------------------------------------------ verify windows

uint64_t Q35Model::state_hash() {
    auto& q = Dev::get().q;
    dec_ev.wait();
    rep_ev.wait();
    q.wait();
    const size_t cb = conv_slot_bytes(), rb = rec_slot_bytes(), sl = cslot(cur_);
    std::vector<uint64_t> buf(rb / 8);
    uint64_t h = 1469598103934665603ull;
    for (int li = 0; li < L; li++) {
        if (blocks[li].attn) continue;
        q.memcpy(buf.data(), static_cast<char*>(gst[li].conv.p) + cb * sl, cb).wait();
        for (size_t i = 0; i < cb / 8; i++) h = (h ^ buf[i]) * 1099511628211ull;
        q.memcpy(buf.data(), static_cast<char*>(gst[li].rec.p) + rb * sl, rb).wait();
        for (size_t i = 0; i < rb / 8; i++) h = (h ^ buf[i]) * 1099511628211ull;
    }
    return h;
}

void Q35Model::replay_dev(int Q) {
    using namespace q4;
    auto& q = Dev::get().q;
    q.memcpy(rdesc_dev.p, rdesc_h, (size_t)Q * 3 * sizeof(int));
    if (replay_mode) {
        // every GDN layer in two launches (the per-layer pair was 144 launches, 2.0 ms)
        q35ops::gdn_replay(q, static_cast<const q35ops::GdnReplayLayer*>(rep_layers.p), rep_nl, rep_cv.ptr<bf16>(), Q,
                           rdesc_dev.ptr<int>());
        return;
    }
    act.reset();
    bf16* cv = act.take<bf16>((size_t)Q * GQKV);
    bf16* core = act.take<bf16>((size_t)Q * GV * GDV);
    for (int li = 0; li < L; li++) {
        if (blocks[li].attn) continue;
        Q35GdnState& g = gst[li];
        q35ops::gdn_conv_seq(q, g.kq.ptr<float>(), g.conv.ptr<bf16>(), blocks[li].gdn->conv_w.ptr<bf16>(), cv, Q,
                             rdesc_dev.ptr<int>());
        q35ops::gdn_delta_seq(q, cv, g.kb.ptr<float>(), g.kg.ptr<float>(), g.rec.ptr<float>(), core, Q, rdesc_dev.ptr<int>());
    }
}

// The last step's first a rows of this sequence again, from the state slot it read, into its live slot: the GDN state
// after the last accepted row. Attention K/V needs nothing (rows past the position are not read).
bool Q35Model::restore_window(Seq& se, int upto) {
    if (nslot < 2 || se.win_p0 < 0 || upto <= se.win_p0 || upto >= se.win_p0 + se.win_n) return false;
    auto& q = Dev::get().q;
    const int s = (int)(&se - seqs_.data()), a = upto - se.win_p0, Q = win_Q;
    for (int r = 0; r < Q * 3; r++) rdesc_h[r] = 0;
    rdesc_h[se.win_first * 3] = a;
    rdesc_h[se.win_first * 3 + 1] = s * nslot + (se.slot ^ 1);
    rdesc_h[se.win_first * 3 + 2] = cslot(s);
    if (graphs_on && !dump) {
        namespace ex = sycl::ext::oneapi::experimental;
        auto& rg = rep_graphs[(replay_mode ? qmax + 1 : 0) + Q];
        if (!rg) {
            ex::command_graph<ex::graph_state::modifiable> g{q.get_context(), q.get_device()};
            g.begin_recording(q);
            replay_dev(Q);
            g.end_recording(q);
            rg = std::make_unique<ExecGraph>(g.finalize());
        }
        rep_ev = q.ext_oneapi_graph(*rg);
    } else {
        replay_dev(Q);
        q.wait();
    }
    se.pos = upto;
    se.rope_next = se.win_rope + a;       // decode rows are text: one rope position each
    se.win_n = a;                         // a shorter prefix can still be had from the same rows
    if (mtp) {
        for (int r = 0; r < RING; r++) if (se.ring_pos[r] >= upto) se.ring_pos[r] = -1;
        se.mtp_done = std::min(se.mtp_done, upto);
    }
    return true;
}

// ------------------------------------------------------------------------------------------------ MTP

void Q35Model::load_mtp() {
    auto m = std::make_unique<Q35Mtp>();
    m->fc = Exl3Linear::load(*st, "mtp.fc");
    GLM_CHECK(m->fc.in == 2 * D && m->fc.out == D, "mtp.fc: shape");
    m->pe = norm1p(*st, "mtp.pre_fc_norm_embedding.weight");
    m->ph = norm1p(*st, "mtp.pre_fc_norm_hidden.weight");
    m->fn = norm1p(*st, "mtp.norm.weight");
    Q35Block& b = m->blk;
    const std::string p = "mtp.layers.0.";
    b.layer = L;
    b.attn = true;
    b.w1 = norm1p(*st, p + "input_layernorm.weight");
    b.w2 = norm1p(*st, p + "post_attention_layernorm.weight");
    auto a = std::make_unique<Q35Attn>();
    a->q = Exl3Linear::load(*st, p + "self_attn.q_proj");
    a->k = Exl3Linear::load(*st, p + "self_attn.k_proj");
    a->v = Exl3Linear::load(*st, p + "self_attn.v_proj");
    a->o = Exl3Linear::load(*st, p + "self_attn.o_proj");
    GLM_CHECK(a->q.out == AH * 2 * AD && a->k.out == AKV * AD && a->v.out == AKV * AD, "mtp attention: shapes");
    a->qn = norm1p(*st, p + "self_attn.q_norm.weight");
    a->kn = norm1p(*st, p + "self_attn.k_norm.weight");
    b.att = std::move(a);
    b.g = Exl3Linear::load(*st, p + "mlp.gate_proj");
    b.u = Exl3Linear::load(*st, p + "mlp.up_proj");
    b.d = Exl3Linear::load(*st, p + "mlp.down_proj");
    kv[L].init(games(), max_tokens);
    mtp = std::move(m);
    fprintf(stderr, "MTP layer loaded: %.2f GiB VRAM\n", Dev::get().vram_used / 1073741824.0);
}

void Q35Model::build_draft_head() {
    // the lm_head's column tiles of each block (8 tiles of 16 columns), its output scales; the input side is the same
    auto& q = Dev::get().q;
    const int nb = (int)mtp_blocks.size(), tn = nb * 8, rows_t = lm_head.in / 16, src_tn = lm_head.out / 16;
    const int words = (int)lm_head.trellis.size(2);
    std::vector<int> map(tn);
    for (int j = 0; j < tn; j++) map[j] = mtp_blocks[j / 8] * 8 + j % 8;
    Tensor mdev = Tensor::empty({tn}, DType::I32);
    upload(mdev.p, map.data(), map.size() * 4);
    Exl3Linear& h = draft_head;
    h.trellis = Tensor::empty({rows_t, tn, words}, DType::I16);
    q35ops::gather_col_tiles(q, lm_head.trellis.ptr<uint16_t>(), h.trellis.ptr<uint16_t>(), rows_t, src_tn, tn, words,
                             mdev.ptr<int>());
    h.suh = Tensor::empty({lm_head.in}, DType::F16);
    q.memcpy(h.suh.p, lm_head.suh.p, (size_t)lm_head.in * 2);
    h.svh = Tensor::empty({tn * 16}, DType::F16);
    for (int b = 0; b < nb; b++)
        q.memcpy(h.svh.ptr<half>() + (size_t)b * 128, lm_head.svh.ptr<half>() + (size_t)mtp_blocks[b] * 128, 256);
    h.sonb = Tensor::zeros({nb}, DType::I32);
    h.K = lm_head.K;
    h.cb = lm_head.cb;
    h.in = lm_head.in;
    h.out = tn * 16;
    q.wait();
    fprintf(stderr, "MTP draft head: %d blocks of 128 ids (%d columns, %.0f MB)\n", nb, h.out, h.trellis.bytes() / 1048576.0);
}

void Q35Model::emb_row(int token, float* dst) const {
    // an image token's MTP row takes the pad token's own embedding (the page's is not kept)
    const bf16* w = embed + (size_t)token * D;
    for (int d = 0; d < D; d++) dst[d] = f(w[d]);
}

void Q35Model::ring_put(int s, int p, int tok, const int* pid, const half* hid_dev) {
    Seq& se = seqs_[s];
    const int r = p % RING;
    Dev::get().q.memcpy(hring.ptr<half>() + ((size_t)s * RING + r) * D, hid_dev, (size_t)D * 2);
    se.ring_pos[r] = p;
    se.ring_tok[r] = tok;
    for (int a = 0; a < 3; a++) se.ring_pid[r][a] = pid[a];
}

void Q35Model::mtp_catchup(int s, int to, std::vector<MRow>& rows) {
    Seq& se = seqs_[s];
    for (int p = std::max({se.mtp_done, to - RING + 1, 0}); p < to; p++) {
        const int r = p % RING;
        if (se.ring_pos[r] != p) continue;
        int h = -1;                                    // position 0: a zero hidden state, as exllamav3
        if (p > 0) {
            const int rp = (p - 1) % RING;
            if (se.ring_pos[rp] != p - 1) continue;
            h = s * RING + rp;
        }
        rows.push_back({s, p, se.ring_tok[r], {se.ring_pid[r][0], se.ring_pid[r][1], se.ring_pid[r][2]}, h});
    }
}

void Q35Model::mtp_pass_dev(int n, int nhead) {
    auto& q = Dev::get().q;
    const Q35Mtp& m = *mtp;
    const int* ph = mpos_h;
    float* xm = mx.ptr<float>();
    half* hin = mhin.ptr<half>();
    q.memcpy(xm, memb_h, (size_t)n * D * 4);
    q.memcpy(mpos_dev.p, ph, (size_t)n * 4);
    q.memcpy(mseq_dev.p, ph + MQ, (size_t)n * 4);
    q.memcpy(mpid_dev.p, ph + 2 * MQ, (size_t)n * 3 * 4);
    q.memcpy(msrc_dev.p, ph + 5 * MQ, (size_t)n * 4);
    q35ops::gather_hidden(q, msrc_dev.ptr<int>(), n, hring.ptr<half>(), mh16.ptr<half>(), hin);
    half* cat = g16.ptr<half>();
    q35ops::norm_rows<float>(q, xm, m.pe.ptr<float>(), cat, 2 * D, n);
    q35ops::norm_rows<half>(q, hin, m.ph.ptr<float>(), cat + D, 2 * D, n);
    m.fc.forward(q, lin, cat, n, xm, DType::F32);
    // the block's MLP output is added to the head rows only, by their norm (the other rows' streams are not read again)
    const bool fold = !dump;
    block_forward(mtp->blk, xm, n, 0, mpid_dev.ptr<int>(), mpos_dev.ptr<int>(), mseq_dev.ptr<int>(), false, fold);
    if (nhead > 0) {
        const size_t o = (size_t)(n - nhead) * D;
        if (fold) q35ops::add_norm16(q, xm + o, sub.ptr<float>() + o, m.fn.ptr<float>(), mh16.ptr<half>(), nhead);
        else q35ops::norm16(q, xm + o, m.fn.ptr<float>(), mh16.ptr<half>(), nhead);
        // drafts from the gathered head's blocks, or the head's first mtp_head columns; the rest are never proposed
        if (mtp_head) {
            lm_head.forward(q, lin, mh16.ptr<half>(), nhead, mlog.p, DType::F16, -1, mtp_head);
            q35ops::argmax_rows(q, mlog.ptr<half>(), nhead, mtp_head, mbest.ptr<int>());
        } else {
            draft_head.forward(q, lin, mh16.ptr<half>(), nhead, mlog.p, DType::F16);
            q35ops::argmax_rows(q, mlog.ptr<half>(), nhead, draft_head.out, mbest.ptr<int>());
        }
        q.memcpy(mbest_h, mbest.p, (size_t)nhead * 2 * 4);
    }
}

void Q35Model::mtp_decode(const std::vector<MRow>& rows, int nhead) {
    auto& q = Dev::get().q;
    const int T = (int)rows.size();
    GLM_CHECK(nhead <= MQ && nhead <= T, "MTP: %d draft rows", nhead);
    if (T == 0) return;
    const int pieces = (T + MQ - 1) / MQ;
    int r0 = 0;
    for (int pc = 0; pc < pieces; pc++) {
        const int n = pc == 0 ? T - (pieces - 1) * MQ : MQ;    // the last piece holds the draft rows
        const int nh = pc + 1 == pieces ? nhead : 0;
        q.wait();                                              // the pinned rows of the piece before were read
        int* ph = mpos_h;
        for (int i = 0; i < n; i++) {
            const MRow& w = rows[r0 + i];
            emb_row(w.tok, memb_h + (size_t)i * D);
            ph[i] = w.pos;
            ph[MQ + i] = w.seq;
            for (int a = 0; a < 3; a++) ph[2 * MQ + i * 3 + a] = w.pid[a];
            ph[5 * MQ + i] = w.hid;
        }
        if (graphs_on && !dump) {
            namespace ex = sycl::ext::oneapi::experimental;
            auto& g = mtp_graphs[(size_t)n * (MQ + 1) + nh];
            if (!g) {
                ex::command_graph<ex::graph_state::modifiable> gr{q.get_context(), q.get_device()};
                gr.begin_recording(q);
                mtp_pass_dev(n, nh);
                gr.end_recording(q);
                g = std::make_unique<ExecGraph>(gr.finalize());
            }
            q.ext_oneapi_graph(*g);
        } else {
            mtp_pass_dev(n, nh);
        }
        r0 += n;
    }
    q.wait();
    mtp_rows += T;
}

void Q35Model::mtp_prefill_rows(const int* tokens, int nr, int p0, const int* pid_rows, float* Xm) {
    auto& q = Dev::get().q;
    const Q35Mtp& m = *mtp;
    x_ev.wait();
    for (int i = 0; i < nr; i++) emb_row(tokens[i], x_host + (size_t)i * D);
    x_ev = q.memcpy(sub.p, x_host, (size_t)nr * D * 4);
    half* cat = g16.ptr<half>();
    q35ops::norm_rows<float>(q, sub.ptr<float>(), m.pe.ptr<float>(), cat, 2 * D, nr);
    q35ops::norm_rows<half>(q, hfin.ptr<half>(), m.ph.ptr<float>(), cat + D, 2 * D, nr);
    m.fc.forward(q, lin, cat, nr, Xm, DType::F32);
    block_forward(mtp->blk, Xm, nr, p0, pid_rows, nullptr, nullptr);
    mtp_rows += nr;
}

void Q35Model::draft_rows(const int* seq, const int* tok, const int* row, const int* n, int R,
                          std::vector<std::vector<int>>& out) {
    (void)row;
    out.assign(R, {});
    if (!mtp || R <= 0) return;
    dec_ev.wait();
    rep_ev.wait();
    // per chain: the catch-up rows of its sequence (true hidden states, positions the last step read), then its first
    // draft row at the sampled token's position, from the hidden state of the row that sampled it
    std::vector<MRow> rows, heads;
    std::vector<int> live, K(R), rope(R);
    std::vector<float> cum(R, 1.0f);
    for (int i = 0; i < R; i++) {
        if (n[i] <= 0) continue;
        const int s = seq[i];
        Seq& se = seqs_[s];
        const int k = se.pos, rp = (k - 1 + RING) % RING;
        if (k < 1 || se.ring_pos[rp] != k - 1 || se.img_row != 0) continue;
        mtp_catchup(s, k, rows);
        K[i] = k;
        rope[i] = se.rope_next;
        heads.push_back({s, k, tok[i], {rope[i], rope[i], rope[i]}, s * RING + rp});
        live.push_back(i);
    }
    if (live.empty()) return;
    rows.insert(rows.end(), heads.begin(), heads.end());
    mtp_decode(rows, (int)heads.size());
    for (int i : live) seqs_[seq[i]].mtp_done = K[i] + 1;
    // the chains: each draft's row takes the one before's MTP output (mtp.norm'd) as its hidden state
    for (;;) {
        std::vector<int> next;
        std::vector<MRow> cr;
        for (int j = 0; j < (int)live.size(); j++) {
            const int i = live[j];
            const int id = draft_id(mbest_h[2 * j]);
            const float pr = sycl::bit_cast<float>(mbest_h[2 * j + 1]);
            // a chain ends where the product of its drafts' probabilities falls below mtp_min_p
            cum[i] *= pr;
            if (cum[i] < mtp_min_p || id >= V) continue;
            out[i].push_back(id);
            const int d = (int)out[i].size();
            if (d < n[i]) {
                next.push_back(i);
                cr.push_back({seq[i], K[i] + d, id, {rope[i] + d, rope[i] + d, rope[i] + d}, q35ops::MH + j});
            }
        }
        if (next.empty()) break;
        mtp_decode(cr, (int)cr.size());
        live.swap(next);
    }
}

}  // namespace glm
