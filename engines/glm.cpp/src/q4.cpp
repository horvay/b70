#include "q4.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>
#include "json.h"
#include "moe.h"
#include "ops.h"
#include "q4_ops.h"
#include "progress.h"

namespace glm {

using namespace q4;

namespace {
const std::string P = "model.language_model.";
template <class T> float f(T v) { return static_cast<float>(v); }

std::string lp(int L) { return P + "layers." + std::to_string(L) + "."; }

// host copy of a small tensor as fp32 (fp16, bf16 or fp32 source)
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

// (1 + w) of a Qwen4ExpTextRMSNorm weight, fp32 on the device
Tensor norm1p(const SafeTensors& st, const std::string& name) {
    auto v = host_f32(st, name);
    for (auto& x : v) x += 1.0f;
    return dev_f32(v);
}

// fp16 matrices stacked by rows
Tensor stack_f16(const SafeTensors& st, const std::vector<std::string>& names) {
    int64_t rows = 0, cols = -1;
    for (auto& n : names) {
        const StTensor& t = st.at(n);
        GLM_CHECK(t.dt == DType::F16 && t.shape.size() == 2, "%s: expected fp16 2-D", n.c_str());
        GLM_CHECK(cols < 0 || cols == t.shape[1], "%s: columns", n.c_str());
        cols = t.shape[1];
        rows += t.shape[0];
    }
    Tensor out = Tensor::empty({rows, cols}, DType::F16);
    size_t off = 0;
    for (auto& n : names) {
        const StTensor& t = st.at(n);
        upload(static_cast<char*>(out.p) + off, t.data, t.bytes);
        off += t.bytes;
    }
    return out;
}

Q4Site load_site(const SafeTensors& st, const std::string& p, bool inject) {
    Q4Site s;
    s.w1p = norm1p(st, p + ".hc_norm.weight");
    std::vector<std::string> d = {p + ".input_mix_weight_down.weight"};
    if (inject) d.push_back(p + ".block_inject_weight.weight");
    s.wdi = stack_f16(st, d);
    s.nd = (int)s.wdi.size(0);
    s.wu = stack_f16(st, {p + ".input_mix_weight_up.weight"});
    return s;
}

// a small pool of host threads for parallel loops
class Pool {
public:
    explicit Pool(int n) {
        for (int i = 0; i < n; i++) th_.emplace_back([this] { run(); });
    }
    ~Pool() {
        { std::lock_guard<std::mutex> g(mu_); stop_ = true; }
        cv_.notify_all();
        for (auto& t : th_) t.join();
    }
    // fn(i) for i in [0, n), on the pool and the calling thread; returns when all are done
    void each(int n, const std::function<void(int)>& fn) {
        if (n <= 0) return;
        std::atomic<int> next{0}, done{0};
        auto job = [&] {
            int i;
            while ((i = next.fetch_add(1)) < n) { fn(i); done.fetch_add(1); }
        };
        {
            std::lock_guard<std::mutex> g(mu_);
            job_ = job;
            gen_++;
        }
        cv_.notify_all();
        job();
        {   // no worker takes the job from here on; the ones that took it are counted in active_
            std::lock_guard<std::mutex> g(mu_);
            job_ = nullptr;
        }
        while (done.load() < n || active_.load() > 0) std::this_thread::yield();
    }

private:
    void run() {
        uint64_t seen = 0;
        for (;;) {
            std::function<void()> j;
            {
                std::unique_lock<std::mutex> g(mu_);
                cv_.wait(g, [&] { return stop_ || (gen_ != seen && job_); });
                if (stop_) return;
                seen = gen_;
                j = job_;
                active_.fetch_add(1);
            }
            j();
            active_.fetch_sub(1);
        }
    }
    std::vector<std::thread> th_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::function<void()> job_;
    uint64_t gen_ = 0;
    bool stop_ = false;
    std::atomic<int> active_{0};
};

Pool& pool() {
    static Pool p(15);
    return p;
}
}  // namespace

// ------------------------------------------------------------------------------------------------ n-gram table

NgramTable::NgramTable(const std::string& path) {
    fd = open(path.c_str(), O_RDONLY);
    GLM_CHECK(fd >= 0, "cannot open %s", path.c_str());
    uint64_t hlen = 0;
    GLM_CHECK(pread(fd, &hlen, 8, 0) == 8, "%s: header", path.c_str());
    std::string h(hlen, '\0');
    GLM_CHECK(pread(fd, h.data(), hlen, 8) == (ssize_t)hlen, "%s: header", path.c_str());
    Json j = Json::parse(h);
    const std::string pre = P + "layers.1.ple.ple_embedding.ngram_embedding.";
    const int64_t data = 8 + (int64_t)hlen;
    const Json& md = j["__metadata__"];
    K = std::stoi(md["K"].s());
    GLM_CHECK(md["codebook"].s() == "mul1" && std::stoi(md["row_dim"].s()) == 160, "n-gram table: format");
    row_words = 1 + 160 * K / 16;
    int64_t rows = std::stoll(md["rows"].s()), shard_rows = std::stoll(md["shard_rows"].s());
    int64_t row_bytes = row_words * 2;
    auto off = [&](const std::string& n) { return data + j[pre + n]["data_offsets"][0].i(); };
    base = off("shard_0.trellis");
    for (int64_t s = 0; s * shard_rows < rows; s++)
        GLM_CHECK(off("shard_" + std::to_string(s) + ".trellis") == base + s * shard_rows * row_bytes,
                  "n-gram table: shard %lld is not contiguous", (long long)s);
    auto rd = [&](const std::string& n, void* dst, size_t b) {
        GLM_CHECK(pread(fd, dst, b, off(n)) == (ssize_t)b, "n-gram table: %s", n.c_str());
    };
    rd("layer_multipliers", mult, sizeof mult);
    rd("head_vocab_sizes", vsize, sizeof vsize);
    rd("head_offsets", offs, sizeof offs);
    std::vector<half> hb(16 * 160);
    rd("head_bias", hb.data(), hb.size() * 2);
    head_bias.resize(hb.size());
    for (size_t i = 0; i < hb.size(); i++) head_bias[i] = f(hb[i]);
}

NgramTable::~NgramTable() {
    if (fd >= 0) close(fd);
}

void NgramTable::lookup(const int* hist, int n, half* emb) {
    // the mul1 codebook, once
    static std::vector<float> cb = [] {
        std::vector<float> v(65536);
        const float k_inv = f(sycl::bit_cast<half>((uint16_t)0x1eee)), k_bias = f(sycl::bit_cast<half>((uint16_t)0xc931));
        for (uint32_t s = 0; s < 65536; s++) {
            uint32_t pr = s * 0x83DCD12Du;
            uint32_t bs = (pr & 255) + ((pr >> 8) & 255) + ((pr >> 16) & 255) + (pr >> 24);
            v[s] = f(half((1024.0f + bs) * k_inv + k_bias));
        }
        return v;
    }();
    const int64_t m0 = mult[0], m1 = mult[1], m2 = mult[2];
    const int W = row_words, nsym = 160, K_ = K;
    pool().each(n * 16, [&](int i) {
        int t = i / 16, h = i % 16;
        // hist index p = t + 2; shifts that cross an EOS read EOS
        const int* hp = hist + t + 2;
        int64_t t0 = hp[0], t1 = hp[-1], t2 = hp[-2];
        if (t1 == EOS) t2 = EOS;
        int64_t mix = (t0 * m0) ^ (t1 * m1);
        if (h >= 8) mix ^= t2 * m2;
        int64_t row = mix % vsize[h] + offs[h];
        uint16_t w[64];
        ssize_t got = pread(fd, w, W * 2, base + row * W * 2);
        GLM_CHECK(got == W * 2, "n-gram table: read row %lld", (long long)row);
        float scale = f(sycl::bit_cast<half>(w[0]));
        const uint16_t* stream = w + 1;
        const int words = W - 1;
        int sym[160];
        for (int p = 0; p < nsym; p++) {
            int bit = p * K_;
            uint32_t win = stream[bit >> 4] | ((uint32_t)stream[((bit >> 4) + 1) % words] << 16);
            sym[p] = (win >> (bit & 15)) & ((1 << K_) - 1);
        }
        const float* hb = head_bias.data() + h * 160;
        half* out = emb + (size_t)t * D + h * 160;
        for (int p = 0; p < nsym; p++) {
            uint32_t st = 0;
            for (int jj = 0; jj < (15 + K_) / K_; jj++) st |= (uint32_t)sym[((p - jj) % nsym + nsym) % nsym] << (jj * K_);
            st &= 0xFFFF;
            out[p] = half(cb[st] * scale + hb[p]);
        }
    });
}

// ------------------------------------------------------------------------------------------------ state

void Q4GdnState::init() {
    conv = Tensor::zeros({GQKV, GCONV - 1}, DType::BF16);
    rec = Tensor::zeros({GV, GDK, GDV}, DType::F32);
}

void Q4GdnState::zero() {
    auto& q = Dev::get().q;
    q.memset(conv.p, 0, conv.bytes());
    q.memset(rec.p, 0, rec.bytes());
}

void Q4Cache::init(int T) {
    k = Tensor::empty({T, AKV, AD}, DType::F16);
    v = Tensor::empty({T, AKV, AD}, DType::F16);
    rk = Tensor::empty({T, ID}, DType::F16);
    pk = Tensor::empty({T / BLK + 1, ID}, DType::F16);
}

// ------------------------------------------------------------------------------------------------ load

Q4Model::Q4Model(const std::string& dir, int max_tok, double reserve_gib, const std::string& hot) : max_tokens(max_tok) {
    Json c = Json::parse(read_file(dir + "/config.json"));
    const Json& tc = c.has("text_config") ? c["text_config"] : c;
    GLM_CHECK(tc.get_s("model_type", "") == "qwen4_exp_text" && tc.get_i("hidden_size", 0) == D &&
                  tc.get_i("num_hidden_layers", 0) == L && tc.get_i("num_experts", 0) == E &&
                  tc.get_i("moe_intermediate_size", 0) == I && tc.get_i("hc_count", 0) == S,
              "%s: not the Qwen3.8-Flash-Next shape this engine is built for", dir.c_str());
    max_tokens = (max_tokens + BLK - 1) / BLK * BLK;
    st = std::make_unique<SafeTensors>(dir);
    ngram = std::make_unique<NgramTable>(dir + "/ngram_embedding.safetensors");
    auto& q = Dev::get().q;
    if (getenv("Q4_REBALANCE")) rebal_every = atoi(getenv("Q4_REBALANCE"));      // 0: static residency
    if (getenv("GLM_PREFILL_CHUNK")) prefill_chunk = std::max(64, std::min(2048, atoi(getenv("GLM_PREFILL_CHUNK"))));

    const StTensor& et = st->at(P + "embed_tokens.weight");
    GLM_CHECK(et.dt == DType::BF16 && et.shape[0] == V && et.shape[1] == D, "embedding: expected bf16 [V, D]");
    embed = static_cast<const bf16*>(et.data);
    lm_head = Exl3Linear::load(*st, "lm_head");
    mixer = load_site(*st, P + "hyper_connection_mixer", false);

    blocks.resize(L);
    gst.resize(L);
    kv.resize(L);
    for (int li = 0; li < L; li++) {
        Q4Block& b = blocks[li];
        std::string p = lp(li);
        b.layer = li;
        b.attn = tc["layer_types"][li].s() == "full_attention";
        b.site[0] = load_site(*st, p + "attn_hyper_connection", true);
        b.site[1] = load_site(*st, p + "mlp_hyper_connection", true);
        if (b.attn) {
            auto a = std::make_unique<Q4Attn>();
            std::string ap = p + "self_attn.";
            a->q = Exl3Linear::load(*st, ap + "q_proj");
            a->k = Exl3Linear::load(*st, ap + "k_proj");
            a->v = Exl3Linear::load(*st, ap + "v_proj");
            a->o = Exl3Linear::load(*st, ap + "o_proj");
            a->idx = Exl3Linear::load(*st, ap + "indexer.index_qk_proj");
            GLM_CHECK(a->q.out == AH * 2 * AD && a->k.out == AKV * AD && a->idx.out == (IH + 1) * ID, "%s: shapes", ap.c_str());
            a->qn = norm1p(*st, ap + "q_norm.weight");
            a->kn = norm1p(*st, ap + "k_norm.weight");
            a->iqn = norm1p(*st, ap + "indexer.q_layernorm.weight");
            a->ikn = norm1p(*st, ap + "indexer.k_layernorm.weight");
            b.att = std::move(a);
            kv[li].init(max_tokens);
        } else {
            auto g = std::make_unique<Q4Gdn>();
            std::string gp = p + "linear_attn.";
            g->qkv = Exl3Linear::load(*st, gp + "in_proj_qkv");
            g->z = Exl3Linear::load(*st, gp + "in_proj_z");
            g->o = Exl3Linear::load(*st, gp + "out_proj");
            GLM_CHECK(g->qkv.out == GQKV && g->z.out == GV * GDV && g->o.in == GV * GDV, "%s: shapes", gp.c_str());
            g->ab = stack_f16(*st, {gp + "in_proj_b.weight", gp + "in_proj_a.weight"});
            g->conv_w = st->load(gp + "conv1d.weight");
            GLM_CHECK(g->conv_w.dt == DType::BF16 && g->conv_w.numel() == GQKV * GCONV, "%s: conv1d", gp.c_str());
            auto al = host_f32(*st, gp + "A_log");
            for (auto& x : al) x = -std::exp(x);
            g->neg_a = dev_f32(al);
            g->dt_bias = dev_f32(host_f32(*st, gp + "dt_bias"));
            g->norm_w = dev_f32(host_f32(*st, gp + "norm.weight"));
            b.gdn = std::move(g);
            gst[li].init();
        }
        b.router = stack_f16(*st, {p + "mlp.gate.weight", p + "mlp.shared_expert_gate.weight"});
        b.sg = Exl3Linear::load(*st, p + "mlp.shared_expert.gate_proj");
        b.su = Exl3Linear::load(*st, p + "mlp.shared_expert.up_proj");
        b.sd = Exl3Linear::load(*st, p + "mlp.shared_expert.down_proj");
        load_progress("loaded layer %d/%d (%.2f GiB VRAM)", li + 1, L, Dev::get().vram_used / 1073741824.0);
    }
    load_progress_end();
    // PLE in front of layer 1
    {
        std::string pp = lp(1) + "ple.";
        ple_wkv = stack_f16(*st, {pp + "key_proj.weight", pp + "value_proj.weight"});
        GLM_CHECK(ple_wkv.size(0) == SD + D && ple_wkv.size(1) == D, "PLE projections");
        ple_nk = norm1p(*st, pp + "norm_key.weight");
        ple_nq = norm1p(*st, pp + "norm_query.weight");
        ple_nc = norm1p(*st, pp + "norm_conv.weight");
        ple_cw = dev_f32(host_f32(*st, pp + "conv1d.weight"));
        ple_state = Tensor::zeros({SD, 9}, DType::F16);
    }

    // workspaces and activations for prefill_chunk rows
    const int C = std::max(qmax, prefill_chunk);
    size_t lw = lm_head.ws_bytes(qmax);
    auto need = [&](const Exl3Linear& l) { lw = std::max({lw, l.ws_bytes(C), l.ws_bytes(qmax)}); };
    for (auto& b : blocks) {
        need(b.sg); need(b.su); need(b.sd);
        if (b.att) { need(b.att->q); need(b.att->k); need(b.att->v); need(b.att->o); need(b.att->idx); }
        if (b.gdn) { need(b.gdn->qkv); need(b.gdn->z); need(b.gdn->o); }
    }
    lin.init(lw + ((size_t)1 << 20));
    // act: GDN ~110 KB a row; attention: projections ~40 KB a row plus 256-row pieces of the softmax partials
    act.init((size_t)C * 120 * 1024 + q4ops::attn_part_floats(256) * 4 + ((size_t)64 << 20));
    mws.init((size_t)C * TOPK * (D + I) * 2 + ((size_t)128 << 20));
    X = Tensor::empty({qmax, S, D}, DType::F32);
    n16 = Tensor::empty({C, SD}, DType::F16);
    t324 = Tensor::empty({C, R + S}, DType::F32);
    a16 = Tensor::empty({C, R}, DType::F16);
    gpre = Tensor::empty({C, SD}, DType::F32);
    y16 = Tensor::empty({C, D}, DType::F16);
    sub = Tensor::empty({C, D}, DType::F32);
    inj = Tensor::empty({C, S}, DType::F32);
    rl = Tensor::empty({C, E + 1}, DType::F16);
    sel = Tensor::empty({C, TOPK}, DType::I32);
    wts = Tensor::empty({C, TOPK}, DType::F32);
    moe16 = Tensor::empty({C, D}, DType::F16);
    sg16 = Tensor::empty({C, I}, DType::F16);
    su16 = Tensor::empty({C, I}, DType::F16);
    sa16 = Tensor::empty({C, I}, DType::F16);
    sh32 = Tensor::empty({C, D}, DType::F32);
    emb16 = Tensor::empty({C, D}, DType::F16);
    kv32 = Tensor::empty({C, SD + D}, DType::F32);
    gv32 = Tensor::empty({C, SD}, DType::F32);
    cn16 = Tensor::empty({C, SD}, DType::F16);
    fin16 = Tensor::empty({qmax, D}, DType::F16);
    logits = Tensor::empty({qmax, V}, DType::F16);
    pos_dev = Tensor::zeros({1}, DType::I32);
    pos_h = sycl::malloc_host<int>(1, q);
    graphs_on = !(getenv("Q4_GRAPHS") && atoi(getenv("Q4_GRAPHS")) == 0) && !g_prof_points && !g_sync_points;
    x_host = sycl::malloc_host<float>((size_t)C * D, q);
    e_host = sycl::malloc_host<half>((size_t)C * D, q);
    GLM_CHECK(x_host && e_host, "pinned upload buffers");
    q.wait();
    load_experts(hot, reserve_gib);
    // every kernel once (modules load on first use, ~7 s): a prompt piece through the GEMM / DPAS paths, a decode step
    if (!getenv("Q4_WARMUP") || atoi(getenv("Q4_WARMUP")) != 0) {
        auto tw = std::chrono::steady_clock::now();
        std::vector<int> t(std::min(300, max_tokens - 1), 220);
        prefill(t.data(), (int)t.size());
        forward(t.data(), 1);
        q.wait();
        reset();
        fprintf(stderr, "warm-up %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - tw).count());
    }
    fprintf(stderr, "model ready: %.2f GiB VRAM\n", Dev::get().vram_used / 1073741824.0);
}

Q4Model::~Q4Model() {
    auto& q = Dev::get().q;
    q.wait();
    for (auto& c : ckpts_) {
        for (void* p : c.conv) if (p) sycl::free(p, q);
        for (void* p : c.rec) if (p) sycl::free(p, q);
        if (c.ple) sycl::free(c.ple, q);
    }
    dec_graph.reset();
    if (swap_buf) sycl::free(swap_buf, q);
    if (pos_h) sycl::free(pos_h, q);
    if (x_host) sycl::free(x_host, q);
    if (e_host) sycl::free(e_host, q);
    if (vram_pool) sycl::free(vram_pool, q);
    if (ram_pool) sycl::free(ram_pool, q);
}

// ------------------------------------------------------------------------------------------------ experts

// one record: planar4(gate | up), planar4(down), g.suh u.suh g.svh u.svh d.suh d.svh (kernels/exl3_moe_kernels.h
// Geom; llm/glm/xpu/pack_store.py blob_perm for K = 3: 24 words a tile, periods of D = 3 words, G = 8)
void q4_build_record(const SafeTensors& stt, int li, int e, char* dst) {
    constexpr int W = 24, PD = 3, PG = 8;
    const SafeTensors* st = &stt;
    const size_t rec = moe_record_bytes(D, I, 3);
    std::string p = lp(li) + "mlp.experts." + std::to_string(e) + ".";
    auto tr = [&](const char* m) {
        const StTensor& t = st->at(p + m + ".trellis");
        GLM_CHECK(t.shape.size() == 3 && t.shape[2] == 3 * 16, "%s%s: expected K=3", p.c_str(), m);
        return static_cast<const uint32_t*>(t.data);
    };
    const uint32_t *g = tr("gate_proj"), *u = tr("up_proj"), *dn = tr("down_proj");
    uint32_t* o = reinterpret_cast<uint32_t*>(dst);
    auto p4 = [&](int rows, int n, auto src) {      // src(r, tile, word)
        for (int r = 0; r < rows; r++)
            for (int n4 = 0; n4 < n / 4; n4++)
                for (int dd = 0; dd < PD; dd++)
                    for (int t = 0; t < 4; t++)
                        for (int gg = 0; gg < PG; gg++) *o++ = src(r, n4 * 4 + t, gg * PD + dd);
    };
    const int TI = I / 16, TH = D / 16;
    p4(TH, 2 * TI, [&](int r, int n, int w) {
        return n < TI ? g[((size_t)r * TI + n) * W + w] : u[((size_t)r * TI + n - TI) * W + w];
    });
    p4(TI, TH, [&](int r, int n, int w) { return dn[((size_t)r * TH + n) * W + w]; });
    char* s = reinterpret_cast<char*>(o);
    auto cp = [&](const std::string& n, size_t bytes) {
        const StTensor& t = st->at(p + n);
        GLM_CHECK(t.bytes == bytes, "%s%s: %zu bytes", p.c_str(), n.c_str(), t.bytes);
        memcpy(s, t.data, bytes);
        s += bytes;
    };
    cp("gate_proj.suh", D * 2); cp("up_proj.suh", D * 2); cp("gate_proj.svh", I * 2); cp("up_proj.svh", I * 2);
    cp("down_proj.suh", I * 2); cp("down_proj.svh", D * 2);
    GLM_CHECK((size_t)(s - dst) == rec, "expert record: %zu of %zu bytes", (size_t)(s - dst), rec);
}

void Q4Model::load_experts(const std::string& hot, double reserve_gib) {
    auto& q = Dev::get().q;
    const size_t rec = moe_record_bytes(D, I, 3);
    rec_bytes = (rec + 4095) / 4096 * 4096;
    const int NK = L * E;
    // order: the hot lists (hottest first, file by file), then the rest round-robin over the layers
    std::vector<int> order;
    std::vector<uint8_t> placed(NK, 0);
    std::string h = hot;
    for (size_t a = 0; !h.empty() && a <= h.size();) {
        size_t b = h.find(',', a);
        std::string path = h.substr(a, b == std::string::npos ? std::string::npos : b - a);
        a = b == std::string::npos ? h.size() + 1 : b + 1;
        if (path.empty()) continue;
        Json j = Json::parse(read_file(path));
        std::vector<std::vector<int>> per(L);
        for (auto& [k, v] : j.obj) {
            int li = -1;
            if (sscanf(k.c_str(), "layers.%d.mlp.experts", &li) != 1 || li < 0 || li >= L) continue;
            for (auto& e : v.arr) per[li].push_back((int)e.i());
        }
        // interleave the layers rank by rank, so a cut anywhere in the list stays even across layers
        for (size_t r = 0;; r++) {
            bool any = false;
            for (int li = 0; li < L; li++) {
                if (r >= per[li].size()) continue;
                any = true;
                int key = li * E + per[li][r];
                if (per[li][r] >= 0 && per[li][r] < E && !placed[key]) { placed[key] = 1; order.push_back(key); }
            }
            if (!any) break;
        }
    }
    for (int e = 0; e < E; e++)
        for (int li = 0; li < L; li++)
            if (!placed[li * E + e]) order.push_back(li * E + e);

    size_t total = Dev::get().dev.get_info<sycl::info::device::global_mem_size>();
    size_t used = Dev::get().vram_used + (size_t)(reserve_gib * 1073741824.0);
    int nv = total > used ? (int)std::min<size_t>(NK, (total - used) / rec_bytes) : 0;
    int nr = NK - nv;
    if (getenv("Q4_VRAM_EXPERTS")) nv = std::min(nv, atoi(getenv("Q4_VRAM_EXPERTS"))), nr = NK - nv;
    fprintf(stderr, "experts: %d in VRAM (%.1f GiB), %d in RAM (%.1f GiB)\n", nv, nv * (double)rec_bytes / 1073741824.0,
            nr, nr * (double)rec_bytes / 1073741824.0);
    vram_pool = nv ? static_cast<char*>(sycl::malloc_device((size_t)nv * rec_bytes, q)) : nullptr;
    GLM_CHECK(!nv || vram_pool, "expert VRAM pool (%d slots) allocation failed", nv);
    Dev::get().vram_used += (size_t)nv * rec_bytes;
    ram_pool = nr ? static_cast<char*>(sycl::malloc_host((size_t)nr * rec_bytes, q)) : nullptr;
    GLM_CHECK(!nr || ram_pool, "expert RAM pool (%d records, %.1f GiB pinned) allocation failed", nr,
              nr * (double)rec_bytes / 1073741824.0);
    in_vram.assign(NK, 0);
    ptrs_h.assign(NK, 0);
    std::vector<char*> host_dst(NK, nullptr);
    for (int i = 0; i < NK; i++) {
        int key = order[i];
        if (i < nv) {
            in_vram[key] = 1;
            ptrs_h[key] = (int64_t)(vram_pool + (size_t)i * rec_bytes);
        } else {
            host_dst[key] = ram_pool + (size_t)(i - nv) * rec_bytes;
            ptrs_h[key] = (int64_t)host_dst[key];
        }
    }

    // build in file order (the shards' tensors sit expert by expert), each expert's bytes asked for ahead with
    // MADV_WILLNEED: page faults in hotness order read ~0.7 GB/s, this streams near the SSD's rate
    auto span = [&](int key) {
        std::string p = lp(key / E) + "mlp.experts." + std::to_string(key % E) + ".";
        uintptr_t lo = UINTPTR_MAX, hi = 0;
        for (const char* m : {"gate_proj", "up_proj", "down_proj"})
            for (const char* t : {".trellis", ".suh", ".svh"}) {
                const StTensor& x = st->at(p + m + t);
                lo = std::min(lo, (uintptr_t)x.data);
                hi = std::max(hi, (uintptr_t)x.data + x.bytes);
            }
        return std::pair<uintptr_t, uintptr_t>(lo, hi);
    };
    std::vector<std::pair<uintptr_t, int>> fo(NK);
    std::vector<std::pair<uintptr_t, uintptr_t>> spans(NK);
    for (int k = 0; k < NK; k++) { spans[k] = span(k); fo[k] = {spans[k].first, k}; }
    std::sort(fo.begin(), fo.end());
    auto willneed = [&](int i) {
        if (i >= NK) return;
        auto [lo, hi] = spans[fo[i].second];
        uintptr_t a0 = lo & ~(uintptr_t)4095;
        madvise(reinterpret_cast<void*>(a0), hi - a0, MADV_WILLNEED);
    };
    auto t0 = std::chrono::steady_clock::now();
    const int B = 256, AHEAD = 64;
    char* bounce = static_cast<char*>(sycl::malloc_host((size_t)B * rec_bytes, q));
    GLM_CHECK(bounce, "expert bounce buffer");
    for (int i = 0; i < AHEAD; i++) willneed(i);
    for (int i0 = 0; i0 < NK; i0 += B) {
        int n = std::min(B, NK - i0);
        pool().each(n, [&](int j) {
            willneed(i0 + j + AHEAD);
            int key = fo[i0 + j].second;
            q4_build_record(*st, key / E, key % E, in_vram[key] ? bounce + (size_t)j * rec_bytes : host_dst[key]);
        });
        for (int j = 0; j < n; j++) {
            int key = fo[i0 + j].second;
            if (in_vram[key]) q.memcpy(reinterpret_cast<void*>(ptrs_h[key]), bounce + (size_t)j * rec_bytes, rec);
        }
        q.wait();
        load_progress("experts: %d/%d", i0 + n, NK);
    }
    sycl::free(bounce, q);
    load_progress_end("experts: %d records in %.1f s", NK,
                      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    ptrs_dev = Tensor::empty({L, E}, DType::I64);
    upload(ptrs_dev.p, ptrs_h.data(), ptrs_h.size() * 8);
    use_dev = Tensor::zeros({L, E}, DType::I32);
    in_vram_dev = Tensor::empty({L, E}, DType::U8);
    upload(in_vram_dev.p, in_vram.data(), NK);
    stats_dev = Tensor::zeros({2}, DType::I64);
}

void Q4Model::rebalance() {
    auto& q = Dev::get().q;
    const int NK = L * E;
    dec_ev.wait();
    q.wait();
    std::vector<int> use(NK);
    download(use.data(), use_dev.p, (size_t)NK * 4);
    q.memset(use_dev.p, 0, (size_t)NK * 4);
    if (score_.empty()) score_.assign(NK, 0.0f);
    for (int k = 0; k < NK; k++) score_[k] = score_[k] * 0.5f + use[k];
    // candidates: RAM experts by score (at least 2 decayed picks); victims: VRAM experts by score, coldest first
    std::vector<int> up, down;
    for (int k = 0; k < NK; k++) (in_vram[k] ? down : up).push_back(k);
    auto hot = [&](int a, int b) { return score_[a] > score_[b]; };
    size_t nu = std::min<size_t>(rebal_max, up.size()), nd = std::min<size_t>(rebal_max, down.size());
    std::partial_sort(up.begin(), up.begin() + nu, up.end(), hot);
    std::partial_sort(down.begin(), down.begin() + nd, down.end(), [&](int a, int b) { return score_[a] < score_[b]; });
    int n = 0;
    while (n < (int)std::min(nu, nd) && score_[up[n]] >= 2.0f && score_[up[n]] > 2.0f * score_[down[n]] + 1.0f) n++;
    if (!n) return;
    if (!swap_buf) {
        swap_buf = static_cast<char*>(sycl::malloc_host((size_t)rebal_max * rec_bytes, q));
        GLM_CHECK(swap_buf, "swap buffer");
    }
    const size_t rec = moe_record_bytes(D, I, 3);
    // VRAM victims out to the bounce, promoted records into the victims' slots, the victims into the freed RAM slots
    for (int j = 0; j < n; j++) q.memcpy(swap_buf + (size_t)j * rec_bytes, reinterpret_cast<void*>(ptrs_h[down[j]]), rec);
    q.wait();
    for (int j = 0; j < n; j++) q.memcpy(reinterpret_cast<void*>(ptrs_h[down[j]]), reinterpret_cast<void*>(ptrs_h[up[j]]), rec);
    q.wait();
    for (int j = 0; j < n; j++) {
        int a = up[j], b = down[j];
        memcpy(reinterpret_cast<void*>(ptrs_h[a]), swap_buf + (size_t)j * rec_bytes, rec);
        std::swap(ptrs_h[a], ptrs_h[b]);
        in_vram[a] = 1;
        in_vram[b] = 0;
        int64_t pa = ptrs_h[a], pb = ptrs_h[b];
        uint8_t one = 1, zero = 0;
        q.memcpy(ptrs_dev.ptr<int64_t>() + a, &pa, 8);
        q.memcpy(ptrs_dev.ptr<int64_t>() + b, &pb, 8);
        q.memcpy(in_vram_dev.ptr<uint8_t>() + a, &one, 1);
        q.memcpy(in_vram_dev.ptr<uint8_t>() + b, &zero, 1);
        q.wait();
    }
    swapped_ += n;
    // Q4_SWAP_CHECK=1: the GPU's view of every record that moved (both sides) against a fresh build of it
    static const bool check = getenv("Q4_SWAP_CHECK") && atoi(getenv("Q4_SWAP_CHECK"));
    if (check) {
        std::vector<int> keys;
        for (int j = 0; j < n; j++) { keys.push_back(up[j]); keys.push_back(down[j]); }
        const int W = (int)(rec / 8);
        uint64_t* sums = sycl::malloc_shared<uint64_t>(keys.size(), q);
        int64_t* addr = sycl::malloc_shared<int64_t>(keys.size(), q);
        for (size_t i = 0; i < keys.size(); i++) addr[i] = ptrs_h[keys[i]];
        q.parallel_for(sycl::range<1>(keys.size()), [=](sycl::id<1> i) {
            const uint64_t* w = reinterpret_cast<const uint64_t*>(addr[i]);
            uint64_t h = 1469598103934665603ull;
            for (int k = 0; k < W; k++) h = (h ^ w[k]) * 1099511628211ull;
            sums[i] = h;
        }).wait();
        std::vector<char> b(rec);
        int bad = 0;
        for (size_t i = 0; i < keys.size(); i++) {
            q4_build_record(*st, keys[i] / E, keys[i] % E, b.data());
            const uint64_t* w = reinterpret_cast<const uint64_t*>(b.data());
            uint64_t h = 1469598103934665603ull;
            for (int k = 0; k < W; k++) h = (h ^ w[k]) * 1099511628211ull;
            bad += h != sums[i];
        }
        fprintf(stderr, "swap check: %d records moved, %d differ from a fresh build\n", (int)keys.size(), bad);
        sycl::free(sums, q);
        sycl::free(addr, q);
    }
}

std::string Q4Model::expert_stats() {
    auto& q = Dev::get().q;
    q.wait();
    uint64_t st[2];
    download(st, stats_dev.p, 16);
    std::vector<int> use((size_t)L * E);
    download(use.data(), use_dev.p, use.size() * 4);
    q.memset(stats_dev.p, 0, 16);
    q.wait();
    std::vector<int> ram;
    for (size_t k = 0; k < use.size(); k++)
        if (!in_vram[k] && use[k]) ram.push_back(use[k]);
    std::sort(ram.rbegin(), ram.rend());
    auto top = [&](size_t n) {
        uint64_t s = 0;
        for (size_t i = 0; i < std::min(n, ram.size()); i++) s += ram[i];
        return st[1] ? 100.0 * s / st[1] : 0.0;
    };
    char b[320];
    snprintf(b, sizeof b, "experts: %llu picks, %.1f%% in RAM (%zu distinct RAM experts; the hottest 256 / 1024 / 4096 of "
             "them: %.0f / %.0f / %.0f%% of RAM picks)\n", (unsigned long long)st[0], st[0] ? 100.0 * st[1] / st[0] : 0.0,
             ram.size(), top(256), top(1024), top(4096));
    return b;
}

// ------------------------------------------------------------------------------------------------ pieces

void Q4Model::grab(const void* p, int n, bool is16) {
    if (!dump) return;
    auto& q = Dev::get().q;
    q.wait();
    size_t o = dump->size();
    dump->resize(o + n);
    if (is16) {
        std::vector<half> h(n);
        q.memcpy(h.data(), p, n * 2).wait();
        for (int i = 0; i < n; i++) (*dump)[o + i] = f(h[i]);
    } else {
        q.memcpy(dump->data() + o, p, n * 4).wait();
    }
}

void Q4Model::embed_rows(const int* tokens, int n, float* Xd) {
    auto& q = Dev::get().q;
    x_ev.wait();
    for (int t = 0; t < n; t++) {
        const bf16* e = embed + (size_t)tokens[t] * D;
        for (int d = 0; d < D; d++) x_host[(size_t)t * D + d] = f(e[d]);
    }
    float* e = sub.ptr<float>();
    x_ev = q.memcpy(e, x_host, (size_t)n * D * 4);
    q4ops::expand(q, e, Xd, n);
}

void Q4Model::site_mix(const Q4Site& s, const float* Xd, int rows, half* y, float* injp) {
    auto& q = Dev::get().q;
    q4ops::site_norm(q, Xd, s.w1p.ptr<float>(), n16.ptr<half>(), rows);
    ops::gemv_f16<float>(q, n16.ptr<half>(), rows, SD, s.wdi.ptr<half>(), s.nd, t324.ptr<float>());
    q4ops::site_gates(q, t324.ptr<float>(), s.nd, a16.ptr<half>(), injp, rows);
    ops::gemv_f16<float>(q, a16.ptr<half>(), rows, R, s.wu.ptr<half>(), SD, gpre.ptr<float>());
    q4ops::site_combine(q, gpre.ptr<float>(), n16.ptr<half>(), y, rows);
}

void Q4Model::site_apply(float* Xd, const float* injp, const float* o, int rows) {
    q4ops::site_apply(Dev::get().q, Xd, injp, o, rows);
}

void Q4Model::gdn_forward(const Q4Gdn& g, Q4GdnState& s, const half* y, int rows, float* out) {
    auto& q = Dev::get().q;
    act.reset();
    float* qkv = act.take<float>((size_t)rows * GQKV);
    float* z = act.take<float>((size_t)rows * GV * GDV);
    float* ab = act.take<float>((size_t)rows * 2 * GV);
    float* beta = act.take<float>((size_t)rows * GV);
    float* gg = act.take<float>((size_t)rows * GV);
    bf16* cv = act.take<bf16>((size_t)rows * GQKV);
    bf16* core = act.take<bf16>((size_t)rows * GV * GDV);
    half* nrm = act.take<half>((size_t)rows * GV * GDV);
    prof_point("GDN: (entry)");
    g.qkv.forward(q, lin, y, rows, qkv, DType::F32);
    g.z.forward(q, lin, y, rows, z, DType::F32);
    prof_point("GDN: qkv, z");
    ops::gemv_f16<float>(q, y, rows, D, g.ab.ptr<half>(), 2 * GV, ab);
    q4ops::gdn_gates(q, ab, g.neg_a.ptr<float>(), g.dt_bias.ptr<float>(), beta, gg, rows);
    q4ops::gdn_conv(q, qkv, s.conv.ptr<bf16>(), g.conv_w.ptr<bf16>(), cv, rows);
    prof_point("GDN: a b gates conv");
    q4ops::gdn_delta(q, cv, beta, gg, s.rec.ptr<float>(), core, rows);
    prof_point("GDN: delta rule");
    q4ops::gdn_norm(q, core, z, g.norm_w.ptr<float>(), nrm, rows);
    g.o.forward(q, lin, nrm, rows, out, DType::F32);
    prof_point("GDN: norm + out_proj");
}

void Q4Model::attn_forward(const Q4Attn& a, Q4Cache& c, const half* y, int rows, int p0, float* out, const int* pdev) {
    auto& q = Dev::get().q;
    act.reset();
    half* qg = act.take<half>((size_t)rows * AH * 2 * AD);
    half* k = act.take<half>((size_t)rows * AKV * AD);
    half* v = act.take<half>((size_t)rows * AKV * AD);
    half* ik = act.take<half>((size_t)rows * (IH + 1) * ID);
    half* qn16 = act.take<half>((size_t)rows * AH * AD);
    float* qi = act.take<float>((size_t)rows * IH * ID);
    int* cnt = act.take<int>(rows);
    half* o16 = act.take<half>((size_t)rows * AH * AD);
    a.q.forward(q, lin, y, rows, qg, DType::F16);
    a.k.forward(q, lin, y, rows, k, DType::F16);
    a.v.forward(q, lin, y, rows, v, DType::F16);
    a.idx.forward(q, lin, y, rows, ik, DType::F16);
    prof_point("attn: projections");
    q4ops::attn_prep(q, qg, k, v, ik, a.qn.ptr<float>(), a.kn.ptr<float>(), a.iqn.ptr<float>(), rows, p0, qn16,
                     c.k.ptr<half>(), c.v.ptr<half>(), qi, c.rk.ptr<half>(), pdev);
    if (pdev) q4ops::attn_pool_dec(q, c.rk.ptr<half>(), a.ikn.ptr<float>(), c.pk.ptr<half>(), pdev);
    else q4ops::attn_pool(q, c.rk.ptr<half>(), a.ikn.ptr<float>(), c.pk.ptr<half>(), p0 / BLK, (p0 + rows) / BLK);
    // selection and the softmax in pieces of rows (the score matrix and the partials grow with them); a graph's
    // decode step sizes the scores for the whole context
    const int nbmax = pdev ? max_tokens / BLK : (p0 + rows) / BLK;
    const size_t mark = act.off;
    for (int r0 = 0; r0 < rows;) {
        act.off = mark;
        int nr = std::min(rows - r0, 256);
        if (nbmax > BUDGET) nr = std::max(1, std::min<int>(nr, (int)((64u << 20) / ((size_t)nbmax * 4))));
        float* sc = act.take<float>((size_t)nr * std::max(nbmax, 1));
        int* list = act.take<int>((size_t)nr * NSEL);
        float* part = act.take<float>(q4ops::attn_part_floats(nr));
        q4ops::attn_select(q, qi + (size_t)r0 * IH * ID, c.pk.ptr<half>(), nr, p0 + r0, std::max(nbmax, 1), sc, list,
                           cnt + r0, pdev);
        q4ops::attn_core(q, qn16 + (size_t)r0 * AH * AD, c.k.ptr<half>(), c.v.ptr<half>(), list, cnt + r0,
                         qg + (size_t)r0 * AH * 2 * AD, nr, p0 + r0, part, o16 + (size_t)r0 * AH * AD, pdev);
        r0 += nr;
    }
    prof_point("attn: select + softmax");
    a.o.forward(q, lin, o16, rows, out, DType::F32);
    prof_point("attn: o_proj");
}

void Q4Model::moe_forward_(const Q4Block& b, const half* y, int rows, float* out, const int64_t* ptrs) {
    auto& q = Dev::get().q;
    prof_point("MoE: (entry)");
    ops::gemv_f16<half>(q, y, rows, D, b.router.ptr<half>(), E + 1, rl.ptr<half>());
    q4ops::route(q, rl.ptr<half>(), E + 1, rows, sel.ptr<int>(), wts.ptr<float>());
    if (counting)
        q4ops::count_use(q, sel.ptr<int>(), rows, use_dev.ptr<int>() + (size_t)b.layer * E,
                         in_vram_dev.ptr<uint8_t>() + (size_t)b.layer * E, stats_dev.ptr<unsigned long long>());
    prof_point("MoE: router");
    moe_forward(q, mws, y, rows, sel.ptr<int>(), wts.ptr<float>(), TOPK, ptrs + (size_t)b.layer * E, D, I, 3, E,
                moe16.ptr<half>(), 0.0f);
    prof_point("MoE: routed experts");
    b.sg.forward(q, lin, y, rows, sg16.p, DType::F16);
    b.su.forward(q, lin, y, rows, su16.p, DType::F16);
    q4ops::swiglu(q, sg16.ptr<half>(), su16.ptr<half>(), sa16.ptr<half>(), (size_t)rows * I);
    b.sd.forward(q, lin, sa16.ptr<half>(), rows, sh32.p, DType::F32);
    q4ops::moe_combine(q, moe16.ptr<half>(), sh32.ptr<float>(), rl.ptr<half>(), E + 1, out, rows);
    prof_point("MoE: shared expert + combine");
}

void Q4Model::ple_host(const int* tokens, int rows) {
    auto& q = Dev::get().q;
    std::vector<int> hist(rows + 2);
    hist[0] = ids_ctx[0];
    hist[1] = ids_ctx[1];
    for (int t = 0; t < rows; t++) hist[t + 2] = tokens[t];
    e_ev.wait();
    ngram->lookup(hist.data(), rows, e_host);
    ids_ctx[0] = hist[rows];
    ids_ctx[1] = hist[rows + 1];
}

void Q4Model::ple_dev(int rows, float* Xd) {
    auto& q = Dev::get().q;
    sycl::event ev = q.memcpy(emb16.p, e_host, (size_t)rows * D * 2);
    if (!recording) e_ev = ev;          // a recorded copy's event is not one to wait on (dec_ev covers it)
    ops::gemv_f16<float>(q, emb16.ptr<half>(), rows, D, ple_wkv.ptr<half>(), SD + D, kv32.ptr<float>());
    q4ops::ple_gate(q, kv32.ptr<float>(), Xd, ple_nk.ptr<float>(), ple_nq.ptr<float>(), ple_nc.ptr<float>(),
                    gv32.ptr<float>(), cn16.ptr<half>(), rows);
    q4ops::ple_conv(q, cn16.ptr<half>(), gv32.ptr<float>(), ple_cw.ptr<float>(), ple_state.ptr<half>(), Xd, rows);
}

void Q4Model::block_forward(Q4Block& b, float* Xd, int rows, int p0, const int64_t* ptrs, const int* tokens,
                            const int* pdev) {
    prof_point("(between layers)");
    if (b.layer == 1) {
        if (tokens) ple_host(tokens, rows);
        ple_dev(rows, Xd);
        prof_point("PLE");
    }
    site_mix(b.site[0], Xd, rows, y16.ptr<half>(), inj.ptr<float>());
    grab(y16.p, D, true);
    prof_point("site mix");
    if (b.att) { attn_forward(*b.att, kv[b.layer], y16.ptr<half>(), rows, p0, sub.ptr<float>(), pdev); prof_point("attention"); }
    else { gdn_forward(*b.gdn, gst[b.layer], y16.ptr<half>(), rows, sub.ptr<float>()); prof_point("GDN"); }
    grab(sub.p, D, false);
    site_apply(Xd, inj.ptr<float>(), sub.ptr<float>(), rows);
    site_mix(b.site[1], Xd, rows, y16.ptr<half>(), inj.ptr<float>());
    grab(y16.p, D, true);
    prof_point("site apply + mix");
    moe_forward_(b, y16.ptr<half>(), rows, sub.ptr<float>(), ptrs);
    grab(sub.p, D, false);
    site_apply(Xd, inj.ptr<float>(), sub.ptr<float>(), rows);
    grab(Xd, SD, false);
    prof_point("site apply");
}

// ------------------------------------------------------------------------------------------------ forward / prefill

void Q4Model::host_rows(const int* tokens, int n) {
    dec_ev.wait();
    x_ev.wait();
    for (int t = 0; t < n; t++) {
        const bf16* e = embed + (size_t)tokens[t] * D;
        for (int d = 0; d < D; d++) x_host[(size_t)t * D + d] = f(e[d]);
    }
    ple_host(tokens, n);
    *pos_h = pos;
}

void Q4Model::forward_dev(int Q, const int* pdev) {
    auto& q = Dev::get().q;
    counting = true;
    float* Xd = X.ptr<float>();
    float* e = sub.ptr<float>();
    q.memcpy(e, x_host, (size_t)Q * D * 4);
    q.memcpy(pos_dev.p, pos_h, sizeof(int));
    q4ops::expand(q, e, Xd, Q);
    for (auto& b : blocks) block_forward(b, Xd, Q, pos, ptrs_dev.ptr<int64_t>(), nullptr, pdev);
    site_mix(mixer, Xd, Q, fin16.ptr<half>(), nullptr);
    lm_head.forward(q, lin, fin16.ptr<half>(), Q, logits.p, DType::F16);
    prof_point("mixer + lm_head");
    counting = false;
}

const half* Q4Model::forward(const int* tokens, int Q) {
    GLM_CHECK(Q >= 1 && Q <= qmax, "forward: %d rows (at most %d)", Q, qmax);
    GLM_CHECK(pos + Q <= max_tokens, "context full (%d tokens)", max_tokens);
    auto& q = Dev::get().q;
    if (rebal_every > 0 && ++since_rebal >= rebal_every) { since_rebal = 0; rebalance(); }
    host_rows(tokens, Q);
    if (Q == 1 && graphs_on && !dump) {
        namespace ex = sycl::ext::oneapi::experimental;
        if (!dec_graph) {
            ex::command_graph<ex::graph_state::modifiable> g{q.get_context(), q.get_device()};
            recording = true;
            g.begin_recording(q);
            forward_dev(1, pos_dev.ptr<int>());
            g.end_recording(q);
            recording = false;
            dec_graph = std::make_unique<ex::command_graph<ex::graph_state::executable>>(g.finalize());
        }
        dec_ev = q.ext_oneapi_graph(*dec_graph);
    } else {
        forward_dev(Q, nullptr);
    }
    pos += Q;
    return logits.ptr<half>();
}

void Q4Model::prefill(const int* tokens, int n, int ckpt_at) {
    if (n <= 0) return;
    int cut = ckpt_at > pos && ckpt_at <= pos + n ? ckpt_at - pos : -1;
    if (cut > 0 && cut < n) {
        prefill(tokens, cut, ckpt_at);
        prefill(tokens + cut, n - cut, -1);
        return;
    }
    GLM_CHECK(pos + n <= max_tokens, "context full (%d tokens)", max_tokens);
    auto& q = Dev::get().q;
    dec_ev.wait();                      // a decode step may still read the upload buffers
    const int C = prefill_chunk;
    // layer-major over super-chunks of rows: every row's streams resident (40 KB a row)
    const int SC = 8 * C;
    Tensor Xall = Tensor::empty({std::min(n, SC), S, D}, DType::F32);
    for (int s0 = 0; s0 < n; s0 += SC) {
        int rows = std::min(SC, n - s0);
        float* Xp = Xall.ptr<float>();
        for (int c0 = 0; c0 < rows; c0 += C) embed_rows(tokens + s0 + c0, std::min(C, rows - c0), Xp + (size_t)c0 * SD);
        for (auto& b : blocks)
            for (int c0 = 0; c0 < rows; c0 += C) {
                int nr = std::min(C, rows - c0);
                block_forward(b, Xp + (size_t)c0 * SD, nr, pos + c0, ptrs_dev.ptr<int64_t>(), tokens + s0 + c0);
            }
        pos += rows;                    // pos is the super-chunk's start: s0 is in it already
        // the PLE consumed the ids of every row; the next super-chunk continues from its last two
    }
    q.wait();
    if (cut == n) save_checkpoint(new_checkpoint(pos));
}

// ------------------------------------------------------------------------------------------------ state

void Q4Model::reset() {
    pos = 0;
    for (auto& c : ckpts_) c.pos = -1;
    for (int li = 0; li < L; li++)
        if (!blocks[li].attn) gst[li].zero();
    auto& q = Dev::get().q;
    q.memset(ple_state.p, 0, ple_state.bytes());
    ids_ctx[0] = ids_ctx[1] = EOS;
    q.wait();
}

Q4Model::Checkpoint& Q4Model::new_checkpoint(int at) {
    auto& q = Dev::get().q;
    if (ckpts_.size() >= 2) {
        Checkpoint c = std::move(ckpts_.front());
        ckpts_.erase(ckpts_.begin());
        ckpts_.push_back(std::move(c));
    } else {
        Checkpoint c;
        for (int li = 0; li < L; li++) {
            if (blocks[li].attn) { c.conv.push_back(nullptr); c.rec.push_back(nullptr); continue; }
            c.conv.push_back(sycl::malloc_host(gst[li].conv.bytes(), q));
            c.rec.push_back(sycl::malloc_host(gst[li].rec.bytes(), q));
            GLM_CHECK(c.conv.back() && c.rec.back(), "checkpoint: pinned allocation failed");
        }
        c.ple = sycl::malloc_host(ple_state.bytes(), q);
        ckpts_.push_back(std::move(c));
    }
    ckpts_.back().pos = at;
    return ckpts_.back();
}

void Q4Model::save_checkpoint(Checkpoint& c) {
    auto& q = Dev::get().q;
    for (int li = 0; li < L; li++) {
        if (blocks[li].attn) continue;
        q.memcpy(c.conv[li], gst[li].conv.p, gst[li].conv.bytes());
        q.memcpy(c.rec[li], gst[li].rec.p, gst[li].rec.bytes());
    }
    q.memcpy(c.ple, ple_state.p, ple_state.bytes());
    c.ids[0] = ids_ctx[0];
    c.ids[1] = ids_ctx[1];
    q.wait();
}

bool Q4Model::restore(int upto) {
    auto& q = Dev::get().q;
    for (int i = (int)ckpts_.size() - 1; i >= 0; i--) {
        Checkpoint& c = ckpts_[i];
        if (c.pos < 0 || c.pos > upto) continue;
        for (int li = 0; li < L; li++) {
            if (blocks[li].attn) continue;
            q.memcpy(gst[li].conv.p, c.conv[li], gst[li].conv.bytes());
            q.memcpy(gst[li].rec.p, c.rec[li], gst[li].rec.bytes());
        }
        q.memcpy(ple_state.p, c.ple, ple_state.bytes());
        ids_ctx[0] = c.ids[0];
        ids_ctx[1] = c.ids[1];
        q.wait();
        pos = c.pos;
        ckpts_.erase(ckpts_.begin() + i + 1, ckpts_.end());
        return true;
    }
    return false;
}

}  // namespace glm
