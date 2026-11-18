#include "e0.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <mutex>
#include <map>
#include <sstream>
#include <thread>
#include <oneapi/mkl/blas.hpp>
#include "json.h"
#include "ops.h"
#include "progress.h"
#include "q4_ops.h"
#include "st.h"

namespace glm {

using namespace q4;
using namespace e0ops;

namespace {
template <class T> float f(T v) { return static_cast<float>(v); }
std::string bp(int L) { return "blk." + std::to_string(L) + "."; }

// GDN value heads: llama.cpp stores them tiled (v-slot j of k-head k at j * 16 + k), HF grouped (k * 3 + j)
inline int vh_src(int h) { return (h % 3) * GK + h / 3; }

float bf16f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float x; memcpy(&x, &u, 4); return x; }
float f16f(uint16_t b) { return f(sycl::bit_cast<half>(b)); }

// one value of a small tensor as fp32 (F32, F16, BF16)
float elem(const GgufTensor& t, size_t i) {
    switch (t.type) {
        case GG_F32: return static_cast<const float*>(t.data)[i];
        case GG_F16: return f16f(static_cast<const uint16_t*>(t.data)[i]);
        case GG_BF16: return bf16f(static_cast<const uint16_t*>(t.data)[i]);
    }
    GLM_CHECK(false, "%s: type %u is not F32/F16/BF16", t.name.c_str(), t.type);
    return 0;
}

std::vector<float> host_f32(const GgufTensor& t) {
    size_t n = (size_t)t.cols() * t.rows();
    std::vector<float> v(n);
    for (size_t i = 0; i < n; i++) v[i] = elem(t, i);
    return v;
}

Tensor dev_f32(const std::vector<float>& v) {
    Tensor t = Tensor::empty({(int64_t)v.size()}, DType::F32);
    upload(t.p, v.data(), v.size() * 4);
    return t;
}

// count of fp16 conversion trouble across the load (values past fp16's range, or nonzero ones in its subnormals)
size_t g_f16_over = 0, g_f16_sub = 0, g_f16_n = 0;
half to_f16(float x) {
    g_f16_n++;
    float a = std::fabs(x);
    if (a > 65504.0f) g_f16_over++;
    else if (a != 0 && a < 6.103515625e-05f) g_f16_sub++;
    return half(x);
}

// fp16 matrix from F32/F16/BF16 tensors stacked by rows, each [rows, cols]; perm (optional, per tensor): source row of
// each output row
Tensor stack_f16(const std::vector<const GgufTensor*>& ts, const std::vector<std::vector<int>>& perms = {}) {
    int64_t rows = 0, cols = ts[0]->cols();
    for (auto* t : ts) {
        GLM_CHECK(t->cols() == cols, "%s: %lld columns, expected %lld", t->name.c_str(), (long long)t->cols(),
                  (long long)cols);
        rows += t->rows();
    }
    std::vector<half> h((size_t)rows * cols);
    size_t o = 0;
    for (size_t k = 0; k < ts.size(); k++) {
        const GgufTensor& t = *ts[k];
        const std::vector<int>* pm = k < perms.size() && !perms[k].empty() ? &perms[k] : nullptr;
        for (int64_t r = 0; r < t.rows(); r++) {
            int64_t sr = pm ? (*pm)[r] : r;
            for (int64_t c = 0; c < cols; c++) h[o++] = to_f16(elem(t, (size_t)sr * cols + c));
        }
    }
    Tensor out = Tensor::empty({rows, cols}, DType::F16);
    upload(out.p, h.data(), h.size() * 2);
    return out;
}

// Q8_0 blocks (fp16 d, 32 int8) -> Q8W. rperm: source row of each output row; cperm: source 128-column chunk of each
// output chunk (both optional)
Q8W q8_from_blocks(const uint8_t* src, int N, int K, const std::vector<int>& rperm = {}, const std::vector<int>& cperm = {}) {
    GLM_CHECK(K % 32 == 0, "Q8_0 width %d", K);
    const int NB = K / 32;
    const size_t rb = (size_t)NB * 34;
    std::vector<int8_t> qs((size_t)N * K);
    std::vector<uint16_t> d((size_t)N * NB);
    for (int n = 0; n < N; n++) {
        const uint8_t* row = src + (size_t)(rperm.empty() ? n : rperm[n]) * rb;
        for (int b = 0; b < NB; b++) {
            int sb = cperm.empty() ? b : cperm[b / 4] * 4 + b % 4;      // a 128-column chunk is 4 blocks
            const uint8_t* blk = row + (size_t)sb * 34;
            memcpy(&d[(size_t)n * NB + b], blk, 2);
            memcpy(&qs[(size_t)n * K + b * 32], blk + 2, 32);
        }
    }
    Q8W w;
    w.N = N;
    w.K = K;
    w.qs = Tensor::empty({N, K}, DType::U8);
    w.d = Tensor::empty({N, NB}, DType::F16);
    upload(w.qs.p, qs.data(), qs.size());
    upload(w.d.p, d.data(), d.size() * 2);
    return w;
}

Q8W q8_load(const GgufTensor& t, const std::vector<int>& rperm = {}, const std::vector<int>& cperm = {}) {
    GLM_CHECK(t.type == GG_Q8_0 && t.ne.size() == 2, "%s: expected a Q8_0 matrix", t.name.c_str());
    return q8_from_blocks(static_cast<const uint8_t*>(t.data), (int)t.rows(), (int)t.cols(), rperm, cperm);
}

Q4Site load_site(const Gguf& g, const std::string& norm, const std::string& down, const std::string& inject,
                 const std::string& up) {
    Q4Site s;
    s.w1p = dev_f32(host_f32(g.at(norm)));            // the conversion's +1 is in it already
    std::vector<const GgufTensor*> d = {&g.at(down)};
    if (!inject.empty()) d.push_back(&g.at(inject));
    s.wdi = stack_f16(d);
    s.nd = (int)s.wdi.size(0);
    s.wu = stack_f16({&g.at(up)});
    GLM_CHECK(s.wdi.size(1) == SD && s.wu.size(0) == SD && s.wu.size(1) == R, "%s: hyper-connection shapes", down.c_str());
    return s;
}

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
    void each(int n, const std::function<void(int)>& fn) {
        if (n <= 0) return;
        std::atomic<int> next{0}, done{0};
        auto job = [&] {
            int i;
            while ((i = next.fetch_add(1)) < n) { fn(i); done.fetch_add(1); }
        };
        { std::lock_guard<std::mutex> g(mu_); job_ = job; gen_++; }
        cv_.notify_all();
        job();
        { std::lock_guard<std::mutex> g(mu_); job_ = nullptr; }
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

// A large device allocation whose pages are checked: on the B70 (xe) a 22 GiB allocation once came back with two
// 2 MiB pages mapped onto the same memory (Strata, sycl_queue.hpp). A tag per 64 KiB is written, then read back; a
// bad allocation is freed and tried again behind a growing spacer (64 MiB more each time), up to `tries` times.
char* malloc_checked(size_t bytes, sycl::queue& q, int tries = 8) {
    constexpr size_t BLK = 65536;
    std::vector<void*> spacers;
    char* got = nullptr;
    for (int attempt = 0; attempt < tries && !got; attempt++) {
        char* p = static_cast<char*>(sycl::malloc_device(bytes, q));
        if (!p) break;
        const size_t nb = bytes / BLK;
        const uint64_t salt = 0x9E3779B97F4A7C15ull * (attempt + 1);
        q.parallel_for(sycl::range<1>(nb), [=](sycl::id<1> i) {
            *reinterpret_cast<uint64_t*>(p + i * BLK) = (uint64_t)i * 0xD1B54A32D192ED03ull ^ salt;
        }).wait();                                  // (the check must not overtake it, whatever the queue's mode)
        unsigned* bad = sycl::malloc_shared<unsigned>(1, q);
        *bad = 0;
        q.parallel_for(sycl::range<1>(nb), [=](sycl::id<1> i) {
            if (*reinterpret_cast<const uint64_t*>(p + i * BLK) != ((uint64_t)i * 0xD1B54A32D192ED03ull ^ salt)) {
                sycl::atomic_ref<unsigned, sycl::memory_order::relaxed, sycl::memory_scope::device> a(*bad);
                a.fetch_add(1u);
            }
        }).wait();
        unsigned nbad = *bad;
        sycl::free(bad, q);
        if (!nbad) { got = p; break; }
        {   // what the bad blocks hold: another block's tag (aliasing), or something else
            std::vector<uint64_t> v(nb);
            q.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {}).wait();
            for (size_t i = 0; i < nb; i++) q.memcpy(&v[i], p + i * BLK, 8);
            q.wait();
            uint64_t inv = 0xD1B54A32D192ED03ull;     // Newton: inverse of the odd multiplier mod 2^64
            for (int k = 0; k < 6; k++) inv *= 2 - 0xD1B54A32D192ED03ull * inv;
            size_t first = SIZE_MAX, last = 0, alias = 0, shown = 0;
            for (size_t i = 0; i < nb; i++) {
                uint64_t want = (uint64_t)i * 0xD1B54A32D192ED03ull ^ salt;
                if (v[i] == want) continue;
                first = std::min(first, i);
                last = i;
                uint64_t src = (v[i] ^ salt) * inv;
                alias += src < nb;
                if (shown++ < 4)
                    fprintf(stderr, "  block %zu (at %.3f GiB) holds %s %llu\n", i, i * (double)BLK / 1073741824.0,
                            src < nb ? "the tag of block" : "a foreign value, as block", (unsigned long long)src);
            }
            fprintf(stderr, "  bad blocks %zu..%zu (%.3f..%.3f GiB), %zu hold another block's tag\n", first, last,
                    first * (double)BLK / 1073741824.0, last * (double)BLK / 1073741824.0, alias);
        }
        fprintf(stderr, "warning: %.1f GiB device allocation has %u aliased 64 KiB blocks: allocating again\n",
                bytes / 1073741824.0, nbad);
        sycl::free(p, q);                           // and a growing spacer first, so the next one lands elsewhere
        spacers.push_back(sycl::malloc_device((size_t)(attempt + 1) << 26, q));
    }
    for (void* sp : spacers) if (sp) sycl::free(sp, q);
    return got;
}

std::string pick(const std::string& env, const std::string& def) {
    const char* e = getenv(env.c_str());
    return e && *e ? e : def;
}
}  // namespace

// ------------------------------------------------------------------------------------------------ n-gram table

E0Ngram::E0Ngram(const std::string& path, const Gguf& skel) {
    Gguf g(path, false);
    GLM_CHECK(g.has_kv("strata.ple.format") && g.get_s("strata.ple.format") == "f8_e4m3", "%s: not an FP8 PLE table",
              path.c_str());
    scale = (float)g.get_f("strata.ple.scale");
    const GgufTensor& t = g.at("per_layer_token_embd.weight");
    GLM_CHECK(t.type == GG_I8 && t.cols() == 160 && t.rows() == 320001536, "%s: table shape", path.c_str());
    base = t.offset;
    auto arr = [&](const std::string& k, int64_t* dst, size_t n) {
        const GgufValue& v = skel.val(k);
        GLM_CHECK(v.kind == GgufValue::Arr && v.arr.size() == n, "%s: %zu values expected", k.c_str(), n);
        for (size_t i = 0; i < n; i++) dst[i] = v.arr[i].i;
    };
    arr("qwen4exp.ple.layer_multipliers", mult, 3);
    arr("qwen4exp.ple.head_offsets", offs, 16);
    arr("qwen4exp.ple.head_vocab_sizes", vsize, 16);
    GLM_CHECK(skel.get_i("qwen4exp.ple.eos_token_id") == EOS && skel.get_i("qwen4exp.ple.ngram_size") == 3 &&
                  skel.get_i("qwen4exp.ple.heads_per_ngram") == 8, "PLE hash parameters");
    for (int b = 0; b < 256; b++) {                 // E4M3 (fn): bias 7, no infinities, 0x7F / 0xFF NaN
        int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
        float v = e == 0 ? std::ldexp((float)m, -9) : std::ldexp(1.0f + m / 8.0f, e - 7);
        if (e == 15 && m == 7) v = NAN;
        lut[b] = (s ? -v : v) * scale;
    }
    fd = open(path.c_str(), O_RDONLY);
    GLM_CHECK(fd >= 0, "cannot open %s", path.c_str());
}

E0Ngram::~E0Ngram() {
    if (fd >= 0) close(fd);
}

void E0Ngram::lookup(const int* hist, int n, half* emb) {
    const int64_t m0 = mult[0], m1 = mult[1], m2 = mult[2];
    pool().each(n * 16, [&](int i) {
        int t = i / 16, h = i % 16;
        const int* hp = hist + t + 2;
        int64_t t0 = hp[0], t1 = hp[-1], t2 = hp[-2];
        if (t1 == EOS) t2 = EOS;
        int64_t mix = (t0 * m0) ^ (t1 * m1);
        if (h >= 8) mix ^= t2 * m2;
        int64_t row = mix % vsize[h] + offs[h];
        uint8_t b[160];
        ssize_t got = pread(fd, b, 160, (off_t)(base + (uint64_t)row * 160));
        GLM_CHECK(got == 160, "n-gram table: read row %lld", (long long)row);
        half* out = emb + (size_t)t * D + h * 160;
        for (int p = 0; p < 160; p++) out[p] = half(lut[b[p]]);
    });
}

// ------------------------------------------------------------------------------------------------ cache

void E0Cache::init(int T, bool int8) {
    i8 = int8;
    k = Tensor::empty({T, AKV, AD}, i8 ? DType::U8 : DType::F16);
    v = Tensor::empty({T, AKV, AD}, i8 ? DType::U8 : DType::F16);
    if (i8) {
        ks = Tensor::empty({T, AKV, AD / 64}, DType::F16);
        vs = Tensor::empty({T, AKV, AD / 64}, DType::F16);
    }
    rk = Tensor::empty({T, ID}, DType::F16);
    pk = Tensor::empty({T / BLK + 1, ID}, DType::F16);
}

// ------------------------------------------------------------------------------------------------ load

E0Model::E0Model(const std::string& dir, int max_tok, double reserve_gib) : max_tokens(max_tok) {
    max_tokens = (max_tokens + BLK - 1) / BLK * BLK;
    auto& q = Dev::get().q;
    if (getenv("E0_REBALANCE")) rebal_every = atoi(getenv("E0_REBALANCE"));
    kv8 = !(getenv("E0_KV16") && atoi(getenv("E0_KV16")));
    if (getenv("GLM_PREFILL_CHUNK")) prefill_chunk = std::max(64, std::min(2048, atoi(getenv("GLM_PREFILL_CHUNK"))));
    if (getenv("E0_MOE_ROWS")) moe_rows = std::max(prefill_chunk, atoi(getenv("E0_MOE_ROWS")));
    moe_rows = (moe_rows + prefill_chunk - 1) / prefill_chunk * prefill_chunk;
    skel = std::make_unique<Gguf>(dir + "/model-skeleton.gguf");
    const Gguf& g = *skel;
    GLM_CHECK(g.get_s("general.architecture") == "qwen4exp" && g.get_i("qwen4exp.block_count") == L &&
                  g.get_i("qwen4exp.embedding_length") == D && g.get_i("qwen4exp.expert_count") == E &&
                  g.get_i("qwen4exp.expert_used_count") == TOPK && g.get_i("qwen4exp.expert_feed_forward_length") == I &&
                  g.get_i("qwen4exp.hyper_connection.count") == S,
              "%s: not the Edge0 / Qwen3.8-Flash-Next shape this engine is built for", dir.c_str());
    ngram = std::make_unique<E0Ngram>(pick("E0_PLE", dir + "/ple-fp8.gguf"), g);

    const GgufTensor& et = g.at("token_embd.weight");
    GLM_CHECK(et.type == GG_Q8_0 && et.rows() == V && et.cols() == D, "token_embd: expected Q8_0 [V, D]");
    embed = static_cast<const uint8_t*>(et.data);
    lm_head = q8_load(g.at("output.weight"));
    mixer = load_site(g, "output_hc_norm.weight", "output_hc_down.weight", "", "output_hc_up.weight");

    // the GDN permutations back to HF's grouped value heads
    std::vector<int> p_qkv(GQKV), p_z(GV * GDV), p_h(GV), p_oc(GV);
    for (int r = 0; r < GQKV; r++) {
        const int qk = 2 * GK * GDK;
        p_qkv[r] = r < qk ? r : qk + vh_src((r - qk) / GDV) * GDV + (r - qk) % GDV;
    }
    for (int r = 0; r < GV * GDV; r++) p_z[r] = vh_src(r / GDV) * GDV + r % GDV;
    for (int h = 0; h < GV; h++) p_h[h] = p_oc[h] = vh_src(h);
    auto permute = [](const std::vector<float>& v, const std::vector<int>& p, int inner) {
        std::vector<float> o(v.size());
        for (size_t r = 0; r < p.size(); r++)
            for (int i = 0; i < inner; i++) o[r * inner + i] = v[(size_t)p[r] * inner + i];
        return o;
    };

    // latent D/U and labels
    const std::string du_path = pick("E0_DU", dir + "/latent_dense_q8.bin");
    int du_fd = open(du_path.c_str(), O_RDONLY);
    GLM_CHECK(du_fd >= 0, "cannot open %s", du_path.c_str());
    const size_t d_bytes = (size_t)GE * (D / 32) * 34, u_bytes = (size_t)G * D * (ELL / 32) * 34;
    GLM_CHECK(lseek(du_fd, 0, SEEK_END) == (off_t)(L * (d_bytes + u_bytes)), "%s: size (expected the Q8 D/U pack)",
              du_path.c_str());
    {
        std::string lb = read_file(dir + "/labels.bin");
        GLM_CHECK(lb.size() == (size_t)L * E * 4, "labels.bin: %zu bytes", lb.size());
        labels_h.resize((size_t)L * E);
        memcpy(labels_h.data(), lb.data(), lb.size());
        for (int v : labels_h) GLM_CHECK(v >= 0 && v < G, "labels.bin: label %d", v);
        labels_dev = Tensor::empty({L, E}, DType::I32);
        upload(labels_dev.p, labels_h.data(), lb.size());
    }
    std::vector<uint8_t> dbuf(d_bytes), ubuf(u_bytes), ucat((size_t)D * (GE / 32) * 34);

    blocks.resize(L);
    gst.resize(L);
    kv.resize(L);
    for (int li = 0; li < L; li++) {
        E0Block& b = blocks[li];
        std::string p = bp(li);
        b.layer = li;
        b.attn = !g.val("qwen4exp.attention.recurrent_layers").arr[li].i;
        GLM_CHECK(b.attn == (li % 4 == 3), "layer %d: unexpected layer type", li);
        b.site[0] = load_site(g, p + "hc_attn_norm.weight", p + "hc_attn_down.weight", p + "hc_attn_inject.weight",
                              p + "hc_attn_up.weight");
        b.site[1] = load_site(g, p + "hc_ffn_norm.weight", p + "hc_ffn_down.weight", p + "hc_ffn_inject.weight",
                              p + "hc_ffn_up.weight");
        if (b.attn) {
            auto a = std::make_unique<E0Attn>();
            a->q = q8_load(g.at(p + "attn_q.weight"));
            a->k = q8_load(g.at(p + "attn_k.weight"));
            a->v = q8_load(g.at(p + "attn_v.weight"));
            a->o = q8_load(g.at(p + "attn_output.weight"));
            GLM_CHECK(a->q.N == AH * 2 * AD && a->k.N == AKV * AD && a->v.N == AKV * AD && a->o.K == AH * AD,
                      "layer %d: attention shapes", li);
            a->idx = stack_f16({&g.at(p + "indexer.q_proj.weight"), &g.at(p + "indexer.k_proj.weight")});
            GLM_CHECK(a->idx.size(0) == (IH + 1) * ID, "layer %d: indexer shapes", li);
            a->qn = dev_f32(host_f32(g.at(p + "attn_q_norm.weight")));
            a->kn = dev_f32(host_f32(g.at(p + "attn_k_norm.weight")));
            a->iqn = dev_f32(host_f32(g.at(p + "indexer.q_norm.weight")));
            a->ikn = dev_f32(host_f32(g.at(p + "indexer.k_norm.weight")));
            b.att = std::move(a);
            kv[li].init(max_tokens, kv8);
        } else {
            auto gd = std::make_unique<E0Gdn>();
            gd->qkv = q8_load(g.at(p + "attn_qkv.weight"), p_qkv);
            gd->z = q8_load(g.at(p + "attn_gate.weight"), p_z);
            gd->o = q8_load(g.at(p + "ssm_out.weight"), {}, p_oc);
            GLM_CHECK(gd->qkv.N == GQKV && gd->z.N == GV * GDV && gd->o.K == GV * GDV, "layer %d: GDN shapes", li);
            gd->ab = stack_f16({&g.at(p + "ssm_beta.weight"), &g.at(p + "ssm_alpha.weight")}, {p_h, p_h});
            {
                const GgufTensor& c = g.at(p + "ssm_conv1d.weight");
                GLM_CHECK(c.cols() == GCONV && c.rows() == GQKV, "layer %d: conv shape", li);
                auto cv = permute(host_f32(c), p_qkv, GCONV);
                std::vector<bf16> cb(cv.size());
                size_t inexact = 0;
                for (size_t i = 0; i < cv.size(); i++) { cb[i] = bf16(cv[i]); inexact += f(cb[i]) != cv[i]; }
                if (inexact && li == 0) fprintf(stderr, "note: GDN conv weights are not bf16-exact (%zu values)\n", inexact);
                gd->conv_w = Tensor::empty({GQKV, GCONV}, DType::BF16);
                upload(gd->conv_w.p, cb.data(), cb.size() * 2);
            }
            gd->neg_a = dev_f32(permute(host_f32(g.at(p + "ssm_a")), p_h, 1));
            gd->dt_bias = dev_f32(permute(host_f32(g.at(p + "ssm_dt.bias")), p_h, 1));
            gd->norm_w = dev_f32(host_f32(g.at(p + "ssm_norm.weight")));
            b.gdn = std::move(gd);
            gst[li].init();
        }
        b.router = stack_f16({&g.at(p + "ffn_gate_inp.weight"), &g.at(p + "ffn_gate_inp_shexp.weight")});
        GLM_CHECK(b.router.size(0) == E + 1, "layer %d: router", li);
        b.sg = q8_load(g.at(p + "ffn_gate_shexp.weight"));
        b.su = q8_load(g.at(p + "ffn_up_shexp.weight"));
        b.sd = q8_load(g.at(p + "ffn_down_shexp.weight"));
        GLM_CHECK(b.sg.N == I && b.sg.K == D && b.sd.N == D && b.sd.K == I, "layer %d: shared expert", li);
        // D [8 * 1792, 2560] as stored; U's 8 [2560, 1792] side by side
        const off_t lo = (off_t)li * (d_bytes + u_bytes);
        GLM_CHECK(pread(du_fd, dbuf.data(), d_bytes, lo) == (ssize_t)d_bytes &&
                      pread(du_fd, ubuf.data(), u_bytes, lo + d_bytes) == (ssize_t)u_bytes, "%s: read", du_path.c_str());
        b.dcat = q8_from_blocks(dbuf.data(), GE, D);
        const size_t urow = (size_t)(ELL / 32) * 34;
        for (int n = 0; n < D; n++)
            for (int gg = 0; gg < G; gg++)
                memcpy(&ucat[(size_t)n * G * urow + gg * urow], &ubuf[((size_t)gg * D + n) * urow], urow);
        b.ucat = q8_from_blocks(ucat.data(), D, GE);
        load_progress("loaded layer %d/%d (%.2f GiB VRAM)", li + 1, L, Dev::get().vram_used / 1073741824.0);
    }
    load_progress_end();
    close(du_fd);
    {
        std::string pp = bp(1);
        ple_wkv = stack_f16({&g.at(pp + "ple_key.weight"), &g.at(pp + "ple_value.weight")});
        GLM_CHECK(ple_wkv.size(0) == SD + D && ple_wkv.size(1) == D, "PLE projections");
        ple_nk = dev_f32(host_f32(g.at(pp + "ple_norm_key.weight")));
        ple_nq = dev_f32(host_f32(g.at(pp + "ple_norm_query.weight")));
        ple_nc = dev_f32(host_f32(g.at(pp + "ple_norm_conv.weight")));
        const GgufTensor& c = g.at(pp + "ple_conv1d.weight");
        GLM_CHECK(c.cols() == 4 && c.rows() == SD, "PLE conv shape");
        ple_cw = dev_f32(host_f32(c));
        ple_state = Tensor::zeros({SD, 9}, DType::F16);
    }
    if (g_f16_over || g_f16_sub)
        fprintf(stderr, "note: fp16 copies of %zu small-tensor values: %zu past fp16's range, %zu in its subnormals\n",
                g_f16_n, g_f16_over, g_f16_sub);

    const int C = std::max(qmax, prefill_chunk);
    act.init((size_t)C * 120 * 1024 + q4ops::attn_part_floats(256) * 4 + ((size_t)64 << 20));
    X = Tensor::empty({qmax, S, D}, DType::F32);
    n16 = Tensor::empty({C, SD}, DType::F16);
    t324 = Tensor::empty({C, R + S}, DType::F32);
    a16 = Tensor::empty({C, R}, DType::F16);
    gpre = Tensor::empty({C, SD}, DType::F32);
    y16 = Tensor::empty({C, D}, DType::F16);
    sub = Tensor::empty({C, D}, DType::F32);
    inj = Tensor::empty({C, S}, DType::F32);
    rl = Tensor::empty({std::max(C, moe_rows), E + 1}, DType::F16);
    sel = Tensor::empty({std::max(C, moe_rows), TOPK}, DType::I32);
    wts = Tensor::empty({std::max(C, moe_rows), TOPK}, DType::F32);
    sg16 = Tensor::empty({std::max(C, moe_rows), I}, DType::F16);
    su16 = Tensor::empty({std::max(C, moe_rows), I}, DType::F16);
    sa16 = Tensor::empty({std::max(C, moe_rows), I}, DType::F16);
    sh32 = Tensor::empty({std::max(C, moe_rows), D}, DType::F32);
    emb16 = Tensor::empty({C, D}, DType::F16);
    kv32 = Tensor::empty({C, SD + D}, DType::F32);
    gv32 = Tensor::empty({C, SD}, DType::F32);
    cn16 = Tensor::empty({C, SD}, DType::F16);
    fin16 = Tensor::empty({qmax, D}, DType::F16);
    logits = Tensor::empty({qmax, V}, DType::F16);
    xl32 = Tensor::empty({MV_MAX, GE}, DType::F32);
    h32 = Tensor::empty({MV_MAX * TOPK, I}, DType::F32);
    yl32 = Tensor::empty({MV_MAX * TOPK, ELL}, DType::F32);
    z32 = Tensor::empty({MV_MAX, GE}, DType::F32);
    {
        size_t most = 0;                            // the largest Q8 weight a prompt chunk dequantizes
        for (auto& b : blocks) {
            if (b.att) most = std::max(most, (size_t)b.att->q.N * b.att->q.K);
            if (b.gdn) most = std::max(most, (size_t)b.gdn->qkv.N * b.gdn->qkv.K);
        }
        deq = Tensor::empty({(int64_t)most}, DType::F16);
    }
    const int MR = moe_rows;
    moe_grouped = getenv("E0_MOE_GROUPED") && atoi(getenv("E0_MOE_GROUPED"));
    moe_dpas = !(getenv("E0_MOE_PATH") && std::string(getenv("E0_MOE_PATH")) == "expert");
    group_skip = !(getenv("E0_GROUP_SKIP") && atoi(getenv("E0_GROUP_SKIP")) == 0);
    stage_on = getenv("E0_STAGE") && atoi(getenv("E0_STAGE"));
    if (getenv("E0_MOE_BATCH")) moe_batch = std::max(1, std::min(E, atoi(getenv("E0_MOE_BATCH"))));
    if (getenv("E0_PICK_CAP")) pick_cap = std::max(MR, atoi(getenv("E0_PICK_CAP")));
    pick_cap = std::max(pick_cap, MR);                // one expert takes at most MR picks
    const int PC = pick_cap;
    stage = Tensor::empty({64, (int64_t)((NV_REC + 4095) / 4096 * 4096)}, DType::U8);   // RAM records on their way
    blk_cap = MR * TOPK / 64 + 2 * E + 256;
    blk_rec = Tensor::empty({blk_cap}, DType::I64);
    blk_r0 = Tensor::empty({blk_cap + 3 * (int64_t)MR * TOPK + 1024}, DType::I32);   // + the combine lists
    blk_n = Tensor::empty({blk_cap}, DType::I32);
    ddeq = Tensor::empty({GE, D}, DType::F16);
    udeq = Tensor::empty({D, GE}, DType::F16);
    xl16 = Tensor::empty({MR, GE}, DType::F16);
    bgu = Tensor::empty({moe_batch, 2 * I, ELL}, DType::F16);
    bdn = Tensor::empty({moe_batch, ELL, I}, DType::F16);
    Xb = Tensor::empty({PC, ELL}, DType::F16);
    GUb = Tensor::empty({PC, 2 * I}, DType::F32);
    h16b = Tensor::empty({PC, I}, DType::F16);
    Yb = Tensor::empty({PC, ELL}, DType::F32);
    Yh = Tensor::empty({PC, ELL}, DType::F16);
    Ob = Tensor::empty({PC, D}, DType::F32);
    yall = Tensor::empty({MR, D}, DType::F16);
    injall = Tensor::empty({MR, S}, DType::F32);
    outall = Tensor::empty({MR, D}, DType::F32);
    recs_sh = sycl::malloc_shared<int64_t>(E, q);
    pa_sh = sycl::malloc_shared<const half*>(3 * E, q);
    pb_sh = sycl::malloc_shared<const half*>(3 * E, q);
    pc_sh = sycl::malloc_shared<float*>(3 * E, q);
    GLM_CHECK(recs_sh && pa_sh && pb_sh && pc_sh, "grouped GEMM pointer arrays");
    for (auto* v : {&gm_m, &gm_n, &gm_k, &gm_lda, &gm_ldb, &gm_ldc, &gm_gs}) v->assign(3 * E, 1);
    gm_alpha.assign(3 * E, 1.0f);
    gm_beta.assign(3 * E, 0.0f);
    gm_ta.assign(3 * E, oneapi::mkl::transpose::nontrans);
    gm_tb.assign(3 * E, oneapi::mkl::transpose::trans);
    elist = Tensor::empty({5 * (int64_t)MR * TOPK}, DType::I32);   // per pick: row, column, weight, s_g, s_u
    pos_dev = Tensor::zeros({1}, DType::I32);
    pos_h = sycl::malloc_host<int>(1, q);
    graphs_on = !(getenv("E0_GRAPHS") && atoi(getenv("E0_GRAPHS")) == 0) && !g_prof_points && !g_sync_points;
    x_host = sycl::malloc_host<float>((size_t)C * D, q);
    e_host = sycl::malloc_host<half>((size_t)C * D, q);
    GLM_CHECK(x_host && e_host, "pinned upload buffers");
    q.wait();
    {
        int spec = getenv("E0_SPEC") ? atoi(getenv("E0_SPEC")) : 3;
        if (getenv("E0_SPEC_MIN_P")) spec_min_p = (float)atof(getenv("E0_SPEC_MIN_P"));
        if (spec > 0 && access((dir + "/mtp/dense.txt").c_str(), R_OK) == 0) {
            load_mtp(dir);
            draft_tokens = std::min(spec, qmax - 1);
        }
    }
    load_experts(pick("E0_EXPERTS", dir + "/experts.bin"), dir + "/profile.bin", reserve_gib);
    if (!getenv("E0_WARMUP") || atoi(getenv("E0_WARMUP")) != 0) {
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

E0Model::~E0Model() {
    auto& q = Dev::get().q;
    q.wait();
    for (auto& c : ckpts_) {
        for (void* p : c.conv) if (p) sycl::free(p, q);
        for (void* p : c.rec) if (p) sycl::free(p, q);
        if (c.ple) sycl::free(c.ple, q);
    }
    for (auto& gp : dec_graphs) gp.reset();
    if (swap_buf) sycl::free(swap_buf, q);
    if (pos_h) sycl::free(pos_h, q);
    if (mtp_emb_h) sycl::free(mtp_emb_h, q);
    if (mtp && mtp->experts) sycl::free(mtp->experts, q);
    if (x_host) sycl::free(x_host, q);
    if (e_host) sycl::free(e_host, q);
    for (char* c : vram_chunks) sycl::free(c, q);
    for (void* p : {(void*)recs_sh, (void*)pa_sh, (void*)pb_sh, (void*)pc_sh}) if (p) sycl::free(p, q);
    if (ram_pool) sycl::free(ram_pool, q);
}

// ------------------------------------------------------------------------------------------------ experts

void E0Model::load_experts(const std::string& path, const std::string& profile, double reserve_gib) {
    auto& q = Dev::get().q;
    const size_t rec = NV_REC;
    rec_bytes = (rec + 4095) / 4096 * 4096;
    const int NK = L * E;
    // order: profile.bin's ranking (hottest first), then the rest spread over the layers
    std::vector<int> order;
    std::vector<uint8_t> placed(NK, 0);
    if (FILE* pf = fopen(profile.c_str(), "rb")) {
        char magic[4];
        uint32_t hdr[5];
        GLM_CHECK(fread(magic, 1, 4, pf) == 4 && fread(hdr, 4, 5, pf) == 5 && !memcmp(magic, "STRP", 4) && hdr[0] == 1 &&
                      hdr[1] == (uint32_t)L && hdr[2] == (uint32_t)E, "%s: not a STRP v1 profile for 48 x 512", profile.c_str());
        std::vector<uint16_t> pr((size_t)hdr[4] * 2);
        GLM_CHECK(fread(pr.data(), 2, pr.size(), pf) == pr.size(), "%s: truncated", profile.c_str());
        fclose(pf);
        for (size_t i = 0; i < hdr[4]; i++) {
            int key = pr[2 * i] * E + pr[2 * i + 1];
            if (pr[2 * i] < L && pr[2 * i + 1] < E && !placed[key]) { placed[key] = 1; order.push_back(key); }
        }
    } else {
        fprintf(stderr, "note: no %s, experts take the VRAM slots in an even spread\n", profile.c_str());
    }
    for (int e = 0; e < E; e++)
        for (int li = 0; li < L; li++)
            if (!placed[li * E + e]) order.push_back(li * E + e);

    size_t total = Dev::get().dev.get_info<sycl::info::device::global_mem_size>();
    size_t used = Dev::get().vram_used + (size_t)(reserve_gib * 1073741824.0);
    int nv = total > used ? (int)std::min<size_t>(NK, (total - used) / rec_bytes) : 0;
    if (getenv("E0_VRAM_EXPERTS")) nv = std::min(nv, atoi(getenv("E0_VRAM_EXPERTS")));
    int nr = NK - nv;
    // the VRAM slots in checked pieces of E0_VRAM_CHUNK_GIB (default 2): one ~20 GiB allocation came back with its
    // last GiBs mapped over its first (a write 16 GiB in landed at the start); a piece that fails its check is dropped
    // and its experts stay in RAM
    std::vector<char*> slot;                        // VRAM slot addresses
    {
        double cg = getenv("E0_VRAM_CHUNK_GIB") ? atof(getenv("E0_VRAM_CHUNK_GIB")) : 2.0;
        int per = std::max(1, (int)(cg * 1073741824.0 / rec_bytes));
        while ((int)slot.size() < nv) {
            int k = std::min(per, nv - (int)slot.size());
            char* c = malloc_checked((size_t)k * rec_bytes, q, 2);
            if (!c) {
                fprintf(stderr, "warning: a VRAM piece of %d expert slots failed its check: %zu slots in VRAM, the rest "
                        "in RAM\n", k, slot.size());
                break;
            }
            vram_chunks.push_back(c);
            for (int j = 0; j < k; j++) slot.push_back(c + (size_t)j * rec_bytes);
        }
        nv = (int)slot.size();
        nr = NK - nv;
    }
    Dev::get().vram_used += (size_t)nv * rec_bytes;
    fprintf(stderr, "experts: %d in VRAM (%.1f GiB), %d in RAM (%.1f GiB)\n", nv, nv * (double)rec_bytes / 1073741824.0,
            nr, nr * (double)rec_bytes / 1073741824.0);
    ram_pool = nr ? static_cast<char*>(sycl::malloc_host((size_t)nr * rec_bytes, q)) : nullptr;
    GLM_CHECK(!nr || ram_pool, "expert RAM pool (%d records, %.1f GiB pinned) allocation failed", nr,
              nr * (double)rec_bytes / 1073741824.0);
    in_vram.assign(NK, 0);
    ptrs_h.assign(NK, 0);
    for (int i = 0; i < NK; i++) {
        int key = order[i];
        in_vram[key] = i < nv;
        ptrs_h[key] = (int64_t)(i < nv ? slot[i] : ram_pool + (size_t)(i - nv) * rec_bytes);
    }

    int fd = open(path.c_str(), O_RDONLY);
    GLM_CHECK(fd >= 0, "cannot open %s", path.c_str());
    GLM_CHECK(lseek(fd, 0, SEEK_END) == (off_t)((size_t)NK * rec), "%s: size (expected %d records of %zu B)",
              path.c_str(), NK, rec);
    posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
    scales_h.assign((size_t)NK * 3, 0.0f);
    auto t0 = std::chrono::steady_clock::now();
    const int B = 64;
    char* bounce[2] = {static_cast<char*>(sycl::malloc_host((size_t)B * rec, q)),
                       static_cast<char*>(sycl::malloc_host((size_t)B * rec, q))};
    // (ptrs_h is in file order here: key = layer * E + expert, the order the records are read in)
    char* raw_dev[2] = {static_cast<char*>(sycl::malloc_device((size_t)B * rec, q)),
                        static_cast<char*>(sycl::malloc_device((size_t)B * rec, q))};
    int64_t* dst_dev[2] = {sycl::malloc_device<int64_t>(B, q), sycl::malloc_device<int64_t>(B, q)};
    char* tiled_dev[2] = {static_cast<char*>(sycl::malloc_device((size_t)B * rec, q)),
                          static_cast<char*>(sycl::malloc_device((size_t)B * rec, q))};
    GLM_CHECK(tiled_dev[0] && tiled_dev[1], "expert repack buffers");
    GLM_CHECK(bounce[0] && bounce[1] && raw_dev[0] && raw_dev[1] && dst_dev[0] && dst_dev[1], "expert bounce buffers");
    sycl::event done[2];
    for (int i0 = 0, k = 0; i0 < NK; i0 += B, k ^= 1) {
        int n = std::min(B, NK - i0);
        done[k].wait();                             // its last copies to VRAM have read it
        char* bb = bounce[k];
        // the file in 8 pieces on the pool (one pread of 124 MB leaves the SSD idle between requests)
        pool().each(8, [&](int j) {
            size_t a = (size_t)n * rec * j / 8, z = (size_t)n * rec * (j + 1) / 8;
            for (size_t o = a; o < z;) {
                ssize_t got = pread(fd, bb + o, z - o, (off_t)((size_t)i0 * rec + o));
                GLM_CHECK(got > 0, "%s: read at record %d", path.c_str(), i0);
                o += got;
            }
        });
        posix_fadvise(fd, (off_t)((size_t)i0 * rec), (off_t)((size_t)n * rec), POSIX_FADV_DONTNEED);
        // repacked into the tiled layout (e0_ops.h) on the GPU, straight into the VRAM slots and pinned RAM records
        for (int j = 0; j < n; j++) memcpy(&scales_h[(size_t)(i0 + j) * 3], bb + (size_t)j * rec + NV_TAIL, 12);
        q.memcpy(raw_dev[k], bb, (size_t)n * rec);
        // VRAM records straight into their slots; RAM ones into tiled_dev, then one copy each to the host
        std::vector<int64_t> dst(n);
        for (int j = 0; j < n; j++)
            dst[j] = in_vram[i0 + j] ? ptrs_h[i0 + j] : (int64_t)(tiled_dev[k] + (size_t)j * rec);
        q.memcpy(dst_dev[k], dst.data(), (size_t)n * 8);
        nv_repack_dev(q, raw_dev[k], dst_dev[k], n);
        for (int j = 0; j < n; j++)
            if (!in_vram[i0 + j]) q.memcpy(reinterpret_cast<void*>(ptrs_h[i0 + j]), tiled_dev[k] + (size_t)j * rec, rec);
        q.wait();                                   // (dst lives on this frame)
        std::vector<sycl::event> evs = {q.ext_oneapi_submit_barrier()};
        done[k] = q.ext_oneapi_submit_barrier(evs);
        if ((i0 / B) % 16 == 0) load_progress("experts: %d/%d", i0 + n, NK);
    }
    q.wait();
    close(fd);
    for (void* b2 : {(void*)bounce[0], (void*)bounce[1], (void*)raw_dev[0], (void*)raw_dev[1], (void*)dst_dev[0],
                     (void*)dst_dev[1], (void*)tiled_dev[0], (void*)tiled_dev[1]})
        sycl::free(b2, q);
    load_progress_end("experts: %d records in %.1f s", NK,
                      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    ptrs_dev = Tensor::empty({L, E}, DType::I64);
    upload(ptrs_dev.p, ptrs_h.data(), ptrs_h.size() * 8);
    use_dev = Tensor::zeros({L, E}, DType::I32);
    in_vram_dev = Tensor::empty({L, E}, DType::U8);
    upload(in_vram_dev.p, in_vram.data(), NK);
    stats_dev = Tensor::zeros({2}, DType::I64);
}

void E0Model::rebalance() {
    auto& q = Dev::get().q;
    const int NK = L * E;
    dec_ev.wait();
    q.wait();
    std::vector<int> use(NK);
    download(use.data(), use_dev.p, (size_t)NK * 4);
    q.memset(use_dev.p, 0, (size_t)NK * 4);
    if (score_.empty()) score_.assign(NK, 0.0f);
    for (int k = 0; k < NK; k++) score_[k] = score_[k] * 0.5f + use[k];
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
    const size_t rec = NV_REC;
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
    }
    upload(ptrs_dev.p, ptrs_h.data(), ptrs_h.size() * 8);
    upload(in_vram_dev.p, in_vram.data(), NK);
    swapped_ += n;
}

std::string E0Model::expert_stats() {
    auto& q = Dev::get().q;
    q.wait();
    uint64_t st[2];
    download(st, stats_dev.p, 16);
    std::vector<int> use((size_t)L * E);
    download(use.data(), use_dev.p, use.size() * 4);
    q.memset(stats_dev.p, 0, 16);
    q.wait();
    size_t ram = 0;
    for (size_t k = 0; k < use.size(); k++) ram += !in_vram[k] && use[k];
    char b[200];
    snprintf(b, sizeof b, "experts: %llu picks, %.1f%% in RAM (%zu distinct RAM experts), %zu swapped so far\n",
             (unsigned long long)st[0], st[0] ? 100.0 * st[1] / st[0] : 0.0, ram, swapped_);
    return b;
}

// ------------------------------------------------------------------------------------------------ pieces

void E0Model::grab(const void* p, int n, bool is16) {
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

void E0Model::embed_host(const int* tokens, int n) {
    const size_t rb = (size_t)(D / 32) * 34;
    for (int t = 0; t < n; t++) {
        GLM_CHECK(tokens[t] >= 0 && tokens[t] < V, "token id %d", tokens[t]);
        const uint8_t* row = embed + (size_t)tokens[t] * rb;
        float* o = x_host + (size_t)t * D;
        for (int b = 0; b < D / 32; b++) {
            uint16_t db;
            memcpy(&db, row + b * 34, 2);
            float sc = f16f(db);
            const int8_t* qs = reinterpret_cast<const int8_t*>(row + b * 34 + 2);
            for (int j = 0; j < 32; j++) o[b * 32 + j] = sc * qs[j];
        }
    }
}

void E0Model::embed_rows(const int* tokens, int n, float* Xd) {
    auto& q = Dev::get().q;
    x_ev.wait();
    embed_host(tokens, n);
    float* e = sub.ptr<float>();
    x_ev = q.memcpy(e, x_host, (size_t)n * D * 4);
    q4ops::expand(q, e, Xd, n);
}

void E0Model::site_mix(const Q4Site& s, const float* Xd, int rows, half* y, float* injp) {
    auto& q = Dev::get().q;
    q4ops::site_norm(q, Xd, s.w1p.ptr<float>(), n16.ptr<half>(), rows);
    ops::gemv_f16<float>(q, n16.ptr<half>(), rows, SD, s.wdi.ptr<half>(), s.nd, t324.ptr<float>());
    q4ops::site_gates(q, t324.ptr<float>(), s.nd, a16.ptr<half>(), injp, rows);
    ops::gemv_f16<float>(q, a16.ptr<half>(), rows, R, s.wu.ptr<half>(), SD, gpre.ptr<float>());
    q4ops::site_combine(q, gpre.ptr<float>(), n16.ptr<half>(), y, rows);
}

void E0Model::gdn_forward(const E0Gdn& g, Q4GdnState& s, const half* y, int rows, float* out) {
    auto& q = Dev::get().q;
    half* dq = deq.ptr<half>();
    act.reset();
    float* qkv = act.take<float>((size_t)rows * GQKV);
    float* z = act.take<float>((size_t)rows * GV * GDV);
    float* ab = act.take<float>((size_t)rows * 2 * GV);
    float* beta = act.take<float>((size_t)rows * GV);
    float* gg = act.take<float>((size_t)rows * GV);
    bf16* cv = act.take<bf16>((size_t)rows * GQKV);
    bf16* core = act.take<bf16>((size_t)rows * GV * GDV);
    half* nrm = act.take<half>((size_t)rows * GV * GDV);
    q8_linear<float>(q, y, rows, g.qkv, qkv, dq);
    q8_linear<float>(q, y, rows, g.z, z, dq);
    prof_point("GDN: qkv, z");
    ops::gemv_f16<float>(q, y, rows, D, g.ab.ptr<half>(), 2 * GV, ab);
    q4ops::gdn_gates(q, ab, g.neg_a.ptr<float>(), g.dt_bias.ptr<float>(), beta, gg, rows);
    q4ops::gdn_conv(q, qkv, s.conv.ptr<bf16>(), g.conv_w.ptr<bf16>(), cv, rows);
    if (saving) {                                   // a verify window: what a replay of a prefix of its rows needs
        const int li = (int)(&s - gst.data());
        q.memcpy(sv_qkv[li].p, qkv, (size_t)rows * GQKV * 4);
        q.memcpy(sv_cv[li].p, cv, (size_t)rows * GQKV * 2);
        q.memcpy(sv_beta[li].p, beta, (size_t)rows * GV * 4);
        q.memcpy(sv_g[li].p, gg, (size_t)rows * GV * 4);
    }
    prof_point("GDN: a b gates conv");
    q4ops::gdn_delta(q, cv, beta, gg, s.rec.ptr<float>(), core, rows);
    prof_point("GDN: delta rule");
    q4ops::gdn_norm(q, core, z, g.norm_w.ptr<float>(), nrm, rows);
    q8_linear<float>(q, nrm, rows, g.o, out, dq);
    prof_point("GDN: norm + out_proj");
}

void E0Model::attn_forward(const E0Attn& a, E0Cache& c, const half* y, int rows, int p0, float* out, const int* pdev) {
    auto& q = Dev::get().q;
    half* dq = deq.ptr<half>();
    act.reset();
    half* qg = act.take<half>((size_t)rows * AH * 2 * AD);
    half* k = act.take<half>((size_t)rows * AKV * AD);
    half* v = act.take<half>((size_t)rows * AKV * AD);
    half* ik = act.take<half>((size_t)rows * (IH + 1) * ID);
    half* qn16 = act.take<half>((size_t)rows * AH * AD);
    float* qi = act.take<float>((size_t)rows * IH * ID);
    int* cnt = act.take<int>(rows);
    half* o16 = act.take<half>((size_t)rows * AH * AD);
    q8_linear<half>(q, y, rows, a.q, qg, dq);
    q8_linear<half>(q, y, rows, a.k, k, dq);
    q8_linear<half>(q, y, rows, a.v, v, dq);
    ops::gemv_f16<half>(q, y, rows, D, a.idx.ptr<half>(), (IH + 1) * ID, ik);
    prof_point("attn: projections");
    if (c.i8)
        attn_prep_i8(q, qg, k, v, ik, a.qn.ptr<float>(), a.kn.ptr<float>(), a.iqn.ptr<float>(), rows, p0, qn16,
                     c.k.ptr<int8_t>(), c.ks.ptr<half>(), c.v.ptr<int8_t>(), c.vs.ptr<half>(), qi, c.rk.ptr<half>(), pdev);
    else
        q4ops::attn_prep(q, qg, k, v, ik, a.qn.ptr<float>(), a.kn.ptr<float>(), a.iqn.ptr<float>(), rows, p0, qn16,
                         c.k.ptr<half>(), c.v.ptr<half>(), qi, c.rk.ptr<half>(), pdev);
    if (pdev) attn_pool_rows(q, c.rk.ptr<half>(), a.ikn.ptr<float>(), c.pk.ptr<half>(), pdev, rows);
    else q4ops::attn_pool(q, c.rk.ptr<half>(), a.ikn.ptr<float>(), c.pk.ptr<half>(), p0 / BLK, (p0 + rows) / BLK);
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
        if (c.i8)
            attn_core_i8(q, qn16 + (size_t)r0 * AH * AD, c.k.ptr<int8_t>(), c.ks.ptr<half>(), c.v.ptr<int8_t>(),
                         c.vs.ptr<half>(), list, cnt + r0, qg + (size_t)r0 * AH * 2 * AD, nr, p0 + r0, part,
                         o16 + (size_t)r0 * AH * AD, pdev);
        else
            q4ops::attn_core(q, qn16 + (size_t)r0 * AH * AD, c.k.ptr<half>(), c.v.ptr<half>(), list, cnt + r0,
                             qg + (size_t)r0 * AH * 2 * AD, nr, p0 + r0, part, o16 + (size_t)r0 * AH * AD, pdev);
        r0 += nr;
    }
    prof_point("attn: select + softmax");
    q8_linear<float>(q, o16, rows, a.o, out, dq);
    prof_point("attn: o_proj");
}

void E0Model::moe_forward_(const E0Block& b, const half* y, int rows, float* out) {
    auto& q = Dev::get().q;
    half* dq = deq.ptr<half>();
    const int64_t* ptrs = ptrs_dev.ptr<int64_t>() + (size_t)b.layer * E;
    const int* lab = labels_dev.ptr<int>() + (size_t)b.layer * E;
    ops::gemv_f16<half>(q, y, rows, D, b.router.ptr<half>(), E + 1, rl.ptr<half>());
    route512(q, rl.ptr<half>(), E + 1, rows, sel.ptr<int>(), wts.ptr<float>());
    if (counting)
        q4ops::count_use(q, sel.ptr<int>(), rows, use_dev.ptr<int>() + (size_t)b.layer * E,
                         in_vram_dev.ptr<uint8_t>() + (size_t)b.layer * E, stats_dev.ptr<unsigned long long>());
    prof_point("MoE: router");
    uint32_t* gm = reinterpret_cast<uint32_t*>(elist.ptr<int>() + 4 * MV_MAX * TOPK);
    if (rows == 1 && group_skip) {
        group_mask(q, sel.ptr<int>(), lab, TOPK, gm);
        q8_matvec_grouped<half>(q, y, b.dcat, xl32.ptr<float>(), gm, 1);
    } else if (rows <= MV_MAX) {
        q8_matvec<half, float>(q, y, rows, b.dcat, xl32.ptr<float>());
    }
    if (rows <= MV_MAX) {
        prof_point("MoE: D");
        int* lead = nullptr;                        // several rows: each distinct expert read once
        if (rows > 1) {
            lead = elist.ptr<int>();
            pick_leaders(q, sel.ptr<int>(), rows * TOPK, lead);
        }
        nv_gate_up_t(q, xl32.ptr<float>(), sel.ptr<int>(), lead, lab, ptrs, rows, h32.ptr<float>());
        nv_down_t(q, h32.ptr<float>(), sel.ptr<int>(), lead, ptrs, rows, yl32.ptr<float>());
        prof_point("MoE: routed experts");
        latent_combine(q, yl32.ptr<float>(), sel.ptr<int>(), wts.ptr<float>(), lab, rows, z32.ptr<float>());
        if (rows == 1 && group_skip) q8_matvec_grouped<float>(q, z32.ptr<float>(), b.ucat, out, gm, 2);
        else q8_matvec<float, float>(q, z32.ptr<float>(), rows, b.ucat, out);
        prof_point("MoE: U");
    } else {
        // the picks of each expert, on the host (one sync a layer)
        const int NP = rows * TOPK;
        sel_h.resize(NP);
        wts_h.resize(NP);
        q.memcpy(sel_h.data(), sel.p, (size_t)NP * 4);
        q.memcpy(wts_h.data(), wts.p, (size_t)NP * 4).wait();
        std::vector<int> cnt(E + 1, 0);
        for (int p = 0; p < NP; p++) cnt[sel_h[p] + 1]++;
        for (int e = 0; e < E; e++) cnt[e + 1] += cnt[e];
        // per pick (sorted by expert): row, xl column, weight, s_g, s_u
        dp_lst.assign(5 * (size_t)NP, 0);
        std::vector<int>& lst = dp_lst;
        std::vector<int> fill(cnt.begin(), cnt.end() - 1);
        int* L_row = lst.data();
        int* L_col = L_row + NP;
        float* L_w = reinterpret_cast<float*>(L_col + NP);
        float* L_s = L_w + NP;
        for (int p = 0; p < NP; p++) {
            const int e = sel_h[p], key = b.layer * E + e, at = fill[e]++;
            L_row[at] = p / TOPK;
            L_col[at] = labels_h[key] * ELL;
            L_w[at] = wts_h[p];
            L_s[2 * at] = scales_h[(size_t)key * 3];
            L_s[2 * at + 1] = scales_h[(size_t)key * 3 + 1];
        }
        int* d_row = elist.ptr<int>();
        int* d_col = d_row + NP;
        float* d_w = reinterpret_cast<float*>(d_col + NP);
        float* d_s = d_w + NP;
        // (the DPAS path uploads its own, sorted lists into the same buffer: two pageable-memory copies into one
        // place made its results vary from run to run)
        if (!moe_dpas || moe_grouped) q.memcpy(d_row, lst.data(), lst.size() * 4);
        // x = D y for all eight groups at once, U as fp16 for the per-expert products
        q8_linear<half>(q, y, rows, b.dcat, xl16.ptr<half>(), ddeq.ptr<half>());
        q8_dequant(q, b.ucat, udeq.ptr<half>());
        q.memset(out, 0, (size_t)rows * D * 4);
        prof_point("MoE: D, U");
        namespace blas = oneapi::mkl::blas::row_major;
        using oneapi::mkl::transpose;
        int ci = 0;                                 // the layer's running index of non-empty experts
        if (moe_dpas && !moe_grouped) {
            moe_prompt_dpas(b, rows, cnt, out);
            prof_point("MoE: routed experts");
        } else if (!moe_grouped) {
            // one expert at a time; RAM records are copied into a ring of VRAM slots on the copy engine ahead of use,
            // each copy waiting for its slot's previous expert to be dequantized
            auto& cq = Dev::get().copy;
            constexpr int NS = 8;
            std::vector<sycl::event> slot_free(NS);
            int rj = 0;
            for (int e = 0; e < E; e++) {
                const int n = cnt[e + 1] - cnt[e];
                if (!n) continue;
                const int key = b.layer * E + e;
                const char* rec = reinterpret_cast<const char*>(ptrs_h[key]);
                if (!in_vram[key] && stage_on) {
                    const int k = rj++ % NS;
                    char* slot = stage.ptr<char>() + (size_t)k * rec_bytes;
                    sycl::event ce = cq.submit([&](sycl::handler& h) {
                        h.depends_on(slot_free[k]);
                        h.memcpy(slot, rec, NV_REC);
                    });
                    q.ext_oneapi_submit_barrier({ce});
                    nv_dequant(q, slot, bgu.ptr<half>(), bdn.ptr<half>());
                    slot_free[k] = q.ext_oneapi_submit_barrier();
                } else {
                    nv_dequant(q, rec, bgu.ptr<half>(), bdn.ptr<half>());
                }
                gather_picks(q, xl16.ptr<half>(), d_row + cnt[e], d_col + cnt[e], n, Xb.ptr<half>());
                blas::gemm(q, transpose::nontrans, transpose::trans, n, 2 * I, ELL, 1.0f, Xb.ptr<half>(), ELL,
                           bgu.ptr<half>(), ELL, 0.0f, GUb.ptr<float>(), 2 * I);
                nv_act_rows(q, GUb.ptr<float>(), d_s + 2 * (size_t)cnt[e], n, h16b.ptr<half>());
                blas::gemm(q, transpose::nontrans, transpose::trans, n, ELL, I, scales_h[(size_t)key * 3 + 2],
                           h16b.ptr<half>(), I, bdn.ptr<half>(), I, 0.0f, Yb.ptr<float>(), ELL);
                e0ops::f32_to_f16(q, Yb.ptr<float>(), Yh.ptr<half>(), (size_t)n * ELL);
                blas::gemm(q, transpose::nontrans, transpose::trans, n, D, ELL, 1.0f, Yh.ptr<half>(), ELL,
                           udeq.ptr<half>() + (size_t)labels_h[key] * ELL, GE, 0.0f, Ob.ptr<float>(), D);
                scatter_rows(q, Ob.ptr<float>(), d_row + cnt[e], d_w + cnt[e], n, D, out);
            }
        }
        for (int e = moe_grouped ? 0 : E; e < E;) {
            // a batch: up to moe_batch non-empty experts and pick_cap picks
            std::vector<int> be;
            int np = 0;
            for (; e < E && (int)be.size() < moe_batch; e++) {
                int n = cnt[e + 1] - cnt[e];
                if (!n) continue;
                if (np + n > pick_cap) break;
                be.push_back(e);
                np += n;
            }
            if (be.empty()) continue;
            const int nb = (int)be.size(), p0 = cnt[be[0]];
            gather_picks(q, xl16.ptr<half>(), d_row + p0, d_col + p0, np, Xb.ptr<half>());
            prof_point("MoE b: gather");
            for (int j = 0; j < nb; j++) recs_sh[ci + j] = ptrs_h[b.layer * E + be[j]];
            nv_dequant_batch(q, recs_sh + ci, nb, bgu.ptr<half>(), bdn.ptr<half>());
            prof_point("MoE b: dequant");
            // three grouped GEMMs, one group an expert: [n, ELL] x gate|up^T, [n, 640] x down^T * s_d, [n, ELL] x U_g^T
            for (int t = 0; t < 3; t++) {
                const int o = t * E + ci;
                for (int j = 0; j < nb; j++) {
                    const int e2 = be[j], n = cnt[e2 + 1] - cnt[e2], off = cnt[e2] - p0, key = b.layer * E + e2;
                    gm_m[o + j] = n;
                    if (t == 0) {
                        gm_n[o + j] = 2 * I; gm_k[o + j] = ELL; gm_lda[o + j] = ELL; gm_ldb[o + j] = ELL; gm_ldc[o + j] = 2 * I;
                        gm_alpha[o + j] = 1.0f;
                        pa_sh[o + j] = Xb.ptr<half>() + (size_t)off * ELL;
                        pb_sh[o + j] = bgu.ptr<half>() + (size_t)j * 2 * I * ELL;
                        pc_sh[o + j] = GUb.ptr<float>() + (size_t)off * 2 * I;
                    } else if (t == 1) {
                        gm_n[o + j] = ELL; gm_k[o + j] = I; gm_lda[o + j] = I; gm_ldb[o + j] = I; gm_ldc[o + j] = ELL;
                        gm_alpha[o + j] = scales_h[(size_t)key * 3 + 2];
                        pa_sh[o + j] = h16b.ptr<half>() + (size_t)off * I;
                        pb_sh[o + j] = bdn.ptr<half>() + (size_t)j * ELL * I;
                        pc_sh[o + j] = Yb.ptr<float>() + (size_t)off * ELL;
                    } else {
                        gm_n[o + j] = D; gm_k[o + j] = ELL; gm_lda[o + j] = ELL; gm_ldb[o + j] = GE; gm_ldc[o + j] = D;
                        gm_alpha[o + j] = 1.0f;
                        pa_sh[o + j] = Yh.ptr<half>() + (size_t)off * ELL;
                        pb_sh[o + j] = udeq.ptr<half>() + (size_t)labels_h[key] * ELL;
                        pc_sh[o + j] = Ob.ptr<float>() + (size_t)off * D;
                    }
                }
            }
            auto grouped = [&](int t) {
                const int o = t * E + ci;
                blas::gemm_batch(q, gm_ta.data() + o, gm_tb.data() + o, gm_m.data() + o, gm_n.data() + o,
                                 gm_k.data() + o, gm_alpha.data() + o, pa_sh + o, gm_lda.data() + o, pb_sh + o,
                                 gm_ldb.data() + o, gm_beta.data() + o, pc_sh + o, gm_ldc.data() + o, (int64_t)nb,
                                 gm_gs.data() + o);
            };
            grouped(0);
            prof_point("MoE b: gemm gate|up");
            nv_act_rows(q, GUb.ptr<float>(), d_s + 2 * (size_t)p0, np, h16b.ptr<half>());
            prof_point("MoE b: act");
            grouped(1);
            prof_point("MoE b: gemm down");
            e0ops::f32_to_f16(q, Yb.ptr<float>(), Yh.ptr<half>(), (size_t)np * ELL);
            prof_point("MoE b: to fp16");
            grouped(2);
            prof_point("MoE b: gemm U");
            scatter_rows(q, Ob.ptr<float>(), d_row + p0, d_w + p0, np, D, out);
            prof_point("MoE b: scatter");
            ci += nb;
        }
        prof_point("MoE: routed experts");
    }
    q8_linear<half>(q, y, rows, b.sg, sg16.ptr<half>(), dq);
    q8_linear<half>(q, y, rows, b.su, su16.ptr<half>(), dq);
    q4ops::swiglu(q, sg16.ptr<half>(), su16.ptr<half>(), sa16.ptr<half>(), (size_t)rows * I);
    q8_linear<float>(q, sa16.ptr<half>(), rows, b.sd, sh32.ptr<float>(), dq);
    add_shared(q, out, sh32.ptr<float>(), rl.ptr<half>(), E + 1, rows);
    prof_point("MoE: shared expert + combine");
}

void E0Model::ple_host(const int* tokens, int rows) {
    std::vector<int> hist(rows + 2);
    hist[0] = ids_ctx[0];
    hist[1] = ids_ctx[1];
    for (int t = 0; t < rows; t++) hist[t + 2] = tokens[t];
    e_ev.wait();
    ngram->lookup(hist.data(), rows, e_host);
    ids_ctx[0] = hist[rows];
    ids_ctx[1] = hist[rows + 1];
}

void E0Model::ple_dev(int rows, float* Xd) {
    auto& q = Dev::get().q;
    sycl::event ev = q.memcpy(emb16.p, e_host, (size_t)rows * D * 2);
    if (!recording) e_ev = ev;
    ops::gemv_f16<float>(q, emb16.ptr<half>(), rows, D, ple_wkv.ptr<half>(), SD + D, kv32.ptr<float>());
    q4ops::ple_gate(q, kv32.ptr<float>(), Xd, ple_nk.ptr<float>(), ple_nq.ptr<float>(), ple_nc.ptr<float>(),
                    gv32.ptr<float>(), cn16.ptr<half>(), rows);
    if (saving) q.memcpy(sv_cn.p, cn16.p, (size_t)rows * SD * 2);
    q4ops::ple_conv(q, cn16.ptr<half>(), gv32.ptr<float>(), ple_cw.ptr<float>(), ple_state.ptr<half>(), Xd, rows);
}

// The prompt's routed experts on the XMX units (e0_dpas.cpp). The picks sorted by (D/U group, expert) and cut into
// chunks of pick_cap; per chunk: their latent inputs gathered, gate/up + SwiGLU and down by DPAS in blocks of 32 rows of
// one expert, U as one GEMM per group's run of picks, then weighted into the rows' outputs. A RAM expert with more than
// one block is copied to VRAM first (a block reads its weights once; over PCIe that would be once a block).
void E0Model::moe_prompt_dpas(const E0Block& b, int rows, const std::vector<int>& cnt, float* out) {
    auto& q = Dev::get().q;
    namespace blas = oneapi::mkl::blas::row_major;
    using oneapi::mkl::transpose;
    constexpr int MBK = 64;                         // rows a DPAS block (e0_dpas.cpp's MB)
    const int NP = rows * TOPK, base = b.layer * E;
    std::vector<int> ord(E);
    for (int e = 0; e < E; e++) ord[e] = e;
    std::stable_sort(ord.begin(), ord.end(), [&](int x, int y) { return labels_h[base + x] < labels_h[base + y]; });
    // picks in that order: row, xl column, weight (from the per-expert lists built by the caller, sorted by expert)
    const int* E_row = dp_lst.data();
    const int* E_col = E_row + NP;
    const float* E_w = reinterpret_cast<const float*>(E_col + NP);
    dp_sorted.resize(3 * (size_t)NP);
    int* S_row = dp_sorted.data();
    int* S_col = S_row + NP;
    float* S_w = reinterpret_cast<float*>(S_col + NP);
    std::vector<int> spos(E);
    for (int i = 0, at = 0; i < E; i++) {
        int e = ord[i], n = cnt[e + 1] - cnt[e];
        spos[e] = at;
        for (int k = 0; k < n; k++) {
            S_row[at + k] = E_row[cnt[e] + k];
            S_col[at + k] = E_col[cnt[e] + k];
            S_w[at + k] = E_w[cnt[e] + k];
        }
        at += n;
    }
    // chunks, blocks and group runs
    struct Chunk { int p0, np, b0, nb; std::vector<std::pair<int, int>> stage; std::vector<std::array<int, 3>> runs;
                   int c0 = 0, nr = 0; };   // its rows' combine lists: dp_crow[c0 .. c0 + nr)
    std::vector<Chunk> chunks;
    dp_brec.clear(); dp_br0.clear(); dp_bn.clear();
    const int nslots = (int)stage.size(0);
    {
        Chunk c{0, 0, 0, 0, {}, {}};
        for (int i = 0; i < E; i++) {
            const int e = ord[i], key = base + e;
            int left = cnt[e + 1] - cnt[e], off = spos[e];
            while (left > 0) {
                int take = std::min(left, pick_cap - c.np);
                const bool heavy = !in_vram[key] && take > MBK;
                if (take <= 0 || (heavy && (int)c.stage.size() >= nslots)) {   // the chunk is full: start the next
                    chunks.push_back(std::move(c));
                    c = Chunk{off, 0, (int)dp_brec.size(), 0, {}, {}};
                    continue;
                }
                int64_t rec = ptrs_h[key];
                if (heavy) {
                    int slot = (int)c.stage.size();
                    c.stage.push_back({slot, key});
                    rec = (int64_t)(stage.ptr<char>() + (size_t)slot * rec_bytes);
                }
                if (c.np == 0) { c.p0 = off; c.b0 = (int)dp_brec.size(); }
                for (int r = 0; r < take; r += MBK) {
                    dp_brec.push_back(rec);
                    dp_br0.push_back(c.np + r);
                    dp_bn.push_back(std::min(MBK, take - r));
                    c.nb++;
                }
                const int g = labels_h[key];
                if (!c.runs.empty() && c.runs.back()[0] == g) c.runs.back()[2] += take;
                else c.runs.push_back({g, c.np, take});
                c.np += take;
                off += take;
                left -= take;
            }
        }
        if (c.np) chunks.push_back(std::move(c));
    }
    const int NB = (int)dp_brec.size();
    GLM_CHECK(NB <= blk_cap, "moe: %d blocks (room for %d)", NB, blk_cap);
    // per chunk: each touched row once, with its picks (chunk-relative), for a race-free and ordered combine
    dp_crow.clear(); dp_coff.clear(); dp_cpick.clear();
    {
        std::vector<int> first(rows, -1), nxt(NP, -1), last(rows, -1);
        for (Chunk& c : chunks) {
            c.c0 = (int)dp_crow.size();
            std::vector<int> touched;
            for (int k = 0; k < c.np; k++) {
                int r = S_row[c.p0 + k];
                if (first[r] < 0) { first[r] = k; touched.push_back(r); }
                else nxt[last[r]] = k;
                last[r] = k;
            }
            for (int r : touched) {
                dp_crow.push_back(r);
                dp_coff.push_back((int)dp_cpick.size());
                for (int k = first[r]; k >= 0; k = nxt[k]) dp_cpick.push_back(k);
                first[r] = -1;
            }
            for (int k = 0; k < c.np; k++) nxt[k] = -1;
            c.nr = (int)touched.size();
            dp_coff.push_back((int)dp_cpick.size());          // the chunk's closing offset
        }
    }
    int* d_row = elist.ptr<int>();
    int* d_col = d_row + NP;
    float* d_w = reinterpret_cast<float*>(d_col + NP);
    q.memcpy(d_row, S_row, (size_t)NP * 4);
    q.memcpy(d_col, S_col, (size_t)NP * 4);
    q.memcpy(d_w, S_w, (size_t)NP * 4);
    q.memcpy(blk_rec.p, dp_brec.data(), (size_t)NB * 8);
    q.memcpy(blk_r0.p, dp_br0.data(), (size_t)NB * 4);
    q.memcpy(blk_n.p, dp_bn.data(), (size_t)NB * 4);
    int* c_row = blk_r0.ptr<int>() + blk_cap;       // (the combine lists after the block tables)
    int* c_off = c_row + NP;
    int* c_pick = c_off + NP + (int)chunks.size();
    q.memcpy(c_row, dp_crow.data(), dp_crow.size() * 4);
    q.memcpy(c_off, dp_coff.data(), dp_coff.size() * 4);
    q.memcpy(c_pick, dp_cpick.data(), dp_cpick.size() * 4);
    q.wait();                                       // the uploads (pageable memory) done before anything reads them
    for (const Chunk& c : chunks) {
        for (auto [slot, key] : c.stage)
            q.memcpy(stage.ptr<char>() + (size_t)slot * rec_bytes, reinterpret_cast<const void*>(ptrs_h[key]), NV_REC);
        prof_point("MoE p: stage");
        gather_picks(q, xl16.ptr<half>(), d_row + c.p0, d_col + c.p0, c.np, Xb.ptr<half>());
        prof_point("MoE p: gather");
        nv_gate_up_dpas(q, Xb.ptr<half>(), c.np, blk_rec.ptr<int64_t>() + c.b0, blk_r0.ptr<int>() + c.b0, blk_n.ptr<int>() + c.b0,
                        c.nb, h16b.ptr<half>());
        prof_point("MoE p: DPAS gate/up");
        nv_down_dpas(q, h16b.ptr<half>(), c.np, blk_rec.ptr<int64_t>() + c.b0, blk_r0.ptr<int>() + c.b0, blk_n.ptr<int>() + c.b0,
                     c.nb, Yh.ptr<half>());
        prof_point("MoE p: DPAS down");
        for (auto& r : c.runs)
            blas::gemm(q, transpose::nontrans, transpose::trans, r[2], D, ELL, 1.0f, Yh.ptr<half>() + (size_t)r[1] * ELL, ELL,
                       udeq.ptr<half>() + (size_t)r[0] * ELL, GE, 0.0f, Ob.ptr<float>() + (size_t)r[1] * D, D);
        prof_point("MoE p: U GEMMs");
        // c_off for this chunk starts at c.c0 + (its index): each chunk's offsets carry one closing entry
        const int ci = (int)(&c - chunks.data());
        combine_rows(q, Ob.ptr<float>(), c_row + c.c0, c_off + c.c0 + ci, c_pick, d_w + c.p0, c.nr, D, out);
        prof_point("MoE p: scatter");
    }
    q.wait();                                       // (the host lists above are uploaded from pageable memory)
}

void E0Model::pre_moe(E0Block& b, float* Xd, int rows, int p0, const int* tokens, const int* pdev, half* y,
                      float* injp) {
    prof_point("(between layers)");
    if (b.layer == 1) {
        if (tokens) ple_host(tokens, rows);
        ple_dev(rows, Xd);
        prof_point("PLE");
    }
    site_mix(b.site[0], Xd, rows, y16.ptr<half>(), inj.ptr<float>());
    grab(y16.p, D, true);
    prof_point("site mix");
    if (b.att) attn_forward(*b.att, kv[b.layer], y16.ptr<half>(), rows, p0, sub.ptr<float>(), pdev);
    else gdn_forward(*b.gdn, gst[b.layer], y16.ptr<half>(), rows, sub.ptr<float>());
    grab(sub.p, D, false);
    q4ops::site_apply(Dev::get().q, Xd, inj.ptr<float>(), sub.ptr<float>(), rows);
    site_mix(b.site[1], Xd, rows, y, injp);
    grab(y, D, true);
    prof_point("site apply + mix");
}

void E0Model::block_forward(E0Block& b, float* Xd, int rows, int p0, const int* tokens, const int* pdev) {
    pre_moe(b, Xd, rows, p0, tokens, pdev, y16.ptr<half>(), inj.ptr<float>());
    moe_forward_(b, y16.ptr<half>(), rows, sub.ptr<float>());
    grab(sub.p, D, false);
    q4ops::site_apply(Dev::get().q, Xd, inj.ptr<float>(), sub.ptr<float>(), rows);
    grab(Xd, SD, false);
    prof_point("site apply");
}

// ------------------------------------------------------------------------------------------------ forward / prefill

void E0Model::host_rows(const int* tokens, int n) {
    dec_ev.wait();
    x_ev.wait();
    embed_host(tokens, n);
    ple_host(tokens, n);
    *pos_h = pos;
}

void E0Model::forward_dev(int Q, const int* pdev) {
    auto& q = Dev::get().q;
    counting = true;
    float* Xd = X.ptr<float>();
    float* e = sub.ptr<float>();
    q.memcpy(e, x_host, (size_t)Q * D * 4);
    q.memcpy(pos_dev.p, pos_h, sizeof(int));
    q4ops::expand(q, e, Xd, Q);
    for (auto& b : blocks) block_forward(b, Xd, Q, pos, nullptr, pdev);
    site_mix(mixer, Xd, Q, fin16.ptr<half>(), nullptr);
    q8_matvec<half, half>(q, fin16.ptr<half>(), Q, lm_head, logits.ptr<half>());
    prof_point("mixer + lm_head");
    counting = false;
}

const half* E0Model::forward(const int* tokens, int Q) {
    GLM_CHECK(Q >= 1 && Q <= qmax, "forward: %d rows (at most %d)", Q, qmax);
    GLM_CHECK(pos + Q <= max_tokens, "context full (%d tokens)", max_tokens);
    auto& q = Dev::get().q;
    if (rebal_every > 0 && ++since_rebal >= rebal_every) {
        since_rebal = 0;
        rebalance();
    }
    vwin_Q = 0;
    if (Q > 1 && mtp) {                              // a verify window: keep the recurrent state from before it
        dec_ev.wait();
        for (int li = 0; li < L; li++) {
            if (blocks[li].attn) continue;
            q.memcpy(rec_bak[li].p, gst[li].rec.p, rec_bak[li].bytes());
            q.memcpy(conv_bak[li].p, gst[li].conv.p, conv_bak[li].bytes());
        }
        q.memcpy(ple_bak.p, ple_state.p, ple_bak.bytes());
        ids_bak[0] = ids_ctx[0];
        ids_bak[1] = ids_ctx[1];
        vwin_p0 = pos;
        vwin_Q = Q;
        vwin_toks.assign(tokens, tokens + Q);
        saving = true;
    }
    fwd_toks.assign(tokens, tokens + Q);
    fwd_p0 = pos;
    host_rows(tokens, Q);
    // one recorded graph per row count (1: decode; 2..qmax: verify windows, recorded with their saves)
    if ((Q == 1 || (mtp && !getenv("E0_NO_VERIFY_GRAPHS"))) && graphs_on && !dump) {
        namespace ex = sycl::ext::oneapi::experimental;
        auto& gp = dec_graphs[Q];
        if (!gp) {
            ex::command_graph<ex::graph_state::modifiable> gr{q.get_context(), q.get_device()};
            recording = true;
            gr.begin_recording(q);
            forward_dev(Q, pos_dev.ptr<int>());
            gr.end_recording(q);
            recording = false;
            gp = std::make_unique<ex::command_graph<ex::graph_state::executable>>(gr.finalize());
        }
        dec_ev = q.ext_oneapi_graph(*gp);
    } else {
        forward_dev(Q, nullptr);
    }
    saving = false;
    pos += Q;
    return logits.ptr<half>();
}

void E0Model::prefill(const int* tokens, int n, int ckpt_at) {
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
    vwin_Q = 0;
    if (mtp) {
        if (pend_pos >= 0 && pend_pos == pos - 1 && mtp_done == pend_pos) {
            mtp_run(mtp_pend.ptr<float>(), tokens, 1, pend_pos, false);
            mtp_done = pos;
        }
        pend_pos = -1;
        mtp_done = pos;                             // (cells below that were not caught up stay stale)
    }
    const int C = prefill_chunk;
    // pieces of moe_rows rows, layer by layer: attention / GDN in chunks of C, then one MoE over the piece
    const int SC = moe_rows;
    Tensor Xall = Tensor::empty({std::min(n, SC), S, D}, DType::F32);
    for (int s0 = 0; s0 < n; s0 += SC) {
        int rows = std::min(SC, n - s0);
        float* Xp = Xall.ptr<float>();
        for (int c0 = 0; c0 < rows; c0 += C) embed_rows(tokens + s0 + c0, std::min(C, rows - c0), Xp + (size_t)c0 * SD);
        for (auto& b : blocks) {
            for (int c0 = 0; c0 < rows; c0 += C) {
                int nr = std::min(C, rows - c0);
                pre_moe(b, Xp + (size_t)c0 * SD, nr, pos + c0, tokens + s0 + c0, nullptr, yall.ptr<half>() + (size_t)c0 * D,
                        injall.ptr<float>() + (size_t)c0 * S);
            }
            moe_forward_(b, yall.ptr<half>(), rows, outall.ptr<float>());
            q4ops::site_apply(q, Xp, injall.ptr<float>(), outall.ptr<float>(), rows);
            prof_point("site apply");
            if (rows >= 256) read_marks().mark((double)rows / blocks.size());
        }
        if (rows < 256) read_marks().mark(rows);
        if (mtp) {
            mtp_prefill(Xp, tokens + s0, rows, pos, s0 + rows < n);
            prof_point("MTP cells");
        }
        pos += rows;
    }
    q.wait();
    if (cut == n) save_checkpoint(new_checkpoint(pos));
}

// ------------------------------------------------------------------------------------------------ state

void E0Model::reset() {
    pos = 0;
    mtp_done = 0;
    pend_pos = -1;
    vwin_Q = 0;
    for (auto& c : ckpts_) c.pos = -1;
    for (int li = 0; li < L; li++)
        if (!blocks[li].attn) gst[li].zero();
    auto& q = Dev::get().q;
    q.memset(ple_state.p, 0, ple_state.bytes());
    ids_ctx[0] = ids_ctx[1] = EOS;
    q.wait();
}

E0Model::Checkpoint& E0Model::new_checkpoint(int at) {
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

void E0Model::save_checkpoint(Checkpoint& c) {
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

bool E0Model::restore(int upto) {
    if (restore_window(upto)) return true;
    vwin_Q = 0;
    mtp_done = std::min(mtp_done, upto);
    if (pend_pos >= upto) pend_pos = -1;
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

// ------------------------------------------------------------------------------------------------ MTP

namespace glm {

void E0Model::load_mtp(const std::string& dir) {
    auto& q = Dev::get().q;
    const std::string md = dir + "/mtp";
    std::string bin = read_file(md + "/dense.bin");
    struct Ent { std::string kind; int rows, cols; size_t off, bytes; };
    std::map<std::string, Ent> ents;
    {
        std::istringstream in(read_file(md + "/dense.txt"));
        std::string name, kind;
        int rows, cols;
        size_t off, bytes;
        while (in >> name >> kind >> rows >> cols >> off >> bytes) {
            GLM_CHECK(off + bytes <= bin.size(), "mtp/dense.txt: %s past dense.bin", name.c_str());
            ents[name] = {kind, rows, cols, off, bytes};
        }
    }
    auto at = [&](const std::string& n, const char* kind) -> const Ent& {
        auto it = ents.find(n);
        GLM_CHECK(it != ents.end() && it->second.kind == kind, "mtp: %s (%s) missing", n.c_str(), kind);
        return it->second;
    };
    auto q8 = [&](const std::string& n) {
        const Ent& e = at(n, "q8_0");
        return q8_from_blocks(reinterpret_cast<const uint8_t*>(bin.data() + e.off), e.rows, e.cols);
    };
    auto f32 = [&](const std::string& n) {
        const Ent& e = at(n, "f32");
        std::vector<float> v((size_t)e.rows * e.cols);
        memcpy(v.data(), bin.data() + e.off, v.size() * 4);
        return dev_f32(v);
    };
    auto f16 = [&](const std::vector<std::string>& ns) {    // bf16 matrices stacked by rows, as fp16
        int rows = 0, cols = at(ns[0], "bf16").cols;
        for (auto& n : ns) { GLM_CHECK(at(n, "bf16").cols == cols, "mtp: %s columns", n.c_str()); rows += at(n, "bf16").rows; }
        std::vector<half> h((size_t)rows * cols);
        size_t o = 0;
        for (auto& n : ns) {
            const Ent& e = at(n, "bf16");
            const uint16_t* src = reinterpret_cast<const uint16_t*>(bin.data() + e.off);
            for (size_t i = 0; i < (size_t)e.rows * e.cols; i++) h[o++] = to_f16(bf16f(src[i]));
        }
        Tensor t = Tensor::empty({rows, cols}, DType::F16);
        upload(t.p, h.data(), h.size() * 2);
        return t;
    };
    auto site = [&](const std::string& p, bool inject) {
        Q4Site s;
        s.w1p = f32(p + ".hc_norm.weight");
        std::vector<std::string> d = {p + ".input_mix_weight_down.weight"};
        if (inject) d.push_back(p + ".block_inject_weight.weight");
        s.wdi = f16(d);
        s.nd = (int)s.wdi.size(0);
        s.wu = f16({p + ".input_mix_weight_up.weight"});
        return s;
    };
    auto m = std::make_unique<E0Mtp>();
    m->fc_emb = q8("fc_embedding.weight");
    m->fc_hid = q8("fc_hidden.weight");
    m->w_pe = f32("pre_fc_norm_embedding.weight");
    m->w_ph = f32("pre_fc_norm_hidden.weight");
    m->site_attn = site("attn_hyper_connection", true);
    m->site_mlp = site("mlp_hyper_connection", true);
    m->mixer = site("hyper_connection_mixer", false);
    m->q = q8("self_attn.q_proj.weight");
    m->k = q8("self_attn.k_proj.weight");
    m->v = q8("self_attn.v_proj.weight");
    m->o = q8("self_attn.o_proj.weight");
    m->qn = f32("self_attn.q_norm.weight");
    m->kn = f32("self_attn.k_norm.weight");
    m->router = f16({"mlp.gate.weight", "mlp.shared_expert_gate.weight"});
    m->sg = q8("mlp.shared_expert.gate_proj.weight");
    m->su = q8("mlp.shared_expert.up_proj.weight");
    m->sd = q8("mlp.shared_expert.down_proj.weight");
    GLM_CHECK(m->q.N == AH * 2 * AD && m->k.N == AKV * AD && m->o.K == AH * AD && m->router.size(0) == E + 1 &&
                  m->fc_hid.N == D && m->sd.N == D, "mtp: shapes");
    {
        std::string ex = read_file(md + "/experts.bin");
        GLM_CHECK(ex.size() == (size_t)E * Q2_REC, "mtp/experts.bin: %zu bytes", ex.size());
        m->experts = malloc_checked(ex.size(), q, 2);
        GLM_CHECK(m->experts, "mtp experts: VRAM");
        Dev::get().vram_used += ex.size();
        upload(m->experts, ex.data(), ex.size());
        std::vector<int64_t> pt(E);
        for (int e = 0; e < E; e++) pt[e] = (int64_t)(m->experts + (size_t)e * Q2_REC);
        m->ptrs = Tensor::empty({E}, DType::I64);
        upload(m->ptrs.p, pt.data(), E * 8);
    }
    m->Kc = Tensor::empty({max_tokens, AKV, AD}, DType::F16);
    m->Vc = Tensor::empty({max_tokens, AKV, AD}, DType::F16);
    mtp = std::move(m);

    const int C = prefill_chunk;
    mtp_pend = Tensor::empty({1, S, D}, DType::F32);
    mtp_in = Tensor::empty({qmax + 1, S, D}, DType::F32);
    mtp_R = Tensor::empty({C, S, D}, DType::F32);
    mtp_emb = Tensor::empty({C, D}, DType::F32);
    mtp_en = Tensor::empty({C, D}, DType::F16);
    mtp_e2 = Tensor::empty({C, D}, DType::F32);
    mtp_hn = Tensor::empty({C, SD}, DType::F16);
    mtp_h2 = Tensor::empty({C, SD}, DType::F32);
    mtp_k = Tensor::empty({C, AKV * AD}, DType::F16);
    mtp_v = Tensor::empty({C, AKV * AD}, DType::F16);
    mtp_qg = Tensor::empty({1, AH * 2 * AD}, DType::F16);
    mtp_qn = Tensor::empty({1, AH * AD}, DType::F16);
    mtp_o16 = Tensor::empty({1, AH * AD}, DType::F16);
    mtp_o = Tensor::empty({1, D}, DType::F32);
    mtp_part = Tensor::empty({(int64_t)mtp_attn_floats(max_tokens)}, DType::F32);
    mtp_h = Tensor::empty({TOPK, I}, DType::F32);
    mtp_moe = Tensor::empty({1, D}, DType::F32);
    mtp_logits = Tensor::empty({1, V}, DType::F16);
    mtp_best = Tensor::empty({2}, DType::I32);
    mtp_emb_h = sycl::malloc_host<float>((size_t)C * D, q);
    GLM_CHECK(mtp_emb_h, "mtp: pinned embedding rows");
    // the verify windows' backups and replay inputs
    rec_bak.resize(L);
    conv_bak.resize(L);
    sv_qkv.resize(L);
    sv_cv.resize(L);
    sv_beta.resize(L);
    sv_g.resize(L);
    for (int li = 0; li < L; li++) {
        if (blocks[li].attn) continue;
        rec_bak[li] = Tensor::empty({GV, GDK, GDV}, DType::F32);
        conv_bak[li] = Tensor::empty({GQKV, GCONV - 1}, DType::BF16);
        sv_qkv[li] = Tensor::empty({qmax, GQKV}, DType::F32);
        sv_cv[li] = Tensor::empty({qmax, GQKV}, DType::BF16);
        sv_beta[li] = Tensor::empty({qmax, GV}, DType::F32);
        sv_g[li] = Tensor::empty({qmax, GV}, DType::F32);
    }
    ple_bak = Tensor::empty({SD, 9}, DType::F16);
    sv_cn = Tensor::empty({qmax, SD}, DType::F16);
    rp_scratch = Tensor::empty({(int64_t)qmax * (GQKV * 2 + GV * GDV * 2 + SD * 8)}, DType::U8);
    q.wait();
    fprintf(stderr, "mtp: draft layer loaded (%.2f GiB VRAM so far)\n", Dev::get().vram_used / 1073741824.0);
}

void E0Model::mtp_run(const float* Xin, const int* toks, int T, int p0, bool full) {
    auto& q = Dev::get().q;
    const E0Mtp& m = *mtp;
    half* dq = deq.ptr<half>();
    GLM_CHECK(T >= 1 && T <= prefill_chunk, "mtp: %d rows", T);
    // the tokens' embeddings (host, Q8_0 rows) -> mtp_emb
    static sycl::event up_ev;
    up_ev.wait();
    const size_t rb = (size_t)(D / 32) * 34;
    for (int t = 0; t < T; t++) {
        const uint8_t* row = embed + (size_t)toks[t] * rb;
        float* o = mtp_emb_h + (size_t)t * D;
        for (int b = 0; b < D / 32; b++) {
            uint16_t db;
            memcpy(&db, row + b * 34, 2);
            float sc = f16f(db);
            const int8_t* qs = reinterpret_cast<const int8_t*>(row + b * 34 + 2);
            for (int j = 0; j < 32; j++) o[b * 32 + j] = sc * qs[j];
        }
    }
    up_ev = q.memcpy(mtp_emb.p, mtp_emb_h, (size_t)T * D * 4);
    rms_rows(q, mtp_emb.ptr<float>(), m.w_pe.ptr<float>(), T, D, mtp_en.ptr<half>());
    q8_linear<float>(q, mtp_en.ptr<half>(), T, m.fc_emb, mtp_e2.ptr<float>(), dq);
    rms_rows(q, Xin, m.w_ph.ptr<float>(), T, SD, mtp_hn.ptr<half>());           // one RMS over the 4 x 2560 values
    q8_linear<float>(q, mtp_hn.ptr<half>(), T * S, m.fc_hid, mtp_h2.ptr<float>(), dq);
    float* R = mtp_R.ptr<float>();
    add_streams(q, mtp_h2.ptr<float>(), mtp_e2.ptr<float>(), T, R);
    site_mix(m.site_attn, R, T, y16.ptr<half>(), inj.ptr<float>());
    q8_linear<half>(q, y16.ptr<half>(), T, m.k, mtp_k.ptr<half>(), dq);
    q8_linear<half>(q, y16.ptr<half>(), T, m.v, mtp_v.ptr<half>(), dq);
    mtp_prep(q, nullptr, mtp_k.ptr<half>(), mtp_v.ptr<half>(), m.qn.ptr<float>(), m.kn.ptr<float>(), T, p0, nullptr,
             m.Kc.ptr<half>(), m.Vc.ptr<half>());
    if (!full) return;
    const int r = T - 1, p = p0 + r;
    float* Rl = R + (size_t)r * SD;
    q8_matvec<half, half>(q, y16.ptr<half>() + (size_t)r * D, 1, m.q, mtp_qg.ptr<half>());
    mtp_prep(q, mtp_qg.ptr<half>(), mtp_k.ptr<half>() + (size_t)r * AKV * AD, mtp_v.ptr<half>() + (size_t)r * AKV * AD,
             m.qn.ptr<float>(), m.kn.ptr<float>(), 1, p, mtp_qn.ptr<half>(), m.Kc.ptr<half>(), m.Vc.ptr<half>());
    mtp_attn(q, mtp_qn.ptr<half>(), m.Kc.ptr<half>(), m.Vc.ptr<half>(), mtp_qg.ptr<half>(), p, mtp_part.ptr<float>(),
             mtp_o16.ptr<half>());
    q8_matvec<half, float>(q, mtp_o16.ptr<half>(), 1, m.o, mtp_o.ptr<float>());
    q4ops::site_apply(q, Rl, inj.ptr<float>() + (size_t)r * S, mtp_o.ptr<float>(), 1);
    site_mix(m.site_mlp, Rl, 1, y16.ptr<half>(), inj.ptr<float>());
    ops::gemv_f16<half>(q, y16.ptr<half>(), 1, D, m.router.ptr<half>(), E + 1, rl.ptr<half>());
    route512(q, rl.ptr<half>(), E + 1, 1, sel.ptr<int>(), wts.ptr<float>());
    q2_gate_up(q, y16.ptr<half>(), sel.ptr<int>(), m.ptrs.ptr<int64_t>(), 1, mtp_h.ptr<float>());
    q2_down_sum(q, mtp_h.ptr<float>(), sel.ptr<int>(), wts.ptr<float>(), m.ptrs.ptr<int64_t>(), 1, mtp_moe.ptr<float>());
    q8_matvec<half, half>(q, y16.ptr<half>(), 1, m.sg, sg16.ptr<half>());
    q8_matvec<half, half>(q, y16.ptr<half>(), 1, m.su, su16.ptr<half>());
    q4ops::swiglu(q, sg16.ptr<half>(), su16.ptr<half>(), sa16.ptr<half>(), I);
    q8_matvec<half, float>(q, sa16.ptr<half>(), 1, m.sd, sh32.ptr<float>());
    add_shared(q, mtp_moe.ptr<float>(), sh32.ptr<float>(), rl.ptr<half>(), E + 1, 1);
    q4ops::site_apply(q, Rl, inj.ptr<float>(), mtp_moe.ptr<float>(), 1);
    q.memcpy(mtp_in.p, Rl, (size_t)SD * 4);       // the next chain step's stack
    site_mix(m.mixer, Rl, 1, fin16.ptr<half>(), nullptr);
    q8_matvec<half, half>(q, fin16.ptr<half>(), 1, lm_head, mtp_logits.ptr<half>());
    argmax_prob(q, mtp_logits.ptr<half>(), V, mtp_best.ptr<int>());
}

void E0Model::mtp_prefill(const float* Xp, const int* toks, int rows, int p0, bool last_known) {
    auto& q = Dev::get().q;
    const int known = last_known ? rows : rows - 1;
    for (int c0 = 0; c0 < known; c0 += prefill_chunk) {
        int nr = std::min(prefill_chunk, known - c0);
        mtp_run(Xp + (size_t)c0 * SD, toks + c0 + 1, nr, p0 + c0, false);
    }
    mtp_done = p0 + known;
    if (!last_known) {
        q.memcpy(mtp_pend.p, Xp + (size_t)(rows - 1) * SD, (size_t)SD * 4);
        pend_pos = p0 + rows - 1;
    } else {
        pend_pos = -1;
    }
}

int E0Model::draft(int tok, int hidden_row, int n, int* out) {
    if (!mtp || n <= 0) return 0;
    auto& q = Dev::get().q;
    dec_ev.wait();
    const int seed = fwd_p0 + hidden_row;
    // catch-up: the cells from the pending one (if it is right before this forward) through the seed row
    std::vector<int> toks;
    int c0 = fwd_p0, T = 0;
    if (pend_pos >= 0 && pend_pos == fwd_p0 - 1 && mtp_done == pend_pos) {
        q.memcpy(mtp_in.p, mtp_pend.p, (size_t)SD * 4);
        toks.push_back(fwd_toks[0]);
        c0 = pend_pos;
        T = 1;
    }
    for (int i = 0; i <= hidden_row; i++, T++) {
        q.memcpy(mtp_in.ptr<float>() + (size_t)T * SD, X.ptr<float>() + (size_t)i * SD, (size_t)SD * 4);
        toks.push_back(i < hidden_row ? fwd_toks[i + 1] : tok);
    }
    pend_pos = -1;
    int k = 0;
    for (int cell = c0;;) {
        mtp_run(mtp_in.ptr<float>(), toks.data(), T, cell, true);
        int best[2];
        download(best, mtp_best.p, 8);
        float prob = sycl::bit_cast<float>(best[1]);
        if (k == 0) mtp_done = seed + 1;
        if (prob < spec_min_p) break;
        out[k++] = best[0];
        if (k >= n) break;
        cell += T;
        toks.assign(1, best[0]);
        T = 1;                                      // mtp_in holds the last stack (mtp_run copied it)
    }
    return k;
}

bool E0Model::restore_window(int upto) {
    if (!vwin_Q || upto <= vwin_p0 || upto >= vwin_p0 + vwin_Q) return false;
    auto& q = Dev::get().q;
    dec_ev.wait();
    const int a = upto - vwin_p0;
    char* scr = rp_scratch.ptr<char>();
    bf16* cv_out = reinterpret_cast<bf16*>(scr);
    bf16* core = reinterpret_cast<bf16*>(scr + (size_t)qmax * GQKV * 2);
    float* xs = reinterpret_cast<float*>(scr + (size_t)qmax * (GQKV * 2 + GV * GDV * 2));
    for (int li = 0; li < L; li++) {
        if (blocks[li].attn) continue;
        const E0Gdn& g = *blocks[li].gdn;
        q.memcpy(gst[li].rec.p, rec_bak[li].p, rec_bak[li].bytes());
        q.memcpy(gst[li].conv.p, conv_bak[li].p, conv_bak[li].bytes());
        q4ops::gdn_conv(q, sv_qkv[li].ptr<float>(), gst[li].conv.ptr<bf16>(), g.conv_w.ptr<bf16>(), cv_out, a);
        q4ops::gdn_delta(q, sv_cv[li].ptr<bf16>(), sv_beta[li].ptr<float>(), sv_g[li].ptr<float>(),
                         gst[li].rec.ptr<float>(), core, a);
    }
    q.memcpy(ple_state.p, ple_bak.p, ple_bak.bytes());
    q4ops::ple_conv(q, sv_cn.ptr<half>(), xs, ple_cw.ptr<float>(), ple_state.ptr<half>(), xs + (size_t)qmax * SD, a);
    std::vector<int> hist = {ids_bak[0], ids_bak[1]};
    for (int i = 0; i < a; i++) hist.push_back(vwin_toks[i]);
    ids_ctx[0] = hist[hist.size() - 2];
    ids_ctx[1] = hist[hist.size() - 1];
    q.wait();
    pos = upto;
    vwin_Q = 0;
    mtp_done = std::min(mtp_done, upto);
    return true;
}

}  // namespace glm
