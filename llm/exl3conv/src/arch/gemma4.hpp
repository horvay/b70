// exl3conv: Gemma 4 dense text decoder on the B70 for calibration (notes/gemma4-forward.md): config, layer loading
// (BF16 -> fp16), and the forward of one decoder layer over one calibration row with optional Hessian capture.
#pragma once

#include "exl3/dequant.hpp"
#include "exl3/quantize.hpp"
#include "io/safetensors.hpp"
#include "nn/ops.hpp"
#include "rt/rt.hpp"

#include <oneapi/mkl/blas.hpp>

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace g4 {

using json = nlohmann::json;
using exl3::DBuf;
using half = sycl::half;
namespace blas = oneapi::mkl::blas::row_major;
using oneapi::mkl::transpose;

struct G4
{
    int64_t H, L, NH, NKV, NKVG, D, DG, I, window, vocab;
    double eps, theta_s, theta_f, partial;
    bool k_eq_v;
    std::vector<bool> full;
    int bos, eos;
    std::string prefix = "model.language_model.";

    explicit G4(const json& cfg)
    {
        const auto& t = cfg.at("text_config");
        if (cfg.value("model_type", "") != "gemma4") throw std::runtime_error("not a gemma4 model (model_type " + cfg.value("model_type", "?") + ")");
        if (t.value("enable_moe_block", false)) throw std::runtime_error("Gemma 4 MoE is not supported yet");
        if (t.value("hidden_size_per_layer_input", 0) != 0) throw std::runtime_error("Gemma 4 per-layer embeddings (PLE) are not supported yet");
        if (t.value("num_kv_shared_layers", 0) != 0) throw std::runtime_error("Gemma 4 KV-shared layers are not supported yet");
        H = t.at("hidden_size"); L = t.at("num_hidden_layers"); NH = t.at("num_attention_heads");
        NKV = t.at("num_key_value_heads"); NKVG = t.value("num_global_key_value_heads", NKV);
        D = t.at("head_dim"); DG = t.value("global_head_dim", D); I = t.at("intermediate_size");
        window = t.at("sliding_window"); vocab = t.at("vocab_size"); eps = t.at("rms_norm_eps");
        k_eq_v = t.value("attention_k_eq_v", false);
        for (const auto& s : t.at("layer_types")) full.push_back(s == "full_attention");
        const auto& rp = t.at("rope_parameters");
        theta_s = rp.at("sliding_attention").at("rope_theta");
        theta_f = rp.at("full_attention").at("rope_theta");
        partial = rp.at("full_attention").value("partial_rotary_factor", 1.0);
        if (rp.at("full_attention").value("rope_type", "default") != "proportional" || rp.at("sliding_attention").value("rope_type", "default") != "default")
            throw std::runtime_error("unexpected Gemma 4 rope types");
        bos = t.value("bos_token_id", 2);
        eos = cfg.contains("eos_token_id") && cfg["eos_token_id"].is_array() ? (int) cfg["eos_token_id"][0] : t.value("eos_token_id", 1);
        if (t.value("attn_logit_softcapping", 0.0) != 0.0) throw std::runtime_error("attention softcapping is not supported");
    }
    std::string lp(int64_t l) const { return prefix + "layers." + std::to_string(l) + "."; }
    // RoPE inverse frequencies (D/2 entries): sliding = default, full = proportional (notes/gemma4-forward.md §2.3)
    std::vector<float> inv_freq(bool f) const
    {
        const int64_t d = f ? DG : D;
        std::vector<float> v(d / 2, 0.0f);
        if (!f) for (int64_t j = 0; j < d / 2; ++j) v[j] = (float) std::pow(theta_s, -2.0 * j / (double) d);
        else
        {
            const int64_t angles = (int64_t) (partial * (double) d / 2.0);
            for (int64_t j = 0; j < angles; ++j) v[j] = (float) std::pow(theta_f, -2.0 * j / (double) d);
        }
        return v;
    }
};

// Host -> device in 32 MiB pieces through a pinned staging buffer (large single copies have faulted the xe copy
// engine under host memory pressure, notes/toolchain-and-reuse.md §6.1).
struct Upload
{
    sycl::queue& q;
    static constexpr size_t CH = 32ull << 20;
    uint8_t* stage;
    explicit Upload(sycl::queue& q_) : q(q_) { stage = (uint8_t*) sycl::malloc_host(CH, q); if (!stage) throw std::runtime_error("pinned staging alloc failed"); }
    ~Upload() { sycl::free(stage, q); }
    void operator()(void* dst, const void* src, size_t n)
    {
        for (size_t o = 0; o < n; o += CH)
        {
            const size_t m = std::min(CH, n - o);
            std::memcpy(stage, (const uint8_t*) src + o, m);
            q.memcpy((uint8_t*) dst + o, stage, m).wait();
        }
    }
    void download(void* dst, const void* src, size_t n)
    {
        for (size_t o = 0; o < n; o += CH)
        {
            const size_t m = std::min(CH, n - o);
            q.memcpy(stage, (const uint8_t*) src + o, m).wait();
            std::memcpy((uint8_t*) dst + o, stage, m);
        }
    }
};

// A linear's weight for the forward: fp16, either HF [out, in] (y = x W^T) or quantizer/inference (in, out) (y = x W).
struct Lin { DBuf<half> w; int64_t in = 0, out = 0; bool hf = true; };

static void gemm_lin(sycl::queue& q, const half* x, int64_t T, const Lin& l, float* y)
{
    blas::gemm(q, transpose::nontrans, l.hf ? transpose::trans : transpose::nontrans, T, l.out, l.in, 1.0f, x, l.in,
               l.w.get(), l.hf ? l.in : l.out, 0.0f, y, l.out);
}

struct Layer
{
    bool full;
    int64_t D, NKV;
    Lin q, k, v, o, g, u, d;
    DBuf<uint16_t> in_ln, post_attn_ln, pre_ff_ln, post_ff_ln, q_norm, k_norm;
    float scalar = 1.0f;
};

static DBuf<uint16_t> load_bf16_vec(sycl::queue& q, const st::Model& m, const std::string& name, Upload& up)
{
    const auto& t = m.get(name);
    if (t.dtype != st::DType::BF16) throw std::runtime_error(name + ": expected BF16");
    DBuf<uint16_t> b(q, (size_t) t.numel());
    up(b.get(), t.data, t.nbytes);
    return b;
}

static Lin load_linear(sycl::queue& q, const st::Model& m, const std::string& name, Upload& up, DBuf<uint16_t>& scratch)
{
    const auto& t = m.get(name);
    if (t.dtype != st::DType::BF16 || t.shape.size() != 2) throw std::runtime_error(name + ": expected a 2-D BF16 weight");
    Lin l;
    l.out = t.shape[0]; l.in = t.shape[1]; l.hf = true;
    if (l.in % 128 || l.out % 128) throw std::runtime_error(name + ": dimensions must be multiples of 128 (padding is not implemented)");
    const size_t n = (size_t) t.numel();
    if (scratch.n < n) scratch = DBuf<uint16_t>(q, n);
    up(scratch.get(), t.data, t.nbytes);
    l.w = DBuf<half>(q, n);
    nn::bf16_to_half(q, scratch.get(), (int64_t) n, l.w.get()).wait();
    return l;
}

static Layer load_layer(sycl::queue& q, const st::Model& m, const G4& g, int64_t li, Upload& up, DBuf<uint16_t>& scratch)
{
    Layer L;
    const auto p = g.lp(li);
    L.full = g.full[li];
    L.D = L.full ? g.DG : g.D;
    L.NKV = L.full && g.k_eq_v ? g.NKVG : g.NKV;
    L.q = load_linear(q, m, p + "self_attn.q_proj.weight", up, scratch);
    L.k = load_linear(q, m, p + "self_attn.k_proj.weight", up, scratch);
    if (!(L.full && g.k_eq_v)) L.v = load_linear(q, m, p + "self_attn.v_proj.weight", up, scratch);
    L.o = load_linear(q, m, p + "self_attn.o_proj.weight", up, scratch);
    L.g = load_linear(q, m, p + "mlp.gate_proj.weight", up, scratch);
    L.u = load_linear(q, m, p + "mlp.up_proj.weight", up, scratch);
    L.d = load_linear(q, m, p + "mlp.down_proj.weight", up, scratch);
    L.in_ln = load_bf16_vec(q, m, p + "input_layernorm.weight", up);
    L.post_attn_ln = load_bf16_vec(q, m, p + "post_attention_layernorm.weight", up);
    L.pre_ff_ln = load_bf16_vec(q, m, p + "pre_feedforward_layernorm.weight", up);
    L.post_ff_ln = load_bf16_vec(q, m, p + "post_feedforward_layernorm.weight", up);
    L.q_norm = load_bf16_vec(q, m, p + "self_attn.q_norm.weight", up);
    L.k_norm = load_bf16_vec(q, m, p + "self_attn.k_norm.weight", up);
    const auto& ls = m.get(p + "layer_scalar");
    L.scalar = st::bf16_to_float(*(const uint16_t*) ls.data);
    return L;
}

// The four Hessian groups of a decoder layer.
struct Capture
{
    DBuf<float> qkv, o, gu, d;
    int64_t n_qkv = 0, n_o = 0, n_gu = 0, n_d = 0;
    DBuf<double> sens_u, sens_g;   // --out-scales auto: summed squared slopes of the MLP's gated activation (up, gate)
    int64_t n_sens = 0;
    int64_t dropped = 0;
};

struct Scratch
{
    DBuf<half> h, qh, kh, vh, P, oh, ah, h2, m, tmp;
    DBuf<float> qf, kf, vf, S, Of, af, gf, uf, df;
    DBuf<int32_t> ok;
    Scratch(sycl::queue& q, const G4& g, int64_t T)
    {
        const int64_t QD = g.NH * std::max(g.D, g.DG), KD = std::max(g.NKV * g.D, g.NKVG * g.DG);
        h = DBuf<half>(q, T * g.H); qh = DBuf<half>(q, T * QD); kh = DBuf<half>(q, T * KD); vh = DBuf<half>(q, T * KD);
        P = DBuf<half>(q, (size_t) g.NH * T * T); oh = DBuf<half>(q, T * QD); ah = DBuf<half>(q, T * g.H);
        h2 = DBuf<half>(q, T * g.H); m = DBuf<half>(q, T * g.I); tmp = DBuf<half>(q, T * std::max(g.I, QD));
        qf = DBuf<float>(q, T * QD); kf = DBuf<float>(q, T * KD); vf = DBuf<float>(q, T * KD);
        S = DBuf<float>(q, (size_t) g.NH * T * T); Of = DBuf<float>(q, T * QD); af = DBuf<float>(q, T * g.H);
        gf = DBuf<float>(q, T * g.I); uf = DBuf<float>(q, T * g.I); df = DBuf<float>(q, T * g.H);
        ok = DBuf<int32_t>(q, T);
    }
};

// H += X^T X over the finite rows of X (T x k fp16); returns the number of rows used.
// (exllamav3 drops tokens with inf / NaN activations from the Hessian, linear.py; the forward itself goes on.)
static int64_t accumulate(sycl::queue& q, const half* X, int64_t T, int64_t k, float* Hm, Scratch& s)
{
    nn::finite_rows(q, const_cast<half*>(X), T, k, s.ok.get(), false);
    std::vector<int32_t> ok(T);
    s.ok.download(ok.data(), T);
    int64_t good = 0;
    for (auto v : ok) good += v;
    const half* src = X;
    if (good < T)
    {
        q.memcpy(s.tmp.get(), X, sizeof(half) * T * k);
        nn::finite_rows(q, s.tmp.get(), T, k, s.ok.get(), true);
        src = s.tmp.get();
    }
    blas::gemm(q, transpose::trans, transpose::nontrans, k, k, T, 1.0f, src, k, src, k, 1.0f, Hm, k);
    return good;
}

// One decoder layer over one calibration row, x (T x H fp32) updated in place. cap: accumulate Hessians.
static void forward_layer(sycl::queue& q, const G4& g, const Layer& L, float* x, int64_t T, Scratch& s,
                          const float* inv_freq, Capture* cap)
{
    const int64_t H = g.H, NH = g.NH, D = L.D, NKV = L.NKV, QD = NH * D, KD = NKV * D;
    const float eps = (float) g.eps;
    nn::rmsnorm_h(q, x, L.in_ln.get(), T, H, eps, s.h.get());
    if (cap) { cap->n_qkv += accumulate(q, s.h.get(), T, H, cap->qkv.get(), s); }
    gemm_lin(q, s.h.get(), T, L.q, s.qf.get());
    gemm_lin(q, s.h.get(), T, L.k, s.kf.get());
    if (!L.full || !g.k_eq_v) gemm_lin(q, s.h.get(), T, L.v, s.vf.get());
    nn::head_norm_rope(q, s.qf.get(), T, NH, D, L.q_norm.get(), inv_freq, eps, s.qh.get());
    nn::head_norm_rope(q, s.kf.get(), T, NKV, D, L.k_norm.get(), inv_freq, eps, s.kh.get());
    nn::head_norm_rope(q, (L.full && g.k_eq_v) ? s.kf.get() : s.vf.get(), T, NKV, D, nullptr, nullptr, eps, s.vh.get());
    const int64_t grp = NH / NKV;
    for (int64_t hq = 0; hq < NH; ++hq)
    {
        const int64_t kv = hq / grp;
        blas::gemm(q, transpose::nontrans, transpose::trans, T, T, D, 1.0f, s.qh.get() + hq * D, QD, s.kh.get() + kv * D, KD,
                   0.0f, s.S.get() + hq * T * T, T);
    }
    nn::softmax_mask(q, s.S.get(), NH, T, L.full ? 0 : g.window, s.P.get());
    for (int64_t hq = 0; hq < NH; ++hq)
    {
        const int64_t kv = hq / grp;
        blas::gemm(q, transpose::nontrans, transpose::nontrans, T, D, T, 1.0f, s.P.get() + hq * T * T, T, s.vh.get() + kv * D, KD,
                   0.0f, s.Of.get() + hq * D, QD);
    }
    nn::to_half(q, s.Of.get(), T * QD, s.oh.get());
    if (cap) { cap->n_o += accumulate(q, s.oh.get(), T, QD, cap->o.get(), s); }
    gemm_lin(q, s.oh.get(), T, L.o, s.af.get());
    nn::to_half(q, s.af.get(), T * H, s.ah.get());
    nn::rmsnorm_add(q, s.ah.get(), L.post_attn_ln.get(), T, H, eps, x);
    nn::rmsnorm_h(q, x, L.pre_ff_ln.get(), T, H, eps, s.h2.get());
    if (cap) { cap->n_gu += accumulate(q, s.h2.get(), T, H, cap->gu.get(), s); }
    gemm_lin(q, s.h2.get(), T, L.g, s.gf.get());
    gemm_lin(q, s.h2.get(), T, L.u, s.uf.get());
    if (cap && cap->sens_u.get()) { nn::gelu_slopes_acc(q, s.gf.get(), s.uf.get(), T, g.I, cap->sens_u.get(), cap->sens_g.get()); cap->n_sens += T; }
    nn::gelu_mul(q, s.gf.get(), s.uf.get(), T * g.I, s.m.get());
    if (cap) { cap->n_d += accumulate(q, s.m.get(), T, g.I, cap->d.get(), s); }
    gemm_lin(q, s.m.get(), T, L.d, s.df.get());
    nn::rmsnorm_add(q, s.df.get(), L.post_ff_ln.get(), T, H, eps, x);
    nn::scale_inplace(q, x, T * H, L.scalar).wait();
}

} // namespace g4

namespace g4 {

// A decoder layer read back from an EXL3 model (any exllamav3-format directory): every linear dequantized to its
// fp16 (in, out) weight on the card, norms and scalar as stored.
inline Lin load_exl3_linear(sycl::queue& q, const st::Model& m, const std::string& key, Upload& up, const DBuf<int>& d_perm, DBuf<float>& tmp)
{
    const auto& tr = m.get(key + ".trellis");
    const auto& su = m.get(key + ".suh");
    const auto& sv = m.get(key + ".svh");
    const int64_t k = tr.shape[0] * 16, n = tr.shape[1] * 16;
    if (tr.shape[2] % 16) throw std::runtime_error(key + ": fractional bitrates are not supported here");
    const int K = (int) (tr.shape[2] / 16);
    const int cb = m.has(key + ".mul1") ? exl3::CB_MUL1 : m.has(key + ".mcg") ? exl3::CB_MCG : exl3::CB_3INST;
    DBuf<uint16_t> dtr(q, tr.nbytes / 2), dsu(q, su.nbytes / 2), dsv(q, sv.nbytes / 2);
    up(dtr.get(), tr.data, tr.nbytes); up(dsu.get(), su.data, su.nbytes); up(dsv.get(), sv.data, sv.nbytes);
    if (tmp.n < (size_t) (k * n)) tmp = DBuf<float>(q, (size_t) (k * n));
    Lin l; l.in = k; l.out = n; l.hf = false;
    l.w = DBuf<half>(q, (size_t) (k * n));
    exl3::dequant_linear(q, dtr.get(), dsu.get(), dsv.get(), k, n, K, cb, d_perm.get(), tmp.get(), l.w.get());
    return l;
}

inline Layer load_layer_exl3(sycl::queue& q, const st::Model& m, const G4& g, int64_t li, Upload& up, const DBuf<int>& d_perm, DBuf<float>& tmp)
{
    Layer L;
    const auto p = g.lp(li);
    L.full = g.full[li];
    L.D = L.full ? g.DG : g.D;
    L.NKV = L.full && g.k_eq_v ? g.NKVG : g.NKV;
    L.q = load_exl3_linear(q, m, p + "self_attn.q_proj", up, d_perm, tmp);
    L.k = load_exl3_linear(q, m, p + "self_attn.k_proj", up, d_perm, tmp);
    if (!(L.full && g.k_eq_v)) L.v = load_exl3_linear(q, m, p + "self_attn.v_proj", up, d_perm, tmp);
    L.o = load_exl3_linear(q, m, p + "self_attn.o_proj", up, d_perm, tmp);
    L.g = load_exl3_linear(q, m, p + "mlp.gate_proj", up, d_perm, tmp);
    L.u = load_exl3_linear(q, m, p + "mlp.up_proj", up, d_perm, tmp);
    L.d = load_exl3_linear(q, m, p + "mlp.down_proj", up, d_perm, tmp);
    L.in_ln = load_bf16_vec(q, m, p + "input_layernorm.weight", up);
    L.post_attn_ln = load_bf16_vec(q, m, p + "post_attention_layernorm.weight", up);
    L.pre_ff_ln = load_bf16_vec(q, m, p + "pre_feedforward_layernorm.weight", up);
    L.post_ff_ln = load_bf16_vec(q, m, p + "post_feedforward_layernorm.weight", up);
    L.q_norm = load_bf16_vec(q, m, p + "self_attn.q_norm.weight", up);
    L.k_norm = load_bf16_vec(q, m, p + "self_attn.k_norm.weight", up);
    L.scalar = st::bf16_to_float(*(const uint16_t*) m.get(p + "layer_scalar").data);
    return L;
}

} // namespace g4
