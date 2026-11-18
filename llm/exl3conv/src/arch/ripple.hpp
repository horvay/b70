// exl3conv: Ripple-65B (model_type ricursive) for calibration: config, layer loading, and the forward of one decoder layer over
// one calibration row with optional Hessian capture. The forward is the checkpoint's modeling_ricursive.py at exllamav3's rounding
// points, as Gemma 4 (nn/ops.hpp), the same as the engine's (engines/glm.cpp/src/rp.h). For a layer, n = rms(x) * input_layernorm:
//
//     q, k = q_proj(n), k_proj(n), rms norm per head (weights q_norm, k_norm), RoPE on the sliding layers; 80 query heads over 8 KV
//     v = v_proj(lerp(E, n, sigmoid(value_res_lambda))), E the row's scaled embedding, constant across layers; V not normed
//     x += post_attn(o_proj(attention(q, k, v) * sigmoid(g_proj(n)))), scores q . k / sqrt(128), window 1024 on the sliding layers
//     x += post_ff(down(silu(min(gate(m), 20)) * clamp(up(m), +-20))), m = pre_ff(x)
//
// Five Hessian groups: q/k/g (n), v (its lerped input), o (the gated attention), gate/up (m), down (the MLP's activation).
//
// Memory on the 32 GB card (hidden 5120, MLP 40960), rows x 2048 tokens: down's Hessian is 6.25 GiB. E is never stored per row:
// the BF16 table stays on the card (0.96 GiB) and each row's E is computed where it is read. Attention runs one KV group's 10 query
// heads at a time (0.23 GiB of scores instead of 1.9 GiB). The Hessians are accumulated in their lower block triangle, which is
// ~55% of the work for down, the largest cost of a capture.
#pragma once

#include "arch/gemma4.hpp"
#include "nn/ripple_ops.hpp"

#include <cmath>
#include <string>
#include <vector>

namespace rp {

using json = nlohmann::json;
using exl3::DBuf;
using half = sycl::half;
using g4::Lin;
using g4::Upload;
using g4::gemm_lin;
using g4::load_bf16_vec;
using g4::load_linear;
using g4::load_exl3_linear;
namespace blas = oneapi::mkl::blas::row_major;
using oneapi::mkl::transpose;

struct Rp
{
    int64_t H, L, NH, NKV, D, I, window, vocab;
    float eps, theta, clamp;
    float mult = 1.0f;       // bf16(embed_scale)
    std::vector<bool> full;
    int bos, eos;
    std::string prefix = "model.";

    explicit Rp(const json& c)
    {
        if (c.value("model_type", "") != "ricursive") throw std::runtime_error("not a Ripple model (model_type " + c.value("model_type", "?") + ")");
        if (!c.value("use_gated_attention", false) || !c.value("use_valres", false)) throw std::runtime_error("Ripple: gated attention and value residuals expected");
        H = c.at("hidden_size"); L = c.at("num_hidden_layers"); NH = c.at("num_attention_heads"); NKV = c.at("num_key_value_heads");
        D = c.at("head_dim"); I = c.at("intermediate_size"); window = c.at("sliding_window"); vocab = c.at("vocab_size");
        eps = c.at("rms_norm_eps"); theta = c.at("rope_theta"); clamp = c.at("ffn_clamp_threshold");
        if (NH * D % 128 || NKV * D % 128 || NH % NKV) throw std::runtime_error("Ripple: unexpected head shape");
        // folded checkpoints bake the scale into the embedding and have none here
        if (c.contains("embed_scale") && !c["embed_scale"].is_null()) mult = st::bf16_to_float(nn::f2bf16(c["embed_scale"].get<float>()));
        for (const auto& s : c.at("layer_types")) full.push_back(s == "full_attention");
        if ((int64_t) full.size() != L) throw std::runtime_error("Ripple: layer_types does not match num_hidden_layers");
        bos = c.value("bos_token_id", 100257);
        eos = c.contains("eos_token_id") && c["eos_token_id"].is_array() ? (int) c["eos_token_id"][0] : c.value("eos_token_id", 100265);
    }
    std::string lp(int64_t l) const { return prefix + "layers." + std::to_string(l) + "."; }
    std::vector<float> inv_freq() const
    {
        std::vector<float> v(D / 2);
        for (int64_t j = 0; j < D / 2; ++j) v[j] = (float) std::pow((double) theta, -2.0 * j / (double) D);
        return v;
    }
};

struct Layer
{
    bool full = false;
    Lin q, k, v, gt, o, g, u, d;      // gt: g_proj (the attention's gate); g, u, d: the MLP's gate, up, down
    DBuf<uint16_t> in_ln, post_attn_ln, pre_ff_ln, post_ff_ln, q_norm, k_norm;
    float blend = 0.0f;               // sigmoid(value_res_lambda), rounded to bf16
};

inline void load_small(sycl::queue& q, const st::Model& m, const Rp& g, int64_t li, Upload& up, Layer& L)
{
    const auto p = g.lp(li), a = p + "self_attn.";
    L.full = g.full[li];
    L.in_ln = load_bf16_vec(q, m, p + "input_layernorm.weight", up);
    L.post_attn_ln = load_bf16_vec(q, m, p + "post_attention_layernorm.weight", up);
    L.pre_ff_ln = load_bf16_vec(q, m, p + "pre_feedforward_layernorm.weight", up);
    L.post_ff_ln = load_bf16_vec(q, m, p + "post_feedforward_layernorm.weight", up);
    L.q_norm = load_bf16_vec(q, m, a + "q_norm.weight", up);
    L.k_norm = load_bf16_vec(q, m, a + "k_norm.weight", up);
    const auto& lam = m.get(a + "value_res_lambda");
    if (lam.dtype != st::DType::BF16) throw std::runtime_error(a + "value_res_lambda: expected BF16");
    const float lv = st::bf16_to_float(*(const uint16_t*) lam.data);
    L.blend = st::bf16_to_float(nn::f2bf16(1.0f / (1.0f + std::exp(-lv))));
}

inline Layer load_layer(sycl::queue& q, const st::Model& m, const Rp& g, int64_t li, Upload& up, DBuf<uint16_t>& scratch)
{
    Layer L;
    const auto p = g.lp(li), a = p + "self_attn.";
    if (!m.has(a + "q_proj.weight")) throw std::runtime_error("the checkpoint has no layer " + std::to_string(li) + " (a part of it? pass --layers)");
    L.q = load_linear(q, m, a + "q_proj.weight", up, scratch);
    L.k = load_linear(q, m, a + "k_proj.weight", up, scratch);
    L.v = load_linear(q, m, a + "v_proj.weight", up, scratch);
    L.gt = load_linear(q, m, a + "g_proj.weight", up, scratch);
    L.o = load_linear(q, m, a + "o_proj.weight", up, scratch);
    L.g = load_linear(q, m, p + "mlp.gate_proj.weight", up, scratch);
    L.u = load_linear(q, m, p + "mlp.up_proj.weight", up, scratch);
    L.d = load_linear(q, m, p + "mlp.down_proj.weight", up, scratch);
    load_small(q, m, g, li, up, L);
    return L;
}

inline Layer load_layer_exl3(sycl::queue& q, const st::Model& m, const Rp& g, int64_t li, Upload& up, const DBuf<int>& d_perm, DBuf<float>& tmp)
{
    Layer L;
    const auto p = g.lp(li), a = p + "self_attn.";
    L.q = load_exl3_linear(q, m, a + "q_proj", up, d_perm, tmp);
    L.k = load_exl3_linear(q, m, a + "k_proj", up, d_perm, tmp);
    L.v = load_exl3_linear(q, m, a + "v_proj", up, d_perm, tmp);
    L.gt = load_exl3_linear(q, m, a + "g_proj", up, d_perm, tmp);
    L.o = load_exl3_linear(q, m, a + "o_proj", up, d_perm, tmp);
    L.g = load_exl3_linear(q, m, p + "mlp.gate_proj", up, d_perm, tmp);
    L.u = load_exl3_linear(q, m, p + "mlp.up_proj", up, d_perm, tmp);
    L.d = load_exl3_linear(q, m, p + "mlp.down_proj", up, d_perm, tmp);
    load_small(q, m, g, li, up, L);
    return L;
}

// lm_head as a forward weight: BF16 (a source checkpoint), fp16 or EXL3 (a quant).
inline Lin load_head(sycl::queue& q, const st::Model& m, Upload& up, const DBuf<int>& d_perm, DBuf<float>& tmp, DBuf<uint16_t>& scratch)
{
    if (m.has("lm_head.trellis")) return load_exl3_linear(q, m, "lm_head", up, d_perm, tmp);
    const auto& t = m.get("lm_head.weight");
    if (t.dtype == st::DType::BF16) return load_linear(q, m, "lm_head.weight", up, scratch);
    if (t.dtype != st::DType::F16 || t.shape.size() != 2) throw std::runtime_error("lm_head.weight: expected BF16 or F16");
    Lin l;
    l.out = t.shape[0]; l.in = t.shape[1]; l.hf = true;
    l.w = DBuf<half>(q, (size_t) t.numel());
    up(l.w.get(), t.data, t.nbytes);
    return l;
}

// The five Hessian groups of a layer.
struct Capture
{
    DBuf<float> qkg, v, o, gu, d;
    int64_t n_qkg = 0, n_v = 0, n_o = 0, n_gu = 0, n_d = 0;
    DBuf<double> sens_u, sens_g;   // --out-scales auto: summed squared slopes of the MLP's gated activation (up, gate)
    int64_t n_sens = 0;
};

struct Scratch
{
    DBuf<half> h, vin, qh, kh, vh, P, oh, ah, h2, m, tmp;
    DBuf<float> qf, kf, vf, tf, S, Of, af, gf, uf, df;
    DBuf<int32_t> ok;
    Scratch(sycl::queue& q, const Rp& g, int64_t T)
    {
        const int64_t QD = g.NH * g.D, KD = g.NKV * g.D, grp = g.NH / g.NKV;
        h = DBuf<half>(q, T * g.H); vin = DBuf<half>(q, T * g.H); qh = DBuf<half>(q, T * QD); kh = DBuf<half>(q, T * KD); vh = DBuf<half>(q, T * KD);
        P = DBuf<half>(q, (size_t) grp * T * T); oh = DBuf<half>(q, T * QD); ah = DBuf<half>(q, T * g.H);
        h2 = DBuf<half>(q, T * g.H); m = DBuf<half>(q, T * g.I); tmp = DBuf<half>(q, T * std::max(g.I, QD));
        qf = DBuf<float>(q, T * QD); kf = DBuf<float>(q, T * KD); vf = DBuf<float>(q, T * KD); tf = DBuf<float>(q, T * QD);
        S = DBuf<float>(q, (size_t) grp * T * T); Of = DBuf<float>(q, T * QD); af = DBuf<float>(q, T * g.H);
        gf = DBuf<float>(q, T * g.I); uf = DBuf<float>(q, T * g.I); df = DBuf<float>(q, T * g.H);
        ok = DBuf<int32_t>(q, T);
    }
    // what the constructor above allocates, in bytes
    static uint64_t bytes(const Rp& g, int64_t T)
    {
        const uint64_t QD = g.NH * g.D, KD = g.NKV * g.D, grp = g.NH / g.NKV, U = T;
        return 2 * (U * g.H * 4 + U * QD * 2 + 2 * U * KD + grp * U * U + U * g.I + U * std::max<uint64_t>(g.I, QD)) +
               4 * (U * QD * 3 + 2 * U * KD + grp * U * U + 2 * U * g.H + 2 * U * g.I) + 4 * U;
    }
};

// How many blocks accumulate() splits a Hessian of k into: the lower block triangle is (b + 1) / (2 b) of the work.
inline int hessian_blocks(int64_t k) { return k >= 40960 ? 8 : 4; }

// H (lower block triangle) += X^T X over the finite rows of X (T x k fp16); returns the number of rows used.
inline int64_t accumulate(sycl::queue& q, const half* X, int64_t T, int64_t k, float* Hm, Scratch& s)
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
    const int b = hessian_blocks(k);
    const int64_t bs = k / b;
    for (int i = 0; i < b; ++i)
        for (int j = 0; j <= i; ++j)
            blas::gemm(q, transpose::trans, transpose::nontrans, bs, bs, T, 1.0f, src + i * bs, k, src + j * bs, k, 1.0f, Hm + i * bs * k + j * bs, k);
    return good;
}

// The Hessian whole, for the quantizer (accumulate() fills the lower block triangle).
inline void symmetrize(sycl::queue& q, float* Hm, int64_t k) { nn::mirror_lower(q, Hm, k, k / hessian_blocks(k)).wait(); }

// One decoder layer over one calibration row, x (T x H fp32) updated in place; E, ids: the BF16 embedding on the card and the row's
// tokens (v_proj's input). cap: accumulate Hessians.
inline void forward_layer(sycl::queue& q, const Rp& g, const Layer& L, float* x, const uint16_t* E, const int32_t* ids, int64_t T, Scratch& s,
                          const float* inv_freq, Capture* cap)
{
    const int64_t H = g.H, NH = g.NH, NKV = g.NKV, D = g.D, QD = NH * D, KD = NKV * D, grp = NH / NKV;
    const float eps = g.eps, alpha = 1.0f / std::sqrt((float) D);
    nn::rmsnorm_h(q, x, L.in_ln.get(), T, H, eps, s.h.get());
    nn::lerp_embed(q, E, ids, H, g.mult, L.blend, s.h.get(), T, s.vin.get());
    if (cap)
    {
        cap->n_qkg += accumulate(q, s.h.get(), T, H, cap->qkg.get(), s);
        cap->n_v += accumulate(q, s.vin.get(), T, H, cap->v.get(), s);
    }
    gemm_lin(q, s.h.get(), T, L.q, s.qf.get());
    gemm_lin(q, s.h.get(), T, L.k, s.kf.get());
    gemm_lin(q, s.h.get(), T, L.gt, s.tf.get());
    gemm_lin(q, s.vin.get(), T, L.v, s.vf.get());
    const float* inv = L.full ? nullptr : inv_freq;
    nn::head_norm_rope(q, s.qf.get(), T, NH, D, L.q_norm.get(), inv, eps, s.qh.get());
    nn::head_norm_rope(q, s.kf.get(), T, NKV, D, L.k_norm.get(), inv, eps, s.kh.get());
    nn::to_half(q, s.vf.get(), T * KD, s.vh.get());
    for (int64_t kv = 0; kv < NKV; ++kv)
    {
        for (int64_t j = 0; j < grp; ++j)
            blas::gemm(q, transpose::nontrans, transpose::trans, T, T, D, alpha, s.qh.get() + (kv * grp + j) * D, QD, s.kh.get() + kv * D, KD,
                       0.0f, s.S.get() + j * T * T, T);
        nn::softmax_mask(q, s.S.get(), grp, T, L.full ? 0 : g.window, s.P.get());
        for (int64_t j = 0; j < grp; ++j)
            blas::gemm(q, transpose::nontrans, transpose::nontrans, T, D, T, 1.0f, s.P.get() + j * T * T, T, s.vh.get() + kv * D, KD,
                       0.0f, s.Of.get() + (kv * grp + j) * D, QD);
    }
    nn::gate_mul(q, s.Of.get(), s.tf.get(), T * QD, s.oh.get());
    if (cap) cap->n_o += accumulate(q, s.oh.get(), T, QD, cap->o.get(), s);
    gemm_lin(q, s.oh.get(), T, L.o, s.af.get());
    nn::to_half_clamp(q, s.af.get(), T * H, s.ah.get());
    nn::rmsnorm_add(q, s.ah.get(), L.post_attn_ln.get(), T, H, eps, x);
    nn::rmsnorm_h(q, x, L.pre_ff_ln.get(), T, H, eps, s.h2.get());
    if (cap) cap->n_gu += accumulate(q, s.h2.get(), T, H, cap->gu.get(), s);
    gemm_lin(q, s.h2.get(), T, L.g, s.gf.get());
    gemm_lin(q, s.h2.get(), T, L.u, s.uf.get());
    if (cap && cap->sens_u.get())
    {
        nn::silu_slopes_acc(q, s.gf.get(), s.uf.get(), T, g.I, g.clamp, cap->sens_u.get(), cap->sens_g.get());
        cap->n_sens += T;
    }
    nn::silu_mul(q, s.gf.get(), s.uf.get(), T * g.I, g.clamp, s.m.get());
    if (cap) cap->n_d += accumulate(q, s.m.get(), T, g.I, cap->d.get(), s);
    gemm_lin(q, s.m.get(), T, L.d, s.df.get());
    nn::rmsnorm_add(q, s.df.get(), L.post_ff_ln.get(), T, H, eps, x).wait();
}

} // namespace rp
