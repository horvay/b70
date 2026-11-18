// exl3conv: Qwen3.5 / Qwen3.8 dense (model_type qwen3_5, text model qwen3_5_text) for calibration: config, layer loading and
// the forward of one decoder layer over one calibration row with optional Hessian capture. The forward is HF's
// modeling_qwen3_5.py at exllamav3's rounding points (architecture/qwen3_5.py: fp32 residual, fp16 into every linear), the same
// as the engine's (engines/glm.cpp/docs/qwen35-spec.md):
//
//     y = fp16(rms(x) * (1 + input_layernorm))
//     x += mixer(y)          Gated DeltaNet (layer_types "linear_attention") or gated full attention ("full_attention")
//     y = fp16(rms(x) * (1 + post_attention_layernorm))
//     x += down(fp16(silu(gate(y)) * up(y)))
//
// Gated DeltaNet: qkv = in_proj_qkv(y) through a causal depthwise conv (kernel 4) and SiLU; q, k l2-normed per head (q / sqrt(128)),
// 16 k-heads serving 48 v-heads (v-head h reads k-head h / 3); beta = sigmoid(in_proj_b(y)), g = -exp(A_log) softplus(in_proj_a(y)
// + dt_bias); the delta rule; o = fp16(norm.weight * rms(o) * silu(in_proj_z(y))) per v-head; out_proj(o).
// Attention: q_proj gives 24 x (256 query | 256 gate) interleaved; q, k normed per head with (1 + w); RoPE on the first 64 of 256
// dims, pairs (j, j + 32), theta 1e7 (text: the interleaved mRoPE has the same position on every axis, so it is plain RoPE);
// softmax(q k / 16) v over 4 KV heads (query head h reads KV head h / 6), times sigmoid(gate), o_proj.
//
// Four Hessian groups per layer, as exllamav3's qmaps: block.attn.input (in_proj_qkv, in_proj_z | q, k, v), block.attn.output
// (out_proj | o_proj), block.mlp.input (gate, up), block.mlp.down. in_proj_a and in_proj_b stay fp16.
//
// The source can hold its tensors as BF16, F16 or F32 (a GGUF read through st::Model): every loader converts.
#pragma once

#include "arch/gemma4.hpp"           // Upload, Lin, gemm_lin
#include "nn/qwen35_ops.hpp"
#include "nn/qwen4exp_ops.hpp"
#include "nn/ripple_ops.hpp"

#include <cmath>
#include <string>
#include <vector>

namespace q35 {

using json = nlohmann::json;
using exl3::DBuf;
using half = sycl::half;
using g4::Lin;
using g4::Upload;
using g4::gemm_lin;
namespace blas = oneapi::mkl::blas::row_major;
using oneapi::mkl::transpose;

struct Config
{
    int64_t D, L, NH, NKV, Dh, R, HK, HV, DK, DV, KS, I, vocab, mtp_layers;
    double eps, theta;
    int bos, eos;
    std::vector<bool> full;
    std::string prefix = "model.language_model.";

    explicit Config(const json& cfg)
    {
        const std::string mt = cfg.value("model_type", "");
        if (mt != "qwen3_5" && mt != "qwen3_5_text") throw std::runtime_error("not a qwen3_5 model (model_type " + mt + ")");
        const json& t = cfg.contains("text_config") ? cfg.at("text_config") : cfg;
        D = t.at("hidden_size"); L = t.at("num_hidden_layers"); NH = t.at("num_attention_heads"); NKV = t.at("num_key_value_heads");
        Dh = t.at("head_dim"); I = t.at("intermediate_size"); vocab = t.at("vocab_size"); eps = t.at("rms_norm_eps");
        HK = t.at("linear_num_key_heads"); HV = t.at("linear_num_value_heads"); DK = t.at("linear_key_head_dim");
        DV = t.at("linear_value_head_dim"); KS = t.value("linear_conv_kernel_dim", 4);
        mtp_layers = t.value("mtp_num_hidden_layers", 0);
        const json rp = t.contains("rope_parameters") ? t.at("rope_parameters") : json::object();
        const double prf = rp.value("partial_rotary_factor", t.value("partial_rotary_factor", 0.25));
        R = (int64_t) std::llround(prf * (double) Dh);
        theta = rp.value("rope_theta", t.value("rope_theta", 1e7));
        bos = t.value("bos_token_id", 248044);
        eos = t.contains("eos_token_id") && t["eos_token_id"].is_array() ? (int) t["eos_token_id"][0] : t.value("eos_token_id", 248044);
        if (t.contains("layer_types"))
            for (const auto& s : t.at("layer_types")) full.push_back(s == "full_attention");
        else
            for (int64_t l = 0; l < L; ++l) full.push_back((l + 1) % t.value("full_attention_interval", 4) == 0);
        if ((int64_t) full.size() != L) throw std::runtime_error("qwen3_5: layer_types does not match num_hidden_layers");
        if (!t.value("attn_output_gate", true)) throw std::runtime_error("qwen3_5: attention without the output gate is not supported");
        if (t.value("output_gate_type", "swish") != "swish") throw std::runtime_error("qwen3_5: only output_gate_type swish is supported");
        if (t.value("hidden_act", "silu") != "silu") throw std::runtime_error("qwen3_5: only SiLU MLPs are supported");
        if (DK != 128 || DV != 128 || HV % HK || NH % NKV) throw std::runtime_error("qwen3_5: unexpected head shapes");
        if (R % 2 || R > Dh) throw std::runtime_error("qwen3_5: unexpected rotary dims");
        if (cfg.value("tie_word_embeddings", false) || t.value("tie_word_embeddings", false)) throw std::runtime_error("qwen3_5: tied embeddings are not supported");
    }
    std::string lp(int64_t l) const { return prefix + "layers." + std::to_string(l) + "."; }
    int64_t C() const { return 2 * HK * DK + HV * DV; }        // in_proj_qkv's outputs (conv channels)
    std::vector<float> inv_freq() const
    {
        std::vector<float> v(R / 2);
        for (int64_t j = 0; j < R / 2; ++j) v[j] = (float) std::pow(theta, -2.0 * (double) j / (double) R);
        return v;
    }
};

// ---------------------------------------------------------------------------------------------------------------
// Reading the source, whatever its float dtype.

inline float elem_f32(const st::TensorView& t, size_t i)
{
    switch (t.dtype)
    {
        case st::DType::BF16: return st::bf16_to_float(((const uint16_t*) t.data)[i]);
        case st::DType::F16: return exl3::f16_to_float(((const uint16_t*) t.data)[i]);
        case st::DType::F32: return ((const float*) t.data)[i];
        default: throw std::runtime_error(t.name + ": expected BF16, F16 or F32");
    }
}

inline std::vector<float> host_f32(const st::Model& m, const std::string& name)
{
    const auto& t = m.get(name);
    std::vector<float> v((size_t) t.numel());
    for (size_t i = 0; i < v.size(); ++i) v[i] = elem_f32(t, i);
    return v;
}

// bf16 bits of a tensor: the bytes for a BF16 source, rounded for the others.
inline std::vector<uint16_t> host_bf16(const st::Model& m, const std::string& name)
{
    const auto& t = m.get(name);
    std::vector<uint16_t> v((size_t) t.numel());
    if (t.dtype == st::DType::BF16) std::memcpy(v.data(), t.data, t.nbytes);
    else for (size_t i = 0; i < v.size(); ++i) v[i] = nn::f2bf16(elem_f32(t, i));
    return v;
}

// fp16 bits of a tensor (exllamav3 stores the small unquantized linears as fp16).
inline std::vector<uint16_t> host_f16(const st::Model& m, const std::string& name)
{
    const auto& t = m.get(name);
    std::vector<uint16_t> v((size_t) t.numel());
    if (t.dtype == st::DType::F16) std::memcpy(v.data(), t.data, t.nbytes);
    else for (size_t i = 0; i < v.size(); ++i) v[i] = exl3::f16_from_float(elem_f32(t, i));
    return v;
}

inline DBuf<uint16_t> load_bf16(sycl::queue& q, const st::Model& m, const std::string& name, Upload& up)
{
    const auto h = host_bf16(m, name);
    DBuf<uint16_t> d(q, h.size());
    up(d.get(), h.data(), h.size() * 2);
    return d;
}

// fp32 (1 + w) of an RMSNorm weight.
inline DBuf<float> load_norm1p(sycl::queue& q, const st::Model& m, const std::string& name, Upload& up)
{
    auto h = host_f32(m, name);
    for (auto& v : h) v += 1.0f;
    DBuf<float> d(q, h.size());
    up(d.get(), h.data(), h.size() * 4);
    return d;
}

// A 2-D [out, in] weight as fp16 on the card, HF layout. BF16 goes up as it is and is converted there; F16 as it is; F32 is
// converted on the host in pieces. big: dimensions must be multiples of 128 (a linear to quantize).
inline Lin load_lin(sycl::queue& q, const st::Model& m, const std::string& name, Upload& up, DBuf<uint16_t>& scratch, bool big = true)
{
    const auto& t = m.get(name);
    if (t.shape.size() != 2) throw std::runtime_error(name + ": expected a 2-D weight");
    Lin l;
    l.out = t.shape[0]; l.in = t.shape[1]; l.hf = true;
    if (big && (l.in % 128 || l.out % 128)) throw std::runtime_error(name + ": dimensions must be multiples of 128 (padding is not implemented)");
    const size_t n = (size_t) t.numel();
    l.w = DBuf<half>(q, n);
    const size_t piece = 32ull << 20;      // elements
    if (t.dtype == st::DType::F16) up(l.w.get(), t.data, t.nbytes);
    else if (t.dtype == st::DType::BF16)
    {
        if (scratch.n < std::min(n, piece)) scratch = DBuf<uint16_t>(q, std::min(n, piece));
        for (size_t o = 0; o < n; o += piece)
        {
            const size_t c = std::min(piece, n - o);
            up(scratch.get(), (const uint16_t*) t.data + o, c * 2);
            nn::bf16_to_half(q, scratch.get(), (int64_t) c, l.w.get() + o).wait();
        }
    }
    else if (t.dtype == st::DType::F32)
    {
        std::vector<uint16_t> h(std::min(n, piece));
        for (size_t o = 0; o < n; o += piece)
        {
            const size_t c = std::min(piece, n - o);
            const float* s = (const float*) t.data + o;
            for (size_t i = 0; i < c; ++i) h[i] = exl3::f16_from_float(s[i]);
            up(l.w.get() + o, h.data(), c * 2);
        }
    }
    else throw std::runtime_error(name + ": expected BF16, F16 or F32");
    return l;
}

// ---------------------------------------------------------------------------------------------------------------

struct Layer
{
    bool full = false;
    DBuf<float> ln1, ln2;                       // (1 + w): input_layernorm, post_attention_layernorm
    Lin qkv, z, a, b, out;                      // Gated DeltaNet
    DBuf<uint16_t> conv, A_log, dt_bias, gnorm; // bf16 bits
    Lin q, k, v, o;                             // attention
    DBuf<uint16_t> q_norm, k_norm;              // bf16 bits (the kernel adds the 1)
    Lin g, u, d;                                // MLP
};

// A decoder block, from the main model's layers (prefix lp(li)) or the MTP head's (prefix "mtp.layers.0.", always attention).
inline Layer load_layer(sycl::queue& q, const st::Model& m, const std::string& p, bool full, Upload& up, DBuf<uint16_t>& scratch)
{
    Layer L;
    L.full = full;
    if (!m.has(p + "input_layernorm.weight")) throw std::runtime_error("the checkpoint has no " + p + " (a part of it? pass --layers)");
    L.ln1 = load_norm1p(q, m, p + "input_layernorm.weight", up);
    L.ln2 = load_norm1p(q, m, p + "post_attention_layernorm.weight", up);
    if (full)
    {
        const auto a = p + "self_attn.";
        L.q = load_lin(q, m, a + "q_proj.weight", up, scratch);
        L.k = load_lin(q, m, a + "k_proj.weight", up, scratch);
        L.v = load_lin(q, m, a + "v_proj.weight", up, scratch);
        L.o = load_lin(q, m, a + "o_proj.weight", up, scratch);
        L.q_norm = load_bf16(q, m, a + "q_norm.weight", up);
        L.k_norm = load_bf16(q, m, a + "k_norm.weight", up);
    }
    else
    {
        const auto a = p + "linear_attn.";
        L.qkv = load_lin(q, m, a + "in_proj_qkv.weight", up, scratch);
        L.z = load_lin(q, m, a + "in_proj_z.weight", up, scratch);
        L.a = load_lin(q, m, a + "in_proj_a.weight", up, scratch, false);
        L.b = load_lin(q, m, a + "in_proj_b.weight", up, scratch, false);
        L.out = load_lin(q, m, a + "out_proj.weight", up, scratch);
        L.conv = load_bf16(q, m, a + "conv1d.weight", up);
        L.A_log = load_bf16(q, m, a + "A_log", up);
        L.dt_bias = load_bf16(q, m, a + "dt_bias", up);
        L.gnorm = load_bf16(q, m, a + "norm.weight", up);
    }
    L.g = load_lin(q, m, p + "mlp.gate_proj.weight", up, scratch);
    L.u = load_lin(q, m, p + "mlp.up_proj.weight", up, scratch);
    L.d = load_lin(q, m, p + "mlp.down_proj.weight", up, scratch);
    return L;
}

// The four Hessian groups of a layer.
struct Capture
{
    DBuf<float> in, o, gu, d;
    int64_t n_in = 0, n_o = 0, n_gu = 0, n_d = 0;
    DBuf<double> sens_u, sens_g;   // --out-scales auto: summed squared slopes of silu(g) * u along up and gate
    int64_t n_sens = 0;
    void alloc(sycl::queue& q, const Config& c, bool sens)
    {
        const int64_t O = std::max(c.HV * c.DV, c.NH * c.Dh);
        in = DBuf<float>(q, (size_t) c.D * c.D); in.zero();
        o = DBuf<float>(q, (size_t) O * O); o.zero();
        gu = DBuf<float>(q, (size_t) c.D * c.D); gu.zero();
        d = DBuf<float>(q, (size_t) c.I * c.I); d.zero();
        if (sens) { sens_u = DBuf<double>(q, c.I); sens_u.zero(); sens_g = DBuf<double>(q, c.I); sens_g.zero(); }
    }
    static uint64_t bytes(const Config& c)
    {
        const uint64_t O = std::max(c.HV * c.DV, c.NH * c.Dh);
        return 4 * (2 * c.D * c.D + O * O + c.I * c.I) + 16 * c.I;
    }
};

struct Scratch
{
    DBuf<half> y, qh, kh, vh, P, oh, m, tmp;
    DBuf<float> qkv, conv, z, ab_a, ab_b, g, beta, o, S, Of, out, gf, uf;
    DBuf<int32_t> ok;
    Scratch(sycl::queue& q, const Config& c, int64_t T)
    {
        const int64_t C = c.C(), QG = 2 * c.NH * c.Dh, QD = c.NH * c.Dh, KD = c.NKV * c.Dh, VO = c.HV * c.DV, grp = c.NH / c.NKV;
        const int64_t O = std::max(VO, QD);
        y = DBuf<half>(q, T * c.D); qh = DBuf<half>(q, T * QD); kh = DBuf<half>(q, T * KD); vh = DBuf<half>(q, T * KD);
        P = DBuf<half>(q, (size_t) grp * T * T); oh = DBuf<half>(q, T * O); m = DBuf<half>(q, T * c.I);
        tmp = DBuf<half>(q, T * std::max(c.I, O));
        qkv = DBuf<float>(q, T * std::max(C, QG)); conv = DBuf<float>(q, T * C); z = DBuf<float>(q, T * std::max(VO, KD));
        ab_a = DBuf<float>(q, T * c.HV); ab_b = DBuf<float>(q, T * c.HV); g = DBuf<float>(q, T * c.HV); beta = DBuf<float>(q, T * c.HV);
        o = DBuf<float>(q, T * std::max(VO, KD)); S = DBuf<float>(q, (size_t) grp * T * T); Of = DBuf<float>(q, T * QD);
        out = DBuf<float>(q, T * c.D); gf = DBuf<float>(q, T * c.I); uf = DBuf<float>(q, T * c.I);
        ok = DBuf<int32_t>(q, T);
    }
    // what the constructor above allocates, in bytes
    static uint64_t bytes(const Config& c, int64_t T)
    {
        const uint64_t C = c.C(), QG = 2 * c.NH * c.Dh, QD = c.NH * c.Dh, KD = c.NKV * c.Dh, VO = c.HV * c.DV, grp = c.NH / c.NKV, U = T;
        const uint64_t O = std::max(VO, QD);
        return 2 * (U * c.D + U * QD + 2 * U * KD + grp * U * U + U * O + U * c.I + U * std::max<uint64_t>(c.I, O)) +
               4 * (U * std::max(C, QG) + U * C + 2 * U * std::max(VO, KD) + 4 * U * c.HV + grp * U * U + U * QD + U * c.D + 2 * U * c.I) + 4 * U;
    }
};

// How many blocks accumulate() splits a Hessian of k into: the lower block triangle is (b + 1) / (2 b) of the work.
inline int hessian_blocks(int64_t k) { return k % 4 == 0 ? 4 : 1; }

// H (lower block triangle) += X^T X over the finite rows of X (T x k fp16, row stride ld); returns the number of rows used.
// (exllamav3 drops tokens with inf / NaN activations from the Hessian; the forward itself goes on.)
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

inline sycl::event add_inplace(sycl::queue& q, float* x, const float* y, int64_t n)
{
    return q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { x[i] += y[i]; });
}

// The token mixer of a layer over one row: y (T x D fp16, the normed input) -> sc.out (T x D fp32). cap: the output group's Hessian.
inline void mixer(sycl::queue& q, const Config& c, const Layer& L, int64_t T, Scratch& sc, const float* inv, Capture* cap)
{
    const float eps = (float) c.eps;
    if (!L.full)
    {
        const int64_t KD = c.HK * c.DK, C = c.C(), VO = c.HV * c.DV;
        gemm_lin(q, sc.y.get(), T, L.qkv, sc.qkv.get());
        gemm_lin(q, sc.y.get(), T, L.z, sc.z.get());
        gemm_lin(q, sc.y.get(), T, L.a, sc.ab_a.get());
        gemm_lin(q, sc.y.get(), T, L.b, sc.ab_b.get());
        nn::gdn_conv(q, sc.qkv.get(), L.conv.get(), T, C, (int) c.KS, sc.conv.get());
        nn::gdn_gates(q, sc.ab_a.get(), sc.ab_b.get(), L.A_log.get(), L.dt_bias.get(), T, c.HV, sc.g.get(), sc.beta.get());
        nn::l2norm_heads(q, sc.conv.get(), T, C, c.HK, c.DK, 1.0f / std::sqrt((float) c.DK));     // q
        nn::l2norm_heads(q, sc.conv.get() + KD, T, C, c.HK, c.DK, 1.0f);                            // k
        nn::gdn_scan(q, sc.conv.get(), T, C, c.HK, c.HV, c.DK, c.DV, sc.g.get(), sc.beta.get(), sc.o.get());
        nn::gdn_norm_swish(q, sc.o.get(), sc.z.get(), L.gnorm.get(), T, c.HV, c.DV, eps, sc.oh.get());
        if (cap) cap->n_o += accumulate(q, sc.oh.get(), T, VO, cap->o.get(), sc);
        gemm_lin(q, sc.oh.get(), T, L.out, sc.out.get());
    }
    else
    {
        const int64_t NH = c.NH, NKV = c.NKV, Dh = c.Dh, QD = NH * Dh, KD = NKV * Dh, grp = NH / NKV;
        gemm_lin(q, sc.y.get(), T, L.q, sc.qkv.get());                 // [T, NH, 2 Dh]: query | gate per head
        gemm_lin(q, sc.y.get(), T, L.k, sc.o.get());
        gemm_lin(q, sc.y.get(), T, L.v, sc.z.get());
        nn::head_norm1p_rope(q, sc.qkv.get(), T, NH, Dh, 2 * Dh, 0, L.q_norm.get(), inv, (int) c.R, eps, sc.qh.get());
        nn::head_norm1p_rope(q, sc.o.get(), T, NKV, Dh, Dh, 0, L.k_norm.get(), inv, (int) c.R, eps, sc.kh.get());
        nn::head_norm1p_rope(q, sc.z.get(), T, NKV, Dh, Dh, 0, nullptr, nullptr, 0, eps, sc.vh.get());
        const float scale = 1.0f / std::sqrt((float) Dh);
        for (int64_t kv = 0; kv < NKV; ++kv)
        {
            for (int64_t j = 0; j < grp; ++j)
                blas::gemm(q, transpose::nontrans, transpose::trans, T, T, Dh, scale, sc.qh.get() + (kv * grp + j) * Dh, QD,
                           sc.kh.get() + kv * Dh, KD, 0.0f, sc.S.get() + j * T * T, T);
            nn::softmax_mask(q, sc.S.get(), grp, T, 0, sc.P.get());
            for (int64_t j = 0; j < grp; ++j)
                blas::gemm(q, transpose::nontrans, transpose::nontrans, T, Dh, T, 1.0f, sc.P.get() + j * T * T, T, sc.vh.get() + kv * Dh, KD,
                           0.0f, sc.Of.get() + (kv * grp + j) * Dh, QD);
        }
        nn::gate_mul(q, sc.Of.get(), sc.qkv.get(), T, NH, Dh, sc.oh.get());
        if (cap) cap->n_o += accumulate(q, sc.oh.get(), T, QD, cap->o.get(), sc);
        gemm_lin(q, sc.oh.get(), T, L.o, sc.out.get());
    }
}

// One decoder layer over one calibration row, x (T x D fp32) updated in place. cap: accumulate Hessians.
inline void forward_layer(sycl::queue& q, const Config& c, const Layer& L, float* x, int64_t T, Scratch& sc, const float* inv, Capture* cap)
{
    const float eps = (float) c.eps;
    nn::rmsnorm_w(q, x, L.ln1.get(), T, c.D, eps, sc.y.get());
    if (cap) cap->n_in += accumulate(q, sc.y.get(), T, c.D, cap->in.get(), sc);
    mixer(q, c, L, T, sc, inv, cap);
    add_inplace(q, x, sc.out.get(), T * c.D);
    nn::rmsnorm_w(q, x, L.ln2.get(), T, c.D, eps, sc.y.get());
    if (cap) cap->n_gu += accumulate(q, sc.y.get(), T, c.D, cap->gu.get(), sc);
    gemm_lin(q, sc.y.get(), T, L.g, sc.gf.get());
    gemm_lin(q, sc.y.get(), T, L.u, sc.uf.get());
    if (cap && cap->sens_u.get())
    {
        nn::silu_slopes_acc(q, sc.gf.get(), sc.uf.get(), T, c.I, INFINITY, cap->sens_u.get(), cap->sens_g.get());
        cap->n_sens += T;
    }
    nn::silu_mul(q, sc.gf.get(), sc.uf.get(), T * c.I, INFINITY, sc.m.get());
    if (cap) cap->n_d += accumulate(q, sc.m.get(), T, c.I, cap->d.get(), sc);
    gemm_lin(q, sc.m.get(), T, L.d, sc.out.get());
    add_inplace(q, x, sc.out.get(), T * c.D).wait();
}

} // namespace q35
