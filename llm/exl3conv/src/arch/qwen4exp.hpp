// exl3conv: Qwen3.8-Flash-Next (qwen4_exp) text decoder on the B70 for calibration (notes/qwen4exp-forward.md §2-3).
//
// A row's state is 4 fp32 residual streams X [T, 4, D]. A decoder layer is two gated-residual sites: the first wraps
// Gated DeltaNet (L % 4 != 3) or gated full attention, the second the MoE. Layer 1 starts with the PLE (hashed n-gram
// features from the quantized table). Hessians are captured per input group (Capture below); in capture mode the
// MoE routes every token through every expert (exllamav3's all-experts capture) and only records the Hessians,
// since nothing in the layer reads its output.
#pragma once

#include "arch/gemma4.hpp"           // Upload, Lin, gemm_lin, load helpers
#include "nn/qwen4exp_ops.hpp"

#include <map>
#include <string>
#include <vector>

namespace qwen4exp {

using g4::Lin;
using g4::Upload;
using exl3::DBuf;
using half = sycl::half;
using json = nlohmann::json;
namespace blas = oneapi::mkl::blas::row_major;
using oneapi::mkl::transpose;

struct Config
{
    int64_t D, L, S, NH, NKV, Dh, R, HK, HV, DK, DV, KS, E, topk, F, SF, rank, vocab, ple_layer;
    double eps, theta;
    int64_t eos;
    std::vector<bool> full;
    std::string prefix = "model.language_model.";

    explicit Config(const json& cfg)
    {
        if (cfg.value("model_type", "") != "qwen4_exp") throw std::runtime_error("not a qwen4_exp model");
        const auto& t = cfg.at("text_config");
        D = t.at("hidden_size"); L = t.at("num_hidden_layers"); S = t.at("hc_count"); rank = t.at("hc_lowrank");
        NH = t.at("num_attention_heads"); NKV = t.at("num_key_value_heads"); Dh = t.at("head_dim");
        R = (int64_t) (t.value("partial_rotary_factor", 1.0) * (double) Dh);
        HK = t.at("linear_num_key_heads"); HV = t.at("linear_num_value_heads");
        DK = t.at("linear_key_head_dim"); DV = t.at("linear_value_head_dim"); KS = t.at("linear_conv_kernel_dim");
        E = t.at("num_experts"); topk = t.at("num_experts_per_tok"); F = t.at("moe_intermediate_size");
        SF = t.at("shared_expert_intermediate_size"); vocab = t.at("vocab_size"); eps = t.at("rms_norm_eps");
        theta = t.contains("rope_parameters") ? (double) t["rope_parameters"].value("rope_theta", 1e7) : (double) t.value("rope_theta", 1e7);
        eos = t.value("eos_token_id", 248044);
        ple_layer = (int64_t) t.at("ple_layer_ids")[0] - 1;            // 1-based in the config
        for (const auto& s : t.at("layer_types")) full.push_back(s == "full_attention");
        if (t.value("output_gate_type", "sigmoid") != "sigmoid") throw std::runtime_error("GDN output gate must be sigmoid");
        if (DK != DV || DK != 128) throw std::runtime_error("GDN head dims other than 128 are not supported");
    }
    std::string lp(int64_t l) const { return prefix + "layers." + std::to_string(l) + "."; }
    std::vector<float> inv_freq() const
    {
        std::vector<float> v(R / 2);
        for (int64_t j = 0; j < R / 2; ++j) v[j] = (float) std::pow(theta, -2.0 * (double) j / (double) R);
        return v;
    }
};

// A gated-residual site: hc_norm (bf16), the rank-r gate (fp32 copies of the fp16 tables) and the inject weights.
struct Site { DBuf<uint16_t> norm; DBuf<float> down, up, inj; };

// fp32 copy on the card of a checkpoint tensor (BF16 or F16).
inline DBuf<float> load_f32(sycl::queue& q, const st::Model& m, const std::string& name, Upload& up)
{
    const auto& t = m.get(name);
    std::vector<float> h((size_t) t.numel());
    const uint16_t* p = (const uint16_t*) t.data;
    if (t.dtype == st::DType::BF16) for (size_t i = 0; i < h.size(); ++i) h[i] = st::bf16_to_float(p[i]);
    else if (t.dtype == st::DType::F16) for (size_t i = 0; i < h.size(); ++i) h[i] = exl3::f16_to_float(p[i]);
    else throw std::runtime_error(name + ": expected BF16 or F16");
    DBuf<float> d(q, h.size());
    up(d.get(), h.data(), h.size() * 4);
    return d;
}

// fp16 weights of the gate/up and down of all experts, HF layout: gu [E, 2F, D], dn [E, D, F] (or the dequantized
// (in, out) layouts when hf = false: gu [E, D, 2F], dn [E, F, D]).
struct Experts { DBuf<half> gu, dn; bool hf = true; };

struct Layer
{
    bool full = false, ple = false;
    Site sa, sm;
    // GDN
    Lin qkv, z, a, b, out;
    DBuf<uint16_t> conv, A_log, dt_bias, gnorm;
    // attention
    Lin q, k, v, o, idx;
    DBuf<uint16_t> q_norm, k_norm;
    // MoE
    Lin router, sgate, s_gate, s_up, s_down;
    Experts ex;
    // PLE (layer 1)
    Lin ple_key, ple_val;
    DBuf<uint16_t> ple_nk, ple_nq, ple_nc, ple_conv;
};

inline Site load_site(sycl::queue& q, const st::Model& m, const std::string& p, Upload& up)
{
    Site s;
    s.norm = g4::load_bf16_vec(q, m, p + "hc_norm.weight", up);
    s.down = load_f32(q, m, p + "input_mix_weight_down.weight", up);
    s.up = load_f32(q, m, p + "input_mix_weight_up.weight", up);
    if (m.has(p + "block_inject_weight.weight")) s.inj = load_f32(q, m, p + "block_inject_weight.weight", up);
    return s;
}

// A BF16 [out, in] linear without the multiple-of-128 check (routers, a/b projections).
inline Lin load_small(sycl::queue& q, const st::Model& m, const std::string& name, Upload& up)
{
    const auto& t = m.get(name);
    Lin l; l.out = t.shape[0]; l.in = t.shape[1]; l.hf = true;
    DBuf<uint16_t> s(q, (size_t) t.numel());
    up(s.get(), t.data, t.nbytes);
    l.w = DBuf<half>(q, (size_t) t.numel());
    if (t.dtype == st::DType::BF16) nn::bf16_to_half(q, s.get(), t.numel(), l.w.get()).wait();
    else q.memcpy(l.w.get(), s.get(), t.nbytes).wait();
    return l;
}

inline Layer load_layer(sycl::queue& q, const st::Model& m, const Config& c, int64_t li, Upload& up, DBuf<uint16_t>& scratch)
{
    Layer Ly;
    const auto p = c.lp(li);
    Ly.full = c.full[li];
    Ly.ple = li == c.ple_layer;
    Ly.sa = load_site(q, m, p + "attn_hyper_connection.", up);
    Ly.sm = load_site(q, m, p + "mlp_hyper_connection.", up);
    if (Ly.full)
    {
        Ly.q = g4::load_linear(q, m, p + "self_attn.q_proj.weight", up, scratch);
        Ly.k = g4::load_linear(q, m, p + "self_attn.k_proj.weight", up, scratch);
        Ly.v = g4::load_linear(q, m, p + "self_attn.v_proj.weight", up, scratch);
        Ly.o = g4::load_linear(q, m, p + "self_attn.o_proj.weight", up, scratch);
        Ly.idx = g4::load_linear(q, m, p + "self_attn.indexer.index_qk_proj.weight", up, scratch);
        Ly.q_norm = g4::load_bf16_vec(q, m, p + "self_attn.q_norm.weight", up);
        Ly.k_norm = g4::load_bf16_vec(q, m, p + "self_attn.k_norm.weight", up);
    }
    else
    {
        Ly.qkv = g4::load_linear(q, m, p + "linear_attn.in_proj_qkv.weight", up, scratch);
        Ly.z = g4::load_linear(q, m, p + "linear_attn.in_proj_z.weight", up, scratch);
        Ly.a = load_small(q, m, p + "linear_attn.in_proj_a.weight", up);
        Ly.b = load_small(q, m, p + "linear_attn.in_proj_b.weight", up);
        Ly.out = g4::load_linear(q, m, p + "linear_attn.out_proj.weight", up, scratch);
        Ly.conv = g4::load_bf16_vec(q, m, p + "linear_attn.conv1d.weight", up);
        Ly.A_log = g4::load_bf16_vec(q, m, p + "linear_attn.A_log", up);
        Ly.dt_bias = g4::load_bf16_vec(q, m, p + "linear_attn.dt_bias", up);
        Ly.gnorm = g4::load_bf16_vec(q, m, p + "linear_attn.norm.weight", up);
    }
    Ly.router = load_small(q, m, p + "mlp.gate.weight", up);
    Ly.sgate = load_small(q, m, p + "mlp.shared_expert_gate.weight", up);
    Ly.s_gate = g4::load_linear(q, m, p + "mlp.shared_expert.gate_proj.weight", up, scratch);
    Ly.s_up = g4::load_linear(q, m, p + "mlp.shared_expert.up_proj.weight", up, scratch);
    Ly.s_down = g4::load_linear(q, m, p + "mlp.shared_expert.down_proj.weight", up, scratch);
    {
        const auto& gu = m.get(p + "mlp.experts.gate_up_proj");
        const auto& dn = m.get(p + "mlp.experts.down_proj");
        for (auto* t : {&gu, &dn})
        {
            DBuf<half>& dst = t == &gu ? Ly.ex.gu : Ly.ex.dn;
            dst = DBuf<half>(q, (size_t) t->numel());
            // In 256 MB pieces through a small BF16 staging buffer.
            const int64_t piece = 128ll << 20;
            if (scratch.n < (size_t) piece) scratch = DBuf<uint16_t>(q, piece);
            for (int64_t o = 0; o < t->numel(); o += piece)
            {
                const int64_t nn_ = std::min(piece, t->numel() - o);
                up(scratch.get(), (const uint16_t*) t->data + o, nn_ * 2);
                nn::bf16_to_half(q, scratch.get(), nn_, dst.get() + o).wait();
            }
        }
        Ly.ex.hf = true;
    }
    if (Ly.ple)
    {
        const auto pp = p + "ple.";
        Ly.ple_key = g4::load_linear(q, m, pp + "key_proj.weight", up, scratch);
        Ly.ple_val = g4::load_linear(q, m, pp + "value_proj.weight", up, scratch);
        Ly.ple_nk = g4::load_bf16_vec(q, m, pp + "norm_key.weight", up);
        Ly.ple_nq = g4::load_bf16_vec(q, m, pp + "norm_query.weight", up);
        Ly.ple_nc = g4::load_bf16_vec(q, m, pp + "norm_conv.weight", up);
        Ly.ple_conv = g4::load_bf16_vec(q, m, pp + "conv1d.weight", up);
    }
    return Ly;
}

// Hessians of a layer, by qmap name: "attn.input", "attn.o" (o_proj / out_proj), "mlp.input", "mlp.down" (shared
// expert) and "e<E>.down" for the routed experts (one buffer of E x F x F).
struct Capture
{
    DBuf<float> attn_in, attn_o, mlp_in, s_down, e_down;
    int64_t n_attn_in = 0, n_attn_o = 0, n_mlp_in = 0, n_s_down = 0, n_e_down = 0;
    void alloc(sycl::queue& q, const Config& c, bool full)
    {
        const int64_t O = full ? c.NH * c.Dh : c.HV * c.DV;
        attn_in = DBuf<float>(q, (size_t) c.D * c.D); attn_in.zero();
        attn_o = DBuf<float>(q, (size_t) O * O); attn_o.zero();
        mlp_in = DBuf<float>(q, (size_t) c.D * c.D); mlp_in.zero();
        s_down = DBuf<float>(q, (size_t) c.SF * c.SF); s_down.zero();
        e_down = DBuf<float>(q, (size_t) c.E * c.F * c.F); e_down.zero();
    }
};

struct Scratch
{
    DBuf<float> N, a, Gl, inj, out, qkv, conv, z, ab_a, ab_b, g, beta, o, S, Of, gu, sg, rl, rw_f, ple_k, ple_v, ple_g;
    DBuf<half> y, oh, qh, kh, vh, P, m, sm, eh, eh2;
    DBuf<int32_t> rid;
    DBuf<float> rwt;
    int64_t T;
    static constexpr int64_t ECH = 32;    // experts per chunk in the all-experts capture
    Scratch(sycl::queue& q, const Config& c, int64_t T_) : T(T_)
    {
        const int64_t SD = c.S * c.D, C = 2 * c.HK * c.DK + c.HV * c.DV, QG = c.NH * 2 * c.Dh;
        N = DBuf<float>(q, T * SD); a = DBuf<float>(q, T * c.rank); Gl = DBuf<float>(q, T * SD); inj = DBuf<float>(q, T * c.S);
        out = DBuf<float>(q, T * c.D); qkv = DBuf<float>(q, T * std::max(C, QG)); conv = DBuf<float>(q, T * C);
        z = DBuf<float>(q, T * c.HV * c.DV); ab_a = DBuf<float>(q, T * c.HV); ab_b = DBuf<float>(q, T * c.HV);
        g = DBuf<float>(q, T * c.HV); beta = DBuf<float>(q, T * c.HV); o = DBuf<float>(q, T * std::max(c.HV * c.DV, c.NH * c.Dh));
        S = DBuf<float>(q, (size_t) c.NH * T * T); Of = DBuf<float>(q, T * c.NH * c.Dh);
        gu = DBuf<float>(q, T * ECH * 2 * c.F); sg = DBuf<float>(q, T * 2 * c.SF); rl = DBuf<float>(q, T * c.E);
        rw_f = DBuf<float>(q, T);
        ple_k = DBuf<float>(q, T * SD); ple_v = DBuf<float>(q, T * c.D); ple_g = DBuf<float>(q, T * SD);
        y = DBuf<half>(q, T * c.D); oh = DBuf<half>(q, T * std::max(c.HV * c.DV, c.NH * c.Dh));
        qh = DBuf<half>(q, T * c.NH * c.Dh); kh = DBuf<half>(q, T * c.NKV * c.Dh); vh = DBuf<half>(q, T * c.NKV * c.Dh);
        P = DBuf<half>(q, (size_t) c.NH * T * T); m = DBuf<half>(q, T * ECH * c.F); sm = DBuf<half>(q, T * c.SF);
        eh = DBuf<half>(q, T * c.D); eh2 = DBuf<half>(q, T * c.D);
        rid = DBuf<int32_t>(q, T * c.topk); rwt = DBuf<float>(q, T * c.topk);
    }
};

// H (k x k) += X^T X over the T rows of X (fp16, row stride ld), batch of nb Hessians spaced strideH / strideX apart.
inline void hess(sycl::queue& q, const half* X, int64_t T, int64_t k, int64_t ld, float* H, int64_t nb = 1, int64_t sx = 0, int64_t sh = 0)
{
    if (nb == 1) blas::gemm(q, transpose::trans, transpose::nontrans, k, k, T, 1.0f, X, ld, X, ld, 1.0f, H, k);
    else blas::gemm_batch(q, transpose::trans, transpose::nontrans, k, k, T, 1.0f, X, ld, sx, X, ld, sx, 1.0f, H, k, sh, nb);
}

// f32 GEMM y = x W^T for the site tables (W [out, in] fp32).
inline void gemm_f32(sycl::queue& q, const float* x, int64_t T, int64_t in, const float* W, int64_t out, float* y)
{
    blas::gemm(q, transpose::nontrans, transpose::trans, T, out, in, 1.0f, x, in, W, in, 0.0f, y, out);
}

// The mix of a site: N = group-normed streams, y (fp16) = the sublayer input; inj = 2 sigmoid(N_flat Winj^T / S).
inline void site_mix(sycl::queue& q, const Config& c, const Site& s, const float* X, int64_t T, Scratch& sc, bool with_inj)
{
    const int64_t SD = c.S * c.D;
    nn::group_norm1p(q, X, s.norm.get(), T, (int) c.S, c.D, (float) c.eps, sc.N.get());
    gemm_f32(q, sc.N.get(), T, SD, s.down.get(), c.rank, sc.a.get());
    nn::silu_scaled(q, sc.a.get(), T * c.rank, (float) c.S);
    gemm_f32(q, sc.a.get(), T, c.rank, s.up.get(), SD, sc.Gl.get());
    nn::hc_mix_out(q, sc.Gl.get(), sc.N.get(), T, (int) c.S, c.D, sc.y.get());
    if (with_inj)
    {
        gemm_f32(q, sc.N.get(), T, SD, s.inj.get(), c.S, sc.inj.get());
        nn::hc_inject(q, sc.inj.get(), T, (int) c.S);
    }
}

// The PLE layer (§2.4): X += Gv + silu(conv(GroupNorm1p(Gv))) with Gv = sigmoid(signed sqrt(<Kn, Qn> / sqrt(D))) * V,
// from E (T x 2560 fp16, the decoded n-gram rows of the row's tokens).
inline void ple(sycl::queue& q, const Config& c, const Layer& L, const half* E, float* X, int64_t T, Scratch& sc)
{
    const int64_t SD = c.S * c.D, Dd = c.D;
    const int S = (int) c.S;
    const float eps = (float) c.eps;
    g4::gemm_lin(q, E, T, L.ple_key, sc.ple_k.get());                     // [T, S*D]
    g4::gemm_lin(q, E, T, L.ple_val, sc.ple_v.get());                     // [T, D]
    nn::group_norm1p(q, sc.ple_k.get(), L.ple_nk.get(), T, S, Dd, eps, sc.Gl.get());   // Kn -> Gl
    nn::group_norm1p(q, X, L.ple_nq.get(), T, S, Dd, eps, sc.N.get());                 // Qn -> N
    float* Kn = sc.Gl.get(); float* Qn = sc.N.get(); float* V = sc.ple_v.get(); float* Gv = sc.ple_g.get();
    // g[t, s] = signed sqrt(sum_d Kn Qn / sqrt(D)); Gv = sigmoid(g) * V
    constexpr int WG = 256;
    q.parallel_for(sycl::nd_range<1>(T * S * WG, WG), [=](sycl::nd_item<1> it) {
        const int64_t ts = it.get_group(0), t = ts / S; const int l = (int) it.get_local_id(0);
        float acc = 0;
        for (int64_t d = l; d < Dd; d += WG) acc += Kn[ts * Dd + d] * Qn[ts * Dd + d];
        acc = sycl::reduce_over_group(it.get_group(), acc, sycl::plus<float>()) / sycl::sqrt((float) Dd);
        const float sg = acc > 0 ? 1.0f : acc < 0 ? -1.0f : 0.0f;
        const float gv = nn::sigm(sg * sycl::sqrt(sycl::fmax(sycl::fabs(acc), 1e-6f)));
        for (int64_t d = l; d < Dd; d += WG) Gv[ts * Dd + d] = gv * V[t * Dd + d];
    });
    nn::group_norm1p(q, Gv, L.ple_nc.get(), T, S, Dd, eps, sc.Gl.get());               // Cn -> Gl
    const float* Cn = sc.Gl.get();
    const uint16_t* wc = L.ple_conv.get();
    const int64_t KSz = 4, dil = 3;
    q.parallel_for(sycl::range<2>(T, SD), [=](sycl::id<2> id) {
        const int64_t t = id[0], ch = id[1];
        float acc = 0;
        for (int64_t j = 0; j < KSz; ++j)
        {
            const int64_t tt = t - dil * (KSz - 1 - j);
            if (tt >= 0) acc += nn::bf16w(wc[ch * KSz + j]) * Cn[tt * SD + ch];
        }
        X[t * SD + ch] += Gv[t * SD + ch] + nn::silu(acc);
    }).wait();
}

// One decoder layer over one row: X (T x S x D fp32) updated in place. cap: capture Hessians (all-experts MoE).
// ple_rows: the decoded n-gram rows of this row's tokens (layer 1 only).
inline void forward_layer(sycl::queue& q, const Config& c, const Layer& L, float* X, int64_t T, Scratch& sc, const float* inv,
                          Capture* cap, const half* ple_rows)
{
    const int S = (int) c.S;
    const float eps = (float) c.eps;
    if (L.ple) ple(q, c, L, ple_rows, X, T, sc);
    // --- attention / GDN site
    site_mix(q, c, L.sa, X, T, sc, true);
    if (cap) { hess(q, sc.y.get(), T, c.D, c.D, cap->attn_in.get()); cap->n_attn_in += T; }
    if (!L.full)
    {
        const int64_t KD = c.HK * c.DK, C = 2 * KD + c.HV * c.DV;
        g4::gemm_lin(q, sc.y.get(), T, L.qkv, sc.qkv.get());
        g4::gemm_lin(q, sc.y.get(), T, L.z, sc.z.get());
        g4::gemm_lin(q, sc.y.get(), T, L.a, sc.ab_a.get());
        g4::gemm_lin(q, sc.y.get(), T, L.b, sc.ab_b.get());
        nn::gdn_conv(q, sc.qkv.get(), L.conv.get(), T, C, (int) c.KS, sc.conv.get());
        nn::gdn_gates(q, sc.ab_a.get(), sc.ab_b.get(), L.A_log.get(), L.dt_bias.get(), T, c.HV, sc.g.get(), sc.beta.get());
        nn::l2norm_heads(q, sc.conv.get(), T, C, c.HK, c.DK, 1.0f / std::sqrt((float) c.DK));               // q
        nn::l2norm_heads(q, sc.conv.get() + KD, T, C, c.HK, c.DK, 1.0f);                                      // k
        nn::gdn_scan(q, sc.conv.get(), T, C, c.HK, c.HV, c.DK, c.DV, sc.g.get(), sc.beta.get(), sc.o.get());
        nn::gdn_gated_norm(q, sc.o.get(), sc.z.get(), L.gnorm.get(), T, c.HV, c.DV, eps, sc.oh.get());
        if (cap) { hess(q, sc.oh.get(), T, c.HV * c.DV, c.HV * c.DV, cap->attn_o.get()); cap->n_attn_o += T; }
        g4::gemm_lin(q, sc.oh.get(), T, L.out, sc.out.get());
    }
    else
    {
        const int64_t NH = c.NH, NKV = c.NKV, Dh = c.Dh, QD = NH * Dh, KD = NKV * Dh;
        g4::gemm_lin(q, sc.y.get(), T, L.q, sc.qkv.get());                    // [T, NH, 2 Dh]: q | gate per head
        g4::gemm_lin(q, sc.y.get(), T, L.k, sc.o.get());                      // k -> o (scratch)
        g4::gemm_lin(q, sc.y.get(), T, L.v, sc.z.get());                      // v -> z (scratch)
        nn::head_norm1p_rope(q, sc.qkv.get(), T, NH, Dh, 2 * Dh, 0, L.q_norm.get(), inv, (int) c.R, eps, sc.qh.get());
        nn::head_norm1p_rope(q, sc.o.get(), T, NKV, Dh, Dh, 0, L.k_norm.get(), inv, (int) c.R, eps, sc.kh.get());
        nn::head_norm1p_rope(q, sc.z.get(), T, NKV, Dh, Dh, 0, nullptr, nullptr, 0, eps, sc.vh.get());
        const int64_t grp = NH / NKV;
        const float scale = 1.0f / std::sqrt((float) Dh);
        for (int64_t hq = 0; hq < NH; ++hq)
            blas::gemm(q, transpose::nontrans, transpose::trans, T, T, Dh, scale, sc.qh.get() + hq * Dh, QD,
                       sc.kh.get() + (hq / grp) * Dh, KD, 0.0f, sc.S.get() + hq * T * T, T);
        nn::softmax_mask(q, sc.S.get(), NH, T, 0, sc.P.get());
        for (int64_t hq = 0; hq < NH; ++hq)
            blas::gemm(q, transpose::nontrans, transpose::nontrans, T, Dh, T, 1.0f, sc.P.get() + hq * T * T, T,
                       sc.vh.get() + (hq / grp) * Dh, KD, 0.0f, sc.Of.get() + hq * Dh, QD);
        nn::gate_mul(q, sc.Of.get(), sc.qkv.get(), T, NH, Dh, sc.oh.get());
        if (cap) { hess(q, sc.oh.get(), T, QD, QD, cap->attn_o.get()); cap->n_attn_o += T; }
        g4::gemm_lin(q, sc.oh.get(), T, L.o, sc.out.get());
    }
    nn::hc_apply(q, X, sc.inj.get(), sc.out.get(), T, S, c.D);
    // --- MoE site
    site_mix(q, c, L.sm, X, T, sc, true);
    const int64_t F = c.F, Dd = c.D;
    if (cap)
    {
        hess(q, sc.y.get(), T, Dd, Dd, cap->mlp_in.get()); cap->n_mlp_in += T;
        // Shared expert intermediate.
        g4::gemm_lin(q, sc.y.get(), T, L.s_gate, sc.sg.get());
        g4::gemm_lin(q, sc.y.get(), T, L.s_up, sc.sg.get() + T * c.SF);
        float* sg = sc.sg.get(); half* smp = sc.sm.get(); const int64_t SF = c.SF;
        q.parallel_for(sycl::range<1>(T * SF), [=](sycl::id<1> i) {
            smp[i] = half(nn::silu((float) half(sg[i])) * (float) half(sg[T * SF + i]));
        });
        hess(q, sc.sm.get(), T, SF, SF, cap->s_down.get()); cap->n_s_down += T;
        // All experts: gate/up in chunks, then every expert's down Hessian over every token.
        const int64_t ECH = Scratch::ECH;
        for (int64_t e0 = 0; e0 < c.E; e0 += ECH)
        {
            const int64_t ne = std::min(ECH, c.E - e0);
            blas::gemm(q, transpose::nontrans, transpose::trans, T, ne * 2 * F, Dd, 1.0f, sc.y.get(), Dd,
                       L.ex.gu.get() + e0 * 2 * F * Dd, Dd, 0.0f, sc.gu.get(), ne * 2 * F);
            // m[t, e, :] = silu(gate) * up; the gate/up block of expert e sits at columns e * 2F
            float* gu = sc.gu.get(); half* mp = sc.m.get();
            q.parallel_for(sycl::range<2>(T, ne * F), [=](sycl::id<2> id) {
                const int64_t t = id[0], ef = id[1], e = ef / F, f = ef % F;
                const float gg = (float) half(gu[t * ne * 2 * F + e * 2 * F + f]), uu = (float) half(gu[t * ne * 2 * F + e * 2 * F + F + f]);
                mp[t * ne * F + ef] = half(nn::silu(gg) * uu);
            });
            hess(q, sc.m.get(), T, F, ne * F, cap->e_down.get() + e0 * F * F, ne, F, F * F);
        }
        q.wait();
        cap->n_e_down += T;
        return;                       // nothing in the layer reads the MoE output during capture
    }
    // Routed experts, real top-k, plus the shared expert.
    g4::gemm_lin(q, sc.y.get(), T, L.router, sc.rl.get());
    nn::router_topk(q, sc.rl.get(), T, (int) c.E, (int) c.topk, sc.rid.get(), sc.rwt.get());
    std::vector<int32_t> ids(T * c.topk);
    std::vector<float> wts(T * c.topk);
    sc.rid.download(ids.data(), ids.size());
    sc.rwt.download(wts.data(), wts.size());
    // Group (token, weight) pairs by expert.
    std::vector<std::vector<std::pair<int32_t, float>>> by(c.E);
    for (int64_t t = 0; t < T; ++t) for (int64_t s = 0; s < c.topk; ++s) by[ids[t * c.topk + s]].push_back({(int32_t) t, wts[t * c.topk + s]});
    q.memset(sc.out.get(), 0, sizeof(float) * T * Dd).wait();
    DBuf<int32_t> tok(q, T * c.topk); DBuf<float> tw(q, T * c.topk);
    std::vector<int32_t> htok; std::vector<float> htw; std::vector<int64_t> start(c.E + 1, 0);
    for (int64_t e = 0; e < c.E; ++e) { start[e] = (int64_t) htok.size(); for (auto& [t, w] : by[e]) { htok.push_back(t); htw.push_back(w); } }
    start[c.E] = (int64_t) htok.size();
    tok.upload(htok.data(), htok.size()); tw.upload(htw.data(), htw.size());
    // Gather the routed tokens' inputs once, in expert order.
    const int64_t NR = (int64_t) htok.size();
    DBuf<half> xs(q, NR * Dd), ms(q, NR * F);
    DBuf<float> gus(q, NR * 2 * F), ds(q, NR * Dd);
    { half* xp = xs.get(); const half* yp = sc.y.get(); const int32_t* tp = tok.get();
      q.parallel_for(sycl::range<2>(NR, Dd), [=](sycl::id<2> id) { xp[id[0] * Dd + id[1]] = yp[(int64_t) tp[id[0]] * Dd + id[1]]; }); }
    for (int64_t e = 0; e < c.E; ++e)
    {
        const int64_t n = start[e + 1] - start[e];
        if (!n) continue;
        const half* x = xs.get() + start[e] * Dd;
        if (L.ex.hf)
            blas::gemm(q, transpose::nontrans, transpose::trans, n, 2 * F, Dd, 1.0f, x, Dd, L.ex.gu.get() + e * 2 * F * Dd, Dd, 0.0f,
                       gus.get() + start[e] * 2 * F, 2 * F);
        else
            blas::gemm(q, transpose::nontrans, transpose::nontrans, n, 2 * F, Dd, 1.0f, x, Dd, L.ex.gu.get() + e * Dd * 2 * F, 2 * F, 0.0f,
                       gus.get() + start[e] * 2 * F, 2 * F);
    }
    nn::silu_mul(q, gus.get(), NR, F, 2 * F, ms.get(), F);
    for (int64_t e = 0; e < c.E; ++e)
    {
        const int64_t n = start[e + 1] - start[e];
        if (!n) continue;
        if (L.ex.hf)
            blas::gemm(q, transpose::nontrans, transpose::trans, n, Dd, F, 1.0f, ms.get() + start[e] * F, F, L.ex.dn.get() + e * Dd * F, F, 0.0f,
                       ds.get() + start[e] * Dd, Dd);
        else
            blas::gemm(q, transpose::nontrans, transpose::nontrans, n, Dd, F, 1.0f, ms.get() + start[e] * F, F, L.ex.dn.get() + e * F * Dd, Dd, 0.0f,
                       ds.get() + start[e] * Dd, Dd);
    }
    // out[t] = sum over its experts of w * down
    { float* op = sc.out.get(); const float* dp = ds.get(); const int32_t* tp = tok.get(); const float* wp = tw.get();
      q.parallel_for(sycl::range<2>(NR, Dd), [=](sycl::id<2> id) {
          sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device> a(op[(int64_t) tp[id[0]] * Dd + id[1]]);
          a.fetch_add(wp[id[0]] * dp[id[0] * Dd + id[1]]);
      }); }
    // Shared expert: sigmoid(y Wsg^T) * down(silu(gate) * up)
    g4::gemm_lin(q, sc.y.get(), T, L.s_gate, sc.sg.get());
    g4::gemm_lin(q, sc.y.get(), T, L.s_up, sc.sg.get() + T * c.SF);
    { float* sg = sc.sg.get(); half* smp = sc.sm.get(); const int64_t SF = c.SF;
      q.parallel_for(sycl::range<1>(T * SF), [=](sycl::id<1> i) { smp[i] = half(nn::silu((float) half(sg[i])) * (float) half(sg[T * SF + i])); }); }
    g4::gemm_lin(q, sc.sm.get(), T, L.s_down, sc.Gl.get());                // [T, D] -> Gl (scratch)
    g4::gemm_lin(q, sc.y.get(), T, L.sgate, sc.rw_f.get());                 // [T, 1]
    { float* op = sc.out.get(); const float* sd = sc.Gl.get(); const float* gz = sc.rw_f.get();
      q.parallel_for(sycl::range<2>(T, Dd), [=](sycl::id<2> id) { op[id[0] * Dd + id[1]] += nn::sigm(gz[id[0]]) * sd[id[0] * Dd + id[1]]; }); }
    nn::hc_apply(q, X, sc.inj.get(), sc.out.get(), T, S, c.D).wait();
}

} // namespace qwen4exp
