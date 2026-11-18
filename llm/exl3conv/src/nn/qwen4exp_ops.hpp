// exl3conv: device kernels for the Qwen3.8-Flash-Next calibration forward (notes/qwen4exp-forward.md §2). Streams fp32,
// linear inputs fp16 (what the Hessians see), everything else fp32.
//
//   group_norm1p   N[t, s, :] = rms(X[t, s, :]) * (1 + w[s*D ..])                 4 streams x D, eps inside the root
//   hc_mix_out     y[t] = fp16(mean_s G[t, s] * N[t, s])                          the sublayer input of a site
//   hc_apply       X[t, s] += inj[t, s] * out[t]                                   on the raw streams
//   head_norm1p_rope  per head: n = fp16(rms(v) * (1 + w)); RoPE on pairs (j, j + R/2), j < R/2; fp16
//   gdn_conv       c[t, ch] = silu(sum_j w[ch, j] * x[t - 3 + j, ch])              causal, kernel 4, zero history
//   gdn_prep       beta = sigmoid(b); g = -exp(A_log) * softplus(a + dt_bias); q, k l2-normed (q / sqrt(dk))
//   gdn_scan       the delta rule, sequential, state (dk x dv) fp32 in SLM, one work-group per v-head
//   gdn_gated_norm o = fp16(w * rms(o) * sigmoid(z)) per v-head
//   gate_mul       o = fp16(o * sigmoid(gate)), the attention output gate (per-head interleaved q/gate layout)
//   router_topk    softmax over E, the top k renormalized (ties to the lower index)
//   silu_mul       m = fp16(silu(g) * u)
//   ple_*          the PLE layer pieces
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>

namespace nn {

using half = sycl::half;

inline float bf16w(uint16_t b) { return sycl::bit_cast<float>((uint32_t) b << 16); }
inline float sigm(float x) { return 1.0f / (1.0f + sycl::exp(-x)); }
inline float silu(float x) { return x * sigm(x); }

// N (T x S x D fp32) = per-stream rms norm of X with weight (1 + w), w bf16 [S*D].
inline sycl::event group_norm1p(sycl::queue& q, const float* X, const uint16_t* w, int64_t T, int S, int64_t D, float eps, float* N)
{
    constexpr int WG = 256;
    return q.parallel_for(sycl::nd_range<1>(T * S * WG, WG), [=](sycl::nd_item<1> it) {
        const int64_t g = it.get_group(0); const int s = (int) (g % S); const int l = (int) it.get_local_id(0);
        const float* x = X + g * D;
        float ss = 0;
        for (int64_t i = l; i < D; i += WG) ss += x[i] * x[i];
        ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
        const float r = sycl::rsqrt(ss / (float) D + eps);
        for (int64_t i = l; i < D; i += WG) N[g * D + i] = x[i] * r * (1.0f + bf16w(w[s * D + i]));
    });
}

// a = silu(a / div) in place (the rank-320 hidden of a site, divided by the stream count)
inline sycl::event silu_scaled(sycl::queue& q, float* a, int64_t n, float div)
{
    return q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { a[i] = silu(a[i] / div); });
}

// y[t, d] = fp16(mean_s sigmoid(Gl[t, s, d]) * N[t, s, d]); Gl: the gate logits (T x S*D)
inline sycl::event hc_mix_out(sycl::queue& q, const float* Gl, const float* N, int64_t T, int S, int64_t D, half* y)
{
    return q.parallel_for(sycl::range<2>(T, D), [=](sycl::id<2> id) {
        const int64_t t = id[0], d = id[1];
        float acc = 0;
        for (int s = 0; s < S; ++s) acc += sigm(Gl[(t * S + s) * D + d]) * N[(t * S + s) * D + d];
        y[t * D + d] = half(acc / (float) S);
    });
}

// inj (T x S) = 2 * sigmoid(logit / S) in place
inline sycl::event hc_inject(sycl::queue& q, float* inj, int64_t T, int S)
{
    return q.parallel_for(sycl::range<1>(T * S), [=](sycl::id<1> i) { inj[i] = 2.0f * sigm(inj[i] / (float) S); });
}

// X[t, s, :] += inj[t, s] * out[t, :], out fp32
inline sycl::event hc_apply(sycl::queue& q, float* X, const float* inj, const float* out, int64_t T, int S, int64_t D)
{
    return q.parallel_for(sycl::range<2>(T * S, D), [=](sycl::id<2> id) {
        const int64_t ts = id[0], t = ts / S, d = id[1];
        X[ts * D + d] += inj[ts] * out[t * D + d];
    });
}

// Per (token, head) of p [T, NH, ld-strided heads of width Dh at offset off within each head slot of width slot]:
// out[t, h, :] = fp16 of RoPE(fp16(rms(v) * (1 + w))) with rotary pairs (j, j + R/2) for j < R/2 at position t.
// w null = no norm (just the fp16 rounding); inv null = no RoPE.
inline sycl::event head_norm1p_rope(sycl::queue& q, const float* p, int64_t T, int64_t NH, int64_t Dh, int64_t slot, int64_t off,
                                    const uint16_t* w, const float* inv, int R, float eps, half* out)
{
    const int WG = 128;
    return q.parallel_for(sycl::nd_range<1>(T * NH * WG, WG), [=](sycl::nd_item<1> it) {
        const int64_t g = it.get_group(0), t = g / NH, h = g % NH;
        const int l = (int) it.get_local_id(0);
        const float* src = p + (t * NH + h) * slot + off;
        float ss = 0;
        for (int64_t i = l; i < Dh; i += WG) { const float v = (float) half(src[i]); ss += v * v; }
        ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
        const float r = w ? sycl::rsqrt(ss / (float) Dh + eps) : 1.0f;
        half* dst = out + (t * NH + h) * Dh;
        for (int64_t i = l; i < Dh; i += WG)
        {
            const float v = (float) half(src[i]);
            float n = w ? (float) half(v * r * (1.0f + bf16w(w[i]))) : v;
            if (inv && i < R)
            {
                const int64_t j = i < R / 2 ? i : i - R / 2;
                const float o = w ? (float) half((float) half(src[i < R / 2 ? i + R / 2 : i - R / 2]) * r *
                                                 (1.0f + bf16w(w[i < R / 2 ? i + R / 2 : i - R / 2])))
                                  : (float) half(src[i < R / 2 ? i + R / 2 : i - R / 2]);
                const float ang = (float) t * inv[j], c = sycl::cos(ang), sn = sycl::sin(ang);
                n = i < R / 2 ? n * c - o * sn : n * c + o * sn;
            }
            dst[i] = half(n);
        }
    });
}

// o (T x NH x Dh, fp32 from the attention) -> fp16(o * sigmoid(gate)), gate read from the interleaved q projection
// qg [T, NH, 2*Dh] at offset Dh.
inline sycl::event gate_mul(sycl::queue& q, const float* o, const float* qg, int64_t T, int64_t NH, int64_t Dh, half* out)
{
    return q.parallel_for(sycl::range<1>(T * NH * Dh), [=](sycl::id<1> id) {
        const int64_t i = id[0], th = i / Dh, d = i % Dh;
        out[i] = half(o[i] * sigm((float) half(qg[th * 2 * Dh + Dh + d])));
    });
}

// Causal depthwise conv (kernel KS) + SiLU over channels of x (T x C fp32; x rounded to fp16 first, like the fp16
// projection output exllamav3 feeds its conv). w: bf16 [C, KS].
inline sycl::event gdn_conv(sycl::queue& q, const float* x, const uint16_t* w, int64_t T, int64_t C, int KS, float* y)
{
    return q.parallel_for(sycl::range<2>(T, C), [=](sycl::id<2> id) {
        const int64_t t = id[0], c = id[1];
        float acc = 0;
        for (int j = 0; j < KS; ++j)
        {
            const int64_t tt = t - (KS - 1) + j;
            if (tt >= 0) acc += bf16w(w[c * KS + j]) * (float) half(x[tt * C + c]);
        }
        y[t * C + c] = silu(acc);
    });
}

// beta = sigmoid(b), g = -exp(A_log) * softplus(a + dt_bias) (softplus(x) = x above 20), per (t, v-head).
inline sycl::event gdn_gates(sycl::queue& q, const float* a, const float* b, const uint16_t* A_log, const uint16_t* dt_bias,
                             int64_t T, int64_t HV, float* g, float* beta)
{
    return q.parallel_for(sycl::range<2>(T, HV), [=](sycl::id<2> id) {
        const int64_t t = id[0], h = id[1];
        const float x = a[t * HV + h] + bf16w(dt_bias[h]);
        const float sp = x > 20.0f ? x : sycl::log1p(sycl::exp(x));
        g[t * HV + h] = -sycl::exp(bf16w(A_log[h])) * sp;
        beta[t * HV + h] = sigm(b[t * HV + h]);
    });
}

// l2-normalize heads of width dk in place (eps 1e-6 inside the root), then multiply by mul.
inline sycl::event l2norm_heads(sycl::queue& q, float* x, int64_t rows, int64_t stride, int64_t heads, int64_t dk, float mul)
{
    const int WG = 128;
    return q.parallel_for(sycl::nd_range<1>(rows * heads * WG, WG), [=](sycl::nd_item<1> it) {
        const int64_t g = it.get_group(0), r = g / heads, h = g % heads; const int l = (int) it.get_local_id(0);
        float* v = x + r * stride + h * dk;
        float ss = 0;
        for (int64_t i = l; i < dk; i += WG) ss += v[i] * v[i];
        ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
        const float f = mul / sycl::sqrt(ss + 1e-6f);
        for (int64_t i = l; i < dk; i += WG) v[i] *= f;
    });
}

// The gated delta rule, sequential over T. One work-group per v-head (HV of them), DK == DV == 128 work-items: item j
// owns column j of the state St (DK x DV, fp32, in SLM). Per step, with k/q of k-head h / ratio:
//   St = exp(g) St;  kv[j] = sum_i St[i, j] k[i];  d[j] = beta (v[j] - kv[j]);  St[i, j] += k[i] d[j];  o[j] = sum_i St[i, j] q[i]
// qkv: the conv output (T x C) with q at column 0, k at KD, v at 2 KD (q already l2-normed and scaled, k l2-normed).
inline sycl::event gdn_scan(sycl::queue& q, const float* qkv, int64_t T, int64_t C, int64_t HK, int64_t HV, int64_t DK,
                            int64_t DV, const float* g, const float* beta, float* o)
{
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> St(sycl::range<1>(DK * DV), h);
        sycl::local_accessor<float, 1> kq(sycl::range<1>(2 * DK), h);
        const int64_t KD = HK * DK, ratio = HV / HK;
        h.parallel_for(sycl::nd_range<1>(HV * DV, DV), [=](sycl::nd_item<1> it) {
            const int64_t hv = it.get_group(0), hk = hv / ratio;
            const int j = (int) it.get_local_id(0);
            for (int64_t i = 0; i < DK; ++i) St[i * DV + j] = 0.0f;
            for (int64_t t = 0; t < T; ++t)
            {
                const float* row = qkv + t * C;
                for (int64_t i = j; i < DK; i += DV) { kq[i] = row[KD + hk * DK + i]; kq[DK + i] = row[hk * DK + i]; }
                sycl::group_barrier(it.get_group());
                const float decay = sycl::exp(g[t * HV + hv]), bt = beta[t * HV + hv];
                const float vj = row[2 * KD + hv * DV + j];
                float kv = 0;
                for (int64_t i = 0; i < DK; ++i) { const float s = St[i * DV + j] * decay; St[i * DV + j] = s; kv += s * kq[i]; }
                const float d = bt * (vj - kv);
                float oj = 0;
                for (int64_t i = 0; i < DK; ++i) { const float s = St[i * DV + j] + kq[i] * d; St[i * DV + j] = s; oj += s * kq[DK + i]; }
                o[(t * HV + hv) * DV + j] = oj;
                sycl::group_barrier(it.get_group());
            }
        });
    });
}

// out (T x HV x DV fp16) = w * rms(o) * sigmoid(z) per v-head (w used as-is, norm before gate).
inline sycl::event gdn_gated_norm(sycl::queue& q, const float* o, const float* z, const uint16_t* w, int64_t T, int64_t HV,
                                  int64_t DV, float eps, half* out)
{
    const int WG = 128;
    return q.parallel_for(sycl::nd_range<1>(T * HV * WG, WG), [=](sycl::nd_item<1> it) {
        const int64_t g = it.get_group(0); const int l = (int) it.get_local_id(0);
        const float* v = o + g * DV;
        float ss = 0;
        for (int64_t i = l; i < DV; i += WG) ss += v[i] * v[i];
        ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
        const float r = sycl::rsqrt(ss / (float) DV + eps);
        for (int64_t i = l; i < DV; i += WG) out[g * DV + i] = half(bf16w(w[i]) * v[i] * r * sigm(z[g * DV + i]));
    });
}

// Router: logits (T x E fp32, from fp16 weights) -> top-k ids and renormalized weights (softmax over the k chosen, which
// equals renormalizing the top k of the full softmax). Ties go to the lower expert index. One work-group per token.
inline sycl::event router_topk(sycl::queue& q, const float* logits, int64_t T, int E, int k, int32_t* ids, float* wts)
{
    return q.parallel_for(sycl::range<1>(T), [=](sycl::id<1> id) {
        const int64_t t = id[0];
        const float* x = logits + t * E;
        int sel[16]; float val[16];
        for (int s = 0; s < k; ++s)
        {
            int best = -1; float bv = -INFINITY;
            for (int e = 0; e < E; ++e)
            {
                bool used = false;
                for (int u = 0; u < s; ++u) used |= sel[u] == e;
                if (!used && (best < 0 || x[e] > bv)) { best = e; bv = x[e]; }
            }
            sel[s] = best; val[s] = bv;
        }
        float m = val[0], z = 0;
        for (int s = 0; s < k; ++s) z += sycl::exp(val[s] - m);
        for (int s = 0; s < k; ++s) { ids[t * k + s] = sel[s]; wts[t * k + s] = sycl::exp(val[s] - m) / z; }
    });
}

// m = fp16(silu(g) * u), with g, u read as fp16 (the projection outputs), from a gate/up block gu (rows x 2F: gate
// columns 0..F-1, up F..2F-1) with row stride ld.
inline sycl::event silu_mul(sycl::queue& q, const float* gu, int64_t rows, int64_t F, int64_t ld, half* m, int64_t ldm)
{
    return q.parallel_for(sycl::range<2>(rows, F), [=](sycl::id<2> id) {
        const int64_t r = id[0], f = id[1];
        const float g = (float) half(gu[r * ld + f]), u = (float) half(gu[r * ld + F + f]);
        m[r * ldm + f] = half(silu(g) * u);
    });
}

} // namespace nn
