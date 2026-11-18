// exl3conv: device kernels for the calibration forward pass. Numerics follow exllamav3's conversion forward
// (notes/gemma4-forward.md §5): residual stream fp32; norm math fp32 with the output rounded once (to fp16 for a
// linear's input, or added into the fp32 stream); linear inputs fp16 with fp32 accumulation (oneMKL half x half ->
// float GEMM); q/k/v/o/gate/up outputs rounded to fp16, down_proj kept fp32.
//
//   embed          x = fp32(bf16(E[id] * bf16(sqrt(hidden))))               (rows gathered on the host)
//   rmsnorm_h      y_h = fp16(x * rsqrt(mean(x^2) + eps) * w)               (one work-group per token)
//   rmsnorm_add    x += fp32(norm(y) * w), y fp16 or fp32
//   head_norm_rope per (token, head): fp16(norm(v) * w) then RoPE (NeoX layout) in fp32, rounded to fp16
//   softmax_mask   P = fp16(softmax(S)) over the allowed keys (causal, optional sliding window), fp32 math
//   gelu_mul       m = fp16(fp16(gelu_tanh(fp16 g)) * fp16 u)               (the product in fp16, like __hmul2)
//   finite_rows    per token: 1 if every element is finite (rows with inf / NaN are kept out of the Hessians)
//   transpose / casts / dequant_exl3 (packed trellis -> W[in, out] fp16, the weight inference sees)
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>

namespace nn {

using half = sycl::half;
using bf16 = sycl::ext::oneapi::bfloat16;

inline float bf16f(uint16_t b) { return sycl::bit_cast<float>((uint32_t) b << 16); }
inline uint16_t f2bf16(float f)           // round to nearest even (no NaN inputs here)
{
    const uint32_t u = sycl::bit_cast<uint32_t>(f);
    return (uint16_t) ((u + 0x7FFFu + ((u >> 16) & 1)) >> 16);
}

// x[t, :] = fp32(bf16(bf16(e[t, :]) * mult)), e: T x H bf16 bits, mult already a bf16 value.
inline sycl::event embed(sycl::queue& q, const uint16_t* e, int64_t T, int64_t H, float mult, float* x)
{
    return q.parallel_for(sycl::range<1>(T * H), [=](sycl::id<1> i) { x[i] = bf16f(f2bf16(bf16f(e[i]) * mult)); });
}

// y (T x H fp16) = norm(x) * w, x fp32, w bf16 bits.
inline sycl::event rmsnorm_h(sycl::queue& q, const float* x, const uint16_t* w, int64_t T, int64_t H, float eps, half* y)
{
    constexpr int WG = 256;
    return q.parallel_for(sycl::nd_range<1>(T * WG, WG), [=](sycl::nd_item<1> it) {
        const int64_t t = it.get_group(0); const int l = (int) it.get_local_id(0);
        const float* xr = x + t * H;
        float s = 0.0f;
        for (int64_t i = l; i < H; i += WG) s += xr[i] * xr[i];
        s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
        const float r = sycl::rsqrt(s / (float) H + eps);
        for (int64_t i = l; i < H; i += WG) y[t * H + i] = half(xr[i] * r * bf16f(w[i]));
    });
}

// x (T x H fp32) += norm(y) * w, y of type Y (half or float).
template <typename Y>
inline sycl::event rmsnorm_add(sycl::queue& q, const Y* y, const uint16_t* w, int64_t T, int64_t H, float eps, float* x)
{
    constexpr int WG = 256;
    return q.parallel_for(sycl::nd_range<1>(T * WG, WG), [=](sycl::nd_item<1> it) {
        const int64_t t = it.get_group(0); const int l = (int) it.get_local_id(0);
        const Y* yr = y + t * H;
        float s = 0.0f;
        for (int64_t i = l; i < H; i += WG) { const float v = (float) yr[i]; s += v * v; }
        s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
        const float r = sycl::rsqrt(s / (float) H + eps);
        for (int64_t i = l; i < H; i += WG) x[t * H + i] += (float) yr[i] * r * bf16f(w[i]);
    });
}

// Per (token, head) of a [T, NH, D] fp32 projection output p: v = fp16(p); n = fp16(rmsnorm(v) * w) (w null = no
// weight); if inv_freq, rotate pairs (j, j + D/2) by pos * inv_freq[j] in fp32 and round to fp16. out: [T, NH, D].
// D <= 512, one work-group of D/2 work-items per head.
inline sycl::event head_norm_rope(sycl::queue& q, const float* p, int64_t T, int64_t NH, int64_t D, const uint16_t* w,
                                  const float* inv_freq, float eps, half* out)
{
    const int WG = (int) (D / 2);
    return q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<1>(T * NH * WG, WG), [=](sycl::nd_item<1> it) {
            const int64_t g = it.get_group(0), t = g / NH;
            const int j = (int) it.get_local_id(0);
            const float* pr = p + g * D;
            const float a0 = (float) half(pr[j]), b0 = (float) half(pr[j + D / 2]);
            float s = sycl::reduce_over_group(it.get_group(), a0 * a0 + b0 * b0, sycl::plus<float>());
            const float r = sycl::rsqrt(s / (float) D + eps);
            float a = (float) half(a0 * r * (w ? bf16f(w[j]) : 1.0f));
            float b = (float) half(b0 * r * (w ? bf16f(w[j + D / 2]) : 1.0f));
            if (inv_freq)
            {
                const float ang = (float) t * inv_freq[j];
                const float c = sycl::cos(ang), sn = sycl::sin(ang);
                const float na = a * c - b * sn, nb = b * c + a * sn;
                a = na; b = nb;
            }
            out[g * D + j] = half(a);
            out[g * D + j + D / 2] = half(b);
        });
    });
}

// S: [NHQ, T, T] scores (fp32), P: same shape fp16. Key u is allowed for query t iff u <= t and u > t - window
// (window 0 = no window).
inline sycl::event softmax_mask(sycl::queue& q, const float* S, int64_t NHQ, int64_t T, int64_t window, half* P)
{
    constexpr int WG = 256;
    return q.parallel_for(sycl::nd_range<1>(NHQ * T * WG, WG), [=](sycl::nd_item<1> it) {
        const int64_t row = it.get_group(0), t = row % T;
        const int l = (int) it.get_local_id(0);
        const float* sr = S + row * T;
        half* pr = P + row * T;
        const int64_t u0 = window > 0 ? sycl::max<int64_t>(0, t - window + 1) : 0;
        float m = -INFINITY;
        for (int64_t u = u0 + l; u <= t; u += WG) m = sycl::max(m, sr[u]);
        m = sycl::reduce_over_group(it.get_group(), m, sycl::maximum<float>());
        float s = 0.0f;
        for (int64_t u = u0 + l; u <= t; u += WG) s += sycl::exp(sr[u] - m);
        s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
        const float inv = 1.0f / s;
        for (int64_t u = l; u < T; u += WG) pr[u] = (u >= u0 && u <= t) ? half(sycl::exp(sr[u] - m) * inv) : half(0.0f);
    });
}

// m = fp16(fp16(gelu_tanh(fp16(g))) * fp16(u))
inline sycl::event gelu_mul(sycl::queue& q, const float* g, const float* u, int64_t n, half* m)
{
    return q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        const float z = (float) half(g[i]);
        const float ge = 0.5f * z * (1.0f + sycl::tanh(0.7978845608028654f * (z + 0.044715f * z * z * z)));
        m[i] = half(ge) * half(u[i]);
    });
}

// Output sensitivity of a gated GELU MLP (exllamav3 1.6.1 capture_out_sensitivity, "gelu"): per intermediate channel
// j, over the T tokens, su[j] += sum (da/du)^2 and sg[j] += sum (da/dg)^2 for a = gelu_tanh(g) * u, on the fp16
// values the forward uses; non-finite slopes count as zero.
inline sycl::event gelu_slopes_acc(sycl::queue& q, const float* g, const float* u, int64_t T, int64_t I, double* su, double* sg)
{
    return q.parallel_for(sycl::range<1>(I), [=](sycl::id<1> id) {
        const int64_t j = id[0];
        float au = 0.0f, ag = 0.0f;
        for (int64_t t = 0; t < T; ++t)
        {
            const float z = (float) half(g[t * I + j]), v = (float) half(u[t * I + j]);
            const float th = sycl::tanh(0.7978845608028654f * (z + 0.044715f * z * z * z));
            const float f = 0.5f * z * (1.0f + th);
            const float df = 0.5f * (1.0f + th) + 0.5f * z * (1.0f - th * th) * 0.7978845608028654f * (1.0f + 0.134145f * z * z);
            float du = f, dg = df * v;
            if (!sycl::isfinite(du)) du = 0.0f;
            if (!sycl::isfinite(dg)) dg = 0.0f;
            au += du * du; ag += dg * dg;
        }
        su[j] += (double) au; sg[j] += (double) ag;
    });
}

inline sycl::event to_half(sycl::queue& q, const float* a, int64_t n, half* b)
{
    return q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { b[i] = half(a[i]); });
}

inline sycl::event scale_inplace(sycl::queue& q, float* x, int64_t n, float s)
{
    return q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { x[i] *= s; });
}

// ok[t] = 1 if row t (width W) of x is all finite; with zero, also zero the bad rows in place (so they add nothing
// to X^T X).
inline sycl::event finite_rows(sycl::queue& q, half* x, int64_t T, int64_t W, int32_t* ok, bool zero)
{
    constexpr int WG = 256;
    return q.parallel_for(sycl::nd_range<1>(T * WG, WG), [=](sycl::nd_item<1> it) {
        const int64_t t = it.get_group(0); const int l = (int) it.get_local_id(0);
        int bad = 0;
        for (int64_t i = l; i < W; i += WG) bad |= !sycl::isfinite((float) x[t * W + i]);
        bad = sycl::reduce_over_group(it.get_group(), bad, sycl::bit_or<int>());
        if (bad && zero) for (int64_t i = l; i < W; i += WG) x[t * W + i] = half(0.0f);
        if (l == 0) ok[t] = !bad;
    });
}

// bf16 bits -> fp16 (weights: BF16 checkpoint values through fp16, as exllamav3 loads them)
inline sycl::event bf16_to_half(sycl::queue& q, const uint16_t* a, int64_t n, half* b)
{
    return q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { b[i] = half(bf16f(a[i])); });
}

// out[c, r] = (float) in[r, c] for an R x C fp16 matrix (HF [out, in] -> quantizer (in, out) fp32).
inline sycl::event transpose_h2f(sycl::queue& q, const half* in, int64_t R, int64_t C, float* out)
{
    constexpr int B = 16;
    return q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 2> tile(sycl::range<2>(B, B + 1), h);
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>((R + B - 1) / B * B, (C + B - 1) / B * B), sycl::range<2>(B, B)),
                       [=](sycl::nd_item<2> it) {
            const int64_t r = it.get_global_id(0), c = it.get_global_id(1);
            const int lr = (int) it.get_local_id(0), lc = (int) it.get_local_id(1);
            if (r < R && c < C) tile[lr][lc] = (float) in[r * C + c];
            sycl::group_barrier(it.get_group());
            const int64_t orow = it.get_group(1) * B + lr, ocol = it.get_group(0) * B + lc;   // out is C x R
            if (orow < C && ocol < R) out[orow * R + ocol] = tile[lc][lr];
        });
    });
}

} // namespace nn

namespace nn {

// Per token t of a chunk: reference and test logits (Tc x V fp32, raw), both softcapped c * tanh(x / c) (c = 0: off).
// out[t] = {KL(ref || test), ref argmax == test argmax, -log p_test(target), -log p_ref(target)}; target < 0: skip NLL.
inline sycl::event logit_stats(sycl::queue& q, const float* Lr, const float* Lt, int64_t Tc, int64_t V, float cap,
                               const int32_t* target, float* out)
{
    constexpr int WG = 1024;
    return q.parallel_for(sycl::nd_range<1>(Tc * WG, WG), [=](sycl::nd_item<1> it) {
        const int64_t t = it.get_group(0); const int l = (int) it.get_local_id(0);
        auto g = it.get_group();
        const float* r = Lr + t * V; const float* x = Lt + t * V;
        auto sc = [&](float v) { return cap > 0 ? cap * sycl::tanh(v / cap) : v; };
        float mr = -INFINITY, mt = -INFINITY; int64_t ar = 0, at = 0;
        for (int64_t i = l; i < V; i += WG)
        {
            const float a = sc(r[i]), b = sc(x[i]);
            if (a > mr) { mr = a; ar = i; }
            if (b > mt) { mt = b; at = i; }
        }
        // argmax: the max value, ties to the lowest index
        const float gmr = sycl::reduce_over_group(g, mr, sycl::maximum<float>());
        const float gmt = sycl::reduce_over_group(g, mt, sycl::maximum<float>());
        const int64_t gar = sycl::reduce_over_group(g, mr == gmr ? ar : V, sycl::minimum<int64_t>());
        const int64_t gat = sycl::reduce_over_group(g, mt == gmt ? at : V, sycl::minimum<int64_t>());
        float zr = 0, zt = 0;
        for (int64_t i = l; i < V; i += WG) { zr += sycl::exp(sc(r[i]) - gmr); zt += sycl::exp(sc(x[i]) - gmt); }
        zr = sycl::reduce_over_group(g, zr, sycl::plus<float>());
        zt = sycl::reduce_over_group(g, zt, sycl::plus<float>());
        const float lzr = sycl::log(zr), lzt = sycl::log(zt);
        float kl = 0;
        for (int64_t i = l; i < V; i += WG)
        {
            const float lr = sc(r[i]) - gmr - lzr, lt = sc(x[i]) - gmt - lzt;
            kl += sycl::exp(lr) * (lr - lt);
        }
        kl = sycl::reduce_over_group(g, kl, sycl::plus<float>());
        if (l == 0)
        {
            out[t * 4 + 0] = kl;
            out[t * 4 + 1] = gar == gat ? 1.0f : 0.0f;
            const int32_t y = target[t];
            out[t * 4 + 2] = y >= 0 ? -(sc(x[y]) - gmt - lzt) : 0.0f;
            out[t * 4 + 3] = y >= 0 ? -(sc(r[y]) - gmr - lzr) : 0.0f;
        }
    });
}

} // namespace nn
