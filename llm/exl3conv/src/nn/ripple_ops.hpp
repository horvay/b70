// exl3conv: forward-pass kernels only Ripple-65B needs (arch/ripple.hpp), at the rounding points of the engine's kernels
// (engines/glm.cpp/src/rp_ops.cpp):
//
//   embed_row      x[t, :] = fp32(bf16(bf16(E[id[t], :]) * mult)), E on the card, mult already a bf16 value
//   lerp_embed     y[t, :] = fp16(a + w * (h - a)), a the row's embedding as above (v_proj's input), never stored
//   gate_mul       o = fp16(fp16(sigmoid(fp16 g)) * fp16 o)
//   silu_mul       m = fp16(fp16(silu(min(fp16 g, c))) * clamp(fp16 u, +-c))
//   to_half_clamp  fp16, inf-free (+-65504): what the norm after o_proj reads
//   mirror_lower   the Hessian's upper block triangle from its lower (accumulate only computes the lower one)
#pragma once

#include "nn/ops.hpp"

namespace nn {

inline sycl::event embed_row(sycl::queue& q, const uint16_t* E, const int32_t* ids, int64_t H, float mult, int64_t T, float* x)
{
    return q.parallel_for(sycl::range<2>(T, H), [=](sycl::id<2> id) {
        const int64_t t = id[0], i = id[1];
        x[t * H + i] = bf16f(f2bf16(bf16f(E[(int64_t) ids[t] * H + i]) * mult));
    });
}

inline sycl::event lerp_embed(sycl::queue& q, const uint16_t* E, const int32_t* ids, int64_t H, float mult, float w, const half* h, int64_t T, half* y)
{
    return q.parallel_for(sycl::range<2>(T, H), [=](sycl::id<2> id) {
        const int64_t t = id[0], i = id[1];
        const float a = bf16f(f2bf16(bf16f(E[(int64_t) ids[t] * H + i]) * mult));
        y[t * H + i] = half(a + w * ((float) h[t * H + i] - a));
    });
}

inline sycl::event gate_mul(sycl::queue& q, const float* o, const float* g, int64_t n, half* out)
{
    return q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        const float s = 1.0f / (1.0f + sycl::exp(-(float) half(g[i])));
        out[i] = half((float) half(s) * (float) half(o[i]));
    });
}

inline sycl::event silu_mul(sycl::queue& q, const float* g, const float* u, int64_t n, float c, half* m)
{
    return q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        const float z = sycl::fmin((float) half(g[i]), c);
        const float v = sycl::fmin(sycl::fmax((float) half(u[i]), -c), c);
        const float si = z / (1.0f + sycl::exp(-z));
        m[i] = half((float) half(si) * v);
    });
}

// Output sensitivity of Ripple's MLP, for --out-scales auto (nn::gelu_slopes_acc's counterpart): per intermediate channel j, over
// the T tokens, su[j] += sum (da/du)^2 and sg[j] += sum (da/dg)^2 for a = silu(min(g, c)) * clamp(u, +-c). A clamped input has slope 0.
inline sycl::event silu_slopes_acc(sycl::queue& q, const float* g, const float* u, int64_t T, int64_t I, float c, double* su, double* sg)
{
    return q.parallel_for(sycl::range<1>(I), [=](sycl::id<1> id) {
        const int64_t j = id[0];
        float au = 0.0f, ag = 0.0f;
        for (int64_t t = 0; t < T; ++t)
        {
            const float z0 = (float) half(g[t * I + j]), v0 = (float) half(u[t * I + j]);
            const float z = sycl::fmin(z0, c), v = sycl::fmin(sycl::fmax(v0, -c), c);
            const float s = 1.0f / (1.0f + sycl::exp(-z));
            float du = sycl::fabs(v0) < c ? z * s : 0.0f;
            float dg = z0 < c ? s * (1.0f + z * (1.0f - s)) * v : 0.0f;
            if (!sycl::isfinite(du)) du = 0.0f;
            if (!sycl::isfinite(dg)) dg = 0.0f;
            au += du * du; ag += dg * dg;
        }
        su[j] += (double) au; sg[j] += (double) ag;
    });
}

inline sycl::event to_half_clamp(sycl::queue& q, const float* a, int64_t n, half* b)
{
    return q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { b[i] = half(sycl::fmin(sycl::fmax(a[i], -65504.0f), 65504.0f)); });
}

// H[r, c] = H[c, r] above the diagonal, except inside the diagonal blocks of bs (those are computed whole).
inline sycl::event mirror_lower(sycl::queue& q, float* H, int64_t k, int64_t bs)
{
    return q.parallel_for(sycl::range<2>(k, k), [=](sycl::id<2> id) {
        const int64_t r = id[0], c = id[1];
        if (r / bs < c / bs) H[r * k + c] = H[c * k + r];
    });
}

} // namespace nn
