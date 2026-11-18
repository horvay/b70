// exl3conv: the kernels only Qwen3.5 / Qwen3.8 dense (qwen3_5) needs; the rest of its forward is the Gated DeltaNet and
// gated-attention kernels of nn/qwen4exp_ops.hpp and the SiLU MLP of nn/ripple_ops.hpp.
//
//   rmsnorm_w      y = fp16(x * rsqrt(mean(x^2) + eps) * w), w fp32 (the norms' (1 + w) computed once on the host)
//   gdn_norm_swish o = fp16(w * rms(o) * silu(z)) per v-head (output_gate_type "swish": Qwen3_5RMSNormGated)
//   bf16_round     x = fp32(bf16(x)) in place (the embedding rows of a non-BF16 source, as the BF16 checkpoint holds them)
#pragma once

#include "nn/ops.hpp"

namespace nn {

inline sycl::event rmsnorm_w(sycl::queue& q, const float* x, const float* w, int64_t T, int64_t H, float eps, half* y)
{
    constexpr int WG = 256;
    return q.parallel_for(sycl::nd_range<1>(T * WG, WG), [=](sycl::nd_item<1> it) {
        const int64_t t = it.get_group(0); const int l = (int) it.get_local_id(0);
        const float* xr = x + t * H;
        float s = 0.0f;
        for (int64_t i = l; i < H; i += WG) s += xr[i] * xr[i];
        s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
        const float r = sycl::rsqrt(s / (float) H + eps);
        for (int64_t i = l; i < H; i += WG) y[t * H + i] = half(xr[i] * r * w[i]);
    });
}

// o: T x HV x DV fp32 (the delta rule's output), z: the same shape fp32 (in_proj_z), w: bf16 bits [DV].
inline sycl::event gdn_norm_swish(sycl::queue& q, const float* o, const float* z, const uint16_t* w, int64_t T, int64_t HV, int64_t DV, float eps, half* out)
{
    const int WG = 128;
    return q.parallel_for(sycl::nd_range<1>(T * HV * WG, WG), [=](sycl::nd_item<1> it) {
        const int64_t g = it.get_group(0); const int l = (int) it.get_local_id(0);
        const float* v = o + g * DV;
        float ss = 0;
        for (int64_t i = l; i < DV; i += WG) ss += v[i] * v[i];
        ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
        const float r = sycl::rsqrt(ss / (float) DV + eps);
        for (int64_t i = l; i < DV; i += WG)
        {
            const float zz = z[g * DV + i];
            out[g * DV + i] = half(bf16f(w[i]) * v[i] * r * (zz / (1.0f + sycl::exp(-zz))));
        }
    });
}

} // namespace nn
