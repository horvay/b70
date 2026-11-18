#pragma once
// Plain SYCL kernels for the dense parts of a GLM-5.3 block: RMSNorm, the mHC mix / apply / final collapse, clamped
// SwiGLU, fp16 GEMV, the exact bsz-1 router. Math and rounding points follow docs/forward-spec.md (§3, §6).
#include "rt.h"

namespace glm::ops {

// y[r] = (clamp16(x[r]) * w) * rsqrt(mean(x[r]^2) + eps); x fp16 (clamped to +-65504) or fp32, w bf16 (or null)
template <class TX, class TY>
void rms_norm(sycl::queue& q, const TX* x, const bf16* w, TY* y, int R, int D, float eps);

// mHC mix of R rows of 4 streams (R [R,4,4096] fp32, fn [24,16384] as TF (fp16 at decode, fp32 in prefill)):
// post [R,4], comb [R,4,4] (row-major [g][h]), collapsed [R,4096] fp16. part: [R * nc * 25] fp32 scratch.
struct HcSite {
    const void* fn;          // [24, 16384], fp16 or fp32 (fn_f16 says which)
    bool fn_f16;
    const float* base;       // [24]
    const float* scale;      // [3]
};
int hc_mix_chunks(int R);
void hc_mix(sycl::queue& q, const float* streams, const HcSite& s, float* post, float* comb, half* collapsed,
            float* part, int R, float rms_eps = 1e-5f, float hc_eps = 1e-6f, int iters = 20);
// R[r,h,:] = post[r,h] * y[r,:] + sum_g comb[r,g,h] * R[r,g,:]   (in place, fp32; y fp32)
void hc_apply(sycl::queue& q, float* streams, const float* y, const float* post, const float* comb, int R);
// out[r] = mean over the 4 streams (fp32)
void hc_mean(sycl::queue& q, const float* streams, float* out, int R);
// x[r,:] (fp32 [R,4096]) -> streams: 4 copies
void hc_expand(sycl::queue& q, const float* x, float* streams, int R);

// a = fp16(min(silu(g), lim) * clamp(u, -lim, lim)), fp32 math; g, u fp16
void swiglu(sycl::queue& q, const half* g, const half* u, half* a, size_t n, float lim = 10.f);

// y[m, n] = sum_k x[m, k] * W[n, k] for an fp16 weight [N, K] (torch Linear layout); fp32 accumulate.
// TY fp16: rounded once. TY float with round16: rounded to fp16 first (the PyTorch stack's hgemm), else plain fp32.
template <class TY>
void gemv_f16(sycl::queue& q, const half* x, int M, int K, const half* W, int N, TY* y, bool round16 = false);

// Exact bsz-1 router (spec §6.4a): logits fp16 = y @ gate^T, s = sigmoid, key = s + bias_c (fp16, mean-centred),
// top-k sorted, w = s[idx] * rsf / (sum + 1e-20) rounded to fp16. sel int32 [Q, k], w fp32 [Q, k]; logits [Q, E] fp16.
void route_exact(sycl::queue& q, const half* y, int Q, const half* gate, const half* bias_c, int E, int k, float rsf,
                 half* logits, int* sel, float* w);

// Tier router (the PyTorch stack's k_router, spec §6.4b/c): s = sigmoid(lc), key = s + esb (fp32, uncentred);
// exact = top-k(key); sel = top-k(key + bonus); w = s[sel] * rsf / (sum + 1e-20) in fp32; pred = top-w1 of the next MoE
// layer's sigmoid(ln) + esb_n + bonus_n. Top-k as k rounds of argmax, ties to the lower index. lc / ln: fp16 [Q, E]
// logits (ln null: no next layer); bonus / bonus_n null: 0. picks int64: [sel Q*k | exact Q*k | pred Q*w1].
void route_tier(sycl::queue& q, const half* lc, const half* ln, int Q, const float* esb, const float* esb_n,
                const float* bonus, const float* bonus_n, int E, int k, int w1, float rsf, int* sel, float* w,
                int64_t* picks);

// Exact routing from precomputed fp16 logits (route_exact's numerics: fp16 mean-centred bias, top-k ties to the lower
// index, weights summed in pick order and rounded to fp16), parallel over the experts, plus the next MoE layer's
// prediction as route_tier makes it (ln null: none). picks int64: [sel | sel (exact = selected) | pred].
void route_exact2(sycl::queue& q, const half* lc, const half* ln, int Q, const half* bias_c, const float* esb_n, int E,
                  int k, int w1, float rsf, int* sel, float* w, int64_t* picks);

// Picks in another order for moe_split_begin: position j gets pick perm[j] (sel_p, w_p), pair_of[perm[j]] = j.
void permute_picks(sycl::queue& q, const int* sel, const float* w, const int* perm, int n, int* sel_p, float* w_p,
                   int* pair_of);

// f32 <-> f16 helpers
void f32_to_f16(sycl::queue& q, const float* x, half* y, size_t n);
void add_f32(sycl::queue& q, float* y, const float* x, size_t n);     // y += x

}  // namespace glm::ops
