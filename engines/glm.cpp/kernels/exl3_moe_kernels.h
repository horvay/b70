#pragma once
// Routed-expert (MoE) ESIMD kernels over pointer tables: GateUp / Down GEMV (decode, grouped small M), the DPAS
// prefill kernels and the combine. Extracted verbatim from llm/glm/xpu/csrc/exl3_moe.n128.sycl (the torch-free part);
// the GLM-5.3 swiglu clamp (EXL3_ACT_LIMIT 10) is fixed here.
#include "exl3_esimd.h"
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#define EXL3_ACT_LIMIT 10.0f
// glm-b70: exllamav3 convention (activation_kernels.cuh): a = min(silu(g), L) * clamp(u, -L, L), i.e. the gate is
// clamped AFTER SiLU and from above only (sglang's swiglu_clamped clamps g before SiLU).
#define EXL3_ACT_CLAMP(g, u) do { } while (0)
// lim: the kernel's act_lim field (GLM-5.3: EXL3_ACT_LIMIT; 0 = plain SwiGLU, Qwen3.8-Flash-Next)
#define EXL3_SILU_CLAMPED_DECL(act, g, u) simd<float, 128> act = g / (1.0f + exp(-g)); \
    if (act_lim > 0.0f) { act.merge(simd<float, 128>(act_lim), act > act_lim); \
        u.merge(simd<float, 128>(act_lim), u > act_lim); \
        u.merge(simd<float, 128>(-act_lim), u < -act_lim); } \
    act = act * u


using namespace exl3;
using bf16 = sycl::ext::oneapi::bfloat16;

namespace glm_moe {
using namespace exl3;
using bf16 = sycl::ext::oneapi::bfloat16;
struct Geom {
    int H, I, K;
    int W() const { return exl3::words_of(K); }      // K: template code (exl3_esimd.h Geo)
    size_t gu_bytes() const { return (size_t)(H / 16) * (2 * I / 16) * W() * 4; }
    size_t d_bytes() const { return (size_t)(I / 16) * (H / 16) * W() * 4; }
    size_t off_suh_g() const { return gu_bytes() + d_bytes(); }
    size_t off_suh_u() const { return off_suh_g() + (size_t)H * 2; }
    size_t off_svh_g() const { return off_suh_u() + (size_t)H * 2; }
    size_t off_svh_u() const { return off_svh_g() + (size_t)I * 2; }
    size_t off_suh_d() const { return off_svh_u() + (size_t)I * 2; }
    size_t off_svh_d() const { return off_suh_d() + (size_t)I * 2; }
    size_t blob() const { return off_svh_d() + (size_t)H * 2; }
};

struct Route {
    const int* act_expert; const int* act_start; const int* act_cnt; const int* n_act;
    const int* act_pstart;          // padded (to MB) row start per active expert (prefill)
    const int* pair_tok; const float* pair_w;
    // pair mode (small M, no align launch): work item a = pair m*topk + k, expert act_expert[a] = topk_ids (flat),
    // one row each, pair_w = topk_w (flat). A duplicated expert is read once per pair (rare at M <= 4).
    int pair_topk = 0;              // > 0: pair mode
    int npairs = 0;
    // device expert cache: fill[e] != 0 -> expert e is read from its (host) pointer and written through to the
    // device slot at address fill[e] by the GEMV kernels (combine then repoints ptrs[e] to it)
    const int64_t* fill = nullptr;
    ESIMD_INLINE void get(int a, int& e, int& s0, int& cnt) const {
        if (pair_topk) { e = act_expert[a]; s0 = a; cnt = 1; }
        else { e = act_expert[a]; s0 = act_start[a]; cnt = act_cnt[a]; }
    }
    ESIMD_INLINE int nact() const { return pair_topk ? npairs : n_act[0]; }
    ESIMD_INLINE int tok(int s) const { return pair_topk ? s / pair_topk : pair_tok[s]; }
};

// Prefetch n_rows tile-row strips (NW dwords each, row stride in dwords) into the cache hierarchy, no registers.
// L1C: also allocate in L1 (else L2 only).
#ifndef EXL3_MOE_PF
#define EXL3_MOE_PF 0   // REJECTED on B70: 1 (L2) M=1 82 -> 128 us, 2 (L1+L2) -> 130 us
#endif
template <int NW, bool L1C>
ESIMD_INLINE void pf_rows(const uint32_t* p, size_t stride, int n_rows) {
    namespace xe = sycl::ext::intel::experimental::esimd;
    constexpr auto h1 = L1C ? xe::cache_hint::cached : xe::cache_hint::uncached;
    for (int r = 0; r < n_rows; ++r) {
#pragma unroll
        for (int c = 0; c < NW / 32; ++c)
            xe::lsc_prefetch<uint32_t, 32, xe::lsc_data_size::default_size, h1, xe::cache_hint::cached>(p + r * stride + 32 * c);
    }
}

// prev plane 0 of NT pre-planarized tiles = plane D-1 rotated by one period within each tile.
template <int K, int NT>
ESIMD_INLINE void planar_prev(simd<uint32_t, NT * Geo<K>::WORDS>& words, simd<uint32_t, NT * Geo<K>::WORDS>& prev);

// Load one tile-row strip of NT pre-planarized tiles and derive prev.
template <int K, int NT>
ESIMD_INLINE void load_planar(const uint32_t* p, simd<uint32_t, NT * Geo<K>::WORDS>& words,
                              simd<uint32_t, NT * Geo<K>::WORDS>& prev) {
    words = block_load<uint32_t, NT * Geo<K>::WORDS>(p);
    planar_prev<K, NT>(words, prev);
}

template <int K, int NT>
ESIMD_INLINE void planar_prev(simd<uint32_t, NT * Geo<K>::WORDS>& words, simd<uint32_t, NT * Geo<K>::WORDS>& prev) {
    constexpr int D = Geo<K>::D, G = Geo<K>::G, W = Geo<K>::WORDS;
    if constexpr (D > 1) {
        static_assert(NT * G == 32 || NT * G == 16, "planar group = 32 lanes (16 for half-integer K)");
        auto pm = prev.template bit_cast_view<uint32_t, NT * W / G, G>();      // rows of G lanes; row t = tile t of plane 0
        auto wm = words.template bit_cast_view<uint32_t, NT * W / G, G>();
        pm.template select<NT, 1, G - 1, 1>(0, 1) = wm.template select<NT, 1, G - 1, 1>((D - 1) * NT, 0);
        pm.template select<NT, 1, 1, 1>(0, 0) = wm.template select<NT, 1, 1, 1>((D - 1) * NT, G - 1);
    } else {
        prev.template select<NT * W - 1, 1>(1) = words.template select<NT * W - 1, 1>(0);
#pragma unroll
        for (int j = 0; j < NT; ++j) prev[j * W] = words[j * W + W - 1];
    }
}

// ------------------------------------------------------------------------------------------------
// gate|up (+ SiLU + down input Hadamard). Threads: T = 4 * P  (proj x 2 strips of NT=4 tiles x P k-splits).
template <int K, int MR, int P, typename TIn>
struct GateUpKernel {
    const TIn* x; int x_stride;
    const int64_t* ptrs;
    Route rt;
    fp16* abuf;                     // [npairs, I]
    int H, I;
    size_t off_suh_g, off_svh_g, off_suh_d;
    int tail_dw = 0;                // dwords from off_suh_g to the blob end (vector tail, write-through copy)
    float act_lim = EXL3_ACT_LIMIT; // SwiGLU clamp (0: none)

    static constexpr int NT = 4;
    static constexpr int W = Geo<K>::WORDS, D = Geo<K>::D;
    using GK = GemvKernel<K, 2, MR, NT>;
    static constexpr int ACC = GK::ACC, NP = GK::NP, GV = GK::GV, G = GK::G, TP = GK::TP;
    static constexpr int SLM = P * MR * 256 * 4;

    void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
        slm_init<SLM>();
        int t = it.get_local_id(0);
        int nbn = I / 128;
        int wg = it.get_group(0);
        int a = wg / nbn, nb = wg % nbn;
        if (a >= rt.nact()) return;
        int e, s0, cnt;
        rt.get(a, e, s0, cnt);
        const uint8_t* base = reinterpret_cast<const uint8_t*>(ptrs[e]);
        const uint32_t* tr = reinterpret_cast<const uint32_t*>(base);
        int proj = t & 1, strip = (t >> 1) & 1, p = t >> 2;
        int tiles_n = 2 * I / 16;
        int tile_n0 = proj * (I / 16) + nb * 8 + strip * NT;
        const fp16* suh = reinterpret_cast<const fp16*>(base + off_suh_g) + (size_t)proj * H;
        int tk = H / 16, rps = tk / P, r0 = p * rps;
        if constexpr (EXL3_MOE_PF) pf_rows<NT * W, EXL3_MOE_PF == 2>(tr + ((size_t)r0 * tiles_n + tile_n0) * W, (size_t)tiles_n * W, rps);
        uint32_t* dst = rt.fill ? reinterpret_cast<uint32_t*>(rt.fill[e]) : nullptr;
        if (dst && nb == 0 && t == 0) {
            const uint32_t* src = reinterpret_cast<const uint32_t*>(base + off_suh_g);
            uint32_t* d2 = dst + off_suh_g / 4;
            int i = 0;
            for (; i + 64 <= tail_dw; i += 64) block_store<uint32_t, 64>(d2 + i, block_load<uint32_t, 64>(src + i));
            for (; i < tail_dw; ++i) d2[i] = src[i];
        }

        for (int c0 = 0; c0 < cnt; c0 += MR) {
            int tok[MR];
#pragma unroll
            for (int m = 0; m < MR; ++m) tok[m] = (c0 + m < cnt) ? rt.tok(s0 + c0 + m) : -1;
            simd<float, ACC> acc = 0.0f;
            for (int rb = r0; rb < r0 + rps; rb += 8) {
                int kb = rb / 8;
#ifdef EXL3_MOE_BATCHLOAD
                // issue all 8 tile-rows' loads first (memory-level parallelism), then the input Hadamard
                simd<uint32_t, 8 * NT * W> wb;
#pragma unroll
                for (int i = 0; i < 8; ++i)
                    wb.template select<NT * W, 1>(i * NT * W) =
                        block_load<uint32_t, NT * W>(tr + ((size_t)(rb + i) * tiles_n + tile_n0) * W);
#endif
                simd<fp16, MR * 128> xblk;
                simd<float, 128> su = convert<float>(block_load<fp16, 128>(suh + kb * 128));
#pragma unroll
                for (int m = 0; m < MR; ++m) {
                    if (tok[m] >= 0) {
#ifdef EXL3_MOE_PROBE_NOHAD
                        xblk.template select<128, 1>(m * 128) = convert<fp16>(block_load<TIn, 128>(x + (size_t)tok[m] * x_stride + kb * 128));
                        continue;
#endif
                        simd<float, 128> v = convert<float>(block_load<TIn, 128>(x + (size_t)tok[m] * x_stride + kb * 128)) * su;
                        v = convert<float>(convert<fp16>(v));
                        fwht128(v);
                        xblk.template select<128, 1>(m * 128) = convert<fp16>(v * kRsqrt128);
                    } else {
                        xblk.template select<128, 1>(m * 128) = 0;
                    }
                }
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    simd<uint32_t, NT * W> words, prev;
#ifdef EXL3_MOE_BATCHLOAD
                    words = wb.template select<NT * W, 1>(i * NT * W);
                    prev.template select<NT * W - 1, 1>(1) = words.template select<NT * W - 1, 1>(0);
#pragma unroll
                    for (int j = 0; j < NT; ++j) prev[j * W] = words[j * W + W - 1];
#else
                    size_t t0 = (size_t)(rb + i) * tiles_n + tile_n0;
                    load_planar<K, NT>(tr + t0 * W, words, prev);
                    if (dst && c0 == 0) block_store<uint32_t, NT * W>(dst + t0 * W, words);
#endif
                    simd<fp16, MR * 16> xr;
#pragma unroll
                    for (int m = 0; m < MR; ++m) xr.template select<16, 1>(m * 16) = xblk.template select<16, 1>(m * 128 + i * 16);
                    simd<fp16, ACC> hacc = 0;
#ifdef EXL3_MOE_PROBE_LOADONLY
                    acc.template select<32, 1>(0) += convert<float>(simd<uint32_t, 32>(words.template select<32, 1>(0)) ^ simd<uint32_t, 32>(prev.template select<32, 1>(32)));
                    acc.template select<16, 1>(0) += convert<float>(simd<fp16, 16>(xr.template select<16, 1>(0)));
#else
                    GK::template step<0>(words, prev, xr, hacc, nullptr);
                    acc += convert<float>(hacc);
#endif
                }
            }
            // lanes -> columns; SLM part[p][m][proj*128 + strip*64 + j*16 + c]
#pragma unroll
            for (int m = 0; m < MR; ++m) {
                simd<float, NT * 16> o;
#pragma unroll
                for (int j = 0; j < NT; ++j)
#pragma unroll
                    for (int h = 0; h < 2; ++h) o.template select<8, 1>(j * 16 + h * 8) = GK::cols(acc, m, j, h);
                slm_block_store<float, NT * 16>(((p * MR + m) * 256 + proj * 128 + strip * 64) * 4, o);
            }
            barrier();
            if (t < MR && tok[t] >= 0) {
                int m = t;
                simd<float, 128> g = 0.0f, u = 0.0f;
                for (int pp = 0; pp < P; ++pp) {
                    g += slm_block_load<float, 128>(((pp * MR + m) * 256) * 4);
                    u += slm_block_load<float, 128>(((pp * MR + m) * 256 + 128) * 4);
                }
                fwht128(g); fwht128(u);
                const fp16* svg = reinterpret_cast<const fp16*>(base + off_svh_g);
                g = convert<float>(convert<fp16>(g * kRsqrt128 * convert<float>(block_load<fp16, 128>(svg + nb * 128))));
                u = convert<float>(convert<fp16>(u * kRsqrt128 * convert<float>(block_load<fp16, 128>(svg + I + nb * 128))));
                EXL3_ACT_CLAMP(g, u);
                EXL3_SILU_CLAMPED_DECL(act, g, u);
                act = convert<float>(convert<fp16>(act));
                const fp16* sud = reinterpret_cast<const fp16*>(base + off_suh_d);
                act = convert<float>(convert<fp16>(act * convert<float>(block_load<fp16, 128>(sud + nb * 128))));
                fwht128(act);
                block_store<fp16, 128>(abuf + (size_t)(s0 + c0 + m) * I + nb * 128, convert<fp16>(act * kRsqrt128));
            }
            barrier();
        }
    }
};

// ------------------------------------------------------------------------------------------------
// down: T = 2 * P (2 strips of NT=4 tiles x P k-splits); WG per (active expert, 128-col block of H).
template <int K, int MR, int P>
struct DownKernel {
    const fp16* abuf;
    const int64_t* ptrs;
    Route rt;
    float* y;                       // [npairs, H]
    int H, I;
    size_t off_d, off_svh_d;

    static constexpr int NT = 4;
    static constexpr int W = Geo<K>::WORDS, D = Geo<K>::D;
    using GK = GemvKernel<K, 2, MR, NT>;
    static constexpr int ACC = GK::ACC, NP = GK::NP, GV = GK::GV, G = GK::G, TP = GK::TP;
    static constexpr int SLM = P * MR * 128 * 4;
    static constexpr int RB = (40 / P) < 8 ? (40 / P) : 8;   // tile-rows per batched load (I/16 = 40 for I=640)

    void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
        slm_init<SLM>();
        int t = it.get_local_id(0);
        int nbn = H / 128;
        int wg = it.get_group(0);
        int a = wg / nbn, nb = wg % nbn;
        if (a >= rt.nact()) return;
        int e, s0, cnt;
        rt.get(a, e, s0, cnt);
        const uint8_t* base = reinterpret_cast<const uint8_t*>(ptrs[e]);
        const uint32_t* tr = reinterpret_cast<const uint32_t*>(base + off_d);
        int strip = t & 1, p = t >> 1;
        int tiles_n = H / 16;
        int tile_n0 = nb * 8 + strip * NT;
        int tk = I / 16, rps = tk / P, r0 = p * rps;
        if constexpr (EXL3_MOE_PF) pf_rows<NT * W, EXL3_MOE_PF == 2>(tr + ((size_t)r0 * tiles_n + tile_n0) * W, (size_t)tiles_n * W, rps);
        uint32_t* dst = rt.fill ? reinterpret_cast<uint32_t*>(rt.fill[e]) : nullptr;
        if (dst) dst += off_d / 4;

        for (int c0 = 0; c0 < cnt; c0 += MR) {
            bool valid[MR];
#pragma unroll
            for (int m = 0; m < MR; ++m) valid[m] = c0 + m < cnt;
            simd<float, ACC> acc = 0.0f;
            for (int rb = r0; rb < r0 + rps; rb += RB) {
#ifdef EXL3_MOE_BATCHLOAD
            simd<uint32_t, RB * NT * W> wb;
#pragma unroll
            for (int i = 0; i < RB; ++i)
                wb.template select<NT * W, 1>(i * NT * W) = block_load<uint32_t, NT * W>(tr + ((size_t)(rb + i) * tiles_n + tile_n0) * W);
#endif
#pragma unroll
            for (int i = 0; i < RB; ++i) {
                int r = rb + i;
                simd<uint32_t, NT * W> words, prev;
#ifdef EXL3_MOE_BATCHLOAD
                words = wb.template select<NT * W, 1>(i * NT * W);
                prev.template select<NT * W - 1, 1>(1) = words.template select<NT * W - 1, 1>(0);
#pragma unroll
                for (int j = 0; j < NT; ++j) prev[j * W] = words[j * W + W - 1];
#else
                load_planar<K, NT>(tr + ((size_t)r * tiles_n + tile_n0) * W, words, prev);
                if (dst && c0 == 0) block_store<uint32_t, NT * W>(dst + ((size_t)r * tiles_n + tile_n0) * W, words);
#endif
                simd<fp16, MR * 16> xr = 0;
#pragma unroll
                for (int m = 0; m < MR; ++m)
                    if (valid[m]) xr.template select<16, 1>(m * 16) = block_load<fp16, 16>(abuf + (size_t)(s0 + c0 + m) * I + r * 16);
                simd<fp16, ACC> hacc = 0;
#ifdef EXL3_MOE_PROBE_LOADONLY
                acc.template select<32, 1>(0) += convert<float>(simd<uint32_t, 32>(words.template select<32, 1>(0)) ^ simd<uint32_t, 32>(prev.template select<32, 1>(32)));
                acc.template select<16, 1>(0) += convert<float>(simd<fp16, 16>(xr.template select<16, 1>(0)));
#else
                GK::template step<0>(words, prev, xr, hacc, nullptr);
                acc += convert<float>(hacc);
#endif
            }
            }
#pragma unroll
            for (int m = 0; m < MR; ++m) {
                simd<float, NT * 16> o;
#pragma unroll
                for (int j = 0; j < NT; ++j)
#pragma unroll
                    for (int h = 0; h < 2; ++h) o.template select<8, 1>(j * 16 + h * 8) = GK::cols(acc, m, j, h);
                slm_block_store<float, NT * 16>(((p * MR + m) * 128 + strip * 64) * 4, o);
            }
            barrier();
            if (t < MR && valid[t]) {
                int m = t;
                simd<float, 128> v = 0.0f;
                for (int pp = 0; pp < P; ++pp) v += slm_block_load<float, 128>(((pp * MR + m) * 128) * 4);
                fwht128(v);
                const fp16* svd = reinterpret_cast<const fp16*>(base + off_svh_d);
                float w = rt.pair_w[s0 + c0 + m];
                block_store<float, 128>(y + (size_t)(s0 + c0 + m) * H + nb * 128,
                                        v * (kRsqrt128 * w) * convert<float>(block_load<fp16, 128>(svd + nb * 128)));
            }
            barrier();
        }
    }
};

// out[m, nb*128 : +128] = sum_k y[pair_of[m*topk + k], ...]
template <typename TOut, typename TY = float>
struct CombineKernel {
    const TY* y; const int* pair_of; TOut* out;
    int M, H, topk, out_stride;
    int64_t* commit_ptrs = nullptr; int64_t* commit_fill = nullptr; const int* fill_list = nullptr;
    void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
        int id = it.get_global_id(0);
        if (id == 0 && fill_list) {
            // expert cache commit: the filled slots are complete (GEMV kernels ran before this one)
            int nf = fill_list[0];
            for (int i = 0; i < nf; ++i) {
                int e = fill_list[1 + i];
                commit_ptrs[e] = commit_fill[e];
                commit_fill[e] = 0;
            }
        }
        int nbn = H / 128;
        int m = id / nbn, nb = id % nbn;
        if (m >= M) return;
        simd<float, 128> v = 0.0f;
        for (int k = 0; k < topk; ++k)
            v += convert<float>(block_load<TY, 128>(y + (size_t)(pair_of ? pair_of[m * topk + k] : m * topk + k) * H + nb * 128));
        block_store<TOut, 128>(out + (size_t)m * out_stride + nb * 128, convert<TOut>(v));
    }
};


// ------------------------------------------------------------------------------------------------
// PREFILL (DPAS / XMX). Rows of each active expert are split in MB-row blocks (align pads each expert to MB).
// Trellis decode: SIMD32 across one PLANAR4 group of 4 tiles (the GEMV state function), scattered straight into
// four VNNI B operands (K=3: P=1, MS=4 geometry of DpasKernel::build). A operands: MB x 16 fp16 per tile-row.

// Other K (glm-b70, 2.51 bpw: K=2 layers, K=4 MTP layer): the GEMV kernel's value mapping (GemvKernel::step) for
// any Geo<K>: value u of lane grp in a tile sits at input row k = R0(u) + STR * (grp % P) and output column
// n = grp / P + 8 * h(u) (P = 32 / V... 1, 2, 4 for V = 32, 16, 8). TP tiles are decoded per SIMD32 op. B is the
// VNNI form: Bm[k / 2][2 * n + k % 2] per tile.
template <int K, int NT, int u>
ESIMD_INLINE void build4_any(simd<uint32_t, NT * Geo<K>::WORDS>& words, simd<uint32_t, NT * Geo<K>::WORDS>& prev,
                             simd<fp16, 256 * NT>& B) {
    constexpr int G = Geo<K>::G, V = Geo<K>::V, W = Geo<K>::WORDS;
    if constexpr (u < V) {
        constexpr int TP = G >= 32 ? 1 : (32 / G < NT ? 32 / G : NT);
        constexpr int NP = NT / TP;
        constexpr int GV = TP * G;
        static_assert(NT % TP == 0 && (GV == 32 || GV == 16), "build4_any: SIMD32 (16) over TP whole tiles");
        constexpr int P = V == 8 ? 4 : (V == 16 ? 2 : 1);
        constexpr int STR = V == 8 ? 2 : (V == 16 ? 4 : 1);
        constexpr int h = (u >> 2) & 1;
        constexpr int R0 = 8 * ((u >> 1) & 1) + 2 * ((u >> 3) & 3) + (u & 1);
        // V = 64 (half-integer K): lane grp holds columns 2 * grp + (u >> 5), i.e. VNNI column 4 * grp + 2 * (u >> 5)
        constexpr int C0 = 16 * h + (u & 1) + (V == 64 ? 2 * (u >> 5) : 0);
#pragma unroll
        for (int jp = 0; jp < NP; ++jp) {
            simd<uint32_t, W * TP> dw = words.template select<W * TP, 1>(jp * W * TP);
            simd<uint32_t, W * TP> pw = prev.template select<W * TP, 1>(jp * W * TP);
            simd<fp16, GV> v = decode_cb_h<2, GV>(tile_states<K, u, TP, (Geo<K>::D > 1)>(dw, pw));
#pragma unroll
            for (int t = 0; t < TP; ++t) {
                auto Bm = B.template select<256, 1>(256 * (jp * TP + t)).template bit_cast_view<fp16, 8, 32>();
                if constexpr (V == 64) {
                    Bm.template select<1, 1, G, 4>(R0 / 2, C0) = v.template select<G, 1>(t * G);
                } else {
#pragma unroll
                    for (int r = 0; r < P; ++r)
                        Bm.template select<1, 1, 8, 2>((R0 + STR * r) / 2, C0) = v.template select<8, P>(t * G + r);
                }
            }
        }
        build4_any<K, NT, u + 1>(words, prev, B);
    }
}

template <int K, int NT, int u>
ESIMD_INLINE void build4(simd<uint32_t, NT * Geo<K>::WORDS>& words, simd<uint32_t, NT * Geo<K>::WORDS>& prev,
                         simd<fp16, 256 * NT>& B) {
    if constexpr (Geo<K>::V != 32) {
        if constexpr (u == 0) build4_any<K, NT, 0>(words, prev, B);
    } else if constexpr (u < Geo<K>::V) {
        static_assert(Geo<K>::V == 32 && NT * Geo<K>::G == 32, "build4: K with 32 values per period, 4-tile groups");
        simd<fp16, 32> v = decode_cb_h<2, 32>(tile_states<K, u, NT, true>(words, prev));
        constexpr int h = (u >> 2) & 1;
        constexpr int R = 4 * ((u >> 1) & 1) + ((u >> 3) & 3);
        constexpr int C0 = 16 * h + (u & 1);
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            auto Bm = B.template select<256, 1>(256 * j).template bit_cast_view<fp16, 8, 32>();
            Bm.template select<1, 1, 8, 2>(R, C0) = v.template select<8, 1>(8 * j);
        }
        build4<K, NT, u + 1>(words, prev, B);
    }
}

// gate|up: WG per row block b; T = 2 * (I/16) / NT threads (I=640: 20), thread = (proj, 4-tile group).
template <int K, int MB, typename TIn>
struct GateUpDpasKernel {
    const TIn* x; int x_stride;
    const int64_t* ptrs;
    Route rt;
    const int* blk_a; const int* n_blk;
    fp16* ablk;                     // [blocks][I/16][MB][16] (down A operand, Hadamard domain)
    int H, I;
    size_t off_suh_g, off_svh_g, off_suh_d;
    int S;                          // glm-b70: work-groups per row block, each owning I/S columns of gate and up
    float act_lim = EXL3_ACT_LIMIT; // SwiGLU clamp (0: none)

    static constexpr int NT = 4, W = Geo<K>::WORDS;
    static constexpr int SLM_A = 2 * 2 * 8 * MB * 16 * 2;          // [buf][proj][8 tile-rows][MB][16] fp16
    static constexpr int SLM_E = MB * 256 * 4;                       // [MB][g 128 | u 128] fp32
    static ESIMD_INLINE int a_off(int buf, int pj, int i, int m) { return (((buf * 2 + pj) * 8 + i) * MB + m) * 16 * 2; }

    void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
        slm_init<SLM_A + SLM_E>();
        int t = it.get_local_id(0), T = it.get_local_range(0);
        int b = it.get_group(0) / S, part = it.get_group(0) % S;
        if (b >= n_blk[0]) return;
        int a = blk_a[b];
        int e, s0, cnt;
        rt.get(a, e, s0, cnt);
        int r0 = (b - rt.act_pstart[a] / MB) * MB;
        int rows = cnt - r0; if (rows > MB) rows = MB;
        const uint8_t* base = reinterpret_cast<const uint8_t*>(ptrs[e]);
        const uint32_t* tr = reinterpret_cast<const uint32_t*>(base);
        const fp16* suh = reinterpret_cast<const fp16*>(base + off_suh_g);
        int half = T / 2;
        int proj = t / half, grp = part * half + t % half;     // grp: global group of NT tiles within the projection
        int tiles_n = 2 * I / 16;
        int tile_n0 = proj * (I / 16) + grp * NT;

        simd<float, NT * MB * 16> acc = 0.0f;      // [NT][MB][16]
        int nkb = H / 128;
#ifndef EXL3_MOE_NO_PF_REG
        simd<uint32_t, NT * W> wnext = block_load<uint32_t, NT * W>(tr + (size_t)tile_n0 * W);   // software pipeline
#endif
        for (int kb = 0; kb < nkb; ++kb) {
            int buf = kb & 1;
            for (int w = t; w < 2 * MB; w += T) {
                int pj = w / MB, m = w % MB;
                simd<fp16, 128> vh = 0;
                if (m < rows) {
                    int tok = rt.tok(s0 + r0 + m);
                    simd<float, 128> v = convert<float>(block_load<TIn, 128>(x + (size_t)tok * x_stride + kb * 128)) *
                                         convert<float>(block_load<fp16, 128>(suh + (size_t)pj * H + kb * 128));
                    v = convert<float>(convert<fp16>(v));
#ifndef EXL3_MOE_PROBE_DPAS_NOHAD
                    fwht128(v);
#endif
                    vh = convert<fp16>(v * kRsqrt128);
                }
#pragma unroll
                for (int i = 0; i < 8; ++i) slm_block_store<fp16, 16>(a_off(buf, pj, i, m), vh.template select<16, 1>(i * 16));
            }
            barrier();
#pragma unroll 1
            for (int i = 0; i < 8; ++i) {
                int r = kb * 8 + i;
                simd<uint32_t, NT * W> words, prev;
#ifndef EXL3_MOE_NO_PF_REG
                words = wnext;
                if (r + 1 < H / 16) wnext = block_load<uint32_t, NT * W>(tr + ((size_t)(r + 1) * tiles_n + tile_n0) * W);
                planar_prev<K, NT>(words, prev);
#else
                load_planar<K, NT>(tr + ((size_t)r * tiles_n + tile_n0) * W, words, prev);
#endif
                simd<fp16, 256 * NT> B;
#ifdef EXL3_MOE_PROBE_DPAS_NODECODE
                B.template select<NT * W * 2, 1>(0) = words.template bit_cast_view<fp16>();
#else
                build4<K, NT, 0>(words, prev, B);
#endif
#pragma unroll
                for (int rb = 0; rb < MB / 8; ++rb) {
                    simd<fp16, 128> A = slm_block_load<fp16, 128>(a_off(buf, proj, i, rb * 8));
#ifdef EXL3_MOE_PROBE_DPAS_NODPAS
                    acc.template select<128, 1>(rb * 128) += convert<float>(A) + convert<float>(B.template select<128, 1>(rb * 128));
                    continue;
#endif
#pragma unroll
                    for (int j = 0; j < NT; ++j) {
                        auto c = acc.template select<128, 1>((j * MB + rb * 8) * 16);
                        c = xmx::dpas<8, 8, float, float, fp16, fp16>(simd<float, 128>(c), simd<fp16, 256>(B.template select<256, 1>(256 * j)), A);
                    }
                }
            }
        }
        // epilogue, one 128-col block of g and u at a time (2 groups of 4 tiles per proj)
        const fp16* svg = reinterpret_cast<const fp16*>(base + off_svh_g);
        const fp16* sud = reinterpret_cast<const fp16*>(base + off_suh_d);
        int tk_d = I / 16;
        int nb_per = (I / 128) / S;
        for (int nb = part * nb_per; nb < (part + 1) * nb_per; ++nb) {
            barrier();
            if (grp / 2 == nb) {
                int hf = grp % 2;
#pragma unroll
                for (int m = 0; m < MB; ++m) {
                    simd<float, NT * 16> o;
#pragma unroll
                    for (int j = 0; j < NT; ++j) o.template select<16, 1>(j * 16) = acc.template select<16, 1>((j * MB + m) * 16);
                    slm_block_store<float, NT * 16>(SLM_A + (m * 256 + proj * 128 + hf * 64) * 4, o);
                }
            }
            barrier();
            for (int m = t; m < rows; m += T) {
                simd<float, 128> g = slm_block_load<float, 128>(SLM_A + (m * 256) * 4);
                simd<float, 128> u = slm_block_load<float, 128>(SLM_A + (m * 256 + 128) * 4);
                fwht128(g); fwht128(u);
                g = convert<float>(convert<fp16>(g * kRsqrt128 * convert<float>(block_load<fp16, 128>(svg + nb * 128))));
                u = convert<float>(convert<fp16>(u * kRsqrt128 * convert<float>(block_load<fp16, 128>(svg + I + nb * 128))));
                EXL3_ACT_CLAMP(g, u);
                EXL3_SILU_CLAMPED_DECL(act, g, u);
                act = convert<float>(convert<fp16>(act));
                act = convert<float>(convert<fp16>(act * convert<float>(block_load<fp16, 128>(sud + nb * 128))));
                fwht128(act);
                simd<fp16, 128> ah = convert<fp16>(act * kRsqrt128);
#pragma unroll
                for (int i = 0; i < 8; ++i)
                    block_store<fp16, 16>(ablk + (((size_t)b * tk_d + nb * 8 + i) * MB + m) * 16, ah.template select<16, 1>(i * 16));
            }
        }
    }
};

// down: WG per (row block b, quarter q of H); T = (H/4/16)/NT threads (H=2560: 10).
template <int K, int MB, typename TY>
struct DownDpasKernel {
    const fp16* ablk;
    const int64_t* ptrs;
    Route rt;
    const int* blk_a; const int* n_blk;
    TY* y;                          // [npairs, H] (sorted pair order), routing weight applied
    int H, I;
    size_t off_d, off_svh_d;

    static constexpr int NT = 4, W = Geo<K>::WORDS;
    static constexpr int NQ = 4;
    static constexpr int SLM_E = MB * 128 * 4;

    void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
        slm_init<SLM_E>();
        int t = it.get_local_id(0), T = it.get_local_range(0);
        int wg = it.get_group(0);
        int b = wg / NQ, qd = wg % NQ;
        if (b >= n_blk[0]) return;
        int a = blk_a[b];
        int e, s0, cnt;
        rt.get(a, e, s0, cnt);
        int r0 = (b - rt.act_pstart[a] / MB) * MB;
        int rows = cnt - r0; if (rows > MB) rows = MB;
        const uint8_t* base = reinterpret_cast<const uint8_t*>(ptrs[e]);
        const uint32_t* tr = reinterpret_cast<const uint32_t*>(base + off_d);
        int tiles_n = H / 16, tk = I / 16;
        int qcols = H / NQ;
        int tile_n0 = qd * (qcols / 16) + t * NT;

        simd<float, NT * MB * 16> acc = 0.0f;
#ifndef EXL3_MOE_NO_PF_REG
        simd<uint32_t, NT * W> wnext = block_load<uint32_t, NT * W>(tr + (size_t)tile_n0 * W);
        simd<fp16, MB * 16> anext = block_load<fp16, MB * 16>(ablk + ((size_t)b * tk) * MB * 16);
#endif
#pragma unroll 1
        for (int r = 0; r < tk; ++r) {
            simd<uint32_t, NT * W> words, prev;
#ifndef EXL3_MOE_NO_PF_REG
            words = wnext;
            simd<fp16, MB * 16> Ab = anext;
            if (r + 1 < tk) {
                wnext = block_load<uint32_t, NT * W>(tr + ((size_t)(r + 1) * tiles_n + tile_n0) * W);
                anext = block_load<fp16, MB * 16>(ablk + (((size_t)b * tk + r + 1) * MB) * 16);
            }
            planar_prev<K, NT>(words, prev);
#else
            load_planar<K, NT>(tr + ((size_t)r * tiles_n + tile_n0) * W, words, prev);
            simd<fp16, MB * 16> Ab = block_load<fp16, MB * 16>(ablk + (((size_t)b * tk + r) * MB) * 16);
#endif
            simd<fp16, 256 * NT> B;
#ifdef EXL3_MOE_PROBE_DPAS_NODECODE
            B.template select<NT * W * 2, 1>(0) = words.template bit_cast_view<fp16>();
#else
            build4<K, NT, 0>(words, prev, B);
#endif
#pragma unroll
            for (int rb = 0; rb < MB / 8; ++rb) {
                simd<fp16, 128> A = Ab.template select<128, 1>(rb * 128);
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    auto c = acc.template select<128, 1>((j * MB + rb * 8) * 16);
                    c = xmx::dpas<8, 8, float, float, fp16, fp16>(simd<float, 128>(c), simd<fp16, 256>(B.template select<256, 1>(256 * j)), A);
                }
            }
        }
        const fp16* svd = reinterpret_cast<const fp16*>(base + off_svh_d);
        for (int nbl = 0; nbl < qcols / 128; ++nbl) {
            barrier();
            if (t / 2 == nbl) {
                int hf = t % 2;
#pragma unroll
                for (int m = 0; m < MB; ++m) {
                    simd<float, NT * 16> o;
#pragma unroll
                    for (int j = 0; j < NT; ++j) o.template select<16, 1>(j * 16) = acc.template select<16, 1>((j * MB + m) * 16);
                    slm_block_store<float, NT * 16>((m * 128 + hf * 64) * 4, o);
                }
            }
            barrier();
            int nb = qd * (qcols / 128) + nbl;
            for (int m = t; m < rows; m += T) {
                simd<float, 128> v = slm_block_load<float, 128>(m * 128 * 4);
                fwht128(v);
                float w = rt.pair_w[s0 + r0 + m];
                v = v * (kRsqrt128 * w) * convert<float>(block_load<fp16, 128>(svd + nb * 128));
                block_store<TY, 128>(y + (size_t)(s0 + r0 + m) * H + nb * 128, convert<TY>(v));
            }
        }
    }
};
}  // namespace glm_moe
