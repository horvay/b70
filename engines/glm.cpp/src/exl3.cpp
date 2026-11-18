// Host launchers for the EXL3 ESIMD kernels, without torch: a port of exl3_gemm_small from
// llm/glm/xpu/csrc/exl3_ops.sycl (same split-K sizing, same kernels, so the same summation order and results).
#include "exl3.h"
#include <algorithm>
#include <type_traits>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include "../kernels/exl3_esimd.h"
#include <oneapi/mkl/blas.hpp>

namespace glm {

int exl3_large_above = 128;
int exl3_vec_max = 8;                      // Gemma: 3 rows 49 ms a step, DPAS 57; 8 rows 57, DPAS 70
bool exl3_same_split = false;

using namespace exl3;
using fp16 = sycl::half;

namespace {

int round_up(int a, int b) { return (a + b - 1) / b * b; }

constexpr int kLocal = 8;                 // work-group size of the GEMV / DPAS launches
constexpr int kTargetThreads = 1024;      // split-K sizing: vector GEMV and DPAS MB=32
constexpr int kTargetThreadsMb16 = 1408;  // DPAS MB=8/16
constexpr int kTargetThreadsMb64 = 2048;  // DPAS MB>=40

constexpr int tp_of(int K) { return (256 / (32 / cgcd(K, 32))) >= 32 ? 1 : 32 / (256 / (32 / cgcd(K, 32))); }
constexpr int nt_raw(int K, int MR) { return MR == 8 ? 2 : (K <= 4 ? (MR == 1 ? 8 : 4) : 4); }
constexpr int nt_of(int K, int MR) { return nt_raw(K, MR) < tp_of(K) ? tp_of(K) : nt_raw(K, MR); }
int dpas_nt(int MB) { return MB == 8 ? 4 : MB == 24 ? 2 : (MB == 40 || MB == 48) ? 2 : (MB <= 16 ? 4 : (MB == 32 ? 2 : 2)); }

template <bool LargeGrf = false, class Kern>
void launch(sycl::queue& q, int n, const Kern& k) {
    if constexpr (LargeGrf) {
        sycl::ext::oneapi::experimental::properties props{sycl::ext::intel::experimental::grf_size<256>};
        q.submit([&](sycl::handler& h) { h.parallel_for(sycl::nd_range<1>(round_up(n, kLocal), kLocal), props, k); });
    } else {
        q.submit([&](sycl::handler& h) { h.parallel_for(sycl::nd_range<1>(round_up(n, kLocal), kLocal), k); });
    }
}

// a kernel under another name, for a second launch property set (the 256-register mode)
template <class Kern> struct BigGrf : Kern {
    BigGrf(const Kern& k) : Kern(k) {}
};

// NT: tiles per thread, the native choice or the narrow one (alt: 4 for one row, 2 for 2..4 rows)
template <int K, int MR> constexpr int nt_alt() {
    constexpr int n = MR == 1 ? 4 : 2;
    return n < tp_of(K) ? tp_of(K) : n;
}
template <int K, int MR, int CB, bool ALT = false>
void gemv(sycl::queue& q, const fp16* xh, const uint32_t* tr, const int* sonb, float* part, int M, int Kdim, int N,
          int P, int rps, int tiles_n) {
    constexpr int NT = ALT ? nt_alt<K, MR>() : nt_of(K, MR);
    int n_strips = N / 16 / NT;
    GemvKernel<K, CB, MR, NT> k{xh, tr, sonb, part, M, Kdim, N, tiles_n, rps, n_strips};
    // GLM_EXL3_BIGGRF=1: 3 and 4 rows in the 256-register mode (an experiment)
    static const bool big = getenv("GLM_EXL3_BIGGRF") && atoi(getenv("GLM_EXL3_BIGGRF")) != 0;
    if constexpr (MR == 3 || MR == 4) {
        if (big) return launch<true>(q, n_strips * P, BigGrf<GemvKernel<K, CB, MR, NT>>{k});
    }
    launch(q, n_strips * P, k);
}

template <int K, int MB, int NT, int CB>
void dpas(sycl::queue& q, const fp16* xh, const uint32_t* tr, const int* sonb, float* part, int M, int Kdim, int N,
          int P, int rps, int tiles_n) {
    int n_strips = N / 16 / NT, m_blocks = (M + MB - 1) / MB;
    DpasKernel<K, CB, MB, NT> k{xh, tr, sonb, part, M, Kdim, N, tiles_n, rps, n_strips, m_blocks, m_blocks * MB};
    launch<(MB >= 40)>(q, n_strips * m_blocks * P, k);    // 256-register mode for the MB x 16 x NT accumulator
}

// the rates served since the first packs, every row block; the rest (K 1, 2, 8 and the half-integer ones, SAGE's and
// the "sc" quants) come with models that set exl3_large_above to 32 or less
template <int K> constexpr bool all_blocks() { return K >= 3 && K <= 7; }

template <int K, int CB = 2>
void run_k(sycl::queue& q, bool vec, int MR, int MB, const fp16* xh, const uint32_t* tr, const int* sonb, float* part,
           int M, int Kdim, int N, int P, int rps, bool alt, int tn) {
    if (vec) {
        if (MR == 1) alt ? gemv<K, 1, CB, true>(q, xh, tr, sonb, part, M, Kdim, N, P, rps, tn) : gemv<K, 1, CB>(q, xh, tr, sonb, part, M, Kdim, N, P, rps, tn);
        else if (MR == 2) alt ? gemv<K, 2, CB, true>(q, xh, tr, sonb, part, M, Kdim, N, P, rps, tn) : gemv<K, 2, CB>(q, xh, tr, sonb, part, M, Kdim, N, P, rps, tn);
        else if (MR == 3) alt ? gemv<K, 3, CB, true>(q, xh, tr, sonb, part, M, Kdim, N, P, rps, tn) : gemv<K, 3, CB>(q, xh, tr, sonb, part, M, Kdim, N, P, rps, tn);
        else if (MR == 4) alt ? gemv<K, 4, CB, true>(q, xh, tr, sonb, part, M, Kdim, N, P, rps, tn) : gemv<K, 4, CB>(q, xh, tr, sonb, part, M, Kdim, N, P, rps, tn);
        else gemv<K, 8, CB>(q, xh, tr, sonb, part, M, Kdim, N, P, rps, tn);
        return;
    }
    if constexpr (CB != 2) {                              // a head: at most the 16 rows of a step
        GLM_CHECK(MB <= 16, "EXL3 codebook %d: %d rows", CB, M);
        dpas<K, 16, 4, CB>(q, xh, tr, sonb, part, M, Kdim, N, P, rps, tn);
    } else {
        switch (MB) {
            case 4: dpas<K, 4, 4, CB>(q, xh, tr, sonb, part, M, Kdim, N, P, rps, tn); break;
            case 8: dpas<K, 8, 4, CB>(q, xh, tr, sonb, part, M, Kdim, N, P, rps, tn); break;
            case 16: dpas<K, 16, 4, CB>(q, xh, tr, sonb, part, M, Kdim, N, P, rps, tn); break;
            case 24: dpas<K, 24, 2, CB>(q, xh, tr, sonb, part, M, Kdim, N, P, rps, tn); break;
            case 32: dpas<K, 32, 2, CB>(q, xh, tr, sonb, part, M, Kdim, N, P, rps, tn); break;
            default:
                if constexpr (all_blocks<K>()) {
                    if (MB == 40) dpas<K, 40, 2, CB>(q, xh, tr, sonb, part, M, Kdim, N, P, rps, tn);
                    else if (MB == 48) dpas<K, 48, 2, CB>(q, xh, tr, sonb, part, M, Kdim, N, P, rps, tn);
                    else dpas<K, 64, 2, CB>(q, xh, tr, sonb, part, M, Kdim, N, P, rps, tn);
                } else {
                    GLM_CHECK(false, "EXL3 K code %d: %d rows (set exl3_large_above to 32)", K, M);
                }
        }
    }
}

// f(integral_constant<int, K>) for K's code
template <class F>
void with_k(int K, F&& f) {
    switch (K) {
        case 1: f(std::integral_constant<int, 1>{}); break;
        case 2: f(std::integral_constant<int, 2>{}); break;
        case 3: f(std::integral_constant<int, 3>{}); break;
        case 4: f(std::integral_constant<int, 4>{}); break;
        case 5: f(std::integral_constant<int, 5>{}); break;
        case 6: f(std::integral_constant<int, 6>{}); break;
        case 7: f(std::integral_constant<int, 7>{}); break;
        case 8: f(std::integral_constant<int, 8>{}); break;
        case 15: f(std::integral_constant<int, 15>{}); break;
        case 25: f(std::integral_constant<int, 25>{}); break;
        case 35: f(std::integral_constant<int, 35>{}); break;
        default: GLM_CHECK(false, "EXL3 K code %d not built", K);
    }
}

// the original codebook's rates (heads)
template <class F>
void with_k_cb0(int K, F&& f) {
    switch (K) {
        case 3: f(std::integral_constant<int, 3>{}); break;
        case 4: f(std::integral_constant<int, 4>{}); break;
        case 5: f(std::integral_constant<int, 5>{}); break;
        case 6: f(std::integral_constant<int, 6>{}); break;
        case 7: f(std::integral_constant<int, 7>{}); break;
        case 8: f(std::integral_constant<int, 8>{}); break;
        default: GLM_CHECK(false, "EXL3 original codebook, K code %d not built", K);
    }
}

}  // namespace

int Exl3Linear::k_code(int words, bool mul1) {
    int K = words % 16 == 8 ? 10 * (words / 16) + 5 : words / 16;
    GLM_CHECK(words % 8 == 0 && ((K >= 1 && K <= 8) || K == 15 || K == 25 || K == 35), "EXL3: %d words a tile", words);
    GLM_CHECK(mul1 || K < 10, "EXL3: half-integer K without the mul1 codebook");
    return K;
}

Exl3Linear Exl3Linear::load(const SafeTensors& st, const std::string& prefix) {
    Exl3Linear l;
    l.trellis = st.load(prefix + ".trellis");
    l.suh = st.load(prefix + ".suh");
    l.svh = st.load(prefix + ".svh");
    GLM_CHECK(l.trellis.dt == DType::I16 && l.trellis.shape.size() == 3, "%s.trellis: expected I16 3-D", prefix.c_str());
    GLM_CHECK(!st.has(prefix + ".mcg"), "%s: mcg codebook not supported", prefix.c_str());
    l.in = l.trellis.size(0) * 16;
    l.out = l.trellis.size(1) * 16;
    l.cb = st.has(prefix + ".mul1") ? 2 : 0;
    l.K = k_code(l.trellis.size(2), l.cb == 2);
    l.sonb = Tensor::zeros({l.out / 128}, DType::I32);
    return l;
}

// M > 128 (prompts): input Hadamard (row-major), the trellis decoded to fp16 W (Hadamard domain) in 16384-column slices,
// one fp16 GEMM per slice (oneMKL, fp32 accumulation), output Hadamard. The GEMM output is fp16, so an fp32-output
// linear is fp16-rounded here, as in the PyTorch stack (exl3_linear).
template <int K>
static void reconstruct(sycl::queue& q, const uint32_t* tr, fp16* w, int tiles_n, int n0, int n_out, int tk) {
    constexpr int NT = 8;                              // 128 columns per thread
    int n_strips = n_out / (16 * NT);
    ReconstructKernel<K, 2, NT> k{tr, w, tiles_n, n0 / 16, n_out, n_strips, tk};
    int n = n_strips * tk;
    q.submit([&](sycl::handler& h) { h.parallel_for(sycl::nd_range<1>(round_up(n, 8), 8), k); });
}

void Exl3Linear::forward_large(sycl::queue& q, Workspace& ws, const half* x, int M, void* y, DType ydt,
                               int x_stride) const {
    GLM_CHECK(cb == 2, "EXL3 original codebook: %d rows", M);
    const int Kdim = in, N = out, tk = Kdim / 16, tiles_n = N / 16;
    constexpr int SLICE = 16384;
    fp16* xh = ws.take<fp16>((size_t)M * Kdim);
    fp16* part = ws.take<fp16>((size_t)M * N);
    fp16* w = ws.take<fp16>((size_t)Kdim * std::min(N, SLICE));
    {
        HadInKernel<fp16> k{x, suh.ptr<fp16>(), xh, M, Kdim, 1, x_stride, M, 1};
        int n = M * (Kdim / 128);
        q.submit([&](sycl::handler& h) { h.parallel_for(sycl::nd_range<1>(round_up(n, 8), 8), k); });
    }
    const uint32_t* tr = trellis.ptr<uint32_t>();
    for (int n0 = 0; n0 < N; n0 += SLICE) {
        int ns = std::min(SLICE, N - n0);
        with_k(K, [&](auto kc) { reconstruct<decltype(kc)::value>(q, tr, w, tiles_n, n0, ns, tk); });
        // part[:, n0:n0+ns] = xh @ w  (row major)
        oneapi::mkl::blas::row_major::gemm(q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, M, ns,
                                           Kdim, fp16(1.0f), xh, Kdim, w, ns, fp16(0.0f), part + n0, N);
    }
    int n = M * (N / 128);
    if (ydt == DType::F16) {
        HadOutKernel<fp16, fp16> k{part, svh.ptr<fp16>(), static_cast<fp16*>(y), M, N, 1, N};
        q.submit([&](sycl::handler& h) { h.parallel_for(sycl::nd_range<1>(round_up(n, 8), 8), k); });
    } else {
        HadOutKernel<float, fp16> k{part, svh.ptr<fp16>(), static_cast<float*>(y), M, N, 1, N};
        q.submit([&](sycl::handler& h) { h.parallel_for(sycl::nd_range<1>(round_up(n, 8), 8), k); });
    }
}

void Exl3Linear::forward(sycl::queue& q, Workspace& ws, const half* x, int M, void* y, DType ydt, int x_stride,
                         int n_cols) const {
    if (x_stride < 0) x_stride = in;
    GLM_CHECK(M >= 1, "Exl3Linear: M=%d", M);
    ws.reset();
    if (n_cols >= 0 && n_cols < out) {
        GLM_CHECK(n_cols % 128 == 0 && M <= exl3_large_above && !lora, "Exl3Linear: %d of %d columns", n_cols, out);
        forward_base(q, ws, x, M, y, ydt, x_stride, n_cols);
        return;
    }
    forward_base(q, ws, x, M, y, ydt, x_stride);
    if (lora) {
        GLM_CHECK(x_stride == in, "LoRA on a strided input");
        lora->apply(q, ws, x, M, y, ydt);
    }
}

void Exl3Linear::forward_base(sycl::queue& q, Workspace& ws, const half* x, int M, void* y, DType ydt,
                              int x_stride, int n_cols) const {
    if (M > exl3_large_above) return forward_large(q, ws, x, M, y, ydt, x_stride);
    const int Kdim = in, N = n_cols >= 0 ? n_cols : out, tn = out / 16;
    static const int vec_env = getenv("GLM_EXL3_VECMAX") ? atoi(getenv("GLM_EXL3_VECMAX")) : -1;
    // K under 3 packs few values into a vector load: 8 rows of Ripple's whole model at K 2 / 3 / 4, 143 ms a step, DPAS 90 (at flat K 3: 98, DPAS 127)
    const int vec_max = vec_env >= 0 ? vec_env : K < 3 || (K >= 10 && K < 25) ? std::min(exl3_vec_max, 4) : exl3_vec_max;
    bool vec = M <= vec_max;
    int MR = M <= 1 ? 1 : M == 2 ? 2 : M == 3 ? 3 : M == 4 ? 4 : 8;
    int MB = M <= 4 ? 4 : M <= 8 ? 8 : (M <= 16 ? 16 : (M <= 24 ? 24 : (M <= 32 ? 32 : (M <= 40 ? 40 : (M <= 48 ? 48 : 64)))));
    int m_blocks = vec ? 1 : (M + MB - 1) / MB;
    int Mp = vec ? M : m_blocks * MB;
    // blocked activations [1][Kdim/16][Mp][16]; DPAS pad rows must be zero
    fp16* xh = ws.take<fp16>((size_t)Mp * Kdim);
    if (Mp != M) q.memset(xh, 0, (size_t)Mp * Kdim * sizeof(fp16));
    {
        HadInKernel<fp16> k{x, suh.ptr<fp16>(), xh, M, Kdim, 1, x_stride, Mp, 0};
        int n = M * (Kdim / 128);
        q.submit([&](sycl::handler& h) { h.parallel_for(sycl::nd_range<1>(round_up(n, 8), 8), k); });
    }
    int tk = Kdim / 16;
    // GLM_EXL3_ALT: rows (bit 0: 1 row, 1: 2, 2: 3, 3: 4) that take the narrow NT (nt_alt)
    // (default with the same split: one row narrow, 47.8 ms a Qwen3.8-40B step against 50.1 for NT 8)
    static const int alt_env = getenv("GLM_EXL3_ALT") ? atoi(getenv("GLM_EXL3_ALT")) : -1;
    const int alt_mask = alt_env >= 0 ? alt_env : (exl3_same_split ? 1 : 0);
    const bool alt = cb == 2 && vec && MR <= 4 && ((alt_mask >> (MR - 1)) & 1);
    int nt = 0;
    if (vec) {
        if (alt) with_k(K, [&](auto kc) {
            constexpr int KK = decltype(kc)::value;
            nt = MR == 1 ? nt_alt<KK, 1>() : nt_alt<KK, 2>();
        });
        else nt = nt_of(K, MR);
    }
    int n_strips = (N / 16) / (vec ? nt : dpas_nt(MB));
    static const int same_env = getenv("GLM_EXL3_SAMESPLIT") ? atoi(getenv("GLM_EXL3_SAMESPLIT")) : -1;
    const bool same = vec && (same_env >= 0 ? same_env > 0 : exl3_same_split);
    // the same split: K cut as for 2..4 rows' native NT, whatever this launch's NT
    int units = same ? (N / 16) / nt_of(K, 2) : n_strips * m_blocks;
    static const int vec_target = getenv("GLM_EXL3_TARGET") ? atoi(getenv("GLM_EXL3_TARGET")) : kTargetThreads;
    int target = vec ? vec_target : (MB <= 16 ? kTargetThreadsMb16 : (MB >= 40 ? kTargetThreadsMb64 : kTargetThreads));
    int P = std::max(1, std::min(tk, (target + units - 1) / units));
    int rps = (tk + P - 1) / P;
    P = (tk + rps - 1) / rps;
    float* part = ws.take<float>((size_t)P * M * N);
    const uint32_t* tr = trellis.ptr<uint32_t>();
    if (cb == 2) with_k(K, [&](auto kc) { run_k<decltype(kc)::value>(q, vec, MR, MB, xh, tr, sonb.ptr<int>(), part, M, Kdim, N, P, rps, alt, tn); });
    else with_k_cb0(K, [&](auto kc) { run_k<decltype(kc)::value, 0>(q, vec, MR, MB, xh, tr, sonb.ptr<int>(), part, M, Kdim, N, P, rps, false, tn); });
    int n = M * (N / 128);
    if (ydt == DType::F16) {
        HadOutKernel<fp16> k{part, svh.ptr<fp16>(), static_cast<fp16*>(y), M, N, P, N};
        q.submit([&](sycl::handler& h) { h.parallel_for(sycl::nd_range<1>(round_up(n, 8), 8), k); });
    } else {
        GLM_CHECK(ydt == DType::F32, "Exl3Linear: output dtype");
        HadOutKernel<float> k{part, svh.ptr<fp16>(), static_cast<float*>(y), M, N, P, N};
        q.submit([&](sycl::handler& h) { h.parallel_for(sycl::nd_range<1>(round_up(n, 8), 8), k); });
    }
}

}  // namespace glm
