// Host launchers for the EXL3 ESIMD kernels, without torch: a port of exl3_gemm_small from
// llm/glm/xpu/csrc/exl3_ops.sycl (same split-K sizing, same kernels, so the same summation order and results).
#include "exl3.h"
#include <algorithm>
#include <type_traits>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include "../kernels/exl3_esimd.h"
#include <oneapi/mkl/blas.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>
#include <chrono>
#include <cmath>
#include <map>
#include <mutex>
#include <vector>

namespace glm {

int exl3_large_above = 128;
bool exl3_int8 = !getenv("GLM_EXL3_INT8") || atoi(getenv("GLM_EXL3_INT8")) != 0;
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

// a merged linear's outputs on an output-Hadamard launch (ys null: one output, as set)
template <class Kern>
void set_shards_impl(Kern& k, const Exl3Linear& L, void* const* ys) {
    if (!ys) return;
    using T = std::remove_pointer_t<decltype(k.out)>;
    k.nsh = L.S;
    T* o[4] = {};
    for (int s = 0; s < L.S; s++) o[s] = static_cast<T*>(ys[s]);
    k.o0 = o[0]; k.o1 = o[1]; k.o2 = o[2]; k.o3 = o[3];
    k.c1 = L.shard_col[1]; k.c2 = L.shard_col[2]; k.c3 = L.shard_col[3]; k.c4 = L.shard_col[4];
}
#define set_shards(k, ys) set_shards_impl(k, *this, ys)

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

// The fp16 prompt path (GLM_EXL3_INT8=0): the trellis decoded to fp16 W (Hadamard domain) in 16384-column slices, one
// fp16 GEMM per slice (oneMKL, fp32 accumulation).
template <int K>
static void reconstruct(sycl::queue& q, const uint32_t* tr, fp16* w, int tiles_n, int n0, int n_out, int tk) {
    constexpr int NT = 8;                              // 128 columns per thread
    int n_strips = n_out / (16 * NT);
    ReconstructKernel<K, 2, NT> k{tr, w, tiles_n, n0 / 16, n_out, n_strips, tk};
    int n = n_strips * tk;
    q.submit([&](sycl::handler& h) { h.parallel_for(sycl::nd_range<1>(round_up(n, 8), 8), k); });
}

void Exl3Linear::forward_large_f16(sycl::queue& q, Workspace& ws, const half* x, int M, void* y, DType ydt,
                                   int x_stride, void* const* ys) const {
    const int Kdim = in, N = out, tk = Kdim / 16, tiles_n = N / 16;
    constexpr int SLICE = 16384;
    fp16* xh = ws.take<fp16>((size_t)S * M * Kdim);
    fp16* part = ws.take<fp16>((size_t)M * N);
    fp16* w = ws.take<fp16>((size_t)Kdim * std::min(N, SLICE));
    {
        HadInKernel<fp16> k{x, suh.ptr<fp16>(), xh, M, Kdim, S, x_stride, M, 1};
        int n = S * M * (Kdim / 128);
        q.submit([&](sycl::handler& h) { h.parallel_for(sycl::nd_range<1>(round_up(n, 8), 8), k); });
    }
    const uint32_t* tr = trellis.ptr<uint32_t>();
    for (int sh = 0; sh < S; sh++) {
        const int c1 = S > 1 ? shard_col[sh + 1] : N;
        for (int n0 = S > 1 ? shard_col[sh] : 0; n0 < c1; n0 += SLICE) {
            int ns = std::min(SLICE, c1 - n0);
            with_k(K, [&](auto kc) { reconstruct<decltype(kc)::value>(q, tr, w, tiles_n, n0, ns, tk); });
            // part[:, n0:n0+ns] = xh (its shard's) @ w  (row major)
            oneapi::mkl::blas::row_major::gemm(q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, M, ns,
                                               Kdim, fp16(1.0f), xh + (size_t)sh * M * Kdim, Kdim, w, ns, fp16(0.0f),
                                               part + n0, N);
        }
    }
    had_out(q, part, M, y, ydt, ys);
}

void Exl3Linear::had_out(sycl::queue& q, const half* part, int M, void* y, DType ydt, void* const* ys) const {
    const int N = out;
    int n = M * (N / 128);
    if (ydt == DType::F16) {
        HadOutKernel<fp16, fp16> k{part, svh.ptr<fp16>(), static_cast<fp16*>(y), M, N, 1, N};
        set_shards(k, ys);
        q.submit([&](sycl::handler& h) { h.parallel_for(sycl::nd_range<1>(round_up(n, 8), 8), k); });
    } else {
        HadOutKernel<float, fp16> k{part, svh.ptr<fp16>(), static_cast<float*>(y), M, N, 1, N};
        set_shards(k, ys);
        q.submit([&](sycl::handler& h) { h.parallel_for(sycl::nd_range<1>(round_up(n, 8), 8), k); });
    }
}

// ---- the int8 (W8A8) prompt path --------------------------------------------------------------------------------
// In the Hadamard domain the activations have no outlier channels (QuaRot's setting) and the mul1 codebook is bounded,
// so one scale per row and one static weight scale lose ~1% (trellis-serve, xpu/docs/PROGRESS.md: NLL +0.17%).

namespace {

constexpr float kWMax = 3.453125f;      // max |fp16(h * c1 + c2)| over the mul1 codebook (h = 1024 .. 2044)

template <int K>
void reconstruct_i8(sycl::queue& q, const uint32_t* tr, int8_t* w, int tiles_n, int n0, int n_out, int tk) {
    constexpr int NT = 8;
    int n_strips = n_out / (16 * NT);
    ReconstructKernel<K, 2, NT, int8_t> k{tr, w, tiles_n, n0 / 16, n_out, n_strips, tk, 127.0f / kWMax};
    int n = n_strips * tk;
    q.submit([&](sycl::handler& h) { h.parallel_for(sycl::nd_range<1>(round_up(n, 8), 8), k); });
}

// had_in to int8: one work-group of 16 threads a row, NB 128-blocks a thread kept in registers
template <int NB>
void had_in_q8(sycl::queue& q, const fp16* x, const fp16* suh, int8_t* xq, float* sx, int M, int Kdim, int x_stride,
               int Mp, float smul) {
    constexpr int TPR = 16;
    HadInQ8WgKernel<fp16, TPR, NB> k{x, suh, xq, sx, M, Kdim, 1, x_stride, Mp, smul};
    q.submit([&](sycl::handler& h) { h.parallel_for(sycl::nd_range<1>((size_t)M * TPR, TPR), k); });
}

void had_in_q8_any(sycl::queue& q, const fp16* x, const fp16* suh, int8_t* xq, float* sx, int M, int Kdim, int x_stride,
                   int Mp, float smul) {
    int nb = (Kdim / 128 + 15) / 16;
    if (nb <= 1) had_in_q8<1>(q, x, suh, xq, sx, M, Kdim, x_stride, Mp, smul);
    else if (nb <= 2) had_in_q8<2>(q, x, suh, xq, sx, M, Kdim, x_stride, Mp, smul);
    else if (nb <= 3) had_in_q8<3>(q, x, suh, xq, sx, M, Kdim, x_stride, Mp, smul);
    else if (nb <= 4) had_in_q8<4>(q, x, suh, xq, sx, M, Kdim, x_stride, Mp, smul);
    else if (nb <= 6) had_in_q8<6>(q, x, suh, xq, sx, M, Kdim, x_stride, Mp, smul);
    else if (nb <= 8) had_in_q8<8>(q, x, suh, xq, sx, M, Kdim, x_stride, Mp, smul);
    else if (nb <= 12) had_in_q8<12>(q, x, suh, xq, sx, M, Kdim, x_stride, Mp, smul);
    else if (nb <= 16) had_in_q8<16>(q, x, suh, xq, sx, M, Kdim, x_stride, Mp, smul);
    else {      // past 32K inputs: one thread a row, two passes
        HadInQ8Kernel<fp16> k{x, suh, xq, sx, M, Kdim, 1, x_stride, Mp, smul};
        q.submit([&](sycl::handler& h) { h.parallel_for(sycl::nd_range<1>(round_up(M, 8), 8), k); });
    }
}

// oneDNN s8 x s8 matmul, c[M, N] (ldc) = fp16((a[M, K] * sa[M]) @ b[K, N]): sa holds both scales. Primitives are kept
// per shape; the scratchpad is ours (one buffer, grown), so nothing is allocated behind the engine's VRAM accounting.
struct I8Gemm {
    std::mutex mu;
    dnnl::engine eng;
    std::vector<std::pair<sycl::queue, dnnl::stream>> streams;
    struct Prim {
        dnnl::matmul prim;
        dnnl::memory::desc a, b, c, s, scr;
        size_t scr_bytes = 0;
    };
    std::map<std::tuple<int, int, int, int>, Prim> prims;
    void* scr = nullptr;
    size_t scr_cap = 0;

    static I8Gemm& get() {
        static I8Gemm* g = new I8Gemm();   // never destroyed: no oneDNN teardown after the SYCL runtime's at exit
        return *g;
    }
    I8Gemm() {
        auto& d = Dev::get();
        eng = dnnl::sycl_interop::make_engine(d.dev, d.ctx);
    }
    dnnl::stream& stream(sycl::queue& q) {
        for (auto& [qq, st] : streams)
            if (qq == q) return st;
        streams.emplace_back(q, dnnl::sycl_interop::make_stream(eng, q));
        return streams.back().second;
    }
    void run(sycl::queue& q, const int8_t* a, const float* sa, const int8_t* b, fp16* c, int M, int K, int N, int ldc) {
        using namespace dnnl;
        using tag = memory::format_tag;
        std::lock_guard<std::mutex> lk(mu);
        auto key = std::make_tuple(M, K, N, ldc);
        auto it = prims.find(key);
        if (it == prims.end()) {
            auto t0 = std::chrono::steady_clock::now();
            Prim p;
            p.a = memory::desc({M, K}, memory::data_type::s8, tag::ab);
            p.b = memory::desc({K, N}, memory::data_type::s8, tag::ab);
            p.c = memory::desc({M, N}, memory::data_type::f16, memory::dims{ldc, 1});
            p.s = memory::desc({M}, memory::data_type::f32, tag::a);
            primitive_attr attr;
            attr.set_scratchpad_mode(scratchpad_mode::user);
            attr.set_scales_mask(DNNL_ARG_SRC, 1 << 0);     // a scale per row of a
            matmul::primitive_desc pd(eng, p.a, p.b, p.c, attr);
            p.scr = pd.scratchpad_desc();
            p.scr_bytes = p.scr.get_size();
            p.prim = matmul(pd);
            static const bool verbose = getenv("GLM_EXL3_I8_VERBOSE") != nullptr;
            if (verbose)
                fprintf(stderr, "exl3 int8: GEMM %d x %d x %d (ldc %d): %s, scratch %zu B, %.0f ms to build\n", M, K, N, ldc,
                        pd.impl_info_str(), p.scr_bytes,
                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            it = prims.emplace(key, std::move(p)).first;
        }
        Prim& p = it->second;
        if (p.scr_bytes > scr_cap) {
            if (scr) { q.wait(); sycl::free(scr, q); }
            scr = sycl::malloc_device(p.scr_bytes, q);
            GLM_CHECK(scr, "exl3 int8: %zu B of GEMM scratch", p.scr_bytes);
            scr_cap = p.scr_bytes;
        }
        auto mem = [&](const memory::desc& md, const void* ptr) {
            return sycl_interop::make_memory(md, eng, sycl_interop::memory_kind::usm, const_cast<void*>(ptr));
        };
        std::unordered_map<int, memory> args = {{DNNL_ARG_SRC, mem(p.a, a)}, {DNNL_ARG_WEIGHTS, mem(p.b, b)},
                                                {DNNL_ARG_DST, mem(p.c, c)},
                                                {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, mem(p.s, sa)}};
        if (p.scr_bytes) args[DNNL_ARG_SCRATCHPAD] = mem(p.scr, scr);
        p.prim.execute(stream(q), args);
    }
};

bool recording(sycl::queue& q) {
    return q.ext_oneapi_get_state() == sycl::ext::oneapi::experimental::queue_state::recording;
}

}  // namespace

// GLM_EXL3_I8_STAGES=1: each stage waited for and timed (a measurement; exl3_i8_stages_report() prints the sums)
static const bool g_i8_stages = getenv("GLM_EXL3_I8_STAGES") && atoi(getenv("GLM_EXL3_I8_STAGES")) != 0;
static double g_i8_t[4];
static void i8_stage(sycl::queue& q, int i, std::chrono::steady_clock::time_point& t) {
    if (!g_i8_stages) return;
    q.wait();
    auto n = std::chrono::steady_clock::now();
    g_i8_t[i] += std::chrono::duration<double>(n - t).count();
    t = n;
}
std::string exl3_i8_stages_report() {
    char b[256];
    snprintf(b, sizeof b, "had_in %.1f ms, reconstruct %.1f ms, GEMM %.1f ms, had_out %.1f ms", g_i8_t[0] * 1e3, g_i8_t[1] * 1e3,
             g_i8_t[2] * 1e3, g_i8_t[3] * 1e3);
    for (double& t : g_i8_t) t = 0;
    return b;
}

// The decoded int8 weights of the last linears, so a prompt read in pieces decodes each layer's trellis once per
// layer, not once per piece (Q35 calls a layer's 6-8 linears for each 2048-row piece of a 16K-row super-chunk). An arena
// filled in call order, wrapping to its start; an entry is gone once something is written over it.
size_t exl3_i8_cache_mb = 0;
namespace {
struct I8Cache {
    Tensor arena;                    // never freed: the cache lives as long as the process (no teardown after SYCL's)
    char* base = nullptr;
    size_t cap = 0, top = 0;
    struct E { const void* key; int n0; size_t off, bytes; };
    std::vector<E> es;
    // the cached weights of (key, n0), or null after reserving room for them in *dst (null: no room at all)
    const int8_t* find(sycl::queue& q, const void* key, int n0, size_t bytes, int8_t** dst) {
        *dst = nullptr;
        static const long env = getenv("GLM_EXL3_I8_CACHE_MB") ? atol(getenv("GLM_EXL3_I8_CACHE_MB")) : -1;
        const size_t want = (env >= 0 ? (size_t)env : exl3_i8_cache_mb) << 20;
        if (want == 0 || bytes > want) return nullptr;
        for (auto& e : es) if (e.key == key && e.n0 == n0 && e.bytes == bytes) return reinterpret_cast<int8_t*>(base + e.off);
        if (cap != want) {
            q.wait();
            arena = Tensor::empty({(int64_t)want}, DType::U8);
            base = arena.ptr<char>();
            cap = want;
            top = 0;
            es.clear();
        }
        bytes = (bytes + 255) & ~size_t(255);
        if (top + bytes > cap) top = 0;
        const size_t off = top;
        top += bytes;
        es.erase(std::remove_if(es.begin(), es.end(), [&](const E& e) { return e.off < off + bytes && off < e.off + e.bytes; }),
                 es.end());
        es.push_back({key, n0, off, bytes});
        *dst = reinterpret_cast<int8_t*>(base + off);
        return nullptr;
    }
};
I8Cache& g_i8c = *new I8Cache();
}  // namespace

void Exl3Linear::forward_large_i8(sycl::queue& q, Workspace& ws, const half* x, int M, void* y, DType ydt,
                                  int x_stride, void* const* ys) const {
    const int Kdim = in, N = out, tk = Kdim / 16, tiles_n = N / 16;
    if (g_i8_stages) q.wait();
    auto t = std::chrono::steady_clock::now();
    const int Mp = (M + kI8RowPad - 1) / kI8RowPad * kI8RowPad;
    int8_t* xq0 = ws.take<int8_t>((size_t)S * Mp * Kdim);
    float* sx0 = ws.take<float>((size_t)S * Mp);
    fp16* part = ws.take<fp16>((size_t)Mp * N);
    int8_t* w = ws.take<int8_t>((size_t)Kdim * std::min(N, kI8Slice));
    for (int sh = 0; sh < S; sh++) {
        int8_t* xq = xq0 + (size_t)sh * Mp * Kdim;
        float* sx = sx0 + (size_t)sh * Mp;
        // the pad rows: zero inputs and scales (their outputs are never read)
        if (Mp > M) {
            q.memset(xq + (size_t)M * Kdim, 0, (size_t)(Mp - M) * Kdim);
            q.memset(sx + M, 0, (size_t)(Mp - M) * 4);
        }
        had_in_q8_any(q, x, suh.ptr<fp16>() + (size_t)sh * Kdim, xq, sx, M, Kdim, x_stride, Mp, kWMax / 127.0f);
    }
    i8_stage(q, 0, t);
    const uint32_t* tr = trellis.ptr<uint32_t>();
    auto& g = I8Gemm::get();
    for (int sh = 0; sh < S; sh++) {
    const int8_t* xq = xq0 + (size_t)sh * Mp * Kdim;
    const float* sx = sx0 + (size_t)sh * Mp;
    const int c1 = S > 1 ? shard_col[sh + 1] : N;
    for (int n0 = S > 1 ? shard_col[sh] : 0; n0 < c1; n0 += kI8Slice) {
        int ns = std::min(kI8Slice, c1 - n0);
        int8_t* fill = nullptr;
        const int8_t* wc = g_i8c.find(q, tr, n0, (size_t)Kdim * ns, &fill);
        int8_t* wd = fill ? fill : w;
        if (!wc) with_k(K, [&](auto kc) { reconstruct_i8<decltype(kc)::value>(q, tr, wd, tiles_n, n0, ns, tk); });
        i8_stage(q, 1, t);
        g.run(q, xq, sx, wc ? wc : wd, part + n0, Mp, Kdim, ns, N);
        i8_stage(q, 2, t);
    }
    }
    had_out(q, part, M, y, ydt, ys);
    i8_stage(q, 3, t);
}

// M > exl3_large_above (prompts): input Hadamard (row-major), the trellis decoded to W (Hadamard domain) in column
// slices, one GEMM per slice (int8, or fp16 with GLM_EXL3_INT8=0), output Hadamard. The GEMM output is fp16, so an
// fp32-output linear is fp16-rounded here, as in the PyTorch stack (exl3_linear).
void Exl3Linear::forward_large(sycl::queue& q, Workspace& ws, const half* x, int M, void* y, DType ydt,
                               int x_stride, void* const* ys) const {
    GLM_CHECK(cb == 2, "EXL3 original codebook: %d rows", M);
    if (!exl3_int8 || recording(q)) return forward_large_f16(q, ws, x, M, y, ydt, x_stride, ys);
    forward_large_i8(q, ws, x, M, y, ydt, x_stride, ys);
    static const bool check = getenv("GLM_EXL3_INT8_CHECK") && atoi(getenv("GLM_EXL3_INT8_CHECK")) != 0;
    if (!check || ys) return;
    // the same call on the fp16 path, compared on the host
    static Workspace cws;
    static void* yref = nullptr;
    static size_t yref_cap = 0;
    static int calls = 0;
    size_t need = ws_bytes(M), ybytes = (size_t)M * out * dsize(ydt);
    q.wait();
    if (cws.cap < need) {
        if (cws.base) sycl::free(cws.base, q);
        cws.base = nullptr;
        cws.init(need);
    }
    if (yref_cap < ybytes) {
        if (yref) sycl::free(yref, q);
        yref = sycl::malloc_device(ybytes, q);
        yref_cap = ybytes;
    }
    cws.reset();
    forward_large_f16(q, cws, x, M, yref, ydt, x_stride);
    q.wait();
    std::vector<float> a((size_t)M * out), b((size_t)M * out);
    if (ydt == DType::F16) {
        std::vector<fp16> ha(a.size()), hb(b.size());
        q.memcpy(ha.data(), y, ybytes).wait();
        q.memcpy(hb.data(), yref, ybytes).wait();
        for (size_t i = 0; i < a.size(); i++) { a[i] = ha[i]; b[i] = hb[i]; }
    } else {
        q.memcpy(a.data(), y, ybytes).wait();
        q.memcpy(b.data(), yref, ybytes).wait();
    }
    double num = 0, den = 0, worst_row = 0;
    for (int m = 0; m < M; m++) {
        double rn = 0, rd = 0;
        for (int n = 0; n < out; n++) {
            double d = (double)a[(size_t)m * out + n] - b[(size_t)m * out + n];
            rn += d * d;
            rd += (double)b[(size_t)m * out + n] * b[(size_t)m * out + n];
        }
        num += rn;
        den += rd;
        if (rd > 0) worst_row = std::max(worst_row, std::sqrt(rn / rd));
    }
    fprintf(stderr, "exl3 int8 check #%d: %d -> %d K=%d, %d rows: rel err %.3e (worst row %.3e)\n", calls++, in, out, K, M,
            std::sqrt(num / std::max(den, 1e-30)), worst_row);
}

void Exl3Linear::forward_split(sycl::queue& q, Workspace& ws, const half* x, int M, void* const* ys, DType ydt) const {
    GLM_CHECK(M >= 1 && !lora, "Exl3Linear::forward_split: M=%d", M);
    ws.reset();
    forward_base(q, ws, x, M, nullptr, ydt, in, -1, ys);
}

bool Exl3Linear::can_merge(const std::vector<const Exl3Linear*>& ps) {
    if (ps.size() < 2 || ps.size() > 4) return false;
    for (auto* p : ps)
        if (p->S != 1 || p->lora || p->in != ps[0]->in || p->K != ps[0]->K || p->cb != ps[0]->cb ||
            p->trellis.size(2) != ps[0]->trellis.size(2) || p->out % 128)
            return false;
    return true;
}

namespace {
// dst rows of dst_pitch uint32 <- src rows of width uint32 (a multiple of 64), 64 a work-item (ESIMD: this library holds no
// plain SYCL kernels, and ext_oneapi_memcpy2d's fallback kernel is not in the AOT image)
struct Copy2DKernel {
    const uint32_t* src; uint32_t* dst;
    size_t src_pitch, dst_pitch;
    int width, rows;
    void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
        const int per = width / 64;
        const int id = it.get_global_id(0), r = id / per, c = (id % per) * 64;
        if (r >= rows) return;
        simd<uint32_t, 64> v = block_load<uint32_t, 64>(src + (size_t)r * src_pitch + c);
        block_store<uint32_t, 64>(dst + (size_t)r * dst_pitch + c, v);
    }
};
}  // namespace

Exl3Linear Exl3Linear::merge(const std::vector<Exl3Linear*>& ps) {
    GLM_CHECK(can_merge(std::vector<const Exl3Linear*>(ps.begin(), ps.end())), "Exl3Linear::merge: parts that do not match");
    auto& q = Dev::get().q;
    Exl3Linear m;
    m.S = (int)ps.size();
    m.in = ps[0]->in;
    m.K = ps[0]->K;
    m.cb = ps[0]->cb;
    for (int i = 0; i < m.S; i++) m.shard_col[i + 1] = m.shard_col[i] + ps[i]->out;
    m.out = m.shard_col[m.S];
    const int rows_t = m.in / 16, words = (int)ps[0]->trellis.size(2), tn = m.out / 16;
    const size_t tb = (size_t)words * 2;      // bytes a tile
    m.trellis = Tensor::empty({rows_t, tn, words}, DType::I16);
    m.suh = Tensor::empty({(int64_t)m.S * m.in}, DType::F16);
    m.svh = Tensor::empty({m.out}, DType::F16);
    std::vector<int> sonb(m.out / 128);
    for (int i = 0; i < m.S; i++) {
        const Exl3Linear& p = *ps[i];
        // each tile row's columns of part i, side by side with the others'
        {
            const int width = (int)((size_t)p.out / 16 * tb / 4);
            GLM_CHECK(width % 64 == 0, "Exl3Linear::merge: %d words a row", width);
            Copy2DKernel k{p.trellis.ptr<uint32_t>(), m.trellis.ptr<uint32_t>() + (size_t)m.shard_col[i] / 16 * tb / 4,
                           (size_t)width, (size_t)tn * tb / 4, width, rows_t};
            const size_t n = (size_t)rows_t * (width / 64);
            q.submit([&](sycl::handler& h) { h.parallel_for(sycl::nd_range<1>(round_up((int)n, 8), 8), k); });
        }
        q.memcpy(m.suh.ptr<half>() + (size_t)i * m.in, p.suh.p, (size_t)m.in * 2);
        q.memcpy(m.svh.ptr<half>() + m.shard_col[i], p.svh.p, (size_t)p.out * 2);
        for (int b = m.shard_col[i] / 128; b < m.shard_col[i + 1] / 128; b++) sonb[b] = i;
    }
    m.sonb = Tensor::empty({(int64_t)sonb.size()}, DType::I32);
    q.memcpy(m.sonb.p, sonb.data(), sonb.size() * 4).wait();
    for (auto* p : ps) *p = Exl3Linear();       // the parts' weights now live in m
    return m;
}

void Exl3Linear::forward(sycl::queue& q, Workspace& ws, const half* x, int M, void* y, DType ydt, int x_stride,
                         int n_cols) const {
    if (x_stride < 0) x_stride = in;
    GLM_CHECK(M >= 1, "Exl3Linear: M=%d", M);
    GLM_CHECK(S == 1, "Exl3Linear: a merged linear takes forward_split");
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
                              int x_stride, int n_cols, void* const* ys) const {
    if (M > exl3_large_above) return forward_large(q, ws, x, M, y, ydt, x_stride, ys);
    const int Kdim = in, N = n_cols >= 0 ? n_cols : out, tn = out / 16;
    static const int vec_env = getenv("GLM_EXL3_VECMAX") ? atoi(getenv("GLM_EXL3_VECMAX")) : -1;
    // K under 3 packs few values into a vector load: 8 rows of Ripple's whole model at K 2 / 3 / 4, 143 ms a step, DPAS 90 (at flat K 3: 98, DPAS 127)
    const int vec_max = vec_env >= 0 ? vec_env : K < 3 || (K >= 10 && K < 25) ? std::min(exl3_vec_max, 4) : exl3_vec_max;
    bool vec = M <= vec_max;
    int MR = M <= 1 ? 1 : M == 2 ? 2 : M == 3 ? 3 : M == 4 ? 4 : 8;
    // 33..40 rows take the 48-row block: the 40-row block gives wrong sums at K=4 (exl3_check M=33..40 on Qwen3.8's
    // in_proj_z and o_proj: rel err 0.42; K=5 was right; found 2026-11-19); GLM_EXL3_MB40=1 brings it back for a test
    static const bool mb40 = getenv("GLM_EXL3_MB40") && atoi(getenv("GLM_EXL3_MB40")) != 0;
    int MB = M <= 4 ? 4 : M <= 8 ? 8 : (M <= 16 ? 16 : (M <= 24 ? 24 : (M <= 32 ? 32 : (mb40 && M <= 40 ? 40 : (M <= 48 ? 48 : 64)))));
    int m_blocks = vec ? 1 : (M + MB - 1) / MB;
    int Mp = vec ? M : m_blocks * MB;
    // blocked activations [1][Kdim/16][Mp][16]; DPAS pad rows must be zero
    fp16* xh = ws.take<fp16>((size_t)S * Mp * Kdim);
    {
        // the pad rows' zeros come from the same launch (they were a memset before it); a merged linear: S shards
        HadInKernel<fp16> k{x, suh.ptr<fp16>(), xh, M, Kdim, S, x_stride, Mp, 0, 1};
        int n = S * Mp * (Kdim / 128);
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
        set_shards(k, ys);
        q.submit([&](sycl::handler& h) { h.parallel_for(sycl::nd_range<1>(round_up(n, 8), 8), k); });
    } else {
        GLM_CHECK(ydt == DType::F32, "Exl3Linear: output dtype");
        HadOutKernel<float> k{part, svh.ptr<fp16>(), static_cast<float*>(y), M, N, P, N};
        set_shards(k, ys);
        q.submit([&](sycl::handler& h) { h.parallel_for(sycl::nd_range<1>(round_up(n, 8), 8), k); });
    }
}

}  // namespace glm
