// Edge0 kernels against host references on the release's own weights:
//   e0_check <model dir> [experts.bin] [layer]
// 1. Q8_0 matvec (1, 5, 16 rows) and the dequantize + GEMM path (40 rows) on the layer's D and an attention/GDN weight
// 2. the NVFP4 record tails (global scales) are well formed
// 3. one expert, decode kernels (VRAM and pinned-RAM record) and the prompt path (dequantize, GEMMs), against the host
// 4. the whole latent MoE of a layer (D, 10 picks, combine, U) for 3 rows against the host
#include <fcntl.h>
#include <unistd.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <random>
#include <oneapi/mkl/blas.hpp>
#include "../src/e0_ops.h"
#include "../src/gguf.h"
#include "../src/q4_ops.h"
#include "../src/st.h"

using namespace glm;
using namespace glm::e0ops;

static int g_bad = 0;
static void report(const char* what, const std::vector<float>& got, const std::vector<float>& ref, double tol) {
    double num = 0, den = 0, mx = 0;
    for (size_t i = 0; i < ref.size(); i++) {
        double d = (double)got[i] - ref[i];
        num += d * d;
        den += (double)ref[i] * ref[i];
        mx = std::max(mx, std::fabs(d));
    }
    double rel = std::sqrt(num / (den + 1e-30));
    bool ok = rel <= tol && std::isfinite(rel);
    g_bad += !ok;
    printf("  %-44s rel %.2e  max|d| %.3g  %s\n", what, rel, mx, ok ? "ok" : "FAIL");
}

static float h2f(uint16_t b) { return (float)sycl::bit_cast<half>(b); }
// ggml's own definitions (ggml-common.h kvalues_mxfp4, ggml-impl.h ggml_ue4m3_to_fp32), written out independently
static const int8_t KV4[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
static float ue4m3_half(uint8_t x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    int e = (x >> 3) & 0xF, m = x & 7;
    float raw = e == 0 ? std::ldexp((float)m, -9) : std::ldexp(1.0f + m / 8.0f, e - 7);
    return raw * 0.5f;
}
static void nv_row(const uint8_t* row, int n, float* out) {        // dequantize_row_nvfp4
    for (int b = 0; b < n / 64; b++) {
        const uint8_t* blk = row + b * 36;
        for (int s = 0; s < 4; s++) {
            float d = ue4m3_half(blk[s]);
            for (int j = 0; j < 8; j++) {
                uint8_t q = blk[4 + s * 8 + j];
                out[b * 64 + s * 16 + j] = KV4[q & 15] * d;
                out[b * 64 + s * 16 + 8 + j] = KV4[q >> 4] * d;
            }
        }
    }
}
static void q8_row(const uint8_t* row, int K, float* out) {
    for (int b = 0; b < K / 32; b++) {
        uint16_t d;
        memcpy(&d, row + b * 34, 2);
        for (int j = 0; j < 32; j++) out[b * 32 + j] = h2f(d) * (int8_t)row[b * 34 + 2 + j];
    }
}
static float silu(float x) { return x / (1 + std::exp(-x)); }

// host expert: h = silu(sg G x) * (su U x), y = sd Dn h
static void expert_ref(const uint8_t* rec, const float* x, float* y, float* hout = nullptr) {
    float sc[3];
    memcpy(sc, rec + NV_TAIL, 12);
    std::vector<float> w(ELL), h(EI);
    for (int j = 0; j < EI; j++) {
        double g = 0, u = 0;
        nv_row(rec + (size_t)j * NV_ROW_GU, ELL, w.data());
        for (int c = 0; c < ELL; c++) g += (double)w[c] * x[c];
        nv_row(rec + (size_t)(EI + j) * NV_ROW_GU, ELL, w.data());
        for (int c = 0; c < ELL; c++) u += (double)w[c] * x[c];
        h[j] = silu(sc[0] * (float)g) * (sc[1] * (float)u);
    }
    if (hout) memcpy(hout, h.data(), EI * 4);
    std::vector<float> wd(EI);
    for (int c = 0; c < ELL; c++) {
        nv_row(rec + NV_DOWN_OFF + (size_t)c * NV_ROW_D, EI, wd.data());
        double a = 0;
        for (int j = 0; j < EI; j++) a += (double)wd[j] * h[j];
        y[c] = sc[2] * (float)a;
    }
}

int main(int argc, char** argv) {
    GLM_CHECK(argc >= 2, "usage: e0_check <model dir> [experts.bin] [layer]");
    std::string dir = argv[1];
    if (dir == "bench") {   // e0_check bench: kernel times for 1..8 rows (synthetic weights)
        auto& q = Dev::get().q;
        auto timeit = [&](auto fn) {
            fn(); q.wait();
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < 20; i++) fn();
            q.wait();
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 20;
        };
        for (auto [N, K] : {std::pair<int, int>{GE, 2560}, {2560, GE}, {10240, 2560}}) {
            Q8W w;
            w.N = N; w.K = K;
            w.qs = Tensor::empty({N, K}, DType::U8);
            w.d = Tensor::zeros({N, K / 32}, DType::F16);
            {   // random bytes: zeros compress or cache, and read faster than real weights
                uint8_t* p = w.qs.ptr<uint8_t>();
                q.parallel_for(sycl::range<1>((size_t)N * K), [=](sycl::id<1> i) {
                    uint64_t z = (uint64_t)i * 0x9E3779B97F4A7C15ull;
                    z ^= z >> 29; z *= 0xBF58476D1CE4E5B9ull; z ^= z >> 32;
                    p[i] = (uint8_t)z;
                }).wait();
            }
            Tensor x = Tensor::zeros({16, K}, DType::F32), y = Tensor::empty({16, N}, DType::F32);
            printf("q8_matvec [%d, %d] (%.1f MB):", N, K, w.bytes() / 1e6);
            for (int M : {1, 2, 4, 8})
                printf("  M=%d %.3f ms", M, timeit([&] { q8_matvec<float, float>(q, x.ptr<float>(), M, w, y.ptr<float>()); }));
            printf("\n");
        }
        // routed experts: 512 records in VRAM, M rows of 10 distinct picks (rows share none)
        char* recs = static_cast<char*>(sycl::malloc_device((size_t)512 * NV_REC, q));
        q.parallel_for(sycl::range<1>((size_t)512 * NV_REC), [=](sycl::id<1> i) {
            uint64_t z = (uint64_t)i * 0x9E3779B97F4A7C15ull;
            z ^= z >> 29; z *= 0xBF58476D1CE4E5B9ull; z ^= z >> 32;
            recs[i] = (char)(z & 0x77);                 // keeps the UE4M3 scales and tails finite
        }).wait();
        std::vector<int64_t> pt(512);
        for (int e = 0; e < 512; e++) pt[e] = (int64_t)(recs + (size_t)e * NV_REC);
        Tensor ptd = Tensor::empty({512}, DType::I64), lab = Tensor::zeros({512}, DType::I32);
        upload(ptd.p, pt.data(), 4096);
        std::vector<int> sl(160);
        for (int i = 0; i < 160; i++) sl[i] = (i * 37) % 512;
        Tensor sld = Tensor::empty({160}, DType::I32), lead = Tensor::empty({160}, DType::I32);
        upload(sld.p, sl.data(), 640);
        Tensor xl = Tensor::zeros({16, GE}, DType::F32), h = Tensor::empty({160, EI}, DType::F32), yl = Tensor::empty({160, ELL}, DType::F32);
        printf("experts (VRAM):");
        for (int M : {1, 2, 4, 8}) {
            double a = timeit([&] { nv_gate_up(q, xl.ptr<float>(), sld.ptr<int>(), lab.ptr<int>(), ptd.ptr<int64_t>(), M, h.ptr<float>());
                                    nv_down(q, h.ptr<float>(), sld.ptr<int>(), ptd.ptr<int64_t>(), M, yl.ptr<float>()); });
            double b = timeit([&] { pick_leaders(q, sld.ptr<int>(), M * 10, lead.ptr<int>());
                                    nv_gate_up_g(q, xl.ptr<float>(), sld.ptr<int>(), lead.ptr<int>(), lab.ptr<int>(), ptd.ptr<int64_t>(), M, h.ptr<float>());
                                    nv_down_g(q, h.ptr<float>(), sld.ptr<int>(), lead.ptr<int>(), ptd.ptr<int64_t>(), M, yl.ptr<float>()); });
            double c = timeit([&] { nv_gate_up_t(q, xl.ptr<float>(), sld.ptr<int>(), nullptr, lab.ptr<int>(), ptd.ptr<int64_t>(), M, h.ptr<float>());
                                    nv_down_t(q, h.ptr<float>(), sld.ptr<int>(), nullptr, ptd.ptr<int64_t>(), M, yl.ptr<float>()); });
            printf("\n  M=%d raw %.3f / raw grouped %.3f / tiled %.3f ms (%.0f / %.0f / %.0f GB/s)", M, a, b, c,
                   M * 10 * NV_REC / a / 1e6, M * 10 * NV_REC / b / 1e6, M * 10 * NV_REC / c / 1e6);
        }
        printf("\n");
        sycl::free(recs, q);
        return 0;
    }
    std::string ep = argc > 2 ? argv[2] : dir + "/experts.bin";
    int layer = argc > 3 ? atoi(argv[3]) : 0;
    auto& q = Dev::get().q;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    auto randv = [&](size_t n, float s = 1.0f) { std::vector<float> v(n); for (auto& x : v) x = nd(rng) * s; return v; };
    auto dev = [&](const std::vector<float>& v) { Tensor t = Tensor::empty({(int64_t)v.size()}, DType::F32); upload(t.p, v.data(), v.size() * 4); return t; };
    auto dev16 = [&](const std::vector<float>& v) {
        std::vector<half> h(v.begin(), v.end());
        Tensor t = Tensor::empty({(int64_t)v.size()}, DType::F16);
        upload(t.p, h.data(), h.size() * 2);
        return t;
    };
    auto back = [&](const Tensor& t, size_t n) {
        std::vector<float> v(n);
        if (t.dt == DType::F16) { std::vector<half> h(n); download(h.data(), t.p, n * 2); for (size_t i = 0; i < n; i++) v[i] = h[i]; }
        else download(v.data(), t.p, n * 4);
        return v;
    };
    auto r16 = [](std::vector<float> v) { for (auto& x : v) x = (float)half(x); return v; };   // the fp16 inputs

    // ---------------------------------------------------------------- 1. Q8_0
    printf("1. Q8_0 linears\n");
    Gguf g(dir + "/model-skeleton.gguf");
    const std::string wn = layer % 4 == 3 ? "attn_output.weight" : "attn_qkv.weight";
    const GgufTensor& wt = g.at("blk." + std::to_string(layer) + "." + wn);
    const int N = (int)wt.rows(), K = (int)wt.cols();
    Q8W w;
    {
        // the same split as the model's loader, done here by hand
        const uint8_t* src = static_cast<const uint8_t*>(wt.data);
        std::vector<int8_t> qs((size_t)N * K);
        std::vector<uint16_t> d((size_t)N * K / 32);
        for (size_t b = 0; b < (size_t)N * K / 32; b++) { memcpy(&d[b], src + b * 34, 2); memcpy(&qs[b * 32], src + b * 34 + 2, 32); }
        w.N = N; w.K = K;
        w.qs = Tensor::empty({N, K}, DType::U8);
        w.d = Tensor::empty({N, K / 32}, DType::F16);
        upload(w.qs.p, qs.data(), qs.size());
        upload(w.d.p, d.data(), d.size() * 2);
    }
    std::vector<float> wrow(K);
    auto q8ref = [&](const std::vector<float>& x, int M) {
        std::vector<float> y((size_t)M * N);
        for (int n = 0; n < N; n++) {
            q8_row(static_cast<const uint8_t*>(wt.data) + (size_t)n * (K / 32) * 34, K, wrow.data());
            for (int m = 0; m < M; m++) {
                double a = 0;
                for (int k = 0; k < K; k++) a += (double)wrow[k] * x[(size_t)m * K + k];
                y[(size_t)m * N + n] = (float)a;
            }
        }
        return y;
    };
    Tensor scratch = Tensor::empty({(int64_t)N * K}, DType::F16);
    for (int M : {1, 5, 16, 40}) {
        auto x = r16(randv((size_t)M * K));
        auto ref = q8ref(x, M);
        Tensor xd = dev16(x), yd = Tensor::empty({M, N}, DType::F32);
        q8_linear<float>(q, xd.ptr<half>(), M, w, yd.ptr<float>(), scratch.ptr<half>());
        q.wait();
        char nm[96];
        snprintf(nm, sizeof nm, "%s, %d rows (%s)", wn.c_str(), M, M <= MV_MAX ? "matvec" : "GEMM");
        report(nm, back(yd, (size_t)M * N), ref, M <= MV_MAX ? 1e-5 : 2e-3);
    }
    {   // fp32 input, as the U projection takes it at decode
        auto x = randv((size_t)3 * K);
        auto ref = q8ref(x, 3);
        Tensor xd = dev(x), yd = Tensor::empty({3, N}, DType::F32);
        q8_matvec<float, float>(q, xd.ptr<float>(), 3, w, yd.ptr<float>());
        q.wait();
        report("fp32 input, 3 rows", back(yd, (size_t)3 * N), ref, 1e-5);
    }

    // ---------------------------------------------------------------- 2. record tails
    printf("2. NVFP4 records of layer %d\n", layer);
    int efd = open(ep.c_str(), O_RDONLY);
    GLM_CHECK(efd >= 0, "cannot open %s", ep.c_str());
    off_t esize = lseek(efd, 0, SEEK_END);
    const off_t lbase = (off_t)layer * 512 * (off_t)NV_REC;
    GLM_CHECK(esize >= lbase + 512 * (off_t)NV_REC, "%s holds %lld bytes: layer %d is not in it yet", ep.c_str(),
              (long long)esize, layer);
    std::vector<uint8_t> recs((size_t)512 * NV_REC);
    GLM_CHECK(pread(efd, recs.data(), recs.size(), lbase) == (ssize_t)recs.size(), "read layer");
    {
        // the record tails: {s_gate, s_up, s_down, 0}, positive and finite, s_gate == s_up (the pack's convention).
        // dense.bin's ffn_*_exps.scale are NOT these: index.txt says Strata packed it "from model.gguf", the source
        // model's full-width experts (scales ~7e-5 against the latent experts' ~3e-3 / 1e-4)
        int bad = 0;
        double mg = 0, md = 0;
        for (int e = 0; e < 512; e++) {
            float t[4];
            memcpy(t, recs.data() + (size_t)e * NV_REC + NV_TAIL, 16);
            bad += !(std::isfinite(t[0]) && t[0] > 0 && t[1] == t[0] && std::isfinite(t[2]) && t[2] > 0 && t[3] == 0);
            mg += t[0] / 512;
            md += t[2] / 512;
        }
        g_bad += bad != 0;
        printf("  record tails: %d of 512 malformed (mean s_gate %.3g, s_down %.3g)  %s\n", bad, mg, md, bad ? "FAIL" : "ok");
    }

    // ---------------------------------------------------------------- 3. one expert
    printf("3. one expert (decode kernels from VRAM and from pinned RAM; prompt path)\n");
    const int ex = 37;
    const uint8_t* rec = recs.data() + (size_t)ex * NV_REC;
    char* rv = static_cast<char*>(sycl::malloc_device(NV_REC, q));
    char* rh = static_cast<char*>(sycl::malloc_host(NV_REC, q));
    std::vector<char> tiled_rec(NV_REC);
    nv_repack(reinterpret_cast<const char*>(rec), tiled_rec.data());     // the engine's layout
    q.memcpy(rv, tiled_rec.data(), NV_REC).wait();
    memcpy(rh, tiled_rec.data(), NV_REC);
    {
        // xl: one row; label 0 for every expert so x = xl[0 .. ELL)
        auto xl = randv(GE, 0.5f);
        std::vector<float> ref_y(ELL), ref_h(EI);
        expert_ref(rec, xl.data(), ref_y.data(), ref_h.data());
        std::vector<int> lab(512, 0), sl(10, ex);
        Tensor xld = dev(xl), labd = Tensor::empty({512}, DType::I32), sld = Tensor::empty({10}, DType::I32);
        upload(labd.p, lab.data(), 2048);
        upload(sld.p, sl.data(), 40);
        for (int where = 0; where < 2; where++) {
            std::vector<int64_t> pt(512, (int64_t)(where ? rh : rv));
            Tensor ptd = Tensor::empty({512}, DType::I64);
            upload(ptd.p, pt.data(), 4096);
            Tensor hd = Tensor::empty({10, EI}, DType::F32), yd = Tensor::empty({10, ELL}, DType::F32);
            nv_gate_up_t(q, xld.ptr<float>(), sld.ptr<int>(), nullptr, labd.ptr<int>(), ptd.ptr<int64_t>(), 1, hd.ptr<float>());
            nv_down_t(q, hd.ptr<float>(), sld.ptr<int>(), nullptr, ptd.ptr<int64_t>(), 1, yd.ptr<float>());
            q.wait();
            auto hh = back(hd, EI), yy = back(yd, ELL);
            report(where ? "gate/up (RAM record)" : "gate/up (VRAM record)", hh, ref_h, 1e-5);
            report(where ? "down (RAM record)" : "down (VRAM record)", yy, ref_y, 1e-5);
        }
        // prompt path: 7 rows of xl (fp16), all to this expert
        const int n = 7;
        auto A = r16(randv((size_t)n * ELL, 0.5f));
        std::vector<float> ref((size_t)n * ELL);
        for (int r = 0; r < n; r++) expert_ref(rec, A.data() + (size_t)r * ELL, ref.data() + (size_t)r * ELL);
        Tensor Ad = dev16(A), gu = Tensor::empty({2 * EI, ELL}, DType::F16), dn = Tensor::empty({ELL, EI}, DType::F16);
        Tensor GU = Tensor::empty({n, 2 * EI}, DType::F32), h16 = Tensor::empty({n, EI}, DType::F16);
        Tensor Y = Tensor::empty({n, ELL}, DType::F32);
        float sd;
        memcpy(&sd, rec + NV_TAIL + 8, 4);
        nv_dequant(q, rh, gu.ptr<half>(), dn.ptr<half>());
        oneapi::mkl::blas::row_major::gemm(q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::trans, n, 2 * EI,
                                           ELL, 1.0f, Ad.ptr<half>(), ELL, gu.ptr<half>(), ELL, 0.0f, GU.ptr<float>(), 2 * EI);
        nv_act(q, GU.ptr<float>(), rh, n, h16.ptr<half>());
        oneapi::mkl::blas::row_major::gemm(q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::trans, n, ELL, EI,
                                           sd, h16.ptr<half>(), EI, dn.ptr<half>(), EI, 0.0f, Y.ptr<float>(), ELL);
        q.wait();
        report("prompt path, 7 rows (fp16 h)", back(Y, (size_t)n * ELL), ref, 2e-3);
        // the DPAS kernels: one expert, 45 rows (a full block of 32 and a partial one), X fp16
        {
            const int nr = 45;
            auto Xh = r16(randv((size_t)nr * ELL, 0.5f));
            std::vector<float> refY((size_t)nr * ELL), refH((size_t)nr * EI);
            for (int r = 0; r < nr; r++)
                expert_ref(rec, Xh.data() + (size_t)r * ELL, refY.data() + (size_t)r * ELL, refH.data() + (size_t)r * EI);
            Tensor Xd = dev16(Xh), Hd = Tensor::empty({nr, EI}, DType::F16), Yd = Tensor::empty({nr, ELL}, DType::F16);
            std::vector<int64_t> brec = {(int64_t)rv, (int64_t)rh};      // block 0 from VRAM, block 1 from pinned RAM
            std::vector<int> br0 = {0, 32}, bn = {32, nr - 32};
            Tensor bd = Tensor::empty({2}, DType::I64), b0 = Tensor::empty({2}, DType::I32), b1 = Tensor::empty({2}, DType::I32);
            upload(bd.p, brec.data(), 16);
            upload(b0.p, br0.data(), 8);
            upload(b1.p, bn.data(), 8);
            nv_gate_up_dpas(q, Xd.ptr<half>(), nr, bd.ptr<int64_t>(), b0.ptr<int>(), b1.ptr<int>(), 2, Hd.ptr<half>());
            q.wait();
            report("DPAS gate/up, 45 rows (h vs host)", back(Hd, (size_t)nr * EI), refH, 2e-3);
            // down from the host's own h (fp16), so its error is the down kernel's alone
            Tensor Hh = dev16(r16(refH));
            std::vector<float> refY2((size_t)nr * ELL);
            {
                float sc[3];
                memcpy(sc, rec + NV_TAIL, 12);
                std::vector<float> wd(EI);
                auto hr = r16(refH);
                for (int c = 0; c < ELL; c++) {
                    nv_row(rec + NV_DOWN_OFF + (size_t)c * NV_ROW_D, EI, wd.data());
                    for (int r = 0; r < nr; r++) {
                        double a = 0;
                        for (int j = 0; j < EI; j++) a += (double)wd[j] * hr[(size_t)r * EI + j];
                        refY2[(size_t)r * ELL + c] = sc[2] * (float)a;
                    }
                }
            }
            nv_down_dpas(q, Hh.ptr<half>(), nr, bd.ptr<int64_t>(), b0.ptr<int>(), b1.ptr<int>(), 2, Yd.ptr<half>());
            q.wait();
            report("DPAS down, 45 rows (y vs host)", back(Yd, (size_t)nr * ELL), refY2, 2e-3);
        }
        // dequantized weights must be exact
        std::vector<float> wref((size_t)2 * EI * ELL);
        for (int j = 0; j < 2 * EI; j++) nv_row(rec + (size_t)j * NV_ROW_GU, ELL, wref.data() + (size_t)j * ELL);
        auto wg = back(gu, wref.size());
        // ggml's values carry the 0.5 in the scale and 2x in the codes: the products are equal
        size_t diff = 0;
        for (size_t i = 0; i < wref.size(); i++) diff += wg[i] != wref[i];
        g_bad += diff != 0;
        printf("  dequantized gate/up vs ggml's dequantize_row_nvfp4: %zu of %zu differ  %s\n", diff, wref.size(),
               diff ? "FAIL" : "ok");
    }

    // ---------------------------------------------------------------- 4. the latent MoE of a layer
    printf("4. latent MoE, layer %d, 3 rows\n", layer);
    {
        const int M = 3;
        const size_t d_bytes = (size_t)GE * (2560 / 32) * 34, u_bytes = (size_t)G * 2560 * (ELL / 32) * 34;
        std::vector<uint8_t> dbuf(d_bytes), ubuf(u_bytes);
        int fd = open((dir + "/latent_dense_q8.bin").c_str(), O_RDONLY);
        GLM_CHECK(fd >= 0, "latent_dense_q8.bin");
        off_t lo = (off_t)layer * (d_bytes + u_bytes);
        GLM_CHECK(pread(fd, dbuf.data(), d_bytes, lo) == (ssize_t)d_bytes && pread(fd, ubuf.data(), u_bytes, lo + d_bytes) == (ssize_t)u_bytes, "D/U read");
        close(fd);
        std::string lb = read_file(dir + "/labels.bin");
        std::vector<int> lab(512);
        memcpy(lab.data(), lb.data() + (size_t)layer * 2048, 2048);
        // picks: 10 distinct experts a row, weights a softmax of random logits
        std::vector<int> sl(M * 10);
        std::vector<float> wv(M * 10);
        for (int r = 0; r < M; r++) {
            std::vector<int> all(512);
            for (int e = 0; e < 512; e++) all[e] = e;
            std::shuffle(all.begin(), all.end(), rng);
            float s = 0;
            // row 2 reuses row 0's experts (a verify window's shared picks), in another order
            for (int k = 0; k < 10; k++) {
                sl[r * 10 + k] = r == 2 ? sl[9 - k] : all[k];
                wv[r * 10 + k] = std::exp(nd(rng));
                s += wv[r * 10 + k];
            }
            for (int k = 0; k < 10; k++) wv[r * 10 + k] /= s;
        }
        auto y = r16(randv((size_t)M * 2560, 0.3f));
        // host: xl = D y per group; per pick the expert on its group's xl; z by group; out = sum_g U_g z_g
        std::vector<float> ref((size_t)M * 2560, 0.0f), rowv(2560), urow(ELL);
        for (int r = 0; r < M; r++) {
            std::vector<float> xl(GE), z(GE, 0.0f);
            for (int o = 0; o < GE; o++) {
                q8_row(dbuf.data() + (size_t)o * 80 * 34, 2560, rowv.data());
                double a = 0;
                for (int k = 0; k < 2560; k++) a += (double)rowv[k] * y[(size_t)r * 2560 + k];
                xl[o] = (float)a;
            }
            for (int k = 0; k < 10; k++) {
                int e = sl[r * 10 + k], gg = lab[e];
                std::vector<float> ye(ELL);
                expert_ref(recs.data() + (size_t)e * NV_REC, xl.data() + (size_t)gg * ELL, ye.data());
                for (int c = 0; c < ELL; c++) z[(size_t)gg * ELL + c] += wv[r * 10 + k] * ye[c];
            }
            for (int n = 0; n < 2560; n++) {
                double a = 0;
                for (int gg = 0; gg < G; gg++) {
                    q8_row(ubuf.data() + ((size_t)gg * 2560 + n) * 56 * 34, ELL, urow.data());
                    for (int c = 0; c < ELL; c++) a += (double)urow[c] * z[(size_t)gg * ELL + c];
                }
                ref[(size_t)r * 2560 + n] = (float)a;
            }
        }
        // device: the model's own layout (D as stored, U's groups side by side)
        auto split = [&](const uint8_t* src, int Nn, int Kk) {
            Q8W ww;
            ww.N = Nn; ww.K = Kk;
            std::vector<int8_t> qs((size_t)Nn * Kk);
            std::vector<uint16_t> d((size_t)Nn * Kk / 32);
            for (size_t b = 0; b < d.size(); b++) { memcpy(&d[b], src + b * 34, 2); memcpy(&qs[b * 32], src + b * 34 + 2, 32); }
            ww.qs = Tensor::empty({Nn, Kk}, DType::U8);
            ww.d = Tensor::empty({Nn, Kk / 32}, DType::F16);
            upload(ww.qs.p, qs.data(), qs.size());
            upload(ww.d.p, d.data(), d.size() * 2);
            return ww;
        };
        std::vector<uint8_t> ucat(ubuf.size());
        const size_t ur = (size_t)56 * 34;
        for (int n = 0; n < 2560; n++)
            for (int gg = 0; gg < G; gg++) memcpy(&ucat[(size_t)n * G * ur + gg * ur], &ubuf[((size_t)gg * 2560 + n) * ur], ur);
        Q8W Dw = split(dbuf.data(), GE, 2560), Uw = split(ucat.data(), 2560, GE);
        // records: half in VRAM, half in pinned RAM
        char* vr = static_cast<char*>(sycl::malloc_device((size_t)512 * NV_REC, q));
        char* hr = static_cast<char*>(sycl::malloc_host((size_t)512 * NV_REC, q));
        for (int e = 0; e < 512; e++)
            nv_repack(reinterpret_cast<const char*>(recs.data()) + (size_t)e * NV_REC, hr + (size_t)e * NV_REC);
        q.memcpy(vr, hr, recs.size()).wait();
        std::vector<int64_t> pt(512);
        for (int e = 0; e < 512; e++) pt[e] = (int64_t)((e & 1 ? hr : vr) + (size_t)e * NV_REC);
        Tensor ptd = Tensor::empty({512}, DType::I64), labd = Tensor::empty({512}, DType::I32);
        Tensor sld = Tensor::empty({M * 10}, DType::I32), wd = dev(wv);
        upload(ptd.p, pt.data(), 4096);
        upload(labd.p, lab.data(), 2048);
        upload(sld.p, sl.data(), M * 40);
        Tensor yd = dev16(y), xl = Tensor::empty({M, GE}, DType::F32), h = Tensor::empty({M * 10, EI}, DType::F32);
        Tensor yl = Tensor::empty({M * 10, ELL}, DType::F32), z = Tensor::empty({M, GE}, DType::F32);
        Tensor out = Tensor::empty({M, 2560}, DType::F32);
        q8_matvec<half, float>(q, yd.ptr<half>(), M, Dw, xl.ptr<float>());
        nv_gate_up_t(q, xl.ptr<float>(), sld.ptr<int>(), nullptr, labd.ptr<int>(), ptd.ptr<int64_t>(), M, h.ptr<float>());
        nv_down_t(q, h.ptr<float>(), sld.ptr<int>(), nullptr, ptd.ptr<int64_t>(), M, yl.ptr<float>());
        latent_combine(q, yl.ptr<float>(), sld.ptr<int>(), wd.ptr<float>(), labd.ptr<int>(), M, z.ptr<float>());
        q8_matvec<float, float>(q, z.ptr<float>(), M, Uw, out.ptr<float>());
        q.wait();
        report("decode path (D, experts, combine, U)", back(out, (size_t)M * 2560), ref, 1e-4);
        Tensor lead = Tensor::empty({M * 10}, DType::I32);
        pick_leaders(q, sld.ptr<int>(), M * 10, lead.ptr<int>());
        nv_gate_up_t(q, xl.ptr<float>(), sld.ptr<int>(), lead.ptr<int>(), labd.ptr<int>(), ptd.ptr<int64_t>(), M, h.ptr<float>());
        nv_down_t(q, h.ptr<float>(), sld.ptr<int>(), lead.ptr<int>(), ptd.ptr<int64_t>(), M, yl.ptr<float>());
        latent_combine(q, yl.ptr<float>(), sld.ptr<int>(), wd.ptr<float>(), labd.ptr<int>(), M, z.ptr<float>());
        q8_matvec<float, float>(q, z.ptr<float>(), M, Uw, out.ptr<float>());
        q.wait();
        report("grouped kernels, row 2 = row 0's experts", back(out, (size_t)M * 2560), ref, 1e-4);
        sycl::free(vr, q);
        sycl::free(hr, q);
    }
    sycl::free(rv, q);
    sycl::free(rh, q);
    close(efd);
    printf("%s\n", g_bad ? "SOME CHECKS FAILED" : "all checks passed");
    return g_bad ? 1 : 0;
}
