// The EXL3 int8 prompt path (GLM_EXL3_INT8) against the fp16 one, on every EXL3 linear of a model: relative error of
// the int8 output (a pseudo-random input with a few strong channels, as RMSNorm outputs have) and the GPU time of each
// path at M rows. The sums are the linear-only time of one M-row prompt piece through the whole model.
//   exl3_i8_check <model dir> [M 2048] [iters 3] [name filter]
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include "../src/exl3.h"

using namespace glm;

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int main(int argc, char** argv) {
    GLM_CHECK(argc >= 2, "usage: exl3_i8_check <model dir> [M] [iters] [filter]");
    SafeTensors st(argv[1]);
    const int M = argc > 2 ? atoi(argv[2]) : 2048;
    const int iters = argc > 3 ? atoi(argv[3]) : 3;
    const char* filt = argc > 4 ? argv[4] : nullptr;
    auto& q = Dev::get().q;
    exl3_large_above = getenv("GLM_EXL3_LARGE") ? atoi(getenv("GLM_EXL3_LARGE")) : 32;   // 128: Qwen's decode-size paths below
    std::vector<std::string> pre;
    for (auto& n : st.names_with_prefix("")) {
        if (n.size() < 8 || n.compare(n.size() - 8, 8, ".trellis")) continue;
        std::string p = n.substr(0, n.size() - 8);
        if (p.find("lm_head") != std::string::npos || p.find("visual") != std::string::npos) continue;
        if (filt && p.find(filt) == std::string::npos) continue;
        pre.push_back(p);
    }
    printf("%zu EXL3 linears, M = %d\n", pre.size(), M);
    std::map<int, Tensor> xs;          // per input width: fp16 [M, in]
    Workspace ws;
    size_t ws_cap = 0;
    double t16 = 0, t8 = 0, worst = 0, sum_rel = 0;
    struct Agg { int n = 0; double t16 = 0, t8 = 0, rel = 0; };
    std::map<std::pair<int, int>, Agg> by_shape;
    int n_done = 0;
    std::mt19937 rng(7);
    for (auto& p : pre) {
        Exl3Linear L = Exl3Linear::load(st, p);
        if (L.cb != 2) { printf("%-60s codebook %d: skipped\n", p.c_str(), L.cb); continue; }
        if (!xs.count(L.in)) {
            std::normal_distribution<float> nd(0.f, 1.f);
            std::vector<half> x((size_t)M * L.in);
            std::vector<float> chan(L.in, 1.0f);
            for (int k = 0; k < 8; k++) chan[(rng() % L.in)] = 20.0f;       // a few outlier channels
            for (int m = 0; m < M; m++)
                for (int k = 0; k < L.in; k++) x[(size_t)m * L.in + k] = (half)(nd(rng) * chan[k] * 0.3f);
            xs[L.in] = Tensor::empty({M, L.in}, DType::F16);
            upload(xs[L.in].p, x.data(), x.size() * 2);
        }
        size_t need = L.ws_bytes(M) + (1 << 20);
        if (need > ws_cap) {
            if (ws.base) sycl::free(ws.base, q);
            ws.base = nullptr;
            ws.init(need);
            ws_cap = need;
        }
        Tensor y16 = Tensor::empty({M, L.out}, DType::F16), y8 = Tensor::empty({M, L.out}, DType::F16);
        const half* x = xs[L.in].ptr<half>();
        double tt[2];
        for (int i8 = 0; i8 < 2; i8++) {
            exl3_int8 = i8;
            Tensor& y = i8 ? y8 : y16;
            L.forward(q, ws, x, M, y.p, DType::F16);      // warm-up (and the GEMM primitive)
            q.wait();
            double a = now();
            for (int it = 0; it < iters; it++) L.forward(q, ws, x, M, y.p, DType::F16);
            q.wait();
            tt[i8] = (now() - a) / iters;
        }
        std::vector<half> a((size_t)M * L.out), b(a.size());
        download(a.data(), y8.p, a.size() * 2);
        download(b.data(), y16.p, b.size() * 2);
        double num = 0, den = 0;
        for (size_t i = 0; i < a.size(); i++) {
            double d = (double)(float)a[i] - (double)(float)b[i];
            num += d * d;
            den += (double)(float)b[i] * (float)b[i];
        }
        double rel = std::sqrt(num / std::max(den, 1e-30));
        bool finite = std::isfinite(rel);
        printf("%-60s %5d -> %5d K=%d  fp16 %7.3f ms  int8 %7.3f ms  (%.2fx)  rel err %.3e%s\n", p.c_str(), L.in, L.out, L.K,
               tt[0] * 1e3, tt[1] * 1e3, tt[0] / tt[1], rel, finite ? "" : "  NOT FINITE");
        fflush(stdout);
        t16 += tt[0];
        t8 += tt[1];
        worst = std::max(worst, finite ? rel : 1e9);
        sum_rel += rel;
        auto& g = by_shape[{L.in, L.out}];
        g.n++, g.t16 += tt[0], g.t8 += tt[1], g.rel += rel;
        n_done++;
    }
    printf("\nby shape (in -> out: count, fp16 / int8 ms each, mean rel err):\n");
    for (auto& [k, g] : by_shape)
        printf("  %5d -> %5d: %3d  %7.3f / %7.3f ms  (%.2fx)  %.3e\n", k.first, k.second, g.n, g.t16 * 1e3 / g.n,
               g.t8 * 1e3 / g.n, g.t16 / g.t8, g.rel / g.n);
    if (getenv("GLM_EXL3_I8_STAGES")) printf("int8 stages (warm-up included): %s\n", exl3_i8_stages_report().c_str());
    printf("all %d linears at M = %d: fp16 %.1f ms, int8 %.1f ms (%.2fx); rel err mean %.3e, worst %.3e\n", n_done, M,
           t16 * 1e3, t8 * 1e3, t16 / t8, sum_rel / std::max(1, n_done), worst);
    bool ok = worst < 3e-2;
    printf(ok ? "EXL3 INT8 CHECK OK\n" : "EXL3 INT8 CHECK FAILED\n");
    return ok ? 0 : 1;
}
