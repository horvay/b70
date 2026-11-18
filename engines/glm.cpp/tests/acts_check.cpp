// One token through the engine vs the eager cache-less forward of the PyTorch stack (llm/glm/xpu/tests/dumpacts.py):
// relative error per layer of the attention input / output, MLP input / output and the residual streams.
//   acts_check <model dir> <store .bin> <ref prefix>
#include <cmath>
#include <cstdio>
#include <fstream>
#include "../src/model.h"

using namespace glm;

static double rel(const float* a, const float* b, int n) {
    double num = 0, den = 0;
    for (int i = 0; i < n; i++) { double d = a[i] - b[i]; num += d * d; den += (double)b[i] * b[i]; }
    return std::sqrt(num / (den + 1e-30));
}

int main(int argc, char** argv) {
    GLM_CHECK(argc >= 4, "usage: acts_check <model dir> <store .bin> <ref prefix>");
    std::string ref = argv[3];
    Json j = Json::parse(read_file(ref + ".json"));
    int tok = j["token"].i(), NL = j["layers"].i(), V = j["vocab"].i();
    const int per = 4 * 4096 + 4 * 4096;
    std::vector<float> r((size_t)NL * per), lpr(V);
    std::ifstream(ref + ".f32", std::ios::binary).read(reinterpret_cast<char*>(r.data()), r.size() * 4);
    std::ifstream(ref + ".lp.f32", std::ios::binary).read(reinterpret_cast<char*>(lpr.data()), V * 4);
    TierConfig tc;
    tc.bonus_vram = tc.bonus_ram = 0;          // exact routing
    tc.ram_bytes = 4ull << 30;
    tc.vram_bytes = 8ull << 30;
    Model m(argv[1], argv[2], 4096, tc);
    std::vector<float> d;
    m.dump = &d;
    const half* lg = m.forward(&tok, 1);
    std::vector<sycl::half> l(m.config().vocab);
    Dev::get().q.memcpy(l.data(), lg, l.size() * 2).wait();
    GLM_CHECK(d.size() == r.size(), "dump size %zu vs %zu", d.size(), r.size());
    const char* nm[5] = {"attn_in", "attn_out", "mlp_in", "mlp_out", "R"};
    const int off[5] = {0, 4096, 8192, 12288, 16384}, len[5] = {4096, 4096, 4096, 4096, 16384};
    for (int L = 0; L < NL; L++) {
        printf("layer %2d %s:", L, m.config().mla[L] ? "MLA" : "KDA");
        for (int k = 0; k < 5; k++)
            printf("  %s %.1e", nm[k], rel(&d[(size_t)L * per + off[k]], &r[(size_t)L * per + off[k]], len[k]));
        printf("\n");
    }
    double mx = -1e30, se = 0, kl = 0;
    for (int v = 0; v < V; v++) mx = std::max(mx, (double)(float)l[v]);
    for (int v = 0; v < V; v++) se += std::exp((double)(float)l[v] - mx);
    for (int v = 0; v < V; v++)
        if (lpr[v] > -1e4) kl += std::exp((double)lpr[v]) * (lpr[v] - ((float)l[v] - mx - std::log(se)));
    printf("ACTS CHECK: final KL %.4f\n", kl);
    // injected: every sublayer on the reference's own inputs (no compounding); the mix of layer L reads the reference R
    // of layer L-1 (layer 0: the embedding streams, skipped)
    printf("injected inputs (rel err of the sublayer output alone):\n");
    for (int L = 0; L < NL; L++) {
        const float* b = &r[(size_t)L * per];
        std::vector<half> yi(4096), ym(4096), yo;
        for (int i = 0; i < 4096; i++) { yi[i] = half(b[i]); ym[i] = half(b[8192 + i]); }
        std::vector<float> ao, mo;
        std::vector<int> picks;
        m.probe_attn(L, yi, ao);
        m.probe_mlp(L, ym, mo, picks);
        printf("layer %2d %s: attn %.1e  mlp %.1e", L, m.config().mla[L] ? "MLA" : "KDA", rel(ao.data(), b + 4096, 4096),
               rel(mo.data(), b + 12288, 4096));
        if (L > 0) {
            std::vector<float> Rp(r.begin() + (size_t)(L - 1) * per + 16384, r.begin() + (size_t)(L - 1) * per + 32768);
            m.probe_mix(L, 0, Rp, yo);
            std::vector<float> yf(4096);
            for (int i = 0; i < 4096; i++) yf[i] = yo[i];
            printf("  mix %.1e", rel(yf.data(), b, 4096));
        }
        if (!picks.empty()) {
            printf("  picks");
            for (int p : picks) printf(" %d", p);
        }
        printf("\n");
    }
}
