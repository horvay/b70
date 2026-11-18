// The engine's decode path against the PyTorch stack's (llm/glm/xpu/tests/dumpref.py): the same tokens one at a time,
// each step's distribution vs the reference's (KL(ref || engine), top-1 agreement), and the time per token.
//   decode_check <model dir> <store .bin> <ref prefix> [steps] [bonus] [warm stats .json] [RAM tier GiB] [LoRA dir]
// bonus 0 (default): exact routing, as the reference (dumpref.py runs with GLM_XPU_RES_BONUS=0)
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include "model-snapshot.h"

using namespace glm;
extern "C" void review_trace_reset();
extern "C" void review_trace_report(int);

int main(int argc, char** argv) {
    GLM_CHECK(argc >= 4, "usage: decode_check <model dir> <store .bin> <ref prefix> [steps] [bonus] [warm .json] [RAM GiB]");
    std::string ref = argv[3];
    Json j = Json::parse(read_file(ref + ".json"));
    int steps = j["steps"].i(), V = j["vocab"].i();
    if (argc > 4) steps = std::min(steps, atoi(argv[4]));
    TierConfig tc;
    tc.bonus_vram = tc.bonus_ram = argc > 5 ? (float)atof(argv[5]) : 0.0f;
    std::string warm = argc > 6 ? argv[6] : "";
    tc.ram_bytes = (size_t)((argc > 7 ? atof(argv[7]) : 14.0) * 1073741824.0);
    std::vector<int> tok;
    for (auto& t : j["tokens"].arr) tok.push_back(t.i());
    int plen = j.get_i("prompt_len", 1);
    std::vector<float> lpr((size_t)steps * V);
    std::ifstream f(ref + ".f32", std::ios::binary);
    f.read(reinterpret_cast<char*>(lpr.data()), lpr.size() * 4);
    GLM_CHECK(f, "short reference file");

    auto t0 = std::chrono::steady_clock::now();
    std::string lora = argc > 8 ? argv[8] : "";
    Model m(argv[1], argv[2], ((int)tok.size() + 64 + 255) / 256 * 256, tc, warm, 3.0, lora);
    printf("loaded in %.1f s\n", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    auto& q = Dev::get().q;
    int Vm = m.config().vocab;
    std::vector<sycl::half> lg(Vm);
    GLM_CHECK(steps <= (int)tok.size() - plen, "steps past the forced tokens");
    double kl_sum = 0, kl_max = 0, nll_r = 0, nll_e = 0, t_dec = 0;
    int n_nll = 0;
    int agree = 0, bad = 0;
    // the prompt: all but its last token as a prefill (GLM_PIECES=1: pieces of 8 through the decode path), the last
    // token through forward(); its row predicts the first forced token
    auto tp = std::chrono::steady_clock::now();
    const half* last = nullptr;
    if (getenv("GLM_PIECES")) {
        for (int i = 0; i < plen; i += 8) {
            int n = std::min(8, plen - i);
            last = m.forward(&tok[i], n) + (size_t)(n - 1) * Vm;
        }
    } else {
        m.prefill(tok.data(), plen - 1);
        last = m.forward(&tok[plen - 1], 1);
    }
    q.wait();
    printf("tier during the prompt: %zu staged from VRAM, %zu from RAM, %zu from NVMe (%zu read ahead), %.1f s waiting\n",
           m.tier().stats.staged_vram, m.tier().stats.staged_ram, m.tier().stats.staged_nvme,
           m.tier().stats.readahead_used, m.tier().stats.t_stage_wait);
    printf("prompt: %d tokens in %.1f s\n", plen, std::chrono::duration<double>(std::chrono::steady_clock::now() - tp).count());
    m.tier().reset_stats();     // decode-only statistics from here
    m.prof_reset();
    review_trace_reset();
    for (int s = 0; s < steps; s++) {
        auto t1 = std::chrono::steady_clock::now();
        const half* d = s == 0 ? last : m.forward(&tok[plen + s - 1], 1);
        q.wait();
        q.memcpy(lg.data(), d, Vm * 2).wait();
        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
        if (s > 0) t_dec += dt;
        // engine log-softmax over the reference's vocabulary
        double mx = -1e30;
        for (int v = 0; v < V; v++) mx = std::max(mx, (double)(float)lg[v]);
        double se = 0;
        for (int v = 0; v < V; v++) se += std::exp((double)(float)lg[v] - mx);
        double lse = mx + std::log(se), kl = 0;
        const float* r = &lpr[(size_t)s * V];
        int am_r = 0, am_e = 0;
        for (int v = 0; v < V; v++) {
            double le = (double)(float)lg[v] - lse;
            if (r[v] > -1e4) kl += std::exp((double)r[v]) * ((double)r[v] - le);
            if (r[v] > r[am_r]) am_r = v;
            if ((float)lg[v] > (float)lg[am_e]) am_e = v;
        }
        if (plen + s < (int)tok.size()) {
            nll_r -= r[tok[plen + s]];
            nll_e -= (double)(float)lg[tok[plen + s]] - lse;
            n_nll++;
        }
        kl_sum += kl;
        kl_max = std::max(kl_max, kl);
        agree += am_r == am_e;
        bad += kl > 0.5;
        printf("step %3d: KL %.4f  top-1 %s (ref %d, engine %d)  %.0f ms  NVMe reads %zu\n", s, kl,
               am_r == am_e ? "same" : "DIFF", am_r, am_e, dt * 1e3, m.tier().stats.nvme);
        fflush(stdout);
        if (!std::isfinite(kl)) break;
    }
    review_trace_report(steps - 1);
    printf("DECODE CHECK: %d steps, KL mean %.4f max %.4f, top-1 agreement %.3f, steps with KL > 0.5: %d\n", steps,
           kl_sum / steps, kl_max, (double)agree / steps, bad);
    printf("REVIEW_TIER raw calls=%zu picks=%zu zero_copy=%zu vram=%zu ram_hit=%zu admits=%zu demand=%zu pref_used=%zu\n", m.tier().stats.calls,m.tier().stats.picks,m.tier().stats.zero_copy,m.tier().stats.vram_hit,m.tier().stats.ram_hit,m.tier().stats.admit,m.tier().stats.nvme,m.tier().stats.prefetch_used);
    printf("tier %s\n", m.tier().summary().c_str());
    if (getenv("GLM_PROF")) printf("profile (decode):\n%s", m.prof_report(steps).c_str());
    printf("decode: %.1f ms per token (steps 1..%d)\n", t_dec * 1e3 / std::max(steps - 1, 1), steps - 1);
    printf("NLL of the next tokens: reference %.4f, engine %.4f (PPL %.3f vs %.3f)\n", nll_r / n_nll, nll_e / n_nll,
           std::exp(nll_r / n_nll), std::exp(nll_e / n_nll));
    return 0;
}
