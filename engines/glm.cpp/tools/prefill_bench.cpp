// Prompt (prefill) speed and the prompt path's correctness at the client's lengths: a text file's tokens, repeated up to N
// tokens, through Model::prefill in one call (as Engine::run does), then the tier's prompt counters.
//   prefill_bench <model dir> <store .bin> <text file> [tokens 8192] [RAM GiB 14] [LoRA dir] [warm stats .json]
// With the routing profile (llm/glm/xpu/data/route_stats_glm53.json) the tiers start filled as glm_server's do;
// without it they start empty and every non-resident expert comes from NVMe (the worst case).
// GLM_PROF=5 sums the named GPU points per step; GLM_PREFILL_TRACE=1|2 prints each layer's (or step's) finish time, GLM_PREFILL_CHUNK the rows per piece.
#include <chrono>
#include <cstdio>
#include "../src/model.h"
#include "../src/tokenizer.h"

using namespace glm;

int main(int argc, char** argv) {
    GLM_CHECK(argc >= 4, "usage: prefill_bench <model dir> <store .bin> <text file> [tokens] [RAM GiB] [LoRA] [warm stats]");
    int N = argc > 4 ? atoi(argv[4]) : 8192;
    TierConfig tc;
    // GLM_BENCH_EXACT=1: no residency bonus, so the logits token's routing does not depend on what is resident
    tc.bonus_vram = tc.bonus_ram = getenv("GLM_BENCH_EXACT") && atoi(getenv("GLM_BENCH_EXACT")) ? 0.0f : 0.1f;
    tc.ram_bytes = (size_t)((argc > 5 ? atof(argv[5]) : 14.0) * 1073741824.0);
    std::string lora = argc > 6 ? argv[6] : "", warm = argc > 7 ? argv[7] : "";
    Tokenizer tok = Tokenizer::load(argv[1]);
    FILE* f = fopen(argv[3], "rb");
    GLM_CHECK(f, "%s", argv[3]);
    std::string text;
    char buf[65536];
    for (size_t r; (r = fread(buf, 1, sizeof buf, f)) > 0;) text.append(buf, r);
    fclose(f);
    std::vector<int> base = tok.encode(text), ids;
    GLM_CHECK(!base.empty(), "empty prompt");
    while ((int)ids.size() < N) ids.insert(ids.end(), base.begin(), base.begin() + std::min<size_t>(base.size(), N - ids.size()));
    Model m(argv[1], argv[2], (N + 64 + 255) / 256 * 256, tc, warm, 3.0, lora);
    auto& q = Dev::get().q;
    q.wait();
    auto t0 = std::chrono::steady_clock::now();
    // GLM_BENCH_LOGITS=<file>: the prompt's last token goes through forward() (as Engine::run) and its fp32 logits
    // are written there, for comparing two builds or paths on the same prompt
    const char* lf = getenv("GLM_BENCH_LOGITS");
    m.prefill(ids.data(), lf ? N - 1 : N, -1);
    q.wait();
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (lf) {
        int V = m.config().vocab;
        std::vector<half> lg(V);
        download(lg.data(), m.forward(&ids[N - 1], 1), V * sizeof(half));
        std::vector<float> l32(lg.begin(), lg.end());
        FILE* o = fopen(lf, "wb");
        GLM_CHECK(o, "%s", lf);
        fwrite(l32.data(), sizeof(float), V, o);
        fclose(o);
    }
    auto& st = m.tier().stats;
    printf("PREFILL BENCH: %d tokens in %.2f s (%.0f tok/s); staged vram %zu, ram %zu, nvme %zu (read-ahead used %zu), "
           "stage wait %.2f s\n", N, s, N / s, st.staged_vram, st.staged_ram, st.staged_nvme, st.readahead_used,
           st.t_stage_wait);
    printf("%s\n", m.tier().summary().c_str());
    if (g_prof_points) printf("%s", prof_points_report(N).c_str());   // GLM_PROF=5: the prompt's named GPU points
}
