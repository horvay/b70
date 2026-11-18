// Free-running generation speed, measured like llm/glm/xpu/tests/startbench.py --decode N --window W: the same prompt
// ("Write a long, detailed story about a lighthouse keeper ...", exllamav3's default_chat_prompt), temperature 1,
// top-p 0.95, no stop tokens, tok/s per window after the first token.
//   gen_bench <model dir> <store .bin> [tokens 400] [window 100] [bonus 0.1] [warm .json] [RAM GiB 14] [LoRA dir]
#include <chrono>
#include <cstdio>
#include "../src/engine.h"
#include "../src/model.h"
#include "../src/tokenizer.h"

using namespace glm;

int main(int argc, char** argv) {
    GLM_CHECK(argc >= 3, "usage: gen_bench <model dir> <store .bin> [tokens] [window] [bonus] [warm] [RAM GiB] [LoRA]");
    int N = argc > 3 ? atoi(argv[3]) : 400, Wn = argc > 4 ? atoi(argv[4]) : 100;
    TierConfig tc;
    tc.bonus_vram = tc.bonus_ram = argc > 5 ? (float)atof(argv[5]) : 0.1f;
    std::string warm = argc > 6 ? argv[6] : "";
    tc.ram_bytes = (size_t)((argc > 7 ? atof(argv[7]) : 14.0) * 1073741824.0);
    std::string lora = argc > 8 ? argv[8] : "";
    Tokenizer tok = Tokenizer::load(argv[1]);
    Model m(argv[1], argv[2], 8192, tc, warm, 3.0, lora);
    Engine eng(m);
    std::string q = "Write a long, detailed story about a lighthouse keeper who finds a message in a bottle.";
    GenRequest r;
    r.prompt = tok.encode("[gMASK]<sop><|user|>\n" + q + "<|assistant|>\n");
    r.sampling.temperature = 1.0f;
    r.sampling.top_p = 0.95f;
    r.sampling.has_seed = true;
    r.sampling.seed = 1234;
    r.max_new = N;
    using clk = std::chrono::steady_clock;
    clk::time_point t0;
    int n = 0, n0 = 0;
    Tier::Stats s0;
    std::string text;
    StreamDecoder dec(tok);
    r.on_token = [&](int id) {
        if (n == 0) { t0 = clk::now(); s0 = m.tier().stats; }
        text += dec.push(id);
        n++;
        if (n - 1 - n0 >= Wn || n == N) {
            double dt = std::chrono::duration<double>(clk::now() - t0).count();
            int k = n - 1 - n0;
            auto& s = m.tier().stats;
            printf("tokens %4d-%4d: %.2f tok/s (%.1f ms/token); per token nvme %.1f, prefetch %.1f (used %.1f), ram zero-copy %.1f, "
                   "admit %.1f, NVMe wait %.1f ms, tier host %.1f ms\n", n0, n - 1, k / dt, dt * 1e3 / k, (s.nvme - s0.nvme) / (double)k,
                   (s.prefetch - s0.prefetch) / (double)k, (s.prefetch_used - s0.prefetch_used) / (double)k,
                   (s.zero_copy - s0.zero_copy) / (double)k, (s.admit - s0.admit) / (double)k, (s.t_nvme - s0.t_nvme) * 1e3 / k,
                   (s.t_host - s0.t_host) * 1e3 / k);
            fflush(stdout);
            n0 = n - 1;
            t0 = clk::now();
            s0 = s;
        }
        return true;
    };
    GenResult g = eng.run(r);
    printf("GEN BENCH: %d tokens, prompt %d tokens in %.2f s, %.2f tok/s overall\n", g.completion_tokens, g.prompt_tokens,
           g.prompt_s, g.decode_tok_s);
    printf("text: %.300s\n", text.c_str());
}
