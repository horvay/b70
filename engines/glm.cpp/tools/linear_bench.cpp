// Trunk EXL3 linears at decode shape (M = 1): GPU time per call and the trellis bytes it streams, for a few real
// projections of the model. GLM_EXL3_TARGET overrides the vector GEMV's split-K thread target (default 1024).
//   linear_bench <model dir> [iters 2000]
#include <chrono>
#include <cstdio>
#include "../src/exl3.h"

using namespace glm;

int main(int argc, char** argv) {
    GLM_CHECK(argc >= 2, "usage: linear_bench <model dir> [iters]");
    SafeTensors st(argv[1]);
    int iters = argc > 2 ? atoi(argv[2]) : 2000;
    auto& q = Dev::get().q;
    const char* pre = "model.language_model.layers.";
    for (const char* name : {"0.self_attn.qkv_proj", "0.self_attn.o_proj", "0.mlp.gate_proj", "0.mlp.down_proj",
                             "3.self_attn.q_b_proj", "3.self_attn.q_a_proj", "3.self_attn.o_proj",
                             "4.mlp.shared_experts.gate_proj", "4.mlp.shared_experts.down_proj"}) {
        std::string p = std::string(pre) + name;
        if (!Exl3Linear::present(st, p)) { printf("%-34s (absent)\n", name); continue; }
        Exl3Linear L = Exl3Linear::load(st, p);
        Workspace ws;
        ws.init(L.ws_bytes(1) + (1 << 20));
        half* x = sycl::malloc_device<half>(L.in, q);
        half* y = sycl::malloc_device<half>(L.out, q);
        q.memset(x, 0, L.in * 2).wait();
        for (int i = 0; i < 50; i++) L.forward(q, ws, x, 1, y, DType::F16);
        q.wait();
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; i++) L.forward(q, ws, x, 1, y, DType::F16);
        q.wait();
        double us = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e6 / iters;
        double mb = L.trellis.bytes() / 1e6;
        printf("%-34s %5d -> %5d K=%d  %6.1f us  %5.2f MB  %6.1f GB/s\n", name, L.in, L.out, L.K, us, mb, mb / us * 1e3);
        sycl::free(x, q);
        sycl::free(y, q);
    }
    printf("LINEAR BENCH DONE\n");
}
