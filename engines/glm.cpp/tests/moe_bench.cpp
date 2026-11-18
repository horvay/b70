// The routed-expert kernel alone at decode shape (M = 1, top-8): 8 real experts from the store, in VRAM slots, in
// pinned host memory read zero-copy, or mixed (the C++ twin of llm/glm/xpu/tests/moebench.py).
//   moe_bench <store .bin> [tier layer, default 1 = model layer 4] [iters]
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <random>
#include "../src/json.h"
#include "../src/moe.h"
#include "../src/st.h"

using namespace glm;

int main(int argc, char** argv) {
    GLM_CHECK(argc >= 2, "usage: moe_bench <store .bin> [li] [iters]");
    std::string bin = argv[1];
    int li = argc > 2 ? atoi(argv[2]) : 1, iters = argc > 3 ? atoi(argv[3]) : 200;
    Json j = Json::parse(read_file(bin.substr(0, bin.rfind('.')) + ".json"));
    auto& l = j["layers"][li];
    size_t off = l["offset"].i(), rec = l["record_bytes"].i();
    int kc = kcode(l["K"].f()), E = j["experts_per_layer"].i();
    int experts[8] = {3, 17, 40, 77, 101, 150, 201, 260};
    auto& q = Dev::get().q;
    char* host = static_cast<char*>(sycl::malloc_host(8 * rec, q));
    char* dev = static_cast<char*>(sycl::malloc_device(8 * rec, q));
    int fd = open(bin.c_str(), O_RDONLY);
    for (int i = 0; i < 8; i++) GLM_CHECK(pread(fd, host + i * rec, rec, off + experts[i] * rec) == (ssize_t)rec, "read");
    q.memcpy(dev, host, 8 * rec).wait();
    std::vector<half> xh(4096);
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0, 2);
    for (auto& v : xh) v = half(nd(rng));
    half* x = sycl::malloc_device<half>(4096, q);
    half* out = sycl::malloc_device<half>(4096, q);
    int* sel = sycl::malloc_device<int>(8, q);
    float* w = sycl::malloc_device<float>(8, q);
    int64_t* ptrs = sycl::malloc_device<int64_t>(E, q);
    std::vector<float> wh(8, 0.125f);
    q.memcpy(x, xh.data(), 8192);
    q.memcpy(sel, experts, 32);
    q.memcpy(w, wh.data(), 32).wait();
    Workspace ws;
    ws.init(64 << 20);
    printf("layer li %d, K code %d, record %.2f MiB, M=1 top-8\n", li, kc, rec / 1048576.0);
    const char* cases[] = {"vvvvvvvv", "vvvvvvrr", "vvvvrrrr", "rrrrrrrr"};
    for (const char* c : cases) {
        std::vector<int64_t> p(E, 0);
        for (int i = 0; i < 8; i++) p[experts[i]] = (int64_t)((c[i] == 'v' ? dev : host) + i * rec);
        q.memcpy(ptrs, p.data(), E * 8).wait();
        for (int i = 0; i < 5; i++) moe_forward(q, ws, x, 1, sel, w, 8, ptrs, 4096, 2048, kc, E, out);
        q.wait();
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; i++) moe_forward(q, ws, x, 1, sel, w, 8, ptrs, 4096, 2048, kc, E, out);
        q.wait();
        double ms = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e3 / iters;
        int nr = 0;
        for (int i = 0; i < 8; i++) nr += c[i] == 'r';
        printf("  %d VRAM + %d RAM: %.3f ms per call\n", 8 - nr, nr, ms);
    }
    // the same experts copied into a 14 GB pinned host tier made of < 2 GiB chunks (as src/tier.cpp allocates it),
    // each at a far-apart slot: does spreading RAM reads over the whole tier cost more (GPU address translation)?
    {
        std::vector<char*> chunks;
        size_t per = ((size_t)2 << 30) - 1, slots = per / rec;
        for (int c = 0; c < 7; c++) {
            char* ch = static_cast<char*>(sycl::malloc_host(slots * rec, q));
            GLM_CHECK(ch, "chunk");
            memset(ch, 0, slots * rec);
            chunks.push_back(ch);
        }
        std::mt19937 r2(7);
        for (int spread = 0; spread < 2; spread++) {
            std::vector<int64_t> p(E, 0);
            for (int i = 0; i < 8; i++) {
                char* dst = spread ? chunks[r2() % chunks.size()] + (r2() % slots) * rec : chunks[0] + i * rec;
                memcpy(dst, host + i * rec, rec);
                p[experts[i]] = (int64_t)dst;
            }
            q.memcpy(ptrs, p.data(), E * 8).wait();
            for (int i = 0; i < 5; i++) moe_forward(q, ws, x, 1, sel, w, 8, ptrs, 4096, 2048, kc, E, out);
            q.wait();
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; i++) moe_forward(q, ws, x, 1, sel, w, 8, ptrs, 4096, 2048, kc, E, out);
            q.wait();
            double ms = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e3 / iters;
            printf("  0 VRAM + 8 RAM in a 14 GB tier, %s: %.3f ms per call\n", spread ? "spread over it" : "adjacent", ms);
        }
        // many different experts in turn (as decode: each call reads other experts), spread over the tier
        std::vector<std::vector<int64_t>> tabs(64, std::vector<int64_t>(E, 0));
        for (auto& t : tabs)
            for (int i = 0; i < 8; i++) {
                char* dst = chunks[r2() % chunks.size()] + (r2() % slots) * rec;
                memcpy(dst, host + i * rec, rec);
                t[experts[i]] = (int64_t)dst;
            }
        int64_t* ptab = sycl::malloc_device<int64_t>((size_t)64 * E, q);
        for (int t = 0; t < 64; t++) q.memcpy(ptab + (size_t)t * E, tabs[t].data(), E * 8);
        q.wait();
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 64; i++) moe_forward(q, ws, x, 1, sel, w, 8, ptab + (size_t)i * E, 4096, 2048, kc, E, out);
        q.wait();
        double ms = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e3 / 64;
        printf("  0 VRAM + 8 RAM, a different spread set every call (first touch each): %.3f ms per call\n", ms);
        t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 64; i++) moe_forward(q, ws, x, 1, sel, w, 8, ptab + (size_t)i * E, 4096, 2048, kc, E, out);
        q.wait();
        ms = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e3 / 64;
        printf("  the same 64 sets again: %.3f ms per call\n", ms);
        // DMA instead: copy each set's 8 RAM experts into VRAM rows (main queue, then a separate copy queue), cold sets
        std::vector<std::vector<int64_t>> tabs2(64, std::vector<int64_t>(E, 0));
        std::vector<char*> src2(64 * 8);
        for (int t = 0; t < 64; t++)
            for (int i = 0; i < 8; i++) {
                char* dst = chunks[r2() % chunks.size()] + (r2() % slots) * rec;
                memcpy(dst, host + i * rec, rec);
                src2[t * 8 + i] = dst;
            }
        sycl::queue cq{q.get_context(), q.get_device(), {sycl::property::queue::in_order{}}};
        for (int pass = 0; pass < 2; pass++) {
            sycl::queue& qq = pass ? cq : q;
            auto t1 = std::chrono::steady_clock::now();
            for (int t = 0; t < 64; t++)
                for (int i = 0; i < 8; i++) qq.memcpy(dev + i * rec, src2[t * 8 + i], rec);
            qq.wait();
            double ms2 = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count() * 1e3 / 64;
            printf("  DMA of 8 cold spread RAM experts to VRAM (%s queue): %.3f ms per 8 (%.3f per expert)\n",
                   pass ? "a second" : "the main", ms2, ms2 / 8);
        }
        // overlap: RAM -> VRAM staging copies on a second queue while the main queue runs expert kernels. Does the copy
        // run beside the compute (a copy engine), and what does each cost the other? (VRAM-only kernels: no PCIe use;
        // half-RAM kernels: both use the PCIe read direction)
        {
            char* stage = static_cast<char*>(sycl::malloc_device(8 * rec, q));
            for (const char* c : {"vvvvvvvv", "vvvvrrrr"}) {
                std::vector<int64_t> p(E, 0);
                for (int i = 0; i < 8; i++) p[experts[i]] = (int64_t)((c[i] == 'v' ? dev : host) + i * rec);
                q.memcpy(ptrs, p.data(), E * 8).wait();
                int nk = c[4] == 'v' ? 600 : 150;
                auto kernels = [&] { for (int i = 0; i < nk; i++) moe_forward(q, ws, x, 1, sel, w, 8, ptrs, 4096, 2048, kc, E, out); };
                auto copies = [&] { for (int t = 0; t < 32; t++) for (int i = 0; i < 8; i++) cq.memcpy(stage + i * rec, src2[t * 8 + i], rec); };
                kernels(); q.wait();
                auto t0 = std::chrono::steady_clock::now();
                kernels(); q.wait();
                double tk = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e3;
                t0 = std::chrono::steady_clock::now();
                copies(); cq.wait();
                double tc = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e3;
                t0 = std::chrono::steady_clock::now();
                copies();
                kernels();
                q.wait();
                double tk2 = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e3;
                cq.wait();
                double tb = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() * 1e3;
                printf("  overlap (%s kernels): kernels alone %.1f ms, 256 copies alone %.1f ms; together: kernels done at "
                       "%.1f ms, both at %.1f ms (serial would be %.1f)\n", c, tk, tc, tk2, tb, tk + tc);
            }
        }

    }
    printf("MOE BENCH DONE\n");
}
