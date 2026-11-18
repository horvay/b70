// Can the RAM tier be ordinary memory that NVMe reads land in directly (O_DIRECT, no bounce copy) and that the GPU
// still reads in place over PCIe? memfd pages -> /dev/udmabuf dma-buf (64 MB windows) -> Level Zero import as a
// device allocation backed by those system pages. Checks: O_DIRECT works, the expert kernel's output matches the
// same experts in VRAM, its speed is host-memory speed (not migrated into VRAM), and a CPU-side overwrite is seen.
//   udmabuf_test <store .bin> [tier layer 1]
#include <fcntl.h>
#include <linux/udmabuf.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <level_zero/ze_api.h>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include "../src/json.h"
#include "../src/moe.h"
#include "../src/st.h"

using namespace glm;

int main(int argc, char** argv) {
    GLM_CHECK(argc >= 2, "usage: udmabuf_test <store .bin> [li]");
    std::string bin = argv[1];
    int li = argc > 2 ? atoi(argv[2]) : 1;
    Json j = Json::parse(read_file(bin.substr(0, bin.rfind('.')) + ".json"));
    auto& l = j["layers"][li];
    size_t off = l["offset"].i(), rec = l["record_bytes"].i();
    int kc = kcode(l["K"].f()), E = j["experts_per_layer"].i();
    const size_t W = 64ull << 20;                  // udmabuf window (size_limit_mb)
    const int per = (int)(W / rec);                // records per window
    auto& q = Dev::get().q;

    // memfd, sealed against shrinking (udmabuf's requirement), mapped
    size_t total = 4 * W;
    int mfd = memfd_create("glm-ram-tier", MFD_ALLOW_SEALING);
    GLM_CHECK(mfd >= 0 && ftruncate(mfd, total) == 0 && fcntl(mfd, F_ADD_SEALS, F_SEAL_SHRINK) == 0, "memfd");
    char* host = static_cast<char*>(mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, mfd, 0));
    GLM_CHECK(host != MAP_FAILED, "mmap");

    // O_DIRECT straight into the memfd pages
    int sfd = open(bin.c_str(), O_RDONLY | O_DIRECT);
    int experts[8] = {3, 17, 40, 77, 101, 150, 201, 260};
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 8; i++) {
        ssize_t r = pread(sfd, host + i * rec, rec, off + experts[i] * rec);
        GLM_CHECK(r == (ssize_t)rec, "O_DIRECT pread into memfd pages: %zd (%s)", r, strerror(errno));
    }
    double gbs = 8.0 * rec / 1e9 / std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("O_DIRECT into memfd pages: ok, %.2f GB/s\n", gbs);

    // one dma-buf per 64 MB window, imported into Level Zero as device allocations over those pages
    int ufd = open("/dev/udmabuf", O_RDWR);
    GLM_CHECK(ufd >= 0, "open /dev/udmabuf");
    auto zctx = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_context());
    auto zdev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(q.get_device());
    std::vector<char*> win;
    for (size_t o = 0; o < total; o += W) {
        udmabuf_create c{(__u32)mfd, UDMABUF_FLAGS_CLOEXEC, o, W};
        int dfd = ioctl(ufd, UDMABUF_CREATE, &c);
        GLM_CHECK(dfd >= 0, "UDMABUF_CREATE: %s", strerror(errno));
        ze_external_memory_import_fd_t imp{ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD, nullptr,
                                           ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF, dfd};
        ze_device_mem_alloc_desc_t d{ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, &imp, 0, 0};
        void* p = nullptr;
        ze_result_t zr = zeMemAllocDevice(zctx, &d, W, 4096, zdev, &p);
        GLM_CHECK(zr == ZE_RESULT_SUCCESS && p, "zeMemAllocDevice(dma-buf import): 0x%x", (unsigned)zr);
        win.push_back(static_cast<char*>(p));
        close(dfd);
    }
    printf("imported %zu windows of 64 MB (%d records each)\n", win.size(), per);

    // the same experts in VRAM, as the reference
    char* vram = static_cast<char*>(sycl::malloc_device(8 * rec, q));
    q.memcpy(vram, host, 8 * rec).wait();
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
    // window-local address of slot i (slots 0..7 all in window 0 here)
    auto run = [&](bool imported, std::vector<half>& res) {
        std::vector<int64_t> p(E, 0);
        for (int i = 0; i < 8; i++)
            p[experts[i]] = (int64_t)(imported ? win[(i * rec) / W] + (i * rec) % W : vram + i * rec);
        q.memcpy(ptrs, p.data(), E * 8).wait();
        for (int i = 0; i < 5; i++) moe_forward(q, ws, x, 1, sel, w, 8, ptrs, 4096, 2048, kc, E, out);
        q.wait();
        auto t = std::chrono::steady_clock::now();
        for (int i = 0; i < 200; i++) moe_forward(q, ws, x, 1, sel, w, 8, ptrs, 4096, 2048, kc, E, out);
        q.wait();
        double ms = std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count() * 1e3 / 200;
        res.resize(4096);
        q.memcpy(res.data(), out, 8192).wait();
        return ms;
    };
    std::vector<half> rv, ri;
    double tv = run(false, rv), ti = run(true, ri);
    double num = 0, den = 0;
    for (int i = 0; i < 4096; i++) { double d = (float)ri[i] - (float)rv[i]; num += d * d; den += (float)rv[i] * (float)rv[i]; }
    printf("8 experts: VRAM %.3f ms, imported memfd pages %.3f ms per call; rel diff %.1e\n", tv, ti, std::sqrt(num / den));
    printf("  (pinned USM host memory read the same way takes ~2.5 ms: about that = read in place over PCIe)\n");

    // coherence: O_DIRECT other experts into the same pages, run again without re-importing
    int ex2[8] = {5, 19, 44, 80, 120, 170, 222, 280};
    for (int i = 0; i < 8; i++)
        GLM_CHECK(pread(sfd, host + i * rec, rec, off + ex2[i] * rec) == (ssize_t)rec, "pread 2");
    q.memcpy(vram, host, 8 * rec).wait();
    q.memcpy(sel, ex2, 32).wait();
    memcpy(experts, ex2, sizeof ex2);
    rv.clear(); ri.clear();
    run(false, rv);
    run(true, ri);
    num = den = 0;
    for (int i = 0; i < 4096; i++) { double d = (float)ri[i] - (float)rv[i]; num += d * d; den += (float)rv[i] * (float)rv[i]; }
    printf("after O_DIRECT overwrites of the same pages: rel diff %.1e (0 = the GPU sees the new bytes)\n",
           std::sqrt(num / den));
    printf("UDMABUF TEST DONE\n");
    return 0;
}
