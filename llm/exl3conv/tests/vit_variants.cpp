// exl3conv: encoder kernel shapes, side by side. Build with ./build.sh vit_variants; the compiler's spill report names
// each instance as viterbi_launch<K, CB, WG, CACHE, BIG, UNROLL>. Run on the B70: vit_variants [tiles]
// For every K it times each shape on the same Gaussian tiles (mul1) and checks they all give identical states.
#include "exl3/viterbi.hpp"
#include "rt/rt.hpp"

#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace exl3;

using LaunchFn = sycl::event (*)(sycl::queue&, const float*, uint16_t*, float*, uint8_t*, half*, int64_t);
struct Variant { int K; const char* name; LaunchFn fn; };

#define V(KK, WG, C, B, U, O) {KK, #WG "/c" #C "/" #B "/u" #U "/o" #O, &viterbi_launch<KK, 2, WG, C, B, U, O>}
#define ALL(KK)                                                                                               \
    V(KK, 1024, 64, false, 8, 3), V(KK, 1024, 64, false, 8, 15), V(KK, 1024, 64, false, 8, 6),              \
    V(KK, 1024, 64, false, 8, 19), V(KK, 1024, 64, false, 8, 18), V(KK, 1024, 32, false, 8, 19),            \
    V(KK, 512, 128, true, 8, 19), V(KK, 512, 256, true, 8, 2), V(KK, 512, 256, true, 8, 3)

static const Variant variants[] = {ALL(6), ALL(7), ALL(8)};

int main(int argc, char** argv)
{
    const int64_t n = argc > 1 ? std::atoll(argv[1]) : 4096;
    const int only_k = argc > 2 ? std::atoi(argv[2]) : 0;
    auto q = rt::make_queue();
    rt::print_device(q.get_device());
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0, 1);
    std::vector<float> w((size_t) n * 256);
    for (auto& v : w) v = nd(rng);
    for (int K = 1; K <= 8; ++K)
    {
        if (only_k && K != only_k) continue;
        const int64_t batch = std::min<int64_t>(n, std::max<int64_t>(32, (int64_t) (512ull << 20) / (int64_t) vit_history_bytes(K)));
        float* d_in = sycl::malloc_device<float>(n * 256, q);
        float* d_out = sycl::malloc_device<float>(batch * 256, q);
        uint16_t* d_idx = sycl::malloc_device<uint16_t>(n * 256, q);
        uint8_t* d_hist = sycl::malloc_device<uint8_t>(batch * vit_history_bytes(K), q);
        half* d_gc = viterbi_needs_gcost(K) ? sycl::malloc_device<half>(batch * viterbi_gcost_halves(K), q) : nullptr;
        q.memcpy(d_in, w.data(), n * 1024).wait();
        std::vector<uint16_t> first, got((size_t) n * 256);
        std::printf("K %d (batch %lld)\n", K, (long long) batch);
        for (const auto& v : variants)
        {
            if (v.K != K) continue;
            v.fn(q, d_in, d_idx, d_out, d_hist, d_gc, std::min<int64_t>(batch, 32)).wait();     // warm-up
            rt::Timer t;
            for (int64_t s = 0; s < n; s += batch)
                v.fn(q, d_in + s * 256, d_idx + s * 256, d_out, d_hist, d_gc, std::min(batch, n - s));
            q.wait();
            const double secs = t.s();
            q.memcpy(got.data(), d_idx, n * 512).wait();
            bool same = true;
            if (first.empty()) first = got; else same = std::memcmp(first.data(), got.data(), n * 512) == 0;
            std::printf("  %-22s %8.0f tiles/s  %s\n", v.name, n / secs, same ? "" : "DIFFERENT STATES");
        }
        sycl::free(d_in, q); sycl::free(d_out, q); sycl::free(d_idx, q); sycl::free(d_hist, q); if (d_gc) sycl::free(d_gc, q);
    }
    return 0;
}
