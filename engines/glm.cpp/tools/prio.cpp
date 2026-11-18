// Does the B70 honor SYCL queue priority? A "decode" loop on a high-priority in-order queue (per token: 42 GPU bursts,
// each waited on by the host, then a short host gap; a 3.5 ms stall every 14 bursts, like NVMe demand reads) against a
// low-priority queue that runs short compute kernels nonstop. Reports the decode loop's time per token alone and beside
// the background queue, and how much work the background queue got done alone and beside decode.
//   prio [tokens] [burst_us] [bg_us]
#include <sycl/sycl.hpp>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

using clk = std::chrono::steady_clock;
namespace prop = sycl::ext::oneapi::property::queue;

static double now() { return std::chrono::duration<double>(clk::now().time_since_epoch()).count(); }

static void spin(double s) {
    double t = now() + s;
    while (now() < t) {}
}

// iters FMA rounds over n floats: a compute-bound kernel whose length scales with iters
static sycl::event burn(sycl::queue& q, float* p, size_t n, int iters) {
    return q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        float a = p[i], b = 1.0001f;
        for (int k = 0; k < iters; k++) a = a * b + 0.5f;
        p[i] = a;
    });
}

// iterations for a kernel of about `us` microseconds
static int calibrate(sycl::queue& q, float* p, size_t n, double us) {
    int it = 64;
    for (;;) {
        burn(q, p, n, it).wait();
        double t0 = now();
        for (int r = 0; r < 10; r++) burn(q, p, n, it);
        q.wait();
        double per = (now() - t0) / 10 * 1e6;
        if (per >= us * 0.8 || it > (1 << 24)) return (int)(it * us / per);
        it *= 2;
    }
}

int main(int argc, char** argv) {
    int tokens = argc > 1 ? atoi(argv[1]) : 100;
    double burst_us = argc > 2 ? atof(argv[2]) : 1500;
    double bg_us = argc > 3 ? atof(argv[3]) : 300;
    sycl::device dev{sycl::gpu_selector_v};
    sycl::context ctx{dev};
    sycl::queue hi{ctx, dev, {sycl::property::queue::in_order{}, prop::priority_high{}}};
    sycl::queue lo{ctx, dev, {sycl::property::queue::in_order{}, prop::priority_low{}}};
    printf("device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
    size_t n = 1 << 20;
    float* a = sycl::malloc_device<float>(n, hi);
    float* b = sycl::malloc_device<float>(n, lo);
    hi.fill(a, 1.0f, n).wait();
    lo.fill(b, 1.0f, n).wait();
    int it_burst = calibrate(hi, a, n, burst_us), it_bg = calibrate(lo, b, n, bg_us);

    auto decode = [&](int ntok) {
        double t0 = now();
        for (int t = 0; t < ntok; t++)
            for (int l = 0; l < 42; l++) {
                burn(hi, a, n, it_burst).wait();          // the host waits on the layer's picks
                spin(l % 14 == 13 ? 3.5e-3 : 0.13e-3);    // host bookkeeping, or an NVMe demand read
            }
        return (now() - t0) / ntok * 1e3;
    };
    std::atomic<bool> run{false};
    std::atomic<long> done{0};
    auto bg = [&] {
        while (run) {
            for (int k = 0; k < 8; k++) burn(lo, b, n, it_bg);
            lo.wait();
            done += 8;
        }
    };
    auto bg_rate = [&](double sec) {
        done = 0;
        run = true;
        std::thread th(bg);
        double t0 = now();
        std::this_thread::sleep_for(std::chrono::duration<double>(sec));
        run = false;
        th.join();
        return done / (now() - t0);
    };

    decode(5);
    double alone = bg_rate(5);
    double d0 = decode(tokens);
    done = 0;
    run = true;
    std::thread th(bg);
    double t0 = now();
    double d1 = decode(tokens);
    double got = done / (now() - t0);
    run = false;
    th.join();
    double d2 = decode(tokens);
    double gpu_ms = 42 * burst_us * 1e-3;
    printf("decode burst %.0f us x 42 (%.1f ms GPU per token), bg kernel %.0f us\n", burst_us, gpu_ms, bg_us);
    printf("decode alone %.1f / %.1f ms per token; beside the low-priority queue %.1f ms (%+.0f%%)\n", d0, d2, d1,
           100 * (d1 / ((d0 + d2) / 2) - 1));
    printf("background alone %.0f kernels/s; beside decode %.0f/s = %.0f%% (the GPU is idle %.0f%% of a decode token)\n",
           alone, got, 100 * got / alone, 100 * (1 - gpu_ms / ((d0 + d2) / 2)));
    sycl::free(a, hi);
    sycl::free(b, lo);
}
