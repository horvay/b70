// What one kernel launch costs on the B70 in an in-order queue, for decode's ~2,900 small launches per token:
// host submission time and GPU time per kernel, for queue.parallel_for (an event per launch) and the
// sycl_ext_oneapi_enqueue_functions launches (no event), with empty and small (one work-group) kernels.
//   launch_bench [n]
#include <chrono>
#include <cstdio>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/enqueue_functions.hpp>

namespace syclex = sycl::ext::oneapi::experimental;
using clk = std::chrono::steady_clock;

static double ms(clk::time_point a, clk::time_point b) { return std::chrono::duration<double>(b - a).count() * 1e3; }

int main(int argc, char** argv) {
    int n = argc > 1 ? atoi(argv[1]) : 3000;
    sycl::queue q{sycl::gpu_selector_v, {sycl::property::queue::in_order{}}};
    sycl::queue qd{q.get_context(), q.get_device(),
                   {sycl::property::queue::in_order{}, sycl::ext::oneapi::property::queue::discard_events{}}};
    float* buf = sycl::malloc_device<float>(1 << 20, q);
    q.memset(buf, 0, 4 << 20).wait();
    auto run = [&](const char* name, sycl::queue& Q, auto launch) {
        for (int r = 0; r < 2; r++) {
            Q.wait();
            auto t0 = clk::now();
            for (int i = 0; i < n; i++) launch(Q, i);
            auto t1 = clk::now();
            Q.wait();
            auto t2 = clk::now();
            if (r) printf("%-44s host %6.2f us/launch, total %6.2f us/kernel\n", name, ms(t0, t1) * 1e3 / n,
                          ms(t0, t2) * 1e3 / n);
        }
    };
    // empty single work-item
    run("parallel_for, empty", q, [&](sycl::queue& Q, int) { Q.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {}); });
    run("nd_launch (no event), empty", q, [&](sycl::queue& Q, int) {
        syclex::nd_launch(Q, sycl::nd_range<1>(16, 16), [=](sycl::nd_item<1>) {});
    });
    run("discard_events queue, empty", qd, [&](sycl::queue& Q, int) {
        Q.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) {});
    });
    // a dependent chain: each kernel reads what the previous wrote (decode's shape), 256 work-groups of 128
    auto small = [&](sycl::queue& Q, int i, bool ev) {
        auto k = [=](sycl::nd_item<1> it) {
            size_t g = it.get_global_id(0);
            buf[g] = buf[(g + 1) & 32767] * 0.5f + (float)i;
        };
        if (ev) Q.parallel_for(sycl::nd_range<1>(32768, 128), k);
        else syclex::nd_launch(Q, sycl::nd_range<1>(32768, 128), k);
    };
    run("parallel_for, 256 WG x 128 dependent", q, [&](sycl::queue& Q, int i) { small(Q, i, true); });
    run("nd_launch, 256 WG x 128 dependent", q, [&](sycl::queue& Q, int i) { small(Q, i, false); });
    run("discard_events, 256 WG x 128 dependent", qd, [&](sycl::queue& Q, int i) { small(Q, i, true); });
    // one work-group (a norm / router tail)
    run("parallel_for, 1 WG x 128", q, [&](sycl::queue& Q, int i) {
        Q.parallel_for(sycl::nd_range<1>(128, 128), [=](sycl::nd_item<1> it) { buf[it.get_global_id(0)] += 1.f; });
    });
    // the same work as one graph replay
    {
        namespace sx = sycl::ext::oneapi::experimental;
        sx::command_graph g{q.get_context(), q.get_device()};
        g.begin_recording(q);
        for (int i = 0; i < 200; i++) small(q, i, true);
        g.end_recording(q);
        auto ge = g.finalize();
        q.ext_oneapi_graph(ge).wait();
        q.wait();
        auto t0 = clk::now();
        for (int r = 0; r < n / 200; r++) q.ext_oneapi_graph(ge);
        auto t1 = clk::now();
        q.wait();
        auto t2 = clk::now();
        int k = n / 200 * 200;
        printf("%-44s host %6.2f us/launch, total %6.2f us/kernel\n", "graph of 200, 256 WG x 128 dependent",
               ms(t0, t1) * 1e3 / k, ms(t0, t2) * 1e3 / k);
    }
    printf("LAUNCH BENCH DONE\n");
}
