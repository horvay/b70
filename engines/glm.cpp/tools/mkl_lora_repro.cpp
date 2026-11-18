// Does oneMKL's fp16 GEMM hang on the prompt's small-N shapes (KDA's fused 320 x 4096, LoRA rank 64) when M=612 calls come before M=2048 ones, as
// a short request before a long one did (2026-11-02)? Each call waits, with a timestamp; a hang stops the output.
//   mkl_lora_repro [first M 612] [second M 2048] [rounds 3]
#include <chrono>
#include <cstdio>
#include <oneapi/mkl.hpp>
#include <sycl/sycl.hpp>

int main(int argc, char** argv) {
    int m1 = argc > 1 ? atoi(argv[1]) : 612, m2 = argc > 2 ? atoi(argv[2]) : 2048, rounds = argc > 3 ? atoi(argv[3]) : 3;
    sycl::queue q{sycl::gpu_selector_v, {sycl::property::queue::in_order{}}};
    // (N, K): LoRA deltas (rank 64), KDA's fused g_a|b|f_a (320 x 4096), g_b / f_b (8192 x 128)
    const std::pair<int, int> shapes[] = {{320, 4096}, {8192, 128}, {128, 64}, {64, 64}, {8192, 64}, {12288, 64}};
    size_t big = (size_t)8192 * 12288;
    sycl::half* t = sycl::malloc_device<sycl::half>(big, q);
    sycl::half* b = sycl::malloc_device<sycl::half>(big, q);
    sycl::half* d = sycl::malloc_device<sycl::half>(big, q);
    q.memset(t, 0, big * 2).wait();
    q.memset(b, 0, big * 2).wait();
    auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < rounds; r++)
        for (int M : {m1, m2})
            for (auto [N, K] : shapes)
                for (float beta : {0.0f, 1.0f}) {
                    oneapi::mkl::blas::row_major::gemm(q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::trans,
                                                       M, N, K, sycl::half(1.0f), t, K, b, K, sycl::half(beta), d, N);
                    q.wait();
                    printf("round %d M %4d N %5d K %4d beta %.0f ok at %.3f s\n", r, M, N, K, beta,
                           std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
                    fflush(stdout);
                }
    printf("MKL REPRO DONE\n");
}
