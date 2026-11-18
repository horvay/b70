// NVMe read throughput of the expert store the way the tier reads it: O_DIRECT preads of random expert records,
// split into pieces, from N threads. Run with nothing else reading the drive.
//   read_bench <store .bin> [record MiB 7.5] [seconds per case 4] [threads pieces: one case] [destination MiB per thread]
#include <fcntl.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <thread>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: read_bench <store> [record MiB] [s]\n"); return 1; }
    int fd = open(argv[1], O_RDONLY | O_DIRECT);
    if (fd < 0) { perror("open"); return 1; }
    off_t size = lseek(fd, 0, SEEK_END);
    size_t rec = (size_t)((argc > 2 ? atof(argv[2]) : 7.5) * 1048576) & ~(size_t)4095;
    double secs = argc > 3 ? atof(argv[3]) : 4;
    int64_t nrec = size / rec - 1;
    std::vector<int> tl{4, 8, 16, 32}, pl{4, 8, 16};
    if (argc > 5) { tl = {atoi(argv[4])}; pl = {atoi(argv[5])}; }
    size_t dest = argc > 6 ? (size_t)atof(argv[6]) * 1048576 : 0;    // spread the landing pages like a RAM tier
    for (int threads : tl) {
        for (int pieces : pl) {
            size_t piece = (rec / pieces + 4095) & ~(size_t)4095;
            std::atomic<int64_t> bytes{0};
            std::atomic<bool> stop{false};
            std::atomic<int64_t> next_rec{0};
            std::vector<std::thread> th;
            // a shared queue of (record, piece) work like the tier's reader threads
            std::vector<int64_t> order(1 << 16);
            std::mt19937_64 rng(threads * 100 + pieces);
            for (auto& o : order) o = rng() % nrec;
            for (int t = 0; t < threads; t++)
                th.emplace_back([&, t] {
                    size_t region = std::max(dest, piece) / piece * piece;
                    void* buf;
                    if (posix_memalign(&buf, 4096, region)) return;
                    size_t nb = region / piece, bi = 0;
                    while (!stop) {
                        int64_t w = next_rec++;
                        int64_t r = order[(w / pieces) % order.size()], p = w % pieces;
                        size_t n = std::min(piece, rec - p * piece);
                        n = (n + 4095) & ~(size_t)4095;
                        char* dst = (char*)buf + (bi++ % nb) * piece;
                        ssize_t got = pread(fd, dst, n, (off_t)(r * rec + p * piece) & ~(off_t)4095);
                        if (got > 0) bytes += got;
                    }
                    free(buf);
                });
            auto t0 = std::chrono::steady_clock::now();
            std::this_thread::sleep_for(std::chrono::duration<double>(secs));
            stop = true;
            for (auto& x : th) x.join();
            double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            printf("threads %2d, %2d pieces of %4zu KiB: %5.2f GB/s, %.2f ms per record\n", threads, pieces, piece >> 10,
                   bytes / dt / 1e9, rec / (bytes / dt) * 1e3);
            fflush(stdout);
        }
    }
}
