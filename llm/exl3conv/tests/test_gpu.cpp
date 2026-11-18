// exl3conv GPU tests (the B70; run inside a queue turn or a `b70q pause` window).
//
//   test_gpu gen <file>     CPU only: random tiles + the CPU reference encoder's states, written to <file>
//   test_gpu probe          print the device limits
//   test_gpu viterbi <file> the SYCL encoder must reproduce every state of <file> exactly
//   test_gpu bench [K] [n]  encoder throughput on n Gaussian tiles (default K 4, 8192 tiles), and what it means for
//                           a 31B model (1.2e8 tiles)
//
// gen is split from viterbi so the slow CPU reference (about a second per tile) never runs while holding the card.
#include "exl3/codebook.hpp"
#include "exl3/pack.hpp"
#include "exl3/viterbi.hpp"
#include "exl3/viterbi_ref.hpp"
#include "rt/rt.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace exl3;

struct Case { int K, cb; std::vector<float> w; std::vector<uint16_t> idx; };

static const int CASE_CBS[] = {CB_MUL1, CB_3INST, CB_MCG};

static void gen(const std::string& path)
{
    std::vector<Case> cases;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0, 1);
    for (int cb : CASE_CBS)
        for (int K = 1; K <= 8; ++K)
        {
            if (cb != CB_MUL1 && K != 2 && K != 4 && K != 8) continue;
            const int n = cb == CB_MUL1 ? 6 : 2;
            for (int t = 0; t < n; ++t)
            {
                Case c{K, cb, std::vector<float>(256), std::vector<uint16_t>(256)};
                const float scale = cb == CB_MUL1 ? 1.0f : 1.24371088f;
                for (int i = 0; i < 256; ++i)
                {
                    float v = nd(rng) * scale;
                    if (t == 1) v *= 0.01f;                    // small values: fp16 subnormal squares
                    if (t == 2) v *= 30.0f;                    // large values: costs near fp16 overflow
                    if (t == 3 && i % 7 == 0) v = 0.0f;        // exact zeros (ties)
                    c.w[i] = v;
                }
                cases.push_back(std::move(c));
            }
        }
    const CodebookTable tables[3] = {CodebookTable(0), CodebookTable(1), CodebookTable(2)};
    std::vector<std::thread> th;
    const int nth = 8;
    for (int p = 0; p < nth; ++p)
        th.emplace_back([&, p] {
            for (size_t i = p; i < cases.size(); i += nth)
            {
                std::vector<float> out(256);
                ViterbiRef(cases[i].K, cases[i].cb, tables[cases[i].cb]).encode(cases[i].w.data(), cases[i].idx.data(), out.data());
            }
        });
    for (auto& t : th) t.join();
    std::ofstream f(path, std::ios::binary);
    const uint32_t n = (uint32_t) cases.size();
    f.write((const char*) &n, 4);
    for (auto& c : cases)
    {
        f.write((const char*) &c.K, 4); f.write((const char*) &c.cb, 4);
        f.write((const char*) c.w.data(), 1024); f.write((const char*) c.idx.data(), 512);
    }
    std::printf("wrote %u cases to %s\n", n, path.c_str());
}

static std::vector<Case> load(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + path + " (run `test_gpu gen` first)");
    uint32_t n; f.read((char*) &n, 4);
    std::vector<Case> cases(n);
    for (auto& c : cases)
    {
        c.w.resize(256); c.idx.resize(256);
        f.read((char*) &c.K, 4); f.read((char*) &c.cb, 4);
        f.read((char*) c.w.data(), 1024); f.read((char*) c.idx.data(), 512);
    }
    return cases;
}

struct Bufs
{
    sycl::queue& q;
    float* in = nullptr; float* out = nullptr; uint16_t* idx = nullptr; uint8_t* hist = nullptr; half* gcost = nullptr;
    Bufs(sycl::queue& q_, int64_t n, int K) : q(q_)
    {
        in = sycl::malloc_device<float>(n * 256, q);
        out = sycl::malloc_device<float>(n * 256, q);
        idx = sycl::malloc_device<uint16_t>(n * 256, q);
        hist = sycl::malloc_device<uint8_t>(n * vit_history_bytes(K), q);
        if (viterbi_needs_gcost(K)) gcost = sycl::malloc_device<half>(n * viterbi_gcost_halves(K), q);
        if (!in || !out || !idx || !hist || (viterbi_needs_gcost(K) && !gcost)) throw std::runtime_error("device allocation failed");
    }
    ~Bufs() { for (void* p : {(void*) in, (void*) out, (void*) idx, (void*) hist, (void*) gcost}) if (p) sycl::free(p, q); }
};

static int test_viterbi(const std::string& path)
{
    const auto cases = load(path);
    auto q = rt::make_queue();
    rt::print_device(q.get_device());
    int bad_cases = 0;
    for (auto& c : cases)
    {
        Bufs b(q, 1, c.K);
        q.memcpy(b.in, c.w.data(), 1024).wait();
        viterbi(q, c.K, c.cb, b.in, b.idx, b.out, b.hist, b.gcost, 1).wait();
        std::vector<uint16_t> idx(256); std::vector<float> out(256);
        q.memcpy(idx.data(), b.idx, 512).wait();
        q.memcpy(out.data(), b.out, 1024).wait();
        int diff = 0, first = -1;
        for (int i = 0; i < 256; ++i) if (idx[i] != c.idx[i]) { if (first < 0) first = i; ++diff; }
        const bool tb = is_tail_biting(idx.data(), Rate::integer(c.K));
        double mse_g = 0, mse_r = 0;
        const CodebookTable tab(c.cb);
        for (int i = 0; i < 256; ++i)
        {
            mse_g += (out[i] - c.w[i]) * (out[i] - c.w[i]);
            const double r = f16_to_double(tab[c.idx[i]]);
            mse_r += (r - c.w[i]) * (r - c.w[i]);
        }
        if (diff || !tb) ++bad_cases;
        std::printf("  K %d cb %d: %s  (%d states differ%s; mse gpu %.5g ref %.5g)%s\n", c.K, c.cb,
                    diff ? "DIFF" : "same", diff, first >= 0 ? (", first at " + std::to_string(first)).c_str() : "",
                    mse_g / 256, mse_r / 256, tb ? "" : "  NOT TAIL-BITING");
    }
    if (bad_cases) std::printf("\n%d of %zu cases differ\n", bad_cases, cases.size());
    else std::printf("\nall %zu cases bit-identical\n", cases.size());
    return bad_cases ? 1 : 0;
}

static int bench(int K, int64_t n)
{
    auto q = rt::make_queue();
    rt::print_device(q.get_device());
    std::mt19937 rng(9);
    std::normal_distribution<float> nd(0, 1);
    std::vector<float> w((size_t) n * 256);
    for (auto& v : w) v = nd(rng);
    // Launch in batches so no single kernel runs near the xe driver's 5 s timeout.
    const int64_t batch = std::min<int64_t>(n, std::max<int64_t>(64, (int64_t) (512ull << 20) / (int64_t) vit_history_bytes(K)));
    Bufs b(q, batch, K);
    std::vector<float> out((size_t) n * 256);
    // Warm-up (JIT / first-launch costs).
    q.memcpy(b.in, w.data(), batch * 1024).wait();
    viterbi(q, K, CB_MUL1, b.in, b.idx, b.out, b.hist, b.gcost, std::min<int64_t>(batch, 64)).wait();
    rt::Timer t;
    double kernel_max = 0;
    for (int64_t s = 0; s < n; s += batch)
    {
        const int64_t m = std::min(batch, n - s);
        q.memcpy(b.in, w.data() + s * 256, m * 1024).wait();
        rt::Timer tk;
        viterbi(q, K, CB_MUL1, b.in, b.idx, b.out, b.hist, b.gcost, m).wait();
        kernel_max = std::max(kernel_max, tk.s());
        q.memcpy(out.data() + s * 256, b.out, m * 1024).wait();
    }
    const double secs = t.s();
    double mse = 0;
    for (size_t i = 0; i < w.size(); ++i) mse += (out[i] - w[i]) * (out[i] - w[i]);
    mse /= (double) w.size();
    const double tps = n / secs;
    std::printf("K %d: %lld tiles in %.2f s = %.0f tiles/s (batch %lld, longest launch %.3f s), mse %.5f\n", K, (long long) n,
                secs, tps, (long long) batch, kernel_max, mse);
    std::printf("  a 31B model (1.2e8 tiles, LDLQ pass only): %.1f h\n", 1.2e8 / tps / 3600);
    return 0;
}

int main(int argc, char** argv)
{
    const std::string cmd = argc > 1 ? argv[1] : "";
    try
    {
        if (cmd == "gen" && argc > 2) { gen(argv[2]); return 0; }
        if (cmd == "probe") { auto q = rt::make_queue(); rt::print_device(q.get_device()); return 0; }
        if (cmd == "viterbi" && argc > 2) return test_viterbi(argv[2]);
        if (cmd == "bench") return bench(argc > 2 ? std::atoi(argv[2]) : 4, argc > 3 ? std::atoll(argv[3]) : 8192);
    }
    catch (const std::exception& e) { std::fprintf(stderr, "error: %s\n", e.what()); return 2; }
    std::fprintf(stderr, "usage: test_gpu gen <file> | probe | viterbi <file> | bench [K] [tiles]\n");
    return 2;
}
