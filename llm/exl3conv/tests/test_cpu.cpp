// exl3conv CPU tests: no GPU, no SYCL. Build and run with tests/run_cpu.sh.
//
//   f16        rounding against GCC's _Float16 conversion; fma against a double fma where that is exact
//   codebook   statistics and test values of all three codebooks (notes/exl3-algorithm.md §4.1)
//   pack       unpack(random packed tile) is tail-biting and packs back to the same words, K = 1..8 and 1.5/2.5/3.5
//   encode     the CPU reference encoder: ideal codewords come back with zero error; Gaussian tiles are tail-biting,
//              decode to the reported values, and stay under exllamav3's MSE bounds (test_quantize_tiles.py)
//   reference  decode one 128x128 block of a tensor in an EXL3 model made by exllamav3 and correlate it with the
//              BF16 weight of the same architecture (EXL3_REF and EXL3_BF16 directories; skipped if unset)
#include "exl3/codebook.hpp"
#include "exl3/f16.hpp"
#include "exl3/pack.hpp"
#include "exl3/viterbi_ref.hpp"
#include "io/safetensors.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using namespace exl3;

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++failures; std::printf("  FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static void test_f16()
{
    std::printf("f16\n");
    std::mt19937_64 rng(1);
    int bad = 0;
    for (int i = 0; i < 2000000; ++i)
    {
        // Doubles across the fp16 range, including subnormals and overflow.
        const double v = std::ldexp(std::uniform_real_distribution<double>(-1, 1)(rng), (int) (rng() % 50) - 30);
        const _Float16 g = (_Float16) v;
        uint16_t gb; std::memcpy(&gb, &g, 2);
        if (f16_from_double(v) != gb) ++bad;
    }
    CHECK(bad == 0, "f16_from_double differs from _Float16 on %d of 2M values", bad);
    // Ties: 1 + 2^-11 is halfway between 1 and 1 + 2^-10; ties to even gives 1.
    CHECK(f16_from_double(1.0 + std::ldexp(1.0, -11)) == 0x3C00, "tie to even");
    CHECK(f16_from_double(65520.0) == 0x7C00 && f16_from_double(65519.0) == 0x7BFF, "overflow boundary");
    CHECK(f16_from_double(std::ldexp(1.0, -25)) == 0 && f16_from_double(std::ldexp(1.5, -25)) == 1, "subnormal rounding");

    // fma: where the double fma is exact (no rounding), both must agree.
    int compared = 0; bad = 0;
    for (int i = 0; i < 2000000; ++i)
    {
        const uint16_t a = (uint16_t) (rng() & 0x7BFF), c = (uint16_t) (rng() & 0x7BFF);
        const double da = f16_to_double(a), dc = f16_to_double(c);
        const double r = std::fma(da, da, dc);
        if (r - dc != da * da) continue;           // not exact in double (rare), skip
        ++compared;
        if (f16_fma(a, a, c) != f16_from_double(r)) ++bad;
    }
    CHECK(bad == 0, "f16_fma differs on %d of %d exact cases", bad, compared);
    // A case where a separate multiply and add round differently from the fused op.
    {
        // a*a + c with a = 1 + 2^-10, c = -1: exact 2^-9 + 2^-20, fused rounds once.
        const uint16_t a = 0x3C01, c = 0xBC00;
        CHECK(f16_fma(a, a, c) == f16_from_double(std::ldexp(1.0, -9) + std::ldexp(1.0, -20)), "fused rounding");
    }
}

static void test_codebooks()
{
    std::printf("codebook\n");
    struct Expect { const char* name; double mean, rms, mn, mx; double v[5]; };
    const Expect ex[3] = {
        {"3inst", 0.00021, 1.24371, -3.9570, 3.9727, {0.76806640625, -0.91943359375, 0.931640625, 0.29443359375, -2.19921875}},
        {"mcg", -0.00004, 1.24411, -3.9492, 3.9492, {1.84375, 0.134521484375, -0.75927734375, 0.83984375, 1.5146484375}},
        {"mul1", -0.00193, 1.00031, -3.453125, 3.3477, {-3.453125, 0.64111328125, -0.448486328125, 0.1944580078125, -0.5771484375}},
    };
    for (int cb = 0; cb < 3; ++cb)
    {
        CodebookTable t(cb);
        double s = 0, s2 = 0, mn = 1e9, mx = -1e9;
        for (uint32_t i = 0; i < 65536; ++i)
        {
            const double v = f16_to_double(t[i]);
            s += v; s2 += v * v; mn = std::min(mn, v); mx = std::max(mx, v);
        }
        const double mean = s / 65536, rms = std::sqrt(s2 / 65536);
        std::printf("  %-5s mean %+.5f rms %.5f min %.4f max %.4f\n", ex[cb].name, mean, rms, mn, mx);
        CHECK(std::fabs(mean - ex[cb].mean) < 1e-5 && std::fabs(rms - ex[cb].rms) < 1e-5, "%s mean/rms", ex[cb].name);
        CHECK(std::fabs(mn - ex[cb].mn) < 1e-3 && std::fabs(mx - ex[cb].mx) < 1e-3, "%s min/max", ex[cb].name);
        const uint32_t ids[5] = {0, 1, 2, 3, 0x1234};
        for (int j = 0; j < 5; ++j)
            CHECK(f16_to_double(t[ids[j]]) == ex[cb].v[j], "%s v[%#x] = %.12g, expected %.12g", ex[cb].name, ids[j],
                  f16_to_double(t[ids[j]]), ex[cb].v[j]);
    }
}

static void test_pack()
{
    std::printf("pack\n");
    std::mt19937 rng(2);
    const int k2s[] = {2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16};
    for (int k2 : k2s)
    {
        Rate rate(k2);
        int bad = 0;
        for (int t = 0; t < 200; ++t)
        {
            uint16_t packed[128], states[256], again[128];
            for (int i = 0; i < rate.words16(); ++i) packed[i] = (uint16_t) rng();
            unpack_tile(packed, states, rate);
            if (!is_tail_biting(states, rate)) ++bad;
            pack_tile(states, again, rate);
            for (int i = 0; i < rate.words16(); ++i) if (again[i] != packed[i]) { ++bad; break; }
        }
        CHECK(bad == 0, "K = %.1f: %d of 200 tiles failed the round trip", k2 / 2.0, bad);
    }
    // Tile permutation is a permutation.
    std::vector<int> seen(256, 0);
    for (int p : tile_perm()) seen[p]++;
    bool ok = true; for (int s : seen) ok &= s == 1;
    CHECK(ok, "tile_perm is not a permutation");
    CHECK(tile_perm()[0] == 0 && tile_perm()[1] == 16 && tile_perm()[2] == 128 && tile_perm()[8] == 32, "tile_perm values");
}

static void test_encode()
{
    std::printf("encode (CPU reference; slow)\n");
    const double max_mse[9] = {0, 0.3, 0.1, 0.1, 0.1, 0.1, 0.07, 0.05, 0.04};
    std::mt19937 rng(3);
    std::normal_distribution<float> nd(0, 1);
    const CodebookTable tables[3] = {CodebookTable(0), CodebookTable(1), CodebookTable(2)};
    const int cbs[] = {CB_MUL1, CB_3INST};
    for (int cb : cbs)
    {
        const double scale = cb == CB_MUL1 ? 1.0 : 1.24371088;
        for (int K = 1; K <= 8; ++K)
        {
            const auto t0 = std::chrono::steady_clock::now();
            ViterbiRef enc(K, cb, tables[cb]);
            Rate rate = Rate::integer(K);
            // Ideal codeword: random packed bits -> tail-biting states -> their values.
            uint16_t packed[128], states[256], idx[256];
            float w[256], out[256];
            for (int i = 0; i < rate.words16(); ++i) packed[i] = (uint16_t) rng();
            unpack_tile(packed, states, rate);
            for (int i = 0; i < 256; ++i) w[i] = f16_to_float(tables[cb][states[i]]);
            enc.encode(w, idx, out);
            double err = 0;
            for (int i = 0; i < 256; ++i) err += (out[i] - w[i]) * (out[i] - w[i]);
            CHECK(err == 0, "cb %d K %d: ideal codeword error %g", cb, K, err);
            CHECK(is_tail_biting(idx, rate), "cb %d K %d: ideal codeword output not tail-biting", cb, K);

            // Gaussian tile.
            for (int i = 0; i < 256; ++i) w[i] = nd(rng) * (float) scale;
            enc.encode(w, idx, out);
            double mse = 0; bool consistent = true;
            for (int i = 0; i < 256; ++i)
            {
                mse += ((out[i] - w[i]) / scale) * ((out[i] - w[i]) / scale);
                consistent &= out[i] == f16_to_float(tables[cb][idx[i]]);
            }
            mse /= 256;
            CHECK(is_tail_biting(idx, rate), "cb %d K %d: output not tail-biting", cb, K);
            CHECK(consistent, "cb %d K %d: output values are not the decoded states", cb, K);
            CHECK(mse < max_mse[K], "cb %d K %d: mse %.4f >= %.2f", cb, K, mse, max_mse[K]);
            // Packing the states and unpacking them gives the states back (tail-biting makes that exact).
            uint16_t p2[128], s2[256];
            pack_tile(idx, p2, rate); unpack_tile(p2, s2, rate);
            bool same = true; for (int i = 0; i < 256; ++i) same &= s2[i] == idx[i];
            CHECK(same, "cb %d K %d: pack/unpack of encoded states", cb, K);
            const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::printf("  cb %d K %d  gaussian mse %.4f  (%.1f s for 2 tiles)\n", cb, K, mse, secs);
        }
    }
}

// W[in, out] block of an EXL3 linear: diag(suh) H128 W_hat H128 diag(svh), rows r0.., cols c0.., 128 x 128.
static std::vector<double> decode_block(const st::Model& m, const std::string& key, int r0, int c0)
{
    const auto& tr = m.get(key + ".trellis");
    const auto& suh = m.get(key + ".suh");
    const auto& svh = m.get(key + ".svh");
    const int tk = (int) tr.shape[0], tn = (int) tr.shape[1], words = (int) tr.shape[2];
    const int K = words / 16;
    (void) tk;
    const CodebookTable table(m.has(key + ".mul1") ? CB_MUL1 : m.has(key + ".mcg") ? CB_MCG : CB_3INST);
    Rate rate = Rate::integer(K);
    std::vector<double> W(128 * 128);
    const uint16_t* td = (const uint16_t*) tr.data;
    for (int a = 0; a < 8; ++a)
        for (int b = 0; b < 8; ++b)
        {
            const int ta = r0 / 16 + a, tb = c0 / 16 + b;
            uint16_t states[256];
            unpack_tile(td + ((size_t) ta * tn + tb) * words, states, rate);
            for (int i = 0; i < 256; ++i)
            {
                const int e = tile_perm()[i], r = e / 16, c = e % 16;
                W[(a * 16 + r) * 128 + b * 16 + c] = f16_to_double(table[states[i]]);
            }
        }
    // H128 = Sylvester / sqrt(128): H[i][j] = (-1)^popcount(i & j) / sqrt(128).
    auto H = [](int i, int j) { return (__builtin_popcount(i & j) & 1 ? -1.0 : 1.0) / std::sqrt(128.0); };
    std::vector<double> T(128 * 128, 0), R(128 * 128, 0);
    for (int i = 0; i < 128; ++i) for (int k = 0; k < 128; ++k) { const double h = H(i, k); for (int j = 0; j < 128; ++j) T[i * 128 + j] += h * W[k * 128 + j]; }
    for (int i = 0; i < 128; ++i) for (int k = 0; k < 128; ++k) { const double t = T[i * 128 + k]; for (int j = 0; j < 128; ++j) R[i * 128 + j] += t * H(k, j); }
    const uint16_t* su = (const uint16_t*) suh.data; const uint16_t* sv = (const uint16_t*) svh.data;
    for (int i = 0; i < 128; ++i) for (int j = 0; j < 128; ++j) R[i * 128 + j] *= f16_to_double(su[r0 + i]) * f16_to_double(sv[c0 + j]);
    return R;
}

static double corr(const std::vector<double>& a, const std::vector<double>& b)
{
    double ma = 0, mb = 0; for (size_t i = 0; i < a.size(); ++i) { ma += a[i]; mb += b[i]; }
    ma /= a.size(); mb /= b.size();
    double sab = 0, saa = 0, sbb = 0;
    for (size_t i = 0; i < a.size(); ++i) { sab += (a[i] - ma) * (b[i] - mb); saa += (a[i] - ma) * (a[i] - ma); sbb += (b[i] - mb) * (b[i] - mb); }
    return sab / std::sqrt(saa * sbb);
}

static void test_reference()
{
    const char* ref = std::getenv("EXL3_REF");
    const char* bf = std::getenv("EXL3_BF16");
    if (!ref || !bf) { std::printf("reference (skipped: set EXL3_REF and EXL3_BF16)\n"); return; }
    std::printf("reference\n");
    st::Model mq(ref), mb(bf);
    const char* keys[] = {"model.language_model.layers.59.self_attn.q_proj", "model.language_model.layers.59.mlp.down_proj",
                          "model.language_model.layers.0.mlp.gate_proj"};
    for (const char* key : keys)
    {
        const auto W = decode_block(mq, key, 0, 0);
        const auto& hf = mb.get(std::string(key) + ".weight");     // [out, in] BF16
        const int in = (int) hf.shape[1];
        const uint16_t* h = (const uint16_t*) hf.data;
        std::vector<double> B(128 * 128), Bt(128 * 128);
        for (int i = 0; i < 128; ++i) for (int j = 0; j < 128; ++j)
        {
            B[i * 128 + j] = st::bf16_to_float(h[(size_t) j * in + i]);    // W[in i, out j] = hf[j, i]
            Bt[i * 128 + j] = st::bf16_to_float(h[(size_t) i * in + j]);   // the wrong orientation
        }
        const double c = corr(W, B), ct = corr(W, Bt);
        std::printf("  %s: corr %.4f (transposed %.4f)\n", key, c, ct);
        CHECK(c > 0.95 && std::fabs(ct) < 0.2, "%s decode does not match the BF16 weight", key);
    }
}

int main(int argc, char** argv)
{
    const std::string only = argc > 1 ? argv[1] : "";
    auto want = [&](const char* n) { return only.empty() || only == n; };
    if (want("f16")) test_f16();
    if (want("codebook")) test_codebooks();
    if (want("pack")) test_pack();
    if (want("reference")) test_reference();
    if (want("encode")) test_encode();
    std::printf(failures ? "\n%d FAILED\n" : "\nall passed\n", failures);
    return failures ? 1 : 0;
}
