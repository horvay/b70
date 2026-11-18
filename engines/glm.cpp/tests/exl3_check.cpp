// EXL3 linear on the GPU vs a CPU reference that decodes the trellis from the format description
// (docs/forward-spec.md §7.2-7.3) in double precision. For M rows of a pseudo-random input, reports the relative
// error of the GPU output (fp16 and fp32 out).
//   exl3_check <model dir> <tensor prefix> [M...]
//   exl3_check synth <K code> <codebook 2|0> <in> <out> [M...]     a pseudo-random trellis and scales, of any rate
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include "../src/exl3.h"

using namespace glm;

static float h2f(uint16_t h) { return (float)sycl::bit_cast<sycl::half>(h); }

// value t of a 16x16 tile (8K uint32 words, MSB-first bit stream that wraps inside the tile)
static double decode(const uint32_t* w, int K, int t, int cb) {
    // half-integer K (code 10 KA + 5): alternating KA, KA + 1 bits, the odd positions carrying the extra one
    int nbits = K >= 10 ? 256 * (K / 10) + 128 : 256 * K;
    int end = K >= 10 ? (t + 1) * (K / 10) + (t + 1) / 2 : (t + 1) * K;
    uint32_t st = 0;
    for (int i = end - 16; i < end; i++) {
        int b = ((i % nbits) + nbits) % nbits;
        st = (st << 1) | ((w[b / 32] >> (31 - b % 32)) & 1);
    }
    if (cb == 0) {
        // two fp16 halves of a hash, added once
        uint32_t x = ((st * 89226354u + 64248484u) & 0x8FFF8FFFu) ^ 0x3B603B60u;
        return (double)(sycl::half)((float)sycl::bit_cast<sycl::half>((uint16_t)(x & 0xFFFF)) +
                                    (float)sycl::bit_cast<sycl::half>((uint16_t)(x >> 16)));
    }
    uint32_t x = st * 0x83DCD12Du;
    uint32_t s = (x & 255) + (x >> 8 & 255) + (x >> 16 & 255) + (x >> 24);
    double h = 1024.0 + s;
    // one fp16 FMA: h * fp16(0.00676727294921875) + fp16(-10.3828125), rounded once
    return (double)(sycl::half)(float)(h * 0.00676727294921875 - 10.3828125);
}

static void fwht(std::vector<double>& v, size_t off) {
    for (int len = 1; len < 128; len <<= 1)
        for (int i = 0; i < 128; i += 2 * len)
            for (int j = i; j < i + len; j++) {
                double a = v[off + j], b = v[off + j + len];
                v[off + j] = a + b;
                v[off + j + len] = a - b;
            }
}

int main(int argc, char** argv) {
    GLM_CHECK(argc >= 3, "usage: exl3_check <model dir> <prefix> [M...]");
    const bool synth = !strcmp(argv[1], "synth");
    std::unique_ptr<SafeTensors> st;
    std::string pre = synth ? "synth" : argv[2];
    Exl3Linear L;
    std::vector<uint32_t> synth_tr;
    std::vector<uint16_t> suh, svh;
    std::vector<int> Ms;
    if (synth) {
        GLM_CHECK(argc >= 6, "usage: exl3_check synth <K code> <codebook> <in> <out> [M...]");
        L.K = atoi(argv[2]);
        L.cb = atoi(argv[3]);
        L.in = atoi(argv[4]);
        L.out = atoi(argv[5]);
        exl3_large_above = 32;
        const int wt = L.K >= 10 ? 8 * (L.K / 10) + 4 : 8 * L.K;     // uint32 words a tile
        Exl3Linear::k_code(wt * 2, L.cb == 2);
        std::mt19937 rt(L.K * 1000 + L.cb);
        synth_tr.resize((size_t)(L.in / 16) * (L.out / 16) * wt);
        for (auto& w : synth_tr) w = rt();
        std::uniform_real_distribution<float> us(0.5f, 2.0f);
        suh.resize(L.in);
        svh.resize(L.out);
        for (auto& v : suh) v = sycl::bit_cast<uint16_t>((sycl::half)us(rt));
        for (auto& v : svh) v = sycl::bit_cast<uint16_t>((sycl::half)us(rt));
        L.trellis = Tensor::empty({L.in / 16, L.out / 16, wt * 2}, DType::I16);
        L.suh = Tensor::empty({L.in}, DType::F16);
        L.svh = Tensor::empty({L.out}, DType::F16);
        L.sonb = Tensor::zeros({L.out / 128}, DType::I32);
        upload(L.trellis.p, synth_tr.data(), synth_tr.size() * 4);
        upload(L.suh.p, suh.data(), suh.size() * 2);
        upload(L.svh.p, svh.data(), svh.size() * 2);
        for (int i = 6; i < argc; i++) Ms.push_back(atoi(argv[i]));
    } else {
        st = std::make_unique<SafeTensors>(argv[1]);
        L = Exl3Linear::load(*st, pre);
        suh.resize(L.in);
        svh.resize(L.out);
        memcpy(suh.data(), st->at(pre + ".suh").data, L.in * 2);
        memcpy(svh.data(), st->at(pre + ".svh").data, L.out * 2);
        for (int i = 3; i < argc; i++) Ms.push_back(atoi(argv[i]));
    }
    printf("%s: %d -> %d, K code %d, codebook %d\n", pre.c_str(), L.in, L.out, L.K, L.cb);
    const auto* words = synth ? synth_tr.data() : static_cast<const uint32_t*>(st->at(pre + ".trellis").data);
    int tiles_n = L.out / 16, wpt = L.K >= 10 ? 8 * (L.K / 10) + 4 : 8 * L.K;
    if (Ms.empty()) Ms = {1, 2, 5, 17};
    int Mmax = *std::max_element(Ms.begin(), Ms.end());
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<sycl::half> x((size_t)Mmax * L.in);
    for (auto& v : x) v = (sycl::half)nd(rng);

    // reference: y = svh * H_n(xh @ W) / sqrt(128), xh = fp16(H(fp16(x * suh)) / sqrt(128))
    std::vector<double> W((size_t)L.in * L.out);
    for (int r = 0; r < L.in / 16; r++)
        for (int c = 0; c < tiles_n; c++) {
            const uint32_t* w = words + ((size_t)r * tiles_n + c) * wpt;
            for (int t = 0; t < 256; t++) {
                int kr = 8 * ((t >> 1) & 1) + 2 * ((t >> 3) & 3) + (t & 1), nc = (t >> 5) + 8 * ((t >> 2) & 1);
                W[(size_t)(r * 16 + kr) * L.out + c * 16 + nc] = decode(w, L.K, t, L.cb);
            }
        }
    const double rs = 1.0 / std::sqrt(128.0);
    std::vector<double> ref((size_t)Mmax * L.out);
    for (int m = 0; m < Mmax; m++) {
        std::vector<double> xh(L.in);
        for (int k = 0; k < L.in; k++) xh[k] = (double)(sycl::half)((float)x[(size_t)m * L.in + k] * h2f(suh[k]));
        for (int b = 0; b < L.in; b += 128) fwht(xh, b);
        for (int k = 0; k < L.in; k++) xh[k] = (double)(sycl::half)(float)(xh[k] * rs);
        std::vector<double> acc(L.out, 0.0);
        for (int k = 0; k < L.in; k++) {
            const double* wr = &W[(size_t)k * L.out];
            for (int n = 0; n < L.out; n++) acc[n] += xh[k] * wr[n];
        }
        for (int b = 0; b < L.out; b += 128) fwht(acc, b);
        for (int n = 0; n < L.out; n++) ref[(size_t)m * L.out + n] = acc[n] * rs * h2f(svh[n]);
    }

    auto& q = Dev::get().q;
    Workspace ws;
    ws.init(256 << 20);
    Tensor xd = Tensor::empty({Mmax, L.in}, DType::F16);
    upload(xd.p, x.data(), x.size() * 2);
    bool ok = true;
    for (int M : Ms)
        for (int f32 = 0; f32 < 2; f32++) {
            Tensor yd = Tensor::empty({M, L.out}, f32 ? DType::F32 : DType::F16);
            L.forward(q, ws, xd.ptr<half>(), M, yd.p, yd.dt);
            q.wait();
            std::vector<float> y((size_t)M * L.out);
            if (f32) download(y.data(), yd.p, y.size() * 4);
            else {
                std::vector<sycl::half> yh(y.size());
                download(yh.data(), yd.p, yh.size() * 2);
                for (size_t i = 0; i < y.size(); i++) y[i] = yh[i];
            }
            double num = 0, den = 0, mx = 0;
            for (size_t i = 0; i < y.size(); i++) {
                double d = y[i] - ref[i];
                num += d * d;
                den += ref[i] * ref[i];
                mx = std::max(mx, std::fabs(d));
            }
            double rel = std::sqrt(num / den);
            bool good = rel < 3e-3;
            ok &= good;
            printf("  M=%-3d %s out: rel err %.2e, max abs err %.3e  %s\n", M, f32 ? "fp32" : "fp16", rel, mx,
                   good ? "ok" : "BAD");
        }
    printf(ok ? "EXL3 CHECK OK\n" : "EXL3 CHECK FAILED\n");
    return ok ? 0 : 1;
}
