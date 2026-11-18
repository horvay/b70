// Edge0's routed experts for prompts on the XMX units (ESIMD DPAS), over tiled NVFP4 records (e0_ops.h): a tile of
// 16 rows x 16 columns is one DPAS B operand, decoded exactly to fp16. Blocks of up to 32 rows of one expert; a thread
// owns 16 output columns (gate and up: the same 16 rows of each, so it writes h itself; down: 32 columns).
// Blocks of up to 64 rows (e0_ops.h says 32 for the callers' tables: any block size up to 64 works).
// Built into libglm_moe_pf.so (256 registers a thread).
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include "e0_ops.h"

namespace glm::e0ops {

namespace {
using namespace sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;
using fp16 = sycl::half;
constexpr int MB = 64;                             // rows a block (each block decodes the expert's weights once)

// 2 x E2M1 of 4-bit codes (sign bit 3): 0, 1, 2, 3, 4, 6, 8, 12
template <int N>
ESIMD_INLINE simd<int16_t, N> e2m1x2v(simd<uint16_t, N> q) {
    simd<int16_t, N> n = q & 7;
    simd<int16_t, N> sh = (n >> 1) - 1;
    sh.merge(simd<int16_t, N>(0), sh < 0);
    simd<int16_t, N> v = (2 + (n & 1)) << sh;
    v.merge(n, n < 2);
    simd<int16_t, N> neg = -v;
    v.merge(neg, (q & 8) != 0);
    return v;
}

// tile t -> B, VNNI fp16 [8 k pairs][16 n][2]: B[kp][n][j] = value of row n at column 2 kp + j (exact in fp16)
ESIMD_INLINE simd<fp16, 256> tile_b(const uint8_t* t) {
    simd<uint8_t, 16> scb = block_load<uint8_t, 16>(t);
    simd<uint8_t, 128> cb = block_load<uint8_t, 128>(t + 16);
    // 0.5 x UE4M3 (ggml: 0 and 0x7F are zero; bit 7 ignored), the codes below being 2 x E2M1
    simd<uint32_t, 16> sc = scb;
    simd<uint32_t, 16> e = (sc >> 3) & 15, m = sc & 7;
    simd<uint32_t, 16> bits = ((e + 119) << 23) | (m << 20);
    simd<float, 16> sv = bits.template bit_cast_view<float>();
    simd<float, 16> subn = convert<float>(m) * (1.0f / 1024.0f);
    sv.merge(subn, e == 0);
    sv.merge(simd<float, 16>(0.0f), sc == 0x7F);
    simd<fp16, 16> s16 = convert<fp16>(sv);
    simd<fp16, 32> s32;
    s32.template select<16, 2>(0) = s16;
    s32.template select<16, 2>(1) = s16;
    simd<uint16_t, 128> c = cb;
    simd<fp16, 128> lo = convert<fp16>(e2m1x2v<128>(c & 15));      // [n][8]: columns 0..7
    simd<fp16, 128> hi = convert<fp16>(e2m1x2v<128>(c >> 4));      // [n][8]: columns 8..15
    simd<fp16, 256> B;
#pragma unroll
    for (int kp = 0; kp < 4; kp++) {
        B.template select<16, 2>(kp * 32) = lo.template select<16, 8>(2 * kp);
        B.template select<16, 2>(kp * 32 + 1) = lo.template select<16, 8>(2 * kp + 1);
        B.template select<16, 2>((kp + 4) * 32) = hi.template select<16, 8>(2 * kp);
        B.template select<16, 2>((kp + 4) * 32 + 1) = hi.template select<16, 8>(2 * kp + 1);
    }
    return B * s32.template replicate<8>();
}

// A: rows rb*8 .. rb*8+7 of a block (clamped to its last row), 16 columns at col
ESIMD_INLINE simd<fp16, 128> load_a(const fp16* X, int stride, int r0, int n, int rb, int col) {
    simd<fp16, 128> A;
#pragma unroll
    for (int rr = 0; rr < 8; rr++) {
        int row = rb * 8 + rr;
        row = row < n ? row : n - 1;
        A.template select<16, 1>(rr * 16) = block_load<fp16, 16>(X + (size_t)(r0 + row) * stride + col);
    }
    return A;
}

struct GuDpas {
    const fp16* X;                                  // [rows, ELL]
    int rows;
    const int64_t* brec;
    const int* br0;
    const int* bn;
    fp16* H;                                        // [rows, 640]
    void operator()(sycl::nd_item<2> it) const SYCL_ESIMD_KERNEL {
        const int b = it.get_group(0), tp = it.get_group(1) * 8 + it.get_local_id(1);   // 16 h columns: 0..39
        const int n = bn[b], r0 = br0[b], nrb = (n + 7) / 8;
        const uint8_t* rec = reinterpret_cast<const uint8_t*>(brec[b]);
        const uint8_t* gt = rec + (size_t)tp * (ELL / 16) * NV_TILE;
        const uint8_t* ut = rec + (size_t)(EI / 16 + tp) * (ELL / 16) * NV_TILE;
        simd<float, (MB / 8) * 128> ag = 0.0f, au = 0.0f;
        for (int s = 0; s < ELL / 16; s++) {
            simd<fp16, 256> Bg = tile_b(gt + (size_t)s * NV_TILE);
            simd<fp16, 256> Bu = tile_b(ut + (size_t)s * NV_TILE);
#pragma unroll
            for (int r4 = 0; r4 < MB / 32; r4++) {
                if (r4 * 4 < nrb) {
                    // 32 rows x 16 columns in one 2D block load (rows past the chunk read as zero)
                    simd<fp16, 512> A4 = load_2d<fp16, 16, 32>(X, ELL * 2 - 1, rows - 1, ELL * 2 - 1, s * 16, r0 + r4 * 32);
#pragma unroll
                    for (int q4 = 0; q4 < 4; q4++) {
                        const int rb = r4 * 4 + q4;
                        if (rb < nrb) {
                            simd<fp16, 128> A = A4.template select<128, 1>(q4 * 128);
                            auto cg = ag.template select<128, 1>(rb * 128);
                            cg = xmx::dpas<8, 8, float, float, fp16, fp16>(simd<float, 128>(cg), Bg, A);
                            auto cu = au.template select<128, 1>(rb * 128);
                            cu = xmx::dpas<8, 8, float, float, fp16, fp16>(simd<float, 128>(cu), Bu, A);
                        }
                    }
                }
            }
        }
        simd<float, 4> tl = block_load<float, 4>(reinterpret_cast<const float*>(rec + NV_TAIL));
        const float sg = tl[0], su = tl[1];
#pragma unroll
        for (int r = 0; r < MB; r++) {
            if (r < n) {
                simd<float, 16> g = ag.template select<16, 1>(r * 16) * sg;
                simd<float, 16> u = au.template select<16, 1>(r * 16) * su;
                simd<float, 16> h = g / (1.0f + exp(-g)) * u;
                block_store<fp16, 16>(H + (size_t)(r0 + r) * EI + tp * 16, convert<fp16>(h));
            }
        }
    }
};

struct DnDpas {
    const fp16* H;                                  // [rows, 640]
    int rows;
    const int64_t* brec;
    const int* br0;
    const int* bn;
    fp16* Y;                                        // [rows, ELL]
    void operator()(sycl::nd_item<2> it) const SYCL_ESIMD_KERNEL {
        const int b = it.get_group(0), tp = it.get_group(1) * 8 + it.get_local_id(1);   // 32 columns: 0..55
        const int n = bn[b], r0 = br0[b], nrb = (n + 7) / 8;
        const uint8_t* rec = reinterpret_cast<const uint8_t*>(brec[b]);
        const uint8_t* t0 = rec + NV_DOWN_OFF + (size_t)(2 * tp) * (EI / 16) * NV_TILE;
        const uint8_t* t1 = t0 + (size_t)(EI / 16) * NV_TILE;
        simd<float, (MB / 8) * 128> a0 = 0.0f, a1 = 0.0f;
        for (int s = 0; s < EI / 16; s++) {
            simd<fp16, 256> B0 = tile_b(t0 + (size_t)s * NV_TILE);
            simd<fp16, 256> B1 = tile_b(t1 + (size_t)s * NV_TILE);
#pragma unroll
            for (int r4 = 0; r4 < MB / 32; r4++) {
                if (r4 * 4 < nrb) {
                    simd<fp16, 512> A4 = load_2d<fp16, 16, 32>(H, EI * 2 - 1, rows - 1, EI * 2 - 1, s * 16, r0 + r4 * 32);
#pragma unroll
                    for (int q4 = 0; q4 < 4; q4++) {
                        const int rb = r4 * 4 + q4;
                        if (rb < nrb) {
                            simd<fp16, 128> A = A4.template select<128, 1>(q4 * 128);
                            auto c0 = a0.template select<128, 1>(rb * 128);
                            c0 = xmx::dpas<8, 8, float, float, fp16, fp16>(simd<float, 128>(c0), B0, A);
                            auto c1 = a1.template select<128, 1>(rb * 128);
                            c1 = xmx::dpas<8, 8, float, float, fp16, fp16>(simd<float, 128>(c1), B1, A);
                        }
                    }
                }
            }
        }
        simd<float, 4> tl = block_load<float, 4>(reinterpret_cast<const float*>(rec + NV_TAIL));
        const float sd = tl[2];
#pragma unroll
        for (int r = 0; r < MB; r++) {
            if (r < n) {
                fp16* yr = Y + (size_t)(r0 + r) * ELL + tp * 32;
                block_store<fp16, 16>(yr, convert<fp16>(simd<float, 16>(a0.template select<16, 1>(r * 16) * sd)));
                block_store<fp16, 16>(yr + 16, convert<fp16>(simd<float, 16>(a1.template select<16, 1>(r * 16) * sd)));
            }
        }
    }
};
}  // namespace

void nv_gate_up_dpas(sycl::queue& q, const half* X, int rows, const int64_t* brec, const int* br0, const int* bn, int nb,
                     half* H) {
    if (nb <= 0) return;
    sycl::ext::oneapi::experimental::properties props{sycl::ext::intel::experimental::grf_size<256>};
    GuDpas k{X, rows, brec, br0, bn, H};
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<2>({(size_t)nb, (size_t)(EI / 16)}, {1, 8}), props, k);
    });
}

void nv_down_dpas(sycl::queue& q, const half* H, int rows, const int64_t* brec, const int* br0, const int* bn, int nb,
                  half* Y) {
    if (nb <= 0) return;
    sycl::ext::oneapi::experimental::properties props{sycl::ext::intel::experimental::grf_size<256>};
    DnDpas k{H, rows, brec, br0, bn, Y};
    q.submit([&](sycl::handler& h) {
        h.parallel_for(sycl::nd_range<2>({(size_t)nb, (size_t)(ELL / 32)}, {1, 8}), props, k);
    });
}

}  // namespace glm::e0ops
