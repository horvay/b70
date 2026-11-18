// exl3conv: the EXL3 trellis codebooks (host side; the SYCL encoder has its own copy of the same integer math).
//
// A trellis state is 16 bits; the codebook maps it to one fp16 weight (notes/exl3-algorithm.md §4.1):
//   3INST (cb 0): x = s * 89226354 + 64248484 (mod 2^32)
//   MCG   (cb 1): x = s * 0xCBAC1FED          (mod 2^32)
//                 both: x = (x & 0x8FFF8FFF) ^ 0x3B603B60; value = rne(half(x & 0xFFFF) + half(x >> 16))
//   MUL1  (cb 2): x = s * 0x83DCD12D          (mod 2^32)
//                 b = sum of the 4 bytes of x; value = rne((1024 + b) * K_INV + K_BIAS)
//                 K_INV = half 0x1EEE, K_BIAS = half 0xC931
// Each value has one rounding, as the CUDA kernels' __hadd / __hfma; double holds the exact sum first.
#pragma once

#include "f16.hpp"

#include <array>
#include <cstdint>

namespace exl3 {

enum Codebook : int { CB_3INST = 0, CB_MCG = 1, CB_MUL1 = 2 };

constexpr uint32_t MUL1_MULT = 0x83DCD12Du;     // also the value of the `.mul1` marker tensor (as int32)
constexpr uint32_t MCG_MULT = 0xCBAC1FEDu;      // the `.mcg` marker
constexpr uint16_t MUL1_K_INV = 0x1EEE;
constexpr uint16_t MUL1_K_BIAS = 0xC931;
constexpr float CODEBOOK_SCALE = 1.24371088f;   // RMS of the 3INST codebook; regularize() divides by it for every codebook

inline uint16_t cb_decode(int cb, uint32_t s)
{
    s &= 0xFFFF;
    if (cb == CB_MUL1)
    {
        const uint32_t x = s * MUL1_MULT;
        const uint32_t b = (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + (x >> 24);
        const double h = 1024.0 + (double) b;
        return f16_from_double(h * f16_to_double(MUL1_K_INV) + f16_to_double(MUL1_K_BIAS));
    }
    uint32_t x = (cb == CB_3INST) ? s * 89226354u + 64248484u : s * MCG_MULT;
    x = (x & 0x8FFF8FFFu) ^ 0x3B603B60u;
    return f16_from_double(f16_to_double((uint16_t) (x & 0xFFFF)) + f16_to_double((uint16_t) (x >> 16)));
}

// 65,536-entry table of one codebook, built once.
struct CodebookTable
{
    std::array<uint16_t, 65536> v;
    explicit CodebookTable(int cb) { for (uint32_t s = 0; s < 65536; ++s) v[s] = cb_decode(cb, s); }
    uint16_t operator[](uint32_t s) const { return v[s & 0xFFFF]; }
};

} // namespace exl3
