// exl3conv: EXL3 tile order and trellis packing (host side). notes/exl3-algorithm.md §5.
//
// Tile order: ring position i = 8t + j (t = 0..31, j = 0..7) holds the row-major tile element perm[i] = 16r + c with
//   r = 2 (t % 4) + (j & 1) + 8 ((j >> 1) & 1),   c = t / 4 + 8 ((j >> 2) & 1)
// The encoder walks the ring in this order; indices are packed in it; decoded values are scattered back through it.
//
// Packing, K bits per position (integer K; half-integer K = KA + 0.5 uses D(i) = KA + bit (i & 15) of 0xAAAA):
//   ring bitstream B of R = 256 K bits; position i writes the low D(i) bits of its state, MSB first, at
//   B[S(i) - D(i) .. S(i)), S(i) = D(0) + .. + D(i). 32-bit word j = B[32j .. 32j + 31] with B[32j] at bit 31;
//   stored little-endian as uint16[2j] = low half, uint16[2j + 1] = high half. A tile is R / 16 uint16.
// Unpacking: the state of position i is the 16 ring bits ending at S(i) (wrapping), so the encoder's states must be
// tail-biting: each state's top 16 - D(i) bits equal the previous state's low 16 - D(i) bits, around the ring.
#pragma once

#include <array>
#include <cstdint>

namespace exl3 {

constexpr int TILE = 256;
constexpr uint16_t FRAC_MASK = 0xAAAA;

inline const std::array<int, TILE>& tile_perm()
{
    static const std::array<int, TILE> perm = [] {
        std::array<int, TILE> p{};
        for (int t = 0; t < 32; ++t)
            for (int j = 0; j < 8; ++j)
            {
                const int r = 2 * (t % 4) + (j & 1) + 8 * ((j >> 1) & 1);
                const int c = t / 4 + 8 * ((j >> 2) & 1);
                p[8 * t + j] = 16 * r + c;
            }
        return p;
    }();
    return perm;
}

// Bit widths per ring position. K2 = 2K (so 9 means K = 4.5); integer K when K2 is even.
struct Rate
{
    int K2;
    explicit Rate(int k2) : K2(k2) {}
    static Rate integer(int K) { return Rate(2 * K); }
    bool is_int() const { return (K2 & 1) == 0; }
    int ka() const { return K2 / 2; }
    int width(int i) const { return is_int() ? ka() : ka() + ((FRAC_MASK >> (i & 15)) & 1); }
    int ring_bits() const { return 128 * K2; }          // 256 K
    int words16() const { return 8 * K2; }              // 16 K uint16 per tile
};

// states[256] (ring order) -> out[16 K] uint16. Only the low D(i) bits of each state are stored.
inline void pack_tile(const uint16_t* states, uint16_t* out, Rate rate)
{
    const int R = rate.ring_bits(), W32 = R / 32;
    uint32_t words[64] = {};
    int pos = 0;
    for (int i = 0; i < TILE; ++i)
    {
        const int d = rate.width(i);
        for (int b = d - 1; b >= 0; --b, ++pos)
            if ((states[i] >> b) & 1) words[pos >> 5] |= 0x80000000u >> (pos & 31);
    }
    for (int j = 0; j < W32; ++j)
    {
        out[2 * j] = (uint16_t) (words[j] & 0xFFFF);
        out[2 * j + 1] = (uint16_t) (words[j] >> 16);
    }
}

// in[16 K] uint16 -> states[256] in ring order.
inline void unpack_tile(const uint16_t* in, uint16_t* states, Rate rate)
{
    const int R = rate.ring_bits(), W32 = R / 32;
    uint32_t words[64];
    for (int j = 0; j < W32; ++j) words[j] = (uint32_t) in[2 * j] | ((uint32_t) in[2 * j + 1] << 16);
    auto bit = [&](int p) { p %= R; if (p < 0) p += R; return (words[p >> 5] >> (31 - (p & 31))) & 1u; };
    int S = 0;
    for (int i = 0; i < TILE; ++i)
    {
        S += rate.width(i);
        uint32_t s = 0;
        for (int b = 0; b < 16; ++b) s = (s << 1) | bit(S - 16 + b);
        states[i] = (uint16_t) s;
    }
}

inline bool is_tail_biting(const uint16_t* states, Rate rate)
{
    for (int i = 0; i < TILE; ++i)
    {
        const int d = rate.width(i);
        const uint32_t prev = states[(i + TILE - 1) % TILE];
        if ((uint32_t) (states[i] >> d) != (prev & ((1u << (16 - d)) - 1))) return false;
    }
    return true;
}

} // namespace exl3
