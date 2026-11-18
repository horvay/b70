// exl3conv: exact IEEE binary16 helpers for host code (tests, the CPU reference encoder, file IO).
//
// Every fp16 here is its 16-bit pattern (uint16_t). The encoder's arithmetic is reproduced with one rounding per
// operation, the way the CUDA kernel's __hsub / __hmul / __hfma round:
//   f16_sub(a, b)      = rne(a - b)        the difference of two fp16 values is exact in double
//   f16_mul(a, b)      = rne(a * b)        exact in double (22-bit product)
//   f16_fma(a, b, c)   = rne(a * b + c)    exact in __int128 units of 2^-48, then one rounding
// rne() is round-to-nearest-even with fp16 subnormals and overflow to +-inf.
//
// Finite fp16 values are integer multiples of 2^-24 below 2^16, so a product is a multiple of 2^-48 below 2^32 and
// a sum of a product and an fp16 fits an __int128 exactly. That is why the fma needs no wider format.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

namespace exl3 {

inline bool f16_is_nan(uint16_t h) { return (h & 0x7C00) == 0x7C00 && (h & 0x03FF); }
inline bool f16_is_inf(uint16_t h) { return (h & 0x7FFF) == 0x7C00; }
inline bool f16_finite(uint16_t h) { return (h & 0x7C00) != 0x7C00; }

inline double f16_to_double(uint16_t h)
{
    const int s = h >> 15, e = (h >> 10) & 0x1F, m = h & 0x3FF;
    double v;
    if (e == 0) v = std::ldexp((double) m, -24);
    else if (e == 31) v = m ? NAN : INFINITY;
    else v = std::ldexp((double) (m | 0x400), e - 25);
    return s ? -v : v;
}

inline float f16_to_float(uint16_t h) { return (float) f16_to_double(h); }

// Round sign * mag * 2^exp to fp16 (round to nearest, ties to even). mag is a non-negative integer.
inline uint16_t f16_round_scaled(bool neg, unsigned __int128 mag, int exp)
{
    const uint16_t sign = neg ? 0x8000 : 0;
    if (mag == 0) return sign;
    // Position of the highest set bit.
    const uint64_t hi = (uint64_t) (mag >> 64), lo = (uint64_t) mag;
    const int hb = hi ? 127 - __builtin_clzll(hi) : 63 - __builtin_clzll(lo);
    // Value = mag * 2^exp, leading bit at 2^(hb + exp). fp16 normal: leading bit 2^E with E in [-14, 15],
    // 10 fraction bits, so the unit in the last place is 2^(E - 10); subnormal ulp is 2^-24.
    int E = hb + exp;
    int ulp_exp = (E < -14) ? -24 : E - 10;
    int shift = ulp_exp - exp;                      // bits of mag below the ulp
    unsigned __int128 q;
    if (shift <= 0) q = mag << (-shift);
    else
    {
        if (shift >= 127) return sign;              // far below the smallest subnormal
        q = mag >> shift;
        const unsigned __int128 rem = mag - (q << shift);
        const unsigned __int128 half = (unsigned __int128) 1 << (shift - 1);
        if (rem > half || (rem == half && (q & 1))) ++q;
    }
    // q is the significand in units of 2^ulp_exp. Rounding up may carry into the next binade.
    if (ulp_exp == -24)
    {
        // Subnormal range (or the carry from it into the smallest normal, which has the same bit pattern).
        return sign | (uint16_t) q;
    }
    if (q >= 2048) { q >>= 1; ++E; }                // carry: 2^11 -> 2^10 at the next exponent (q is even then)
    if (E > 15) return sign | 0x7C00;
    return sign | (uint16_t) ((E + 15) << 10) | (uint16_t) (q & 0x3FF);
}

// Exact value of a finite fp16 as (neg, integer magnitude in units of 2^-24).
inline void f16_split(uint16_t h, bool& neg, uint64_t& units)
{
    neg = h >> 15;
    const int e = (h >> 10) & 0x1F, m = h & 0x3FF;
    units = (e == 0) ? (uint64_t) m : ((uint64_t) (m | 0x400) << (e - 1));
}

inline uint16_t f16_from_double(double v)
{
    if (std::isnan(v)) return 0x7E00;
    if (std::isinf(v)) return v < 0 ? 0xFC00 : 0x7C00;
    const bool neg = std::signbit(v);
    double a = std::fabs(v);
    if (a == 0) return neg ? 0x8000 : 0;
    // a = mant * 2^(exp - 53), mant a 53-bit integer: exact.
    int ex;
    double fr = std::frexp(a, &ex);                  // a = fr * 2^ex, fr in [0.5, 1)
    uint64_t mant = (uint64_t) std::ldexp(fr, 53);
    return f16_round_scaled(neg, mant, ex - 53);
}

inline uint16_t f16_from_float(float v) { return f16_from_double((double) v); }

// rne(a - b), rne(a * b) for finite operands; inf / nan follow IEEE through double.
inline uint16_t f16_sub(uint16_t a, uint16_t b) { return f16_from_double(f16_to_double(a) - f16_to_double(b)); }
inline uint16_t f16_mul(uint16_t a, uint16_t b) { return f16_from_double(f16_to_double(a) * f16_to_double(b)); }

// rne(a * b + c) with a single rounding.
inline uint16_t f16_fma(uint16_t a, uint16_t b, uint16_t c)
{
    if (!f16_finite(a) || !f16_finite(b) || !f16_finite(c))
        return f16_from_double(std::fma(f16_to_double(a), f16_to_double(b), f16_to_double(c)));
    bool na, nb, nc; uint64_t ua, ub, uc;
    f16_split(a, na, ua); f16_split(b, nb, ub); f16_split(c, nc, uc);
    // Units of 2^-48.
    __int128 p = (__int128) ((unsigned __int128) ua * ub);
    if (na != nb) p = -p;
    __int128 cc = (__int128) ((unsigned __int128) uc << 24);
    if (nc) cc = -cc;
    __int128 s = p + cc;
    if (s == 0)
    {
        // IEEE: an exact zero sum is +0 in round-to-nearest unless both addends are -0.
        const bool p_negzero = p == 0 && na != nb;
        const bool c_negzero = uc == 0 && nc;
        return p_negzero && c_negzero ? 0x8000 : 0;
    }
    const bool neg = s < 0;
    return f16_round_scaled(neg, (unsigned __int128) (neg ? -s : s), -48);
}

// Order of non-negative fp16 values (and +inf): their bit patterns compare as unsigned integers.
inline bool f16_lt_nonneg(uint16_t a, uint16_t b) { return a < b; }

} // namespace exl3
