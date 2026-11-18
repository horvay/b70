// exl3conv: the Unicode the native tokenizer needs: UTF-8 <-> code points, \p{L}, \p{N} and \s classes, and NFC.
// Tables come from tools/gen_unicode.py (Python's unicodedata). \s is the Unicode White_Space property, as in the
// regex engines Hugging Face tokenizers uses.
#pragma once

#include "io/unicode_tables.hpp"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace uni {

inline std::vector<uint32_t> decode_utf8(const std::string& s)
{
    std::vector<uint32_t> out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size();)
    {
        const unsigned char c = (unsigned char) s[i];
        uint32_t cp; size_t n;
        if (c < 0x80) { cp = c; n = 1; }
        else if ((c >> 5) == 6) { cp = c & 0x1F; n = 2; }
        else if ((c >> 4) == 14) { cp = c & 0x0F; n = 3; }
        else if ((c >> 3) == 30) { cp = c & 0x07; n = 4; }
        else { out.push_back(0xFFFD); ++i; continue; }
        if (i + n > s.size()) { out.push_back(0xFFFD); break; }
        for (size_t k = 1; k < n; ++k) cp = (cp << 6) | ((unsigned char) s[i + k] & 0x3F);
        out.push_back(cp);
        i += n;
    }
    return out;
}

inline void append_utf8(std::string& o, uint32_t cp)
{
    if (cp < 0x80) o += (char) cp;
    else if (cp < 0x800) { o += (char) (0xC0 | (cp >> 6)); o += (char) (0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) { o += (char) (0xE0 | (cp >> 12)); o += (char) (0x80 | ((cp >> 6) & 0x3F)); o += (char) (0x80 | (cp & 0x3F)); }
    else { o += (char) (0xF0 | (cp >> 18)); o += (char) (0x80 | ((cp >> 12) & 0x3F)); o += (char) (0x80 | ((cp >> 6) & 0x3F)); o += (char) (0x80 | (cp & 0x3F)); }
}

inline std::string encode_utf8(const std::vector<uint32_t>& cps, size_t a = 0, size_t b = SIZE_MAX)
{
    std::string o;
    for (size_t i = a; i < std::min(b, cps.size()); ++i) append_utf8(o, cps[i]);
    return o;
}

template <size_t N>
inline bool in_ranges(const Range (&r)[N], uint32_t cp)
{
    const Range* e = r + N;
    const Range* it = std::upper_bound(r, e, cp, [](uint32_t v, const Range& x) { return v < x.a; });
    return it != r && cp <= (it - 1)->b;
}
inline bool is_letter(uint32_t cp) { return in_ranges(LETTER, cp); }
inline bool is_number(uint32_t cp) { return in_ranges(NUMBER, cp); }
inline bool is_space(uint32_t cp)
{
    return (cp >= 0x09 && cp <= 0x0D) || cp == 0x20 || cp == 0x85 || cp == 0xA0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) ||
           cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

inline uint8_t ccc(uint32_t cp)
{
    const CCC* e = COMBINING + sizeof(COMBINING) / sizeof(COMBINING[0]);
    const CCC* it = std::lower_bound(COMBINING, e, cp, [](const CCC& x, uint32_t v) { return x.cp < v; });
    return it != e && it->cp == cp ? it->c : 0;
}

// Hangul syllables (UAX #15 / Unicode 3.12)
constexpr uint32_t SBase = 0xAC00, LBase = 0x1100, VBase = 0x1161, TBase = 0x11A7, LCount = 19, VCount = 21, TCount = 28,
                   NCount = VCount * TCount, SCount = LCount * NCount;

inline void decompose(uint32_t cp, std::vector<uint32_t>& out)
{
    if (cp >= SBase && cp < SBase + SCount)
    {
        const uint32_t s = cp - SBase;
        out.push_back(LBase + s / NCount); out.push_back(VBase + (s % NCount) / TCount);
        if (s % TCount) out.push_back(TBase + s % TCount);
        return;
    }
    const Decomp* e = DECOMP + sizeof(DECOMP) / sizeof(DECOMP[0]);
    const Decomp* it = std::lower_bound(DECOMP, e, cp, [](const Decomp& x, uint32_t v) { return x.cp < v; });
    if (it != e && it->cp == cp) for (int k = 0; k < it->len; ++k) out.push_back(DECOMP_DATA[it->off + k]);
    else out.push_back(cp);
}

inline uint32_t compose(uint32_t a, uint32_t b)   // 0 = none
{
    if (a >= LBase && a < LBase + LCount && b >= VBase && b < VBase + VCount) return SBase + ((a - LBase) * VCount + (b - VBase)) * TCount;
    if (a >= SBase && a < SBase + SCount && (a - SBase) % TCount == 0 && b > TBase && b < TBase + TCount) return a + (b - TBase);
    const Comp* e = COMPOSE + sizeof(COMPOSE) / sizeof(COMPOSE[0]);
    const Comp* it = std::lower_bound(COMPOSE, e, std::make_pair(a, b), [](const Comp& x, const std::pair<uint32_t, uint32_t>& v) {
        return x.a != v.first ? x.a < v.first : x.b < v.second;
    });
    return it != e && it->a == a && it->b == b ? it->c : 0;
}

inline std::vector<uint32_t> nfc(const std::vector<uint32_t>& in)
{
    // Fast path: plain ASCII is already NFC.
    if (std::all_of(in.begin(), in.end(), [](uint32_t c) { return c < 0x300; }) &&
        std::none_of(in.begin(), in.end(), [](uint32_t c) { return c >= 0xC0; }))
        return in;
    std::vector<uint32_t> d;
    d.reserve(in.size());
    for (uint32_t c : in) decompose(c, d);
    // Canonical ordering: stable sort of every run of non-starters by combining class.
    for (size_t i = 0; i < d.size();)
    {
        if (!ccc(d[i])) { ++i; continue; }
        size_t j = i;
        while (j < d.size() && ccc(d[j])) ++j;
        std::stable_sort(d.begin() + i, d.begin() + j, [](uint32_t x, uint32_t y) { return ccc(x) < ccc(y); });
        i = j;
    }
    // Canonical composition.
    std::vector<uint32_t> out;
    out.reserve(d.size());
    long starter = -1;
    int last = 0;
    for (uint32_t c : d)
    {
        const int cc = ccc(c);
        if (starter >= 0 && (last < cc || last == 0))
            if (const uint32_t p = compose(out[starter], c); p && !(last == 0 && (long) out.size() - 1 != starter && cc == 0))
            {
                out[starter] = p;
                continue;
            }
        if (cc == 0) { starter = (long) out.size(); last = 0; }
        else last = cc;
        out.push_back(c);
    }
    return out;
}

} // namespace uni
