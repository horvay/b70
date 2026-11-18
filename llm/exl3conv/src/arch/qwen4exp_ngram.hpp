// exl3conv: Qwen3.8-Flash-Next's hashed n-gram ids (HF Qwen4ExpTextNgramEmbedding.forward; notes/qwen4exp-forward.md
// §2.4). A row's history is [EOS, EOS] + ids; the token s places back counts only if no EOS lies between it and the
// current token (an n-gram never reaches back across an EOS; the EOS itself still sees its own context):
//   mix2 = t0*m0 ^ t1*m1,  mix3 = mix2 ^ t2*m2   (int64, no overflow: every multiplier < 2^63 / vocab)
//   id[h] = (h < 8 ? mix2 : mix3) % size[h] + offset[h],  h = 0..15
#pragma once

#include <cstdint>
#include <vector>

namespace qwen4exp {

inline void ngram_ids(const int32_t* ids, int64_t T, int64_t eos, const int64_t mult[3], const int64_t* sizes,
                      const int64_t* offsets, int64_t* out /* T x 16 */)
{
    std::vector<int64_t> hist(T + 2);
    hist[0] = hist[1] = eos;
    for (int64_t t = 0; t < T; ++t) hist[t + 2] = ids[t];
    int64_t seg = 2;                                     // first position after the last EOS before p
    for (int64_t p = 2; p < T + 2; ++p)
    {
        auto tok = [&](int s) { return p - seg >= s ? hist[p - s] : eos; };
        const int64_t t0 = hist[p], t1 = tok(1), t2 = tok(2);
        const int64_t mix2 = (t0 * mult[0]) ^ (t1 * mult[1]);
        const int64_t mix3 = mix2 ^ (t2 * mult[2]);
        int64_t* o = out + (p - 2) * 16;
        for (int h = 0; h < 16; ++h) o[h] = (h < 8 ? mix2 : mix3) % sizes[h] + offsets[h];
        if (hist[p] == eos) seg = p + 1;
    }
}

} // namespace qwen4exp
