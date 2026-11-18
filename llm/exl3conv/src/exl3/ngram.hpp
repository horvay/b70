// exl3conv: the hashed n-gram embedding table of Qwen3.8-Flash-Next as exllamav3's exl3_ngram_trellis format
// (notes/qwen4exp-forward.md §3.7; exllamav3 conversion/ngram.py, modules/quant/exl3_lib/ngram_codec.py; the B70
// server reads the same codec, trellis-serve xpu/tests/ngram_codec_ref.py).
//
// Each 160-wide row is one tail-biting mul1 ring of 160 positions in natural order, plus an fp16 scale, after a
// per-hash-head bias is subtracted:
//   w      = row - bias[head(row)]
//   cs     = clamp(gamma / max(absmax(w) / max(rms(w), 1e-12), 1e-6), 0.55, cs_hi)      (gamma, cs_hi) per K
//   scale0 = f16(max(rms(w) / cs, 1e-8))  (1 if that rounds to 0)
//   states = viterbi_160(w / scale0);  q = decoded values
//   scale  = f16(sum(w q) / max(sum(q q), 1e-12))                                         least squares
// Packed row: 1 + 10K uint16 words; word 0 = the scale's bits; stream bit b is bit (b & 15) of word 1 + (b >> 4),
// and stream bits [iK, iK + K) are the low K bits of state i, bit j at iK + j (LSB first, unlike linear tiles).
// Decode: state i = sym(i) | sym(i-1) << K | sym(i-2) << 2K ... (mod 160) & 0xFFFF; row = f16(mul1(state)) * scale
// + bias.
#pragma once

#include "exl3/f16.hpp"
#include "exl3/viterbi.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace exl3::ngram {

constexpr int ROW = 160;
inline int words_per_row(int K) { return 1 + ROW * K / 16; }

// (gamma, cs_hi) of exllamav3's CS_HEURISTIC, indexed by K.
inline void cs_heuristic(int K, float& gamma, float& hi)
{
    static const float G[9] = {0, 4.0f, 3.6f, 3.2f, 3.0f, 3.0f, 3.0f, 3.0f, 3.0f};
    static const float H[9] = {0, 1.16f, 0.98f, 0.98f, 0.98f, 0.95f, 0.92f, 0.90f, 0.86f};
    gamma = G[K]; hi = H[K];
}
constexpr float CS_MIN = 0.55f;

// Head of a global row: searchsorted(offsets, r, right) - 1, clamped to 0..15 (padding rows count as the last head).
inline int head_of(const int64_t* offsets, int nheads, int64_t r)
{
    int h = (int) (std::upper_bound(offsets, offsets + nheads, r) - offsets) - 1;
    return std::clamp(h, 0, nheads - 1);
}

// Host pack / unpack of one row (tests and the CPU reference).
inline void pack_row(const uint16_t* states, uint16_t scale_bits, int K, uint16_t* out)
{
    const int W = ROW * K / 16;
    out[0] = scale_bits;
    std::fill(out + 1, out + 1 + W, 0);
    for (int i = 0; i < ROW; ++i)
        for (int j = 0; j < K; ++j)
            if ((states[i] >> j) & 1) { const int b = i * K + j; out[1 + (b >> 4)] |= (uint16_t) (1u << (b & 15)); }
}
inline void unpack_row(const uint16_t* in, int K, uint16_t* states, uint16_t& scale_bits)
{
    scale_bits = in[0];
    auto sym = [&](int i) {
        i = ((i % ROW) + ROW) % ROW;
        uint32_t v = 0;
        for (int j = 0; j < K; ++j) { const int b = i * K + j; v |= ((in[1 + (b >> 4)] >> (b & 15)) & 1u) << j; }
        return v;
    };
    for (int i = 0; i < ROW; ++i)
    {
        uint32_t s = 0;
        for (int m = 0; (m * K) < 16; ++m) s |= sym(i - m) << (m * K);
        states[i] = (uint16_t) (s & 0xFFFF);
    }
}

// Device batch quantizer: rows (n x 160, BF16 bits) with global row indices r0 .. r0 + n - 1 -> packed rows
// (n x words_per_row(K) uint16). bias: 16 x 160 fp16 bits; offsets: 16 int64 (device).
struct RowQuantizer
{
    sycl::queue& q;
    int K;
    int64_t cap;
    VitWork vw;
    float *w = nullptr, *tin = nullptr, *tout = nullptr;
    uint16_t* idx = nullptr;
    uint16_t* scales = nullptr;

    RowQuantizer(sycl::queue& q_, int K_, int64_t cap_) : q(q_), K(K_), cap(cap_), vw(q_, K_, 512ull << 20, ROW)
    {
        w = sycl::malloc_device<float>(cap * ROW, q); tin = sycl::malloc_device<float>(cap * ROW, q);
        tout = sycl::malloc_device<float>(cap * ROW, q); idx = sycl::malloc_device<uint16_t>(cap * ROW, q);
        scales = sycl::malloc_device<uint16_t>(cap, q);
        if (!w || !tin || !tout || !idx || !scales) throw std::runtime_error("RowQuantizer: device allocation failed");
    }
    ~RowQuantizer() { for (void* p : {(void*) w, (void*) tin, (void*) tout, (void*) idx, (void*) scales}) sycl::free(p, q); }

    // packed: device, n x words_per_row(K)
    void run(const uint16_t* rows, int64_t r0, int64_t n, const uint16_t* bias, const int64_t* offsets, int nheads, uint16_t* packed)
    {
        float gamma, hi;
        cs_heuristic(K, gamma, hi);
        float* W = w; float* TI = tin; const int Kk = K;
        // 1: subtract the bias, per-row scale, pre-scaled ring input. One work-group of 32 per row.
        q.parallel_for(sycl::nd_range<1>(n * 32, 32), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const int64_t r = it.get_group(0); const int l = (int) it.get_local_id(0);
            const int64_t gr = r0 + r;
            int h = 0;
            for (int k = 1; k < nheads; ++k) if (gr >= offsets[k]) h = k;
            float ss = 0, am = 0;
            for (int i = l; i < ROW; i += 32)
            {
                float v = sycl::bit_cast<float>((uint32_t) rows[r * ROW + i] << 16);
                if (!sycl::isfinite(v)) v = 0;
                v -= (float) sycl::bit_cast<sycl::half>(bias[h * ROW + i]);
                W[r * ROW + i] = v;
                ss += v * v; am = sycl::fmax(am, sycl::fabs(v));
            }
            ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
            am = sycl::reduce_over_group(it.get_group(), am, sycl::maximum<float>());
            const float rms = sycl::sqrt(ss / ROW);
            const float an = am / sycl::fmax(rms, 1e-12f);
            const float cs = sycl::clamp(gamma / sycl::fmax(an, 1e-6f), CS_MIN, hi);
            float s0 = (float) sycl::half(sycl::fmax(rms / cs, 1e-8f));
            if (!(s0 > 0)) s0 = 1.0f;
            for (int i = l; i < ROW; i += 32) TI[r * ROW + i] = W[r * ROW + i] / s0;
        });
        // 2: the rings
        vw.encode(2, tin, idx, tout, n);
        // 3: least-squares scale, then pack (LSB first) with the scale in word 0.
        const float* TO = tout; const uint16_t* ID = idx; uint16_t* SC = scales;
        const int words = words_per_row(K);
        q.parallel_for(sycl::nd_range<1>(n * 32, 32), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const int64_t r = it.get_group(0); const int l = (int) it.get_local_id(0);
            float wq = 0, qq = 0;
            for (int i = l; i < ROW; i += 32) { const float qv = TO[r * ROW + i]; wq += W[r * ROW + i] * qv; qq += qv * qv; }
            wq = sycl::reduce_over_group(it.get_group(), wq, sycl::plus<float>());
            qq = sycl::reduce_over_group(it.get_group(), qq, sycl::plus<float>());
            const uint16_t sb = sycl::bit_cast<uint16_t>(sycl::half(wq / sycl::fmax(qq, 1e-12f)));
            if (l == 0) { SC[r] = sb; packed[r * words] = sb; }
            for (int wd = l; wd < words - 1; wd += 32)
            {
                uint32_t v = 0;
                for (int bb = 0; bb < 16; ++bb)
                {
                    const int b = wd * 16 + bb, i = b / Kk, j = b % Kk;
                    v |= (uint32_t) ((ID[r * ROW + i] >> j) & 1u) << bb;
                }
                packed[r * words + 1 + wd] = (uint16_t) v;
            }
        }).wait();
    }
};

// Device decode of packed rows to fp16 rows (what the calibration's PLE layer reads):
// row[i] = f16(f16(mul1(state_i)) * scale + bias[head][i])
inline sycl::event decode_rows(sycl::queue& q, const uint16_t* packed, const int64_t* row_ids, int64_t n, int K,
                               const uint16_t* bias, const int64_t* offsets, int nheads, sycl::half* out)
{
    const int words = words_per_row(K);
    return q.parallel_for(sycl::range<2>(n, ROW), [=](sycl::id<2> id) {
        const int64_t r = id[0]; const int i = (int) id[1];
        const uint16_t* pr = packed + r * words;
        auto sym = [&](int p) {
            p = (p % ROW + ROW) % ROW;
            uint32_t v = 0;
            for (int j = 0; j < K; ++j) { const int b = p * K + j; v |= ((pr[1 + (b >> 4)] >> (b & 15)) & 1u) << j; }
            return v;
        };
        uint32_t s = 0;
        for (int m = 0; m * K < 16; ++m) s |= sym(i - m) << (m * K);
        s &= 0xFFFF;
        int h = 0;
        for (int k = 1; k < nheads; ++k) if (row_ids[r] >= offsets[k]) h = k;
        const float scale = (float) sycl::bit_cast<sycl::half>(pr[0]);
        const float v = (float) dev::decode<2>(s) * scale + (float) sycl::bit_cast<sycl::half>(bias[h * ROW + i]);
        out[r * ROW + i] = sycl::half(v);
    });
}

// Per-head bias (exllamav3 compute_head_bias): mean of 16 stripes of 8192 rows per head, fp64, rounded to fp16.
// read(lo, hi, dst) fills dst with rows lo .. hi - 1 (160 floats each).
template <typename ReadRows>
inline std::vector<uint16_t> head_bias(const std::vector<int64_t>& offsets, const std::vector<int64_t>& sizes, ReadRows read)
{
    const int H = (int) offsets.size();
    const int64_t stripe = 8192, nstripes = 131072 / stripe;
    std::vector<uint16_t> bias((size_t) H * ROW);
    std::vector<float> buf(stripe * ROW);
    for (int h = 0; h < H; ++h)
    {
        std::vector<double> acc(ROW, 0.0);
        int64_t n = 0;
        for (int64_t j = 0; j < nstripes; ++j)
        {
            const int64_t lo = offsets[h] + (sizes[h] - stripe) * j / std::max<int64_t>(1, nstripes - 1);
            read(lo, lo + stripe, buf.data());
            for (int64_t r = 0; r < stripe; ++r) for (int i = 0; i < ROW; ++i) acc[i] += buf[r * ROW + i];
            n += stripe;
        }
        for (int i = 0; i < ROW; ++i) bias[h * ROW + i] = f16_from_double(acc[i] / (double) n);
    }
    return bias;
}

} // namespace exl3::ngram
