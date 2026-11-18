// exl3conv: the n-gram table codec on real rows of a Qwen3.8-Flash-Next table (B70).
//   test_ngram <bf16 model dir> [K = 3] [rows = 1000000]
// Checks: head bias (printed next to the reference model's values); the 160-ring encoder bit for bit against the CPU
// reference on rings the card prepared; every packed ring unpacks to tail-biting states; the device decode equals a
// host decode of the packed rows; reconstruction error; rows per second.
#include "exl3/ngram.hpp"
#include "exl3/viterbi_ref.hpp"
#include "io/safetensors.hpp"
#include "rt/dbuf.hpp"
#include "rt/rt.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using namespace exl3;

static int failures = 0;

int main(int argc, char** argv)
{
    if (argc < 2) { std::fprintf(stderr, "usage: test_ngram <bf16 model dir> [K] [rows]\n"); return 2; }
    const int K = argc > 2 ? std::atoi(argv[2]) : 3;
    const int64_t N = argc > 3 ? std::atoll(argv[3]) : 1000000;
    try
    {
        st::Model m(argv[1]);
        const std::string P = "model.language_model.layers.1.ple.ple_embedding.";
        const auto& to = m.get(P + "ngram_heads_offsets");
        const auto& ts = m.get(P + "ngram_heads_vocab_sizes");
        std::vector<int64_t> offs(16), sizes(16);
        std::memcpy(offs.data(), to.data, 128); std::memcpy(sizes.data(), ts.data, 128);
        const auto& s0 = m.get(P + "ngram_embedding.shard_0.weight");
        const int64_t shard_rows = s0.shape[0];
        auto read = [&](int64_t lo, int64_t hi, float* dst) {
            for (int64_t r = lo; r < hi; ++r)
            {
                const auto& t = m.get(P + "ngram_embedding.shard_" + std::to_string(r / shard_rows) + ".weight");
                const uint16_t* src = (const uint16_t*) t.data + (r % shard_rows) * ngram::ROW;
                for (int i = 0; i < ngram::ROW; ++i) dst[(r - lo) * ngram::ROW + i] = st::bf16_to_float(src[i]);
            }
        };
        rt::Timer tb;
        const auto bias = ngram::head_bias(offs, sizes, read);
        std::printf("head bias in %.1f s; head 0: %.6g %.6g %.6g (reference model: 0.00117 9.98e-05 0.000387)\n", tb.s(),
                    f16_to_double(bias[0]), f16_to_double(bias[1]), f16_to_double(bias[2]));

        auto q = rt::make_queue();
        rt::print_device(q.get_device());
        const int64_t B = std::min<int64_t>(N, 262144);
        ngram::RowQuantizer rq(q, K, B);
        exl3::DBuf<uint16_t> dbias(q, bias.size()), drows(q, B * ngram::ROW), dpacked(q, B * ngram::words_per_row(K));
        exl3::DBuf<int64_t> doffs(q, 16), dids(q, B);
        dbias.upload(bias.data(), bias.size()); doffs.upload(offs.data(), 16);
        const int words = ngram::words_per_row(K);
        double err = 0, ref = 0;
        int64_t bad_tb = 0, bad_dec = 0;
        rt::Timer tq;
        double t_quant = 0;
        for (int64_t r0 = 0; r0 < N; r0 += B)
        {
            const int64_t n = std::min(B, N - r0);
            const uint16_t* src = (const uint16_t*) s0.data + r0 * ngram::ROW;
            drows.upload(src, n * ngram::ROW);
            rt::Timer t1;
            rq.run(drows.get(), r0, n, dbias.get(), doffs.get(), 16, dpacked.get());
            t_quant += t1.s();
            if (r0 == 0)
            {
                // Encoder bit-exactness on the first rings, against the CPU reference on the card's own inputs.
                const int nc = 6;
                std::vector<float> tin(nc * ngram::ROW);
                std::vector<uint16_t> gidx(nc * ngram::ROW);
                q.memcpy(tin.data(), rq.tin, tin.size() * 4).wait();
                q.memcpy(gidx.data(), rq.idx, gidx.size() * 2).wait();
                const CodebookTable tab(CB_MUL1);
                std::vector<uint16_t> cidx(nc * ngram::ROW);
                std::vector<std::thread> th;
                for (int c = 0; c < nc; ++c)
                    th.emplace_back([&, c] { std::vector<float> out(ngram::ROW); ViterbiRef(K, CB_MUL1, tab, ngram::ROW).encode(&tin[c * ngram::ROW], &cidx[c * ngram::ROW], out.data()); });
                for (auto& t : th) t.join();
                const int diff = (int) std::count_if(cidx.begin(), cidx.end(), [&, i = 0](uint16_t v) mutable { return v != gidx[i++]; });
                std::printf("160-ring encoder vs CPU reference on %d rings: %s (%d states differ)\n", nc, diff ? "DIFF" : "bit-identical", diff);
                if (diff) ++failures;
            }
            std::vector<uint16_t> packed(n * words);
            dpacked.download(packed.data(), packed.size());
            // Host unpack + decode, tail-biting, error against the source row.
            std::vector<int64_t> ids(n);
            for (int64_t r = 0; r < n; ++r) ids[r] = r0 + r;
            dids.upload(ids.data(), n);
            exl3::DBuf<sycl::half> ddec(q, n * ngram::ROW);
            ngram::decode_rows(q, dpacked.get(), dids.get(), n, K, dbias.get(), doffs.get(), 16, ddec.get()).wait();
            std::vector<uint16_t> dec(n * ngram::ROW);
            q.memcpy(dec.data(), ddec.get(), dec.size() * 2).wait();
            const CodebookTable tab(CB_MUL1);
            for (int64_t r = 0; r < n; ++r)
            {
                uint16_t st[ngram::ROW], sb;
                ngram::unpack_row(&packed[r * words], K, st, sb);
                bool tb_ok = true;
                for (int i = 0; i < ngram::ROW; ++i)
                {
                    const uint32_t prev = st[(i + ngram::ROW - 1) % ngram::ROW];
                    tb_ok &= (uint32_t) (st[i] >> K) == (prev & ((1u << (16 - K)) - 1));
                }
                bad_tb += !tb_ok;
                const int h = ngram::head_of(offs.data(), 16, r0 + r);
                for (int i = 0; i < ngram::ROW; ++i)
                {
                    // fp32 multiply, fp32 add, one fp16 rounding: exllamav3's codebook[s].float() * scale.float() + bias.float()
                    const float prod = f16_to_float(tab[st[i]]) * f16_to_float(sb);
                    const double hv = f16_to_double(f16_from_float(prod + f16_to_float(bias[h * ngram::ROW + i])));
                    const double dv = f16_to_double(dec[r * ngram::ROW + i]);
                    if (hv != dv) ++bad_dec;
                    const double sv = st::bf16_to_float(src[r * ngram::ROW + i]);
                    err += (dv - sv) * (dv - sv); ref += sv * sv;
                }
            }
        }
        std::printf("%lld rows at K %d: quantized at %.0f rows/s (whole loop %.1f s); non-tail-biting rings %lld; "
                    "device vs host decode mismatches %lld; relative RMS error %.4f\n", (long long) N, K, N / t_quant, tq.s(),
                    (long long) bad_tb, (long long) bad_dec, std::sqrt(err / ref));
        std::printf("  -> the full table (320M rows) at this rate: %.0f min\n", 320001536.0 / (N / t_quant) / 60);
        if (bad_tb || bad_dec) ++failures;
    }
    catch (const std::exception& e) { std::fprintf(stderr, "error: %s\n", e.what()); return 2; }
    std::printf(failures ? "\n%d FAILED\n" : "\nall passed\n", failures);
    return failures ? 1 : 0;
}
