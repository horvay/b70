// ngram_hash_check <model dir> <cal dir> <out prefix>: build Qwen3.8-Flash-Next calibration rows natively, write their
// tokens (<prefix>.tokens.bin, the first and last rows of every source) and the C++ n-gram ids (<prefix>.ids.txt) for
// tests/ngram_hash_ref.py to compare. CPU only.
#include "arch/qwen4exp_ngram.hpp"
#include "io/caldata.hpp"
#include "io/safetensors.hpp"

#include <cstdio>
#include <fstream>

int main(int argc, char** argv)
{
    if (argc < 4) return 2;
    const std::string P = "model.language_model.layers.1.ple.ple_embedding.";
    std::ifstream ix(std::string(argv[1]) + "/model.safetensors.index.json");
    const auto wm = nlohmann::json::parse(ix)["weight_map"];
    auto get = [&](const std::string& n, void* dst, size_t bytes) {   // opens just the shard that holds n
        st::File f(std::string(argv[1]) + "/" + wm[n].get<std::string>());
        std::memcpy(dst, f.tensors().at(n).data, bytes);
    };
    int64_t mult[3], sizes[16], offs[16];
    get(P + "layer_multipliers", mult, 24);
    get(P + "ngram_heads_vocab_sizes", sizes, 128);
    get(P + "ngram_heads_offsets", offs, 128);
    tok::BPE tk((std::string(argv[1]) + "/tokenizer.json"));
    const int eos = 248044;
    const auto rows = cal::default_rows(tk, argv[2], 250, 2048, eos, eos, 248077);
    std::ofstream tb(std::string(argv[3]) + ".tokens.bin", std::ios::binary);
    std::FILE* f = std::fopen((std::string(argv[3]) + ".ids.txt").c_str(), "w");
    const int pick[] = {0, 1, 36, 37, 73, 74, 92, 93, 110, 111, 202, 203, 210, 211, 249};
    std::vector<int64_t> ids(2048 * 16);
    int eos_seen = 0;
    for (int r : pick)
    {
        tb.write((const char*) rows[r].data(), 2048 * 4);
        for (int v : rows[r]) eos_seen += v == eos;
        qwen4exp::ngram_ids(rows[r].data(), 2048, eos, mult, sizes, offs, ids.data());
        for (int t = 0; t < 2048; ++t)
        {
            for (int h = 0; h < 16; ++h) std::fprintf(f, h ? " %lld" : "%lld", (long long) ids[t * 16 + h]);
            std::fprintf(f, "\n");
        }
    }
    std::fclose(f);
    std::printf("%zu rows; checked %zu rows with %d EOS tokens; multipliers %lld %lld %lld\n", rows.size(), std::size(pick), eos_seen,
                (long long) mult[0], (long long) mult[1], (long long) mult[2]);
    std::printf("sizes"); for (auto s : sizes) std::printf(",%lld", (long long) s); std::printf("\noffsets"); for (auto o : offs) std::printf(",%lld", (long long) o); std::printf("\n");
}
