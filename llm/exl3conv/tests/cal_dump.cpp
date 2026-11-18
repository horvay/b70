// cal_dump <tokenizer.json> <standard_cal_data dir> [rows] [cols]: build the default calibration rows natively and
// print their make-up (CPU only).
#include "io/caldata.hpp"
#include "rt/timer.hpp"

#include <cstdio>
#include <cstdlib>

int main(int argc, char** argv)
{
    if (argc < 3) { std::fprintf(stderr, "usage: cal_dump <tokenizer.json> <cal dir> [rows] [cols]\n"); return 2; }
    const int rows = argc > 3 ? std::atoi(argv[3]) : 250, cols = argc > 4 ? std::atoi(argv[4]) : 2048;
    rt::Timer t;
    tok::BPE tk(argv[1]);
    std::printf("tokenizer: %zu tokens, loaded in %.1f s\n", tk.vocab_size(), t.s());
    rt::Timer t2;
    const auto r = cal::default_rows(tk, argv[2], rows, cols, 2, 1, 262144);
    size_t bos = 0, eos = 0, first_bos = 0;
    for (auto& row : r) { for (int x : row) { bos += x == 2; eos += x == 1; } first_bos += row[0] == 2; }
    std::printf("%zu rows x %zu tokens in %.1f s; BOS %zu (row starts %zu), EOS %zu\n", r.size(), r[0].size(), t2.s(), bos, first_bos, eos);
    return r.size() == (size_t) rows ? 0 : 1;
}
