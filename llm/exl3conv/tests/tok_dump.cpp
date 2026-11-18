// tok_dump <tokenizer.json> <text file>: print the native BPE token ids of the file, space-separated (for
// tests/tok_check.py, which compares them with Hugging Face tokenizers).
#include "io/tokenizer.hpp"

#include <cstdio>
#include <fstream>
#include <sstream>

int main(int argc, char** argv)
{
    if (argc < 3) { std::fprintf(stderr, "usage: tok_dump <tokenizer.json> <text file>\n"); return 2; }
    tok::BPE t(argv[1]);
    std::ifstream f(argv[2], std::ios::binary);
    std::stringstream ss; ss << f.rdbuf();
    for (int id : t.encode(ss.str())) std::printf("%d ", id);
    std::printf("\n");
    return 0;
}
