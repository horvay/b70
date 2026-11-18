// exl3conv: calibration token rows, after exllamav3 1.5.3 `get_default_calibration` (conversion/calibration_data.py).
//
// rows x cols tokens from the bundled texts in exllamav3's conversion/standard_cal_data/ (found in this repo's
// exllamav3 submodule; --cal-dir elsewhere), weighted as exllamav3 weights them (sum 135; 250 rows give 37 / 37 / 18
// / 18 / 92 / 9 rows plus 39 random ones):
//   c4.utf8 20 (shuffled lines), code.utf8 20 (raw slices), multilingual.utf8 10 (shuffled lines),
//   technical.utf8 10 (raw slices), wiki.utf8 50 (articles), tiny.utf8 5 (stories), random tokens (the rest)
// Articles are packed into rows; a row is cut at cols tokens and the rest of the last article is dropped. Articles
// in a source's even rows are wrapped in BOS ... EOS, odd rows get no special tokens. Raw slices have none.
// Differences from exllamav3, on purpose and harmless for quality: the line shuffle and the random rows use this
// file's seeded generator instead of Python's random.shuffle and torch.randint, so the rows aren't bit-identical.
#pragma once

#include "io/tokenizer.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cal {

namespace fs = std::filesystem;
using Row = std::vector<int32_t>;

inline uint64_t splitmix(uint64_t& s)
{
    uint64_t z = (s += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

inline std::string read_file(const fs::path& p)
{
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read calibration file " + p.string());
    std::stringstream ss; ss << f.rdbuf(); return ss.str();
}

inline std::vector<std::string> split(const std::string& s, const std::string& sep)
{
    std::vector<std::string> out;
    size_t a = 0;
    for (size_t b = s.find(sep); b != std::string::npos; b = s.find(sep, a)) { out.push_back(s.substr(a, b - a)); a = b + sep.size(); }
    out.push_back(s.substr(a));
    return out;
}

inline std::string strip(const std::string& s)
{
    const char* ws = " \t\n\r\f\v";
    const size_t a = s.find_first_not_of(ws);
    if (a == std::string::npos) return "";
    return s.substr(a, s.find_last_not_of(ws) - a + 1);
}

inline bool is_space(const std::string& s)   // Python str.isspace(): non-empty and all whitespace
{
    if (s.empty()) return false;
    for (unsigned char c : s) if (!std::isspace(c)) return false;
    return true;
}

struct Builder
{
    const tok::BPE& tk;
    int bos, eos, cols;

    std::vector<Row> split_art(const std::vector<std::string>& arts, int rows) const
    {
        std::vector<Row> out;
        Row row;
        for (size_t idx = 0; (int) out.size() < rows; ++idx)
        {
            if (idx >= arts.size()) throw std::runtime_error("calibration source ran out of text");
            const bool sp = out.size() % 2 == 0;
            if (sp) row.push_back(bos);
            for (int t : tk.encode(arts[idx])) row.push_back(t);
            if (sp) row.push_back(eos);
            if ((int) row.size() > cols) row.resize(cols);
            if ((int) row.size() == cols) { out.push_back(row); row.clear(); }
        }
        return out;
    }
    std::vector<Row> wiki(const std::string& text, int rows) const
    {
        std::vector<std::string> arts;
        for (auto& a : split(text, "</doc>\n"))
        {
            const size_t nl = a.find('\n');                   // Python a[a.find("\n") + 1:]: no newline keeps all of a
            const std::string body = nl == std::string::npos ? a : a.substr(nl + 1);
            if (body.size() > 50) arts.push_back(body);
        }
        return split_art(arts, rows);
    }
    std::vector<Row> tiny(const std::string& text, int rows) const
    {
        std::vector<std::string> arts;
        for (auto& a : split(text, "<|endoftext|>")) arts.push_back(strip(a));
        return split_art(arts, rows);
    }
    std::vector<Row> lines(const std::string& text, int rows, uint64_t seed) const
    {
        std::vector<std::string> arts;
        for (auto& a : split(text, "\n")) if (!is_space(a)) arts.push_back(a);
        for (size_t i = arts.size(); i > 1; --i) std::swap(arts[i - 1], arts[splitmix(seed) % i]);   // Fisher-Yates
        return split_art(arts, rows);
    }
    std::vector<Row> raw(const std::string& text, int rows) const
    {
        const auto all = tk.encode(text);
        std::vector<Row> out;
        for (int i = 0; i < rows; ++i)
        {
            const size_t a = (size_t) i * cols;
            if (a + cols > all.size()) throw std::runtime_error("raw calibration source too short");
            out.emplace_back(all.begin() + a, all.begin() + a + cols);
        }
        return out;
    }
    std::vector<Row> random(int rows, int vocab, uint64_t seed) const
    {
        std::vector<Row> out(rows, Row(cols));
        for (auto& r : out) for (auto& t : r) t = (int32_t) (splitmix(seed) % (uint64_t) vocab);
        return out;
    }
};

// The default calibration set: rows x cols token ids.
inline std::vector<Row> default_rows(const tok::BPE& tk, const fs::path& dir, int rows, int cols, int bos, int eos, int vocab)
{
    Builder b{tk, bos, eos, cols};
    struct Src { const char* file; int weight; int kind; };      // kind: 0 lines, 1 raw, 2 wiki, 3 tiny, 4 random
    const Src srcs[] = {{"c4.utf8", 20, 0}, {"code.utf8", 20, 1}, {"multilingual.utf8", 10, 0}, {"technical.utf8", 10, 1},
                        {"wiki.utf8", 50, 2}, {"tiny.utf8", 5, 3}, {nullptr, 20, 4}};
    const int sum = 135;
    std::vector<Row> all;
    for (const auto& s : srcs)
    {
        int target = std::max(1, (int) ((double) s.weight / sum * rows));
        std::vector<Row> r;
        switch (s.kind)
        {
            case 0: r = b.lines(read_file(dir / s.file), target, 0); break;
            case 1: r = b.raw(read_file(dir / s.file), target); break;
            case 2: r = b.wiki(read_file(dir / s.file), target); break;
            case 3: r = b.tiny(read_file(dir / s.file), target); break;
            default: target = std::max(1, rows - (int) all.size()); r = b.random(target, vocab, 0); break;
        }
        for (auto& x : r) all.push_back(std::move(x));
    }
    return all;
}

} // namespace cal
