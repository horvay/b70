// exl3conv: a native Hugging Face tokenizer.json BPE encoder, enough for calibration text (no Python).
//
// Supported: model.type "BPE", in two families. SentencePiece-style (Gemma): byte_fallback, a normalizer of "Replace"
// (string pattern) / "Prepend" / "Sequence", and a pre-tokenizer that is absent or a "Split" on a string.
// Byte-level (Qwen): an "NFC" normalizer, a "Sequence" of a "Split" on Qwen's regex (Isolated) and "ByteLevel"
// (use_regex false); each piece's UTF-8 bytes are mapped to GPT-2's printable byte characters before the merges.
// Ripple is the same family without a normalizer: its Split keeps the regex's matches (Removed with invert, which is
// Isolated here since every character matches) and takes digits in runs of up to 3.
// The regex is matched by hand (qwen_split below); another regex throws. Anything unsupported throws, so a new
// tokenizer family fails loudly instead of tokenizing wrongly.
//
// Encoding a pre-token: start from its UTF-8 characters (a character missing from the vocabulary becomes its
// <0xXX> byte tokens), then repeatedly merge the adjacent pair with the lowest merge rank, leftmost first, as HF's
// BPE does. Added/special tokens are not matched inside the text: exllamav3 encodes calibration text that way
// (special-token strings in the corpus stay plain text), and BOS / EOS are added as ids by the caller.
#pragma once

#include "io/unicode.hpp"
#include "json.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace tok {

using json = nlohmann::json;

class BPE
{
public:
    explicit BPE(const std::string& path)
    {
        std::ifstream f(path);
        if (!f) throw std::runtime_error("cannot read " + path);
        const json j = json::parse(f);
        const auto& m = j.at("model");
        if (m.value("type", "") != "BPE") throw std::runtime_error("tokenizer: only BPE models are supported");
        byte_fallback_ = m.value("byte_fallback", false);
        if (!m["continuing_subword_prefix"].is_null() && m["continuing_subword_prefix"] != "")
            throw std::runtime_error("tokenizer: continuing_subword_prefix is not supported");
        if (!m["end_of_word_suffix"].is_null() && m["end_of_word_suffix"] != "")
            throw std::runtime_error("tokenizer: end_of_word_suffix is not supported");
        for (auto& [s, id] : m.at("vocab").items()) vocab_[s] = id.get<int>();
        if (m.contains("unk_token") && m["unk_token"].is_string())
        {
            auto it = vocab_.find(m["unk_token"].get<std::string>());
            if (it != vocab_.end()) unk_ = it->second;
        }
        int rank = 0;
        for (const auto& mg : m.at("merges"))
        {
            std::string a, b;
            if (mg.is_array()) { a = mg[0]; b = mg[1]; }
            else { const std::string s = mg; const auto sp = s.find(' '); a = s.substr(0, sp); b = s.substr(sp + 1); }
            auto ia = vocab_.find(a), ib = vocab_.find(b), ic = vocab_.find(a + b);
            if (ia != vocab_.end() && ib != vocab_.end() && ic != vocab_.end())
                merges_[key(ia->second, ib->second)] = {rank, ic->second};
            ++rank;
        }
        // Non-special added tokens are cut out of the raw text first and emitted as single ids (HF's AddedVocabulary;
        // exllamav3 encodes calibration text with special tokens as plain text, which leaves these active).
        for (const auto& t : j.at("added_tokens"))
            if (!t.value("special", true))
            {
                if (t.value("lstrip", false) || t.value("rstrip", false) || t.value("single_word", false))
                    throw std::runtime_error("tokenizer: added token options lstrip/rstrip/single_word are not supported");
                added_.push_back({t.at("content").get<std::string>(), t.at("id").get<int>()});
            }
        parse_normalizer(j["normalizer"]);
        parse_pretokenizer(j["pre_tokenizer"]);
        // GPT-2 bytes_to_unicode: printable bytes map to themselves, the others to U+0100 + n in order.
        for (int b = 0, k = 0; b < 256; ++b)
        {
            const bool keep = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
            std::string u;
            uni::append_utf8(u, keep ? (uint32_t) b : 256u + (uint32_t) k++);
            byte_char_[b] = u;
        }
        for (int b = 0; b < 256; ++b)
        {
            char name[8];
            std::snprintf(name, sizeof name, "<0x%02X>", b);
            auto it = vocab_.find(name);
            byte_ids_[b] = it == vocab_.end() ? -1 : it->second;
        }
    }

    size_t vocab_size() const { return vocab_.size(); }

    // The id of a vocabulary entry by its exact string (special tokens such as "<|turn>"); throws if absent.
    int id(const std::string& piece) const
    {
        auto it = vocab_.find(piece);
        if (it == vocab_.end()) throw std::runtime_error("tokenizer: no token " + piece);
        return it->second;
    }

    std::vector<int> encode(const std::string& text) const
    {
        std::vector<int> out;
        if (added_.empty()) { encode_plain(text, out); return out; }
        // Leftmost match, longest on ties at the same position.
        size_t pos = 0, seg = 0;
        while (pos < text.size())
        {
            int best = -1; size_t best_len = 0;
            for (size_t a = 0; a < added_.size(); ++a)
                if (added_[a].first.size() > best_len && text.compare(pos, added_[a].first.size(), added_[a].first) == 0)
                { best = (int) a; best_len = added_[a].first.size(); }
            if (best < 0) { ++pos; continue; }
            if (pos > seg) encode_plain(text.substr(seg, pos - seg), out);
            out.push_back(added_[best].second);
            pos += best_len; seg = pos;
        }
        if (seg < text.size()) encode_plain(text.substr(seg), out);
        return out;
    }

private:
    void encode_plain(const std::string& text, std::vector<int>& out) const
    {
        if (bytelevel_)
        {
            auto cps = uni::decode_utf8(text);
            if (nfc_) cps = uni::nfc(cps);
            for (const auto& [a, b] : qwen_split(cps, num_run_))
            {
                std::string mapped;
                for (unsigned char c : uni::encode_utf8(cps, a, b)) mapped += byte_char_[c];
                bpe(mapped, out);
            }
            return;
        }
        std::string s = normalize(text);
        if (nfc_) s = uni::encode_utf8(uni::nfc(uni::decode_utf8(s)));
        for (const auto& piece : pretokenize(s)) bpe(piece, out);
    }

public:

    // Qwen's pre-tokenizer regex, the only Split regex accepted:
    static constexpr const char* QWEN_REGEX =
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

    // Ripple's: the same with \p{N}{1,3}.
    static constexpr const char* RIPPLE_REGEX =
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

    // Its matches over code points, as [begin, end) pieces. Alternatives are tried in order at each position
    // (leftmost-first), with the backtracking results worked out per alternative:
    //   1 '(s|t|re|ve|m|ll|d) case-insensitive   2 [^\r\n L N]? L+   3 N (up to num_run of them)   4 ' '? [^\s L N]+ [\r\n]*
    //   5 \s*[\r\n]+: the whitespace run up to and including its last CR/LF   6 \s+(?!\S): the run, minus its last
    //   character when a non-space follows (fails for a single space before a non-space)   7 \s+: the run
    static std::vector<std::pair<size_t, size_t>> qwen_split(const std::vector<uint32_t>& s, int num_run = 1)
    {
        std::vector<std::pair<size_t, size_t>> out;
        const size_t n = s.size();
        auto L = [&](size_t i) { return i < n && uni::is_letter(s[i]); };
        auto N = [&](size_t i) { return i < n && uni::is_number(s[i]); };
        auto S = [&](size_t i) { return i < n && uni::is_space(s[i]); };
        auto CRLF = [&](size_t i) { return i < n && (s[i] == '\r' || s[i] == '\n'); };
        auto P = [&](size_t i) { return i < n && !uni::is_space(s[i]) && !uni::is_letter(s[i]) && !uni::is_number(s[i]); };
        auto lower = [&](size_t i) { return i < n && s[i] < 128 ? (uint32_t) std::tolower((int) s[i]) : (i < n ? s[i] : 0u); };
        size_t i = 0;
        while (i < n)
        {
            size_t e = 0;
            // 1: contractions
            if (s[i] == '\'')
            {
                const uint32_t a = lower(i + 1), b = lower(i + 2);
                if (a == 's' || a == 't' || a == 'm' || a == 'd') e = i + 2;
                else if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) e = i + 3;
            }
            // 2: an optional non-letter/number/CR/LF, then letters
            if (!e)
            {
                if (!CRLF(i) && !L(i) && !N(i) && L(i + 1)) { e = i + 1; while (L(e)) ++e; }
                else if (L(i)) { e = i; while (L(e)) ++e; }
            }
            // 3: number characters
            if (!e && N(i)) { e = i + 1; while (e < i + num_run && N(e)) ++e; }
            // 4: an optional space, punctuation, then CR/LF
            if (!e)
            {
                size_t j = i;
                if (s[j] == ' ' && P(j + 1)) ++j;
                if (P(j)) { while (P(j)) ++j; while (CRLF(j)) ++j; e = j; }
            }
            if (!e && S(i))
            {
                size_t w = i; while (S(w)) ++w;            // whitespace run [i, w)
                size_t last = SIZE_MAX;
                for (size_t k = i; k < w; ++k) if (CRLF(k)) last = k;
                if (last != SIZE_MAX) e = last + 1;                        // 5
                else if (w == n) e = w;                                    // 6, at the end
                else if (w - i >= 2) e = w - 1;                            // 6, before a non-space
                else e = w;                                                // 7
            }
            if (!e) e = i + 1;                                           // (unreachable: every character matches)
            out.push_back({i, e});
            i = e;
        }
        return out;
    }

private:
    struct Replace { std::string from, to; bool prepend = false; };
    static uint64_t key(int a, int b) { return ((uint64_t) (uint32_t) a << 32) | (uint32_t) b; }

    void parse_normalizer(const json& n)
    {
        if (n.is_null()) return;
        const std::string t = n.value("type", "");
        if (t == "Sequence") { for (const auto& x : n.at("normalizers")) parse_normalizer(x); return; }
        if (t == "Replace" && n.at("pattern").contains("String"))
        { norm_.push_back({n["pattern"]["String"].get<std::string>(), n.at("content").get<std::string>()}); return; }
        if (t == "Prepend") { norm_.push_back({"", n.at("prepend").get<std::string>(), true}); return; }
        if (t == "NFC") { nfc_ = true; return; }
        throw std::runtime_error("tokenizer: unsupported normalizer " + t);
    }
    void parse_pretokenizer(const json& p)
    {
        if (p.is_null()) return;
        const std::string t = p.value("type", "");
        if (t == "Sequence")
        {
            const auto& ps = p.at("pretokenizers");
            if (ps.size() == 2 && ps[0].value("type", "") == "Split" && ps[0].at("pattern").contains("Regex") &&
                ps[1].value("type", "") == "ByteLevel" && !ps[1].value("use_regex", true) && !ps[1].value("add_prefix_space", false))
            {
                const std::string re = ps[0]["pattern"]["Regex"].get<std::string>(), beh = ps[0].value("behavior", "");
                const bool inv = ps[0].value("invert", false);
                if (re == QWEN_REGEX && beh == "Isolated" && !inv) { bytelevel_ = true; return; }
                if (re == RIPPLE_REGEX && beh == "Removed" && inv) { bytelevel_ = true; num_run_ = 3; return; }
            }
            throw std::runtime_error("tokenizer: unsupported pre_tokenizer sequence");
        }
        if (t == "Split" && p.at("pattern").contains("String") && !p.value("invert", false))
        {
            split_ = p["pattern"]["String"].get<std::string>();
            behavior_ = p.value("behavior", "");
            if (behavior_ != "MergedWithPrevious" && behavior_ != "Isolated" && behavior_ != "Removed")
                throw std::runtime_error("tokenizer: unsupported Split behavior " + behavior_);
            return;
        }
        throw std::runtime_error("tokenizer: unsupported pre_tokenizer " + t);
    }

    std::string normalize(std::string s) const
    {
        for (const auto& r : norm_)
        {
            if (r.prepend) { s = r.to + s; continue; }
            std::string o;
            o.reserve(s.size() + s.size() / 4);
            for (size_t i = 0; i < s.size();)
                if (!r.from.empty() && s.compare(i, r.from.size(), r.from) == 0) { o += r.to; i += r.from.size(); }
                else o += s[i++];
            s.swap(o);
        }
        return s;
    }

    std::vector<std::string> pretokenize(const std::string& s) const
    {
        if (split_.empty()) return {s};
        std::vector<std::string> out;
        size_t start = 0;
        for (size_t i = s.find(split_); i != std::string::npos; i = s.find(split_, i + split_.size()))
        {
            if (behavior_ == "MergedWithPrevious") { out.push_back(s.substr(start, i + split_.size() - start)); start = i + split_.size(); }
            else
            {
                if (i > start) out.push_back(s.substr(start, i - start));
                if (behavior_ == "Isolated") out.push_back(split_);
                start = i + split_.size();
            }
        }
        if (start < s.size()) out.push_back(s.substr(start));
        out.erase(std::remove_if(out.begin(), out.end(), [](const std::string& x) { return x.empty(); }), out.end());
        return out;
    }

    static size_t utf8_len(unsigned char c) { return c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1; }

    void bpe(const std::string& w, std::vector<int>& out) const
    {
        struct Sym { int id, prev, next; bool alive; };
        std::vector<Sym> sy;
        for (size_t i = 0; i < w.size();)
        {
            const size_t n = std::min(utf8_len((unsigned char) w[i]), w.size() - i);
            auto it = vocab_.find(w.substr(i, n));
            if (it != vocab_.end()) sy.push_back({it->second, 0, 0, true});
            else if (byte_fallback_)
            {
                bool ok = true;
                for (size_t b = 0; b < n; ++b) ok &= byte_ids_[(unsigned char) w[i + b]] >= 0;
                if (ok) for (size_t b = 0; b < n; ++b) sy.push_back({byte_ids_[(unsigned char) w[i + b]], 0, 0, true});
                else if (unk_ >= 0) sy.push_back({unk_, 0, 0, true});
            }
            else if (unk_ >= 0) sy.push_back({unk_, 0, 0, true});
            i += n;
        }
        const int n = (int) sy.size();
        for (int i = 0; i < n; ++i) { sy[i].prev = i - 1; sy[i].next = i + 1 < n ? i + 1 : -1; }
        // (rank, left position, left id, right id); smallest rank first, then leftmost.
        struct Cand { int rank, pos, a, b; bool operator>(const Cand& o) const { return rank != o.rank ? rank > o.rank : pos > o.pos; } };
        std::priority_queue<Cand, std::vector<Cand>, std::greater<Cand>> pq;
        auto push = [&](int l) {
            if (l < 0 || sy[l].next < 0) return;
            const int r = sy[l].next;
            auto it = merges_.find(key(sy[l].id, sy[r].id));
            if (it != merges_.end()) pq.push({it->second.first, l, sy[l].id, sy[r].id});
        };
        for (int i = 0; i + 1 < n; ++i) push(i);
        while (!pq.empty())
        {
            const Cand c = pq.top(); pq.pop();
            if (!sy[c.pos].alive || sy[c.pos].id != c.a) continue;
            const int r = sy[c.pos].next;
            if (r < 0 || sy[r].id != c.b) continue;
            sy[c.pos].id = merges_.at(key(c.a, c.b)).second;
            sy[r].alive = false;
            sy[c.pos].next = sy[r].next;
            if (sy[r].next >= 0) sy[sy[r].next].prev = c.pos;
            push(sy[c.pos].prev);
            push(c.pos);
        }
        for (int i = n ? 0 : -1; i >= 0; i = sy[i].next) out.push_back(sy[i].id);
    }

    std::unordered_map<std::string, int> vocab_;
    std::unordered_map<uint64_t, std::pair<int, int>> merges_;
    std::vector<Replace> norm_;
    std::string split_, behavior_;
    bool byte_fallback_ = false;
    bool bytelevel_ = false, nfc_ = false;
    int num_run_ = 1;                                  // digits per piece in qwen_split
    std::vector<std::pair<std::string, int>> added_;   // non-special added tokens
    std::string byte_char_[256];
    int unk_ = -1;
    int byte_ids_[256];
};

} // namespace tok
