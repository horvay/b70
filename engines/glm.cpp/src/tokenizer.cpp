#include "tokenizer.h"
#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>
#include <utf8proc.h>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <queue>
#include <sstream>
#include <unordered_map>
#include "json.h"
#include "rt.h"

namespace glm {

// ---------------------------------------------------------------------------------------------------- UTF-8

// Length of the valid UTF-8 sequence at s[i] (Unicode table 3-7), or -(length of the maximal invalid prefix).
static int utf8_seq(const unsigned char* s, size_t n, size_t i) {
    unsigned char c = s[i];
    if (c < 0x80) return 1;
    int w;
    unsigned char lo = 0x80, hi = 0xBF;
    if (c >= 0xC2 && c <= 0xDF) w = 2;
    else if (c >= 0xE0 && c <= 0xEF) { w = 3; if (c == 0xE0) lo = 0xA0; if (c == 0xED) hi = 0x9F; }
    else if (c >= 0xF0 && c <= 0xF4) { w = 4; if (c == 0xF0) lo = 0x90; if (c == 0xF4) hi = 0x8F; }
    else return -1;
    for (int k = 1; k < w; k++) {
        if (i + k >= n) return -k;
        unsigned char b = s[i + k];
        if (k == 1 ? (b < lo || b > hi) : (b < 0x80 || b > 0xBF)) return -k;
    }
    return w;
}

// The code point of the valid w-byte UTF-8 sequence at p
static uint32_t utf8_cp(const unsigned char* p, int w) {
    return w == 1 ? p[0] : w == 2 ? (p[0] & 0x1F) << 6 | (p[1] & 0x3F)
         : w == 3 ? (p[0] & 0x0F) << 12 | (p[1] & 0x3F) << 6 | (p[2] & 0x3F)
                  : (p[0] & 0x07) << 18 | (p[1] & 0x3F) << 12 | (p[2] & 0x3F) << 6 | (p[3] & 0x3F);
}

static bool utf8_valid(const std::string& s) {
    auto p = reinterpret_cast<const unsigned char*>(s.data());
    for (size_t i = 0; i < s.size();) {
        int w = utf8_seq(p, s.size(), i);
        if (w < 0) return false;
        i += w;
    }
    return true;
}

std::string utf8_lossy(const std::string& b) {
    if (utf8_valid(b)) return b;
    auto p = reinterpret_cast<const unsigned char*>(b.data());
    std::string o;
    o.reserve(b.size() + 8);
    for (size_t i = 0; i < b.size();) {
        int w = utf8_seq(p, b.size(), i);
        if (w > 0) { o.append(b, i, w); i += w; }
        else { o += "\xEF\xBF\xBD"; i += -w; }
    }
    return o;
}

size_t utf8_length(const std::string& s) {
    size_t n = 0;
    for (unsigned char c : s) n += (c & 0xC0) != 0x80;
    return n;
}

static std::string slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    GLM_CHECK(f, "cannot open %s", path.c_str());
    std::stringstream s;
    s << f.rdbuf();
    return s.str();
}

// ---------------------------------------------------------------------------------------------------- impl

struct Tokenizer::Impl {
    struct Added { int id; std::string content; bool special; };

    std::vector<std::string> bytes;                  // id -> raw bytes (byte-level decoded); added: content
    std::vector<std::string> text;                   // id -> decode({id}) with special tokens
    std::unordered_map<std::string, int> vocab;      // raw bytes -> id, model vocabulary
    std::unordered_map<std::string, int> vocab_bl;   // byte-level form -> id (id_of)
    std::unordered_map<uint64_t, std::pair<int, int>> merges;   // (left id, right id) -> (rank, merged id)
    int byte_id[256];                                // id of each single byte, -1 if the vocabulary lacks it
    bool ignore_merges = false;
    bool nfc = false;                                // normalizer NFC (Qwen): on the stretches between added tokens
    std::vector<Added> added;                        // by id
    std::vector<int> added_by_byte[256];             // indices into added, by first byte, longest first
    std::vector<char> special;                       // id -> special added token
    int n_vocab = 0;
    pcre2_code* re = nullptr;
    bool keep_matches = false;                       // Split(behavior Removed, invert) (Ripple): gaps are dropped
    // byte fallback (SentencePiece-style BPE, Gemma); vocab then maps the token strings themselves
    bool sp = false;
    std::vector<std::pair<std::string, std::string>> sp_replace;   // normalizer Replace steps, in order
    std::string sp_split;                            // pre-tokenizer Split on this string, MergedWithPrevious ("": none)
    std::unordered_map<uint32_t, int> sp_char;       // code point -> its single-character token
    int sp_byte[256];                                // id of <0xNN>
    std::vector<int16_t> byte_val;                   // id -> the byte its <0xNN> text stands for in the decoder, or -1

    ~Impl() { if (re) pcre2_code_free(re); }

    void load_sp(const Json& j);

    // a stretch without added tokens: normalized (NFC) when the tokenizer says so, then encode_plain
    void encode_stretch(const char* s, size_t n, pcre2_match_data* md, std::vector<int>& out) const {
        if (sp) return encode_sp(s, n, out);
        bool ascii = true;
        for (size_t k = 0; k < n && ascii; k++) ascii = (unsigned char)s[k] < 0x80;
        if (!nfc || ascii) return encode_plain(s, n, md, out);
        utf8proc_uint8_t* r = utf8proc_NFC(reinterpret_cast<const utf8proc_uint8_t*>(std::string(s, n).c_str()));
        GLM_CHECK(r, "tokenizer: NFC failed");
        std::string t(reinterpret_cast<char*>(r));
        free(r);
        encode_plain(t.data(), t.size(), md, out);
    }

    // HF Split on one stretch. behavior Isolated: matches and the gaps between them, in order. behavior Removed with
    // invert (keep_matches): the matches alone
    template <class F> void split(const char* s, size_t n, pcre2_match_data* md, F&& piece) const {
        size_t off = 0;
        uint32_t opts = PCRE2_NO_UTF_CHECK;
        while (off < n) {
            int rc = pcre2_match(re, (PCRE2_SPTR)s, n, off, opts, md, nullptr);
            if (rc == PCRE2_ERROR_JIT_STACKLIMIT) {
                opts |= PCRE2_NO_JIT;
                continue;
            }
            if (rc == PCRE2_ERROR_NOMATCH) break;
            GLM_CHECK(rc > 0, "pre-tokenizer regex failed (%d)", rc);
            PCRE2_SIZE* ov = pcre2_get_ovector_pointer(md);
            size_t a = ov[0], b = ov[1];
            if (a > off && !keep_matches) piece(s + off, a - off);
            if (b == a) {   // empty match (not possible with these patterns): step one character, HF's next gap
                b = a + 1;
                while (b < n && ((unsigned char)s[b] & 0xC0) == 0x80) b++;
                if (!keep_matches) piece(s + a, b - a);
            } else {
                piece(s + a, b - a);
            }
            off = b;
        }
        if (off < n && !keep_matches) piece(s + off, n - off);
    }

    // HF BPE::tokenize of one pre-token (raw bytes)
    void bpe(const char* s, size_t n, std::vector<int>& out) const {
        if (ignore_merges) {
            auto it = vocab.find(std::string(s, n));
            if (it != vocab.end()) { out.push_back(it->second); return; }
        }
        std::vector<Sym> sy;
        sy.reserve(n);
        for (size_t k = 0; k < n; k++) {
            int id = byte_id[(unsigned char)s[k]];
            if (id < 0) continue;          // HF drops characters missing from the vocabulary (no unk token)
            int i = (int)sy.size();
            sy.push_back({id, i - 1, i + 1, true});
        }
        merge_all(sy, out);
    }

    // HF Word::merge_all: the pair of lowest merge rank first (leftmost on a tie), until none is left
    struct Sym { int c, prev, next; bool live; };
    void merge_all(std::vector<Sym>& sy, std::vector<int>& out) const {
        if (sy.empty()) return;
        sy.back().next = -1;
        struct M { int rank, pos, id; };
        auto worse = [](const M& a, const M& b) { return a.rank != b.rank ? a.rank > b.rank : a.pos > b.pos; };
        std::priority_queue<M, std::vector<M>, decltype(worse)> q(worse);
        auto lookup = [&](int l, int r) -> const std::pair<int, int>* {
            auto it = merges.find((uint64_t)(uint32_t)l << 32 | (uint32_t)r);
            return it == merges.end() ? nullptr : &it->second;
        };
        for (int i = 0; i + 1 < (int)sy.size(); i++)
            if (auto m = lookup(sy[i].c, sy[i + 1].c)) q.push({m->first, i, m->second});
        while (!q.empty()) {
            M top = q.top();
            q.pop();
            Sym& cur = sy[top.pos];
            if (!cur.live || cur.next < 0) continue;
            int nx = cur.next;
            auto m = lookup(cur.c, sy[nx].c);
            if (!m || m->second != top.id) continue;   // stale entry
            cur.c = top.id;
            cur.next = sy[nx].next;
            sy[nx].live = false;
            if (cur.next >= 0) sy[cur.next].prev = top.pos;
            if (cur.prev >= 0)
                if (auto p = lookup(sy[cur.prev].c, cur.c)) q.push({p->first, cur.prev, p->second});
            if (cur.next >= 0)
                if (auto p = lookup(cur.c, sy[cur.next].c)) q.push({p->first, top.pos, p->second});
        }
        for (auto& x : sy)
            if (x.live) out.push_back(x.c);
    }

    // a stretch without added tokens: split, then BPE per piece
    void encode_plain(const char* s, size_t n, pcre2_match_data* md, std::vector<int>& out) const {
        split(s, n, md, [&](const char* p, size_t m) { bpe(p, m, out); });
    }

    // ---- byte fallback

    std::string sp_normalize(const char* s, size_t n) const {
        std::string t(s, n);
        for (auto& [from, to] : sp_replace) {
            std::string o;
            o.reserve(t.size() + t.size() / 2);
            size_t p = 0;
            for (size_t q; (q = t.find(from, p)) != std::string::npos; p = q + from.size()) o.append(t, p, q - p) += to;
            o.append(t, p, std::string::npos);
            t.swap(o);
        }
        return t;
    }

    // HF Split(String, MergedWithPrevious): each piece ends with the delimiter. (Gemma splits on ' ' after a
    // normalizer that leaves none, so a whole stretch is one piece.)
    template <class F> void sp_pieces(const std::string& t, F&& piece) const {
        if (sp_split.empty()) { if (!t.empty()) piece(t.data(), t.size()); return; }
        size_t p = 0;
        for (size_t q; (q = t.find(sp_split, p)) != std::string::npos; p = q + sp_split.size())
            piece(t.data() + p, q + sp_split.size() - p);
        if (p < t.size()) piece(t.data() + p, t.size() - p);
    }

    // HF BPE::merge_word with byte_fallback: one symbol per character, a character outside the vocabulary as its
    // UTF-8 bytes' <0xNN> tokens (load() checked that all 256 exist, so unk never occurs), then the merges
    void bpe_sp(const char* s, size_t n, std::vector<int>& out) const {
        if (ignore_merges) {
            auto it = vocab.find(std::string(s, n));
            if (it != vocab.end()) { out.push_back(it->second); return; }
        }
        auto p = reinterpret_cast<const unsigned char*>(s);
        std::vector<Sym> sy;
        sy.reserve(n);
        auto add = [&](int id) { int i = (int)sy.size(); sy.push_back({id, i - 1, i + 1, true}); };
        for (size_t i = 0; i < n;) {
            int w = utf8_seq(p, n, i);     // the text is valid UTF-8 (encode() repaired it)
            auto it = sp_char.find(utf8_cp(p + i, w));
            if (it != sp_char.end()) add(it->second);
            else for (int k = 0; k < w; k++) add(sp_byte[p[i + k]]);
            i += w;
        }
        merge_all(sy, out);
    }

    void encode_sp(const char* s, size_t n, std::vector<int>& out) const {
        std::string t = sp_normalize(s, n);
        sp_pieces(t, [&](const char* q, size_t m) { bpe_sp(q, m, out); });
    }

    // HF decoder Sequence[Replace, ByteFallback, Fuse] over the token texts (bytes[] has the Replace applied)
    std::string decode_sp(const std::vector<int>& ids, bool skip_special) const {
        std::string o, run;
        auto flush = [&] {
            if (run.empty()) return;
            if (utf8_valid(run)) o += run;
            else for (size_t k = 0; k < run.size(); k++) o += "\xEF\xBF\xBD";   // ByteFallback: one U+FFFD per byte
            run.clear();
        };
        for (int id : ids) {
            if (id < 0 || id >= n_vocab) continue;
            if (skip_special && special[id]) continue;
            if (byte_val[id] >= 0) { run += (char)byte_val[id]; continue; }
            flush();
            o += bytes[id];
        }
        flush();
        return o;
    }
};

// GPT-2 bytes_to_unicode: printable bytes map to themselves, the rest to U+0100 onwards
static void byte_level_table(uint32_t cp_of[256]) {
    int n = 0;
    for (int b = 0; b < 256; b++) {
        bool keep = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
        cp_of[b] = keep ? b : 256 + n++;
    }
}

// Onig's \s (Unicode White_Space), spelled out so the match does not depend on how a PCRE2 version defines UCP \s
static const char* kSpace = "\\t-\\r \\x{85}\\x{A0}\\x{1680}\\x{2000}-\\x{200A}\\x{2028}\\x{2029}\\x{202F}\\x{205F}\\x{3000}";

static std::string onig_to_pcre2(const std::string& pat) {
    std::string o;
    bool in_class = false;
    for (size_t i = 0; i < pat.size(); i++) {
        char c = pat[i];
        if (c == '\\' && i + 1 < pat.size()) {
            char e = pat[i + 1];
            i++;
            if (e == 's') { o += in_class ? std::string(kSpace) : "[" + std::string(kSpace) + "]"; continue; }
            if (e == 'S') {
                GLM_CHECK(!in_class, "tokenizer regex: \\S inside a class is not supported");
                o += "[^" + std::string(kSpace) + "]";
                continue;
            }
            o += c;
            o += e;
            continue;
        }
        if (!in_class && c == '[') {
            in_class = true;
            o += c;
            if (i + 1 < pat.size() && pat[i + 1] == '^') o += pat[++i];
            if (i + 1 < pat.size() && pat[i + 1] == ']') o += pat[++i];   // a leading ] is literal
            continue;
        }
        if (in_class && c == ']') in_class = false;
        o += c;
    }
    return o;
}

// HF's ByteFallback decoder test for one token text: <0xNN>, with what u8::from_str_radix(.., 16) takes for NN
// (either case, a leading '+'). The byte, or -1.
static int byte_token_value(const std::string& t) {
    if (t.size() != 6 || t.compare(0, 3, "<0x") != 0 || t[5] != '>') return -1;
    auto hex = [](char c) { return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1; };
    int a = hex(t[3]), b = hex(t[4]);
    if (b < 0) return -1;
    if (t[3] == '+') return b;
    return a < 0 ? -1 : a * 16 + b;
}

// The tokenizer.json of Gemma 4: anything this port does not cover exactly fails here.
void Tokenizer::Impl::load_sp(const Json& j) {
    const Json& m = j["model"];
    sp = true;
    GLM_CHECK(!m.get("dropout") || m["dropout"].kind == Json::Null, "tokenizer: BPE dropout is not supported");
    GLM_CHECK(m.get_s("continuing_subword_prefix", "").empty() && m.get_s("end_of_word_suffix", "").empty(),
              "tokenizer: subword prefix / suffix not supported");
    ignore_merges = m.get_b("ignore_merges", false);

    auto replace_step = [&](const Json& n) {
        GLM_CHECK(n.get_s("type", "") == "Replace" && n["pattern"].has("String") && !n["pattern"]["String"].s().empty(),
                  "tokenizer: only Replace on a string is supported in the normalizer");
        sp_replace.emplace_back(n["pattern"]["String"].s(), n["content"].s());
    };
    if (const Json* n = j.get("normalizer"); n && n->kind != Json::Null) {
        if (n->get_s("type", "") == "Sequence") for (auto& x : (*n)["normalizers"].arr) replace_step(x);
        else replace_step(*n);
    }
    if (const Json* pt = j.get("pre_tokenizer"); pt && pt->kind != Json::Null) {
        GLM_CHECK(pt->get_s("type", "") == "Split" && (*pt)["pattern"].has("String") && !pt->get_b("invert", false) &&
                  pt->get_s("behavior", "") == "MergedWithPrevious", "tokenizer: unexpected pre_tokenizer");
        sp_split = (*pt)["pattern"]["String"].s();
    }
    const Json& de = j["decoder"];
    GLM_CHECK(de.get_s("type", "") == "Sequence" && de["decoders"].size() == 3, "tokenizer: unexpected decoder");
    const Json& dr = de["decoders"][0];
    GLM_CHECK(dr.get_s("type", "") == "Replace" && dr["pattern"].has("String") && dr["pattern"]["String"].s() == "\xE2\x96\x81" &&
              dr["content"].s() == " " && de["decoders"][1].get_s("type", "") == "ByteFallback" &&
              de["decoders"][2].get_s("type", "") == "Fuse", "tokenizer: unexpected decoder");

    const Json& vj = m["vocab"];
    int max_id = -1;
    for (auto& [k, v] : vj.obj) max_id = std::max(max_id, (int)v.i());
    for (auto& a : j["added_tokens"].arr) max_id = std::max(max_id, (int)a["id"].i());
    n_vocab = max_id + 1;
    bytes.assign(n_vocab, "");
    special.assign(n_vocab, 0);
    byte_val.assign(n_vocab, -1);
    // what the decoder makes of a token: Replace applied, ByteFallback's value
    auto set_text = [&](int id, const std::string& k) {
        std::string o;
        size_t p = 0;
        for (size_t q; (q = k.find("\xE2\x96\x81", p)) != std::string::npos; p = q + 3) o.append(k, p, q - p) += ' ';
        o.append(k, p, std::string::npos);
        byte_val[id] = (int16_t)byte_token_value(o);
        bytes[id] = std::move(o);
    };
    vocab.reserve(vj.size() * 2);
    for (auto& [k, v] : vj.obj) {
        int id = (int)v.i();
        vocab.emplace(k, id);
        set_text(id, k);
        auto p = reinterpret_cast<const unsigned char*>(k.data());
        int w = k.empty() ? 0 : utf8_seq(p, k.size(), 0);
        if (w > 0 && (size_t)w == k.size()) sp_char.emplace(utf8_cp(p, w), id);
    }
    for (int b = 0; b < 256; b++) {
        char name[8];
        snprintf(name, sizeof name, "<0x%02X>", b);
        auto it = vocab.find(name);
        GLM_CHECK(it != vocab.end(), "tokenizer: byte_fallback without the token %s", name);
        sp_byte[b] = it->second;
    }

    const Json& mg = m["merges"];
    merges.reserve(mg.size() * 2);
    for (size_t r = 0; r < mg.size(); r++) {
        const Json& e = mg.arr[r];
        std::string a, b;
        if (e.kind == Json::Arr) { a = e[0].s(); b = e[1].s(); }
        else {
            size_t sp2 = e.s().find(' ', 1);
            GLM_CHECK(sp2 != std::string::npos, "tokenizer: bad merge %s", e.s().c_str());
            a = e.s().substr(0, sp2);
            b = e.s().substr(sp2 + 1);
        }
        auto ia = vocab.find(a), ib = vocab.find(b), im = vocab.find(a + b);
        GLM_CHECK(ia != vocab.end() && ib != vocab.end() && im != vocab.end(),
                  "tokenizer: merge %s %s outside the vocabulary", a.c_str(), b.c_str());
        // HF collects the merges into a HashMap: a pair listed twice takes its last rank
        merges.insert_or_assign((uint64_t)(uint32_t)ia->second << 32 | (uint32_t)ib->second, std::make_pair((int)r, im->second));
    }

    for (auto& a : j["added_tokens"].arr) {
        GLM_CHECK(!a.get_b("lstrip", false) && !a.get_b("rstrip", false) && !a.get_b("single_word", false) &&
                  !a.get_b("normalized", false),
                  "tokenizer: added token %s uses lstrip / rstrip / single_word / normalized", a["content"].s().c_str());
        int id = (int)a["id"].i();
        bool spc = a.get_b("special", false);
        added.push_back({id, a["content"].s(), spc});
        special[id] = spc;
        set_text(id, a["content"].s());
    }
    for (size_t i = 0; i < added.size(); i++)
        if (!added[i].content.empty()) added_by_byte[(unsigned char)added[i].content[0]].push_back((int)i);
    for (auto& v : added_by_byte)
        std::stable_sort(v.begin(), v.end(), [&](int x, int y) { return added[x].content.size() > added[y].content.size(); });

    text.resize(n_vocab);
    std::vector<int> one(1);
    for (int id = 0; id < n_vocab; id++) {
        one[0] = id;
        text[id] = decode_sp(one, false);
    }
}

Tokenizer::Tokenizer() : d(std::make_unique<Impl>()) {}
Tokenizer::Tokenizer(Tokenizer&&) noexcept = default;
Tokenizer& Tokenizer::operator=(Tokenizer&&) noexcept = default;
Tokenizer::~Tokenizer() = default;

Tokenizer Tokenizer::load(const std::string& model_dir) {
    Tokenizer t;
    Impl& d = *t.d;
    Json j = Json::parse(slurp(model_dir + "/tokenizer.json"));
    const Json& m = j["model"];
    GLM_CHECK(m.get_s("type", "") == "BPE", "tokenizer: model type %s, expected BPE", m.get_s("type", "?").c_str());
    if (m.get_b("byte_fallback", false)) {
        d.load_sp(j);
        return t;
    }
    GLM_CHECK(!m.get("dropout") || m["dropout"].kind == Json::Null, "tokenizer: BPE dropout is not supported");
    GLM_CHECK(m.get_s("continuing_subword_prefix", "").empty() && m.get_s("end_of_word_suffix", "").empty(),
              "tokenizer: subword prefix / suffix not supported");
    if (j.get("normalizer") && j["normalizer"].kind != Json::Null) {
        GLM_CHECK(j["normalizer"].get_s("type", "") == "NFC", "tokenizer: only the NFC normalizer is supported");
        d.nfc = true;
    }
    d.ignore_merges = m.get_b("ignore_merges", false);

    // pre-tokenizer: Sequence[Split(regex, Isolated or inverted Removed), ByteLevel(add_prefix_space false, use_regex false)]
    const Json& pt = j["pre_tokenizer"];
    GLM_CHECK(pt.get_s("type", "") == "Sequence" && pt["pretokenizers"].size() == 2, "tokenizer: unexpected pre_tokenizer");
    const Json& sp = pt["pretokenizers"][0];
    const Json& bl = pt["pretokenizers"][1];
    bool invert = sp.get_b("invert", false);
    std::string behavior = sp.get_s("behavior", "");
    GLM_CHECK(sp.get_s("type", "") == "Split" && ((behavior == "Isolated" && !invert) || (behavior == "Removed" && invert)) &&
              sp["pattern"].has("Regex"), "tokenizer: unexpected Split pre-tokenizer");
    d.keep_matches = invert;
    GLM_CHECK(bl.get_s("type", "") == "ByteLevel" && !bl.get_b("add_prefix_space", true) && !bl.get_b("use_regex", true),
              "tokenizer: unexpected ByteLevel pre-tokenizer");
    GLM_CHECK(j["decoder"].get_s("type", "") == "ByteLevel", "tokenizer: decoder is not ByteLevel");

    std::string pat = onig_to_pcre2(sp["pattern"]["Regex"].s());
    int ec;
    PCRE2_SIZE eo;
    d.re = pcre2_compile((PCRE2_SPTR)pat.c_str(), pat.size(), PCRE2_UTF | PCRE2_UCP, &ec, &eo, nullptr);
    if (!d.re) {
        PCRE2_UCHAR msg[256];
        pcre2_get_error_message(ec, msg, sizeof msg);
        GLM_CHECK(false, "tokenizer regex: %s at %zu", (const char*)msg, (size_t)eo);
    }
    pcre2_jit_compile(d.re, PCRE2_JIT_COMPLETE);   // falls back to the interpreter if JIT is unavailable

    // byte-level alphabet
    uint32_t cp_of[256];
    byte_level_table(cp_of);
    std::unordered_map<uint32_t, unsigned char> byte_of;
    for (int b = 0; b < 256; b++) byte_of[cp_of[b]] = (unsigned char)b;
    // HF ByteLevel decoder on one token: every character mapped back to its byte; false (raw = the token's own
    // UTF-8) when one is outside the alphabet
    auto bl_decode = [&](const std::string& s, std::string& raw) {
        raw.clear();
        auto p = reinterpret_cast<const unsigned char*>(s.data());
        for (size_t i = 0; i < s.size();) {
            int w = utf8_seq(p, s.size(), i);
            uint32_t cp = w == 1 ? p[i] : w == 2 ? (p[i] & 0x1F) << 6 | (p[i + 1] & 0x3F)
                        : w == 3 ? (p[i] & 0x0F) << 12 | (p[i + 1] & 0x3F) << 6 | (p[i + 2] & 0x3F)
                        : w == 4 ? (p[i] & 0x07) << 18 | (p[i + 1] & 0x3F) << 12 | (p[i + 2] & 0x3F) << 6 | (p[i + 3] & 0x3F)
                                 : 0xFFFFFFFFu;
            auto it = byte_of.find(cp);
            if (it == byte_of.end()) { raw = s; return false; }
            raw += (char)it->second;
            i += w;
        }
        return true;
    };

    const Json& vocab = m["vocab"];
    int max_id = -1;
    for (auto& [k, v] : vocab.obj) max_id = std::max(max_id, (int)v.i());
    for (auto& a : j["added_tokens"].arr) max_id = std::max(max_id, (int)a["id"].i());
    d.n_vocab = max_id + 1;
    d.bytes.assign(d.n_vocab, "");
    d.special.assign(d.n_vocab, 0);
    std::vector<char> present(d.n_vocab, 0);
    d.vocab.reserve(vocab.size() * 2);
    d.vocab_bl.reserve(vocab.size() * 2);
    std::string raw;
    for (auto& [k, v] : vocab.obj) {
        int id = (int)v.i();
        // a token with a character outside the byte-level alphabet can never come out of BPE on mapped bytes
        if (bl_decode(k, raw)) d.vocab.emplace(raw, id);
        d.bytes[id] = raw;
        present[id] = 1;
        d.vocab_bl.emplace(k, id);
    }
    for (int b = 0; b < 256; b++) {
        auto it = d.vocab.find(std::string(1, (char)b));
        d.byte_id[b] = it == d.vocab.end() ? -1 : it->second;
    }

    const Json& merges = m["merges"];
    d.merges.reserve(merges.size() * 2);
    for (size_t r = 0; r < merges.size(); r++) {
        const Json& e = merges.arr[r];
        std::string a, b;
        if (e.kind == Json::Arr) { a = e[0].s(); b = e[1].s(); }
        else {
            size_t sp2 = e.s().find(' ', 1);
            GLM_CHECK(sp2 != std::string::npos, "tokenizer: bad merge %s", e.s().c_str());
            a = e.s().substr(0, sp2);
            b = e.s().substr(sp2 + 1);
        }
        auto ia = d.vocab_bl.find(a), ib = d.vocab_bl.find(b), im = d.vocab_bl.find(a + b);
        GLM_CHECK(ia != d.vocab_bl.end() && ib != d.vocab_bl.end() && im != d.vocab_bl.end(),
                  "tokenizer: merge %s %s outside the vocabulary", a.c_str(), b.c_str());
        bool fresh = d.merges.emplace((uint64_t)(uint32_t)ia->second << 32 | (uint32_t)ib->second,
                                      std::make_pair((int)r, im->second)).second;
        GLM_CHECK(fresh, "tokenizer: merge %s %s listed twice", a.c_str(), b.c_str());
    }

    for (auto& a : j["added_tokens"].arr) {
        GLM_CHECK(!a.get_b("lstrip", false) && !a.get_b("rstrip", false) && !a.get_b("single_word", false),
                  "tokenizer: added token %s uses lstrip / rstrip / single_word", a["content"].s().c_str());
        int id = (int)a["id"].i();
        bool spc = a.get_b("special", false);
        d.added.push_back({id, a["content"].s(), spc});
        d.special[id] = spc;
        bl_decode(a["content"].s(), d.bytes[id]);
        present[id] = 1;
    }
    for (size_t i = 0; i < d.added.size(); i++)
        if (!d.added[i].content.empty()) d.added_by_byte[(unsigned char)d.added[i].content[0]].push_back((int)i);
    for (auto& v : d.added_by_byte)
        std::stable_sort(v.begin(), v.end(), [&](int x, int y) { return d.added[x].content.size() > d.added[y].content.size(); });

    d.text.resize(d.n_vocab);
    for (int id = 0; id < d.n_vocab; id++)
        if (present[id]) d.text[id] = utf8_lossy(d.bytes[id]);
    return t;
}

std::vector<int> Tokenizer::encode(const std::string& in, bool special) const {
    const Impl& m = *d;
    const std::string* sp = &in;
    std::string fixed;
    if (!utf8_valid(in)) { fixed = utf8_lossy(in); sp = &fixed; }
    const std::string& s = *sp;
    std::vector<int> out;
    out.reserve(s.size() / 3 + 8);
    pcre2_match_data* md = m.re ? pcre2_match_data_create_from_pattern(m.re, nullptr) : nullptr;   // (byte fallback: no regex)
    size_t start = 0;   // of the pending plain stretch
    for (size_t i = 0; i < s.size();) {
        const Impl::Added* hit = nullptr;
        for (int ai : m.added_by_byte[(unsigned char)s[i]]) {
            const auto& a = m.added[ai];
            if (!special && a.special) continue;
            if (s.compare(i, a.content.size(), a.content) == 0) { hit = &a; break; }
        }
        if (!hit) { i++; continue; }
        if (i > start) m.encode_stretch(s.data() + start, i - start, md, out);
        out.push_back(hit->id);
        i += hit->content.size();
        start = i;
    }
    if (start < s.size()) m.encode_stretch(s.data() + start, s.size() - start, md, out);
    if (md) pcre2_match_data_free(md);
    return out;
}

std::string Tokenizer::decode(const std::vector<int>& ids, bool skip_special) const {
    if (d->sp) return d->decode_sp(ids, skip_special);
    std::string b;
    for (int id : ids) {
        if (id < 0 || id >= d->n_vocab) continue;
        if (skip_special && d->special[id]) continue;
        b += d->bytes[id];
    }
    return utf8_lossy(b);
}

const std::string& Tokenizer::token_text(int id) const {
    static const std::string none;
    return id >= 0 && id < d->n_vocab ? d->text[id] : none;
}

int Tokenizer::vocab_size() const { return d->n_vocab; }

bool Tokenizer::is_special(int id) const { return id >= 0 && id < d->n_vocab && d->special[id]; }

std::optional<int> Tokenizer::id_of(const std::string& tok) const {
    for (auto& a : d->added)
        if (a.content == tok) return a.id;
    const auto& v = d->sp ? d->vocab : d->vocab_bl;
    auto it = v.find(tok);
    if (it != v.end()) return it->second;
    return std::nullopt;
}

std::vector<std::string> Tokenizer::pre_tokens(const std::string& in) const {
    std::string s = utf8_lossy(in);
    std::vector<std::string> out;
    if (d->sp) {
        d->sp_pieces(d->sp_normalize(s.data(), s.size()), [&](const char* p, size_t n) { out.emplace_back(p, n); });
        return out;
    }
    pcre2_match_data* md = pcre2_match_data_create_from_pattern(d->re, nullptr);
    d->split(s.data(), s.size(), md, [&](const char* p, size_t n) { out.emplace_back(p, n); });
    pcre2_match_data_free(md);
    return out;
}

// ---------------------------------------------------------------------------------------------------- streaming

static bool ends_with_replacement(const std::string& s) {
    return s.size() >= 3 && s.compare(s.size() - 3, 3, "\xEF\xBF\xBD") == 0;
}

std::string StreamDecoder::push(int id, bool last) {
    held_ += tok_.token_text(id);
    held_ids_.push_back(id);
    if (last) return flush();
    if (ends_with_replacement(held_)) {
        std::string joint = tok_.decode(held_ids_, true);
        if (ends_with_replacement(joint) && utf8_length(joint) <= max_stop_len_ + 20) return {};
        held_ = std::move(joint);
    }
    return flush();
}

std::string StreamDecoder::flush() {
    std::string o = std::move(held_);
    held_.clear();
    held_ids_.clear();
    return o;
}

}  // namespace glm
