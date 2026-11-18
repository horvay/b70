#pragma once
// JSON: objects, arrays, strings, numbers, true/false/null. Objects keep insertion order (as Python dicts do, which
// the chat template's tojson and items() depend on); integers keep their exact literal; dump() is Python's
// json.dumps(x, ensure_ascii=...) with its default separators, byte for byte.
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace glm {

// Object members in insertion order. A key set again keeps its first position and takes the new value (Python dict
// semantics, also for duplicate keys in parsed text). Objects above 8 members get a hash index. (A template only so
// that its members are instantiated once Json is complete.)
template <class J>
class JsonObjT {
public:
    using Item = std::pair<std::string, J>;
    using Items = std::vector<Item>;
    const J* find(const std::string& k) const {
        ptrdiff_t p = pos(k);
        return p < 0 ? nullptr : &items_[p].second;
    }
    J* find(const std::string& k) {
        ptrdiff_t p = pos(k);
        return p < 0 ? nullptr : &items_[p].second;
    }
    size_t count(const std::string& k) const { return find(k) ? 1 : 0; }
    const J& at(const std::string& k) const {
        const J* v = find(k);
        if (!v) throw std::runtime_error("json: no key " + k);
        return *v;
    }
    J& operator[](const std::string& k) {      // inserts null when absent
        ptrdiff_t p = pos(k);
        return p >= 0 ? items_[p].second : set(k, J());
    }
    J& set(std::string k, J v) {
        ptrdiff_t p = pos(k);
        if (p >= 0) return items_[p].second = std::move(v);
        items_.emplace_back(std::move(k), std::move(v));
        if (!index_.empty()) index_.emplace(items_.back().first, items_.size() - 1);
        else if (items_.size() > kIndexAbove) reindex();
        return items_.back().second;
    }
    bool erase(const std::string& k) {
        ptrdiff_t p = pos(k);
        if (p < 0) return false;
        items_.erase(items_.begin() + p);
        reindex();
        return true;
    }
    size_t size() const { return items_.size(); }
    bool empty() const { return items_.empty(); }
    const Item* begin() const { return items_.data(); }
    const Item* end() const { return items_.data() + items_.size(); }
    Item* begin() { return items_.data(); }
    Item* end() { return items_.data() + items_.size(); }

private:
    static constexpr size_t kIndexAbove = 8;
    Items items_;
    std::unordered_map<std::string, size_t> index_;   // empty while size() <= kIndexAbove

    ptrdiff_t pos(const std::string& k) const {
        if (!index_.empty()) {
            auto it = index_.find(k);
            return it == index_.end() ? -1 : (ptrdiff_t)it->second;
        }
        for (size_t i = 0; i < items_.size(); i++)
            if (items_[i].first == k) return (ptrdiff_t)i;
        return -1;
    }
    void reindex() {
        index_.clear();
        if (items_.size() <= kIndexAbove) return;
        index_.reserve(items_.size() * 2);
        for (size_t i = 0; i < items_.size(); i++) index_.emplace(items_[i].first, i);
    }
};

struct Json {
    enum Kind { Null, Bool, Num, Str, Arr, Obj } kind = Null;
    bool b = false;
    double num = 0;
    bool integer = false;   // Num written without fraction or exponent; str then holds its literal (exact)
    std::string str;
    std::vector<Json> arr;
    JsonObjT<Json> obj;

    // defined out of line: the implicit ones would need std::pair<std::string, Json> complete inside its own
    // instantiation (libstdc++ 16)
    Json();
    ~Json();
    Json(const Json&);
    Json(Json&&) noexcept;
    Json& operator=(const Json&);
    Json& operator=(Json&&) noexcept;

    bool has(const std::string& k) const { return kind == Obj && obj.count(k); }
    const Json& operator[](const std::string& k) const {
        const Json* v = kind == Obj ? obj.find(k) : nullptr;
        if (!v) throw std::runtime_error("json: no key " + k);
        return *v;
    }
    const Json& operator[](size_t i) const { return arr.at(i); }
    size_t size() const { return kind == Arr ? arr.size() : kind == Obj ? obj.size() : 0; }
    int64_t i() const { return integer ? std::strtoll(str.c_str(), nullptr, 10) : (int64_t)num; }
    double f() const { return num; }
    const std::string& s() const { return str; }
    // value at key k, or def when absent / null
    int64_t get_i(const std::string& k, int64_t def) const { auto v = get(k); return v && v->kind == Num ? v->i() : def; }
    double get_f(const std::string& k, double def) const { auto v = get(k); return v && v->kind == Num ? v->f() : def; }
    bool get_b(const std::string& k, bool def) const { auto v = get(k); return v && v->kind == Bool ? v->b : def; }
    std::string get_s(const std::string& k, const std::string& def) const {
        auto v = get(k);
        return v && v->kind == Str ? v->str : def;
    }
    // member k, or nullptr (also when this is not an object)
    const Json* get(const std::string& k) const { return kind == Obj ? obj.find(k) : nullptr; }

    // builders
    static Json make_bool(bool v);
    static Json make_num(double v);
    static Json make_int(int64_t v);
    static Json make_str(std::string v);
    static Json make_arr();
    static Json make_obj();
    Json& push(Json v);
    Json& set(std::string k, Json v);

    // Python json.loads grammar (NaN / Infinity / -Infinity accepted, control characters in strings and unknown
    // escapes rejected). try_parse returns nullopt on malformed text; parse throws.
    static std::optional<Json> try_parse(const std::string& text, std::string* err = nullptr);
    static Json parse(const std::string& text);
};

// Python json.dumps(j, ensure_ascii=ensure_ascii): separators ", " and ": ", insertion order, floats as repr().
std::string dump(const Json& j, bool ensure_ascii = false);

using JsonObj = JsonObjT<Json>;

inline Json::Json() = default;
inline Json::~Json() = default;
inline Json::Json(const Json&) = default;
inline Json::Json(Json&&) noexcept = default;
inline Json& Json::operator=(const Json&) = default;
inline Json& Json::operator=(Json&&) noexcept = default;

inline Json Json::make_bool(bool v) { Json j; j.kind = Bool; j.b = v; return j; }
inline Json Json::make_num(double v) { Json j; j.kind = Num; j.num = v; return j; }
inline Json Json::make_int(int64_t v) {
    Json j; j.kind = Num; j.num = (double)v; j.integer = true; j.str = std::to_string(v); return j;
}
inline Json Json::make_str(std::string v) { Json j; j.kind = Str; j.str = std::move(v); return j; }
inline Json Json::make_arr() { Json j; j.kind = Arr; return j; }
inline Json Json::make_obj() { Json j; j.kind = Obj; return j; }
inline Json& Json::push(Json v) { arr.push_back(std::move(v)); return arr.back(); }
inline Json& Json::set(std::string k, Json v) { return obj.set(std::move(k), std::move(v)); }

// ---------------------------------------------------------------------------------------------------- parser

namespace json_detail {

inline void put_utf8(std::string& o, uint32_t cp) {
    if (cp < 0x80) o += (char)cp;
    else if (cp < 0x800) { o += (char)(0xC0 | cp >> 6); o += (char)(0x80 | (cp & 63)); }
    else if (cp < 0x10000) { o += (char)(0xE0 | cp >> 12); o += (char)(0x80 | (cp >> 6 & 63)); o += (char)(0x80 | (cp & 63)); }
    else { o += (char)(0xF0 | cp >> 18); o += (char)(0x80 | (cp >> 12 & 63)); o += (char)(0x80 | (cp >> 6 & 63));
           o += (char)(0x80 | (cp & 63)); }
}

struct Parser {
    const std::string& t;
    size_t p = 0;
    std::string err;

    bool fail(const char* what) {
        if (err.empty()) err = std::string(what) + " at offset " + std::to_string(p);
        return false;
    }
    void ws() { while (p < t.size() && (t[p] == ' ' || t[p] == '\n' || t[p] == '\r' || t[p] == '\t')) p++; }
    bool lit(const char* s) {
        size_t n = strlen(s);
        if (t.compare(p, n, s) != 0) return false;
        p += n;
        return true;
    }
    bool hex4(uint32_t& v) {
        if (p + 4 > t.size()) return false;
        v = 0;
        for (int k = 0; k < 4; k++) {
            char c = t[p + k];
            int d = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1;
            if (d < 0) return false;
            v = v * 16 + d;
        }
        p += 4;
        return true;
    }
    bool string(std::string& o) {
        p++;   // opening quote
        for (;;) {
            size_t s = p;
            while (p < t.size() && t[p] != '"' && t[p] != '\\' && (unsigned char)t[p] >= 0x20) p++;
            o.append(t, s, p - s);
            if (p >= t.size()) return fail("unterminated string");
            char c = t[p];
            if (c == '"') { p++; return true; }
            if (c != '\\') return fail("control character in string");
            if (++p >= t.size()) return fail("unterminated string");
            char e = t[p++];
            switch (e) {
                case '"': o += '"'; break;
                case '\\': o += '\\'; break;
                case '/': o += '/'; break;
                case 'b': o += '\b'; break;
                case 'f': o += '\f'; break;
                case 'n': o += '\n'; break;
                case 'r': o += '\r'; break;
                case 't': o += '\t'; break;
                case 'u': {
                    uint32_t cp;
                    if (!hex4(cp)) return fail("bad \\u escape");
                    if (cp >= 0xD800 && cp < 0xDC00 && t.compare(p, 2, "\\u") == 0) {
                        size_t save = p;
                        p += 2;
                        uint32_t lo;
                        if (hex4(lo) && lo >= 0xDC00 && lo < 0xE000) cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        else p = save;
                    }
                    if (cp >= 0xD800 && cp < 0xE000) cp = 0xFFFD;   // a lone surrogate has no UTF-8 form
                    put_utf8(o, cp);
                    break;
                }
                default: p--; return fail("invalid escape");
            }
        }
    }
    bool number(Json& j) {
        size_t s = p;
        if (p < t.size() && t[p] == '-') p++;
        if (lit("Infinity")) { j.kind = Json::Num; j.num = t[s] == '-' ? -INFINITY : INFINITY; return true; }
        if (p >= t.size() || !isdigit((unsigned char)t[p])) return fail("bad value");
        if (t[p] == '0') p++;
        else while (p < t.size() && isdigit((unsigned char)t[p])) p++;
        bool integral = true;
        if (p + 1 < t.size() && t[p] == '.' && isdigit((unsigned char)t[p + 1])) {
            integral = false;
            p++;
            while (p < t.size() && isdigit((unsigned char)t[p])) p++;
        }
        if (p < t.size() && (t[p] == 'e' || t[p] == 'E')) {
            size_t q = p + 1;
            if (q < t.size() && (t[q] == '+' || t[q] == '-')) q++;
            if (q < t.size() && isdigit((unsigned char)t[q])) {
                integral = false;
                p = q;
                while (p < t.size() && isdigit((unsigned char)t[p])) p++;
            }
        }
        std::string lexeme = t.substr(s, p - s);
        j.kind = Json::Num;
        j.num = strtod(lexeme.c_str(), nullptr);
        if (integral) {
            j.integer = true;
            j.str = lexeme == "-0" ? "0" : std::move(lexeme);
        }
        return true;
    }
    bool value(Json& j, int depth) {
        if (depth > 512) return fail("nesting too deep");
        ws();
        if (p >= t.size()) return fail("unexpected end");
        char c = t[p];
        if (c == '{') {
            j.kind = Json::Obj;
            p++;
            ws();
            if (p < t.size() && t[p] == '}') { p++; return true; }
            for (;;) {
                ws();
                if (p >= t.size() || t[p] != '"') return fail("expected a key");
                std::string k;
                if (!string(k)) return false;
                ws();
                if (p >= t.size() || t[p] != ':') return fail("expected ':'");
                p++;
                Json v;
                if (!value(v, depth + 1)) return false;
                j.obj.set(std::move(k), std::move(v));
                ws();
                if (p < t.size() && t[p] == ',') { p++; continue; }
                if (p < t.size() && t[p] == '}') { p++; return true; }
                return fail("expected ',' or '}'");
            }
        }
        if (c == '[') {
            j.kind = Json::Arr;
            p++;
            ws();
            if (p < t.size() && t[p] == ']') { p++; return true; }
            for (;;) {
                j.arr.emplace_back();
                if (!value(j.arr.back(), depth + 1)) return false;
                ws();
                if (p < t.size() && t[p] == ',') { p++; continue; }
                if (p < t.size() && t[p] == ']') { p++; return true; }
                return fail("expected ',' or ']'");
            }
        }
        if (c == '"') { j.kind = Json::Str; return string(j.str); }
        if (lit("true")) { j.kind = Json::Bool; j.b = true; return true; }
        if (lit("false")) { j.kind = Json::Bool; return true; }
        if (lit("null")) return true;
        if (lit("NaN")) { j.kind = Json::Num; j.num = NAN; return true; }
        return number(j);
    }
};

// Python float.__repr__: shortest round-trip digits; positional when -4 < exponent10 + 1 <= 16, else d.ddde+XX.
inline std::string float_repr(double v) {
    if (std::isnan(v)) return "nan";
    if (std::isinf(v)) return v < 0 ? "-inf" : "inf";
    if (v == 0) return std::signbit(v) ? "-0.0" : "0.0";
    char buf[64];
    auto r = std::to_chars(buf, buf + sizeof buf, v, std::chars_format::scientific);
    std::string s(buf, r.ptr);
    bool neg = s[0] == '-';
    if (neg) s.erase(0, 1);
    size_t e = s.find('e');
    int exp10 = atoi(s.c_str() + e + 1);
    std::string digits;
    for (size_t k = 0; k < e; k++)
        if (s[k] != '.') digits += s[k];
    int decpt = exp10 + 1;   // value = 0.digits * 10^decpt
    std::string o = neg ? "-" : "";
    if (decpt <= -4 || decpt > 16) {
        o += digits[0];
        if (digits.size() > 1) o += "." + digits.substr(1);
        int x = decpt - 1;
        o += x < 0 ? "e-" : "e+";
        std::string xs = std::to_string(x < 0 ? -x : x);
        if (xs.size() < 2) xs = "0" + xs;
        o += xs;
    } else if (decpt <= 0) {
        o += "0." + std::string(-decpt, '0') + digits;
    } else if ((size_t)decpt >= digits.size()) {
        o += digits + std::string(decpt - digits.size(), '0') + ".0";
    } else {
        o += digits.substr(0, decpt) + "." + digits.substr(decpt);
    }
    return o;
}

inline void dump_str(std::string& o, const std::string& s, bool ascii) {
    static const char* hex = "0123456789abcdef";
    o += '"';
    for (size_t k = 0; k < s.size(); k++) {
        unsigned char c = s[k];
        switch (c) {
            case '"': o += "\\\""; continue;
            case '\\': o += "\\\\"; continue;
            case '\n': o += "\\n"; continue;
            case '\r': o += "\\r"; continue;
            case '\t': o += "\\t"; continue;
            case '\b': o += "\\b"; continue;
            case '\f': o += "\\f"; continue;
        }
        if (c < 0x20) { o += "\\u00"; o += hex[c >> 4]; o += hex[c & 15]; continue; }
        if (c < 0x80 || !ascii) { o += (char)c; continue; }
        // ensure_ascii: one code point as \uXXXX (a surrogate pair above the BMP)
        uint32_t cp;
        int n = c >= 0xF0 ? 3 : c >= 0xE0 ? 2 : 1;
        cp = c & (0x3F >> n);
        for (int m = 0; m < n && k + 1 < s.size(); m++) cp = cp << 6 | (s[++k] & 0x3F);
        auto u = [&](uint32_t x) {
            o += "\\u";
            for (int sh = 12; sh >= 0; sh -= 4) o += hex[x >> sh & 15];
        };
        if (cp >= 0x10000) { cp -= 0x10000; u(0xD800 | cp >> 10); u(0xDC00 | (cp & 0x3FF)); }
        else u(cp);
    }
    o += '"';
}

inline void dump_to(std::string& o, const Json& j, bool ascii) {
    switch (j.kind) {
        case Json::Null: o += "null"; return;
        case Json::Bool: o += j.b ? "true" : "false"; return;
        case Json::Num:
            if (j.integer) o += j.str;
            else if (std::isnan(j.num)) o += "NaN";
            else if (std::isinf(j.num)) o += j.num < 0 ? "-Infinity" : "Infinity";
            else o += float_repr(j.num);
            return;
        case Json::Str: dump_str(o, j.str, ascii); return;
        case Json::Arr: {
            o += '[';
            for (size_t k = 0; k < j.arr.size(); k++) {
                if (k) o += ", ";
                dump_to(o, j.arr[k], ascii);
            }
            o += ']';
            return;
        }
        case Json::Obj: {
            o += '{';
            bool first = true;
            for (auto& [k, v] : j.obj) {
                if (!first) o += ", ";
                first = false;
                dump_str(o, k, ascii);
                o += ": ";
                dump_to(o, v, ascii);
            }
            o += '}';
            return;
        }
    }
}

}  // namespace json_detail

inline std::optional<Json> Json::try_parse(const std::string& text, std::string* err) {
    json_detail::Parser ps{text};
    Json j;
    bool ok = ps.value(j, 0);
    if (ok) {
        ps.ws();
        if (ps.p != text.size()) ok = ps.fail("trailing data");
    }
    if (!ok) {
        if (err) *err = ps.err;
        return std::nullopt;
    }
    return j;
}

inline Json Json::parse(const std::string& text) {
    std::string err;
    auto j = try_parse(text, &err);
    if (!j) throw std::runtime_error("json: " + err);
    return std::move(*j);
}

inline std::string dump(const Json& j, bool ensure_ascii) {
    std::string o;
    json_detail::dump_to(o, j, ensure_ascii);
    return o;
}

}  // namespace glm
