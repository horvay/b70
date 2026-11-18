// Ripple-65B's tokenizer, streaming detokenizer, chat template and output parsers, against tests/ripple_ref.py's outputs (HF
// tokenizers, HF's Jinja environment, a transcription of ripple_format.py's reasoning detector) and, for the parsers, cases
// derived by hand from ripple_format.py and the hermes layout. CPU only.
//   ripple_check <model dir> <ref dir>       ref dir: where ripple_ref.py wrote chat.json, encode.json, sweep.json, decode.json,
//                                            split.json, tiny.json and the tiny* tokenizers
// Every case must match byte for byte; prints the first mismatches of each kind and a summary.
#include <chrono>
#include <cstdio>
#include <fstream>
#include <random>
#include <sstream>
#include "../src/chat.h"
#include "../src/rt.h"
#include "../src/tokenizer.h"

using namespace glm;

static std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    GLM_CHECK(f, "cannot open %s", path.c_str());
    std::stringstream s;
    s << f.rdbuf();
    return s.str();
}

static std::vector<int> ids_of(const Json& a) {
    std::vector<int> v;
    v.reserve(a.arr.size());
    for (auto& x : a.arr) v.push_back((int)x.i());
    return v;
}

static std::string esc(const std::string& s, size_t max = 160) {
    std::string o;
    for (unsigned char c : s.substr(0, max)) {
        if (c == '\n') o += "\\n";
        else if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\x%02x", c); o += b; }
        else o += (char)c;
    }
    return o + (s.size() > max ? "..." : "");
}

static std::string first_diff(const std::string& a, const std::string& b) {
    size_t i = 0;
    while (i < a.size() && i < b.size() && a[i] == b[i]) i++;
    size_t s = i > 40 ? i - 40 : 0;
    return "at byte " + std::to_string(i) + ": got [" + esc(a.substr(s), 120) + "] want [" + esc(b.substr(s), 120) + "]";
}

static std::string ids_str(const std::vector<int>& v, size_t from, size_t n = 12) {
    std::string o;
    for (size_t i = from; i < v.size() && i < from + n; i++) o += std::to_string(v[i]) + " ";
    return o;
}

struct Tally {
    const char* what;
    int ok = 0, bad = 0;
    bool report() { return bad++ < 5; }
    void print() const { printf("%-38s %8d identical, %d different\n", what, ok, bad); }
};

static uint64_t fnv(const std::vector<int>& ids) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (int t : ids)
        for (int k = 0; k < 4; k++) { h ^= (uint32_t)t >> (8 * k) & 0xFF; h *= 0x100000001b3ull; }
    return h;
}

static std::string probe(uint32_t c) {
    std::string s;
    json_detail::put_utf8(s, c);
    return "x" + s + "x 1" + s + "1 " + s + "x !" + s + "! \n" + s + "\n'" + s + "'" + s + s + " " + s + " \t" + s + "\t ";
}

// ------------------------------------------------------------------------------------------- tool calls, by hand

struct CallCase {
    const char* text;
    const char* rest;                            // what stays content
    std::vector<std::pair<const char*, const char*>> calls;   // name, arguments as dumped
};

static std::vector<CallCase> call_cases() {
#define CALL(n, a) "<tool_call>\n{\"name\": \"" n "\", \"arguments\": " a "}\n</tool_call>"
    return {
        // the template's layout
        {CALL("f", "{\"a\": 1}"), "", {{"f", "{\"a\": 1}"}}},
        {"Hello\n" CALL("f", "{}"), "Hello", {{"f", "{}"}}},
        {"Hello\n\n" CALL("f", "{}"), "Hello\n", {{"f", "{}"}}},                                             // only the one newline goes
        {"  Hello  " CALL("f", "{}"), "  Hello  ", {{"f", "{}"}}},
        {"\n\n" CALL("f", "{}"), "\n", {{"f", "{}"}}},
        {CALL("f", "{}") "\n" CALL("g", "{\"b\": 2.5}"), "", {{"f", "{}"}, {"g", "{\"b\": 2.5}"}}},
        {"A\n" CALL("f", "{}") "\nB\n" CALL("g", "{}") "\nC", "A", {{"f", "{}"}, {"g", "{}"}}},               // text after a call: dropped
        // compact and otherwise spaced JSON
        {"<tool_call>{\"name\":\"f\",\"arguments\":{\"a\":1,\"b\":[1,2],\"c\":{\"d\":null}}}</tool_call>", "", {{"f", "{\"a\": 1, \"b\": [1, 2], \"c\": {\"d\": null}}"}}},
        {"<tool_call>\r\n  {  \"name\" : \"f\" ,\n \"arguments\" : { } }\t\n</tool_call>", "", {{"f", "{}"}}},
        {"<tool_call>{\"arguments\":{\"a\":1},\"name\":\"f\"}</tool_call>", "", {{"f", "{\"a\": 1}"}}},       // key order
        // values come back as json.dumps writes them
        {CALL("f", "{\"s\": \"q\\\"\\n\\t\\\\\", \"u\": \"\\u00e9\xE4\xBD\xA0\\ud83d\\ude00\xF0\x9F\x99\x82\", \"i\": 123456789012345678901234567890, \"x\": 1.50, \"e\": 1e3, \"t\": true, \"n\": null}"), "",
         {{"f", "{\"s\": \"q\\\"\\n\\t\\\\\", \"u\": \"\xC3\xA9\xE4\xBD\xA0\xF0\x9F\x98\x80\xF0\x9F\x99\x82\", \"i\": 123456789012345678901234567890, \"x\": 1.5, \"e\": 1000.0, \"t\": true, \"n\": null}"}}},
        // markers inside strings are text
        {CALL("f", "{\"s\": \"a</tool_call>b\"}"), "", {{"f", "{\"s\": \"a</tool_call>b\"}"}}},
        {CALL("f", "{\"s\": \"<tool_call>\"}") "\n" CALL("g", "{}"), "", {{"f", "{\"s\": \"<tool_call>\"}"}, {"g", "{}"}}},
        {"x\n" CALL("f", "{\"s\": \"</tool_call></tool_call>\"}") "tail", "x", {{"f", "{\"s\": \"</tool_call></tool_call>\"}"}}},
        // hermes' variants
        {"<tool_call>\n{\"name\": \"f\", \"parameters\": {\"a\": 1}}\n</tool_call>", "", {{"f", "{\"a\": 1}"}}},
        {"<tool_call>\n{\"name\": \"f\", \"parameters\": {}, \"arguments\": {\"a\": 1}}\n</tool_call>", "", {{"f", "{\"a\": 1}"}}},
        {"<tool_call>\n[{\"name\": \"f\", \"arguments\": {}}, {\"name\": \"g\", \"arguments\": {\"a\": 1}}]\n</tool_call>", "", {{"f", "{}"}, {"g", "{\"a\": 1}"}}},
        {"<tool_call>\n{\"name\": \"f\"}\n</tool_call>", "", {{"f", "{}"}}},
        {CALL("f", "null"), "", {{"f", "{}"}}},
        {CALL("f", "\"{\\\"a\\\": 1}\""), "", {{"f", "{\"a\": 1}"}}},                                         // a string holding an object
        {CALL("f", "\"plain\""), "", {{"f", "\"plain\""}}},                                                   // json.dumps of a string, as hermes
        {CALL("f", "[1, 2]"), "", {{"f", "[1, 2]"}}},
        {CALL("do.thing-x", "{}"), "", {{"do.thing-x", "{}"}}},                                               // any name, as the other parsers
        // malformed: nothing to parse, the text stays whole, newline included
        {"x\n<tool_call>\nnot json\n</tool_call>", "x\n<tool_call>\nnot json\n</tool_call>", {}},
        {"<tool_call>\n{\"arguments\": {}}\n</tool_call>", "<tool_call>\n{\"arguments\": {}}\n</tool_call>", {}},
        {"<tool_call>\n{\"name\": \"\", \"arguments\": {}}\n</tool_call>", "<tool_call>\n{\"name\": \"\", \"arguments\": {}}\n</tool_call>", {}},
        {"<tool_call>\n{\"name\": 5, \"arguments\": {}}\n</tool_call>", "<tool_call>\n{\"name\": 5, \"arguments\": {}}\n</tool_call>", {}},
        {"<tool_call>\n5\n</tool_call>", "<tool_call>\n5\n</tool_call>", {}},
        {"<tool_call></tool_call>", "<tool_call></tool_call>", {}},
        {"<tool_call>\n{\"name\": \"f\", \"arguments\": {\"a\": 1}\n</tool_call>", "<tool_call>\n{\"name\": \"f\", \"arguments\": {\"a\": 1}\n</tool_call>", {}},
        {"<tool_call>\n{\"name\": \"f\", \"arguments\": {\"s\": \"a\nb\"}}\n</tool_call>", "<tool_call>\n{\"name\": \"f\", \"arguments\": {\"s\": \"a\nb\"}}\n</tool_call>", {}},   // a raw newline in a string
        // cut off
        {"x\n<tool_call>\n{\"name\": \"f\"", "x\n<tool_call>\n{\"name\": \"f\"", {}},
        {"x\n<tool_call>", "x\n<tool_call>", {}},
        {"no marker here </tool_call>", "no marker here </tool_call>", {}},
        {"no marker", "no marker", {}},
        {"", "", {}},
        // a complete call without its closer
        {"x\n<tool_call>\n{\"name\": \"f\", \"arguments\": {}}\n", "x", {{"f", "{}"}}},
        // the good ones survive the bad ones
        {"x\n<tool_call>\nbad\n</tool_call>\n" CALL("g", "{}"), "x", {{"g", "{}"}}},
        {CALL("f", "{}") "\n<tool_call>\nbad\n</tool_call>", "", {{"f", "{}"}}},
        {CALL("f", "{}") "\n<tool_call>\n{\"name\": \"g\"", "", {{"f", "{}"}}},
        {"<tool_call>\n{\"name\": \"f\", \"arguments\": {}}\n<tool_call>\n{\"name\": \"g\", \"arguments\": {}}\n</tool_call>", "<tool_call>\n{\"name\": \"f\", \"arguments\": {}}\n<tool_call>\n{\"name\": \"g\", \"arguments\": {}}\n</tool_call>", {}},
    };
#undef CALL
}

// ------------------------------------------------------------------------------------------- RippleSplitter, by hand

struct SplitCase {
    const char* name;
    bool force, tools, separate;
    std::vector<std::string> chunks;
    std::string reasoning, content, held;
};

static std::vector<SplitCase> split_cases() {
    const std::string call = "<tool_call>\n{\"name\": \"f\", \"arguments\": {}}\n</tool_call>";
    return {
        {"the template's layout", false, false, true, {"<think>", "\n", "I", " think", "\n", "</think>", "\n", "Hello", " there"}, "I think", "Hello there", ""},
        {"one chunk", false, false, true, {"<think>\nr\n</think>\nc"}, "r", "c", ""},
        {"no thought", false, false, true, {"Hello", " world\n"}, "", "Hello world\n", ""},
        {"a thought later is text", false, false, true, {"x<think>y</think>z"}, "", "x<think>y</think>z", ""},
        {"a newline before the opener is text", false, false, true, {"\n<think>a</think>b"}, "", "\n<think>a</think>b", ""},
        {"a space before the opener is text", false, false, true, {" <think>a</think>b"}, "", " <think>a</think>b", ""},
        {"held prefix that is not the opener", false, false, true, {"<thi", "s"}, "", "<this", ""},
        {"cut off in the opener", false, false, true, {"<thi"}, "", "<thi", ""},
        {"cut off in the thought", false, false, true, {"<think>\nabc"}, "abc", "", ""},
        {"cut off in the closer", false, false, true, {"<think>\nabc\n</thi"}, "abc\n</thi", "", ""},
        {"cut off at a newline", false, false, true, {"<think>\nabc\n"}, "abc\n", "", ""},
        {"cut off after the opener", false, false, true, {"<think>"}, "", "", ""},
        {"cut off after its newline", false, false, true, {"<think>\n"}, "", "", ""},
        {"exactly the separators", false, false, true, {"<think>\n\nabc\n\n</think>\n\nC\n"}, "\nabc\n", "\nC\n", ""},
        {"no separators", false, false, true, {"<think>abc</think>C"}, "abc", "C", ""},
        {"empty thought", false, false, true, {"<think>\n\n</think>\nC"}, "", "C", ""},
        {"no newline between closer and end", false, false, true, {"<think>\nr\n</think>"}, "r", "", ""},
        {"a second thought is text", false, false, true, {"<think>\na\n</think>\nb<think>\nc\n</think>\nd"}, "a", "b<think>\nc\n</think>\nd", ""},
        {"newlines that are not the separator", false, false, true, {"<think>\na\n", "b\n", "\n", "</think>\nc"}, "a\nb\n", "c", ""},
        {"half a closer that is not one", false, false, true, {"<think>\na\n</", "b</think>"}, "a\n</b", "", ""},
        {"closer split everywhere", false, false, true, {"<think>", "\nr\n<", "/", "thi", "nk", ">\nc"}, "r", "c", ""},
        {"force", true, false, true, {"abc\n", "</think>\nC"}, "abc", "C", ""},
        {"force, an opener stays thought", true, false, true, {"<think>x</think>y"}, "<think>x", "y", ""},
        {"force, a first newline stays", true, false, true, {"\nx\n</think>y"}, "\nx", "y", ""},
        {"force, cut off", true, false, true, {"abc", "def\n"}, "abcdef\n", "", ""},
        {"force, a tool call inside stays thought", true, true, true, {"a" + call + "b\n</think>\nc"}, "a" + call + "b", "c", ""},
        {"tool call", false, true, true, {"Let me\n", "<tool_call>\n", "{\"name\": \"f\", \"arguments\": {}}", "\n</tool_call>"}, "", "Let me", call},
        {"tool call after two newlines", false, true, true, {"Let me\n\n" + call}, "", "Let me\n", call},
        {"tool call, no newline before", false, true, true, {"Let me" + call}, "", "Let me", call},
        {"tool call first", false, true, true, {call}, "", "", call},
        {"tool call after a newline only", false, true, true, {"\n", call}, "", "", call},
        {"tool marker split", false, true, true, {"x\n<tool", "_call>\n{\"name\": \"f\", \"arguments\": {}}\n</tool_call>"}, "", "x", call},
        {"not the tool marker", false, true, true, {"x\n<to", "ast"}, "", "x\n<toast", ""},
        {"tool marker cut off", false, true, true, {"x\n<to"}, "", "x\n<to", ""},
        {"newline held until the next text", false, true, true, {"x\n", "y\n", "z"}, "", "x\ny\nz", ""},
        {"newline at the end kept", false, true, true, {"x\n"}, "", "x\n", ""},
        {"text after the call is held", false, true, true, {call + "\nafter"}, "", "", call + "\nafter"},
        {"thought then call, the layout", false, true, true, {"<think>\nhm\n</think>\n\n" + call}, "hm", "", call},
        {"thought then call, one newline", false, true, true, {"<think>\nhm\n</think>\n" + call}, "hm", "", call},
        {"thought, text, call", false, true, true, {"<think>\nhm\n</think>\nT\n" + call}, "hm", "T", call},
        {"a call in the thought stays thought", false, true, true, {"<think>\n" + call + "\n</think>\nx"}, call, "x", ""},
        {"no tools: the call is text", false, false, true, {"x\n", call}, "", "x\n" + call, ""},
        {"separate_reasoning false", false, false, false, {"<think>\na\n</think>\nb"}, "", "<think>\na\n</think>\nb", ""},
        {"separate_reasoning false, force", true, false, false, {"\nanswer\n"}, "", "\nanswer\n", ""},
        {"separate_reasoning false, tools", false, true, false, {"answer\n", call}, "", "answer", call},
    };
}

int main(int argc, char** argv) {
    GLM_CHECK(argc >= 3, "usage: ripple_check <model dir> <ref dir>");
    std::string ref = argv[2];
    auto t0 = std::chrono::steady_clock::now();
    Tokenizer tok = Tokenizer::load(argv[1]);
    printf("tokenizer loaded in %.0f ms, %d ids\n",
           std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(), tok.vocab_size());
    std::vector<Tally> all;
    size_t tokens_compared = 0;

    // ---- token pieces, decode, streaming
    {
        Json d = Json::parse(read_file(ref + "/decode.json"));
        Tally tp{"token_text (all ids)"}, de{"decode"}, st{"stream decode (streams)"};
        const auto& pieces = d["pieces"].arr;
        GLM_CHECK((int)pieces.size() == tok.vocab_size(), "vocab %d vs %zu pieces", tok.vocab_size(), pieces.size());
        for (size_t i = 0; i < pieces.size(); i++) {
            if (tok.token_text((int)i) == pieces[i].s()) { tp.ok++; continue; }
            if (tp.report()) printf("token_text(%zu): got [%s] want [%s]\n", i, esc(tok.token_text((int)i)).c_str(), esc(pieces[i].s()).c_str());
        }
        for (auto& c : d["decode"].arr) {
            std::string got = tok.decode(ids_of(c["ids"]), c["skip"].b);
            if (got == c["text"].s()) { de.ok++; continue; }
            if (de.report()) printf("decode skip=%d: %s\n", (int)c["skip"].b, first_diff(got, c["text"].s()).c_str());
        }
        for (auto& c : d["stream"].arr) {
            auto ids = ids_of(c["ids"]);
            StreamDecoder sd(tok, (size_t)c["max_stop"].i());
            bool same = true;
            for (size_t k = 0; k < ids.size(); k++) {
                std::string got = sd.push(ids[k], k + 1 == ids.size());
                if (got != c["chunks"][k].s()) {
                    same = false;
                    if (st.bad < 5) printf("stream token %zu/%zu (id %d): %s\n", k, ids.size(), ids[k], first_diff(got, c["chunks"][k].s()).c_str());
                    break;
                }
            }
            if (same) st.ok++;
            else st.bad++;
        }
        all.push_back(tp);
        all.push_back(de);
        all.push_back(st);
    }

    // ---- encode
    {
        Json e = Json::parse(read_file(ref + "/encode.json"));
        Tally en{"encode"}, pr{"encode: tokenizer probes"}, sy{"encode: special tokens, synthetic"}, ad{"added tokens: ids, id_of, is_special"};
        for (auto& c : e["cases"].arr) {
            auto want = ids_of(c["ids"]);
            auto got = tok.encode(c["text"].s(), c["special"].b);
            tokens_compared += want.size();
            const std::string& name = c["name"].s();
            Tally& t = name.compare(0, 16, "tokenizer_probe:") == 0 ? pr : name.compare(0, 10, "synthetic:") == 0 ? sy : en;
            if (got == want) { t.ok++; continue; }
            if (t.report()) {
                size_t i = 0;
                while (i < got.size() && i < want.size() && got[i] == want[i]) i++;
                printf("encode %s special=%d: %zu vs %zu ids, first difference at %zu: got %s want %s\n",
                       name.c_str(), (int)c["special"].b, got.size(), want.size(), i, ids_str(got, i).c_str(),
                       ids_str(want, i).c_str());
                if (c["text"].s().size() < 200) printf("   text [%s]\n", esc(c["text"].s()).c_str());
            }
        }
        for (auto& a : e["added"].arr) {
            int id = (int)a["id"].i();
            auto one = tok.encode(a["content"].s());
            bool same = tok.id_of(a["content"].s()) == id && tok.is_special(id) == a["special"].b && one.size() == 1 && one[0] == id &&
                        tok.token_text(id) == a["content"].s();
            (same ? ad.ok : ad.bad)++;
            if (!same) printf("added token %s (%d) differs\n", a["content"].s().c_str(), id);
        }
        all.push_back(en);
        all.push_back(pr);
        all.push_back(sy);
        all.push_back(ad);
        const std::string& big = e["big"]["text"].s();
        auto want = ids_of(e["big"]["ids"]);
        double best = 1e30;
        std::vector<int> got;
        for (int r = 0; r < 5; r++) {
            auto a = std::chrono::steady_clock::now();
            got = tok.encode(big);
            best = std::min(best, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count());
        }
        tokens_compared += want.size();
        printf("long mixed text: %zu tokens (%s), %.1f ms (HF %.1f ms)\n", got.size(), got == want ? "identical" : "DIFFERENT",
               best, e["big"]["py_ms"].f());
        Tally bg{"encode long mixed text"};
        (got == want ? bg.ok : bg.bad)++;
        all.push_back(bg);
    }

    // ---- code point sweep
    {
        Json s = Json::parse(read_file(ref + "/sweep.json"));
        Tally sw{"code point sweep (blocks)"};
        for (auto& b : s["blocks"].arr) {
            uint32_t lo = (uint32_t)b["lo"].i(), hi = (uint32_t)b["hi"].i();
            std::string text;
            for (uint32_t c = lo; c <= hi; c++)
                if (c < 0xD800 || c > 0xDFFF) text += probe(c);
            auto ids = tok.encode(text, false);
            tokens_compared += ids.size();
            char h[32];
            snprintf(h, sizeof h, "%016llx", (unsigned long long)fnv(ids));
            if (h == b["hash"].s() && (int64_t)ids.size() == b["n"].i()) { sw.ok++; continue; }
            if (sw.report()) printf("sweep block U+%04X..U+%04X: %zu ids vs %lld\n", lo, hi, ids.size(), (long long)b["n"].i());
        }
        all.push_back(sw);
    }

    // ---- Split(Removed, invert) with gaps to drop, and the shapes the port must refuse
    {
        Tally ti{"tiny tokenizer (gaps dropped)"}, rf{"tokenizers refused"};
        Tokenizer tiny = Tokenizer::load(ref + "/tiny");
        Json tj = Json::parse(read_file(ref + "/tiny.json"));
        for (auto& c : tj["cases"].arr) {
            auto want = ids_of(c["ids"]);
            auto got = tiny.encode(c["text"].s());
            if (got == want) { ti.ok++; continue; }
            if (ti.report()) printf("tiny [%s]: got %s want %s\n", esc(c["text"].s()).c_str(), ids_str(got, 0).c_str(), ids_str(want, 0).c_str());
        }
        for (const char* dir : {"/tiny_removed", "/tiny_isolated_invert"}) {
            bool refused = false;
            try {
                Tokenizer::load(ref + dir);
            } catch (const std::runtime_error&) {
                refused = true;
            }
            (refused ? rf.ok : rf.bad)++;
            if (!refused) printf("%s: loaded, should be refused\n", dir);
        }
        all.push_back(ti);
        all.push_back(rf);
    }

    // ---- prompts
    {
        Json c = Json::parse(read_file(ref + "/chat.json"));
        Tally rr{"render_chat_ripple (template)"}, sv{"render_prompt_ripple (serve)"}, si{"prompt token ids"};
        for (auto& k : c["render"].arr) {
            const Json* tools = k["tools"].kind == Json::Null ? nullptr : &k["tools"];
            bool want_err = k["text"].kind == Json::Null;
            std::string got, err;
            try {
                got = render_chat_ripple(k["messages"], tools, k["vars"]);
            } catch (const std::runtime_error& e) {
                err = e.what();
            }
            bool same = want_err ? !err.empty() : err.empty() && got == k["text"].s();
            if (same) { rr.ok++; continue; }
            if (rr.report())
                printf("render %s: %s\n", k["name"].s().c_str(),
                       want_err ? ("expected an error, got [" + esc(got) + "]").c_str() : !err.empty() ? err.c_str() : first_diff(got, k["text"].s()).c_str());
        }
        for (auto& k : c["serve"].arr) {
            const Json& body = k["body"];
            bool want_err = k["text"].kind == Json::Null;
            ChatPrompt p;
            std::string err;
            try {
                p = render_prompt_ripple(body["messages"], body.get("tools"), body.get("chat_template_kwargs"), body.get("reasoning_effort"));
            } catch (const std::runtime_error& e) {
                err = e.what();
            }
            if (want_err) {
                if (!err.empty()) { sv.ok++; si.ok++; continue; }
                sv.bad++;
                printf("serve %s: expected an error\n", k["name"].s().c_str());
                continue;
            }
            bool same = err.empty() && p.text == k["text"].s() && p.thinking == k["thinking"].b && p.reply_in_thought == k["reply_in_thought"].b;
            if (same) sv.ok++;
            else if (sv.report())
                printf("serve %s: %s (thinking %d/%d, in thought %d/%d)\n", k["name"].s().c_str(), !err.empty() ? err.c_str() : first_diff(p.text, k["text"].s()).c_str(),
                       (int)p.thinking, (int)k["thinking"].b, (int)p.reply_in_thought, (int)k["reply_in_thought"].b);
            auto want = ids_of(k["ids"]);
            tokens_compared += want.size();
            if (err.empty() && tok.encode(p.text) == want) si.ok++;
            else if (si.report()) printf("serve %s: token ids differ\n", k["name"].s().c_str());
        }
        all.push_back(rr);
        all.push_back(sv);
        all.push_back(si);
    }

    // ---- the output parsers
    auto run = [](RippleSplitter& s, const std::vector<std::string>& chunks, std::string& r, std::string& t) {
        for (size_t k = 0; k <= chunks.size(); k++)
            for (auto& [is_r, x] : k < chunks.size() ? s.feed(chunks[k]) : s.feed("", true)) (is_r ? r : t) += x;
    };
    {
        Tally pc{"parse_tool_calls_ripple (by hand)"}, sp{"RippleSplitter (by hand)"}, by{"RippleSplitter, byte by byte"}, fz{"RippleSplitter, against the detector"};
        Json tools = Json::make_arr();
        for (auto& c : call_cases()) {
            auto [rest, calls] = parse_tool_calls_ripple(c.text, &tools);
            bool same = rest == c.rest && calls.size() == c.calls.size();
            for (size_t i = 0; same && i < c.calls.size(); i++) {
                const Json& call = calls[i];
                same = call["function"]["name"].s() == c.calls[i].first && call["function"]["arguments"].s() == c.calls[i].second &&
                       call["type"].s() == "function" && call["id"].s().size() == 29 && call["id"].s().compare(0, 5, "call_") == 0;
            }
            if (same) { pc.ok++; continue; }
            if (pc.report()) {
                printf("tool calls [%s]: rest [%s], %zu calls", esc(c.text).c_str(), esc(rest).c_str(), calls.size());
                for (size_t i = 0; i < calls.size(); i++) printf(" %s", dump(calls[i]["function"]).c_str());
                printf("\n");
            }
        }
        auto [r0, c0] = parse_tool_calls_ripple("<tool_call>{\"name\":\"f\"}</tool_call>", nullptr);   // tools off: no parsing
        (r0 == "<tool_call>{\"name\":\"f\"}</tool_call>" && c0.size() == 0 ? pc.ok : pc.bad)++;
        for (auto& c : split_cases()) {
            RippleSplitter s(c.force, c.tools, c.separate);
            std::string r, t;
            run(s, c.chunks, r, t);
            if (r == c.reasoning && t == c.content && s.held == c.held) sp.ok++;
            else if (sp.report())
                printf("splitter %s: reasoning [%s] want [%s], content [%s] want [%s], held [%s] want [%s]\n", c.name, esc(r).c_str(), esc(c.reasoning).c_str(),
                       esc(t).c_str(), esc(c.content).c_str(), esc(s.held).c_str(), esc(c.held).c_str());
            // by the byte instead of by the token: partial markers everywhere
            std::string whole;
            for (auto& x : c.chunks) whole += x;
            std::vector<std::string> bytes;
            for (char ch : whole) bytes.emplace_back(1, ch);
            RippleSplitter s2(c.force, c.tools, c.separate);
            std::string r2, t2;
            run(s2, bytes, r2, t2);
            if (r2 == c.reasoning && t2 == c.content && s2.held == c.held) by.ok++;
            else if (by.report())
                printf("splitter by byte %s: reasoning [%s] want [%s], content [%s] want [%s], held [%s] want [%s]\n", c.name, esc(r2).c_str(), esc(c.reasoning).c_str(),
                       esc(t2).c_str(), esc(c.content).c_str(), esc(s2.held).c_str(), esc(c.held).c_str());
        }
        // random replies and chunkings against the transcribed detector; the tool-call side checked for not depending on
        // the chunking (held plus content, a call anywhere in the text)
        Tally ch{"RippleSplitter tools, any chunking"};
        std::mt19937 rng(5);
        Json sj = Json::parse(read_file(ref + "/split.json"));
        for (auto& c : sj["cases"].arr) {
            std::vector<std::string> chunks;
            std::string whole;
            for (auto& x : c["chunks"].arr) { chunks.push_back(x.s()); whole += x.s(); }
            RippleSplitter s(c["force"].b, false);
            std::string r, t;
            run(s, chunks, r, t);
            if (r == c["reasoning"].s() && t == c["content"].s()) fz.ok++;
            else if (fz.report())
                printf("splitter [%s]: reasoning [%s] want [%s], content [%s] want [%s]\n", esc(whole).c_str(), esc(r).c_str(), esc(c["reasoning"].s()).c_str(),
                       esc(t).c_str(), esc(c["content"].s()).c_str());
            RippleSplitter one(c["force"].b, true);
            std::string r1, t1;
            run(one, {whole}, r1, t1);
            std::vector<std::string> cut;
            for (size_t p = 0; p < whole.size();) {
                size_t n = 1 + rng() % 5;
                cut.push_back(whole.substr(p, n));
                p += n;
            }
            RippleSplitter many(c["force"].b, true);
            std::string r2, t2;
            run(many, cut, r2, t2);
            if (r1 == r2 && t1 == t2 && one.held == many.held) ch.ok++;
            else if (ch.report()) printf("splitter tools [%s]: content [%s] vs [%s]\n", esc(whole).c_str(), esc(t1).c_str(), esc(t2).c_str());
        }
        all.push_back(pc);
        all.push_back(sp);
        all.push_back(by);
        all.push_back(fz);
        all.push_back(ch);
    }

    // ---- ripple_format.py's promise: a reply sent back unchanged re-renders to the tokens the model generated. From ids:
    // StreamDecoder, RippleSplitter, parse_tool_calls_ripple, the template.
    {
        Tally rt{"reply through parsers and back"};
        const std::string c1 = "<tool_call>\n{\"name\": \"f\", \"arguments\": {\"a\": 1, \"s\": \"x\\ny \xC3\xA9\xF0\x9F\x99\x82\"}}\n</tool_call>";
        const std::string c2 = "<tool_call>\n{\"name\": \"g\", \"arguments\": {}}\n</tool_call>";
        Json tools = Json::make_arr();
        for (const std::string& reply : {
                 std::string("Hello"),
                 std::string("a\n\n  b\n"),
                 std::string("<think>\nR\n</think>\nHello"),
                 std::string("<think>\nline one\n\nline two\n\n</think>\n\nC \xE4\xBD\xA0\xE5\xA5\xBD\xF0\x9F\x99\x82"),
                 "<think>\nR\n</think>\n\n" + c1,
                 "<think>\nR\n</think>\nT\n" + c1,
                 "<think>\nR\n</think>\nT\n\n" + c1 + "\n" + c2,
                 "T\n" + c1 + "\n" + c2,
                 c1,
             }) {
            auto ids = tok.encode(reply);
            for (size_t max_stop : {0, 40}) {
                StreamDecoder sd(tok, max_stop);
                RippleSplitter sp(false, true);
                std::string r, t;
                for (int id : ids)
                    for (auto& [is_r, x] : sp.feed(sd.push(id))) (is_r ? r : t) += x;
                for (auto& [is_r, x] : sp.feed(sd.flush(), true)) (is_r ? r : t) += x;
                Json m = Json::make_obj();
                m.set("role", Json::make_str("assistant"));
                m.set("content", Json::make_str(t));
                if (!r.empty()) m.set("reasoning_content", Json::make_str(r));
                if (!sp.held.empty()) {
                    auto [rest, calls] = parse_tool_calls_ripple(sp.held, &tools);
                    if (!rest.empty() || !calls.size()) { rt.bad++; printf("reply [%s]: rest [%s]\n", esc(reply).c_str(), esc(rest).c_str()); continue; }
                    m.set("tool_calls", calls);
                }
                Json msgs = Json::make_arr();
                Json u = Json::make_obj();
                u.set("role", Json::make_str("user"));
                u.set("content", Json::make_str("u"));
                msgs.push(u);
                msgs.push(m);
                std::string got = render_prompt_ripple(msgs, nullptr, nullptr, nullptr).text;
                std::string want = "<|im_start|>user\nu<|im_end|>\n<|im_start|>assistant\n" + reply + "<|im_end|>\n<|im_start|>assistant\n";
                if (got == want) rt.ok++;
                else if (rt.report()) printf("reply [%s]: %s\n", esc(reply).c_str(), first_diff(got, want).c_str());
            }
        }
        all.push_back(rt);
    }

    int bad = 0;
    printf("\n");
    for (auto& t : all) {
        t.print();
        bad += t.bad;
    }
    printf("%zu tokens compared with HF\n", tokens_compared);
    printf("%s\n", bad ? "MISMATCHES" : "all identical");
    return bad ? 1 : 0;
}
