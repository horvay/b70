// Gemma 4's tokenizer, streaming detokenizer, chat template and output parsers, against tests/gemma_ref.py's outputs (HF
// tokenizers, HF's Jinja rendering) and, for the parsers, cases derived by hand from SGLang's source. CPU only.
//   gemma_check <model dir> <ref dir>        ref dir: where gemma_ref.py wrote chat.json, encode.json, sweep.json, decode.json
// Every case must match byte for byte; prints the first mismatches of each kind and a summary.
#include <chrono>
#include <cstdio>
#include <fstream>
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
    void print() const { printf("%-34s %8d identical, %d different\n", what, ok, bad); }
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

// Each worked through Gemma4Detector's Python by hand. The ones marked SGLang-quirk keep what that code does.
static std::vector<CallCase> call_cases() {
#define Q "<|\"|>"
#define CALL(args) "<|tool_call>call:f{" args "}<tool_call|>"
    return {
        {CALL("a:" Q "x" Q), "", {{"f", "{\"a\": \"x\"}"}}},
        {"Hello " "<|tool_call>call:f{}<tool_call|>" " after", "Hello ", {{"f", "{}"}}},                    // after: dropped
        {"A<|tool_call>call:f{a:1}<tool_call|>B<|tool_call>call:g{b:2.5}<tool_call|>C", "A", {{"f", "{\"a\": 1}"}, {"g", "{\"b\": 2.5}"}}},
        {"x<|tool_call>call:f{a:1}", "x<|tool_call>call:f{a:1}", {}},                                      // no end marker
        {"<|tool_call>call:f{a:1}<|tool_call>call:g{b:2}<tool_call|>", "", {{"f", "{\"a\": 1}"}}},
        {"<|tool_call>foo{a:1}<tool_call|>", "<|tool_call>foo{a:1}<tool_call|>", {}},                      // not call:
        {"<|tool_call> call:f{}<tool_call|>", "<|tool_call> call:f{}<tool_call|>", {}},                    // no whitespace allowed
        {"<|tool_call>call:f<tool_call|>", "<|tool_call>call:f<tool_call|>", {}},                          // no brace
        {"<|tool_call>foo<tool_call|>x<|tool_call>call:g{}<tool_call|>", "", {{"g", "{}"}}},               // the bad one skipped, content before it
        {"no marker here <tool_call|>", "no marker here <tool_call|>", {}},
        {"<|tool_call>call:do.thing-x{}<tool_call|>", "", {{"do.thing-x", "{}"}}},                          // any name, tools or not
        {"<|tool_call>call:{}<tool_call|>", "", {{"", "{}"}}},
        {"<|tool_call>call:f{a:{b:1}<tool_call|>", "", {{"f", "{\"a\": {\"b\": 1}}"}}},                     // unbalanced: all of it
        {"<|tool_call>call:f{a:" Q "xyz<tool_call|>", "", {{"f", "{\"a\": \"xyz\"}"}}},                     // unterminated string
        // bare values: Python's int() and float() decide; null is JSON null; SGLang-quirk: 1e5 stays a string
        {CALL("a:true,b:false,c:null,d:5,e:-3,f:+7,g:007,h:1_000,i:2.5,j:1.,k:.5,l:1e5,m:1.5e3,n:abc def,o:0x10,p:-0,q:1.2.3,r:1_.5,s:1.0e999,t:123456789012345678901234567890"),
         "", {{"f", "{\"a\": true, \"b\": false, \"c\": null, \"d\": 5, \"e\": -3, \"f\": 7, \"g\": 7, \"h\": 1000, \"i\": 2.5, \"j\": 1.0, \"k\": 0.5, \"l\": \"1e5\", "
                    "\"m\": 1500.0, \"n\": \"abc def\", \"o\": \"0x10\", \"p\": 0, \"q\": \"1.2.3\", \"r\": \"1_.5\", \"s\": Infinity, \"t\": 123456789012345678901234567890}"}}},
        {CALL("a:1__0,b:_1,c:1_,d:1._5,e:1.5_,f:.,g:-,h:1.5e,i:1.5e1_0"), "",
         {{"f", "{\"a\": \"1__0\", \"b\": \"_1\", \"c\": \"1_\", \"d\": \"1._5\", \"e\": \"1.5_\", \"f\": \".\", \"g\": \"-\", \"h\": \"1.5e\", \"i\": 15000000000.0}"}}},
        {CALL("a : 1 ,\n b:\n\t" Q "x" Q " , c: [ 1 , 2 ]"), "", {{"f", "{\"a\": 1, \"b\": \"x\", \"c\": [1, 2]}"}}},
        {CALL("a:1,\rb:2,c:\r5"), "", {{"f", "{\"a\": 1, \"b\": 2, \"c\": 5}"}}},                           // \r only falls to strip()
        {CALL("a:1,b:2,a:3"), "", {{"f", "{\"a\": 3, \"b\": 2}"}}},                                         // a dict: first position, last value
        {CALL("a:"), "", {{"f", "{\"a\": \"\"}"}}},
        {CALL("a:  "), "", {{"f", "{\"a\": \"\"}"}}},
        {CALL("a"), "", {{"f", "{}"}}},
        {CALL("a:1,b"), "", {{"f", "{\"a\": 1}"}}},
        {CALL("  \n "), "", {{"f", "{}"}}},
        {CALL("o:{x:{y:[1,{z:" Q "}" Q "}]}},l:[[1,2],[" Q "a" Q ",[3]]]"), "",
         {{"f", "{\"o\": {\"x\": {\"y\": [1, {\"z\": \"}\"}]}}, \"l\": [[1, 2], [\"a\", [3]]]}"}}},
        {CALL("l:[a}b]"), "", {{"f", "{\"l\": []}"}}},                                                      // SGLang-quirk: } ends the args, then an open [
        {CALL("a:1]b:2"), "", {{"f", "{\"a\": 1, \"]b\": 2}"}}},                                            // SGLang-quirk: ] stops a bare value
        {CALL("l:[" Q "abc]"), "", {{"f", "{\"l\": [\"abc]\"]}"}}},                                         // unterminated, the last character cut
        // SGLang's array parser would loop for ever on this one (a ] left at its level); here it ends
        {CALL("l:[[" Q "]x],y" Q "]]"), "", {{"f", "{\"l\": [[\"\"], \"x\", \"y<|\\\"|>\"]}"}}},
        {CALL("\xC3\x9Cn\xC3\xAF:1,\xE3\x80\x80k\xE3\x80\x80:2"), "", {{"f", "{\"\xC3\x9Cn\xC3\xAF\": 1, \"k\": 2}"}}},   // str.strip() takes U+3000
        {CALL("a:" Q "\xC3\xA9\xE4\xBD\xA0\xF0\x9F\x98\x80" Q), "", {{"f", "{\"a\": \"\xC3\xA9\xE4\xBD\xA0\xF0\x9F\x98\x80\"}"}}},
        {CALL("a:" Q "q\"\n\t\\" Q), "", {{"f", "{\"a\": \"q\\\"\\n\\t\\\\\"}"}}},
        {CALL("a:" Q Q), "", {{"f", "{\"a\": \"\"}"}}},
        // a string that lost its opening quote: whole, not cut at its commas and brackets, and the next arguments intact
        {CALL("content:---\nname: Tala\naliases: []\npersonality: \"Proud, feminine.\"\n---\n- Sharp, always.\n" Q ",i:" Q "Update." Q ",path:" Q "d/t.md" Q), "",
         {{"f", "{\"content\": \"---\\nname: Tala\\naliases: []\\npersonality: \\\"Proud, feminine.\\\"\\n---\\n- Sharp, always.\\n\", \"i\": \"Update.\", \"path\": \"d/t.md\"}"}}},
        {CALL("a:1,b:true,c:one, two" Q), "", {{"f", "{\"a\": 1, \"b\": true, \"c\": \"one, two\"}"}}},        // numbers before it stand
        {CALL("a:abc,b:" Q "x" Q), "", {{"f", "{\"a\": \"abc\", \"b\": \"x\"}"}}},                         // that quote opens b
        {CALL("a:abc," Q "x" Q), "", {{"f", "{\"a\": \"abc\"}"}}},                                             // no key before it
    };
#undef CALL
#undef Q
}

// ------------------------------------------------------------------------------------------- GemmaSplitter, by hand

struct SplitCase {
    const char* name;
    bool force, tools, separate;
    std::vector<std::string> chunks;
    std::string reasoning, content, held;
};

static std::vector<SplitCase> split_cases() {
    const std::string to = "<|channel>thought\n", tc = "<channel|>", call = "<|tool_call>call:f{}<tool_call|>";
    return {
        {"thinking on, model opens", true, false, true, {"<|channel>", "thought", "\n", "I", " think", "\n", tc, "Hello", " there"}, "I think\n", "Hello there", ""},
        {"thinking on, one chunk", true, false, true, {to + "r\n" + tc + "c"}, "r\n", "c", ""},
        {"prompt opened it", true, false, true, {"abc", tc, "x"}, "abc", "x", ""},
        {"cut off in the thought", true, false, true, {"abc", "def"}, "abcdef", "", ""},
        {"cut off in a marker", true, false, true, {"abc<chann"}, "abc<chann", "", ""},
        {"text and half a closer", true, false, true, {"abc<chann", "el|>x"}, "abc", "x", ""},
        {"text and half an opener", true, false, true, {"abc<|chan", "nel>thought\nr"}, "abcr", "", ""},
        {"cut off in the opener", true, false, true, {"<|channel>tho"}, "<|channel>tho", "", ""},
        {"thinking on, never opens: all thought", true, false, true, {"Just", " an", " answer"}, "Just an answer", "", ""},
        {"thinking off", false, false, true, {"Hello", " <", "b>"}, "", "Hello <b>", ""},
        {"thinking off, model opens", false, false, true, {"<|channel>", "thought", "\n", "a", tc, "b"}, "a", "b", ""},
        {"thinking off, split marker", false, false, true, {"<", "|channel>thought\nr" + tc + "c"}, "r", "c", ""},
        {"held prefix that is not a marker", false, false, true, {"<|chan", "ge"}, "", "<|change", ""},
        {"second thought is text", true, false, true, {to, "r", tc, "x", to, "y"}, "r", "x" + to + "y", ""},
        {"tool call", false, true, true, {"Let me", " roll", "<|tool_call>", "call:f{", "}", "<tool_call|>"}, "", "Let me roll", call},
        {"tool call after thought", true, true, true, {to, "hm", tc, "<|tool_call>", "call:f{}<tool_call|>"}, "hm", "", call},
        {"tool marker split", false, true, true, {"x<|tool", "_call>call:f{}<tool_call|>"}, "", "x", call},
        {"not the tool marker", false, true, true, {"<|to", "ast"}, "", "<|toast", ""},
        {"tool marker cut off", false, true, true, {"<|to"}, "", "<|to", ""},
        {"no tools: the call is text", false, false, true, {"<|tool_call>", "call:f{}", "<tool_call|>"}, "", call, ""},
        {"a tool call inside the thought stays thought", true, true, true, {"a" + call + "b", tc, "c"}, "a" + call + "b", "c", ""},
        {"separate_reasoning false", true, false, false, {to, "a", tc, "b"}, "", to + "a" + tc + "b", ""},
        {"separate_reasoning false, tools", true, true, false, {to, "a", tc, "b", call}, "", to + "a" + tc + "b", call},
    };
}

int main(int argc, char** argv) {
    GLM_CHECK(argc >= 3, "usage: gemma_check <model dir> <ref dir>");
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

    // ---- a character split across <0xNN> tokens streams whole: every code point outside the vocabulary
    {
        Tally bf{"byte fallback characters streamed"};
        std::vector<char> is_byte(tok.vocab_size(), 0);
        for (int b = 0; b < 256; b++) {
            char name[8];
            snprintf(name, sizeof name, "<0x%02X>", b);
            is_byte[*tok.id_of(name)] = 1;
        }
        size_t multi = 0;
        for (uint32_t c = 0x80; c < 0x110000; c++) {
            if (c >= 0xD800 && c <= 0xDFFF) continue;
            std::string ch;
            json_detail::put_utf8(ch, c);
            std::string text = ch + " ok";
            auto ids = tok.encode(text);
            bool fallback = ids.size() >= 2 && ch.size() >= 2;
            for (size_t k = 0; k < ch.size() && fallback; k++) fallback = k < ids.size() && is_byte[ids[k]];
            if (!fallback) continue;
            multi++;
            StreamDecoder sd(tok);
            std::string out;
            bool early = false;
            for (size_t k = 0; k < ids.size(); k++) {
                std::string piece = sd.push(ids[k]);
                early |= k + 1 < ch.size() && !piece.empty();    // nothing before the character is complete
                out += piece;
            }
            out += sd.flush();
            if (out == text && !early) { bf.ok++; continue; }
            if (bf.report()) printf("byte fallback U+%04X: got [%s]%s\n", c, esc(out).c_str(), early ? " (emitted early)" : "");
        }
        printf("%zu code points fall back to <0xNN> tokens\n", multi);
        all.push_back(bf);
    }

    // ---- encode
    {
        Json e = Json::parse(read_file(ref + "/encode.json"));
        Tally en{"encode"};
        for (auto& c : e["cases"].arr) {
            auto want = ids_of(c["ids"]);
            auto got = tok.encode(c["text"].s(), c["special"].b);
            tokens_compared += want.size();
            if (got == want) { en.ok++; continue; }
            if (en.report()) {
                size_t i = 0;
                while (i < got.size() && i < want.size() && got[i] == want[i]) i++;
                printf("encode %s special=%d: %zu vs %zu ids, first difference at %zu: got %s want %s\n",
                       c["name"].s().c_str(), (int)c["special"].b, got.size(), want.size(), i, ids_str(got, i).c_str(),
                       ids_str(want, i).c_str());
                if (c["text"].s().size() < 200) printf("   text [%s]\n", esc(c["text"].s()).c_str());
            }
        }
        all.push_back(en);
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
        printf("big text: %zu tokens (%s), %.1f ms (HF %.1f ms)\n", got.size(), got == want ? "identical" : "DIFFERENT",
               best, e["big"]["py_ms"].f());
        Tally bg{"encode big text"};
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
            auto ids = tok.encode(text);
            tokens_compared += ids.size();
            char h[32];
            snprintf(h, sizeof h, "%016llx", (unsigned long long)fnv(ids));
            if (h == b["hash"].s() && (int64_t)ids.size() == b["n"].i()) { sw.ok++; continue; }
            if (sw.report()) printf("sweep block U+%04X..U+%04X: %zu ids vs %lld\n", lo, hi, ids.size(), (long long)b["n"].i());
        }
        all.push_back(sw);
    }

    // ---- prompts
    {
        Json c = Json::parse(read_file(ref + "/chat.json"));
        Tally rr{"render_chat_gemma (template)"}, sv{"render_prompt_gemma (serve)"}, si{"prompt token ids"}, rt{"tool calls round trip"};
        for (auto& k : c["render"].arr) {
            const Json* tools = k["tools"].kind == Json::Null ? nullptr : &k["tools"];
            bool want_err = k["text"].kind == Json::Null;
            std::string got, err;
            try {
                got = render_chat_gemma(k["messages"], tools, k["vars"]);
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
                p = render_prompt_gemma(body["messages"], body.get("tools"), body.get("chat_template_kwargs"), body.get("reasoning_effort"));
            } catch (const std::runtime_error& e) {
                err = e.what();
            }
            if (want_err) {
                if (!err.empty()) { sv.ok++; si.ok++; continue; }
                sv.bad++;
                printf("serve %s: expected an error\n", k["name"].s().c_str());
                continue;
            }
            bool same = err.empty() && p.text == k["text"].s() && p.thinking == k["thinking"].b && p.reply_in_thought == k["thinking"].b;
            if (same) sv.ok++;
            else if (sv.report())
                printf("serve %s: %s (thinking %d/%d)\n", k["name"].s().c_str(), !err.empty() ? err.c_str() : first_diff(p.text, k["text"].s()).c_str(),
                       (int)p.thinking, (int)k["thinking"].b);
            auto want = ids_of(k["ids"]);
            tokens_compared += want.size();
            if (err.empty() && tok.encode(p.text) == want) si.ok++;
            else if (si.report()) printf("serve %s: token ids differ\n", k["name"].s().c_str());
        }
        Json tools = Json::make_arr();
        for (auto& k : c["roundtrip"].arr) {
            auto [rest, calls] = parse_tool_calls_gemma(k["text"].s(), &tools);
            std::string got = calls.size() == 1 ? calls[0]["function"]["arguments"].s() : "(no call)";
            std::string want = dump(k["args"]);
            bool name_ok = calls.size() == 1 && calls[0]["function"]["name"].s() == "fn_name";
            if (rest.empty() && name_ok && got == want) { rt.ok++; continue; }
            if (rt.report()) printf("round trip [%s]: %s\n", esc(k["text"].s()).c_str(), first_diff(got, want).c_str());
        }
        all.push_back(rr);
        all.push_back(sv);
        all.push_back(si);
        all.push_back(rt);
    }

    // ---- the output parsers
    {
        Tally pc{"parse_tool_calls_gemma (by hand)"}, sp{"GemmaSplitter (by hand)"}, cu{"GemmaSplitter, any chunking"}, tf{"reply through the tokenizer"};
        Json tools = Json::make_arr();
        for (auto& c : call_cases()) {
            auto [rest, calls] = parse_tool_calls_gemma(c.text, &tools);
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
        auto [r0, c0] = parse_tool_calls_gemma("<|tool_call>call:f{}<tool_call|>", nullptr);   // tools off: no parsing
        (r0 == "<|tool_call>call:f{}<tool_call|>" && c0.size() == 0 ? pc.ok : pc.bad)++;
        auto run = [](GemmaSplitter& s, const std::vector<std::string>& chunks, std::string& r, std::string& t) {
            for (size_t k = 0; k <= chunks.size(); k++)
                for (auto& [is_r, x] : k < chunks.size() ? s.feed(chunks[k]) : s.feed("", true)) (is_r ? r : t) += x;
        };
        for (auto& c : split_cases()) {
            GemmaSplitter s(c.force, c.tools, c.separate);
            std::string r, t;
            run(s, c.chunks, r, t);
            if (r == c.reasoning && t == c.content && s.held == c.held) sp.ok++;
            else if (sp.report())
                printf("splitter %s: reasoning [%s] want [%s], content [%s] want [%s], held [%s] want [%s]\n", c.name, esc(r).c_str(), esc(c.reasoning).c_str(),
                       esc(t).c_str(), esc(c.content).c_str(), esc(s.held).c_str(), esc(c.held).c_str());
            // by the byte instead of by the token: partial markers everywhere, for cases that have one thought or none
            std::string whole;
            for (auto& x : c.chunks) whole += x;
            std::vector<std::string> bytes;
            for (char ch : whole) bytes.emplace_back(1, ch);
            if (std::string(c.name) == "second thought is text") continue;   // the reply's split depends on the chunks here
            GemmaSplitter s2(c.force, c.tools, c.separate);
            std::string r2, t2;
            run(s2, bytes, r2, t2);
            if (r2 == c.reasoning && t2 == c.content && s2.held == c.held) cu.ok++;
            else if (cu.report())
                printf("splitter by byte %s: reasoning [%s] want [%s], content [%s] want [%s], held [%s] want [%s]\n", c.name, esc(r2).c_str(), esc(c.reasoning).c_str(),
                       esc(t2).c_str(), esc(c.content).c_str(), esc(s2.held).c_str(), esc(c.held).c_str());
        }
        // A reply with characters in byte fallback, from ids: StreamDecoder, GemmaSplitter, parse_tool_calls_gemma
        const std::string thought = "The player \xF0\x9F\xAB\xA0 wants \xE4\xBD\xA0\xE5\xA5\xBD \xF0\xA0\x80\x80 a roll.\n";
        const std::string answer = "Rolling \xF0\x9F\xAB\xA1 now.";
        const std::string reply = "<|channel>thought\n" + thought + "<channel|>" + answer +
                                  "<|tool_call>call:roll{note:<|\"|>\xF0\xA0\x80\x81 luck \xF0\x9F\xAB\xA2<|\"|>,n:20}<tool_call|>";
        auto ids = tok.encode(reply);
        std::string want_calls = "{\"note\": \"\xF0\xA0\x80\x81 luck \xF0\x9F\xAB\xA2\", \"n\": 20}";
        for (size_t max_stop : {0, 40}) {
            StreamDecoder sd(tok, max_stop);
            GemmaSplitter sg(true, true);
            std::string r, t;
            for (int id : ids) {
                for (auto& [is_r, x] : sg.feed(sd.push(id))) (is_r ? r : t) += x;
            }
            for (auto& [is_r, x] : sg.feed(sd.flush(), true)) (is_r ? r : t) += x;
            auto [rest, calls] = parse_tool_calls_gemma(sg.held, &tools);
            bool same = r == thought && t == answer && rest.empty() && calls.size() == 1 && calls[0]["function"]["arguments"].s() == want_calls;
            if (same) tf.ok++;
            else if (tf.report()) printf("reply: reasoning [%s], content [%s], held [%s]\n", esc(r).c_str(), esc(t).c_str(), esc(sg.held).c_str());
        }
        all.push_back(pc);
        all.push_back(sp);
        all.push_back(cu);
        all.push_back(tf);
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
