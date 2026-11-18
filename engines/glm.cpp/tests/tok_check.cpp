// Tokenizer, streaming detokenizer and chat template against the PyTorch stack (tests/tok_ref.py outputs).
//   tok_check <model dir> <ref dir>        ref dir: where tok_ref.py wrote chat.json, encode.json, sweep.json, decode.json
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
    void print() const { printf("%-28s %6d identical, %d different\n", what, ok, bad); }
};

static uint64_t fnv(const std::vector<int>& ids) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (int t : ids)
        for (int k = 0; k < 4; k++) { h ^= (uint32_t)t >> (8 * k) & 0xFF; h *= 0x100000001b3ull; }
    return h;
}

static void put_utf8(std::string& o, uint32_t cp) { json_detail::put_utf8(o, cp); }

static std::string probe(uint32_t c) {
    std::string s;
    put_utf8(s, c);
    return "x" + s + "x 1" + s + "1 " + s + "x !" + s + "! \n" + s + "\n'" + s + "'" + s + s + " " + s + " \t" + s + "\t ";
}

int main(int argc, char** argv) {
    GLM_CHECK(argc >= 3, "usage: tok_check <model dir> <ref dir>");
    std::string ref = argv[2];
    auto t0 = std::chrono::steady_clock::now();
    Tokenizer tok = Tokenizer::load(argv[1]);
    printf("tokenizer loaded in %.0f ms, %d ids\n",
           std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(), tok.vocab_size());
    std::vector<Tally> all;

    // ---- token pieces, decode, streaming
    {
        Json d = Json::parse(read_file(ref + "/decode.json"));
        Tally tp{"token_text (all ids)"}, de{"decode"}, st{"stream decode (chunks)"};
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
        Tally en{"encode"};
        for (auto& c : e["cases"].arr) {
            auto want = ids_of(c["ids"]);
            auto got = tok.encode(c["text"].s(), c["special"].b);
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
        printf("big prompt: %zu tokens (%s), %.1f ms (HF %.1f ms)\n", got.size(), got == want ? "identical" : "DIFFERENT",
               best, e["big"]["py_ms"].f());
        Tally bg{"encode 20k-token prompt"};
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
        Tally rr{"render_chat (template)"}, sv{"render_prompt (serve.py)"}, si{"prompt token ids"};
        for (auto& k : c["render"].arr) {
            const Json* tools = k["tools"].kind == Json::Null ? nullptr : &k["tools"];
            bool want_err = k["text"].kind == Json::Null;
            std::string got, err;
            try {
                got = render_chat(k["messages"], tools, k["vars"], k["template"].s() == "nq");
            } catch (const std::runtime_error& e) {
                err = e.what();
            }
            bool same = want_err ? !err.empty() : err.empty() && got == k["text"].s();
            if (same) { rr.ok++; continue; }
            if (rr.report())
                printf("render %s: %s\n", k["name"].s().c_str(),
                       want_err ? "expected an error" : !err.empty() ? err.c_str() : first_diff(got, k["text"].s()).c_str());
        }
        for (auto& k : c["serve"].arr) {
            const Json& body = k["body"];
            const Json* tools = body.get("tools");
            if (tools && body.get_s("tool_choice", "") == "none") tools = nullptr;
            ServeConfig d;
            d.effort = k["effort"].s();
            d.clear_thinking = k["clear"].b;
            d.nq_prefill_hook = k["template"].s() == "nq";
            bool want_err = k["text"].kind == Json::Null;
            ChatPrompt p;
            std::string err;
            try {
                p = render_prompt(body["messages"], tools, body.get("chat_template_kwargs"), body.get("reasoning_effort"), d);
            } catch (const std::runtime_error& e) {
                err = e.what();
            }
            if (want_err) {
                if (!err.empty()) { sv.ok++; si.ok++; continue; }
                sv.bad++;
                printf("serve %s: expected an error\n", k["name"].s().c_str());
                continue;
            }
            bool same = err.empty() && p.text == k["text"].s() && p.thinking == k["thinking"].b && p.reply_in_thought == k["in_thought"].b;
            if (same) sv.ok++;
            else if (sv.report())
                printf("serve %s: %s (thinking %d/%d, in thought %d/%d)\n", k["name"].s().c_str(),
                       !err.empty() ? err.c_str() : first_diff(p.text, k["text"].s()).c_str(), (int)p.thinking,
                       (int)k["thinking"].b, (int)p.reply_in_thought, (int)k["in_thought"].b);
            auto ids = tok.encode(p.text);
            if (ids == ids_of(k["ids"])) si.ok++;
            else if (si.report()) printf("serve %s: token ids differ\n", k["name"].s().c_str());
        }
        all.push_back(rr);
        all.push_back(sv);
        all.push_back(si);
    }

    // ---- Json round trips through dump (Python json.dumps of what json.loads read)
    {
        Tally js{"json dump"};
        const char* cases[][2] = {
            {"{\"b\": 1, \"a\": [1.0, -0.0, 1e16, 1e15, 0.0001, 1e-05, 123456789012345678901234567890, 1.5e-07, 2.5]}",
             "{\"b\": 1, \"a\": [1.0, -0.0, 1e+16, 1000000000000000.0, 0.0001, 1e-05, 123456789012345678901234567890, 1.5e-07, 2.5]}"},
            {"[\"\\u00e9\\ud83d\\ude00\\n\\t\\\"\\\\\\/\\u0001\\u007f\", true, false, null, -0, 1E2, 0.1, NaN, -Infinity]",
             "[\"\xc3\xa9\xf0\x9f\x98\x80\\n\\t\\\"\\\\/\\u0001\x7f\", true, false, null, 0, 100.0, 0.1, NaN, -Infinity]"},
            {"{\"k\": 1, \"j\": 2, \"k\": 3}", "{\"k\": 3, \"j\": 2}"},
        };
        for (auto& c : cases) {
            std::string got = dump(Json::parse(c[0]));
            if (got == c[1]) js.ok++;
            else if (js.report()) printf("dump %s: %s\n", c[0], first_diff(got, c[1]).c_str());
        }
        std::string a = dump(Json::parse("[\"\\u00e9\\ud83d\\ude00\"]"), true);
        if (a == "[\"\\u00e9\\ud83d\\ude00\"]") js.ok++;
        else if (js.report()) printf("dump ensure_ascii: %s\n", a.c_str());
        const char* bad[] = {"[1,]", "{\"a\" 1}", "\"\x01\"", "\"\\x\"", "01", "1.", "-", "tru", "[1] x", ""};
        for (auto b : bad) {
            if (!Json::try_parse(b)) js.ok++;
            else if (js.report()) printf("parse accepted bad JSON [%s]\n", esc(b).c_str());
        }
        all.push_back(js);
    }

    int bad = 0;
    printf("\n");
    for (auto& t : all) {
        t.print();
        bad += t.bad;
    }
    printf("%s\n", bad ? "MISMATCHES" : "all identical");
    return bad ? 1 : 0;
}
