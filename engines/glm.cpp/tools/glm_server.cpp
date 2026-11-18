// OpenAI-compatible HTTP server for GLM-5.3-Flash on the B70: a drop-in for llm/glm/xpu/serve.py (same endpoints,
// arguments, environment and response shapes), one request at a time on the GPU. A Qwen3.8-Flash-Next folder
// (model_type qwen4_exp) runs on Q4Model instead, with its own chat template and qwen3_coder tool calls, as the
// SGLang path serves it (no --store: every expert is resident; --hot: the VRAM ranking files). A Gemma 4 folder (model_type
// gemma4) runs on G4Model, with its template, reasoning channel and tool calls as SGLang's gemma4 parsers read them. A
// Ripple-65B folder (model_type ricursive) runs on RpModel, with its template and the checkpoint's ripple_format.py parsers.
//   glm_server --model-dir <dir> --store <.bin> [--host 127.0.0.1] [--port 8053] [--cache 49152] [--ram-gb 14]
//              [--served-name glm-5.3-flash] [--lora dir[,dir]] [--lora-scale 1] [--warm-stats <json>] [--no-warm]
//              [--hot a.json,b.json] [--drafter <assistant model dir> (Gemma 4), <DFlash2 dir> (Ripple)] [--parallel N (Gemma 4, Ripple: games at once)]
// Environment: GLM_XPU_RES_BONUS (default 0.1 = quality mode, 0 = exact routing), GLM_XPU_RES_BONUS_RAM,
// GLM_SERVE_REASONING_EFFORT, GLM_SERVE_CLEAR_THINKING (as serve.py). GLM_DUMP_TOKENS=<file>: append each finished
// request's prompt and reply token ids there as a JSON line (training data for a drafter, llm/g4draft). Off unless set;
// never set it on a server that serves someone's own chats.
//   GET  /health, /v1/models
//   POST /v1/chat/completions   messages / tools / chat_template_kwargs / reasoning_effort; stream or not;
//                               reasoning -> reasoning_content, tool calls -> tool_calls; return_progress
//   POST /v1/completions        raw prompt text, stream or not
//   POST /quit                  stop the server (127.0.0.1 only)
#include <filesystem>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdarg>
#include <ctime>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <mutex>
#include <random>
#include <thread>
#include "../src/chat.h"
#include "../src/e0.h"
#include "../src/engine.h"
#include "../src/g4.h"
#include "../src/g4_engine.h"
#include "../src/rp.h"
#include "../src/model.h"
#include "../src/progress.h"
#include "../src/q4.h"
#include "../src/q35v.h"
#include "../src/http.h"
#include "../src/tokenizer.h"

using namespace glm;

namespace {

void log(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
void log(const char* fmt, ...) {
    // one write a line: the engine thread and the progress reporter both log
    char b[1024];
    int n = snprintf(b, sizeof b, " -- glm-server: ");
    va_list ap;
    va_start(ap, fmt);
    n += vsnprintf(b + n, sizeof b - n - 1, fmt, ap);
    va_end(ap);
    n = std::min(n, (int)sizeof b - 2);
    b[n++] = '\n';
    fwrite(b, 1, n, stdout);
    fflush(stdout);
}

int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// One request's progress, for the log: written by the engine thread, read by Server::report
struct Meter {
    int id = 0;
    std::atomic<int> phase{0};                  // 0 waiting, 1 reading its prompt, 2 writing its reply, 3 over
    std::atomic<int64_t> since{0};              // when the phase began (now_ns)
    std::atomic<int> total{0}, read{0}, first{0};   // prompt tokens; read as of the engine's last word, and at the phase's start
    std::atomic<int> tokens{0};                 // reply tokens
    // the reporter's own: the phase, time and count of this request's last line
    int r_phase = 0;
    int64_t r_at = 0;
    double r_n = 0;
};

// GLM_DUMP_TOKENS: one JSON line per finished request, written from the engine thread
FILE* g_dump = nullptr;
void dump_request(const std::vector<int>& prompt, const std::vector<int>& reply, const GenResult& g) {
    if (!g_dump) return;
    std::string s = "{\"prompt\":[";
    for (size_t i = 0; i < prompt.size(); i++) { if (i) s += ','; s += std::to_string(prompt[i]); }
    s += "],\"reply\":[";
    for (size_t i = 0; i < reply.size(); i++) { if (i) s += ','; s += std::to_string(reply[i]); }
    s += "],\"drafted\":" + std::to_string(g.drafted) + ",\"accepted\":" + std::to_string(g.accepted) +
         ",\"finish\":\"" + g.finish + "\",\"decode_tok_s\":" + std::to_string(g.decode_tok_s) + "}\n";
    fwrite(s.data(), 1, s.size(), g_dump);
    fflush(g_dump);
}

std::string hex24() {
    static std::mt19937_64 rng{std::random_device{}()};
    static const char* h = "0123456789abcdef";
    std::string s;
    for (int i = 0; i < 24; i++) s += h[rng() & 15];
    return s;
}

// Python's round(v, nd): the decimal nearest v (not v * 10^nd rounded and divided back, which can print as 3.6830000000000003)
double round_to(double v, int nd) {
    char b[64];
    snprintf(b, sizeof b, "%.*f", nd, v);
    return strtod(b, nullptr);
}

bool is_ws(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; }
std::string lstrip(const std::string& s) { size_t i = 0; while (i < s.size() && is_ws(s[i])) i++; return s.substr(i); }
std::string strip(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && is_ws(s[a])) a++;
    while (b > a && is_ws(s[b - 1])) b--;
    return s.substr(a, b - a);
}

// length of the longest end of text that could be the start of marker
size_t partial_suffix(const std::string& text, const std::string& marker) {
    for (size_t k = std::min(marker.size() - 1, text.size()); k > 0; k--)
        if (marker.compare(0, k, text, text.size() - k, k) == 0) return k;
    return 0;
}

// serve.py Splitter: streamed model text -> reasoning (until </think>), visible content, a held tool-call tail
struct Splitter {
    bool in_think, tools, started = false, lead;
    std::string buf, held;
    Splitter(bool thinking, bool has_tools) : in_think(thinking), tools(has_tools), lead(!thinking) {}

    void content(const std::string& c0, std::vector<std::pair<bool, std::string>>& out) {
        std::string c = c0;
        if (c.empty()) return;
        if (!started) {
            c = lstrip(c);
            if (c.empty()) return;
            started = true;
        }
        out.push_back({false, c});
    }
    static std::string drop_think(std::string s) {
        size_t p;
        while ((p = s.find("<think>")) != std::string::npos) s.erase(p, 7);
        return s;
    }
    // (is_reasoning, text) pieces
    std::vector<std::pair<bool, std::string>> feed(const std::string& text, bool final = false) {
        std::vector<std::pair<bool, std::string>> out;
        buf += text;
        if (lead) {
            std::string head = lstrip(buf);
            if (head.compare(0, 7, "<think>") == 0) {
                in_think = true;
                lead = false;
                buf = head.substr(7);
            } else if (!final && std::string("<think>").compare(0, head.size(), head) == 0) {
                return out;
            } else {
                lead = false;
            }
        }
        while (!buf.empty()) {
            if (in_think) {
                size_t p = buf.find("</think>");
                if (p == std::string::npos) {
                    size_t hold = final ? 0 : partial_suffix(buf, "</think>");
                    std::string emit = buf.substr(0, buf.size() - hold);
                    buf = buf.substr(buf.size() - hold);
                    if (!emit.empty()) out.push_back({true, drop_think(emit)});
                    break;
                }
                if (p) out.push_back({true, drop_think(buf.substr(0, p))});
                in_think = false;
                buf = buf.substr(p + 8);
                continue;
            }
            if (!held.empty()) {
                held += buf;
                buf.clear();
                break;
            }
            if (tools) {
                size_t p = buf.find("<tool_call>");
                if (p != std::string::npos) {
                    content(buf.substr(0, p), out);
                    held = buf.substr(p);
                    buf.clear();
                    break;
                }
                size_t hold = final ? 0 : partial_suffix(buf, "<tool_call>");
                content(buf.substr(0, buf.size() - hold), out);
                buf = buf.substr(buf.size() - hold);
                break;
            }
            content(buf, out);
            buf.clear();
            break;
        }
        return out;
    }
};

// serve.py parse_tool_calls: <tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value>...</tool_call>; string-typed
// parameters (per the tool's schema) stay raw text, others are parsed as JSON when they parse
std::pair<std::string, Json> parse_tool_calls(const std::string& content, const Json* tools) {
    Json calls = Json::make_arr();
    if (!tools || content.find("<tool_call>") == std::string::npos) return {content, calls};
    auto prop_type = [&](const std::string& fn, const std::string& k) -> std::string {
        for (auto& t : tools->arr) {
            const Json& f = t.has("function") ? t["function"] : t;
            if (f.get_s("name", "") != fn) continue;
            const Json* p = f.get("parameters");
            const Json* props = p ? p->get("properties") : nullptr;
            const Json* pk = props ? props->get(k) : nullptr;
            return pk ? pk->get_s("type", "") : "";
        }
        return "";
    };
    std::string rest;
    size_t i = 0;
    while (true) {
        size_t a = content.find("<tool_call>", i);
        if (a == std::string::npos) { rest += content.substr(i); break; }
        rest += content.substr(i, a - i);
        size_t b = content.find("</tool_call>", a + 11);
        std::string body = content.substr(a + 11, b == std::string::npos ? std::string::npos : b - a - 11);
        i = b == std::string::npos ? content.size() : b + 12;
        std::string name = strip(body.substr(0, body.find("<arg_key>")));
        if (name.empty()) continue;
        Json args = Json::make_obj();
        size_t p = 0;
        while (true) {
            size_t k0 = body.find("<arg_key>", p);
            if (k0 == std::string::npos) break;
            size_t k1 = body.find("</arg_key>", k0);
            if (k1 == std::string::npos) break;
            size_t v0 = k1 + 10;
            while (v0 < body.size() && is_ws(body[v0])) v0++;
            if (body.compare(v0, 11, "<arg_value>") != 0) { p = k1; continue; }
            size_t v1 = body.find("</arg_value>", v0);
            if (v1 == std::string::npos) break;
            std::string k = strip(body.substr(k0 + 9, k1 - k0 - 9)), v = strip(body.substr(v0 + 11, v1 - v0 - 11));
            if (prop_type(name, k) == "string") {
                args.set(k, Json::make_str(v));
            } else {
                auto pj = Json::try_parse(v);
                args.set(k, pj ? *pj : Json::make_str(v));
            }
            p = v1 + 12;
        }
        Json fn = Json::make_obj();
        fn.set("name", Json::make_str(name));
        fn.set("arguments", Json::make_str(dump(args)));
        Json c = Json::make_obj();
        c.set("id", Json::make_str("call_" + hex24()));
        c.set("type", Json::make_str("function"));
        c.set("function", fn);
        calls.push(c);
    }
    return {strip(rest), calls};
}

// SGLang's qwen3_coder parser: <tool_call>\n<function=name>\n<parameter=k>\nv\n</parameter>\n...</function>\n</tool_call>.
// A value loses one leading and one trailing newline; string-typed parameters (per the tool's schema) stay text,
// others are parsed as JSON when they parse
std::pair<std::string, Json> parse_tool_calls_qwen(const std::string& content, const Json* tools) {
    Json calls = Json::make_arr();
    if (!tools || content.find("<tool_call>") == std::string::npos) return {content, calls};
    auto prop_type = [&](const std::string& fn, const std::string& k) -> std::string {
        for (auto& t : tools->arr) {
            const Json& f = t.has("function") ? t["function"] : t;
            if (f.get_s("name", "") != fn) continue;
            const Json* p = f.get("parameters");
            const Json* props = p ? p->get("properties") : nullptr;
            const Json* pk = props ? props->get(k) : nullptr;
            return pk ? pk->get_s("type", "") : "";
        }
        return "";
    };
    std::string rest;
    size_t i = 0;
    while (true) {
        size_t a = content.find("<tool_call>", i);
        if (a == std::string::npos) { rest += content.substr(i); break; }
        rest += content.substr(i, a - i);
        size_t b = content.find("</tool_call>", a + 11);
        std::string body = content.substr(a + 11, b == std::string::npos ? std::string::npos : b - a - 11);
        i = b == std::string::npos ? content.size() : b + 12;
        size_t f0 = body.find("<function=");
        if (f0 == std::string::npos) continue;
        size_t f1 = body.find('>', f0);
        if (f1 == std::string::npos) continue;
        std::string name = strip(body.substr(f0 + 10, f1 - f0 - 10));
        if (name.empty()) continue;
        Json args = Json::make_obj();
        size_t p = f1 + 1;
        while (true) {
            size_t k0 = body.find("<parameter=", p);
            if (k0 == std::string::npos) break;
            size_t k1 = body.find('>', k0);
            if (k1 == std::string::npos) break;
            size_t v1 = body.find("</parameter>", k1);
            size_t nx = body.find("<parameter=", k1);
            if (v1 == std::string::npos || (nx != std::string::npos && nx < v1)) v1 = nx;   // an unclosed parameter
            if (v1 == std::string::npos) v1 = body.find("</function>", k1);
            if (v1 == std::string::npos) v1 = body.size();
            std::string k = strip(body.substr(k0 + 11, k1 - k0 - 11)), v = body.substr(k1 + 1, v1 - k1 - 1);
            if (!v.empty() && v[0] == '\n') v.erase(0, 1);
            if (!v.empty() && v.back() == '\n') v.pop_back();
            if (prop_type(name, k) == "string") {
                args.set(k, Json::make_str(v));
            } else {
                auto pj = Json::try_parse(v);
                args.set(k, pj ? *pj : Json::make_str(v));
            }
            p = v1;
        }
        Json fn = Json::make_obj();
        fn.set("name", Json::make_str(name));
        fn.set("arguments", Json::make_str(dump(args)));
        Json c = Json::make_obj();
        c.set("id", Json::make_str("call_" + hex24()));
        c.set("type", Json::make_str("function"));
        c.set("function", fn);
        calls.push(c);
    }
    return {strip(rest), calls};
}

// one request's events, from the engine thread to its connection
struct Job {
    std::vector<int> ids;
    std::vector<Q35Pixels> pixels;                          // Qwen3.8-27B: the prompt's images, encoded on the engine thread
    SamplingParams sampling;
    std::vector<std::string> stops;
    int max_new = 0;
    std::vector<int> turn_marks;                            // Gemma: where the prompt's last turns start (GenRequest)
    std::atomic<bool> cancelled{false};
    std::atomic<int> fd{-1};                                // its connection while the handler waits on it, else -1
    std::chrono::steady_clock::time_point held{};           // engine thread: since when it waits for a running game
    std::mutex mu;
    std::condition_variable cv;
    struct Ev { int kind; std::string text; Json info; };   // 0 progress, 1 text, 2 done, 3 error
    std::vector<int> reply_ids;                             // the reply's tokens (return_token_ids)
    std::deque<Ev> evs;
    void put(Ev e) {
        { std::lock_guard<std::mutex> g(mu); evs.push_back(std::move(e)); }
        cv.notify_all();
    }
};

// a handler's hold on its job's connection: the engine thread stops looking at the socket once the handler is done
struct Unwatch {
    std::shared_ptr<Job> j;
    ~Unwatch() { j->fd = -1; }
};

struct Server {
    LM* model = nullptr;
    Engine* engine = nullptr;
    G4Engine* games = nullptr;                                // Gemma: several requests at once
    Q35Vision* vision = nullptr;                              // Qwen3.8-27B
    const Tokenizer* tok = nullptr;
    std::vector<int> eos;
    int turn_id = -1;                                         // Gemma: the token that starts a turn
    std::string name;
    int ctx = 0;
    float default_temp = 1.0f, default_top_p = 1.0f;
    std::mutex qmu;
    std::condition_variable qcv;
    std::deque<std::shared_ptr<Job>> queue;
    bool quit = false;
    std::string broken;
    // the log: a line every 2 s for each request being read or answered (GLM_LOG_PROGRESS=0: none), and what a
    // request's last line says (GLM_LOG_DETAIL=1: its game, shared prefix, drafts and finish reason too)
    bool progress = !(getenv("GLM_LOG_PROGRESS") && atoi(getenv("GLM_LOG_PROGRESS")) == 0);
    bool detail = getenv("GLM_LOG_DETAIL") && atoi(getenv("GLM_LOG_DETAIL"));
    std::mutex mmu;
    std::vector<std::shared_ptr<Meter>> meters;
    int meter_ids = 0;

    // one request's way from the engine's tokens to its connection
    struct Running {
        std::shared_ptr<Job> jp;
        std::shared_ptr<Meter> meter;
        int told = -1;                                        // the prompt progress its client was last sent
        std::unique_ptr<StreamDecoder> dec;
        std::string pending;                                  // text held for a partial stop-string match
        int n = 0;
        bool stopped = false;
        std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        void emit(std::string t, bool final) {
            Job& j = *jp;
            pending += t;
            if (!j.stops.empty()) {
                size_t cut = std::string::npos;
                for (auto& s : j.stops) {
                    size_t p = pending.find(s);
                    if (p != std::string::npos && p < cut) cut = p;
                }
                if (cut != std::string::npos) {
                    if (cut) j.put({1, pending.substr(0, cut), {}});
                    pending.clear();
                    stopped = true;
                    return;
                }
                size_t hold = 0;
                if (!final)
                    for (auto& s : j.stops) hold = std::max(hold, partial_suffix(pending, s));
                std::string out = pending.substr(0, pending.size() - hold);
                pending = pending.substr(pending.size() - hold);
                if (!out.empty()) j.put({1, out, {}});
                return;
            }
            if (!pending.empty()) j.put({1, pending, {}});
            pending.clear();
        }
    };

    GenRequest prep(const std::shared_ptr<Running>& st) {
        Job& j = *st->jp;
        GenRequest r;
        r.prompt = j.ids;
        r.sampling = j.sampling;
        r.max_new = j.max_new;
        r.stop_ids = eos;
        r.cancelled = &j.cancelled;
        r.turn_marks = j.turn_marks;
        if (!j.pixels.empty()) {
            r.fresh = true;
            Q35Vision* vis = vision;
            LM* lm = model;
            Job* job = &j;
            r.before = [vis, lm, job] {
                auto t0 = std::chrono::steady_clock::now();
                std::vector<Q35Image> imgs;
                for (auto& px : job->pixels) imgs.push_back(vis->encode(px));
                static_cast<Q35Model*>(lm)->set_images(std::move(imgs));
                fprintf(stderr, "vision: %zu image(s) in %.2f s\n", job->pixels.size(),
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
            };
        }
        size_t max_stop = 0;
        for (auto& s : j.stops) max_stop = std::max(max_stop, utf8_length(s));
        st->dec = std::make_unique<StreamDecoder>(*tok, max_stop);
        Running* rs = st.get();
        Job* jr = &j;
        st->meter = std::make_shared<Meter>();
        {
            std::lock_guard<std::mutex> g(mmu);
            st->meter->id = ++meter_ids;
            if (progress) meters.push_back(st->meter);
        }
        Meter* mt = st->meter.get();
        const bool marks = progress;
        r.on_token = [rs, jr, mt](int id) {
            mt->tokens.fetch_add(1, std::memory_order_relaxed);
            rs->n++;
            jr->reply_ids.push_back(id);
            rs->emit(rs->dec->push(id, rs->n == jr->max_new), false);
            return !rs->stopped && !jr->cancelled.load();
        };
        r.on_progress = [rs, jr, mt, marks](int done, int total, int cached) {
            // the log's meter: reading until the whole prompt is in, then writing. The engine says so before each
            // piece it reads too, which makes this request the one the model's marks count for
            const int phase = done < total ? 1 : 2;
            mt->total = total;
            mt->read = done;
            if (mt->phase.load() != phase) {
                mt->first = done;
                mt->since = now_ns();
                mt->phase = phase;
            }
            if (marks && phase == 1) read_marks().restart(mt, done);
            if (done == rs->told) return;
            rs->told = done;
            Json p = Json::make_obj();
            p.set("total", Json::make_int(total));
            p.set("cache", Json::make_int(cached));
            p.set("processed", Json::make_int(done));
            p.set("time_ms", Json::make_int((int64_t)(std::chrono::duration<double>(
                                                           std::chrono::steady_clock::now() - rs->t0).count() * 1000)));
            jr->put({0, "", p});
        };
        return r;
    }

    void finish_job(Running& st, const GenResult& g) {
        Job& j = *st.jp;
        if (!st.stopped) st.emit(st.dec->flush(), true);
        if (g.stop_id >= 0) j.reply_ids.push_back(g.stop_id);
        Json info = Json::make_obj();
        std::string fin = g.finish == "length" ? "length" : "stop";
        info.set("finish_reason", Json::make_str(fin));
        info.set("prompt_tokens", Json::make_int(g.prompt_tokens));
        info.set("completion_tokens", Json::make_int(g.completion_tokens));
        info.set("cached_tokens", Json::make_int(g.cached_tokens));
        info.set("prompt_s", Json::make_num(round_to(g.prompt_s, 3)));
        info.set("total_s", Json::make_num(round_to(g.total_s, 3)));
        info.set("decode_tok_s", g.decode_tok_s > 0 ? Json::make_num(round_to(g.decode_tok_s, 2)) : Json());
        st.meter->phase = 3;
        // the request's last line: what it wrote and how fast, and what it had to read of its prompt
        const int read = std::max(0, g.prompt_tokens - g.cached_tokens);
        char rd[96], more[160] = "";
        if (read >= 64 && g.prompt_s > 0)
            snprintf(rd, sizeof rd, "%d read in %.1f s (%.0f t/s)", read, g.prompt_s, read / g.prompt_s);
        else
            snprintf(rd, sizeof rd, "%d read in %.2f s", read, g.prompt_s);
        // "shared" is the prefix the prompt has in common with its game: far above the cached part, no checkpoint was near
        if (detail)
            snprintf(more, sizeof more, "; %s, %d of %d drafts accepted, game %d, %d shared", g.finish.c_str(), g.accepted,
                     g.drafted, g.game, g.shared_tokens);
        if (j.cancelled.load()) {
            log("cancelled after %d tokens; prompt %d tokens, %s%s", g.completion_tokens, g.prompt_tokens, rd, more);
            return;                                   // nobody is listening (serve.py sends no done either)
        }
        char sp[48] = "";
        if (g.decode_tok_s > 0) snprintf(sp, sizeof sp, " at %.1f t/s", g.decode_tok_s);
        log("done: %d token%s%s; prompt %d tokens, %s%s", g.completion_tokens, g.completion_tokens == 1 ? "" : "s", sp,
            g.prompt_tokens, rd, more);
        dump_request(j.ids, j.reply_ids, g);
        j.put({2, "", info});
    }

    // The log's progress lines, on a thread of its own: 2 s after a request starts reading its prompt or writing its
    // reply, and every 2 s from then, a line with its speed since the line before. A request that has nothing new
    // (another is being read first; a model that marks nothing inside a long pass) gets no line.
    void report() {
        for (;;) {
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            std::vector<std::shared_ptr<Meter>> ms;
            {
                std::lock_guard<std::mutex> g(mmu);
                std::erase_if(meters, [](const std::shared_ptr<Meter>& m) { return m->phase.load() == 3; });
                ms = meters;
            }
            int busy = 0;
            for (auto& m : ms) busy += m->phase.load() == 1 || m->phase.load() == 2;
            for (auto& m : ms) {
                const int phase = m->phase.load();
                if (phase != 1 && phase != 2) continue;
                const int64_t now = now_ns();
                if (phase != m->r_phase) {
                    m->r_phase = phase;
                    m->r_at = m->since.load();
                    m->r_n = phase == 1 ? m->first.load() : 0;
                }
                const double dt = (now - m->r_at) * 1e-9;
                if (dt < 2.0) continue;
                const std::string who = busy > 1 ? "#" + std::to_string(m->id) + " " : "";
                if (phase == 1) {
                    const int total = m->total.load();
                    double n = read_marks().done(m.get());
                    if (n < 0) {                              // another prompt is read first: its wait is not its speed
                        m->r_at = now;
                        continue;
                    }
                    n = std::min<double>(std::max<double>(n, m->read.load()), total);
                    if (n <= m->r_n) continue;
                    log("%sprefill %d/%d tokens (%d%%), %.0f t/s", who.c_str(), (int)n, total,
                        total ? (int)(100.0 * n / total) : 0, (n - m->r_n) / dt);
                    m->r_n = n;
                } else {
                    const int n = m->tokens.load();
                    if (n <= m->r_n) continue;
                    log("%sgenerating: %d tokens, %.1f t/s", who.c_str(), n, (n - m->r_n) / dt);
                    m->r_n = n;
                }
                m->r_at = now;
            }
        }
    }

    void run_job(const std::shared_ptr<Job>& jp) {
        auto st = std::make_shared<Running>();
        st->jp = jp;
        GenRequest r = prep(st);
        finish_job(*st, engine->run(r));
    }

    // Gemma: requests start as games free up and decode together; the engine thread is the only one on the GPU
    void loop_games() {
        std::vector<std::shared_ptr<Running>> running;
        for (;;) {
            std::vector<std::shared_ptr<Running>> fresh;
            bool waiting = false;
            {
                std::unique_lock<std::mutex> lk(qmu);
                qcv.wait(lk, [&] { return quit || !queue.empty() || games->busy(); });
                if (quit) return;
                waiting = !queue.empty();
            }
            // A client that hung up leaves its game before a waiting request picks one. The client cancels a request and sends
            // it again with the reply's thought added: taken while the first still held its game, the second went to
            // another game and read the whole prompt again.
            if (waiting && games->busy()) {
                for (auto& st : running) {
                    Job& j = *st->jp;
                    int fd = j.fd.load();
                    if (fd >= 0 && !j.cancelled.load() && socket_peer_gone(fd)) j.cancelled = true;
                }
                games->reap();
            }
            {
                std::unique_lock<std::mutex> lk(qmu);
                size_t free_now = 0;
                // has_free says one; start one a pass (a start may take the last game)
                free_now = games->has_free() ? 1 : 0;
                // The same request sent again, its cancel still on the way: up to half a second for the game that holds
                // its tokens, when the free games would have it read over 512 more.
                if (free_now && !queue.empty() && games->busy()) {
                    Job& j = *queue.front();
                    auto now = std::chrono::steady_clock::now();
                    if (j.held == std::chrono::steady_clock::time_point{}) j.held = now;
                    if (now - j.held < std::chrono::milliseconds(500) &&
                        games->shared_prefix(j.ids, true) > games->shared_prefix(j.ids, false) + 512)
                        free_now = 0;
                }
                while (free_now-- && !queue.empty()) {
                    auto st = std::make_shared<Running>();
                    st->jp = queue.front();
                    queue.pop_front();
                    fresh.push_back(st);
                }
            }
            try {
                for (auto& st : fresh) {
                    running.push_back(st);
                    games->submit(prep(st), [this, st, &running](const GenResult& g) {
                        finish_job(*st, g);
                        running.erase(std::find(running.begin(), running.end(), st));
                    });
                }
                if (games->busy()) games->step();
            } catch (const std::exception& e) {
                broken = e.what();
                log("request failed: %s", e.what());
                for (auto& st : running) st->jp->put({3, e.what(), {}});
                return;
            }
        }
    }

    void loop() {
        if (games) return loop_games();
        for (;;) {
            std::shared_ptr<Job> j;
            {
                std::unique_lock<std::mutex> lk(qmu);
                qcv.wait(lk, [&] { return quit || !queue.empty(); });
                if (quit) return;
                j = queue.front();
                queue.pop_front();
            }
            try {
                run_job(j);
            } catch (const std::exception& e) {
                broken = e.what();
                log("request failed: %s", e.what());
                j->put({3, e.what(), {}});
            }
        }
    }
};

std::string sse(const Json& j) { return "data: " + dump(j) + "\n\n"; }

Json error_body(int code, const std::string& msg) {
    Json e = Json::make_obj();
    e.set("message", Json::make_str(msg));
    e.set("type", Json::make_str(code < 500 ? "invalid_request_error" : "server_error"));
    e.set("code", Json::make_int(code));
    Json b = Json::make_obj();
    b.set("error", e);
    return b;
}

}  // namespace

int main(int argc, char** argv) {
    std::string host = "127.0.0.1", model_dir, store, name = "glm-5.3-flash", lora, warm, hot, drafter;
    int port = 8053, ctx = 49152, parallel = 1;
    double ram_gb = 14, lora_scale = getenv("GLM_XPU_LORA_SCALE") ? atof(getenv("GLM_XPU_LORA_SCALE")) : 1.0;
    bool no_warm = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto val = [&] { GLM_CHECK(i + 1 < argc, "%s needs a value", a.c_str()); return std::string(argv[++i]); };
        if (a == "--host") host = val();
        else if (a == "--port") port = std::stoi(val());
        else if (a == "--model-dir") model_dir = val();
        else if (a == "--store") store = val();
        else if (a == "--cache") ctx = std::stoi(val());
        else if (a == "--ram-gb") ram_gb = std::stod(val());
        else if (a == "--served-name") name = val();
        else if (a == "--lora") lora = val();
        else if (a == "--lora-scale") lora_scale = std::stod(val());
        else if (a == "--warm-stats") warm = val();
        else if (a == "--no-warm") no_warm = true;
        else if (a == "--hot") hot = val();
        else if (a == "--drafter") drafter = val();
        else if (a == "--parallel") parallel = std::stoi(val());
        else if (a == "--chat-template") val();      // the client's template is built in (chat.h nq_prefill_hook)
        else GLM_CHECK(false, "unknown argument %s", a.c_str());
    }
    GLM_CHECK(!model_dir.empty(), "--model-dir is required");
    if (const char* dp = getenv("GLM_DUMP_TOKENS"); dp && *dp) {
        g_dump = fopen(dp, "a");
        GLM_CHECK(g_dump, "GLM_DUMP_TOKENS: cannot open %s", dp);
        log("dumping request tokens to %s", dp);
    }
    bool qwen = false, gemma = false, q35 = false, ripple = false;
    {
        Json cj = Json::parse(read_file(model_dir + "/config.json"));
        qwen = cj.get_s("model_type", "") == "qwen4_exp";
        gemma = cj.get_s("model_type", "") == "gemma4";
        ripple = cj.get_s("model_type", "") == "ricursive";
        q35 = cj.get_s("model_type", "") == "qwen3_5";
    }
    // Edge0 (a latent-MoE Qwen3.8-Flash-Next): the folder's model.json says so; Qwen's template and tool calls
    bool edge0 = false;
    if (std::filesystem::exists(model_dir + "/model.json") && std::filesystem::exists(model_dir + "/model-skeleton.gguf"))
        edge0 = Json::parse(read_file(model_dir + "/model.json")).get_s("family", "") == "Edge0";
    if (edge0) qwen = true;
    GLM_CHECK(qwen || gemma || q35 || ripple || !store.empty(), "--store is required for GLM");
    if (warm.empty() && !no_warm) warm = std::string(getenv("HOME")) + "/b70/llm/glm/xpu/data/route_stats_glm53.json";
    if (no_warm) warm.clear();
    if (qwen && !edge0 && hot.empty())
        hot = std::string(getenv("HOME")) + "/b70/llm/flashnext/hot/gm-7500.json," + getenv("HOME") +
              "/b70/llm/flashnext/hot/gm-ram-4500.json";

    HttpServer http(host, port);                     // bind first: a taken port fails before the load
    Tokenizer tok = Tokenizer::load(model_dir);
    // VRAM kept out of the expert pool: driver, graphs, oneMKL scratch and the prompt's per-row buffers
    double reserve = getenv("GLM_VRAM_RESERVE_GIB") ? atof(getenv("GLM_VRAM_RESERVE_GIB")) : (qwen ? 2.0 : 3.0);
    std::unique_ptr<LM> lm;
    std::string routing = "exact";
    std::unique_ptr<Q35Vision> vision;
    if (q35) {
        auto m = std::make_unique<Q35Model>(model_dir, ctx, parallel);
        // text-only packs (e.g. converted from a GGUF, whose tower was a separate mmproj) have no model.visual.*
        const bool tower = m->tensors().has("model.visual.patch_embed.proj.weight");
        if (tower) vision = std::make_unique<Q35Vision>(m->tensors());
        else fprintf(stderr, "no vision tower in %s: text only\n", model_dir.c_str());
        lm = std::move(m);
        qwen_allow_images(tower);
    } else if (gemma) {
        lm = std::make_unique<G4Model>(model_dir, ctx, drafter, parallel);
    } else if (ripple) {
        lm = std::make_unique<RpModel>(model_dir, ctx, drafter, parallel);
    } else if (edge0) {
        lm = std::make_unique<E0Model>(model_dir, ctx, reserve);
    } else if (qwen) {
        lm = std::make_unique<Q4Model>(model_dir, ctx, reserve, hot);
    } else {
        TierConfig tc;
        tc.bonus_vram = getenv("GLM_XPU_RES_BONUS") ? (float)atof(getenv("GLM_XPU_RES_BONUS")) : 0.1f;
        tc.bonus_ram = getenv("GLM_XPU_RES_BONUS_RAM") ? (float)atof(getenv("GLM_XPU_RES_BONUS_RAM")) : tc.bonus_vram;
        tc.ram_bytes = (size_t)(ram_gb * 1073741824.0);
        if (lora_scale == 0) lora.clear();
        if (tc.bonus_vram > 0) routing = "quality (bonus)";
        lm = std::make_unique<Model>(model_dir, store, ctx, tc, warm, reserve, lora, (float)lora_scale);
    }
    ctx = lm->max_context();                         // what fit (Gemma may have less room than asked)
    Engine engine(*lm);
    std::unique_ptr<G4Engine> g4_engine;
    if (gemma) g4_engine = std::make_unique<G4Engine>(*static_cast<G4Model*>(lm.get()));
    if (ripple) {
        g4_engine = std::make_unique<G4Engine>(*static_cast<RpModel*>(lm.get()));
        g4_engine->max_rows = 16;                    // DPAS: 16 rows cost 1.6 times one
    }
    if (q35) g4_engine = std::make_unique<G4Engine>(*static_cast<Q35Model*>(lm.get()));
    Server S;
    S.model = lm.get();
    S.engine = &engine;
    S.games = g4_engine.get();
    S.vision = vision.get();
    S.tok = &tok;
    if (gemma)
        if (auto id = tok.id_of("<|turn>")) S.turn_id = *id;
    S.name = name;
    S.ctx = ctx;
    {
        // (a checkpoint without one: config.json's eos_token_id, temperature and top_p 1)
        std::string gcp = model_dir + "/generation_config.json";
        Json gc = Json::parse(read_file(std::filesystem::exists(gcp) ? gcp : model_dir + "/config.json"));
        if (const Json* e = gc.get("eos_token_id")) {
            if (e->kind == Json::Arr) for (auto& x : e->arr) S.eos.push_back(x.i());
            else S.eos.push_back(e->i());
        }
        // and the tokenizer's eos_token, as serve.py adds tok.eos_token_id
        Json tc2 = Json::parse(read_file(model_dir + "/tokenizer_config.json"));
        if (tc2.has("eos_token") && tc2["eos_token"].kind == Json::Str)
            if (auto id = tok.id_of(tc2["eos_token"].str))
                if (std::find(S.eos.begin(), S.eos.end(), *id) == S.eos.end()) S.eos.push_back(*id);
        S.default_temp = (float)gc.get_f("temperature", 1.0);
        S.default_top_p = (float)gc.get_f("top_p", 1.0);
    }
    ServeConfig scfg = ServeConfig::from_env();
    // Qwen: the reasoning effort for requests that name none (xhigh, medium or low; unset: the template's xhigh)
    const std::string qwen_effort = getenv("Q4_SERVE_REASONING_EFFORT") ? getenv("Q4_SERVE_REASONING_EFFORT") : "";

    auto submit = [&](const std::vector<int>& ids, const Json& body, HttpConn& c, std::vector<Q35Pixels> pixels = {}) {
        auto j = std::make_shared<Job>();
        j->ids = ids;
        j->pixels = std::move(pixels);
        j->fd = c.fd();
        // the prompt's last two turn starts: mostly the last user message's and the reply's
        if (S.turn_id >= 0)
            for (int i = (int)ids.size() - 1; i > 0 && j->turn_marks.size() < 2; i--)
                if (ids[i] == S.turn_id) j->turn_marks.insert(j->turn_marks.begin(), i);
        int P = (int)ids.size(), room = ctx - P - 8;
        GLM_CHECK(room >= 1, "prompt is %d tokens; the server's context is %d", P, ctx);
        const Json* mx = body.get("max_completion_tokens");
        if (!mx || mx->kind == Json::Null) mx = body.get("max_tokens");
        j->max_new = mx && mx->kind == Json::Num ? std::max(1, std::min((int)mx->i(), room)) : room;
        if (const Json* s = body.get("stop")) {
            if (s->kind == Json::Str) j->stops.push_back(s->str);
            else if (s->kind == Json::Arr) for (auto& x : s->arr) if (x.kind == Json::Str && !x.str.empty()) j->stops.push_back(x.str);
        }
        SamplingParams& sp = j->sampling;
        sp.temperature = (float)body.get_f("temperature", S.default_temp);
        sp.top_p = (float)body.get_f("top_p", S.default_top_p);
        sp.top_k = (int)body.get_i("top_k", 0);
        sp.min_p = (float)body.get_f("min_p", 0.0);
        sp.rep_p = (float)body.get_f("repetition_penalty", 1.0);
        sp.pres_p = (float)body.get_f("presence_penalty", 0.0);
        sp.freq_p = (float)body.get_f("frequency_penalty", 0.0);
        if (const Json* sd = body.get("seed"); sd && sd->kind == Json::Num) { sp.has_seed = true; sp.seed = sd->i(); }
        {
            std::lock_guard<std::mutex> g(S.qmu);
            S.queue.push_back(j);
        }
        S.qcv.notify_all();
        return j;
    };
    // the job's events; cancels it when the client hangs up
    auto next_event = [&](Job& j, HttpConn& c, Job::Ev& ev) {
        for (;;) {
            std::unique_lock<std::mutex> lk(j.mu);
            if (j.cv.wait_for(lk, std::chrono::milliseconds(500), [&] { return !j.evs.empty(); })) {
                ev = std::move(j.evs.front());
                j.evs.pop_front();
                return true;
            }
            lk.unlock();
            if (c.client_gone()) { j.cancelled = true; return false; }
        }
    };
    auto usage = [](const Json& info) {
        Json u = Json::make_obj();
        int64_t P = info["prompt_tokens"].i(), n = info["completion_tokens"].i();
        u.set("prompt_tokens", Json::make_int(P));
        u.set("completion_tokens", Json::make_int(n));
        u.set("total_tokens", Json::make_int(P + n));
        Json d = Json::make_obj();
        d.set("cached_tokens", Json::make_int(info["cached_tokens"].i()));
        u.set("prompt_tokens_details", d);
        return u;
    };
    auto timings = [](const Json& info) {
        Json t = Json::make_obj();
        for (const char* k : {"cached_tokens", "prompt_s", "total_s", "decode_tok_s"}) t.set(k, info[k]);
        return t;
    };

    http.serve([&](const HttpRequest& rq, HttpConn& c) {
        auto send_json = [&](int code, const Json& j) { c.send(code, "application/json", dump(j)); };
        std::string path = rq.path;
        while (path.size() > 1 && path.back() == '/') path.pop_back();
        if (rq.method == "GET") {
            if (path == "/health" || path == "/") {
                Json b = Json::make_obj();
                b.set("ok", Json::make_bool(S.broken.empty()));
                b.set("error", S.broken.empty() ? Json() : Json::make_str(S.broken));
                return send_json(S.broken.empty() ? 200 : 503, b);
            }
            if (path == "/v1/models" || path == "/models") {
                Json m = Json::make_obj();
                m.set("id", Json::make_str(name));
                m.set("object", Json::make_str("model"));
                m.set("owned_by", Json::make_str("local"));
                m.set("max_model_len", Json::make_int(ctx));
                Json d = Json::make_arr();
                d.push(m);
                Json b = Json::make_obj();
                b.set("object", Json::make_str("list"));
                b.set("data", d);
                return send_json(200, b);
            }
            return send_json(404, error_body(404, "no route " + rq.path));
        }
        if (path == "/quit") {
            if (rq.peer != "127.0.0.1") return send_json(403, error_body(403, "quit only from this machine"));
            Json b = Json::make_obj();
            b.set("ok", Json::make_bool(true));
            send_json(200, b);
            { std::lock_guard<std::mutex> g(S.qmu); S.quit = true; }
            S.qcv.notify_all();
            return;
        }
        auto bj = Json::try_parse(rq.body.empty() ? "{}" : rq.body);
        if (!bj) return send_json(400, error_body(400, "body is not JSON"));
        const Json& body = *bj;
        if (!S.broken.empty())
            return send_json(503, error_body(503, "the model failed earlier (" + S.broken + "); restart the server"));
        std::string created = std::to_string((int64_t)time(nullptr));
        if (path == "/v1/chat/completions" || path == "/chat/completions") {
            const Json* msgs = body.get("messages");
            if (!msgs || msgs->kind != Json::Arr || msgs->arr.empty())
                return send_json(400, error_body(400, "messages must be a non-empty list"));
            const Json* tools = body.get("tools");
            if (tools && (tools->kind != Json::Arr || tools->arr.empty() || body.get_s("tool_choice", "") == "none"))
                tools = nullptr;
            if (gemma || ripple)
                for (auto& m : msgs->arr)
                    if (const Json* mc = m.kind == Json::Obj ? m.get("content") : nullptr; mc && mc->kind == Json::Arr)
                        for (auto& it : mc->arr) {
                            std::string ty = it.kind == Json::Obj ? it.get_s("type", "text") : "text";
                            if (ty != "text")
                                return send_json(400, error_body(400, "only text is supported by this server (no vision tower): content part of type " + ty));
                        }
            ChatPrompt cp = ripple ? render_prompt_ripple(*msgs, tools, body.get("chat_template_kwargs"), body.get("reasoning_effort"))
                          : gemma ? render_prompt_gemma(*msgs, tools, body.get("chat_template_kwargs"), body.get("reasoning_effort"))
                          : qwen || q35 ? render_prompt_qwen(*msgs, tools, body.get("chat_template_kwargs"),
                                                      body.get("reasoning_effort"), qwen_effort)
                                 : render_prompt(*msgs, tools, body.get("chat_template_kwargs"), body.get("reasoning_effort"), scfg);
            std::vector<int> prompt_ids = tok.encode(cp.text);
            std::vector<Q35Pixels> pixels;
            if (q35) {
                // pages in prompt order; each placeholder token becomes one token per merge unit
                static const int max_px = getenv("Q35_MAX_PIXELS") ? atoi(getenv("Q35_MAX_PIXELS")) : 4194304;
                for (auto& m : msgs->arr) {
                    const Json* mc = m.kind == Json::Obj ? m.get("content") : nullptr;
                    if (!mc || mc->kind != Json::Arr) continue;
                    for (auto& it : mc->arr) {
                        if (it.kind != Json::Obj) continue;
                        const Json* iu = it.get("image_url");
                        if (!iu) continue;
                        std::string url = iu->kind == Json::Str ? iu->str : iu->get_s("url", "");
                        if (url.rfind("data:", 0) != 0) return send_json(400, error_body(400, "images must be data: URLs"));
                        std::string raw = base64_decode_url(url);
                        pixels.push_back(q35_load_image(raw.data(), raw.size(), max_px));
                    }
                }
                std::vector<int> expanded;
                size_t im = 0;
                for (int id : prompt_ids) {
                    if (id != q35::IMAGE_PAD) { expanded.push_back(id); continue; }
                    if (im >= pixels.size()) return send_json(400, error_body(400, "image tokens in the text without an image"));
                    expanded.insert(expanded.end(), pixels[im].gh / q35v::MERGE * (pixels[im].gw / q35v::MERGE), id);
                    im++;
                }
                if (im != pixels.size()) return send_json(400, error_body(400, "an image part without its placeholder"));
                prompt_ids = std::move(expanded);
            }
            auto j = submit(prompt_ids, body, c, std::move(pixels));
            Unwatch unwatch{j};
            std::string cid = "chatcmpl-" + hex24();
            auto base = [&] {
                Json b = Json::make_obj();
                b.set("id", Json::make_str(cid));
                b.set("object", Json::make_str("chat.completion.chunk"));
                b.set("created", Json::make_int(std::stoll(created)));
                b.set("model", Json::make_str(name));
                return b;
            };
            Splitter sp_std(cp.reply_in_thought, tools != nullptr);
            // the client's "answer now" sends separate_reasoning false: everything after its closed thought is the answer
            GemmaSplitter sp_gem(cp.reply_in_thought, tools != nullptr, body.get_b("separate_reasoning", true));
            RippleSplitter sp_rip(cp.reply_in_thought, tools != nullptr, body.get_b("separate_reasoning", true));
            auto feed = [&](const std::string& t, bool fin = false) { return ripple ? sp_rip.feed(t, fin) : gemma ? sp_gem.feed(t, fin) : sp_std.feed(t, fin); };
            auto held = [&]() -> const std::string& { return ripple ? sp_rip.held : gemma ? sp_gem.held : sp_std.held; };
            auto parse_calls = [&](const std::string& t) {
                return ripple ? parse_tool_calls_ripple(t, tools) : gemma ? parse_tool_calls_gemma(t, tools) : qwen || q35 ? parse_tool_calls_qwen(t, tools) : parse_tool_calls(t, tools);
            };
            if (body.get_b("stream", false)) {
                auto chunk = [&](Json delta, const char* fin = nullptr) {
                    Json ch = Json::make_obj();
                    ch.set("index", Json::make_int(0));
                    ch.set("delta", delta);
                    ch.set("finish_reason", fin ? Json::make_str(fin) : Json());
                    Json arr = Json::make_arr();
                    arr.push(ch);
                    Json b = base();
                    b.set("choices", arr);
                    return b;
                };
                c.start_stream();
                Json d0 = Json::make_obj();
                d0.set("role", Json::make_str("assistant"));
                d0.set("content", Json::make_str(""));
                if (!c.write(sse(chunk(d0)))) { j->cancelled = true; return; }
                bool progress = body.get_b("return_progress", false);
                Json info;
                Job::Ev ev;
                while (next_event(*j, c, ev)) {
                    if (ev.kind == 3) { c.write(sse(error_body(500, ev.text))); return; }
                    std::string out;
                    if (ev.kind == 0) {
                        if (!progress) continue;
                        Json d = Json::make_obj();
                        d.set("role", Json::make_str("assistant"));
                        d.set("content", Json());
                        Json b = chunk(d);
                        b.set("prompt_progress", ev.info);
                        out = sse(b);
                    } else {
                        if (ev.kind == 2) info = ev.info;
                        for (auto& [r, t] : ev.kind == 2 ? feed("", true) : feed(ev.text)) {
                            Json d = Json::make_obj();
                            d.set(r ? "reasoning_content" : "content", Json::make_str(t));
                            out += sse(chunk(d));
                        }
                    }
                    if (!out.empty() && !c.write(out)) { j->cancelled = true; return; }
                    if (ev.kind == 2) break;
                }
                if (info.kind != Json::Obj) return;     // the client left
                std::string reason = info["finish_reason"].s();
                std::string tail;
                if (!held().empty()) {
                    auto [content, calls] = parse_calls(held());
                    if (!content.empty()) {
                        Json d = Json::make_obj();
                        d.set("content", Json::make_str(content));
                        tail += sse(chunk(d));
                    }
                    if (calls.size()) {
                        reason = "tool_calls";
                        Json arr = Json::make_arr();
                        for (size_t i = 0; i < calls.size(); i++) {
                            Json cc = calls.arr[i];
                            cc.set("index", Json::make_int((int64_t)i));
                            arr.push(cc);
                        }
                        Json d = Json::make_obj();
                        d.set("tool_calls", arr);
                        tail += sse(chunk(d));
                    }
                }
                Json last = chunk(Json::make_obj(), reason.c_str());
                last.set("usage", usage(info));
                last.set("timings", timings(info));
                tail += sse(last) + "data: [DONE]\n\n";
                c.write(tail);
                return;
            }
            std::string reasoning, content;
            Json info;
            Job::Ev ev;
            while (next_event(*j, c, ev)) {
                if (ev.kind == 3) return send_json(500, error_body(500, ev.text));
                if (ev.kind == 0) continue;
                if (ev.kind == 2) info = ev.info;
                for (auto& [r, t] : ev.kind == 2 ? feed("", true) : feed(ev.text)) (r ? reasoning : content) += t;
                if (ev.kind == 2) break;
            }
            if (info.kind != Json::Obj) return;
            Json calls = Json::make_arr();
            std::string text = content;
            if (!held().empty()) {
                auto [rest, cl] = parse_calls(held());
                calls = cl;
                text = (gemma || ripple ? text : strip(text)) + (rest.empty() ? "" : "\n" + rest);
            }
            Json msg = Json::make_obj();
            msg.set("role", Json::make_str("assistant"));
            // Gemma as SGLang returns it: nothing stripped, reasoning_content whenever there was any (a model can open a
            // thought unasked)
            msg.set("content", Json::make_str(gemma || ripple ? text : strip(text)));
            if (gemma || ripple ? !reasoning.empty() : cp.thinking) msg.set("reasoning_content", Json::make_str(gemma || ripple ? reasoning : strip(reasoning)));
            if (calls.size()) msg.set("tool_calls", calls);
            Json ch = Json::make_obj();
            ch.set("index", Json::make_int(0));
            ch.set("message", msg);
            ch.set("finish_reason", Json::make_str(calls.size() ? "tool_calls" : info["finish_reason"].s()));
            // SGLang's return_prompt_token_ids / return_token_ids, for exl3conv's calibration traces
            auto ids_json = [](const std::vector<int>& v) {
                Json a = Json::make_arr();
                for (int x : v) a.push(Json::make_int(x));
                return a;
            };
            if (body.get_b("return_prompt_token_ids", false)) ch.set("prompt_token_ids", ids_json(j->ids));
            if (body.get_b("return_token_ids", false)) ch.set("response_token_ids", ids_json(j->reply_ids));
            Json arr = Json::make_arr();
            arr.push(ch);
            Json b = Json::make_obj();
            b.set("id", Json::make_str(cid));
            b.set("object", Json::make_str("chat.completion"));
            b.set("created", Json::make_int(std::stoll(created)));
            b.set("model", Json::make_str(name));
            b.set("choices", arr);
            b.set("usage", usage(info));
            b.set("timings", timings(info));
            return send_json(200, b);
        }
        if (path == "/v1/completions" || path == "/completions") {
            const Json* pr = body.get("prompt");
            std::string prompt;
            if (pr && pr->kind == Json::Arr) prompt = pr->arr.empty() ? "" : pr->arr[0].s();
            else if (pr && pr->kind == Json::Str) prompt = pr->str;
            else return send_json(400, error_body(400, "prompt must be a string"));
            auto j = submit(tok.encode(prompt), body, c);
            Unwatch unwatch{j};
            std::string cid = "cmpl-" + hex24();
            auto obj = [&](const std::string& text, const char* fin) {
                Json ch = Json::make_obj();
                ch.set("index", Json::make_int(0));
                ch.set("text", Json::make_str(text));
                ch.set("finish_reason", fin ? Json::make_str(fin) : Json());
                Json arr = Json::make_arr();
                arr.push(ch);
                Json b = Json::make_obj();
                b.set("id", Json::make_str(cid));
                b.set("object", Json::make_str("text_completion"));
                b.set("created", Json::make_int(std::stoll(created)));
                b.set("model", Json::make_str(name));
                b.set("choices", arr);
                return b;
            };
            Job::Ev ev;
            Json info;
            std::string text;
            bool stream = body.get_b("stream", false);
            if (stream) c.start_stream();
            while (next_event(*j, c, ev)) {
                if (ev.kind == 3) { if (stream) c.write(sse(error_body(500, ev.text))); else send_json(500, error_body(500, ev.text)); return; }
                if (ev.kind == 0) continue;
                if (ev.kind == 2) { info = ev.info; break; }
                if (stream) { if (!c.write(sse(obj(ev.text, nullptr)))) { j->cancelled = true; return; } }
                else text += ev.text;
            }
            if (info.kind != Json::Obj) return;
            Json last = obj(stream ? "" : text, info["finish_reason"].s().c_str());
            last.set("usage", usage(info));
            last.set("timings", timings(info));
            if (stream) c.write(sse(last) + "data: [DONE]\n\n");
            else send_json(200, last);
            return;
        }
        send_json(404, error_body(404, "no route " + rq.path));
    });
    log("server ready on http://%s:%d/v1 (model %s, context %d, routing %s, LoRA %s)", host.c_str(), port, name.c_str(),
        ctx, qwen || q35 ? "exact" : routing.c_str(), lora.empty() ? "off" : lora.c_str());
    if (S.progress) std::thread([&S] { S.report(); }).detach();
    S.loop();
    log("server stopped");
    Dev::get().q.wait();
    _exit(0);
}
