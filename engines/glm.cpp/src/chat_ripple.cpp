// Ripple-65B's chat_template.jinja (ricursive), ported by hand: every branch of the template, in its order, with Jinja's
// semantics where they matter (undefined vs None, `str + x` raising on anything but a str, tojson =
// json.dumps(ensure_ascii=False)). Plus what its server does around it: ripple_format.py's reasoning and tool-call parsers.
#include <algorithm>
#include <random>
#include "chat.h"
#include "rt.h"

namespace glm {

namespace {

// A template value: nullptr is Jinja's Undefined (a missing key), a Json Null is Python's None.
using V = const Json*;

const std::string kThinkOpen = "<think>", kThinkClose = "</think>", kCallOpen = "<tool_call>", kCallClose = "</tool_call>";

const char* kToolsHead = "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided "
                         "with function signatures within <tools> XML tags:\n<tools>";
const char* kToolsTail = "\n</tools>\n\nFor each function call, return exactly one valid JSON object with function name and "
                         "arguments inside XML tool-call tags. The arguments field must be a JSON object, not a string, and "
                         "strings must use valid JSON escaping:\n<tool_call>\n{\"name\": \"execute_bash\", \"arguments\": "
                         "{\"command\": \"pwd\"}}\n</tool_call><|im_end|>\n";

bool truthy(V x) {
    if (!x) return false;
    switch (x->kind) {
        case Json::Null: return false;
        case Json::Bool: return x->b;
        case Json::Num: return x->num != 0;
        case Json::Str: return !x->str.empty();
        case Json::Arr: return !x->arr.empty();
        case Json::Obj: return !x->obj.empty();
    }
    return false;
}
bool is_str(V x) { return x && x->kind == Json::Str; }
bool eq_str(V x, const char* s) { return is_str(x) && x->str == s; }
bool starts_with(const std::string& s, const std::string& t) { return s.compare(0, t.size(), t) == 0; }

// x.k and x['k']: anything but a mapping holding k is Undefined (no attribute of a Python value is named like the keys read
// here), and reading Undefined itself raises
V attr(V x, const char* k) {
    GLM_CHECK(x, "chat template: '%s' of an undefined value", k);
    return x->kind == Json::Obj ? x->obj.find(k) : nullptr;
}

// the x of 'text' + x
const std::string& str_or_raise(V x, const char* what) {
    GLM_CHECK(is_str(x), "chat template: %s is not a string", what);
    return x->str;
}

// x | tojson: json.dumps raises on Undefined
std::string tojson(V x, const char* what) {
    GLM_CHECK(x, "chat template: %s is undefined", what);
    return dump(*x);
}

// empty_thought: enable_thinking false in render_prompt_ripple, not part of the template
std::string render(const Json& messages, const Json* tools, const Json& vars, bool empty_thought) {
    GLM_CHECK(messages.kind == Json::Arr, "chat template: messages is not a list");
    const auto& M = messages.arr;
    GLM_CHECK(!M.empty(), "chat template: 'role' of an undefined value");       // messages[0]['role'], in both branches
    std::string o;
    bool first_system = eq_str(attr(&M[0], "role"), "system");
    if (truthy(tools)) {
        GLM_CHECK(tools->kind == Json::Arr, "chat template: tools is not a list");
        o += "<|im_start|>system\n";
        if (first_system) o += str_or_raise(attr(&M[0], "content"), "the system message's content") + "\n\n";
        o += kToolsHead;
        for (auto& t : tools->arr) o += "\n" + dump(t);
        o += kToolsTail;
    } else if (first_system) {
        o += "<|im_start|>system\n" + str_or_raise(attr(&M[0], "content"), "the system message's content") + "<|im_end|>\n";
    }
    static const Json empty = Json::make_str("");
    bool clear_thinking = truthy(vars.get("clear_thinking"));
    for (size_t i = 0; i < M.size(); i++) {
        V message = &M[i];
        V role = attr(message, "role");
        V raw = attr(message, "content");
        V content = truthy(raw) ? raw : &empty;      // message.content or ''
        V reasoning = attr(message, "reasoning_content");
        V tool_calls = attr(message, "tool_calls");
        Json with_thought;
        if (eq_str(role, "assistant") && truthy(reasoning) && !clear_thinking) {
            const std::string& c = str_or_raise(content, "an assistant's content");   // content.lstrip()
            if (!starts_with(py_strip_text(c), kThinkOpen)) {
                with_thought = Json::make_str("<think>\n" + str_or_raise(reasoning, "reasoning_content") + "\n</think>\n" + c);
                content = &with_thought;
            }
        }
        if (eq_str(role, "user") || (eq_str(role, "system") && i > 0) || (eq_str(role, "assistant") && !truthy(tool_calls))) {
            o += "<|im_start|>" + role->str + "\n" + str_or_raise(content, "a message's content") + "<|im_end|>\n";
        } else if (eq_str(role, "assistant")) {
            o += "<|im_start|>assistant";
            if (truthy(content)) o += "\n" + str_or_raise(content, "an assistant's content");
            // (a mapping's keys and a string's characters have no .name, and the rest is not iterable: all raise)
            GLM_CHECK(tool_calls->kind == Json::Arr, "chat template: tool_calls is not a list");
            for (auto& item : tool_calls->arr) {
                V tc = &item;
                if (V fn = attr(tc, "function")) tc = fn;
                o += "\n<tool_call>\n{\"name\": " + tojson(attr(tc, "name"), "a tool call's name");
                o += ", \"arguments\": " + tojson(attr(tc, "arguments"), "a tool call's arguments") + "}\n</tool_call>";
            }
            o += "<|im_end|>\n";
        } else if (eq_str(role, "tool")) {
            if (i == 0 || !eq_str(attr(&M[i - 1], "role"), "tool")) o += "<|im_start|>user";
            o += "\n<tool_response>\n" + str_or_raise(raw, "a tool message's content") + "\n</tool_response>";
            if (i + 1 == M.size() || !eq_str(attr(&M[i + 1], "role"), "tool")) o += "<|im_end|>\n";
        }
    }
    if (truthy(vars.get("add_generation_prompt"))) {
        o += "<|im_start|>assistant\n";
        if (empty_thought) o += "<think>\n\n</think>\n";
        // the client's hook, appended to every model's template (exl3_wire.ts)
        V nq = vars.get("nq_prefill");
        if (truthy(nq)) o += py_str_value(*nq);
    }
    return o;
}

// normalize_messages, then a content list as SGLang hands it to a template that wants strings: its text parts joined with
// spaces, the rest dropped
Json shape_messages(const Json& messages) {
    Json out = normalize_messages(messages);
    for (auto& m : out.arr) {
        V c = m.get("content");
        if (c->kind != Json::Arr) continue;
        std::string joined;
        bool first = true;
        for (auto& part : c->arr) {
            if (part.kind != Json::Obj) continue;
            V ty = part.get("type");
            if (!eq_str(ty, "text") && !eq_str(ty, "input_text")) continue;
            V t = part.get("text");
            GLM_CHECK(is_str(t), "a text content part without text");
            if (!first) joined += " ";
            first = false;
            joined += t->str;
        }
        m.set("content", Json::make_str(std::move(joined)));
    }
    return out;
}

// ------------------------------------------------------------------------------------------------- tool calls

std::string call_id() {                      // the id form of glm_server's other tool calls
    static std::mt19937_64 rng{std::random_device{}()};
    static const char* h = "0123456789abcdef";
    std::string s = "call_";
    for (int i = 0; i < 24; i++) s += h[rng() & 15];
    return s;
}

// HermesDetector.parse_base_json: the parameters of a call as a JSON string
std::string arguments_of(const Json& call) {
    V a = call.get("parameters");
    if (!truthy(a)) a = call.get("arguments");
    if (!a || a->kind == Json::Null) return "{}";
    if (a->kind == Json::Str)
        if (auto parsed = Json::try_parse(a->str); parsed && parsed->kind == Json::Obj) return dump(*parsed);
    return dump(*a);
}

void add_calls(const Json& j, Json& calls) {
    if (j.kind == Json::Arr) {
        for (auto& x : j.arr)
            if (x.kind == Json::Obj) add_calls(x, calls);
        return;
    }
    V name = j.get("name");
    if (!is_str(name) || name->str.empty()) return;
    Json fn = Json::make_obj();
    fn.set("name", *name);
    fn.set("arguments", Json::make_str(arguments_of(j)));
    Json c = Json::make_obj();
    c.set("id", Json::make_str(call_id()));
    c.set("type", Json::make_str("function"));
    c.set("function", std::move(fn));
    calls.push(std::move(c));
}

size_t partial_suffix(const std::string& text, const std::string& marker) {
    for (size_t k = std::min(marker.size() - 1, text.size()); k > 0; k--)
        if (marker.compare(0, k, text, text.size() - k, k) == 0) return k;
    return 0;
}

void drop_first_newline(std::string& s) { if (!s.empty() && s[0] == '\n') s.erase(0, 1); }
void drop_last_newline(std::string& s) { if (!s.empty() && s.back() == '\n') s.pop_back(); }

}  // namespace

std::string render_chat_ripple(const Json& messages, const Json* tools, const Json& vars) {
    static const Json none = Json::make_obj();
    return render(messages, tools, vars.kind == Json::Obj ? vars : none, false);
}

// The template has no thinking switch. enable_thinking false does what the other ports' templates do with it (Qwen's and
// Gemma's empty thought in the generation prompt), in this model's layout; other than that the prompt is the template's.
ChatPrompt render_prompt_ripple(const Json& messages, const Json* tools, const Json* chat_template_kwargs, const Json*) {
    Json kw = Json::make_obj();
    if (truthy(chat_template_kwargs)) {
        GLM_CHECK(chat_template_kwargs->kind == Json::Obj, "chat_template_kwargs is not an object");
        kw = *chat_template_kwargs;
    }
    kw.set("add_generation_prompt", Json::make_bool(true));
    V et = kw.get("enable_thinking");
    ChatPrompt p;
    p.thinking = !(et && et->kind == Json::Bool && !et->b);
    p.text = render(shape_messages(messages), truthy(tools) ? tools : nullptr, kw, !p.thinking);
    // the prefill alone decides: earlier turns' text can hold anything, and the generation prompt opens nothing
    V nq = kw.get("nq_prefill");
    std::string prefill = truthy(nq) ? py_str_value(*nq) : "";
    size_t o = prefill.rfind(kThinkOpen), c = prefill.rfind(kThinkClose);
    p.reply_in_thought = o != std::string::npos && (c == std::string::npos || o > c);
    return p;
}

std::pair<std::string, Json> parse_tool_calls_ripple(const std::string& content, const Json* tools) {
    Json calls = Json::make_arr();
    size_t first = content.find(kCallOpen);
    if (!tools || first == std::string::npos) return {content, calls};
    for (size_t from = first; from < content.size();) {
        size_t a = content.find(kCallOpen, from);
        if (a == std::string::npos) break;
        size_t body = a + kCallOpen.size();
        size_t first_close = content.find(kCallClose, body);
        // up to the first close that leaves JSON (a string argument can hold the marker itself), else the end of the reply
        bool done = false;
        for (size_t e = first_close;; e = content.find(kCallClose, e + kCallClose.size())) {
            auto j = Json::try_parse(content.substr(body, e == std::string::npos ? e : e - body));
            if (j) {
                add_calls(*j, calls);
                from = e == std::string::npos ? content.size() : e + kCallClose.size();
                done = true;
                break;
            }
            if (e == std::string::npos) break;
        }
        if (done) continue;
        if (first_close == std::string::npos) break;       // cut off inside a call
        from = first_close + kCallClose.size();            // not JSON: skipped
    }
    if (calls.arr.empty()) return {content, calls};
    std::string before = content.substr(0, first);
    drop_last_newline(before);
    return {before, calls};
}

// ------------------------------------------------------------------------------------------------- RippleSplitter

// RippleReasoningDetector._parse_streaming_increment_impl
void RippleSplitter::reasoning_step(const std::string& text, std::string& reasoning, std::string& normal) {
    static const std::string close_nl = "\n" + kThinkClose;
    buf_ += text;
    while (!buf_.empty()) {
        if (state_ == Start) {
            if (starts_with(buf_, kThinkOpen)) {
                buf_.erase(0, kThinkOpen.size());
                state_ = Lead;
            } else if (starts_with(kThinkOpen, buf_)) {
                break;
            } else {
                state_ = Content;
            }
        } else if (state_ == Lead) {
            drop_first_newline(buf_);
            state_ = Think;
        } else if (state_ == Think) {
            size_t i = buf_.find(kThinkClose);
            if (i != std::string::npos) {
                std::string r = buf_.substr(0, i);
                drop_last_newline(r);
                reasoning += r;
                buf_.erase(0, i + kThinkClose.size());
                state_ = After;
                continue;
            }
            // a possible "\n</think>": its newline is the separator
            size_t hold = std::max(partial_suffix(buf_, close_nl), partial_suffix(buf_, kThinkClose));
            reasoning.append(buf_, 0, buf_.size() - hold);
            buf_.erase(0, buf_.size() - hold);
            break;
        } else if (state_ == After) {
            drop_first_newline(buf_);
            state_ = Content;
        } else {
            normal += buf_;
            buf_.clear();
        }
    }
}

// RippleToolDetector's streaming side: the text before the first call, less the newline the template puts before it
void RippleSplitter::content_step(const std::string& text, bool final, Pieces& out) {
    static const std::string call_nl = "\n" + kCallOpen;
    if (!tools_) {
        if (!text.empty()) out.push_back({false, text});
        return;
    }
    if (!held.empty()) {
        held += text;
        return;
    }
    pending_ += text;
    size_t p = pending_.find(kCallOpen);
    if (p != std::string::npos) {
        std::string before = pending_.substr(0, p);
        drop_last_newline(before);
        if (!before.empty()) out.push_back({false, std::move(before)});
        held = pending_.substr(p);
        pending_.clear();
        return;
    }
    size_t hold = final ? 0 : std::max(partial_suffix(pending_, call_nl), partial_suffix(pending_, kCallOpen));
    if (pending_.size() > hold) out.push_back({false, pending_.substr(0, pending_.size() - hold)});
    pending_.erase(0, pending_.size() - hold);
}

RippleSplitter::Pieces RippleSplitter::feed(const std::string& text, bool final) {
    Pieces out;
    std::string reasoning, normal;
    if (!separate_) normal = text;
    else {
        reasoning_step(text, reasoning, normal);
        if (final) {
            // RippleReasoningDetector.finish: what was held back for a marker that never came
            (state_ == Lead || state_ == Think ? reasoning : normal) += buf_;
            buf_.clear();
        }
    }
    if (!reasoning.empty()) out.push_back({true, std::move(reasoning)});
    content_step(normal, final, out);
    return out;
}

}  // namespace glm
