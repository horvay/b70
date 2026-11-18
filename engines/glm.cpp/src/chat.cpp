#include "chat.h"
#include <cstdlib>
#include <optional>
#include "rt.h"

namespace glm {

namespace {

// A template value: nullptr is Jinja's Undefined (a missing key or attribute), a Json Null is Python's None.
using V = const Json*;

const Json kChar = Json::make_str("c");   // stands for str[0] (Jinja `s.0`): only its kind is ever looked at

V attr(V x, const char* k) {                // Jinja x.k (and x['k'])
    GLM_CHECK(x, "chat template: '%s' of an undefined value", k);
    return x->kind == Json::Obj ? x->obj.find(k) : nullptr;
}

V item0(V x) {                               // Jinja x.0
    GLM_CHECK(x, "chat template: item 0 of an undefined value");
    if (x->kind == Json::Arr) return x->arr.empty() ? nullptr : &x->arr[0];
    if (x->kind == Json::Str) return x->str.empty() ? nullptr : &kChar;
    return nullptr;
}

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
bool is_mapping(V x) { return x && x->kind == Json::Obj; }
bool is_iterable(V x) { return !x || x->kind == Json::Str || x->kind == Json::Arr || x->kind == Json::Obj; }
bool eq_str(V x, const char* s) { return x && x->kind == Json::Str && x->str == s; }

bool eq(V a, V b) {                          // Python ==
    if (!a || !b) return !a && !b;           // Undefined == Undefined
    auto numeric = [](V x) { return x->kind == Json::Num || x->kind == Json::Bool; };
    if (numeric(a) && numeric(b)) {
        auto val = [](V x) { return x->kind == Json::Bool ? (double)x->b : x->num; };
        if (a->kind == Json::Num && b->kind == Json::Num && a->integer && b->integer) return a->str == b->str;
        return val(a) == val(b);
    }
    if (a->kind != b->kind) return false;
    switch (a->kind) {
        case Json::Null: return true;
        case Json::Str: return a->str == b->str;
        case Json::Arr:
            if (a->arr.size() != b->arr.size()) return false;
            for (size_t i = 0; i < a->arr.size(); i++)
                if (!eq(&a->arr[i], &b->arr[i])) return false;
            return true;
        case Json::Obj:
            if (a->obj.size() != b->obj.size()) return false;
            for (auto& [k, v] : a->obj)
                if (!eq(&v, b->obj.find(k))) return false;
            return true;
        default: return false;
    }
}

// ------------------------------------------------------------------------------------------- Python str helpers

uint32_t cp_at(const std::string& s, size_t i, int& w) {
    unsigned char c = s[i];
    w = c < 0x80 ? 1 : c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : 2;
    if (i + w > s.size()) { w = 1; return c; }
    uint32_t cp = w == 1 ? c : c & (0x7F >> w);
    for (int k = 1; k < w; k++) cp = cp << 6 | (s[i + k] & 0x3F);
    return cp;
}

bool py_isspace(uint32_t c) {
    return (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x20) || c == 0x85 || c == 0xA0 || c == 0x1680 ||
           (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

std::string py_strip(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b) {
        int w;
        uint32_t c = cp_at(s, a, w);
        if (!py_isspace(c)) break;
        a += w;
    }
    while (b > a) {
        size_t p = b - 1;
        while (p > a && ((unsigned char)s[p] & 0xC0) == 0x80) p--;
        int w;
        if (!py_isspace(cp_at(s, p, w))) break;
        b = p;
    }
    return s.substr(a, b - a);
}

std::vector<std::string> py_split(const std::string& s, const std::string& sep) {
    std::vector<std::string> out;
    size_t p = 0;
    for (;;) {
        size_t q = s.find(sep, p);
        if (q == std::string::npos) { out.push_back(s.substr(p)); return out; }
        out.push_back(s.substr(p, q - p));
        p = q + sep.size();
    }
}

std::string replace_all(std::string s, const std::string& from, const std::string& to) {
    for (size_t p = 0; (p = s.find(from, p)) != std::string::npos; p += to.size()) s.replace(p, from.size(), to);
    return s;
}

// Python repr() of a str. str.isprintable is approximated outside ASCII (C1 controls, the separators and the
// common format characters are escaped; unassigned code points are not known here).
std::string py_repr_str(const std::string& s) {
    char q = s.find('\'') != std::string::npos && s.find('"') == std::string::npos ? '"' : '\'';
    std::string o(1, q);
    char buf[16];
    for (size_t i = 0; i < s.size();) {
        int w;
        uint32_t c = cp_at(s, i, w);
        std::string ch = s.substr(i, w);
        i += w;
        if (c == (uint32_t)q || c == '\\') { o += '\\'; o += (char)c; continue; }
        if (c == '\t') { o += "\\t"; continue; }
        if (c == '\n') { o += "\\n"; continue; }
        if (c == '\r') { o += "\\r"; continue; }
        bool printable = c >= 0x20 && c != 0x7F;
        if (c >= 0x80) {
            printable = !(c <= 0xA0 || c == 0xAD || c == 0x1680 || c == 0x180E || (c >= 0x2000 && c <= 0x200F) ||
                          (c >= 0x2028 && c <= 0x202F) || (c >= 0x205F && c <= 0x206F) || c == 0x3000 ||
                          (c >= 0xD800 && c <= 0xF8FF) || c == 0xFEFF || (c >= 0xFFF9 && c <= 0xFFFB) ||
                          (c >= 0xE0000 && c <= 0xE007F) || c >= 0xF0000);
        }
        if (printable) { o += ch; continue; }
        if (c <= 0xFF) snprintf(buf, sizeof buf, "\\x%02x", c);
        else if (c <= 0xFFFF) snprintf(buf, sizeof buf, "\\u%04x", c);
        else snprintf(buf, sizeof buf, "\\U%08x", c);
        o += buf;
    }
    return o + q;
}

std::string py_num(const Json& x) {
    if (x.integer) return x.str;
    return json_detail::float_repr(x.num);
}

std::string py_repr(const Json& x) {
    switch (x.kind) {
        case Json::Null: return "None";
        case Json::Bool: return x.b ? "True" : "False";
        case Json::Num: return py_num(x);
        case Json::Str: return py_repr_str(x.str);
        case Json::Arr: {
            std::string o = "[";
            for (size_t i = 0; i < x.arr.size(); i++) o += (i ? ", " : "") + py_repr(x.arr[i]);
            return o + "]";
        }
        case Json::Obj: {
            std::string o = "{";
            bool first = true;
            for (auto& [k, v] : x.obj) {
                o += (first ? "" : ", ") + py_repr_str(k) + ": " + py_repr(v);
                first = false;
            }
            return o + "}";
        }
    }
    return "";
}

std::string py_str(V x) {                    // Jinja {{ x }}
    if (!x) return "";
    if (x->kind == Json::Str) return x->str;
    return py_repr(*x);
}

std::string py_lower_ascii(std::string s) {
    for (auto& c : s)
        if (c >= 'A' && c <= 'Z') c = c - 'A' + 'a';
    return s;
}

bool ends_with(const std::string& s, const std::string& t) {
    return s.size() >= t.size() && s.compare(s.size() - t.size(), t.size(), t) == 0;
}

// ------------------------------------------------------------------------------------------- the template

struct Renderer {
    const Json& msgs;
    V tools;
    const Json& vars;
    bool nq_hook;

    V function_of(V tool) {                  // {% if 'function' in tool %}{% set tool = tool['function'] %}
        GLM_CHECK(is_mapping(tool), "chat template: a tool is not an object");
        V f = tool->obj.find("function");
        if (!f) return tool;
        GLM_CHECK(is_mapping(f), "chat template: a tool's function is not an object");
        return f;
    }

    std::string tool_to_json(V tool) {
        GLM_CHECK(is_mapping(tool), "chat template: a tool is not an object");
        std::string o = "{";
        bool first = true;
        for (auto& [k, v] : tool->obj) {
            if (k == "defer_loading" || k == "strict") continue;
            if (!first) o += ", ";
            first = false;
            o += "\"" + k + "\": " + dump(v);
        }
        return o + "}";
    }

    std::string tool_references_to_response(V refs) {
        GLM_CHECK(truthy(tools), "chat template: a tool_reference without tools");
        std::string o = "<tool_response><tools>\n";
        GLM_CHECK(refs && refs->kind == Json::Arr, "chat template: tool references are not a list");
        for (auto& tr : refs->arr)
            for (auto& tool : tools->arr) {
                V t = function_of(&tool);
                if (eq(attr(t, "name"), attr(&tr, "name"))) o += tool_to_json(t) + "\n";
            }
        return o + "</tools></tool_response>";
    }

    std::string visible_text(V c) {
        if (is_str(c)) return c->str;
        if (is_iterable(c) && !is_mapping(c)) {
            std::string o;
            if (!c) return o;
            for (auto& item : c->arr) {
                V it = &item;
                if (is_mapping(it) && eq_str(attr(it, "type"), "text")) {
                    o += py_str(attr(it, "text"));
                } else if (is_str(it)) {
                    o += it->str;
                } else if (is_mapping(it)) {
                    V ty = attr(it, "type");
                    static const char* media[] = {"image", "image_url", "video", "video_url", "audio", "audio_url",
                                                  "input_audio"};
                    bool m = false;
                    for (auto s : media) m |= eq_str(ty, s);
                    if (m)
                        o += "<reminder>You are unable to process this " +
                             replace_all(replace_all(ty->str, "_url", ""), "input_", "") +
                             " because you don't have multi-modal input ability. Try different methods.</reminder>";
                }
            }
            return o;
        }
        return py_str(c);
    }

    static std::string tool_response(const std::string& t) { return "<tool_response>" + t + "</tool_response>"; }

    bool refs_output(V out) {                // an output that is a non-empty list of tool references
        return is_iterable(out) && !is_str(out) && !is_mapping(out) && truthy(out) &&
               eq_str(attr(item0(out), "type"), "tool_reference");
    }

    std::string render_tool_response(V m) {
        V c = attr(m, "content");
        if (is_str(c)) return tool_response(c->str);
        if (truthy(c) && !is_mapping(c) && eq_str(attr(item0(c), "type"), "tool_reference"))
            return tool_references_to_response(c);
        if (is_list_of_outputs(m)) {
            std::string o;
            for (auto& tr : c->arr) {
                V out = attr(&tr, "output");
                o += refs_output(out) ? tool_references_to_response(out) : tool_response(visible_text(out));
            }
            return o;
        }
        return tool_response(visible_text(c));
    }

    std::string id_of(V obj) {
        V t = attr(obj, "tool_call_id");
        if (truthy(t)) return py_str(t);
        V i = attr(obj, "id");
        if (truthy(i)) return py_str(i);
        return "";
    }

    bool is_list_of_outputs(V m) {
        V c = attr(m, "content");
        return truthy(c) && attr(item0(c), "output") != nullptr;
    }

    void tool_block(size_t i, std::string& o) {
        size_t n = msgs.arr.size(), bs = i, end = i;
        for (size_t j = bs; j < n; j++) {
            if (!eq_str(attr(&msgs.arr[j], "role"), "tool")) break;
            end = j;
        }
        V tcs = nullptr;
        if (bs > 0 && eq_str(attr(&msgs.arr[bs - 1], "role"), "assistant") && truthy(attr(&msgs.arr[bs - 1], "tool_calls")))
            tcs = attr(&msgs.arr[bs - 1], "tool_calls");
        bool can_sort = truthy(tcs);
        if (can_sort) {
            GLM_CHECK(tcs->kind == Json::Arr, "chat template: tool_calls is not a list");
            for (size_t k = bs; k <= end; k++) {
                V m = &msgs.arr[k];
                if (is_list_of_outputs(m)) {
                    for (auto& e : attr(m, "content")->arr)
                        if (id_of(&e).empty()) can_sort = false;
                } else if (id_of(m).empty()) {
                    can_sort = false;
                }
            }
            for (auto& tc : tcs->arr)
                if (id_of(&tc).empty()) can_sort = false;
        }
        if (!can_sort) {
            for (size_t k = bs; k <= end; k++) o += render_tool_response(&msgs.arr[k]);
            return;
        }
        for (auto& tc : tcs->arr) {
            std::string tc_id = id_of(&tc);
            for (size_t k = bs; k <= end; k++) {
                V m = &msgs.arr[k];
                if (is_list_of_outputs(m)) {
                    for (auto& e : attr(m, "content")->arr) {
                        if (id_of(&e) != tc_id) continue;
                        V out = attr(&e, "output");
                        o += refs_output(out) ? tool_references_to_response(out) : tool_response(visible_text(out));
                    }
                } else if (id_of(m) == tc_id) {
                    o += render_tool_response(m);
                }
            }
        }
    }

    void assistant(size_t i, ptrdiff_t last_user, std::string& o) {
        V m = &msgs.arr[i];
        o += "<|assistant|>";
        std::string content = visible_text(attr(m, "content"));
        std::optional<std::string> rc;
        V r = attr(m, "reasoning_content");
        if (is_str(r)) {
            rc = r->str;
        } else if (content.find("</think>") != std::string::npos) {
            auto parts = py_split(content, "</think>");
            rc = py_split(parts[0], "<think>").back();
            content = parts.back();
        }
        V ct = vars.get("clear_thinking");
        if (((ct && !truthy(ct)) || (ptrdiff_t)i > last_user) && rc) o += "<think>" + *rc + "</think>";
        else o += "<think></think>";
        std::string s = py_strip(content);
        if (!s.empty()) o += s;
        V tcs = attr(m, "tool_calls");
        if (!truthy(tcs)) return;
        GLM_CHECK(tcs->kind == Json::Arr, "chat template: tool_calls is not a list");
        for (auto& tc0 : tcs->arr) {
            V tc = &tc0;
            V f = attr(tc, "function");
            if (truthy(f)) tc = f;
            V name = attr(tc, "name");
            GLM_CHECK(is_str(name), "chat template: a tool call without a string name");
            o += "<tool_call>" + name->str;
            V args = attr(tc, "arguments");
            GLM_CHECK(is_mapping(args), "chat template: tool call arguments are not an object");
            for (auto& [k, v] : args->obj)
                o += "<arg_key>" + k + "</arg_key><arg_value>" + (v.kind == Json::Str ? v.str : dump(v)) + "</arg_value>";
            o += "</tool_call>";
        }
    }

    std::string render() {
        GLM_CHECK(msgs.kind == Json::Arr, "chat template: messages is not a list");
        std::string o = "[gMASK]<sop>";
        V re = vars.get("reasoning_effort");
        bool known = eq_str(re, "low") || eq_str(re, "high");
        o += "<|system|>Reasoning Effort: ";
        o += !known ? "Max" : re->str == "low" ? "Low" : "High";
        if (truthy(tools)) {
            GLM_CHECK(tools->kind == Json::Arr, "chat template: tools is not a list");
            o += "<|system|>\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
                 "You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n";
            for (auto& tool : tools->arr) {
                V t = function_of(&tool);
                V dl = attr(t, "defer_loading");
                if (!dl || !truthy(dl)) o += tool_to_json(t) + "\n";
            }
            o += "</tools>\n\nFor each function call, output the function name and arguments within the following "
                 "XML format:\n<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}"
                 "</arg_value><arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>";
        }
        ptrdiff_t last_user = -1;
        for (size_t i = 0; i < msgs.arr.size(); i++)
            if (eq_str(attr(&msgs.arr[i], "role"), "user")) last_user = (ptrdiff_t)i;
        for (size_t i = 0; i < msgs.arr.size(); i++) {
            V m = &msgs.arr[i];
            V role = attr(m, "role");
            if (eq_str(role, "user")) {
                o += "<|user|>" + visible_text(attr(m, "content"));
            } else if (eq_str(role, "assistant")) {
                assistant(i, last_user, o);
            } else if (eq_str(role, "tool")) {
                if (i == 0 || !eq_str(attr(&msgs.arr[i - 1], "role"), "tool")) {
                    o += "<|observation|>";
                    tool_block(i, o);
                }
            } else if (eq_str(role, "system")) {
                o += "<|system|>" + visible_text(attr(m, "content"));
            }
        }
        bool gen = truthy(vars.get("add_generation_prompt"));
        if (gen) o += "<|assistant|><think>";
        V nq = vars.get("nq_prefill");
        if (nq_hook && gen && truthy(nq)) o += py_str(nq);
        return o;
    }
};

const char* map_effort(const std::string& e) {
    static const char* tab[][2] = {{"none", "low"}, {"minimal", "low"}, {"low", "low"}, {"medium", "high"},
                                   {"high", "high"}, {"xhigh", "max"}, {"max", "max"}};
    for (auto& t : tab)
        if (e == t[0]) return t[1];
    return nullptr;
}

}  // namespace

std::string py_strip_text(const std::string& s) { return py_strip(s); }

std::string py_str_value(const Json& x) { return py_str(&x); }

std::string render_chat(const Json& messages, const Json* tools, const Json& vars, bool nq_prefill_hook) {
    static const Json none = Json::make_obj();
    Renderer r{messages, tools, vars.kind == Json::Obj ? vars : none, nq_prefill_hook};
    return r.render();
}

Json normalize_messages(const Json& messages) {
    GLM_CHECK(messages.kind == Json::Arr, "messages is not a list");
    Json out = Json::make_arr();
    for (auto& m0 : messages.arr) {
        GLM_CHECK(m0.kind == Json::Obj, "a message is not an object");
        Json m = m0;
        V c = m.get("content");
        if (!c || c->kind == Json::Null) m.set("content", Json::make_str(""));
        if (eq_str(m.get("role"), "developer")) m.set("role", Json::make_str("system"));
        V tcs = m.get("tool_calls");
        if (truthy(tcs)) {
            GLM_CHECK(tcs->kind == Json::Arr, "tool_calls is not a list");
            Json list = Json::make_arr();
            for (auto& tc0 : tcs->arr) {
                Json tc = tc0;
                GLM_CHECK(tc.kind == Json::Obj, "a tool call is not an object");
                Json* fn = tc.obj.find("function");
                if (!fn) fn = &tc;
                GLM_CHECK(fn->kind == Json::Obj, "a tool call's function is not an object");
                V a = fn->get("arguments");
                if (is_str(a)) {
                    std::string text = a->str;
                    auto parsed = Json::try_parse(text.empty() ? "{}" : text);
                    if (parsed) {
                        fn->set("arguments", std::move(*parsed));
                    } else {
                        Json w = Json::make_obj();
                        w.set("arguments", Json::make_str(text));
                        fn->set("arguments", std::move(w));
                    }
                }
                if (!is_mapping(fn->get("arguments"))) fn->set("arguments", Json::make_obj());
                list.push(std::move(tc));
            }
            m.set("tool_calls", std::move(list));
        }
        out.push(std::move(m));
    }
    return out;
}

ServeConfig ServeConfig::from_env() {
    ServeConfig d;
    const char* e = getenv("GLM_SERVE_REASONING_EFFORT");
    if (e) {
        const char* v = map_effort(py_lower_ascii(py_strip(e)));
        d.effort = v ? v : "low";
    }
    const char* c = getenv("GLM_SERVE_CLEAR_THINKING");
    if (c) {
        std::string s = c;
        d.clear_thinking = !(s == "0" || s == "false" || s == "no" || s.empty());
    }
    return d;
}

ChatPrompt render_prompt(const Json& messages, const Json* tools, const Json* chat_template_kwargs,
                         const Json* reasoning_effort, const ServeConfig& config) {
    Json kw = Json::make_obj();
    if (truthy(chat_template_kwargs)) {
        GLM_CHECK(chat_template_kwargs->kind == Json::Obj, "chat_template_kwargs is not an object");
        kw = *chat_template_kwargs;
    }
    if (reasoning_effort && reasoning_effort->kind != Json::Null && !kw.has("reasoning_effort"))
        kw.set("reasoning_effort", *reasoning_effort);
    V et = kw.get("enable_thinking");
    ChatPrompt p;
    p.thinking = !(et && et->kind == Json::Bool && !et->b);

    // Renderer.render
    kw.obj.erase("enable_thinking");
    V e = kw.get("reasoning_effort");
    std::string effort = config.effort;
    if (e && e->kind != Json::Null) {
        const char* v = map_effort(py_lower_ascii(py_strip(py_str(e))));
        if (v) effort = v;
    }
    kw.obj.erase("reasoning_effort");
    if (effort != "max") kw.set("reasoning_effort", Json::make_str(effort));
    if (!kw.has("clear_thinking")) kw.set("clear_thinking", Json::make_bool(config.clear_thinking));
    kw.set("add_generation_prompt", Json::make_bool(true));
    Json msgs = normalize_messages(messages);
    p.text = render_chat(msgs, truthy(tools) ? tools : nullptr, kw, config.nq_prefill_hook);
    if (!p.thinking && ends_with(p.text, "<think>")) p.text += "</think>";
    size_t o = p.text.rfind("<think>"), c = p.text.rfind("</think>");
    p.reply_in_thought = p.thinking && o != std::string::npos && (c == std::string::npos || o > c);
    return p;
}

}  // namespace glm
