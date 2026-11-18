// Gemma 4's chat_template.jinja (two Gemma 4 31B-based models; "Published: 2026-07-09"), ported by hand: every macro and
// branch in its order, with Jinja's semantics where they matter (undefined vs None, Environment.getitem, dictsort =
// case-insensitive stable sort, trim = str.strip, sequence includes mappings and Undefined). Plus what SGLang does around
// it for `b70 serve`: serving_chat's message shaping, and the gemma4 reasoning and tool-call parsers.
#include <utf8proc.h>
#include <algorithm>
#include <cstdlib>
#include <random>
#include "chat.h"
#include "rt.h"

namespace glm {

namespace {

// A template value: nullptr is Jinja's Undefined (a missing key), a Json Null is Python's None.
using V = const Json*;

const Json kNone;
const std::string kQ = "<|\"|>";                 // the template's string quote, a special token
const std::string kChannelOpen = "<|channel>", kChannelClose = "<channel|>";
const std::string kThoughtOpen = "<|channel>thought\n";
const std::string kCallOpen = "<|tool_call>", kCallClose = "<tool_call|>";

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
bool is_map(V x) { return x && x->kind == Json::Obj; }
bool is_none(V x) { return x && x->kind == Json::Null; }
bool eq_str(V x, const char* s) { return is_str(x) && x->str == s; }

// x['k']: Environment.getitem falls back to getattr, so anything but a mapping holding k is Undefined, and subscripting
// Undefined itself raises. (dict.items would be found as an attribute for 'items', but that key is only read behind
// `is mapping`, which a bound method fails, the same as Undefined.)
V gi(V x, const char* k) {
    GLM_CHECK(x, "chat template: '%s' of an undefined value", k);
    return x->kind == Json::Obj ? x->obj.find(k) : nullptr;
}

// m.get('k'): None when absent. Only a mapping has .get in the sandbox; on anything else the call raises.
V mget(V m, const char* k) {
    GLM_CHECK(is_map(m), "chat template: .get('%s') of a value that is not a mapping", k);
    V v = m->obj.find(k);
    return v ? v : &kNone;
}

// {{ x }}
std::string str_of(V x) { return x ? py_str_value(*x) : ""; }

// Python ==, for tc.get('id') == follow.get('tool_call_id'): None == None holds, so calls and responses that both lack
// an id all match; numbers compare by value whatever their spelling
bool py_eq(V a, V b) {
    if (a->kind == Json::Num && b->kind == Json::Num) {
        if (a->integer && b->integer) return a->str == b->str;
        return a->num == b->num;
    }
    if (a->kind != b->kind) return false;
    switch (a->kind) {
        case Json::Null: return true;
        case Json::Bool: return a->b == b->b;
        case Json::Str: return a->str == b->str;
        default: return dump(*a) == dump(*b);
    }
}

// Python str.upper / str.lower by the simple Unicode case maps; the special ones (ß -> SS) never occur in a JSON schema
// type or a tool's parameter name
template <class F> std::string map_case(const std::string& s, F&& f) {
    std::string o;
    o.reserve(s.size());
    auto p = reinterpret_cast<const utf8proc_uint8_t*>(s.data());
    for (size_t i = 0; i < s.size();) {
        utf8proc_int32_t c;
        auto w = utf8proc_iterate(p + i, s.size() - i, &c);
        if (w <= 0) { o += s[i++]; continue; }
        utf8proc_uint8_t b[4];
        auto n = utf8proc_encode_char(f(c), b);
        o.append(reinterpret_cast<char*>(b), n);
        i += w;
    }
    return o;
}
std::string to_upper(const std::string& s) { return map_case(s, utf8proc_toupper); }
std::string to_lower(const std::string& s) { return map_case(s, utf8proc_tolower); }

// x | upper: str(x).upper(), so Undefined gives '' and a list its repr in capitals
std::string upper_of(V x) { return to_upper(str_of(x)); }

// x | dictsort: by lowercased key, stable
std::vector<const JsonObj::Item*> dictsort(V x) {
    GLM_CHECK(is_map(x), "chat template: dictsort of a value that is not a mapping");
    std::vector<std::pair<std::string, const JsonObj::Item*>> keyed;
    keyed.reserve(x->obj.size());
    for (auto& it : x->obj) keyed.emplace_back(to_lower(it.first), &it);
    std::stable_sort(keyed.begin(), keyed.end(), [](auto& a, auto& b) { return a.first < b.first; });
    std::vector<const JsonObj::Item*> o;
    o.reserve(keyed.size());
    for (auto& k : keyed) o.push_back(k.second);
    return o;
}

// for item in x: a list's items, a string's characters, a mapping's keys, nothing for Undefined; the rest raise
std::vector<Json> iter_of(V x) {
    std::vector<Json> o;
    if (!x) return o;
    if (x->kind == Json::Arr) return x->arr;
    if (x->kind == Json::Obj) {
        for (auto& it : x->obj) o.push_back(Json::make_str(it.first));
        return o;
    }
    GLM_CHECK(x->kind == Json::Str, "chat template: iterating a value that is not iterable");
    auto p = reinterpret_cast<const utf8proc_uint8_t*>(x->str.data());
    for (size_t i = 0; i < x->str.size();) {
        utf8proc_int32_t c;
        auto w = utf8proc_iterate(p + i, x->str.size() - i, &c);
        if (w <= 0) w = 1;
        o.push_back(Json::make_str(x->str.substr(i, w)));
        i += w;
    }
    return o;
}

std::string quoted_list(V x) {               // <|"|>a<|"|>,<|"|>b<|"|>
    std::string o;
    auto items = iter_of(x);
    for (size_t i = 0; i < items.size(); i++) o += kQ + str_of(&items[i]) + kQ + (i + 1 < items.size() ? "," : "");
    return o;
}

// format_argument. Undefined passes `is sequence` (it has __len__ and __getitem__) and iterates to nothing.
std::string format_argument(V a, bool escape_keys = true) {
    if (!a) return "[]";
    switch (a->kind) {
        case Json::Null: return "null";
        case Json::Str: return kQ + a->str + kQ;
        case Json::Bool: return a->b ? "true" : "false";
        case Json::Num: return py_str_value(*a);
        case Json::Obj: {
            std::string o = "{";
            bool first = true;
            for (auto* it : dictsort(a)) {
                if (!first) o += ",";
                first = false;
                o += escape_keys ? kQ + it->first + kQ : it->first;
                o += ":" + format_argument(&it->second, escape_keys);
            }
            return o + "}";
        }
        case Json::Arr: {
            std::string o = "[";
            for (size_t i = 0; i < a->arr.size(); i++) {
                if (i) o += ",";
                o += format_argument(&a->arr[i], escape_keys);
            }
            return o + "]";
        }
    }
    return "";
}

// format_parameters. Its `required` argument is never read, so it is left out here.
std::string format_parameters(V properties, bool filter_keys = false) {
    static const char* standard[] = {"description", "type", "properties", "required", "nullable"};
    std::string o;
    bool found_first = false;
    for (auto* it : dictsort(properties)) {
        const std::string& key = it->first;
        V value = &it->second;
        if (filter_keys && std::any_of(std::begin(standard), std::end(standard), [&](const char* s) { return key == s; }))
            continue;
        if (found_first) o += ",";
        found_first = true;
        bool add_comma = false;
        auto comma = [&] {
            if (add_comma) o += ",";
            add_comma = true;
        };
        o += key + ":{";
        V desc = gi(value, "description");
        if (truthy(desc)) {
            o += "description:" + kQ + str_of(desc) + kQ;
            add_comma = true;
        }
        std::string type = upper_of(gi(value, "type"));
        if (type == "STRING") {
            V en = gi(value, "enum");
            if (truthy(en)) {
                comma();
                o += "enum:" + format_argument(en);
            }
        } else if (type == "ARRAY") {
            V items = gi(value, "items");
            if (is_map(items) && truthy(items)) {
                comma();
                o += "items:{";
                bool first_item = true;
                for (auto* ii : dictsort(items)) {
                    V iv = &ii->second;
                    if (is_none(iv)) continue;
                    if (!first_item) o += ",";
                    first_item = false;
                    if (ii->first == "properties") {
                        o += "properties:{";
                        if (is_map(iv)) o += format_parameters(iv);
                        o += "}";
                    } else if (ii->first == "required") {
                        o += "required:[" + quoted_list(iv) + "]";
                    } else if (ii->first == "type") {
                        if (is_str(iv)) {
                            Json up = Json::make_str(to_upper(iv->str));
                            o += "type:" + format_argument(&up);
                        } else {
                            Json up = Json::make_arr();
                            for (auto& x : iter_of(iv)) up.push(Json::make_str(upper_of(&x)));
                            o += "type:" + format_argument(&up);
                        }
                    } else {
                        o += ii->first + ":" + format_argument(iv);
                    }
                }
                o += "}";
            }
        }
        if (truthy(gi(value, "nullable"))) {
            comma();
            o += "nullable:true";
        }
        if (type == "OBJECT") {
            V props = gi(value, "properties");
            if (is_map(props)) {
                comma();
                o += "properties:{" + format_parameters(props) + "}";
            } else if (is_map(value)) {
                comma();
                o += "properties:{" + format_parameters(value, true) + "}";
            }
            V req = gi(value, "required");
            if (truthy(req)) {
                comma();
                o += "required:[" + quoted_list(req) + "]";
            }
        }
        comma();
        o += "type:" + kQ + type + kQ + "}";
    }
    return o;
}

// format_function_declaration. As in the template, parameters without a type leave their brace open.
std::string format_function_declaration(V tool) {
    V fn = gi(tool, "function");
    std::string o = "declaration:" + str_of(gi(fn, "name")) + "{description:" + kQ + str_of(gi(fn, "description")) + kQ;
    V params = gi(fn, "parameters");
    if (truthy(params)) {
        o += ",parameters:{";
        V props = gi(params, "properties");
        if (truthy(props)) o += "properties:{" + format_parameters(props) + "},";
        V req = gi(params, "required");
        if (truthy(req)) o += "required:[" + quoted_list(req) + "],";
        V ty = gi(params, "type");
        if (truthy(ty)) o += "type:" + kQ + upper_of(ty) + kQ + "}";
    }
    GLM_CHECK(is_map(fn), "chat template: a tool's function is not an object");
    if (V resp = fn->obj.find("response")) {
        o += ",response:{";
        V d = gi(resp, "description");
        if (truthy(d)) o += "description:" + kQ + str_of(d) + kQ + ",";
        if (upper_of(gi(resp, "type")) == "OBJECT") o += "type:" + kQ + upper_of(gi(resp, "type")) + kQ + "}";
    }
    return o + "}";
}

std::string strip_thinking(const std::string& text) {
    std::string r;
    size_t p = 0;
    for (;;) {
        size_t q = text.find(kChannelClose, p);
        std::string part = text.substr(p, q == std::string::npos ? std::string::npos : q - p);
        r += part.substr(0, part.find(kChannelOpen));
        if (q == std::string::npos) break;
        p = q + kChannelClose.size();
    }
    return py_strip_text(r);
}

std::string format_tool_response_block(const std::string& name, V response) {
    std::string o = "<|tool_response>";
    if (is_map(response)) {
        o += "response:" + name + "{";
        auto items = dictsort(response);
        for (size_t i = 0; i < items.size(); i++) {
            o += items[i]->first + ":" + format_argument(&items[i]->second, false);
            if (i + 1 < items.size()) o += ",";
        }
        o += "}";
    } else {
        o += "response:" + name + "{value:" + format_argument(response, false) + "}";
    }
    return o + "<tool_response|>";
}

// item.get('type') in [...]: the placeholder of a media part, "" for anything else
std::string media_placeholder(V part) {
    V ty = mget(part, "type");
    if (eq_str(ty, "image") || eq_str(ty, "image_url")) return "<|image|>";
    if (eq_str(ty, "audio") || eq_str(ty, "input_audio")) return "<|audio|>";
    if (eq_str(ty, "video")) return "<|video|>";
    return "";
}

const std::string& str_or_raise(V x, const char* what) {
    GLM_CHECK(is_str(x), "chat template: %s is not a string", what);
    return x->str;
}

std::string render(const Json& messages, const Json* tools, const Json& vars) {
    GLM_CHECK(messages.kind == Json::Arr, "chat template: messages is not a list");
    bool enable_thinking = truthy(vars.get("enable_thinking"));
    bool preserve_thinking = truthy(vars.get("preserve_thinking"));
    bool has_tools = truthy(tools);
    std::vector<const Json*> loop;
    for (auto& m : messages.arr) loop.push_back(&m);
    std::string o = "<bos>";
    enum { None_, Think, Tool, ToolCall, ToolResponse } prev_type = None_;
    V first_role = messages.arr.empty() ? nullptr : gi(&messages.arr[0], "role");
    bool first_system = eq_str(first_role, "system") || eq_str(first_role, "developer");
    if (enable_thinking || has_tools || first_system) {
        o += "<|turn>system\n";
        if (enable_thinking) {
            o += "<|think|>\n";
            prev_type = Think;
        }
        if (first_system) {
            V c = gi(&messages.arr[0], "content");
            if (is_str(c)) {
                o += py_strip_text(c->str);
            } else if (!c || c->kind == Json::Arr || c->kind == Json::Obj) {   // `is sequence`: mappings too, and Undefined
                for (auto& item : iter_of(c)) o += py_strip_text(str_of(gi(&item, "text"))) + " ";
            }
            loop.erase(loop.begin());
        }
        if (has_tools) {
            GLM_CHECK(tools->kind == Json::Arr, "chat template: tools is not a list");
            for (auto& tool : tools->arr) o += "<|tool>" + py_strip_text(format_function_declaration(&tool)) + "<tool|>";
            prev_type = Tool;
        }
        o += "<turn|>\n";
    }

    ptrdiff_t last_user_idx = -1;
    for (size_t i = 0; i < loop.size(); i++)
        if (eq_str(gi(loop[i], "role"), "user")) last_user_idx = (ptrdiff_t)i;

    V prev_non_tool_role = nullptr;              // None at first
    for (size_t i = 0; i < loop.size(); i++) {
        V message = loop[i];
        V mrole = gi(message, "role");
        if (eq_str(mrole, "tool")) continue;
        prev_type = None_;
        std::string role = eq_str(mrole, "assistant") ? "model" : mrole ? (is_str(mrole) ? mrole->str : "") : "";
        bool is_model = role == "model";
        if (!(is_model && eq_str(prev_non_tool_role, "assistant"))) {
            GLM_CHECK(is_str(mrole), "chat template: a message role is not a string");
            o += "<|turn>" + role + "\n";
        }

        V thinking_text = mget(message, "reasoning");
        if (!truthy(thinking_text)) thinking_text = mget(message, "reasoning_content");
        V tool_calls = mget(message, "tool_calls");
        bool thinking_gate = (ptrdiff_t)i > last_user_idx || (preserve_thinking && truthy(tool_calls));
        if (truthy(thinking_text) && thinking_gate)
            o += kThoughtOpen + str_or_raise(thinking_text, "reasoning") + "\n" + kChannelClose;

        if (truthy(tool_calls)) {
            GLM_CHECK(tool_calls->kind == Json::Arr, "chat template: tool_calls is not a list");
            for (auto& tc : tool_calls->arr) {
                V function = gi(&tc, "function");
                o += "<|tool_call>call:" + str_or_raise(gi(function, "name"), "a tool call's name") + "{";
                V args = gi(function, "arguments");
                if (is_map(args)) {
                    bool first = true;
                    for (auto* it : dictsort(args)) {
                        if (!first) o += ",";
                        first = false;
                        o += it->first + ":" + format_argument(&it->second, false);
                    }
                } else {
                    GLM_CHECK(is_none(args), "chat_template: tool_calls[].function.arguments must be a JSON object (mapping), "
                              "not a string. Deserialize arguments before passing to the template.");
                }
                o += "}<tool_call|>";
            }
            prev_type = ToolCall;
        }

        bool tr_out = false;
        V tool_responses = mget(message, "tool_responses");
        if (truthy(tool_responses)) {
            // legacy (Google/Gemma native): tool_responses embedded on the assistant message
            for (auto& tr : iter_of(tool_responses)) {
                V name = gi(&tr, "name");
                static const Json unknown = Json::make_str("unknown");
                if (!truthy(name)) name = &unknown;      // default('unknown', true)
                o += format_tool_response_block(str_or_raise(name, "a tool response's name"), gi(&tr, "response"));
                tr_out = true;
                prev_type = ToolResponse;
            }
        } else if (truthy(tool_calls)) {
            // OpenAI Chat Completions: the role:tool messages right after this one
            for (size_t k = i + 1; k < loop.size(); k++) {
                V follow = loop[k];
                if (!eq_str(gi(follow, "role"), "tool")) break;
                static const Json unknown = Json::make_str("unknown");
                V tname = mget(follow, "name");
                if (!truthy(tname)) tname = &unknown;
                for (auto& tc : tool_calls->arr)
                    if (py_eq(mget(&tc, "id"), mget(follow, "tool_call_id"))) tname = gi(gi(&tc, "function"), "name");
                const std::string& name = str_or_raise(tname, "a tool response's name");
                V body = mget(follow, "content");
                if (body->kind == Json::Arr || body->kind == Json::Obj) {   // `is sequence and is not string`
                    std::string text;
                    auto parts = iter_of(body);
                    for (auto& part : parts)
                        if (eq_str(mget(&part, "type"), "text"))
                            text += str_or_raise(mget(&part, "text"), "a text part");   // (absent: None, past the default(''))
                    Json text_json = Json::make_str(std::move(text));
                    o += format_tool_response_block(name, &text_json);
                    for (auto& part : parts) o += media_placeholder(&part);
                } else {
                    o += format_tool_response_block(name, body);
                }
                tr_out = true;
                prev_type = ToolResponse;
            }
        }

        std::string captured;
        V content = mget(message, "content");
        if (is_str(content)) {
            captured = is_model ? strip_thinking(content->str) : py_strip_text(content->str);
        } else if (content->kind == Json::Arr || content->kind == Json::Obj) {
            for (auto& item : iter_of(content)) {
                if (eq_str(mget(&item, "type"), "text")) {
                    V t = gi(&item, "text");
                    captured += is_model ? strip_thinking(str_or_raise(t, "a text part")) : py_strip_text(str_of(t));
                } else {
                    captured += media_placeholder(&item);
                }
            }
        }
        o += captured;
        bool has_content = !py_strip_text(captured).empty();

        V next_role = nullptr;
        bool next_found = false;
        for (size_t j = i + 1; j < loop.size() && !next_found; j++) {
            V r = gi(loop[j], "role");
            if (!eq_str(r, "tool")) {
                next_role = r;
                next_found = true;
            }
        }
        bool continues_into_next = is_model && eq_str(next_role, "assistant") && (!truthy(tool_calls) || tr_out);
        if (prev_type == ToolCall && !tr_out) o += "<|tool_response>";
        else if (continues_into_next) {}
        else if (!(tr_out && !has_content && !next_found)) o += "<turn|>\n";
        prev_non_tool_role = mrole;
    }

    bool gen = truthy(vars.get("add_generation_prompt"));
    if (gen) {
        if (prev_type != ToolResponse && prev_type != ToolCall) {
            o += "<|turn>model\n";
            if (!enable_thinking) o += kThoughtOpen + kChannelClose;
        } else if (prev_type == ToolResponse && enable_thinking) {
            o += kThoughtOpen;
        }
        // the client's hook, appended to every model's template (exl3_wire.ts): an open or closed thought after the prompt
        V nq = vars.get("nq_prefill");
        if (truthy(nq)) o += str_of(nq);
    }
    return o;
}

// serving_chat's shaping of a request's messages for a Jinja template: argument strings parsed (strict:
// normalize_assistant_tool_call_arguments), content null -> "", content parts cut to {text, image, video, audio,
// tool_reference} (other types and non-objects dropped), and a tool message's all-text parts joined with spaces. Roles
// stay: unlike normalize_messages, a later `developer` stays `developer` here, as in SGLang. Not ported: pydantic's model
// (it drops `reasoning` and `tool_responses`, adds description None to a tool without one) and the media-only reordering
// of tool results; so those fields reach the template here.
Json shape_messages(const Json& messages) {
    GLM_CHECK(messages.kind == Json::Arr, "messages is not a list");
    Json out = Json::make_arr();
    for (auto& m0 : messages.arr) {
        GLM_CHECK(m0.kind == Json::Obj, "a message is not an object");
        Json m = m0;
        V role = m.get("role");
        if (eq_str(role, "assistant"))
            if (Json* tcs = m.obj.find("tool_calls"); tcs && tcs->kind == Json::Arr)
                for (auto& tc : tcs->arr) {
                    Json* fn = tc.kind == Json::Obj ? tc.obj.find("function") : nullptr;
                    if (!fn || fn->kind != Json::Obj) continue;
                    Json* a = fn->obj.find("arguments");
                    if (!a || a->kind != Json::Str) continue;
                    auto parsed = Json::try_parse(a->str);
                    GLM_CHECK(parsed, "Assistant tool call function.arguments must be valid JSON.");
                    GLM_CHECK(parsed->kind == Json::Obj, "Assistant tool call function.arguments must be a JSON object.");
                    *a = std::move(*parsed);
                }
        V c = m.get("content");
        if (!c || c->kind == Json::Null) m.set("content", Json::make_str(""));
        else if (c->kind == Json::Arr) {
            Json parts = Json::make_arr();
            auto typed = [](const char* t) {
                Json p = Json::make_obj();
                p.set("type", Json::make_str(t));
                return p;
            };
            bool all_text = true;
            std::string joined;
            for (auto& ch : c->arr) {
                if (ch.kind != Json::Obj) continue;
                V ty = ch.get("type");
                if (eq_str(ty, "image_url") || eq_str(ty, "input_image")) { parts.push(typed("image")); all_text = false; }
                else if (eq_str(ty, "video_url")) { parts.push(typed("video")); all_text = false; }
                else if (eq_str(ty, "audio_url")) { parts.push(typed("audio")); all_text = false; }
                else if (eq_str(ty, "tool_reference")) { parts.push(ch); all_text = false; }
                else if (eq_str(ty, "text") || eq_str(ty, "input_text")) {
                    V t = ch.get("text");
                    GLM_CHECK(is_str(t), "a text content part without text");
                    if (parts.size()) joined += " ";
                    joined += t->str;
                    Json p = typed("text");
                    p.set("text", *t);
                    parts.push(std::move(p));
                }
            }
            if (eq_str(role, "tool") && all_text) m.set("content", Json::make_str(std::move(joined)));
            else m.set("content", std::move(parts));
        }
        out.push(std::move(m));
    }
    return out;
}

// ------------------------------------------------------------------------------------------------- tool calls

bool py_ws_arg(char c) { return c == ' ' || c == '\n' || c == '\t'; }          // the parsers' own whitespace, no \r

// Python int(s) / float(s) for the ASCII spellings: digits may have single underscores between them. int: [+-]digits.
// float (the parser tries it only when s has a '.'): [+-](digits[.digits?]|.digits)([eE][+-]?digits)?.
bool digits_at(const std::string& s, size_t& i) {
    size_t a = i;
    for (;;) {
        if (i >= s.size() || s[i] < '0' || s[i] > '9') return i > a && s[i - 1] != '_';
        i++;
        if (i < s.size() && s[i] == '_') i++;
    }
}
std::string no_underscores(const std::string& s) {
    std::string o;
    for (char c : s) if (c != '_') o += c;
    return o;
}
bool parse_py_int(const std::string& s, Json& out) {
    size_t i = 0;
    if (i < s.size() && (s[i] == '+' || s[i] == '-')) i++;
    size_t d0 = i;
    if (!digits_at(s, i) || i != s.size()) return false;
    std::string d = no_underscores(s.substr(d0));
    size_t nz = d.find_first_not_of('0');
    d = nz == std::string::npos ? "0" : d.substr(nz);
    std::string lit = (s[0] == '-' && d != "0" ? "-" : "") + d;
    out = Json();
    out.kind = Json::Num;
    out.integer = true;
    out.str = lit;
    out.num = strtod(lit.c_str(), nullptr);
    return true;
}
bool parse_py_float(const std::string& s, Json& out) {
    size_t i = 0;
    if (i < s.size() && (s[i] == '+' || s[i] == '-')) i++;
    size_t w0 = i;
    bool whole = digits_at(s, i);
    if (!whole && i != w0) return false;          // an underscore right before the point
    if (i < s.size() && s[i] == '.') {
        i++;
        size_t f = i;
        bool frac = digits_at(s, i);
        if (!frac && f != i) return false;        // an underscore right after the point
        if (!whole && !frac) return false;
    } else {
        return false;                             // the caller saw a '.', after something that is not a number
    }
    if (i < s.size() && (s[i] == 'e' || s[i] == 'E')) {
        i++;
        if (i < s.size() && (s[i] == '+' || s[i] == '-')) i++;
        if (!digits_at(s, i)) return false;
    }
    if (i != s.size()) return false;
    out = Json::make_num(strtod(no_underscores(s).c_str(), nullptr));
    return true;
}

// _parse_gemma4_value, except a bare null is JSON null: the template renders None as null, and SGLang's string "null"
// comes back to the game as text.
Json parse_value(const std::string& raw) {
    std::string v = py_strip_text(raw);
    if (v == "null") return Json();
    if (v == "true") return Json::make_bool(true);
    if (v == "false") return Json::make_bool(false);
    Json n;
    if (v.find('.') != std::string::npos ? parse_py_float(v, n) : parse_py_int(v, n)) return n;
    return Json::make_str(v);
}

Json parse_args(const std::string& s);

// the scan past a nested {...} or [...]: on return i is past the closer (or at the end), strings skipped when asked
void skip_nested(const std::string& s, size_t& i, char open, char close, bool skip_strings) {
    int depth = 1;
    size_t n = s.size();
    i++;
    while (i < n && depth > 0) {
        if (skip_strings && s.compare(i, kQ.size(), kQ) == 0) {
            i += kQ.size();
            size_t e = s.find(kQ, i);
            i = e == std::string::npos ? n : e + kQ.size();
            continue;
        }
        if (s[i] == open) depth++;
        else if (s[i] == close) depth--;
        i++;
    }
}
// s[a : i - 1], where i - 1 may fall before a (an opener at the very end of the text)
std::string inner_of(const std::string& s, size_t a, size_t i) { return i - 1 > a ? s.substr(a, i - 1 - a) : ""; }

// _parse_gemma4_array
Json parse_array(const std::string& s) {
    Json items = Json::make_arr();
    size_t i = 0, n = s.size();
    while (i < n) {
        while (i < n && (s[i] == ',' || py_ws_arg(s[i]))) i++;
        if (i >= n) break;
        if (s.compare(i, kQ.size(), kQ) == 0) {
            i += kQ.size();
            size_t e = s.find(kQ, i);
            if (e == std::string::npos) { items.push(Json::make_str(s.substr(i))); break; }
            items.push(Json::make_str(s.substr(i, e - i)));
            i = e + kQ.size();
        } else if (s[i] == '{') {
            size_t a = i + 1;
            skip_nested(s, i, '{', '}', true);
            items.push(parse_args(inner_of(s, a, i)));
        } else if (s[i] == '[') {
            size_t a = i + 1;
            skip_nested(s, i, '[', ']', false);        // (SGLang's array scan does not skip strings here)
            items.push(parse_array(inner_of(s, a, i)));
        } else {
            // A ']' left at this level only comes from a string that the scan above read through. SGLang's loop would
            // spin on it for ever, appending; step over it.
            if (s[i] == ']') { i++; continue; }
            size_t a = i;
            while (i < n && s[i] != ',' && s[i] != ']') i++;
            items.push(parse_value(s.substr(a, i - a)));
        }
    }
    return items;
}

// Where a string whose opening quote the model dropped ends, or npos when the bare value at `a` (running to `end`) is
// not one. The template only ever writes numbers, booleans and null bare, so a bare word is a string gone wrong, and a
// quote that follows it without opening a value closes it. Without this the text is cut at its first comma or ]
// (SGLang's parser does that), and the pieces come back as junk arguments.
size_t lost_quote_close(const std::string& s, size_t a, size_t end) {
    if (end == a || parse_value(s.substr(a, end - a)).kind != Json::Str) return std::string::npos;
    size_t e = s.find(kQ, a);
    if (e == std::string::npos) return std::string::npos;
    size_t before = e;
    while (before > a && py_ws_arg(s[before - 1])) before--;
    if (before == a || s[before - 1] == ':' || s[before - 1] == ',' || s[before - 1] == '[') return std::string::npos;
    size_t after = e + kQ.size();
    while (after < s.size() && py_ws_arg(s[after])) after++;
    return after == s.size() || s[after] == ',' || s[after] == '}' ? e : std::string::npos;
}

// _parse_gemma4_args
Json parse_args(const std::string& s) {
    Json result = Json::make_obj();
    if (py_strip_text(s).empty()) return result;
    size_t i = 0, n = s.size();
    while (i < n) {
        while (i < n && (s[i] == ',' || py_ws_arg(s[i]))) i++;
        if (i >= n) break;
        size_t ks = i;
        while (i < n && s[i] != ':') i++;
        if (i >= n) break;
        std::string key = py_strip_text(s.substr(ks, i - ks));
        i++;
        if (i >= n) { result.set(key, Json::make_str("")); break; }
        while (i < n && py_ws_arg(s[i])) i++;
        if (i >= n) { result.set(key, Json::make_str("")); break; }
        if (s.compare(i, kQ.size(), kQ) == 0) {
            i += kQ.size();
            size_t e = s.find(kQ, i);
            if (e == std::string::npos) { result.set(key, Json::make_str(s.substr(i))); break; }
            result.set(key, Json::make_str(s.substr(i, e - i)));
            i = e + kQ.size();
        } else if (s[i] == '{') {
            size_t a = i + 1;
            skip_nested(s, i, '{', '}', true);
            result.set(key, parse_args(inner_of(s, a, i)));
        } else if (s[i] == '[') {
            size_t a = i + 1;
            skip_nested(s, i, '[', ']', true);
            result.set(key, parse_array(inner_of(s, a, i)));
        } else {
            size_t a = i;
            while (i < n && s[i] != ',' && s[i] != '}' && s[i] != ']') i++;
            size_t close = lost_quote_close(s, a, i);
            if (close != std::string::npos) {
                result.set(key, Json::make_str(s.substr(a, close - a)));
                i = close + kQ.size();
            } else {
                result.set(key, parse_value(s.substr(a, i - a)));
            }
        }
    }
    return result;
}

// _find_matching_brace: of text just after an opening '{'; npos when incomplete
size_t find_matching_brace(const std::string& t) {
    int depth = 1;
    size_t i = 0, n = t.size();
    while (i < n && depth > 0) {
        if (t.compare(i, kQ.size(), kQ) == 0) {
            i += kQ.size();
            size_t e = t.find(kQ, i);
            if (e == std::string::npos) return std::string::npos;
            i = e + kQ.size();
            continue;
        }
        if (t[i] == '{') depth++;
        else if (t[i] == '}') depth--;
        i++;
    }
    return depth == 0 ? i - 1 : std::string::npos;
}

std::string call_id() {                      // the id form of glm_server's other tool calls
    static std::mt19937_64 rng{std::random_device{}()};
    static const char* h = "0123456789abcdef";
    std::string s = "call_";
    for (int i = 0; i < 24; i++) s += h[rng() & 15];
    return s;
}

size_t partial_suffix(const std::string& text, const std::string& marker) {
    for (size_t k = std::min(marker.size() - 1, text.size()); k > 0; k--)
        if (marker.compare(0, k, text, text.size() - k, k) == 0) return k;
    return 0;
}

}  // namespace

std::string render_chat_gemma(const Json& messages, const Json* tools, const Json& vars) {
    static const Json none = Json::make_obj();
    return render(messages, tools, vars.kind == Json::Obj ? vars : none);
}

ChatPrompt render_prompt_gemma(const Json& messages, const Json* tools, const Json* chat_template_kwargs,
                               const Json* reasoning_effort) {
    Json kw = Json::make_obj();
    if (truthy(chat_template_kwargs)) {
        GLM_CHECK(chat_template_kwargs->kind == Json::Obj, "chat_template_kwargs is not an object");
        kw = *chat_template_kwargs;
    }
    if (reasoning_effort && reasoning_effort->kind != Json::Null && !kw.has("reasoning_effort"))
        kw.set("reasoning_effort", *reasoning_effort);
    kw.set("add_generation_prompt", Json::make_bool(true));
    V et = kw.get("enable_thinking");
    ChatPrompt p;
    p.thinking = et && et->kind == Json::Bool && et->b;   // _get_reasoning_from_request: explicit `is True`, default off
    p.reply_in_thought = p.thinking;
    p.text = render(shape_messages(messages), truthy(tools) ? tools : nullptr, kw);
    return p;
}

std::pair<std::string, Json> parse_tool_calls_gemma(const std::string& content, const Json* tools) {
    Json calls = Json::make_arr();
    if (!tools || content.find(kCallOpen) == std::string::npos) return {content, calls};
    // Gemma4Detector._extract_tool_calls
    size_t from = 0;
    for (;;) {
        size_t a = content.find(kCallOpen, from);
        if (a == std::string::npos) break;
        size_t e = content.find(kCallClose, a);
        if (e == std::string::npos) break;
        std::string inner = content.substr(a + kCallOpen.size(), e - a - kCallOpen.size());
        from = e + kCallClose.size();
        if (inner.compare(0, 5, "call:") != 0) continue;
        size_t brace = inner.find('{');
        if (brace == std::string::npos) continue;
        std::string body = inner.substr(brace + 1);
        size_t m = find_matching_brace(body);
        Json fn = Json::make_obj();
        fn.set("name", Json::make_str(inner.substr(5, brace - 5)));
        fn.set("arguments", Json::make_str(dump(parse_args(m == std::string::npos ? body : body.substr(0, m)))));
        Json c = Json::make_obj();
        c.set("id", Json::make_str(call_id()));
        c.set("type", Json::make_str("function"));
        c.set("function", std::move(fn));
        calls.push(std::move(c));
    }
    if (calls.arr.empty()) return {content, calls};
    return {content.substr(0, content.find(kCallOpen)), calls};  // detect_and_parse: the text before the first call
}

// ------------------------------------------------------------------------------------------------- GemmaSplitter

void GemmaSplitter::reasoning_step(const std::string& text, std::string& reasoning, std::string& normal) {
    buf_ += text;
    for (const std::string* t : {&kThoughtOpen, &kChannelClose})
        if (*t != buf_ && t->compare(0, buf_.size(), buf_) == 0) return;   // the buffer could still become this marker
    if (!stripped_) {
        size_t p = buf_.find(kThoughtOpen);
        if (p != std::string::npos) {
            buf_.erase(p, kThoughtOpen.size());
            stripped_ = true;
            in_reasoning_ = true;
        }
    }
    if (in_reasoning_) {
        size_t e = buf_.find(kChannelClose);
        if (e != std::string::npos) {
            reasoning = buf_.substr(0, e);
            normal = buf_.substr(e + kChannelClose.size());
            buf_.clear();
            in_reasoning_ = false;
            return;
        }
        size_t hold = partial_suffix(buf_, kChannelClose);
        if (!stripped_) hold = std::max(hold, partial_suffix(buf_, kThoughtOpen));
        reasoning = buf_.substr(0, buf_.size() - hold);
        buf_ = buf_.substr(buf_.size() - hold);
        return;
    }
    normal = std::move(buf_);
    buf_.clear();
}

void GemmaSplitter::content_step(const std::string& text, bool final, Pieces& out) {
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
        if (p) out.push_back({false, pending_.substr(0, p)});
        held = pending_.substr(p);
        pending_.clear();
        return;
    }
    size_t hold = final ? 0 : partial_suffix(pending_, kCallOpen);
    if (pending_.size() > hold) out.push_back({false, pending_.substr(0, pending_.size() - hold)});
    pending_ = pending_.substr(pending_.size() - hold);
}

GemmaSplitter::Pieces GemmaSplitter::feed(const std::string& text, bool final) {
    Pieces out;
    std::string reasoning, normal;
    if (!separate_) normal = text;
    else {
        reasoning_step(text, reasoning, normal);
        if (final) {
            // BaseReasoningFormatDetector.finish: what was held back for a marker that never came
            if (in_reasoning_) {
                std::string b = buf_.compare(0, kThoughtOpen.size(), kThoughtOpen) == 0 ? buf_.substr(kThoughtOpen.size()) : buf_;
                reasoning += b;
            } else {
                normal += buf_;
            }
            buf_.clear();
        }
    }
    if (!reasoning.empty()) out.push_back({true, std::move(reasoning)});
    content_step(normal, final, out);
    return out;
}

}  // namespace glm
