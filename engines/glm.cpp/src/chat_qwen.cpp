// Qwen3.8-Flash-Next's chat_template.jinja (the Swift 1.5 pack ships it unmodified), ported by hand: every branch of
// the template, in its order, with Jinja's semantics where they matter (undefined vs None, trim = str.strip,
// tojson = json.dumps(ensure_ascii=False)).
#include "chat.h"
#include "rt.h"

namespace glm {

namespace {

using V = const Json*;

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
bool ends_with(const std::string& s, const std::string& t) {
    return s.size() >= t.size() && s.compare(s.size() - t.size(), t.size(), t) == 0;
}

bool g_images_ok = false;

// render_content(content, ...): a string as is, a list's text items joined (images: their placeholder when the model has
// a vision tower; videos: refused), None / undefined as ''
std::string render_content(V c) {
    if (!c || c->kind == Json::Null) return "";
    if (c->kind == Json::Str) return c->str;
    GLM_CHECK(c->kind == Json::Arr, "chat template: Unexpected content type.");
    std::string o;
    for (auto& it : c->arr) {
        GLM_CHECK(it.kind == Json::Obj, "chat template: Unexpected item type in content.");
        bool img = it.has("image") || it.has("image_url") || eq_str(it.get("type"), "image");
        bool vid = it.has("video") || eq_str(it.get("type"), "video");
        GLM_CHECK((!img || g_images_ok) && !vid, "images and videos are not supported by this server (text only)");
        if (img) { o += "<|vision_start|><|image_pad|><|vision_end|>"; continue; }
        V t = it.get("text");
        GLM_CHECK(t, "chat template: Unexpected item type in content.");
        o += is_str(t) ? t->str : dump(*t);
    }
    return o;
}

const char* kToolFormat =
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n"
    "<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n"
    "<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n"
    "</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified "
    "format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural "
    "language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question "
    "like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

std::string render(const Json& msgs, const Json* tools, const Json& vars) {
    GLM_CHECK(msgs.kind == Json::Arr && !msgs.arr.empty(), "chat template: No messages provided.");
    const auto& M = msgs.arr;
    V et = vars.get("enable_thinking");
    V pt = vars.get("preserve_thinking");
    std::string o, instr;
    if (!et || (et->kind == Json::Bool && et->b)) {
        V re = vars.get("reasoning_effort");
        std::string eff = re ? (is_str(re) ? re->str : dump(*re)) : "xhigh";
        GLM_CHECK(eff == "xhigh" || eff == "medium" || eff == "low",
                  "chat template: Unexpected reasoning effort %s. Supported types are xhigh (default), medium, and low.",
                  eff.c_str());
        if (eff == "xhigh")
            instr = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, "
                    "consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final "
                    "answer.";
        else if (eff == "low")
            instr = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the "
                    "conclusion without unnecessary elaboration.";
    }
    bool first_system = eq_str(M[0].get("role"), "system");
    if (tools && tools->kind == Json::Arr && !tools->arr.empty()) {
        o += "<|im_start|>system\n";
        if (!instr.empty()) o += instr + "\n\n";
        o += "# Tools\n\nYou have access to the following functions:\n\n<tools>";
        for (auto& t : tools->arr) o += "\n" + dump(t);
        o += "\n</tools>";
        o += kToolFormat;
        if (first_system) {
            std::string c = py_strip_text(render_content(M[0].get("content")));
            if (!c.empty()) o += "\n\n" + c;
        }
        o += "<|im_end|>\n";
    } else if (first_system) {
        std::string c = py_strip_text(render_content(M[0].get("content")));
        if (!c.empty()) o += "<|im_start|>system\n" + (instr.empty() ? "" : instr + "\n\n") + c + "<|im_end|>\n";
        else if (!instr.empty()) o += "<|im_start|>system\n" + instr + "<|im_end|>\n";
    } else if (!instr.empty()) {
        o += "<|im_start|>system\n" + instr + "<|im_end|>\n";
    }
    // the last user query (tool responses wrapped as user messages don't count)
    bool multi_step_tool = true;
    int last_query = (int)M.size() - 1;
    for (int i = (int)M.size() - 1; i >= 0 && multi_step_tool; i--) {
        if (!eq_str(M[i].get("role"), "user")) continue;
        std::string c = py_strip_text(render_content(M[i].get("content")));
        if (!(starts_with(c, "<tool_response>") && ends_with(c, "</tool_response>"))) {
            multi_step_tool = false;
            last_query = i;
        }
    }
    GLM_CHECK(!multi_step_tool, "chat template: No user query found in messages.");
    for (size_t i = 0; i < M.size(); i++) {
        const Json& m = M[i];
        std::string c = py_strip_text(render_content(m.get("content")));
        V role = m.get("role");
        if (eq_str(role, "system")) {
            GLM_CHECK(i == 0, "chat template: System message must be at the beginning.");
        } else if (eq_str(role, "user")) {
            o += "<|im_start|>user\n" + c + "<|im_end|>\n";
        } else if (eq_str(role, "assistant")) {
            V rc = m.get("reasoning_content");
            std::string r = is_str(rc) ? py_strip_text(rc->str) : "";
            bool keep = !pt || (pt->kind == Json::Bool && pt->b) || (int)i > last_query;
            if (keep) o += "<|im_start|>assistant\n<think>\n" + r + "\n</think>\n\n" + c;
            else o += "<|im_start|>assistant\n" + c;
            V tcs = m.get("tool_calls");
            if (tcs && tcs->kind == Json::Arr) {
                for (size_t k = 0; k < tcs->arr.size(); k++) {
                    const Json* tc = &tcs->arr[k];
                    if (V fn = tc->get("function")) tc = fn;
                    V name = tc->get("name");
                    std::string nm = name ? (is_str(name) ? name->str : dump(*name)) : "";
                    if (k == 0) o += (py_strip_text(c).empty() ? "" : "\n\n") + std::string("<tool_call>\n<function=") + nm + ">\n";
                    else o += "\n<tool_call>\n<function=" + nm + ">\n";
                    V args = tc->get("arguments");
                    if (args && !eq_str(args, "")) {
                        GLM_CHECK(args->kind == Json::Obj, "chat template: tool call arguments are not a mapping");
                        for (auto& [k2, v] : args->obj) {
                            o += "<parameter=" + k2 + ">\n";
                            o += v.kind == Json::Str ? v.str : dump(v);
                            o += "\n</parameter>\n";
                        }
                    }
                    o += "</function>\n</tool_call>";
                }
            }
            o += "<|im_end|>\n";
        } else if (eq_str(role, "tool")) {
            if (i > 0 && !eq_str(M[i - 1].get("role"), "tool")) o += "<|im_start|>user";
            o += "\n<tool_response>\n" + c + "\n</tool_response>";
            if (i + 1 < M.size() && !eq_str(M[i + 1].get("role"), "tool")) o += "<|im_end|>\n";
            else if (i + 1 == M.size()) o += "<|im_end|>\n";
        } else {
            GLM_CHECK(false, "chat template: Unexpected message role.");
        }
    }
    V agp = vars.get("add_generation_prompt");
    if (truthy(agp)) {
        o += "<|im_start|>assistant\n";
        if (et && et->kind == Json::Bool && !et->b) o += "<think>\n\n</think>\n\n";
        else o += "<think>\n";
        V nq = vars.get("nq_prefill");
        if (truthy(nq)) o += is_str(nq) ? nq->str : dump(*nq);
    }
    return o;
}

}  // namespace

void qwen_allow_images(bool on) { g_images_ok = on; }

ChatPrompt render_prompt_qwen(const Json& messages, const Json* tools, const Json* chat_template_kwargs,
                              const Json* reasoning_effort, const std::string& default_effort) {
    Json kw = Json::make_obj();
    if (truthy(chat_template_kwargs)) {
        GLM_CHECK(chat_template_kwargs->kind == Json::Obj, "chat_template_kwargs is not an object");
        kw = *chat_template_kwargs;
    }
    if (reasoning_effort && reasoning_effort->kind != Json::Null && !kw.has("reasoning_effort"))
        kw.set("reasoning_effort", *reasoning_effort);
    if (!kw.has("reasoning_effort") && !default_effort.empty()) kw.set("reasoning_effort", Json::make_str(default_effort));
    kw.set("add_generation_prompt", Json::make_bool(true));
    V et = kw.get("enable_thinking");
    ChatPrompt p;
    p.thinking = !(et && et->kind == Json::Bool && !et->b);
    Json msgs = normalize_messages(messages);
    p.text = render(msgs, truthy(tools) ? tools : nullptr, kw);
    size_t o = p.text.rfind("<think>"), c = p.text.rfind("</think>");
    p.reply_in_thought = p.thinking && o != std::string::npos && (c == std::string::npos || o > c);
    return p;
}

}  // namespace glm
