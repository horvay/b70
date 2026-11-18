#pragma once
// GLM-5.3 prompts: the model's chat_template.jinja ported by hand, plus what llm/glm/xpu/serve.py does around it
// (normalize_messages, Renderer.render, the request's thinking switch). Byte-exact with HF apply_chat_template.
#include <string>
#include <utility>
#include <vector>
#include "json.h"

namespace glm {

// The chat template with HF's Jinja settings (trim_blocks, lstrip_blocks, tojson = json.dumps ensure_ascii=False).
//   messages        array of message objects, as given (no normalization)
//   tools           array of tool objects, or nullptr
//   vars            object of template variables: reasoning_effort ("low" / "high"; anything else renders Max),
//                   clear_thinking, add_generation_prompt (absent = false, as in HF's render_jinja_template),
//                   nq_prefill. Absent = undefined in Jinja.
//   nq_prefill_hook false: chat_template.jinja exactly. true: the client's template (llm/glm/run/nqfake/tpl.jinja, which
//                   b70 serve gets as --chat-template): the model's plus a truthy vars.nq_prefill appended after the
//                   generation prompt. Without nq_prefill both render the same.
// Inputs the template itself would fail on (a tool_call without a name, non-object arguments, ...) fail GLM_CHECK.
std::string render_chat(const Json& messages, const Json* tools, const Json& vars, bool nq_prefill_hook = true);

// serve.py normalize_messages: content null / absent -> "", role developer -> system, each tool call's arguments
// parsed when a JSON string ({"arguments": text} when it does not parse, {} when it is not an object).
Json normalize_messages(const Json& messages);

// serve.py's settings: the defaults for a request that names no effort / clear_thinking, and the template.
struct ServeConfig {
    std::string effort = "low";     // GLM_SERVE_REASONING_EFFORT, mapped like a request's effort
    bool clear_thinking = true;     // GLM_SERVE_CLEAR_THINKING: anything but "0", "false", "no", ""
    bool nq_prefill_hook = true;    // render_chat's: false = serve.py without --chat-template
    static ServeConfig from_env();  // effort and clear_thinking from the environment, as serve.py reads them
};

struct ChatPrompt {
    std::string text;               // the prompt to encode (Tokenizer::encode with special tokens)
    bool thinking;                  // chat_template_kwargs.enable_thinking is not false
    bool reply_in_thought;          // the reply starts inside <think> (serve.py's Splitter starting state)
};

// A /v1/chat/completions prompt exactly as serve.py renders it:
//   messages              the request's messages
//   tools                 the request's tools (nullptr when absent or tool_choice is "none")
//   chat_template_kwargs  the request's object, or nullptr; its other keys pass to the template as variables
//   reasoning_effort      the request's top-level reasoning_effort, or nullptr (chat_template_kwargs' wins)
// Effort: none / minimal / low -> Low, medium / high -> High, xhigh / max -> Max, anything else the default.
// Thinking off appends "</think>" to the open think block of the generation prompt.
ChatPrompt render_prompt(const Json& messages, const Json* tools, const Json* chat_template_kwargs,
                         const Json* reasoning_effort, const ServeConfig& config = ServeConfig::from_env());

// Python's str.strip() (Jinja's trim): Unicode whitespace at both ends
std::string py_strip_text(const std::string& s);
// Python's str(x) (Jinja's {{ x }}): a string as is, the rest their repr (None, True, 2.5, ['a'], {'k': 1})
std::string py_str_value(const Json& x);

// Qwen3.8-Flash-Next (qwen4_exp): its chat_template.jinja ported by hand (chat_qwen.cpp), with the client's prefill hook
// (a truthy chat_template_kwargs.nq_prefill appended after the generation prompt). Template variables come from
// chat_template_kwargs, plus the request's top-level reasoning_effort when the kwargs have none; default_effort
// (Q4_SERVE_REASONING_EFFORT: xhigh, medium or low; empty = the template's default, xhigh) fills an unset one.
// enable_thinking false: the generation prompt's empty think block.
ChatPrompt render_prompt_qwen(const Json& messages, const Json* tools, const Json* chat_template_kwargs,
                              const Json* reasoning_effort, const std::string& default_effort = "");
// Qwen3.8-27B has a vision tower: image parts render as their placeholder instead of being refused.
void qwen_allow_images(bool on);

// Gemma 4 (gemma4): its chat_template.jinja ported by hand (chat_gemma.cpp), plus the client's prefill hook: every model the client
// serves gets the hook appended to its template, and for Gemma nq_prefill is "<|channel>thought\n<prefill>" with
// enable_thinking true (closed with "\n<channel|>" and separate_reasoning false for "answer now"). Byte-exact with HF
// apply_chat_template over that template. The prompt starts with <bos>, as text: encode it with special tokens and add
// nothing. vars: enable_thinking, preserve_thinking, add_generation_prompt, nq_prefill; absent = undefined in Jinja.
// Inputs the template itself would fail on fail GLM_CHECK.
std::string render_chat_gemma(const Json& messages, const Json* tools, const Json& vars);

// A /v1/chat/completions prompt as `b70 serve` renders it through SGLang: its message shaping (argument strings parsed,
// null content, content parts), then the template with add_generation_prompt. Thinking is off unless
// chat_template_kwargs.enable_thinking is true (the template's default, and SGLang's). reasoning_effort only passes
// through as a template variable, which this template ignores.
//   thinking          enable_thinking is exactly true
//   reply_in_thought  the same: SGLang's force_reasoning. The reply counts as thought until <channel|>, whether the model
//                     opens the channel itself or the prompt did (after a tool response, or a client prefill)
ChatPrompt render_prompt_gemma(const Json& messages, const Json* tools, const Json* chat_template_kwargs,
                               const Json* reasoning_effort);

// SGLang's gemma4 tool-call parser (detect_and_parse): <|tool_call>call:NAME{key:value,...}<tool_call|>, values in
// <|"|> quotes, {objects}, [arrays], or bare (true, false, Python int and float spellings, anything else as a string,
// including null). The tool schema's types are not used; a name outside tools is kept. Returns the text before the first
// call (the text itself when no call is complete) and the OpenAI tool_calls, arguments as a JSON string. tools nullptr: no
// parsing, as SGLang's parser is off for such a request.
std::pair<std::string, Json> parse_tool_calls_gemma(const std::string& content, const Json* tools);

// Streamed model text -> reasoning, visible content, a held tool-call tail; glm_server's Splitter for Gemma, with the
// same shape: pass the text with special tokens (the markers are special tokens, SGLang sets skip_special_tokens false
// for this model). The reasoning side is SGLang's streaming gemma4 detector (markers <|channel>thought\n and <channel|>):
//   force_reasoning  ChatPrompt.reply_in_thought: text is thought from the start, "<|channel>thought\n" dropped once
//   otherwise        text is content until the marker appears, so a model that opens a thought unasked is still split
//   a reply cut off inside the thought is all thought; a model that answers with thinking on and never opens or closes
//   the channel is all thought too (SGLang's behaviour)
// separate_reasoning false (the request's field, "answer now" sends it): everything is content. has_tools: from
// <|tool_call> on the text goes to `held` for parse_tool_calls_gemma; without tools it stays content. Nothing is stripped
// (SGLang does not): a thought normally ends in "\n", from the "\n<channel|>" the model writes.
class GemmaSplitter {
public:
    using Pieces = std::vector<std::pair<bool, std::string>>;    // (is_reasoning, text)
    GemmaSplitter(bool force_reasoning, bool has_tools, bool separate_reasoning = true)
        : tools_(has_tools), separate_(separate_reasoning), in_reasoning_(force_reasoning) {}
    Pieces feed(const std::string& text, bool final = false);    // final = true at the end of the reply
    std::string held;                                            // the tool-call tail, once it started

private:
    void reasoning_step(const std::string& text, std::string& reasoning, std::string& normal);
    void content_step(const std::string& text, bool final, Pieces& out);
    bool tools_, separate_, in_reasoning_, stripped_ = false;
    std::string buf_, pending_;
};

// Ripple-65B (ricursive): its chat_template.jinja ported by hand (chat_ripple.cpp), plus the client's prefill hook (a truthy
// vars.nq_prefill appended after the generation prompt). Byte-exact with HF apply_chat_template over that template; the
// template adds no <|endoftext|>, and neither does encoding the prompt. vars: clear_thinking, add_generation_prompt,
// nq_prefill; absent = undefined in Jinja. Inputs the template itself would fail on fail GLM_CHECK.
std::string render_chat_ripple(const Json& messages, const Json* tools, const Json& vars);

// A /v1/chat/completions prompt: normalize_messages, a content list's text parts joined with spaces (SGLang's string
// content format; other parts are dropped, so the server refuses them first), then the template with
// add_generation_prompt. Assistant reasoning_content is rendered back unless chat_template_kwargs.clear_thinking is
// truthy. The template has no effort levels or thinking switch: reasoning_effort is ignored, and enable_thinking exactly
// false (the one addition to the template, as the other ports have their own) renders the empty thought "<think>\n\n
// </think>\n" after the generation prompt.
//   thinking          enable_thinking is not false
//   reply_in_thought  nq_prefill opens a <think> it does not close. The generation prompt opens nothing, so otherwise the
//                     reply counts as thought only if it starts with <think> itself.
// nq_prefill should have the template's layout: "<think>\n<prefill>", closed with "\n</think>\n" for "answer now" (with
// separate_reasoning false).
ChatPrompt render_prompt_ripple(const Json& messages, const Json* tools, const Json* chat_template_kwargs,
                                const Json* reasoning_effort);

// ripple_format.py's tool-call parser (hermes with the template's layout): <tool_call>\n{"name": "f", "arguments": {...}}
// \n</tool_call>, any JSON whitespace (compact too), hermes' "parameters" for "arguments" and a list of calls in one
// block. Arguments are re-dumped as json.dumps does; ones sent as a string holding a JSON object are read as that object,
// null or absent are {}. A name outside tools is kept. A block that is not JSON or has no name is skipped; a complete JSON
// call whose </tool_call> never came still counts. Returns the text before the first call without the one newline the
// template puts there (the whole text when no call parsed) and the OpenAI tool_calls, arguments as a JSON string; text
// after a call is dropped. tools nullptr: no parsing.
std::pair<std::string, Json> parse_tool_calls_ripple(const std::string& content, const Json* tools);

// Streamed model text -> reasoning, visible content, a held tool-call tail; GemmaSplitter's shape, with ripple_format.py's
// RippleReasoningDetector on the reasoning side (states start -> lead -> think -> after -> content):
//   force_reasoning  ChatPrompt.reply_in_thought: text is thought from the start (a prefill of exactly "<think>" keeps the
//                    model's first newline in the thought, as the splitter cannot see the prefill)
//   otherwise        text is thought only if the reply opens with <think>, anywhere later it is content
// Exactly the separator newlines go: one after <think>, one before </think>, one after </think>. "\n</think>" is held back
// while it can still complete; a reply cut off inside the thought is all thought. separate_reasoning false (the request's
// field, "answer now" sends it): everything is content. has_tools: from <tool_call> on the text goes to `held` for
// parse_tool_calls_ripple, and the text before it loses the one newline the template adds (held back until it is known);
// without tools it stays content.
class RippleSplitter {
public:
    using Pieces = std::vector<std::pair<bool, std::string>>;    // (is_reasoning, text)
    RippleSplitter(bool force_reasoning, bool has_tools, bool separate_reasoning = true)
        : tools_(has_tools), separate_(separate_reasoning), state_(force_reasoning ? Think : Start) {}
    Pieces feed(const std::string& text, bool final = false);    // final = true at the end of the reply
    std::string held;                                            // the tool-call tail, once it started

private:
    void reasoning_step(const std::string& text, std::string& reasoning, std::string& normal);
    void content_step(const std::string& text, bool final, Pieces& out);
    enum State { Start, Lead, Think, After, Content };
    bool tools_, separate_;
    State state_;
    std::string buf_, pending_;
};

}  // namespace glm
