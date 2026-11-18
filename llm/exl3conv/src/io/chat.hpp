// exl3conv: conversations rendered as the chat template does, for exl3eval's scoring of assistant replies.
//
// Gemma 4: <bos> then, per message, <|turn>ROLE\nCONTENT<turn|>\n with the assistant's role written as "model".
// Special tokens go in as ids; the text between them is encoded on its own, which is how HF splits a rendered
// template around its special tokens. Scored targets are an assistant turn's content tokens and its <turn|>.
//
// Ripple-65B (ricursive): no BOS, per message <|im_start|>ROLE\nCONTENT<|im_end|>\n (chat_template.jinja without tools);
// scored targets are an assistant turn's content tokens and its <|im_end|>.
#pragma once
#include "io/tokenizer.hpp"

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace chat {

using json = nlohmann::json;

struct Row
{
    std::vector<int32_t> ids;
    std::vector<uint8_t> scored;   // scored[t]: the token at t is a target (predicted from position t - 1)
    std::string id, lang;
    int64_t n_scored() const { int64_t n = 0; for (auto s : scored) n += s; return n; }
};

inline std::vector<json> read_jsonl(const std::string& path)
{
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot read " + path);
    std::vector<json> out;
    for (std::string line; std::getline(f, line);)
        if (!line.empty()) out.push_back(json::parse(line));
    return out;
}

// One conversation, cut to at most max_tokens.
inline Row render_gemma4(const tok::BPE& tk, const json& conv, int bos, size_t max_tokens)
{
    const int turn_open = tk.id("<|turn>"), turn_close = tk.id("<turn|>");
    const auto nl = tk.encode("\n");
    Row r;
    r.id = conv.value("id", ""); r.lang = conv.value("lang", "");
    auto push = [&](int t, bool s) { r.ids.push_back(t); r.scored.push_back(s); };
    push(bos, false);
    for (const auto& m : conv.at("messages"))
    {
        std::string role = m.at("role");
        const bool asst = role == "assistant";
        if (asst) role = "model";
        push(turn_open, false);
        // Role and content are one text segment; the role's own tokens are the prefix that encodes alone.
        const auto pre = tk.encode(role + "\n");
        const auto seg = tk.encode(role + "\n" + m.at("content").get<std::string>());
        size_t k = 0;
        while (k < pre.size() && k < seg.size() && pre[k] == seg[k]) ++k;
        for (size_t i = 0; i < seg.size(); ++i) push(seg[i], asst && i >= k);
        push(turn_close, asst);
        for (int t : nl) push(t, false);
    }
    if (r.ids.size() > max_tokens) { r.ids.resize(max_tokens); r.scored.resize(max_tokens); }
    return r;
}

inline Row render_ripple(const tok::BPE& tk, const json& conv, size_t max_tokens)
{
    const int open = tk.id("<|im_start|>"), close = tk.id("<|im_end|>");
    const auto nl = tk.encode("\n");
    Row r;
    r.id = conv.value("id", ""); r.lang = conv.value("lang", "");
    auto push = [&](int t, bool s) { r.ids.push_back(t); r.scored.push_back(s); };
    for (const auto& m : conv.at("messages"))
    {
        const std::string role = m.at("role");
        const bool asst = role == "assistant";
        push(open, false);
        const auto pre = tk.encode(role + "\n");
        const auto seg = tk.encode(role + "\n" + m.at("content").get<std::string>());
        size_t k = 0;
        while (k < pre.size() && k < seg.size() && pre[k] == seg[k]) ++k;
        for (size_t i = 0; i < seg.size(); ++i) push(seg[i], asst && i >= k);
        push(close, asst);
        for (int t : nl) push(t, false);
    }
    if (r.ids.size() > max_tokens) { r.ids.resize(max_tokens); r.scored.resize(max_tokens); }
    return r;
}

inline std::vector<Row> load(const tok::BPE& tk, const std::string& path, const std::string& model_type, int bos,
                             size_t max_tokens, int64_t min_scored = 32)
{
    if (model_type != "gemma4" && model_type != "ricursive") throw std::runtime_error("chat eval: no chat format for " + model_type + " yet");
    std::vector<Row> rows;
    for (const auto& c : read_jsonl(path))
    {
        Row r = model_type == "ricursive" ? render_ripple(tk, c, max_tokens) : render_gemma4(tk, c, bos, max_tokens);
        if (r.n_scored() >= min_scored) rows.push_back(std::move(r));
    }
    return rows;
}

}  // namespace chat
