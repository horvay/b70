// chat_dump <model dir> <conversations.jsonl> [max tokens]: print each conversation as exl3eval renders it, one line
// of "id scored" pairs per conversation (scored is 1 for a target token), for tests/chat_check.py, which compares
// the ids with Hugging Face's apply_chat_template.
#include "io/chat.hpp"

#include <cstdio>
#include <filesystem>

int main(int argc, char** argv)
{
    if (argc < 3) { std::fprintf(stderr, "usage: chat_dump <model dir> <conversations.jsonl> [max tokens]\n"); return 2; }
    const std::filesystem::path dir = argv[1];
    std::ifstream cf(dir / "config.json");
    const auto cfg = nlohmann::json::parse(cf);
    const auto& tc = cfg.contains("text_config") ? cfg["text_config"] : cfg;
    tok::BPE tk((dir / "tokenizer.json").string());
    const int bos = tc.value("bos_token_id", cfg.value("bos_token_id", 2));
    for (const auto& r : chat::load(tk, argv[2], cfg.value("model_type", std::string()), bos,
                                    argc > 3 ? std::stoul(argv[3]) : (size_t) 1 << 30, 0))
    {
        for (size_t i = 0; i < r.ids.size(); ++i) std::printf("%d:%d ", r.ids[i], (int) r.scored[i]);
        std::printf("\n");
    }
    return 0;
}
