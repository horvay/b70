// Qwen3.8-Flash-Next prompts against HF apply_chat_template (tests/q4_chat_ref.py's chat.json): render_prompt_qwen's
// text byte for byte, and the tokenizer's ids for it.
//   q4_chat_check <model dir> <chat.json>
#include <cstdio>
#include <fstream>
#include <sstream>
#include "../src/chat.h"
#include "../src/rt.h"
#include "../src/tokenizer.h"

using namespace glm;

static std::string slurp(const char* p) {
    std::ifstream f(p, std::ios::binary);
    GLM_CHECK(f, "cannot open %s", p);
    std::stringstream s;
    s << f.rdbuf();
    return s.str();
}

int main(int argc, char** argv) {
    GLM_CHECK(argc == 3, "usage: q4_chat_check <model dir> <chat.json>");
    Tokenizer tok = Tokenizer::load(argv[1]);
    Json cases = Json::parse(slurp(argv[2]));
    int bad = 0;
    for (auto& c : cases.arr) {
        const Json* tools = c["tools"].kind == Json::Arr ? &c["tools"] : nullptr;
        ChatPrompt p = render_prompt_qwen(c["messages"], tools, &c["kwargs"], nullptr);
        const std::string& want = c["text"].s();
        bool text_ok = p.text == want;
        std::vector<int> ids = tok.encode(p.text, true);
        bool ids_ok = ids.size() == c["ids"].size();
        for (size_t i = 0; ids_ok && i < ids.size(); i++) ids_ok = ids[i] == (int)c["ids"][i].i();
        printf("%-16s text %s, ids %s (%zu tokens)\n", c["name"].s().c_str(), text_ok ? "ok" : "DIFF", ids_ok ? "ok" : "DIFF",
               ids.size());
        if (!text_ok) {
            size_t k = 0;
            while (k < want.size() && k < p.text.size() && want[k] == p.text[k]) k++;
            printf("  first difference at byte %zu:\n  want: %s\n  got:  %s\n", k, want.substr(k, 80).c_str(),
                   p.text.substr(k, 80).c_str());
        }
        bad += !text_ok || !ids_ok;
    }
    printf(bad ? "Q4 CHAT CHECK FAILED (%d)\n" : "Q4 CHAT CHECK OK\n", bad);
    return bad ? 1 : 0;
}
