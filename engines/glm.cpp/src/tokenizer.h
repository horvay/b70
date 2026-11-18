#pragma once
// HF `tokenizers` BPE of a tokenizer.json, in the two families the engine serves, picked by load():
//  - byte-level (GLM-5.3, Qwen3.8, Ripple): Split regex (Isolated; Ripple: Removed with invert, gaps dropped) + ByteLevel
//    pre-tokenizer, NFC (Qwen), ignore_merges (GLM), as the PyTorch stack's exllamav3 Tokenizer uses it;
//  - SentencePiece-style (Gemma 4): byte_fallback (characters outside the vocabulary become <0xNN> tokens), the
//    normalizer Replace(' ' -> '▁'), a Split on a string, and the decoder Replace('▁' -> ' ') + ByteFallback
//    (a run of <0xNN> tokens joined into UTF-8, U+FFFD per byte when it is not valid) + Fuse.
// Plus the streaming detokenizer of exllamav3's generator (job.py receive_sample with decode_special_tokens=False).
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace glm {

class Tokenizer {
public:
    // <model_dir>/tokenizer.json. Fails (GLM_CHECK) on a tokenizer.json this port does not cover exactly.
    static Tokenizer load(const std::string& model_dir);
    Tokenizer(Tokenizer&&) noexcept;
    Tokenizer& operator=(Tokenizer&&) noexcept;
    ~Tokenizer();

    // exllamav3 encode(text, add_bos=False, encode_special_tokens=special). Added tokens anywhere in the text are
    // matched first (leftmost, longest); special=false leaves the special ones ([gMASK], <|user|>, ...) to be
    // encoded as plain text, while the non-special ones (<think>, <tool_call>, ...) are still matched, as in HF.
    // Invalid UTF-8 is read as U+FFFD per bad sequence (Python's errors="replace").
    std::vector<int> encode(const std::string& text, bool special = true) const;
    // HF decode(ids, skip_special_tokens=skip_special): byte-level bytes joined, invalid UTF-8 as U+FFFD; or (byte
    // fallback) the token texts joined, each run of <0xNN> tokens as UTF-8 or one U+FFFD per byte. Ids out of range
    // are dropped.
    std::string decode(const std::vector<int>& ids, bool skip_special = false) const;
    // decode({id}) with special tokens: what the generator streams for this token alone ("" when out of range)
    const std::string& token_text(int id) const;
    int vocab_size() const;                   // model vocabulary plus added tokens (GLM 154856, Gemma 262144)
    // id of an added token's content (e.g. "<|user|>") or of a vocabulary entry as tokenizer.json spells it ("Ġthe",
    // Gemma "▁the")
    std::optional<int> id_of(const std::string& tok) const;
    bool is_special(int id) const;            // an added token marked special (skipped by decode(skip_special))
    // the pre-tokenizer's pieces of one stretch of text without added tokens (raw bytes, not byte-level mapped;
    // byte fallback: after the normalizer, so spaces are U+2581)
    std::vector<std::string> pre_tokens(const std::string& text) const;

private:
    Tokenizer();
    struct Impl;
    std::unique_ptr<Impl> d;
};

// Incremental detokenizer: the text exllamav3's generator streams per sampled token. Each token adds its
// token_text(); while the held text ends in U+FFFD (a character split across tokens) the held tokens are decoded
// together (special tokens skipped), and text is held until that decode no longer ends in U+FFFD or is longer than
// max_stop_len + 20 code points (max_stop_len: the longest stop string of the request, 0 without any).
// Stop tokens: do not push them. exllamav3 drops the held text when one ends the reply (the stop token's text
// and any partial character before it), so just discard the decoder.
class StreamDecoder {
public:
    explicit StreamDecoder(const Tokenizer& tok, size_t max_stop_len = 0) : tok_(tok), max_stop_len_(max_stop_len) {}
    // Text to stream after token id; last=true for the token that reaches max_new_tokens, which emits everything
    // held as the raw per-token texts without the joint decode (exllamav3 does exactly that).
    std::string push(int id, bool last = false);
    // Everything held, as raw per-token texts, and reset.
    std::string flush();

private:
    const Tokenizer& tok_;
    size_t max_stop_len_;
    std::string held_;
    std::vector<int> held_ids_;
};

// UTF-8 helpers shared by the tokenizer and the chat template.
std::string utf8_lossy(const std::string& bytes);    // Rust from_utf8_lossy / Python decode(errors="replace")
size_t utf8_length(const std::string& s);            // code points of valid UTF-8 (Python len)

}  // namespace glm
