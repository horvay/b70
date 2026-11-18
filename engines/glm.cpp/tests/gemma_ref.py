"""Reference outputs for tests/gemma_check.cpp: HF `tokenizers` and HF's Jinja rendering of Gemma 4's tokenizer and
chat template, without torch and without a GPU. Any Python with tokenizers, jinja2 and transformers:

  S=<scratch dir>; BWRAP_EXTRA="--bind $S /s --ro-bind $PWD /g" GLM53_MODELS=<Gemma model dir> \
      ~/b70/llm/glm/scripts/in_rootfs.sh python3 -I /g/tests/gemma_ref.py /models /s/ref \
      /w/exllamav3/exllamav3/conversion/standard_cal_data /g

  model_dir out_dir cal_dir repo_dir: repo_dir's *.md files and cal_dir's *.utf8 (exllamav3's calibration texts) are the
  long texts. Nothing here reads a chat log; the conversations below are written out by hand.

- encode: HF encode(text, add_special_tokens=False) with encode_special_tokens = not special.
- stream: exllamav3 generator/job.py receive_sample as tests/tok_ref.py replicates it.
- render: transformers' render_jinja_template (what apply_chat_template calls) over chat_template.jinja plus the client's prefill
  hook, with its tokenizer's special tokens as variables. A template error gives text null: the port must fail as well.
- serve: serving_chat's message shaping as read from SGLang's source, written out in shape() below (SGLang is not imported
  or run), then the same template. A transcription, so it checks the port against my reading of that code.
- roundtrip: tool calls rendered by the template (its format_argument), for the parser to read back. SGLang's parser is
  not run; gemma_check.cpp has hand-derived cases for its quirks, and the expected values here apply two of them
  (a float whose repr has no '.' stays a string).
"""
import json
import os
import random
import sys
import time

sys.modules["torch"] = None          # importing torch would load the GPU stack: make any attempt fail loudly
MODEL, OUT, CAL, REPO = sys.argv[1:5]
os.makedirs(OUT, exist_ok=True)

from tokenizers import Tokenizer  # noqa: E402
from transformers.utils.chat_template_utils import render_jinja_template  # noqa: E402

tok = Tokenizer.from_file(os.path.join(MODEL, "tokenizer.json"))
tok.no_truncation()
RAW_V = tok.get_vocab_size()
PAD, EOS = tok.token_to_id("<pad>"), tok.token_to_id("<eos>")
tj = json.load(open(os.path.join(MODEL, "tokenizer.json"), encoding="utf-8"))
ADDED = [a["content"] for a in tj["added_tokens"]]
SPECIAL = {a["id"] for a in tj["added_tokens"] if a["special"]}


def encode(text, special=True):
    tok.encode_special_tokens = not special
    return tok.encode(text, add_special_tokens=False).ids


t_ids = encode(" t")
assert len(t_ids) == 1 and len(tok.decode([t_ids[0]])) == 2   # exllamav3 _get_fixed_vocab: single-token decodes
PIECES = tok.decode_batch([[i] for i in range(RAW_V)], skip_special_tokens=False)


def ex_decode(ids):
    seq = [t for t in ids if t != PAD and t < RAW_V and t != EOS]
    return tok.decode(seq, skip_special_tokens=True)


def stream(ids, max_stop=0):
    held, held_ids, out = "", [], []
    for n, t in enumerate(ids):
        held += PIECES[t]
        held_ids.append(t)
        if n == len(ids) - 1:
            out.append(held)
            break
        if held.endswith("�"):
            td = ex_decode(held_ids)
            if td.endswith("�") and len(td) <= max_stop + 20:
                out.append("")
                continue
            held = td
        out.append(held)
        held, held_ids = "", []
    return out


# ------------------------------------------------------------------------------------------------ text corpus

def read(p):
    return open(p, encoding="utf-8").read()


def synthetic():
    s = [
        "", " ", "  ", "a", "Hello world", "Hello, world! How are you?", "  leading spaces", "trailing spaces   ",
        "▁", "▁▁", " ▁ ", "a▁b", "▁the the", "<0x41>", "<0xE4><0xBD><0xA0>", "<0xZZ>", "<0x+F>",
        "<unused0>", "<unused99>", "[multimodal]", "<pad> <eos>", "<start_of_image>",
        "I'm you're he'll she'd it's we've don't can't", "1234567890", "3.14159", "1,000,000.00", "١٢٣٤", "①②③", "½ ¼ ⅓",
        "a  b", "a   \n\n  b", "\t\t", "  \n", " \r\n ", "\r\n\r\n", "  x", "　x　", "a b",
        "a\u0085b", "a​b", "  ", "\n", "\n\n\n", "a\nb", "a\r\nb", "a\rb", " \n \n ", "x \n\t y",
        "你好，世界！", "こんにちは、世界。カタカナ", "안녕하세요", "Привет, мир!", "Γειά σου", "مرحبا بالعالم", "שלום", "नमस्ते", "สวัสดี",
        "é à̀ ñ", "\U0001F600\U0001F44D\U0001F3FD", "\U0001F468‍\U0001F469‍\U0001F467", "\U0001F1FA\U0001F1F8", "❤️",
        "emoji\U0001F600text", " \U0001F600 ", "\U0001F600\U0001F600\U0001F600", "\U0001FAE0\U0001FAE1\U0001FAE2",
        "\U00020000\U00020001\U0002A6D6", "𝔘𝔫𝔦𝔠𝔬𝔡𝔢", "", "\U000F0000\U0010FFFD", "�", "a�b", "\x00\x01\x1f\x7f", "tab\x0bvt\x0cff",
        "­‎", "def f(x):\n    return x * 2\n\nprint(f(21))\n", "{\"a\": [1, 2, {\"b\": null}], \"c\": \"d\\n\"}",
        "<div class=\"x\">&amp; &lt;</div>", "```python\nimport os\n```", "| a | b |\n|---|---|\n| 1 | 2 |", "!!!???...",
        "http://example.com/a?b=c&d=e#f", "user@example.com", "$100 €50 £30", "été naïve café über straße", "İstanbul ıİ", "ẞß",
        "a" * 5000, " " * 3000, "\n" * 1000, "\t" * 300, "1" * 100, "ab" * 1000, "a " * 1000, "一" * 500, "! " * 300,
        "x" + " " * 50 + "y", " " * 7 + "\n" * 3 + " " * 2, "▁" * 100, "\U0001F600" * 200,
    ]
    for t in ADDED:
        s += ["x" + t + "y", " " + t + " ", t + t, "\n" + t + "\n", "ab" + t + "cd ef", t, t + " " + t, "  " + t + "  x",
              t + "\n\n", "1" + t + "2", "\U0001F600" + t + "你"]
    s += ["<|turn", "<|channel", "<channel|", "<|\"|", "<|tool_call>call:f{a:<|\"|>x<|\"|>}<tool_call|>",
          "<|channel>thought\nhm<channel|>done<turn|>", "<bos><|turn>user\nhi<turn|>\n<|turn>model\n", "<|image|><|audio|><|video|>"]
    return s


def random_strings(n, seed):
    rnd = random.Random(seed)
    palette = (list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") + list(" " * 12) +
               list("\n\n\n\t\r") + list(".,;:!?'\"()[]{}<>-_=+*/\\|@#$%^&`~") +
               [" ", "　", " ", "\u0085", "​", "é", "ß", "ſ", "İ", "’", "▁", "<0x41>"] +
               list("你好世界こんカ안녕Прمرשनส") +
               ["́", "̀", "‍", "️", "\U0001F600", "\U0001F44D", "\U0001F3FD", "\U0001F1FA", "�", "\U0001FAE0", "\U00020000"] +
               list("١٢²½①Ⅻ"))
    out = []
    for _ in range(n):
        L = rnd.choice([1, 2, 3, 5, 8, 13, 30, 80, 200])
        parts = []
        for _ in range(L):
            if rnd.random() < 0.03:
                parts.append(rnd.choice(ADDED))
            elif rnd.random() < 0.05:
                parts.append(chr(rnd.randrange(0x20, 0x30000)) if rnd.random() < 0.7 else chr(rnd.randrange(0x80, 0x800)))
            else:
                parts.append(rnd.choice(palette))
        out.append("".join(c for c in "".join(parts) if not 0xD800 <= ord(c) <= 0xDFFF))
    return out


def probe(c):
    """Context probe for one code point (the same string gemma_check.cpp builds)."""
    return ("x" + c + "x 1" + c + "1 " + c + "x !" + c + "! \n" + c + "\n'" + c + "'" + c + c + " " + c + " \t" + c + "\t ")


def fnv(ids):
    h = 0xcbf29ce484222325
    for t in ids:
        for b in t.to_bytes(4, "little"):
            h ^= b
            h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


# ------------------------------------------------------------------------------------------------ template battery

HOOK = """
{#- client hook: an open thinking prefill after the generation prompt, from chat_template_kwargs.nq_prefill -#}
{%- if add_generation_prompt and nq_prefill is defined and nq_prefill -%}{{- nq_prefill -}}{%- endif -%}
"""
TEMPLATE_MODEL = read(os.path.join(MODEL, "chat_template.jinja"))
TEMPLATE = TEMPLATE_MODEL + HOOK
TVARS = {"bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>", "unk_token": "<unk>"}


def fn(name, description="d", parameters=None, **extra):
    f = {"name": name, "description": description}
    if parameters is not None:
        f["parameters"] = parameters
    f.update(extra)
    return {"type": "function", "function": f}


TOOLS = [
    fn("roll", "Roll a fair die: uniform in [1, n]. — \"quoted\" \\ slash \n two lines é你\U0001F600",
       {"type": "object", "properties": {"n": {"type": "number", "description": "faces", "minimum": 1, "maximum": 1e6, "default": 6.0}},
        "required": ["n"]}),
    fn("kinds", "every property kind", {"type": "object", "required": ["s", "a"], "properties": {
        "Zeta": {"type": "string", "description": "mixed-case keys sort without case"},
        "alpha": {"type": "string", "enum": ["a", "b", "c"], "description": "an enum"},
        "beta": {"type": "string", "enum": ["only"], "nullable": True},
        "s": {"type": "STRING"},
        "i": {"type": "integer", "nullable": True, "description": "nullable int"},
        "a": {"type": "array", "description": "strings", "items": {"type": "string", "enum": ["x", "y"], "nullable": None}},
        "a2": {"type": "array", "items": {"type": ["string", "null"], "description": "union type", "minItems": 1, "other": {"K": [1, 2.5, None, True, "v"]}}},
        "a3": {"type": "array", "items": {"type": "object", "required": ["p"], "properties": {"p": {"type": "string"}, "q": {"type": "array", "items": {"type": "integer"}}}}},
        "a4": {"type": "array", "items": {}},
        "a5": {"type": "array"},
        "a6": {"type": "array", "items": {"properties": "not a mapping", "required": "pq"}},
        "o": {"type": "object", "description": "nested", "required": ["x"], "properties": {"x": {"type": "boolean"}, "y": {"type": "object", "properties": {"z": {"type": "number"}}}}},
        "o2": {"type": "object", "additionalProperties": {"type": "string"}, "minProperties": 1, "title": "free", "nullable": True},
        "o3": {"type": "object", "required": ["k"], "k": {"type": "string", "description": "a property written in place"}},
        "o4": {"type": "object", "properties": {}},
        "o5": {"type": "object", "properties": "bad"},
        "u": {"type": ["string", "null"], "description": "a union"},
        "n": {"description": "no type"},
        "e": {},
        "t": {"type": "boolean", "enum": [True, False], "default": False},
    }}),
    fn("noparams", "no parameters at all"),
    fn("emptyparams", "empty parameters", {}),
    fn("notype", "parameters without a type leave a brace open", {"properties": {"p": {"type": "string"}}, "required": ["p"]}),
    fn("onlytype", "only a type", {"type": "object"}),
    fn("resp", "with a response", {"type": "object", "properties": {}}, response={"type": "object", "description": "what it returns"}),
    fn("resp2", "response of another type", {"type": "object"}, response={"type": "string"}),
    fn("resp3", "empty response", response={}),
    {"type": "function", "function": {"name": "nodesc", "parameters": {"type": "object"}}},
    fn("nulldesc", None),
]
TOOLS_SMALL = [TOOLS[0]]
ALL_ARGS = {"s": "text \"q\" : , { } [ ] é你\U0001F600\nline", "i": 42, "f": 0.1, "t": True, "fa": False, "n": None, "big": 1e300, "small": 1e-7, "neg0": -0.0,
            "l": [1, "two", 3.0, None, True, [], {}], "d": {"B": "vé\n", "a": [{"z": 1, "Y": 2}]}, "huge": 123456789012345678901234567890, "int_float": 100.0, "e": "", "Upper": 1, "lower": 2}
TC = [{"id": "call_1", "type": "function", "function": {"name": "roll", "arguments": {"n": 20}}},
      {"id": "call_2", "type": "function", "function": {"name": "search", "arguments": ALL_ARGS}}]
TC_NOID = [{"function": {"name": "roll", "arguments": {"n": 20}}}, {"function": {"name": "search", "arguments": {}}}]


def conversations():
    sys_m = {"role": "system", "content": "  You are the narrator. Stay in character.\n\nRules:\n- be terse  \n"}
    u1 = {"role": "user", "content": "I open the door."}
    a1 = {"role": "assistant", "content": "The door creaks open.", "reasoning_content": "The player opens the door; describe."}
    u2 = {"role": "user", "content": "  I step inside. 你好 \U0001F600\n"}
    c = {}
    c["single_user"] = [u1]
    c["system_user"] = [sys_m, u1]
    c["developer_user"] = [{"role": "developer", "content": "dev instructions"}, u1]
    c["developer_later"] = [u1, {"role": "developer", "content": "later dev"}, u2]
    c["system_later"] = [u1, {"role": "system", "content": " later sys "}, u2]
    c["system_only"] = [sys_m]
    c["system_parts"] = [{"role": "system", "content": [{"type": "text", "text": " Part one. "}, {"type": "text", "text": "Part two.\n"}, {"type": "text"}, {"type": "image"}, "bare"]}, u1]
    c["system_none"] = [{"role": "system", "content": None}, u1]
    c["system_nokey"] = [{"role": "system"}, u1]
    c["system_empty"] = [{"role": "system", "content": ""}, u1]
    c["system_dict"] = [{"role": "system", "content": {"a": 1, "b": 2}}, u1]
    c["system_number"] = [{"role": "system", "content": 5}, u1]
    c["multi_turn"] = [sys_m, u1, a1, u2]
    c["assistant_last"] = [sys_m, u1, a1]
    c["reasoning_after_last_user"] = [u1, a1, u2, {"role": "assistant", "content": "ok", "reasoning_content": "thinking now"}]
    c["reasoning_key"] = [u1, {"role": "assistant", "content": "ok", "reasoning": "from reasoning", "reasoning_content": "from reasoning_content"}]
    c["reasoning_empty_falls_through"] = [u1, {"role": "assistant", "content": "ok", "reasoning": "", "reasoning_content": "second"}]
    c["reasoning_number"] = [u1, {"role": "assistant", "content": "ok", "reasoning": 5}]
    c["reasoning_tool_preserve"] = [u1, {"role": "assistant", "content": "", "reasoning_content": "before the call", "tool_calls": [TC[0]]},
                                    {"role": "tool", "tool_call_id": "call_1", "content": "5"}, u2, {"role": "assistant", "content": "done", "reasoning_content": "after"}, u1]
    c["strip_thinking"] = [u1, {"role": "assistant", "content": "  <|channel>thought\nhidden<channel|>Visible  <|channel>x<channel|> tail <|channel>never closed"}, u2]
    c["strip_thinking_close_only"] = [u1, {"role": "assistant", "content": "a<channel|>b<channel|>c"}, u2]
    c["user_keeps_markers"] = [{"role": "user", "content": " <|channel>thought\nnot stripped<channel|> "}]
    c["two_assistants"] = [u1, {"role": "assistant", "content": "one", "reasoning_content": "r1"}, {"role": "assistant", "content": "two"}]
    c["three_assistants_end"] = [u1, {"role": "assistant", "content": "one"}, {"role": "assistant", "content": ""}, {"role": "assistant", "content": "three"}, u2]
    c["model_role_literal"] = [u1, {"role": "model", "content": "as model <|channel>x<channel|>y"}, {"role": "assistant", "content": "next"}]
    c["parts_user"] = [{"role": "user", "content": [{"type": "text", "text": " look: "}, {"type": "image_url", "image_url": {"url": "x"}}, {"type": "image"}, {"type": "audio"},
                                                    {"type": "input_audio"}, {"type": "video"}, {"type": "video_url"}, {"type": "other"}, {"type": "text", "text": " and\n"}]}]
    c["parts_assistant"] = [u1, {"role": "assistant", "content": [{"type": "text", "text": " <|channel>h<channel|> first "}, {"type": "image"}, {"type": "text", "text": "second "}]}, u2]
    c["parts_string_item"] = [{"role": "user", "content": ["bare string"]}]
    c["parts_text_number"] = [{"role": "user", "content": [{"type": "text", "text": 5}]}]
    c["parts_assistant_number"] = [u1, {"role": "assistant", "content": [{"type": "text", "text": 5}]}]
    c["parts_text_missing_user"] = [{"role": "user", "content": [{"type": "text"}]}]
    c["parts_text_missing_assistant"] = [u1, {"role": "assistant", "content": [{"type": "text"}]}]
    c["content_dict"] = [{"role": "user", "content": {"a": 1}}]
    c["content_empty_dict"] = [{"role": "user", "content": {}}]
    c["content_number"] = [{"role": "user", "content": 5}, {"role": "assistant", "content": 2.5}]
    c["content_none"] = [{"role": "user", "content": None}, {"role": "assistant", "content": None}]
    c["content_nokey"] = [{"role": "user"}, {"role": "assistant"}]
    c["whitespace"] = [{"role": "user", "content": "  spaced  \n"}, {"role": "assistant", "content": " \n\t "}, {"role": "user", "content": ""}]
    c["unicode_strip"] = [{"role": "user", "content": "　  reply  \x1c"}, {"role": "assistant", "content": "　 reply \x1c"}]
    c["unknown_role"] = [{"role": "narrator", "content": "told"}, u1]
    c["no_role"] = [{"content": "no role"}]
    c["role_number"] = [{"role": 5, "content": "x"}]
    c["empty"] = []
    # tool calls
    c["call_then_results"] = [sys_m, u1, {"role": "assistant", "content": "Let me roll.", "reasoning_content": "need a roll", "tool_calls": TC},
                              {"role": "tool", "tool_call_id": "call_1", "content": "17"}, {"role": "tool", "tool_call_id": "call_2", "content": "{\"hits\": []}"}]
    c["call_results_then_user"] = [u1, {"role": "assistant", "content": "", "tool_calls": [TC[0]]}, {"role": "tool", "tool_call_id": "call_1", "content": "17"}, u2]
    c["call_results_then_assistant"] = [u1, {"role": "assistant", "content": "", "tool_calls": [TC[0]]}, {"role": "tool", "tool_call_id": "call_1", "content": "17"},
                                        {"role": "assistant", "content": "You rolled 17.", "reasoning_content": "it was 17"}]
    c["call_results_assistant_user"] = [u1, {"role": "assistant", "content": "x", "tool_calls": [TC[0]]}, {"role": "tool", "tool_call_id": "call_1", "content": "17"},
                                        {"role": "assistant", "content": "You rolled 17."}, u2, {"role": "assistant", "content": "", "tool_calls": [TC[1]]},
                                        {"role": "tool", "tool_call_id": "call_2", "content": "none"}]
    c["call_no_results"] = [u1, {"role": "assistant", "content": "", "tool_calls": TC}]
    c["call_no_results_then_user"] = [u1, {"role": "assistant", "content": "x", "tool_calls": [TC[0]]}, u2]
    c["call_then_assistant_no_results"] = [u1, {"role": "assistant", "content": "x", "tool_calls": [TC[0]]}, {"role": "assistant", "content": "y"}]
    c["no_ids"] = [u1, {"role": "assistant", "content": "", "tool_calls": TC_NOID}, {"role": "tool", "content": "a"}, {"role": "tool", "content": "b"}]
    c["ids_mismatch"] = [u1, {"role": "assistant", "content": "", "tool_calls": [TC[0]]}, {"role": "tool", "tool_call_id": "nope", "content": "a"},
                         {"role": "tool", "tool_call_id": "nope2", "name": "named", "content": "b"}, {"role": "tool", "name": "", "tool_call_id": "x", "content": "c"}]
    c["id_numbers"] = [u1, {"role": "assistant", "content": "", "tool_calls": [{"id": 7, "function": {"name": "num", "arguments": {}}}, {"id": 8.0, "function": {"name": "flt", "arguments": {}}}]},
                       {"role": "tool", "tool_call_id": 7, "content": "a"}, {"role": "tool", "tool_call_id": 8, "content": "b"}, {"role": "tool", "tool_call_id": 7.0, "content": "c"}]
    c["tool_contents"] = [u1, {"role": "assistant", "content": "", "tool_calls": [TC[0]]},
                          {"role": "tool", "tool_call_id": "call_1", "content": [{"type": "text", "text": "part1 "}, {"type": "image_url"}, {"type": "text", "text": "part2"}, {"type": "audio"}, {"type": "video"}, {"type": "x"}]},
                          {"role": "tool", "tool_call_id": "call_1", "content": None}, {"role": "tool", "tool_call_id": "call_1"},
                          {"role": "tool", "tool_call_id": "call_1", "content": 5}, {"role": "tool", "tool_call_id": "call_1", "content": 2.5},
                          {"role": "tool", "tool_call_id": "call_1", "content": True}, {"role": "tool", "tool_call_id": "call_1", "content": {}},
                          {"role": "tool", "tool_call_id": "call_1", "content": []}, {"role": "tool", "tool_call_id": "call_1", "content": ""}]
    c["tool_content_dict"] = [u1, {"role": "assistant", "content": "", "tool_calls": [TC[0]]}, {"role": "tool", "tool_call_id": "call_1", "content": {"a": 1}}]
    c["tool_content_text_missing"] = [u1, {"role": "assistant", "content": "", "tool_calls": [TC[0]]}, {"role": "tool", "tool_call_id": "call_1", "content": [{"type": "text"}]}]
    c["tool_content_string_part"] = [u1, {"role": "assistant", "content": "", "tool_calls": [TC[0]]}, {"role": "tool", "tool_call_id": "call_1", "content": ["bare"]}]
    c["tool_orphan"] = [u1, {"role": "tool", "content": "lonely"}]
    c["tool_first"] = [{"role": "tool", "content": "lonely"}, u1]
    c["tool_after_plain_assistant"] = [u1, a1, {"role": "tool", "content": "dropped"}, u2]
    c["args_string"] = [u1, {"role": "assistant", "content": "", "tool_calls": [{"id": "c", "function": {"name": "roll", "arguments": "{\"n\": 6, \"s\": \"x\"}"}}]}]
    c["args_none"] = [u1, {"role": "assistant", "content": "", "tool_calls": [{"id": "c", "function": {"name": "roll", "arguments": None}}]}]
    c["args_missing"] = [u1, {"role": "assistant", "content": "", "tool_calls": [{"id": "c", "function": {"name": "roll"}}]}]
    c["args_list"] = [u1, {"role": "assistant", "content": "", "tool_calls": [{"id": "c", "function": {"name": "roll", "arguments": [1, 2]}}]}]
    c["call_no_name"] = [u1, {"role": "assistant", "content": "", "tool_calls": [{"id": "c", "function": {"arguments": {}}}]}]
    c["call_no_function"] = [u1, {"role": "assistant", "content": "", "tool_calls": [{"id": "c", "name": "roll", "arguments": {}}]}]
    c["empty_tool_calls"] = [u1, {"role": "assistant", "content": "plain", "tool_calls": []}, u2]
    # legacy tool_responses
    c["legacy"] = [u1, {"role": "assistant", "content": "", "tool_calls": [TC[0]], "tool_responses": [{"name": "roll", "response": {"z": 1, "A": "two", "m": {"k": None}}}]}]
    c["legacy_scalars"] = [u1, {"role": "assistant", "content": "after", "tool_responses": [{"name": "", "response": "text"}, {"response": 5}, {"name": None, "response": [1, "x"]}, {"name": "n", "response": None}, {"name": "m"}]}, u2]
    c["legacy_then_assistant"] = [u1, {"role": "assistant", "tool_responses": [{"name": "f", "response": {"a": 1}}]}, {"role": "assistant", "content": "next"}]
    c["legacy_over_tool_messages"] = [u1, {"role": "assistant", "content": "", "tool_calls": [TC[0]], "tool_responses": [{"name": "f", "response": "embedded"}]}, {"role": "tool", "tool_call_id": "call_1", "content": "ignored"}]
    c["legacy_name_number"] = [u1, {"role": "assistant", "tool_responses": [{"name": 5, "response": "x"}]}]
    return c


VAR_SETS = [{}, {"add_generation_prompt": True}, {"add_generation_prompt": True, "enable_thinking": True}, {"add_generation_prompt": True, "enable_thinking": False},
            {"add_generation_prompt": True, "preserve_thinking": True}, {"add_generation_prompt": True, "enable_thinking": True, "preserve_thinking": True},
            {"add_generation_prompt": True, "nq_prefill": "The player wants"}, {"add_generation_prompt": True, "enable_thinking": True, "nq_prefill": "<|channel>thought\nA prefill"},
            {"add_generation_prompt": True, "enable_thinking": True, "nq_prefill": "<|channel>thought\nall of it\n<channel|>"},
            {"add_generation_prompt": True, "enable_thinking": 1}, {"add_generation_prompt": True, "enable_thinking": None}, {"add_generation_prompt": True, "enable_thinking": "no"},
            {"add_generation_prompt": True, "nq_prefill": 7}, {"add_generation_prompt": True, "nq_prefill": True}, {"add_generation_prompt": True, "nq_prefill": ""}, {"add_generation_prompt": False, "nq_prefill": "x"},
            {"add_generation_prompt": True, "reasoning_effort": "low", "something_else": [1, 2]}]


def rt(x):
    return json.loads(json.dumps(x))       # what the C++ side parses


def render(msgs, tools, vars_, template=TEMPLATE):
    out, _ = render_jinja_template(conversations=[msgs], tools=tools, chat_template=template, **TVARS, **vars_)
    return out[0]


def shape(messages):
    """serving_chat.py's jinja path as read: normalize_assistant_tool_call_arguments (strict), content None -> "",
    process_content_for_template_format("openai"), normalize_tool_content. Raises what SGLang raises."""
    out = []
    for m in json.loads(json.dumps(messages)):
        if m.get("role") == "assistant" and isinstance(m.get("tool_calls"), list):
            for item in m["tool_calls"]:
                f = item.get("function") if isinstance(item, dict) else None
                if isinstance(f, dict) and isinstance(f.get("arguments"), str):
                    parsed = json.loads(f["arguments"])
                    if not isinstance(parsed, dict):
                        raise ValueError("not an object")
                    f["arguments"] = parsed
        if m.get("content") is None:
            m["content"] = ""
        if isinstance(m["content"], list):
            parts = []
            for ch in m["content"]:
                if not isinstance(ch, dict):
                    continue
                t = ch.get("type")
                if t in ("image_url", "input_image"):
                    parts.append({"type": "image"})
                elif t == "video_url":
                    parts.append({"type": "video"})
                elif t == "audio_url":
                    parts.append({"type": "audio"})
                elif t in ("text", "input_text"):
                    parts.append({"type": "text", "text": ch["text"]})
                elif t == "tool_reference":
                    parts.append(ch)
            m["content"] = parts
            if m.get("role") == "tool" and all(p.get("type") == "text" for p in parts):
                m["content"] = " ".join(p.get("text", "") for p in parts)
        out.append(m)
    return out


def serve_cases():
    conv = conversations()
    out = []
    for name in ["single_user", "system_user", "multi_turn", "reasoning_after_last_user", "strip_thinking", "call_then_results", "call_results_then_user", "call_no_results", "no_ids", "parts_user",
                 "developer_later", "two_assistants", "tool_orphan", "ids_mismatch"]:
        for kw in [None, {}, {"enable_thinking": True}, {"enable_thinking": False}, {"enable_thinking": True, "nq_prefill": "<|channel>thought\nprefill"}, {"enable_thinking": 1}, {"preserve_thinking": True}]:
            for tools in [None, TOOLS_SMALL]:
                out.append((f"{name}/{kw}/{'tools' if tools else 'notools'}", {"messages": conv[name], "tools": tools, "chat_template_kwargs": kw}))
    out.append(("effort_ignored", {"messages": conv["multi_turn"], "reasoning_effort": "high"}))
    out.append(("null_content", {"messages": [conv["multi_turn"][1], {"role": "assistant", "content": None}, conv["multi_turn"][3]]}))
    tc_str = [{"id": "c1", "type": "function", "function": {"name": "roll", "arguments": "{\"z\": 1, \"a\": \"x\"}"}}]
    out.append(("args_string_parsed", {"messages": [conv["multi_turn"][1], {"role": "assistant", "content": None, "tool_calls": tc_str}, {"role": "tool", "tool_call_id": "c1", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}], "tools": TOOLS_SMALL}))
    out.append(("args_string_bad", {"messages": [conv["multi_turn"][1], {"role": "assistant", "content": "", "tool_calls": [{"id": "c", "function": {"name": "roll", "arguments": "not json {"}}]}]}))
    out.append(("args_string_list", {"messages": [conv["multi_turn"][1], {"role": "assistant", "content": "", "tool_calls": [{"id": "c", "function": {"name": "roll", "arguments": "[1, 2]"}}]}]}))
    out.append(("parts_shaped", {"messages": [{"role": "user", "content": [{"type": "input_text", "text": "in"}, {"type": "input_image"}, {"type": "audio_url"}, {"type": "video_url"}, {"type": "input_audio"}, {"type": "image"}, "bare", {"type": "text", "text": "after"}]}]}))
    out.append(("tool_mixed_parts", {"messages": [conv["multi_turn"][1], {"role": "assistant", "content": "", "tool_calls": [TC[0]]}, {"role": "tool", "tool_call_id": "call_1", "content": [{"type": "text", "text": "a"}, {"type": "image_url"}, {"type": "text", "text": "b"}]}]}))
    return out


def roundtrip_cases():
    """(arguments, what the parser must return for them): the rendered text is the model's own format"""
    def want(x):
        if x is None:
            return None
        if isinstance(x, float):
            r = repr(x)
            return x if "." in r else r       # 1e+300 has no '.': int() fails, so it stays a bare string
        if isinstance(x, list):
            return [want(v) for v in x]
        if isinstance(x, dict):
            return {k: want(x[k]) for k in sorted(x, key=str.lower)}
        return x
    args = [{}, {"n": 20}, ALL_ARGS, {"s": ""}, {"a": []}, {"o": {}}, {"a": [[1, [2, [3]]], {"b": [{"c": {"d": "e"}}]}]}, {"k": "}{][,:"}, {"k": ["}{][,:", {"m": "}{"}]},
            {"long": "line one\nline two\n\n  indented é你\U0001F600 " * 20}, {"f": 1.5e-07, "g": 2.5e+20, "h": 1e16, "i": 1e15, "j": -3.25, "m": 0.0}, {"t": True, "u": False}, {"neg": -7, "zero": 0},
            {"name with space": 1, "Ünï": "x", "a.b": "y"}]
    return [(a, want(a)) for a in args]


def main():
    t0 = time.time()
    conv = conversations()
    # ---- prompts
    render_out, n_err = [], 0
    for name, msgs in conv.items():
        for vi, v in enumerate(VAR_SETS):
            for tname, tools in [("none", None), ("small", TOOLS_SMALL), ("all", TOOLS), ("empty", [])]:
                m, tl = rt(msgs), rt(tools)
                try:
                    text = render(m, tl, v)
                except Exception as e:  # noqa: BLE001  (the port must fail as well)
                    text = None
                    n_err += 1
                    if vi == 1 and tname == "none":
                        print(f"render {name}: {type(e).__name__}: {str(e)[:100]}")
                render_out.append({"name": f"{name}/{vi}/{tname}", "messages": m, "tools": tl, "vars": rt(v), "text": text})
    # the unmodified template renders the same when nq_prefill is not set
    for name, msgs in conv.items():
        for v in VAR_SETS[:6]:
            m = rt(msgs)
            try:
                a = render(m, rt(TOOLS), v)
            except Exception:  # noqa: BLE001
                continue
            assert a == render(m, rt(TOOLS), v, TEMPLATE_MODEL), name
    serve_out = []
    for name, body in serve_cases():
        body = {k: v for k, v in rt(body).items() if v is not None}
        kw = dict(body.get("chat_template_kwargs") or {})
        if body.get("reasoning_effort") is not None:
            kw.setdefault("reasoning_effort", body["reasoning_effort"])
        kw["add_generation_prompt"] = True
        try:
            text = render(shape(body["messages"]), body.get("tools"), kw)
        except Exception as e:  # noqa: BLE001
            print(f"serve {name}: {type(e).__name__}: {str(e)[:100]}")
            text = None
        serve_out.append({"name": name, "body": body, "text": text, "thinking": (body.get("chat_template_kwargs") or {}).get("enable_thinking") is True,
                          "ids": encode(text) if text is not None else None})
    rts = []
    for a, want in roundtrip_cases():
        msgs = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "fn_name", "arguments": rt(a)}}]}]
        text = render(rt(msgs), None, {})
        rts.append({"text": text[text.index("<|tool_call>"):text.index("<tool_call|>") + len("<tool_call|>")], "args": rt(want)})
    json.dump({"render": render_out, "serve": serve_out, "roundtrip": rts}, open(f"{OUT}/chat.json", "w"), ensure_ascii=False)
    print(f"prompts: {len(render_out)} template cases ({n_err} template errors), {len(serve_out)} serve cases, {len(rts)} tool-call round trips ({time.time() - t0:.1f} s)")

    # ---- encode
    texts = []
    long_texts = []
    for f in sorted(os.listdir(CAL)):
        if f.endswith(".utf8"):
            long_texts.append(("cal:" + f, read(os.path.join(CAL, f))))
    for f in sorted(os.listdir(REPO)):
        if f.endswith(".md"):
            long_texts.append(("doc:" + f, read(os.path.join(REPO, f))))
    long_texts.append(("file:chat_template.jinja", TEMPLATE_MODEL))
    long_texts.append(("file:tokenizer_config", read(os.path.join(MODEL, "tokenizer_config.json"))))
    for name, s in long_texts:
        # 16K-character pieces, so a difference shows where it is; each piece is one BPE word in HF as well
        for k in range(0, len(s), 16384):
            texts.append((f"{name}:{k}", s[k:k + 16384], [True]))
    for i, s in enumerate(synthetic()):
        texts.append((f"synthetic:{i}", s, [True, False]))
    for i, s in enumerate(random_strings(4000, 1234)):
        texts.append((f"random:{i}", s, [True, False]))
    for c in serve_out:
        if c["text"] is not None:
            texts.append(("prompt:" + c["name"], c["text"], [True, False]))
    enc = []
    n_tok = 0
    for name, s, specials in texts:
        for special in specials:
            ids = encode(s, special)
            n_tok += len(ids)
            enc.append({"name": name, "text": s, "special": special, "ids": ids})
    big = "\n\n".join(s for _, s in long_texts[:1])[:120000]
    t1 = time.perf_counter()
    big_ids = encode(big)
    t_big = time.perf_counter() - t1
    json.dump({"cases": enc, "big": {"text": big, "ids": big_ids, "py_ms": t_big * 1000}}, open(f"{OUT}/encode.json", "w"), ensure_ascii=False)
    print(f"encode: {len(enc)} cases, {n_tok} tokens; big text {len(big_ids)} tokens in {t_big * 1000:.1f} ms (HF)")

    # ---- code point sweep
    blocks = []
    cps = [c for c in range(0x110000) if not 0xD800 <= c <= 0xDFFF]
    B = 4096
    batch = []
    for k in range(0, len(cps), B):
        blk = cps[k:k + B]
        batch.append((blk[0], blk[-1], "".join(probe(chr(c)) for c in blk)))
    tok.encode_special_tokens = False
    encs = tok.encode_batch([b[2] for b in batch], add_special_tokens=False)
    for (lo, hi, _), e in zip(batch, encs):
        blocks.append({"lo": lo, "hi": hi, "n": len(e.ids), "hash": format(fnv(e.ids), "016x")})
    json.dump({"block": B, "blocks": blocks}, open(f"{OUT}/sweep.json", "w"))
    print(f"sweep: {len(blocks)} blocks of {B} code points")

    # ---- decode, token pieces, streaming
    rnd = random.Random(99)
    byte_ids = [tok.token_to_id(f"<0x{b:02X}>") for b in range(256)]
    specials = sorted(SPECIAL)
    dec = []
    for c in enc[::7]:
        for skip in (False, True):
            dec.append({"ids": c["ids"], "skip": skip, "text": tok.decode(c["ids"], skip_special_tokens=skip)})
    utf8_runs = [[byte_ids[b] for b in ch.encode("utf-8")] for ch in "é你\U0001F600▁\U00020000"]
    for k in range(4000):
        n = rnd.choice([1, 2, 3, 5, 10, 50, 300])
        pool = [byte_ids, range(RAW_V), byte_ids + specials][k % 3]
        ids = []
        while len(ids) < n:
            if rnd.random() < 0.2:
                ids += rnd.choice(utf8_runs)
            else:
                ids.append(rnd.choice(pool))
        for skip in (False, True):
            dec.append({"ids": ids, "skip": skip, "text": tok.decode(ids, skip_special_tokens=skip)})
    streams = []
    stop_ids = {PAD, EOS, tok.token_to_id("<turn|>"), tok.token_to_id("<|tool_response>")}
    for c in enc[::5]:
        ids = [t for t in c["ids"] if t not in stop_ids]
        if ids:
            streams.append({"ids": ids, "max_stop": 0, "chunks": stream(ids)})
    for k in range(3000):
        n = rnd.choice([1, 2, 4, 8, 30, 100, 1000])
        mode = k % 4
        ids = []
        while len(ids) < n:
            if mode == 0:
                t = rnd.randrange(RAW_V)
            elif mode == 1:
                t = rnd.choice(byte_ids) if rnd.random() < 0.7 else rnd.randrange(RAW_V)
            elif mode == 2:
                t = rnd.choice(byte_ids)
            else:
                ids += rnd.choice(utf8_runs)
                t = rnd.choice(specials) if rnd.random() < 0.2 else rnd.randrange(RAW_V)
            if t not in stop_ids:
                ids.append(t)
        streams.append({"ids": ids, "max_stop": [0, 0, 5, 40][k % 4], "chunks": stream(ids, [0, 0, 5, 40][k % 4])})
    json.dump({"pieces": PIECES, "decode": dec, "stream": streams}, open(f"{OUT}/decode.json", "w"), ensure_ascii=False)
    print(f"decode: {len(dec)} cases, {len(streams)} streams, {len(PIECES)} token pieces; total {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
