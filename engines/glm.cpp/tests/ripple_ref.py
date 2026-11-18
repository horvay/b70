"""Reference outputs for tests/ripple_check.cpp: HF `tokenizers` and HF's Jinja rendering of Ripple-65B's tokenizer and
chat template, without torch and without a GPU. Nothing else needs installing:

  uv run --no-project --with jinja2 --with tokenizers python -I tests/ripple_ref.py <model dir> <out dir> <repo dir>

  model_dir: tokenizer.json, chat_template.jinja, and the other text files read as long texts (nothing there is executed)
  repo_dir: this repository; its sources and docs are more long texts. Nothing here reads a chat log; the conversations
  below are written out by hand.
  With `--with transformers` added, every render is also compared with transformers' own render_jinja_template.

- encode: HF encode(text, add_special_tokens=False) with encode_special_tokens = not special.
- stream: exllamav3 generator/job.py receive_sample as tests/tok_ref.py replicates it.
- render: render_jinja_template's environment (trim_blocks, lstrip_blocks, tojson = json.dumps ensure_ascii=False) over
  chat_template.jinja plus the client's prefill hook. A template error gives text null: the port must fail as well.
- serve: normalize_messages and the content-list join transcribed in shape() below, then the same template; enable_thinking
  false adds the empty thought, which is this port's own (see chat_ripple.cpp), not the template's.
- split: ripple_format.py's RippleReasoningDetector, transcribed in detector() below (the file is not imported or run) and
  driven with random chunkings. A transcription, so it checks the port against my reading of that code.
- tiny: a small tokenizer whose regex leaves gaps, which Ripple's own never does, for Split(Removed, invert).
"""
import json
import os
import random
import sys
import time

sys.modules["torch"] = None          # importing torch would load the GPU stack: make any attempt fail loudly
MODEL, OUT, REPO = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)

from jinja2.ext import loopcontrols  # noqa: E402
from jinja2.sandbox import ImmutableSandboxedEnvironment  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

try:
    from transformers.utils.chat_template_utils import render_jinja_template
except ImportError:
    render_jinja_template = None

tok = Tokenizer.from_file(os.path.join(MODEL, "tokenizer.json"))
tok.no_truncation()
RAW_V = tok.get_vocab_size()
PAD, EOS = tok.token_to_id("<|pad|>"), tok.token_to_id("<|im_end|>")
tj = json.load(open(os.path.join(MODEL, "tokenizer.json"), encoding="utf-8"))
ADDED = [a["content"] for a in tj["added_tokens"]]
SPECIAL = {a["id"] for a in tj["added_tokens"] if a["special"]}

# ripple_format.py TOKENIZER_PROBES
PROBES = [
    "x = 1234567890 + 3.14159; y=-42",
    "I'M here, you'RE there, it's 4:05pm. DON'T",
    "a  b   c\n\n\n  d\t\te \r\n f",
    "    def f(self):\n        return {'k': [1, 2]}\n",
    "café naïve 日本語テキスト 한국어 🙂👍 Ελληνικά",
    "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\nok\n</think>\nHello<|im_end|>",
]


def encode(text, special=True, t=tok):
    t.encode_special_tokens = not special
    return t.encode(text, add_special_tokens=False).ids


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


def bytes_to_unicode():
    keep = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    table, n = {}, 0
    for b in range(256):
        if b in keep:
            table[b] = chr(b)
        else:
            table[b] = chr(256 + n)
            n += 1
    return table


BYTE_CHAR = bytes_to_unicode()
BYTE_IDS = [tok.token_to_id(BYTE_CHAR[b]) for b in range(256)]
assert all(i is not None for i in BYTE_IDS)


# ------------------------------------------------------------------------------------------------ text corpus

def read(p):
    return open(p, encoding="utf-8").read()


def synthetic():
    s = [
        "", " ", "  ", "a", "Hello world", "Hello, world! How are you?", "  leading spaces", "trailing spaces   ",
        "I'm you're he'll she'd it's we've don't can't", "I'M YOU'RE HE'LL SHE'D IT'S WE'VE DON'T", "'s 'S 't 're 'RE 'Ve 'm 'll 'd 'x '", "'ſ 'ſs K'K",
        "1234567890", "12345", "123 456", "3.14159", "1,000,000.00", "١٢٣٤", "①②③", "½ ¼ ⅓", "a1b2c3", "1a2b", "x²",
        "a  b", "a   \n\n  b", "\t\t", "  \n", " \r\n ", "\r\n\r\n", "  x", "　x　", "a b",
        "a\u0085b", "a​b", "  ", "\n", "\n\n\n", "a\nb", "a\r\nb", "a\rb", " \n \n ", "x \n\t y", "x\x1cy", " \x1c ", "!!\n\n", "!! \n", " !!\r\n\r\n",
        "你好，世界！", "こんにちは、世界。カタカナ", "안녕하세요", "Привет, мир!", "Γειά σου", "مرحبا بالعالم", "שלום", "नमस्ते", "สวัสดี",
        "é à̀ ñ", "\U0001F600\U0001F44D\U0001F3FD", "\U0001F468‍\U0001F469‍\U0001F467", "\U0001F1FA\U0001F1F8", "❤️",
        "emoji\U0001F600text", " \U0001F600 ", "\U0001F600\U0001F600\U0001F600", "\U0001FAE0\U0001FAE1\U0001FAE2",
        "\U00020000\U00020001\U0002A6D6", "𝔘𝔫𝔦𝔠𝔬𝔡𝔢", "\U000F0000\U0010FFFD", "�", "a�b", "\x00\x01\x1f\x7f", "tab\x0bvt\x0cff",
        "­‎", "def f(x):\n    return x * 2\n\nprint(f(21))\n", "{\"a\": [1, 2, {\"b\": null}], \"c\": \"d\\n\"}",
        "<div class=\"x\">&amp; &lt;</div>", "```python\nimport os\n```", "| a | b |\n|---|---|\n| 1 | 2 |", "!!!???...",
        "http://example.com/a?b=c&d=e#f", "user@example.com", "$100 €50 £30", "été naïve café über straße", "İstanbul ıİ", "ẞß",
        "a" * 5000, " " * 3000, "\n" * 1000, "\t" * 300, "1" * 100, "ab" * 1000, "a " * 1000, "一" * 500, "! " * 300,
        "x" + " " * 50 + "y", " " * 7 + "\n" * 3 + " " * 2, "\U0001F600" * 200, "!" * 2000, "a\n" * 500,
        "<think>\nhm\n</think>\nanswer", "<tool_call>\n{\"name\": \"f\", \"arguments\": {\"a\": 1}}\n</tool_call>", "<tool_response>\nx\n</tool_response>", "<tools>\n{}\n</tools>",
        "<|im_start", "<|im_end", "|||", "||PHONE_NUMBER|||", "<|extra_id_", "<|extra_id_99|>", "<|endoftext|><|endoftext|>",
    ]
    for t in ADDED:
        s += ["x" + t + "y", " " + t + " ", t + t, "\n" + t + "\n", "ab" + t + "cd ef", t, t + " " + t, "  " + t + "  x",
              t + "\n\n", "1" + t + "2", "\U0001F600" + t + "你", "'" + t + "s"]
    return s


def random_strings(n, seed):
    rnd = random.Random(seed)
    palette = (list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") + list(" " * 12) +
               list("\n\n\n\t\r") + list(".,;:!?'\"()[]{}<>-_=+*/\\|@#$%^&`~") + list("''''") +
               [" ", "　", " ", "\u0085", "​", "é", "ß", "ſ", "İ", "’", "<think>", "</think>", "<tool_call>", "'s", "'LL"] +
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
    """Context probe for one code point (the same string ripple_check.cpp builds)."""
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


def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


ENV = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
ENV.filters["tojson"] = tojson
COMPILED = {t: ENV.from_string(t) for t in (TEMPLATE, TEMPLATE_MODEL)}
TVARS = {"bos_token": "<|endoftext|>", "eos_token": "<|im_end|>", "pad_token": "<|pad|>", "unk_token": "<|endoftext|>"}


def fn(name, description="d", parameters=None, **extra):
    f = {"name": name, "description": description}
    if parameters is not None:
        f["parameters"] = parameters
    f.update(extra)
    return {"type": "function", "function": f}


TOOLS = [
    fn("add_numbers", "Add two integers.", {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a", "b"], "additionalProperties": False}, strict=True),
    fn("roll", "Roll a fair die: uniform in [1, n]. — \"quoted\" \\ slash \n two lines é你\U0001F600",
       {"type": "object", "properties": {"n": {"type": "number", "description": "faces", "minimum": 1, "maximum": 1e6, "default": 6.0}}, "required": ["n"]}),
    fn("noparams", "no parameters at all"),
    {"type": "function", "function": {"name": "nodesc", "parameters": {"type": "object"}}},
    fn("nulldesc", None),
    {"name": "flat", "description": "not nested in function", "extra": [1, 2.5, None, True, "v", {"K": []}]},
]
TOOLS_SMALL = [TOOLS[0]]
ALL_ARGS = {"s": "text \"q\" : , { } [ ] é你\U0001F600\nline\ttab", "i": 42, "f": 0.1, "t": True, "fa": False, "n": None, "big": 1e300, "small": 1e-7, "neg0": -0.0,
            "l": [1, "two", 3.0, None, True, [], {}], "d": {"B": "vé\n", "a": [{"z": 1, "Y": 2}]}, "huge": 123456789012345678901234567890, "int_float": 100.0, "e": "", "Upper": 1, "lower": 2,
            "ctl": "\x01\x1f\x7f", "close": "</tool_call>"}
TC = [{"id": "call_1", "type": "function", "function": {"name": "add_numbers", "arguments": {"a": 23, "b": 19}}},
      {"id": "call_2", "type": "function", "function": {"name": "search", "arguments": ALL_ARGS}}]


def conversations():
    sys_m = {"role": "system", "content": "  You are the narrator. Stay in character.\n\nRules:\n- be terse  \n"}
    u1 = {"role": "user", "content": "I open the door."}
    a1 = {"role": "assistant", "content": "The door creaks open.", "reasoning_content": "The player opens the door; describe."}
    u2 = {"role": "user", "content": "  I step inside. 你好 \U0001F600\n"}
    tool1 = {"role": "tool", "tool_call_id": "call_1", "content": "42"}
    tool2 = {"role": "tool", "tool_call_id": "call_2", "content": "{\"hits\": []}"}
    call1 = {"role": "assistant", "content": "", "tool_calls": [TC[0]]}
    call2 = {"role": "assistant", "content": "Let me look.", "reasoning_content": "need both", "tool_calls": TC}
    c = {}
    c["single_user"] = [u1]
    c["system_user"] = [sys_m, u1]
    c["system_only"] = [sys_m]
    c["system_empty"] = [{"role": "system", "content": ""}, u1]
    c["system_none"] = [{"role": "system", "content": None}, u1]
    c["system_nokey"] = [{"role": "system"}, u1]
    c["system_number"] = [{"role": "system", "content": 5}, u1]
    c["system_list"] = [{"role": "system", "content": [{"type": "text", "text": "x"}]}, u1]
    c["system_later"] = [u1, {"role": "system", "content": " later sys "}, u2]
    c["system_twice"] = [sys_m, {"role": "system", "content": "second"}, u1]
    c["developer_first"] = [{"role": "developer", "content": "dev instructions"}, u1]
    c["multi_turn"] = [sys_m, u1, a1, u2]
    c["assistant_last"] = [sys_m, u1, a1]
    c["reasoning_only"] = [u1, {"role": "assistant", "content": "", "reasoning_content": "only thought"}]
    c["reasoning_none"] = [u1, {"role": "assistant", "content": "x", "reasoning_content": None}, u2]
    c["reasoning_empty"] = [u1, {"role": "assistant", "content": "x", "reasoning_content": ""}, u2]
    c["reasoning_number"] = [u1, {"role": "assistant", "content": "x", "reasoning_content": 5}]
    c["reasoning_zero"] = [u1, {"role": "assistant", "content": "x", "reasoning_content": 0}]
    c["reasoning_list"] = [u1, {"role": "assistant", "content": "x", "reasoning_content": ["r"]}]
    c["reasoning_multiline"] = [u1, {"role": "assistant", "content": "x\n\ny", "reasoning_content": "line one\n\n  line two\n"}, u2]
    c["reasoning_on_user"] = [{"role": "user", "content": "u", "reasoning_content": "ignored"}]
    c["content_has_think"] = [u1, {"role": "assistant", "content": "<think>\nmine\n</think>\nkept", "reasoning_content": "dropped"}, u2]
    c["content_has_think_spaced"] = [u1, {"role": "assistant", "content": " \n\t<think>kept", "reasoning_content": "dropped"}]
    c["content_has_think_unicode_space"] = [u1, {"role": "assistant", "content": "　\x1c<think>kept", "reasoning_content": "dropped"}]
    c["content_think_later"] = [u1, {"role": "assistant", "content": "a <think>", "reasoning_content": "added"}]
    c["user_keeps_markers"] = [{"role": "user", "content": " <think>\nnot stripped</think> "}]
    c["two_assistants"] = [u1, {"role": "assistant", "content": "one", "reasoning_content": "r1"}, {"role": "assistant", "content": "two"}]
    c["empty_contents"] = [{"role": "user", "content": ""}, {"role": "assistant", "content": ""}, {"role": "user", "content": ""}]
    c["whitespace"] = [{"role": "user", "content": "  spaced  \n"}, {"role": "assistant", "content": " \n\t "}, {"role": "user", "content": "\n"}]
    c["content_none"] = [{"role": "user", "content": None}, {"role": "assistant", "content": None}]
    c["content_nokey"] = [{"role": "user"}, {"role": "assistant"}]
    c["content_zero"] = [{"role": "user", "content": 0}, {"role": "assistant", "content": False}, {"role": "user", "content": []}, {"role": "assistant", "content": {}}]
    c["content_number"] = [{"role": "user", "content": 5}]
    c["content_true"] = [{"role": "user", "content": True}]
    c["content_list"] = [{"role": "user", "content": [{"type": "text", "text": "x"}]}]
    c["content_dict"] = [{"role": "user", "content": {"a": 1}}]
    c["assistant_number_reasoning"] = [u1, {"role": "assistant", "content": 5, "reasoning_content": "r"}]
    c["unicode"] = [{"role": "system", "content": "Système 日本語 \U0001F600"}, {"role": "user", "content": "مرحبا　שלום \U0001F44D\U0001F3FD é à̀"}, {"role": "assistant", "content": "한국어 Ελληνικά", "reasoning_content": "ß ſ İ \U00020000"}, u2]
    c["unknown_role"] = [{"role": "narrator", "content": "told"}, u1]
    c["unknown_role_first_tools"] = [{"role": "developer", "content": "told"}, u1, {"role": "narrator", "content": "x"}]
    c["no_role"] = [{"content": "no role"}]
    c["role_number"] = [{"role": 5, "content": "x"}]
    c["not_a_mapping"] = ["bare", u1]
    c["none_message"] = [None, u1]
    c["empty"] = []
    # tool calls
    c["call_only"] = [u1, call1]
    c["call_then_result"] = [sys_m, u1, call1, tool1]
    c["call_results_several"] = [sys_m, u1, call2, tool1, tool2]
    c["call_results_three_then_user"] = [u1, call2, tool1, tool2, {"role": "tool", "content": "third"}, u2]
    c["call_results_then_assistant"] = [u1, call1, tool1, {"role": "assistant", "content": "It is 42.", "reasoning_content": "the sum"}]
    c["call_loop"] = [u1, call1, tool1, call2, tool1, tool2, a1, u2, call1, tool1]
    c["call_with_text_no_reasoning"] = [u1, {"role": "assistant", "content": "Let me add.", "tool_calls": [TC[0]]}, tool1]
    c["call_reasoning_no_text"] = [u1, {"role": "assistant", "content": "", "reasoning_content": "add", "tool_calls": [TC[0]]}, tool1]
    c["call_content_none"] = [u1, {"role": "assistant", "content": None, "tool_calls": [TC[0]]}, tool1]
    c["call_content_nokey"] = [u1, {"role": "assistant", "tool_calls": [TC[0]]}]
    c["call_content_has_think"] = [u1, {"role": "assistant", "content": "<think>x</think>", "reasoning_content": "dropped", "tool_calls": [TC[0]]}]
    c["call_content_number"] = [u1, {"role": "assistant", "content": 5, "tool_calls": [TC[0]]}]
    c["call_flat"] = [u1, {"role": "assistant", "content": "", "tool_calls": [{"name": "flat", "arguments": {"x": 1}}]}]
    c["call_args_string"] = [u1, {"role": "assistant", "content": "", "tool_calls": [{"id": "c", "function": {"name": "roll", "arguments": "{\"n\": 6}"}}]}]
    c["call_args_none"] = [u1, {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "roll", "arguments": None}}]}]
    c["call_args_missing"] = [u1, {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "roll"}}]}]
    c["call_args_list"] = [u1, {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "roll", "arguments": [1, "x"]}}]}]
    c["call_args_unicode"] = [u1, {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "énçødé", "arguments": {"ключ": "значение \U0001F600"}}}]}]
    c["call_name_number"] = [u1, {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": 5, "arguments": {}}}]}]
    c["call_name_missing"] = [u1, {"role": "assistant", "content": "", "tool_calls": [{"function": {"arguments": {}}}]}]
    c["call_function_none"] = [u1, {"role": "assistant", "content": "", "tool_calls": [{"function": None, "name": "x", "arguments": {}}]}]
    c["call_function_string"] = [u1, {"role": "assistant", "content": "", "tool_calls": [{"function": "roll"}]}]
    c["call_item_string"] = [u1, {"role": "assistant", "content": "", "tool_calls": ["roll"]}]
    c["call_item_none"] = [u1, {"role": "assistant", "content": "", "tool_calls": [None]}]
    c["tool_calls_string"] = [u1, {"role": "assistant", "content": "", "tool_calls": "roll"}]
    c["tool_calls_dict"] = [u1, {"role": "assistant", "content": "", "tool_calls": {"a": 1}}]
    c["tool_calls_number"] = [u1, {"role": "assistant", "content": "", "tool_calls": 5}]
    c["tool_calls_empty"] = [u1, {"role": "assistant", "content": "plain", "tool_calls": []}, u2]
    c["tool_calls_none"] = [u1, {"role": "assistant", "content": "plain", "tool_calls": None}, u2]
    c["tool_first"] = [{"role": "tool", "content": "lonely"}, u1]
    c["tool_only"] = [{"role": "tool", "content": "lonely"}]
    c["tool_orphan"] = [u1, {"role": "tool", "content": "lonely"}]
    c["tool_after_system"] = [sys_m, {"role": "tool", "content": "a"}, {"role": "tool", "content": "b"}]
    c["tool_empty"] = [u1, call1, {"role": "tool", "content": ""}]
    c["tool_none"] = [u1, call1, {"role": "tool", "content": None}]
    c["tool_nokey"] = [u1, call1, {"role": "tool"}]
    c["tool_number"] = [u1, call1, {"role": "tool", "content": 5}]
    c["tool_list"] = [u1, call1, {"role": "tool", "content": [{"type": "text", "text": "x"}]}]
    c["tool_unicode_multiline"] = [u1, call1, {"role": "tool", "content": "line é\n\n 你好 \U0001F600\n"}]
    c["tool_then_bare_next"] = [u1, call1, tool1, "bare"]
    return c


VAR_SETS = [{}, {"add_generation_prompt": True}, {"add_generation_prompt": True, "clear_thinking": True}, {"add_generation_prompt": True, "clear_thinking": False},
            {"add_generation_prompt": True, "clear_thinking": 1}, {"add_generation_prompt": True, "clear_thinking": 0}, {"add_generation_prompt": True, "clear_thinking": None},
            {"add_generation_prompt": True, "clear_thinking": "no"}, {"add_generation_prompt": True, "clear_thinking": ""},
            {"add_generation_prompt": True, "nq_prefill": "The player wants"}, {"add_generation_prompt": True, "nq_prefill": "<think>\nA prefill"},
            {"add_generation_prompt": True, "nq_prefill": "<think>\nall of it\n</think>\n"}, {"add_generation_prompt": True, "nq_prefill": 7}, {"add_generation_prompt": True, "nq_prefill": True},
            {"add_generation_prompt": True, "nq_prefill": ""}, {"add_generation_prompt": False, "nq_prefill": "x"}, {"add_generation_prompt": 1}, {"add_generation_prompt": None},
            {"add_generation_prompt": True, "enable_thinking": False, "reasoning_effort": "low", "something_else": [1, 2]}]


def rt(x):
    return json.loads(json.dumps(x))       # what the C++ side parses


def render(msgs, tools, vars_, template=TEMPLATE):
    hf = None
    if render_jinja_template is not None:
        try:
            hf = render_jinja_template(conversations=[msgs], tools=tools, chat_template=template, **TVARS, **vars_)[0][0]
        except Exception:  # noqa: BLE001
            pass
    try:
        out = "".join(COMPILED[template].generate(messages=msgs, tools=tools, documents=None, **{"add_generation_prompt": False, **TVARS, **vars_}))
    except Exception:
        assert hf is None, "transformers renders what this environment refuses"
        raise
    assert render_jinja_template is None or hf == out, "transformers renders differently"
    return out


def shape(messages):
    """normalize_messages (chat.cpp, serve.py) and the join of a content list's text parts (chat_ripple.cpp)."""
    out = []
    for m in json.loads(json.dumps(messages)):
        if m.get("content") is None:
            m["content"] = ""
        if m.get("role") == "developer":
            m["role"] = "system"
        if m.get("tool_calls"):
            for tc in m["tool_calls"]:
                f = tc.get("function", tc)
                a = f.get("arguments")
                if isinstance(a, str):
                    try:
                        f["arguments"] = json.loads(a or "{}")
                    except ValueError:
                        f["arguments"] = {"arguments": a}
                if not isinstance(f.get("arguments"), dict):
                    f["arguments"] = {}
        if isinstance(m["content"], list):
            m["content"] = " ".join(p["text"] for p in m["content"] if isinstance(p, dict) and p.get("type") in ("text", "input_text"))
        out.append(m)
    return out


SERVE_KW = [None, {}, {"clear_thinking": True}, {"clear_thinking": False}, {"enable_thinking": True}, {"enable_thinking": False}, {"enable_thinking": None}, {"enable_thinking": 0},
            {"nq_prefill": "<think>\nThe player wants"}, {"nq_prefill": "<think>\nall\n</think>\n"}, {"nq_prefill": "plain words"}, {"nq_prefill": "</think><think>again"},
            {"nq_prefill": 7}, {"nq_prefill": ""}, {"enable_thinking": False, "nq_prefill": "<think>\nx"}, {"reasoning_effort": "high"}]


def serve_cases():
    conv = conversations()
    out = []
    for name in ["single_user", "system_user", "developer_first", "multi_turn", "assistant_last", "content_has_think", "content_none", "call_then_result", "call_results_several",
                 "call_results_three_then_user", "call_loop", "call_args_string", "call_content_none", "tool_none", "unicode", "empty_contents"]:
        for ki, kw in enumerate(SERVE_KW):
            for tools in [None, TOOLS_SMALL, TOOLS]:
                out.append((f"{name}/{ki}/{'tools' if tools else 'notools'}", {"messages": conv[name], "tools": tools, "chat_template_kwargs": kw}))
    u1 = conv["single_user"][0]
    out.append(("effort_ignored", {"messages": conv["multi_turn"], "reasoning_effort": "high"}))
    out.append(("effort_number", {"messages": conv["multi_turn"], "reasoning_effort": 3}))
    out.append(("args_string_bad", {"messages": [u1, {"role": "assistant", "content": "", "tool_calls": [{"id": "c", "function": {"name": "roll", "arguments": "not json {"}}]}]}))
    out.append(("args_string_list", {"messages": [u1, {"role": "assistant", "content": "", "tool_calls": [{"id": "c", "function": {"name": "roll", "arguments": "[1, 2]"}}]}]}))
    out.append(("args_string_empty", {"messages": [u1, {"role": "assistant", "content": "", "tool_calls": [{"id": "c", "function": {"name": "roll", "arguments": ""}}]}]}))
    out.append(("args_flat", {"messages": [u1, {"role": "assistant", "content": "", "tool_calls": [{"name": "roll", "arguments": "{\"n\": 6, \"s\": \"é\"}"}]}]}))
    out.append(("parts_joined", {"messages": [{"role": "user", "content": [{"type": "text", "text": "one"}, {"type": "input_text", "text": "two"}, {"type": "image_url", "image_url": {"url": "x"}}, "bare", {"type": "text", "text": "three"}]}]}))
    out.append(("parts_tool", {"messages": [u1, {"role": "assistant", "content": "", "tool_calls": [TC[0]]}, {"role": "tool", "tool_call_id": "call_1", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}]}))
    out.append(("parts_empty", {"messages": [{"role": "user", "content": []}, {"role": "assistant", "content": [{"type": "text", "text": ""}]}]}))
    out.append(("parts_assistant_reasoning", {"messages": [u1, {"role": "assistant", "content": [{"type": "text", "text": "x"}], "reasoning_content": "r"}]}))
    return out


def detector(chunks, force):
    """RippleReasoningDetector, streaming, then finish(); force: this port's start inside the thought after a prefill."""
    START, END = "<think>", "</think>"

    def drop(text, lead):
        if lead:
            return text[1:] if text.startswith("\n") else text
        return text[:-1] if text.endswith("\n") else text

    def partial(text, token):
        for i in range(min(len(text), len(token) - 1), 0, -1):
            if token.startswith(text[-i:]):
                return i
        return 0

    state, buffer, reasoning, normal = ("think" if force else "start"), "", "", ""
    for new_text in chunks:
        buffer += new_text
        while buffer:
            buf = buffer
            if state == "start":
                if buf.startswith(START):
                    buffer, state = buf[len(START):], "lead"
                elif START.startswith(buf):
                    break
                else:
                    state = "content"
            elif state == "lead":
                buffer, state = drop(buf, True), "think"
            elif state == "think":
                i = buf.find(END)
                if i >= 0:
                    reasoning += drop(buf[:i], False)
                    buffer, state = buf[i + len(END):], "after"
                    continue
                hold = max(partial(buf, "\n" + END), partial(buf, END))
                reasoning += buf[:len(buf) - hold]
                buffer = buf[len(buf) - hold:]
                break
            elif state == "after":
                buffer, state = drop(buf, True), "content"
            else:
                normal += buf
                buffer = ""
    if state in ("lead", "think"):
        reasoning += buffer
    else:
        normal += buffer
    return reasoning, normal


def split_cases(n, seed):
    rnd = random.Random(seed)
    palette = ["<think>", "</think>", "\n", "\n", "\n\n", "\n</think>", "<think>\n", "</think>\n", "<", "</", "</th", "</think", "<thi", "<think", ">", "abc", " ", "é", "你", "\U0001F600", "x\ny", "<tool_call>", "\r\n"]
    out = []
    for k in range(n):
        reply = "".join(rnd.choice(palette) for _ in range(rnd.choice([1, 2, 3, 4, 6, 10, 20])))
        if k % 3 == 0:
            reply = "<think>\n" + reply
        cuts = sorted(rnd.sample(range(len(reply) + 1), min(rnd.choice([0, 1, 2, 4, 8]), len(reply) + 1)))
        chunks = [reply[a:b] for a, b in zip([0] + cuts, cuts + [len(reply)])]
        force = k % 5 == 4
        r, c = detector(chunks, force)
        assert (r, c) == detector([reply], force) == detector(list(reply), force), reply
        out.append({"chunks": chunks, "force": force, "reasoning": r, "content": c})
    return out


def tiny_cases():
    """Split(Removed, invert) with a regex that leaves gaps, and the Split shapes the port must refuse."""
    def write(name, behavior, invert):
        vocab = {BYTE_CHAR[b]: b for b in range(256)}
        vocab["ab"] = 256
        vocab["abc"] = 257
        vocab["<x>"] = 258
        tj = {"version": "1.0", "truncation": None, "padding": None,
              "added_tokens": [{"id": 258, "content": "<x>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True}],
              "normalizer": None,
              "pre_tokenizer": {"type": "Sequence", "pretokenizers": [{"type": "Split", "pattern": {"Regex": "[a-z]+|\\p{N}"}, "behavior": behavior, "invert": invert},
                                                                       {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False}]},
              "post_processor": None, "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
              "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": "", "end_of_word_suffix": "", "fuse_unk": False, "byte_fallback": False,
                        "ignore_merges": False, "vocab": vocab, "merges": [["a", "b"], ["ab", "c"]]}}
        d = os.path.join(OUT, name)
        os.makedirs(d, exist_ok=True)
        json.dump(tj, open(os.path.join(d, "tokenizer.json"), "w", encoding="utf-8"), ensure_ascii=False)
        return d
    d = write("tiny", "Removed", True)
    write("tiny_removed", "Removed", False)
    write("tiny_isolated_invert", "Isolated", True)
    t = Tokenizer.from_file(os.path.join(d, "tokenizer.json"))
    texts = ["abc", "abc abc", "  abc  ", "!!!", "", " ", "ab!c", "x1y2 z", "AB ab", "é abc 你 9", "<x>", " <x> ", "ab<x>c", "!<x>!", "١ ٢", "\n\n", "abc\n", "\U0001F600abc\U0001F600"]
    return [{"text": s, "ids": encode(s, True, t)} for s in texts]


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
                except AssertionError:
                    raise
                except Exception as e:  # noqa: BLE001  (the port must fail as well)
                    text = None
                    n_err += 1
                    if vi == 1 and tname == "none":
                        print(f"render {name}: {type(e).__name__}: {str(e)[:100]}")
                render_out.append({"name": f"{name}/{vi}/{tname}", "messages": m, "tools": tl, "vars": rt(v), "text": text})
    # the unmodified template renders the same when nq_prefill is not set
    for name, msgs in conv.items():
        for v in VAR_SETS[:9]:
            m = rt(msgs)
            try:
                a = render(m, rt(TOOLS), v)
            except AssertionError:
                raise
            except Exception:  # noqa: BLE001
                continue
            assert a == render(m, rt(TOOLS), v, TEMPLATE_MODEL), name
    serve_out = []
    for name, body in serve_cases():
        body = {k: v for k, v in rt(body).items() if v is not None}
        kw = dict(body.get("chat_template_kwargs") or {})
        off = kw.get("enable_thinking") is False
        prefill = kw.get("nq_prefill")
        kw.pop("nq_prefill", None)
        kw["add_generation_prompt"] = True
        try:
            text = render(shape(body["messages"]), body.get("tools"), kw)
            if off:
                text += "<think>\n\n</think>\n"
            prefill_text = str(prefill) if prefill else ""
            text += prefill_text
        except AssertionError:
            raise
        except Exception as e:  # noqa: BLE001
            print(f"serve {name}: {type(e).__name__}: {str(e)[:100]}")
            text, prefill_text = None, ""
        o, c = prefill_text.rfind("<think>"), prefill_text.rfind("</think>")
        serve_out.append({"name": name, "body": body, "text": text, "thinking": not off, "reply_in_thought": o >= 0 and (c < 0 or o > c),
                          "ids": encode(text) if text is not None else None})
    json.dump({"render": render_out, "serve": serve_out}, open(f"{OUT}/chat.json", "w"), ensure_ascii=False)
    print(f"prompts: {len(render_out)} template cases ({n_err} template errors), {len(serve_out)} serve cases ({time.time() - t0:.1f} s)"
          f"{', all also through transformers' if render_jinja_template else ''}")
    json.dump({"cases": split_cases(6000, 77)}, open(f"{OUT}/split.json", "w"), ensure_ascii=False)
    json.dump({"cases": tiny_cases()}, open(f"{OUT}/tiny.json", "w"), ensure_ascii=False)

    # ---- encode
    texts = []
    long_texts = []
    for f in ["README.md", "chat_template.jinja", "tokenizer_config.json", "config.json", "inference/server/README.md", "inference/server/ripple_format.py", "inference/server/model.py", "modeling_ricursive.py"]:
        if os.path.exists(os.path.join(MODEL, f)):
            long_texts.append(("model:" + f, read(os.path.join(MODEL, f))))
    for f in ["README.md", "src/chat_gemma.cpp", "src/tokenizer.cpp", "src/json.h", "tests/gemma_ref.py", "Makefile", "build.sh"]:
        long_texts.append(("repo:" + f, read(os.path.join(REPO, f))))
    docs = os.path.join(REPO, "docs")
    for f in sorted(os.listdir(docs)):
        if f.endswith(".md"):
            long_texts.append(("doc:" + f, read(os.path.join(docs, f))))
    for name, s in long_texts:
        for k in range(0, len(s), 16384):
            texts.append((f"{name}:{k}", s[k:k + 16384], [True]))
    for i, s in enumerate(PROBES):
        texts.append((f"tokenizer_probe:{i}", s, [True, False]))
    for i, s in enumerate(synthetic()):
        texts.append((f"synthetic:{i}", s, [True, False]))
    for i, s in enumerate(random_strings(4000, 1234)):
        texts.append((f"random:{i}", s, [True, False]))
    for i, c in enumerate(serve_out):
        if c["text"] is not None and i % 8 == 0:
            texts.append(("prompt:" + c["name"], c["text"], [True, False]))
    enc = []
    n_tok = 0
    for name, s, specials in texts:
        for special in specials:
            ids = encode(s, special)
            n_tok += len(ids)
            enc.append({"name": name, "text": s, "special": special, "ids": ids})
    # the long mixed code / prose text, with the probes and special tokens in it, as one stretch per added token
    big = "\n\n".join(s for _, s in long_texts)[:150000] + "\n".join(PROBES) + "".join(ADDED) + "\n".join(synthetic()[:90])
    t1 = time.perf_counter()
    big_ids = encode(big)
    t_big = time.perf_counter() - t1
    json.dump({"cases": enc, "big": {"text": big, "ids": big_ids, "py_ms": t_big * 1000}, "added": [{"content": a["content"], "id": a["id"], "special": a["special"]} for a in tj["added_tokens"]]},
              open(f"{OUT}/encode.json", "w"), ensure_ascii=False)
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
    specials = sorted(SPECIAL)
    dec = []
    for c in enc[::7]:
        for skip in (False, True):
            dec.append({"ids": c["ids"], "skip": skip, "text": tok.decode(c["ids"], skip_special_tokens=skip)})
    utf8_runs = [[BYTE_IDS[b] for b in ch.encode("utf-8")] for ch in "é你\U0001F600\U00020000"]
    for k in range(4000):
        n = rnd.choice([1, 2, 3, 5, 10, 50, 300])
        pool = [BYTE_IDS, range(RAW_V), BYTE_IDS + specials][k % 3]
        ids = []
        while len(ids) < n:
            if rnd.random() < 0.2:
                ids += rnd.choice(utf8_runs)
            else:
                ids.append(rnd.choice(pool))
        for skip in (False, True):
            dec.append({"ids": ids, "skip": skip, "text": tok.decode(ids, skip_special_tokens=skip)})
    streams = []
    stop_ids = {PAD, EOS, tok.token_to_id("<|endoftext|>")}
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
                t = rnd.choice(BYTE_IDS) if rnd.random() < 0.7 else rnd.randrange(RAW_V)
            elif mode == 2:
                t = rnd.choice(BYTE_IDS)
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
