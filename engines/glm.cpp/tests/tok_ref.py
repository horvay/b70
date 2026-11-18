"""Reference outputs for tests/tok_check.cpp: the PyTorch stack's tokenizer, streaming detokenizer and prompt
rendering, without torch and without a GPU.

  GLM53_MODELS=<model dir> llm/glm/scripts/in_rootfs.sh python3 -I /w/run/tokcheck/tok_ref.py [out dir]

(copy this file to llm/glm/run/tokcheck/ first; the outputs land in /w/run/tokcheck = llm/glm/run/tokcheck)

- encode: exllamav3 Tokenizer.encode(text, add_bos=False, encode_special_tokens=special). Its wrapper is replicated
  (it imports torch): with no added_tokens_decoder in tokenizer_config.json and no added_tokens.json it reduces to the
  HF tokenizer with encode_special_tokens = not special, add_special_tokens=False (asserted below).
- stream: exllamav3 generator/job.py receive_sample with decode_special_tokens=False, no stop strings / banned
  strings / token healing: per-token pieces from get_id_to_piece_list(False), the joint decode while the held text
  ends in U+FFFD, the max_new_tokens path for the last token.
- prompts: serve.py's own Renderer / normalize_messages / chat() logic, HF apply_chat_template via transformers
  (tokenizer side only; torch is blocked).
"""
import json
import os
import random
import sys
import time

sys.modules["torch"] = None          # importing torch would load the GPU stack: make any attempt fail loudly
sys.argv = [sys.argv[0]] + sys.argv[1:]
OUT = sys.argv[1] if len(sys.argv) > 1 else "/w/run/tokcheck"
MODEL = "/models"
NQ_TEMPLATE = "/w/run/nqfake/tpl.jinja"
os.makedirs(OUT, exist_ok=True)

from tokenizers import Tokenizer  # noqa: E402

tok = Tokenizer.from_file(os.path.join(MODEL, "tokenizer.json"))
tok.no_truncation()
cfg = json.load(open(os.path.join(MODEL, "tokenizer_config.json")))
assert not cfg.get("added_tokens_decoder"), "exllamav3 would see unspecial added tokens: replicate encode_part_base"
assert not os.path.exists(os.path.join(MODEL, "added_tokens.json"))
RAW_V = tok.get_vocab_size()                     # with added tokens, as exllamav3's raw_vocab_size
PAD = EOS = tok.token_to_id(cfg["eos_token"])    # exllamav3: pad = eos = <|endoftext|> for this model


def encode(text, special=True):
    tok.encode_special_tokens = not special
    return tok.encode(text, add_special_tokens=False).ids


# exllamav3 _get_fixed_vocab: single-token decodes when " t" is one token that decodes to " t"
t_ids = encode(" t")
assert len(t_ids) == 1 and len(tok.decode([t_ids[0]])) == 2
PIECES = tok.decode_batch([[i] for i in range(RAW_V)], skip_special_tokens=False)


def ex_decode(ids):
    """exllamav3 Tokenizer.decode(ids, decode_special_tokens=False)"""
    seq = [t for t in ids if t != PAD and t < RAW_V and t != EOS]
    return tok.decode(seq, skip_special_tokens=True)


def stream(ids, max_stop=0):
    """Text emitted after each token; the last token takes the max_new_tokens path."""
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

ADDED = [a["content"] for a in json.load(open(os.path.join(MODEL, "tokenizer.json")))["added_tokens"]]


def synthetic():
    s = [
        "", " ", "a", "Hello world", "Hello, world! How are you?", "  leading spaces", "trailing spaces   ",
        "I'm you're he'll she'd it's we've don't can't", "IT'S I'M YOU'RE HE'LL SHE'D WE'VE", "'s 't 're 've 'm 'll 'd",
        "rock'n'roll o'clock", "it’s don’t", "'ſ 'ſſ 'S'T", "x'sy 'sx ''s '''",
        "1234567890", "3.14159", "1,000,000.00", "12 345 6789", "١٢٣٤", "①②③",
        "½ ¼ ⅓", "Ⅻ Ⅷ", "x² + y³", "a1b22c333d4444", "1a 22b 333c 4444d",
        "a  b", "a   \n\n  b", "\t\t", "  \n", " \r\n ", "\r\n\r\n", "  x", "　x　", "a b",
        "a\u0085b", "a​b", "  ", "     ", "\n", "\n\n\n", "a\nb", "a\r\nb", "a\rb", " \n \n ",
        "x \n\t y", "\t\n", "end \n", "end  \n\n", "word\n\n\n\nword", "  \t  x", "x\t\t\ty",
        "你好，世界！", "こんにちは、世界。カタカナ",
        "안녕하세요", "Привет, мир!",
        "Γειά σου", "مرحبا بالعالم",
        "שלום", "नमस्ते", "สวัสดี",
        "é à̀ ñ", "\U0001F600\U0001F44D\U0001F3FD", "\U0001F468‍\U0001F469‍\U0001F467",
        "\U0001F1FA\U0001F1F8", "❤️", "emoji\U0001F600text", " \U0001F600 ", "\U0001F600\U0001F600\U0001F600",
        "�", "a�b", "\x00\x01\x1f\x7f", "tab\x0bvt\x0cff", "­‎",
        "def f(x):\n    return x * 2\n\nprint(f(21))\n", "{\"a\": [1, 2, {\"b\": null}], \"c\": \"d\\n\"}",
        "<div class=\"x\">&amp; &lt;</div>", "```python\nimport os\n```", "| a | b |\n|---|---|\n| 1 | 2 |",
        "!!!???...", "--->", "<<>>", "\"quoted\"", "'single'", "(parens) [brackets] {braces}", "a.b,c;d:e",
        "C++ C# F# .NET node.js", "http://example.com/a?b=c&d=e#f", "user@example.com", "$100 €50 £30",
        "été naïve café über straße", "İstanbul ıİ", "ẞß",
        "a" * 5000, " " * 3000, "\n" * 1000, "1" * 100, "ab" * 1000, "a " * 1000, "一" * 500, "! " * 300,
        "x" + " " * 50 + "y", " " * 7 + "\n" * 3 + " " * 2,
    ]
    for t in ADDED:
        s += ["x" + t + "y", " " + t + " ", t + t, "\n" + t + "\n", "ab" + t + "cd ef", t, t + " " + t,
              "  " + t + "  x", t + "\n\n", "1" + t + "2", "\U0001F600" + t + "你"]
    s += ["<|user", "<think", "</think>>", "<<think>>", "<think></think>", "[gMASK]<sop><|system|>x",
          "<|user|>hi<|assistant|><think>ok</think>done", "</tool_call><tool_call>", "[MASK][gMASK][sMASK]",
          "<|begin_of_box|><|end_of_box|>", "/nothink /nothinks", "<tool_call>f<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>",
          "<|endoftext|>", "<|code_prefix|>a<|code_middle|>b<|code_suffix|>"]
    return s


def random_strings(n, seed):
    rnd = random.Random(seed)
    palette = (list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") + list(" " * 12) +
               list("\n\n\n\t\r") + list(".,;:!?'\"()[]{}<>-_=+*/\\|@#$%^&`~") +
               [" ", "　", " ", "\u0085", "​", "é", "ß", "ſ", "İ", "’"] +
               list("你好世界こんカ안녕Прمرשनส") +
               ["́", "̀", "‍", "️", "\U0001F600", "\U0001F44D", "\U0001F3FD", "\U0001F1FA", "�"] +
               list("١٢²½①Ⅻ") + ["'s", "'t", "'re", "'ll", "'d", "'m", "'ve", "'S"])
    out = []
    for k in range(n):
        L = rnd.choice([1, 2, 3, 5, 8, 13, 30, 80, 200])
        parts = []
        for _ in range(L):
            if rnd.random() < 0.03:
                parts.append(rnd.choice(ADDED))
            elif rnd.random() < 0.05:
                parts.append(chr(rnd.randrange(0x20, 0x30000)) if rnd.random() < 0.7 else chr(rnd.randrange(0x80, 0x800)))
            else:
                parts.append(rnd.choice(palette))
        s = "".join(parts)
        s = "".join(c for c in s if not 0xD800 <= ord(c) <= 0xDFFF)
        out.append(s)
    return out


def probe(c):
    """Context probe for one code point (the same string tok_check.cpp builds)."""
    return ("x" + c + "x 1" + c + "1 " + c + "x !" + c + "! \n" + c + "\n'" + c + "'" + c + c + " " + c + " \t" + c +
            "\t ")


def fnv(ids):
    h = 0xcbf29ce484222325
    for t in ids:
        for b in t.to_bytes(4, "little"):
            h ^= b
            h = (h * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


def read(p):
    return open(p, encoding="utf-8").read()


# ------------------------------------------------------------------------------------------------ prompts

sys.path.insert(0, "/w/xpu")   # python3 -I ignores PYTHONPATH
import serve  # noqa: E402  (llm/glm/xpu/serve.py: no torch at import)

R_NQ = serve.Renderer(MODEL, NQ_TEMPLATE)
R_MODEL = serve.Renderer(MODEL)
assert R_NQ.hf is not None and R_MODEL.hf is not None, "AutoTokenizer did not load"
from transformers.utils.chat_template_utils import render_jinja_template  # noqa: E402

TEMPLATES = {"nq": read(NQ_TEMPLATE), "model": read(os.path.join(MODEL, "chat_template.jinja"))}
TOOLS = [
    {"type": "function", "function": {"name": "roll", "description": "Roll a fair die: uniform integer in [1, n]. — \"quoted\" \\ slash",
                                      "parameters": {"type": "object", "properties": {"n": {"type": "number", "minimum": 1, "maximum": 1e6, "default": 6.0}},
                                                     "required": ["n"]}, "strict": True}},
    {"type": "function", "function": {"name": "search", "description": "Search memory.\nTwo lines, unicode é你\U0001F600 and tab\t.",
                                      "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "includeSeed": {"type": "boolean", "default": False},
                                                                                      "limit": {"type": "integer", "default": None}, "zeta": {"enum": ["a", "b", 3, -0.5, 1.5e-7, 12345678901234567890]}},
                                                     "required": ["query"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "hidden", "description": "deferred", "defer_loading": True, "parameters": {}}},
    {"name": "bare_tool", "description": "a tool without the function wrapper", "parameters": {"type": "object", "properties": {}}},
]
TC = [{"id": "call_1", "type": "function", "function": {"name": "roll", "arguments": "{\"n\": 20}"}},
      {"id": "call_2", "type": "function", "function": {"name": "search", "arguments": {"query": "the iron ring", "includeSeed": True, "limit": 3, "nested": {"z": [1, 2.5, None, "é"]}}}}]


def conversations():
    sys_m = {"role": "system", "content": "You are the narrator. Stay in character.\n\nRules:\n- be terse"}
    u1 = {"role": "user", "content": "I open the door."}
    a1 = {"role": "assistant", "content": "The door creaks open.", "reasoning_content": "The player opens the door; describe."}
    u2 = {"role": "user", "content": "I step inside. 你好 \U0001F600"}
    c = {}
    c["single_user"] = [u1]
    c["system_user"] = [sys_m, u1]
    c["multi_turn"] = [sys_m, u1, a1, u2]
    c["developer_role"] = [{"role": "developer", "content": "dev instructions"}, u1]
    c["assistant_last"] = [sys_m, u1, a1]
    c["assistant_think_in_content"] = [u1, {"role": "assistant", "content": "<think>hidden plan</think>\n\n  Visible reply.  "}, u2]
    c["assistant_think_after_user"] = [u1, {"role": "assistant", "content": "pre<think>a</think>b</think>  tail  "}]
    c["reasoning_after_last_user"] = [u1, a1, u2, {"role": "assistant", "content": "", "reasoning_content": "thinking now"}]
    c["two_assistants"] = [u1, {"role": "assistant", "content": "one", "reasoning_content": "r1"}, {"role": "assistant", "content": "two"}]
    c["null_content"] = [u1, {"role": "assistant", "content": None, "tool_calls": TC}]
    c["tool_round"] = [sys_m, u1, {"role": "assistant", "content": "Let me roll.", "reasoning_content": "need a roll", "tool_calls": TC},
                       {"role": "tool", "tool_call_id": "call_1", "content": "17"},
                       {"role": "tool", "tool_call_id": "call_2", "content": "{\"hits\": []}"}]
    c["tool_round_reordered"] = [u1, {"role": "assistant", "content": "", "tool_calls": TC},
                                 {"role": "tool", "tool_call_id": "call_2", "content": "second"},
                                 {"role": "tool", "tool_call_id": "call_1", "content": "first"},
                                 {"role": "tool", "tool_call_id": "call_9", "content": "orphan"}]
    c["tool_no_ids"] = [u1, {"role": "assistant", "content": "", "tool_calls": TC},
                        {"role": "tool", "content": "a"}, {"role": "tool", "content": "b"}, u2]
    c["tool_without_assistant"] = [u1, {"role": "tool", "content": "lonely"}]
    c["tool_list_content"] = [u1, {"role": "assistant", "content": "", "tool_calls": TC},
                              {"role": "tool", "tool_call_id": "call_1", "content": [{"type": "text", "text": "part1 "}, {"type": "text", "text": "part2"}]}]
    c["tool_outputs"] = [u1, {"role": "assistant", "content": "", "tool_calls": TC},
                         {"role": "tool", "content": [{"tool_call_id": "call_2", "output": "out2"}, {"id": "call_1", "output": [{"type": "text", "text": "out1"}]}]}]
    c["tool_reference"] = [u1, {"role": "assistant", "content": "", "tool_calls": [TC[0]]},
                           {"role": "tool", "tool_call_id": "call_1", "content": [{"type": "tool_reference", "name": "search"}, {"type": "tool_reference", "name": "roll"}]}]
    c["tool_then_more"] = [u1, {"role": "assistant", "content": "a", "tool_calls": [TC[0]]}, {"role": "tool", "tool_call_id": "call_1", "content": "5"},
                           {"role": "assistant", "content": "You rolled 5.", "reasoning_content": "it was 5"}, u2,
                           {"role": "assistant", "content": "", "tool_calls": [TC[1]]}, {"role": "tool", "tool_call_id": "call_2", "content": "none"}]
    c["list_content"] = [{"role": "user", "content": [{"type": "text", "text": "look: "}, {"type": "image_url", "image_url": {"url": "x"}},
                                                      {"type": "input_audio"}, {"type": "video"}, "plain string item", {"type": "other"}]}]
    c["args_string_bad"] = [u1, {"role": "assistant", "content": "", "tool_calls": [{"id": "c", "type": "function", "function": {"name": "roll", "arguments": "not json {"}}]}]
    c["args_empty_string"] = [u1, {"role": "assistant", "content": "", "tool_calls": [{"id": "c", "function": {"name": "roll", "arguments": ""}}]}]
    c["args_list"] = [u1, {"role": "assistant", "content": "", "tool_calls": [{"id": "c", "function": {"name": "roll", "arguments": "[1, 2]"}}]}]
    c["args_missing"] = [u1, {"role": "assistant", "content": "", "tool_calls": [{"id": "c", "function": {"name": "roll"}}]}]
    c["args_no_wrapper"] = [u1, {"role": "assistant", "content": "", "tool_calls": [{"id": "c", "name": "roll", "arguments": "{\"n\": 6, \"s\": \"x\"}"}]}]
    c["args_types"] = [u1, {"role": "assistant", "content": "", "tool_calls": [{"id": "c", "function": {"name": "f", "arguments":
                        {"s": "text \"q\"", "i": 42, "f": 0.1, "big": 1e300, "small": 1e-7, "neg0": -0.0, "t": True, "n": None,
                         "l": [1, "two", 3.0], "d": {"k": "vé\n"}, "huge": 123456789012345678901234567890, "int_float": 100.0}}}]}]
    c["empty_messages"] = []
    c["whitespace_content"] = [{"role": "user", "content": "  spaced  \n"}, {"role": "assistant", "content": " \n\t "}, {"role": "user", "content": ""}]
    c["unicode_strip"] = [u1, {"role": "assistant", "content": "　  reply  \x1c"}]
    c["unknown_role"] = [{"role": "narrator", "content": "ignored"}, u1]
    c["system_middle"] = [u1, sys_m, u2]
    return c


def serve_cases():
    """(name, body, defaults (effort, clear_thinking), template) for serve.py's chat() path"""
    conv = conversations()
    out = []
    efforts = [None, "none", "minimal", "low", "medium", "high", "xhigh", "max", "MAX ", "bogus", True, 3]
    for i, e in enumerate(efforts):
        b = {"messages": conv["multi_turn"]}
        if e is not None:
            b["reasoning_effort"] = e
        out.append((f"effort_top_{i}", b, ("low", True), "nq"))
        out.append((f"effort_kw_{i}", {"messages": conv["multi_turn"], "chat_template_kwargs": {"reasoning_effort": e}}, ("low", True), "nq"))
    out.append(("effort_kw_wins", {"messages": conv["multi_turn"], "reasoning_effort": "high", "chat_template_kwargs": {"reasoning_effort": "low"}}, ("max", True), "nq"))
    out.append(("effort_kw_null", {"messages": conv["multi_turn"], "reasoning_effort": "high", "chat_template_kwargs": {"reasoning_effort": None}}, ("high", True), "nq"))
    for d in [("low", True), ("high", True), ("max", True), ("low", False), ("max", False)]:
        for name in ["multi_turn", "tool_round", "tool_then_more", "reasoning_after_last_user", "two_assistants"]:
            out.append((f"defaults_{d[0]}_{int(d[1])}_{name}", {"messages": conv[name], "tools": TOOLS}, d, "nq"))
    for ct in [True, False, None, 0, "no"]:
        out.append((f"clear_{ct}", {"messages": conv["tool_then_more"], "tools": TOOLS, "chat_template_kwargs": {"clear_thinking": ct}}, ("low", True), "nq"))
    for et in [True, False, None, 0, "false"]:
        out.append((f"thinking_{et}", {"messages": conv["multi_turn"], "chat_template_kwargs": {"enable_thinking": et}}, ("low", True), "nq"))
    out.append(("thinking_off_tools", {"messages": conv["tool_round"], "tools": TOOLS, "chat_template_kwargs": {"enable_thinking": False}}, ("low", True), "nq"))
    for pf in ["The player", "", None, "</think>Already decided.", 7]:
        for et in [True, False]:
            out.append((f"nq_prefill_{pf}_{et}", {"messages": conv["multi_turn"], "chat_template_kwargs": {"nq_prefill": pf, "enable_thinking": et}}, ("low", True), "nq"))
    out.append(("nq_prefill_model_template", {"messages": conv["multi_turn"], "chat_template_kwargs": {"nq_prefill": "ignored"}}, ("low", True), "model"))
    out.append(("tools_empty", {"messages": conv["multi_turn"], "tools": []}, ("low", True), "nq"))
    out.append(("extra_kwarg", {"messages": conv["multi_turn"], "chat_template_kwargs": {"something_else": 1}}, ("low", True), "nq"))
    for name, msgs in conv.items():
        if not msgs:
            continue
        for tpl in ["nq", "model"]:
            out.append((f"conv_{name}_{tpl}", {"messages": msgs, "tools": TOOLS}, ("low", True), tpl))
            out.append((f"conv_{name}_{tpl}_notools", {"messages": msgs}, ("high", False), tpl))
    req = json.load(open("/w/run/nqfake/req-tiny.json"))
    out.append(("nq_req_tiny", req, ("low", True), "nq"))
    req = json.load(open("/w/run/nqfake/req-mid.json"))
    out.append(("nq_req_mid", req, ("low", True), "nq"))
    return out


def render_serve(body, defaults, tpl):
    serve.DEFAULT_EFFORT, serve.DEFAULT_CLEAR_THINKING = defaults
    r = R_NQ if tpl == "nq" else R_MODEL
    kw = dict(body.get("chat_template_kwargs") or {})
    if body.get("reasoning_effort") is not None:
        kw.setdefault("reasoning_effort", body["reasoning_effort"])
    thinking = kw.get("enable_thinking", True) is not False
    tools = body.get("tools") if body.get("tool_choice") != "none" else None
    prompt = r.render(body["messages"], thinking, tools, kw)
    in_thought = thinking and prompt.rfind("<think>") > prompt.rfind("</think>")
    return prompt, thinking, in_thought


def render_raw(msgs, tools, vars_, tpl):
    out, _ = render_jinja_template(conversations=[msgs], tools=tools, chat_template=TEMPLATES[tpl], **vars_)
    return out[0]


def main():
    t0 = time.time()
    rt = lambda x: json.loads(json.dumps(x))   # what the C++ side parses
    # ---- prompts
    serve_out = []
    for name, body, d, tpl in serve_cases():
        body = rt(body)
        try:
            p, th, it = render_serve(body, d, tpl)
        except Exception as e:  # noqa: BLE001  (the C++ port must fail as well)
            print(f"serve {name}: {type(e).__name__}: {e}")
            p, th, it = None, None, None
        serve_out.append({"name": name, "body": body, "effort": d[0], "clear": d[1], "template": tpl,
                          "text": p, "thinking": th, "in_thought": it, "ids": encode(p) if p is not None else None})
    raw_out = []
    conv = conversations()
    var_sets = [{}, {"add_generation_prompt": True}, {"add_generation_prompt": False, "reasoning_effort": "high"},
                {"add_generation_prompt": True, "reasoning_effort": "low", "clear_thinking": False},
                {"add_generation_prompt": True, "reasoning_effort": "Low", "clear_thinking": True},
                {"add_generation_prompt": True, "clear_thinking": None, "nq_prefill": "prefill text"},
                {"add_generation_prompt": True, "reasoning_effort": None, "clear_thinking": 0, "nq_prefill": 0}]
    for name, msgs in conv.items():
        for vi, v in enumerate(var_sets):
            for tools in [None, TOOLS]:
                for tpl in ["nq", "model"]:
                    if tpl == "model" and "nq_prefill" in v:
                        continue
                    m, t = rt(msgs), rt(tools)
                    try:
                        text = render_raw(m, t, v, tpl)
                    except Exception as e:  # noqa: BLE001
                        text = None
                        print(f"render {name}/{vi}: {type(e).__name__}: {e}")
                    raw_out.append({"name": f"{name}_{vi}_{'tools' if tools else 'notools'}_{tpl}", "messages": m,
                                    "tools": t, "vars": rt(v), "template": tpl, "text": text})
    # template-level dict/number content and similar: rendered by the raw template only (serve normalizes)
    odd = {"content_number": [{"role": "user", "content": 5}, {"role": "assistant", "content": 2.5}],
           "content_dict": [{"role": "user", "content": {"a": 1, "b": [True, None, "x'y"]}}],
           "content_none": [{"role": "user", "content": None}, {"role": "system"}],
           "tool_number_ids": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "", "tool_calls": [{"id": 7, "function": {"name": "roll", "arguments": {}}}]},
                               {"role": "tool", "tool_call_id": 7, "content": "r"}]}
    for name, msgs in odd.items():
        m = rt(msgs)
        try:
            text = render_raw(m, None, {"add_generation_prompt": True}, "nq")
        except Exception as e:  # noqa: BLE001
            print(f"render odd_{name}: {type(e).__name__}: {e}")
            text = None
        raw_out.append({"name": "odd_" + name, "messages": m, "tools": None, "vars": {"add_generation_prompt": True},
                        "template": "nq", "text": text})
    json.dump({"serve": serve_out, "render": raw_out}, open(f"{OUT}/chat.json", "w"), ensure_ascii=False)
    print(f"prompts: {len(serve_out)} serve cases, {len(raw_out)} template cases ({time.time() - t0:.1f} s)")

    # ---- encode
    texts = []
    nq = "/w/run/nqfake"
    for f in sorted(os.listdir(nq)):
        if f.endswith((".txt", ".json", ".py", ".sh", ".jinja")):
            texts.append(("file:" + f, read(os.path.join(nq, f))))
    texts.append(("file:serve.py", read("/w/xpu/serve.py")))
    texts.append(("file:chat_template.jinja", TEMPLATES["model"]))
    for i, s in enumerate(synthetic()):
        texts.append((f"synthetic:{i}", s))
    for i, s in enumerate(random_strings(4000, 1234)):
        texts.append((f"random:{i}", s))
    for c in serve_out:
        if c["text"] is not None:
            texts.append(("prompt:" + c["name"], c["text"]))
    enc = []
    for name, s in texts:
        for special in (True, False):
            enc.append({"name": name, "text": s, "special": special, "ids": encode(s, special)})
    big = read(f"{nq}/mid-prompt.txt")
    while len(encode(big)) < 20000:
        big += "\n\n" + read(f"{nq}/mid-prompt.txt")
    t1 = time.perf_counter()
    big_ids = encode(big)
    t_big = time.perf_counter() - t1
    json.dump({"cases": enc, "big": {"text": big, "ids": big_ids, "py_ms": t_big * 1000}}, open(f"{OUT}/encode.json", "w"),
              ensure_ascii=False)
    print(f"encode: {len(enc)} cases, big prompt {len(big_ids)} tokens in {t_big * 1000:.1f} ms (HF)")

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
    dec = []
    for c in enc[::7]:
        for skip in (False, True):
            dec.append({"ids": c["ids"], "skip": skip, "text": tok.decode(c["ids"], skip_special_tokens=skip)})
    byte_ids = [i for i in range(RAW_V) if "�" in PIECES[i]]
    for k in range(400):
        n = rnd.choice([1, 2, 3, 5, 10, 50, 300])
        pool = byte_ids if k % 2 else range(RAW_V)
        ids = [rnd.choice(pool) if rnd.random() < 0.8 else rnd.randrange(RAW_V) for _ in range(n)]
        for skip in (False, True):
            dec.append({"ids": ids, "skip": skip, "text": tok.decode(ids, skip_special_tokens=skip)})
    streams = []
    stop_ids = {PAD, EOS, tok.token_to_id("<|user|>"), tok.token_to_id("<|observation|>")}
    for c in enc[::5]:
        ids = [t for t in c["ids"] if t not in stop_ids]
        if ids:
            streams.append({"ids": ids, "max_stop": 0, "chunks": stream(ids)})
    for k in range(600):
        n = rnd.choice([1, 2, 4, 8, 30, 100, 1000])
        mode = k % 3
        ids = []
        while len(ids) < n:
            if mode == 0:
                t = rnd.randrange(RAW_V)
            elif mode == 1:
                t = rnd.choice(byte_ids) if rnd.random() < 0.7 else rnd.randrange(RAW_V)
            else:
                t = rnd.choice(byte_ids)
            if t not in stop_ids:
                ids.append(t)
        streams.append({"ids": ids, "max_stop": [0, 0, 5, 40][k % 4], "chunks": stream(ids, [0, 0, 5, 40][k % 4])})
    json.dump({"pieces": PIECES, "decode": dec, "stream": streams}, open(f"{OUT}/decode.json", "w"), ensure_ascii=False)
    print(f"decode: {len(dec)} cases, {len(streams)} streams, {len(PIECES)} token pieces; total {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
