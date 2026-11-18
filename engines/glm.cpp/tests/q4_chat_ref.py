"""Reference prompts for tests/q4_chat_check.cpp: HF apply_chat_template over the Qwen3.8-Flash-Next template (with
The client's prefill hook appended, as the client serves it), plus the HF tokenizer's ids. CPU only, no torch.

  cp tests/q4_chat_ref.py ~/b70/llm/glm/run/q4chat/ &&
  GLM53_MODELS=<Swift model dir> ~/b70/llm/glm/scripts/in_rootfs.sh python3 -I /w/run/q4chat/q4_chat_ref.py
"""
import json, sys
sys.modules["torch"] = None
from transformers import PreTrainedTokenizerFast  # noqa: E402

HOOK = """
{#- client hook: an open thinking prefill after the generation prompt, from chat_template_kwargs.nq_prefill -#}
{%- if add_generation_prompt and nq_prefill is defined and nq_prefill -%}{{- nq_prefill -}}{%- endif -%}
"""
tok = PreTrainedTokenizerFast(tokenizer_file="/models/tokenizer.json")
tpl = open("/models/chat_template.jinja").read() + HOOK
tools = [{"type": "function", "function": {"name": "roll_dice", "description": "Roll dice: e.g. 2d6+1.",
          "parameters": {"type": "object", "properties": {"notation": {"type": "string"}, "advantage": {"type": "boolean"},
                                                          "times": {"type": "integer"}}, "required": ["notation"]}}},
         {"type": "function", "function": {"name": "set_scene", "description": "Set the scene. Ünïcode ok.",
          "parameters": {"type": "object", "properties": {"place": {"type": "string"}, "tags": {"type": "array"}}}}}]
sys_m = {"role": "system", "content": "  You are the narrator.\nKeep it vivid.  "}
cases = [
    ("plain", [{"role": "user", "content": "Hello there"}], None, {}),
    ("system_low", [sys_m, {"role": "user", "content": "Start the adventure."}], None, {"reasoning_effort": "low"}),
    ("medium_multi", [sys_m, {"role": "user", "content": "Hi"},
                      {"role": "assistant", "content": " Welcome, traveler. ", "reasoning_content": "  greet them  "},
                      {"role": "user", "content": "I look around."}], None, {"reasoning_effort": "medium"}),
    ("no_think", [{"role": "user", "content": "Quick: 2+2?"}], None, {"enable_thinking": False}),
    ("empty_system", [{"role": "system", "content": ""}, {"role": "user", "content": "x"}], None, {"reasoning_effort": "low"}),
    ("tools", [sys_m, {"role": "user", "content": "I attack the goblin!"},
               {"role": "assistant", "content": "", "reasoning_content": "need a roll",
                "tool_calls": [{"type": "function", "function": {"name": "roll_dice", "arguments": {"notation": "1d20+3", "advantage": True}}},
                               {"type": "function", "function": {"name": "set_scene", "arguments": {"place": "cave", "tags": ["dark", "wet"]}}}]},
               {"role": "tool", "content": "17"}, {"role": "tool", "content": "ok"},
               {"role": "user", "content": "Did I hit?"}], tools, {}),
    ("tools_text_call", [{"role": "user", "content": "roll"},
                         {"role": "assistant", "content": "Rolling now.", "tool_calls": [{"function": {"name": "roll_dice", "arguments": {"notation": "2d6", "times": 3}}}]},
                         {"role": "tool", "content": "<b>7</b>"}], tools, {"preserve_thinking": False}),
    ("nq_prefill", [sys_m, {"role": "user", "content": "Continue."}], None, {"nq_prefill": "The player wants"}),
    ("parts", [{"role": "system", "content": [{"type": "text", "text": "Sys part. "}]},
               {"role": "user", "content": [{"type": "text", "text": "Part one, "}, {"type": "text", "text": "part two."}]}], None, {}),
    ("unicode", [{"role": "user", "content": "Café naïve — “quotes” 日本語 é"}], None, {"reasoning_effort": "xhigh"}),
]
out = []
for name, msgs, tl, kw in cases:
    text = tok.apply_chat_template(msgs, tools=tl, chat_template=tpl, tokenize=False, add_generation_prompt=True, **kw)
    ids = tok.encode(text, add_special_tokens=False)
    out.append({"name": name, "messages": msgs, "tools": tl, "kwargs": kw, "text": text, "ids": ids})
json.dump(out, open("/w/run/q4chat/chat.json", "w"), ensure_ascii=False, indent=1)
print(len(out), "cases")
