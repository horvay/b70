"""A Game Master-shaped conversation against a running xpu/serve.py: one long first prompt, then follow-up turns that
each add a fresh passage (the server reuses the cached prefix and prefills only the new text). Prints each request's
prompt / cached tokens, time to first token and decode speed, plus a summary line to compare modes by.
  python3 tests/t_turns.py <port> <label>     (host python; the text is tests/corpus.txt prose from offset 30000)
  BASE_WORDS (4500), TURN_WORDS (900), TURNS (3), NEW (120)"""
import json
import os
import sys
import time
import urllib.request

PORT, LABEL = sys.argv[1], sys.argv[2]
BASE = int(os.environ.get("BASE_WORDS", "4500"))
TURN = int(os.environ.get("TURN_WORDS", "900"))
TURNS = int(os.environ.get("TURNS", "3"))
NEW = int(os.environ.get("NEW", "120"))
words = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus.txt")).read().split()
off = 30000                     # prose (offset 60000 makes the model loop on placeholders)


def take(n):
    global off
    off += n
    return " ".join(words[off - n:off])


def ask(messages):
    body = {"model": "dsv", "messages": messages, "max_tokens": NEW, "temperature": 0, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.load(r)
    return out, time.time() - t0


msgs = [{"role": "system", "content": "You are the narrator of a long story. The story so far:\n\n" + take(BASE)},
        {"role": "user", "content": "Continue the story in one short paragraph."}]
rows = []
for t in range(TURNS + 1):
    out, wall = ask(msgs)
    tm, us = out.get("timings") or {}, out.get("usage") or {}
    text = out["choices"][0]["message"].get("content") or ""
    rows.append((us.get("prompt_tokens"), tm.get("cached_tokens"), tm.get("prompt_s"), tm.get("decode_tok_s")))
    print(f"TURN {t}: {rows[-1][0]} prompt tokens ({rows[-1][1]} cached), first token {rows[-1][2]} s, "
          f"decode {rows[-1][3]} tok/s, wall {wall:.1f} s | {text[:80]!r}", flush=True)
    msgs += [{"role": "assistant", "content": text},
             {"role": "user", "content": "Here is what happens next:\n\n" + take(TURN) +
              "\n\nContinue the story in one short paragraph."}]
follow = rows[1:]
print(f"SUMMARY {LABEL}: first prompt {rows[0][2]} s; follow-up first token "
      f"{', '.join(str(r[2]) for r in follow)} s (mean {sum(r[2] for r in follow) / len(follow):.1f}); "
      f"decode {', '.join(str(r[3]) for r in follow)} tok/s", flush=True)
