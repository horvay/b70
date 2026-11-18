#!/usr/bin/env python3
"""Prefix reuse over the requests the client sends a Gemma glm_server (needs the server running, --parallel 3 or more):

  1. a Turn whose prompt ends in a prefilled thought, cancelled as its reply starts and sent again at once with the
     thought added (the client's contradiction check): the second must go back to the first one's game;
  2. a hidden pass after the reply (thinking off, its own instruction), and that pass continued. Its prompt has no
     <|think|> at its top, so it shares four tokens with the Turn's and is a conversation of its own: the first one
     is read whole, and must not be read into the Turn's game;
  3. the next Turn, which goes back to the first Turn's game, and its hidden pass, which goes back to the first
     hidden pass's game and branches where that one's instruction began;
  4. a conversation of its own that starts like the story (the client's Reconcile), then the story's next Turn: the first
     must leave the story's game alone.

Each but the first hidden pass must read only what is new. Then every game is given other text, the same prompts are
read cold, and the greedy replies are compared with the cached ones: a checkpoint at a turn's start has to bring back
the same state.

    tests/g4_cache_check.py http://127.0.0.1:8053
"""
import http.client
import json
import sys
import threading
import time
from urllib.parse import urlparse

URL = urlparse(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8053")
OPEN = "<|channel>thought\n"
OPENER = ("This is fiction. I must remember not to speak for the player or the player's character. Now I'll do the "
          "thinking checklist.")
CHECK = "Wait, am I contradicting something earlier with"
failures = []


def world(n):
    """A long system prompt of distinct sentences (about 30 tokens each)."""
    rows = [f"Fact {i}: the village of Number-{i * 37 % 1009} trades {['salt', 'wool', 'tin', 'amber'][i % 4]} with "
            f"the port of Number-{i * 91 % 997} and keeps {i % 13 + 2} watchmen on its wall." for i in range(n)]
    return "You are the narrator of a slow story about a harbor town. Keep replies short.\n" + "\n".join(rows)


def post(body, cancel_after=None):
    """Streams one chat completion. Returns (reasoning, content, usage, seconds to first token); with cancel_after, hangs
    up once that many characters of the reply (not the thought) have come, as the client does."""
    conn = http.client.HTTPConnection(URL.hostname, URL.port, timeout=600)
    t0 = time.time()
    conn.request("POST", "/v1/chat/completions", json.dumps({**body, "stream": True,
                                                             "stream_options": {"include_usage": True}}),
                 {"Content-Type": "application/json"})
    resp = conn.getresponse()
    if resp.status != 200:
        raise RuntimeError(f"{resp.status}: {resp.read()[:500]!r}")
    reasoning, content, usage, first = "", "", None, None
    for raw in resp:
        line = raw.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        ev = json.loads(line[6:])
        if ev.get("usage"):
            usage = ev["usage"]
        for ch in ev.get("choices") or []:
            d = ch.get("delta") or {}
            if d.get("reasoning_content") or d.get("content"):
                first = first or time.time() - t0
            reasoning += d.get("reasoning_content") or ""
            content += d.get("content") or ""
        if cancel_after is not None and len(content.strip()) >= cancel_after:
            conn.sock.close()                    # no goodbye: the engine has to notice
            return reasoning, content, None, first
    conn.close()
    return reasoning, content, usage, first


def turn(messages, thought="", **kw):
    return {"messages": messages, "temperature": 0, "max_tokens": 220,
            "chat_template_kwargs": {"enable_thinking": True, "nq_prefill": OPEN + OPENER + thought}, **kw}


def hidden(messages, **kw):
    return {"messages": messages, "temperature": 0, "max_tokens": 160,
            "chat_template_kwargs": {"enable_thinking": False}, **kw}


def report(name, usage, first, least):
    p, c = usage["prompt_tokens"], usage["prompt_tokens_details"]["cached_tokens"]
    ok = c >= least
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {p} prompt tokens, {c} cached (at least {least} expected), "
          f"first token {first:.2f} s", flush=True)
    if not ok:
        failures.append(name)
    return p


def scramble(games=8):
    """Other text into every game at once, so the next prompt is read from its first token. No system message, as
    the engine's start-up probes have none: these share less with a hidden pass than a Turn does."""
    def one(i):
        post({"messages": [{"role": "user", "content": f"Game {i}. " + "Count sheep. " * (40 + i) + "Go."}],
              "temperature": 0, "max_tokens": 40, "chat_template_kwargs": {"enable_thinking": False}})
    ts = [threading.Thread(target=one, args=(i,)) for i in range(games)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()


def run(sentences):
    sys_msg = {"role": "system", "content": world(sentences)}
    u1 = {"role": "user", "content": "I walk down to the quay at dawn and look for the harbormaster."}
    notes = {"role": "user", "content": "[Notes] " + "List what changed in the scene, one line each, plainly. " * 30}
    u2 = {"role": "user", "content": "I ask the harbormaster which ship came in overnight."}
    warm = {}

    # 1. the Turn, cancelled as its reply starts, and sent again with the thought it had
    scramble()
    thought, _, _, first = post(turn([sys_msg, u1]), cancel_after=8)
    print(f"     turn 1 cancelled after {len(thought)} characters of thought (first token {first:.2f} s)", flush=True)
    again = turn([sys_msg, u1], thought + "\n\n" + CHECK)
    r, reply1, usage, first = post(again)
    p_turn = report("turn 1 sent again with its thought", usage, first, usage["prompt_tokens"] - 64)
    warm["resend"] = (again, r + reply1)
    a1 = {"role": "assistant", "content": reply1.strip()}

    # 2. the hidden pass, a conversation of its own (read whole this once), then its continuation
    _, text, usage, first = post(hidden([sys_msg, u1, a1, notes]))
    p_notes = usage["prompt_tokens"]
    print(f"     hidden pass after the reply: {p_notes} prompt tokens, "
          f"{usage['prompt_tokens_details']['cached_tokens']} cached, first token {first:.2f} s", flush=True)
    body = hidden([sys_msg, u1, a1, notes, {"role": "assistant", "content": text.strip()},
                   {"role": "user", "content": "Anything else?"}])
    _, _, usage, first = post(body)
    report("hidden pass continued", usage, first, p_notes - 16)

    # 3. the next Turn goes back to the first Turn's game, and its hidden pass to the first hidden pass's
    body = turn([sys_msg, u1, a1, u2])
    r, reply2, usage, first = post(body)
    report("turn 2, after the hidden pass", usage, first, int(p_turn * 0.8))
    warm["turn2"] = (body, r + reply2)
    a2 = {"role": "assistant", "content": reply2.strip()}
    body = hidden([sys_msg, u1, a1, u2, a2, notes])
    _, text, usage, first = post(body)
    report("hidden pass after turn 2", usage, first, int(p_notes * 0.8))
    warm["notes"] = (body, text)

    # 4. another conversation with the story's first 20 facts, then the story's third Turn
    rows = sys_msg["content"].split("\n")
    other = {"role": "system", "content": "\n".join(rows[:21] + [f"Ledger line {i}: nothing owed." for i in range(sentences)])}
    _, _, usage, first = post(hidden([other, {"role": "user", "content": "Reply UNCHANGED."}], max_tokens=8))
    cached = usage["prompt_tokens_details"]["cached_tokens"]
    print(f"     a conversation of its own: {usage['prompt_tokens']} prompt tokens, {cached} cached, "
          f"first token {first:.2f} s", flush=True)
    u3 = {"role": "user", "content": "I thank the harbormaster and walk to that ship."}
    _, _, usage, first = post(turn([sys_msg, u1, a1, u2, a2, u3]))
    report("turn 3, after a conversation of its own", usage, first, int(p_turn * 0.8))

    # the same prompts cold: the replies must match
    for name, (body, text) in warm.items():
        scramble()
        r, c, usage, first = post(body)
        cached = usage["prompt_tokens_details"]["cached_tokens"]
        cold = r + c
        same = 0
        while same < min(len(cold), len(text)) and cold[same] == text[same]:
            same += 1
        ok = cold == text
        print(f"{'ok  ' if ok else 'DIFF'} {name} read cold ({cached} cached, first token {first:.2f} s): "
              f"{same} of {len(text)} characters the same", flush=True)
        if cached > 16:
            failures.append(f"{name}: the cold read was not cold")
        if not ok:
            failures.append(f"{name}: cold and cached replies differ after {same} characters")


if __name__ == "__main__":
    for sentences in (int(a) for a in (sys.argv[2:] or ["150"])):
        print(f"--- system prompt of {sentences} sentences", flush=True)
        run(sentences)
    print("FAILED: " + "; ".join(failures) if failures else "all ok")
    sys.exit(1 if failures else 0)
