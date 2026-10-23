"""Prompt progress against a running xpu/serve.py: a cold streamed prompt with return_progress gets a prompt_progress
chunk per prefill chunk before its first token; the same request without it gets none; a client that hangs up after
the first progress chunk has its job cancelled (the server log says so; check it after).
  python3 tests/t_progress.py <port>     (host python; the text is tests/corpus.txt prose)
  WORDS (4500)"""
import http.client
import json
import os
import sys
import time

PORT = int(sys.argv[1])
WORDS = int(os.environ.get("WORDS", "4500"))
words = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus.txt")).read().split()


def stream(offset, progress, hang_up=False):
    text = " ".join(words[offset:offset + WORDS])
    body = {"model": "dsv", "stream": True, "max_tokens": 16, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "system", "content": "The story so far:\n\n" + text},
                         {"role": "user", "content": "Continue in one sentence."}]}
    if progress:
        body["return_progress"] = True
    c = http.client.HTTPConnection("127.0.0.1", PORT, timeout=1800)
    c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    t0, seen, first = time.time(), [], None
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        ev = json.loads(line[6:])
        if "prompt_progress" in ev:
            seen.append(ev["prompt_progress"])
            print(f"  {time.time() - t0:6.1f} s  prompt_progress {ev['prompt_progress']}", flush=True)
            if hang_up:
                c.close()
                return seen, None
            continue
        delta = (ev.get("choices") or [{}])[0].get("delta") or {}
        if first is None and (delta.get("content") or delta.get("reasoning_content")):
            first = time.time() - t0
    c.close()
    return seen, first


print("with return_progress (cold):", flush=True)
seen, first = stream(30000, True)
print(f"  first token {first:.1f} s", flush=True)
ok = len(seen) >= 2 and seen[-1]["processed"] >= seen[0]["processed"] and seen[-1]["total"] > 0
print("without return_progress (a new prompt):", flush=True)
none, first = stream(40000, False)
print(f"  {len(none)} progress chunks, first token {first:.1f} s", flush=True)
ok = ok and not none
print("hanging up after the first progress chunk (a new prompt):", flush=True)
stream(50000, True, hang_up=True)
time.sleep(90)         # the server finishes the chunk it is on, then cancels
print("PROGRESS", "OK" if ok else "FAILED", flush=True)
