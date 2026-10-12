# Per-token decode timing with a streaming Job on a fresh prompt: python3 xpu/dev.py xpu/snip/stream.py
import random, time
from exllamav3.generator import Job
N_TOK = globals().get("N_TOK", 48)
p = model.default_chat_prompt(f"({random.randint(0, 10**9)}) Tell me a long story about a dragon who learns to bake.")
ids = tok.encode(p, add_bos=False, encode_special_tokens=True)
s0 = dict(tier.stats)
job = Job(input_ids=ids, max_new_tokens=N_TOK, sampler=ArgmaxSampler())
gen.enqueue(job)
stamps = []
t_start = time.perf_counter()
text = ""
while gen.num_remaining_jobs():
    for r in gen.iterate():
        if r.get("stage") == "streaming" and r.get("token_ids") is not None and r["token_ids"].numel():
            stamps.append(time.perf_counter())
            text += r.get("text", "")
n = len(stamps)
ttft = stamps[0] - t_start
dec = (n - 1) / (stamps[-1] - stamps[0]) if n > 1 else 0
d = {k: round((tier.stats[k] - s0.get(k, 0)), 2) for k in tier.stats}
print(f"prompt {ids.shape[-1]} tok: first token after {ttft:.1f} s; decode {dec:.2f} tok/s over {n} steps")
h = n // 2
if n > 8:
    print(f"second half: {(n - 1 - h) / (stamps[-1] - stamps[h]):.2f} steps/s")
print("tier delta", d)
print("text:", text[:160].replace("\n", " | "))
