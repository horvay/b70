# Quick prefill timing: one fresh ~N-token prompt (new nonce, so no prefix-cache hit), then 8 decode tokens.
# Set N first, e.g.  dev.py -c "N = 16000; exec(open('/w/xpu/snip/prefill.py').read())"
import random, time
from exllamav3.generator import Job
N = globals().get("N", 3600)
words = "the river mountain keeper lantern dragon market silver harbor winter letter garden storm candle".split()
body = " ".join(random.choice(words) for _ in range(N))
p = model.default_chat_prompt(f"({random.randint(0, 10**9)}) Summarize this text: {body}")
ids = tok.encode(p, add_bos=False, encode_special_tokens=True)
s0 = dict(tier.stats)
job = Job(input_ids=ids, max_new_tokens=8, sampler=ArgmaxSampler())
gen.enqueue(job)
t0 = time.perf_counter()
while gen.num_remaining_jobs():
    r = gen.iterate()
    if any(x.get("stage") == "streaming" for x in r):
        break
torch.xpu.synchronize()
tp = time.perf_counter() - t0
while gen.num_remaining_jobs():
    gen.iterate()
d = {k: round(tier.stats[k] - s0[k], 2) for k in ("prefill_stage", "prefill_nvme", "t_nvme")}
print(f"{ids.shape[-1]} prompt tokens: prefill {tp:.1f} s ({ids.shape[-1] / tp:.0f} tok/s); {d}")
