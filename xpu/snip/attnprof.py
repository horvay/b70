# GPU time of attention vs the rest: one ~4K-token prefill, then decode steps at ~4K context.
import random, time
from torch.profiler import profile, ProfilerActivity
from exllamav3.generator import Job
random.seed(3)
words = "the river mountain keeper lantern dragon market silver harbor winter letter garden storm candle".split()
body = " ".join(random.choice(words) for _ in range(3600))
p = model.default_chat_prompt(f"({random.randint(0, 10**9)}) Summarize this text: {body}")
ids = tok.encode(p, add_bos=False, encode_special_tokens=True)
print("prompt tokens", ids.shape[-1])
job = Job(input_ids=ids, max_new_tokens=24, sampler=ArgmaxSampler())
gen.enqueue(job)
def is_attn(n):
    n = n.lower()
    return any(t in n for t in ("mla", "dsa", "kda", "chunk", "gla", "conv1d", "l2norm", "cumsum", "recompute_w_u",
                                "delta", "_kda_step", "solve", "intra"))
def report(prof, label):
    tot = attn = 0.0
    rows = []
    for e in prof.key_averages():
        t = e.self_device_time_total if hasattr(e, "self_device_time_total") else e.self_xpu_time_total
        if t <= 0 or e.key.startswith(("Memcpy", "aten::", "ze", "ur", "exl3xpu_C::", "exl3xpu_moe::")):
            continue
        tot += t
        if is_attn(e.key):
            attn += t
            rows.append((t, e.key, e.count))
    print(f"{label}: kernel time {tot / 1e3:.0f} ms, attention kernels {attn / 1e3:.0f} ms ({100 * attn / max(tot, 1):.0f}%)")
    for t, k, c in sorted(rows, reverse=True)[:8]:
        print(f"   {t / 1e3:8.1f} ms  {c:5d}x  {k[:70]}")
t0 = time.perf_counter()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]) as prof:
    while gen.num_remaining_jobs():
        r = gen.iterate()
        if any(x.get("stage") == "streaming" for x in r):
            break
    torch.xpu.synchronize()
print(f"prefill wall {time.perf_counter() - t0:.1f} s")
report(prof, "prefill")
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]) as prof2:
    n = 0
    while gen.num_remaining_jobs() and n < 16:
        gen.iterate(); n += 1
    torch.xpu.synchronize()
report(prof2, f"decode x{n} at ~{ids.shape[-1]} ctx")
while gen.num_remaining_jobs():
    gen.iterate()
