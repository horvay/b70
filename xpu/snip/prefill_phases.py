# Where does a long prompt's time go? One fresh N-token prompt (no prefix reuse), sync-timed per module class, plus
# the tier's staging/NVMe time and peak device memory.
import random, time, collections
from exllamav3.generator import Job
from exllamav3.modules.transformer import TransformerBlock
N = globals().get("PP_N", 2000)   # not "N": snippets share globals
blocks = [m for m in model.modules if isinstance(m, TransformerBlock)]
classes = sorted({type(b.attn) for b in blocks} | {type(b.mlp) for b in blocks}, key=lambda c: c.__name__)
acc = collections.defaultdict(float); cnt = collections.Counter()
saved = {}
def wrap(c, f):
    def g(self, x, params, *a, **k):
        if x.dim() >= 2 and x.shape[-2] == 1:          # decode steps run inside graph capture: no sync there
            return f(self, x, params, *a, **k)
        torch.xpu.synchronize(); t = time.perf_counter()
        y = f(self, x, params, *a, **k)
        torch.xpu.synchronize(); acc[c.__name__] += time.perf_counter() - t; cnt[c.__name__] += 1
        return y
    return g
for c in classes:
    saved[c] = c.forward; c.forward = wrap(c, c.forward)
words = "the river mountain keeper lantern dragon market silver harbor winter letter garden storm candle".split()
try:
    gen.pagetable.reset_page_table()
    ids = tok.encode(model.default_chat_prompt(f"({random.randint(0, 10**9)}) Summarize: " + " ".join(random.choice(words) for _ in range(N))),
                     add_bos=False, encode_special_tokens=True)
    s0 = dict(tier.stats)
    torch.xpu.reset_peak_memory_stats(); base = torch.xpu.memory_allocated()
    gen.enqueue(Job(input_ids=ids, max_new_tokens=1, sampler=ArgmaxSampler()))
    t0 = time.perf_counter()
    while gen.num_remaining_jobs():
        gen.iterate()
    wall = time.perf_counter() - t0
finally:
    for c, f in saved.items():
        c.forward = f
T = ids.shape[-1]
print(f"{T}-token prompt: {wall:.1f} s ({T / wall:.0f} tok/s); peak extra device memory {(torch.xpu.max_memory_allocated() - base) / 2**30:.2f} GiB")
for k, v in sorted(acc.items(), key=lambda kv: -kv[1]):
    print(f"  {v:6.2f} s  {k} (x{cnt[k]})")
print("  tier:", {k: round(tier.stats[k] - s0.get(k, 0), 2) for k in ("prefill_stage", "prefill_nvme", "t_nvme")})
