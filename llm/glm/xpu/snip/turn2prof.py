# Where does a short follow-up turn (cached 2K prefix + ~38 new tokens) spend its time?
import random, time, collections
from exllamav3.generator import Job
from exllamav3.modules.transformer import TransformerBlock
words = "the river mountain keeper lantern dragon market silver harbor winter letter garden storm candle".split()
history = f"({random.randint(0, 10**9)}) Here are my notes: " + " ".join(random.choice(words) for _ in range(2000))
p1 = model.default_chat_prompt(history + "\nSummarize the notes in one sentence.")
def run(text, n):
    ids = tok.encode(text, add_bos=False, encode_special_tokens=True)
    gen.enqueue(Job(input_ids=ids, max_new_tokens=n, sampler=ArgmaxSampler()))
    out, its = [], []
    while gen.num_remaining_jobs():
        t0 = time.perf_counter(); rs = gen.iterate(); its.append(round(time.perf_counter() - t0, 2))
        out += [r.get("text", "") for r in rs]
    return "".join(out), its
gen.pagetable.reset_page_table()
r1, _ = run(p1, 16)
acc = collections.defaultdict(float); cnt = collections.Counter()
blocks = [m for m in model.modules if isinstance(m, TransformerBlock)]
classes = sorted({type(b.attn) for b in blocks} | {type(b.mlp) for b in blocks}, key=lambda c: c.__name__)
saved = {}
def wrap(name, f):
    def g(*a, **k):
        x = a[1] if len(a) > 1 else None
        if isinstance(x, torch.Tensor) and x.dim() >= 2 and x.shape[-2] == 1:
            return f(*a, **k)
        torch.xpu.synchronize(); t = time.perf_counter()
        y = f(*a, **k)
        torch.xpu.synchronize(); acc[name] += time.perf_counter() - t; cnt[name] += 1
        return y
    return g
for c in classes:
    saved[c] = c.forward; c.forward = wrap(c.__name__, c.forward)
tsaved = {n: getattr(tier, n) for n in ("stage_layer", "ensure", "_refill", "_end_large_forward", "_pf_start", "_pf_stop")}
def twrap(n, f):
    def g(*a, **k):
        t = time.perf_counter(); y = f(*a, **k); acc["tier." + n] += time.perf_counter() - t; cnt["tier." + n] += 1
        return y
    return g
for n, f in tsaved.items():
    setattr(tier, n, twrap(n, f))
try:
    r2, its = run(p1 + r1 + "<|user|>\nNow make it shorter.<|assistant|>\n", 4)
finally:
    for c, f in saved.items():
        c.forward = f
    for n in tsaved:
        delattr(tier, n)
print("iterate() times (s):", its)
for k, v in sorted(acc.items(), key=lambda kv: -kv[1]):
    print(f"  {v:6.2f} s  {k} (x{cnt[k]})")
