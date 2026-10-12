# Which part of the model makes prompt memory grow (~0.7 GiB per 1K tokens measured)? One prompt of N tokens (single
# forward: keep N below the generator's chunk size); for each module class: peak extra device memory above what was allocated when it was called,
# and per token. Pass 1 wraps whole blocks, pass 2 the attention and MLP modules inside them (no nesting per pass).
import random
from exllamav3.generator import Job
from exllamav3.modules.transformer import TransformerBlock
from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
from glm_xpu import block_graphs as bg
N = globals().get("MEMPROF_N", 1000)           # not "N": snippets share globals (speed.py sets N)
blocks = [m for m in model.modules if isinstance(m, TransformerBlock)]
targets = {"pass 1 (whole blocks)": [TransformerBlock],
           "pass 2 (attention, MLP)": sorted({type(b.attn) for b in blocks} | {type(b.mlp) for b in blocks}, key=lambda c: c.__name__)}
words = "the river mountain keeper lantern dragon market silver harbor winter letter garden storm candle".split()
for label, classes in targets.items():
    peak = {}
    saved = {}
    def wrap(cls, f):
        def g(self, x, params, *a, **k):
            T_ = x.shape[1] if x.dim() >= 3 else x.shape[0]   # blocks: (1, T, streams, hidden)
            if T_ < 64:
                return f(self, x, params, *a, **k)
            torch.xpu.synchronize()
            base = torch.xpu.memory_allocated()
            held.append(base - start[0])
            torch.xpu.reset_peak_memory_stats()
            y = f(self, x, params, *a, **k)
            torch.xpu.synchronize()
            d = torch.xpu.max_memory_allocated() - base
            p = peak.get(cls.__name__, (0, 0))
            if d > p[0]:
                peak[cls.__name__] = (d, T_)
            return y
        return g
    for c in classes:
        saved[c] = c.forward
        c.forward = wrap(c, c.forward)
    if TransformerBlock in classes:                     # block_graphs patches the block forward through _orig
        saved["orig"] = bg._orig["block"]
        bg._orig["block"] = wrap(TransformerBlock, saved["orig"])
    held = []
    start = [0]
    try:
        gen.pagetable.reset_page_table()
        torch.xpu.synchronize(); start[0] = torch.xpu.memory_allocated()
        p = model.default_chat_prompt(f"({random.randint(0, 10**9)}) Summarize: " + " ".join(random.choice(words) for _ in range(N)))
        ids = tok.encode(p, add_bos=False, encode_special_tokens=True)
        gen.enqueue(Job(input_ids=ids, max_new_tokens=1, sampler=ArgmaxSampler()))
        while gen.num_remaining_jobs():
            gen.iterate()
    finally:
        for c, f in saved.items():
            if c == "orig":
                bg._orig["block"] = f
            else:
                c.forward = f
    print(f"--- {label}, {ids.shape[-1]}-token prompt; held across layers at module entry: max "
          f"{max(held) / 2**20:.1f} MiB ({max(held) / ids.shape[-1] / 1024:.1f} KiB/token)", flush=True)
    for name, (d, T) in sorted(peak.items(), key=lambda kv: -kv[1][0]):
        print(f"  {name:28s} peak +{d / 2**20:8.1f} MiB  ({d / T / 1024:6.1f} KiB per token)", flush=True)
