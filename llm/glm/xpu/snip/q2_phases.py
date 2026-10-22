# Per-layer replay time of each captured phase graph, Q=1 decode vs Q=2 MTP verify, for a KDA block and an MLA block.
import time, torch
from exllamav3.modules.transformer import TransformerBlock
def rep(g, n=50):
    g.replay(); torch.xpu.synchronize(); t0 = time.perf_counter()
    for _ in range(n):
        g.replay()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / n * 1e6
blocks = [b for b in model.modules if isinstance(b, TransformerBlock)]
tot = {}
for b in blocks:
    for (Q, hist), st in getattr(b, "_gstates", {}).items():
        for key, g in st.graphs.items():
            us = rep(g, 10)
            kind = type(b.attn).__name__
            tot.setdefault((Q, key[0], kind), []).append(us)
for (Q, ph, kind), v in sorted(tot.items()):
    print(f"Q={Q} phase {ph:3s} {kind:14s}: {len(v):2d} layers, mean {sum(v) / len(v):7.1f} us, total {sum(v) / 1000:6.2f} ms")
