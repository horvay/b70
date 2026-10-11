# Decode MoE kernel (M=1, 8 VRAM-resident experts) per-layer time for gate/up and down k-splits, graph-replayed.
import time, numpy as np, torch
X = tier.X
dev = tier.dev
def layer_case(li):
    vo = tier.vram_of.reshape(tier.L, tier.E)[li]
    ids = np.nonzero(vo >= 0)[0][:8]
    x = torch.randn(1, 4096, device=dev).half() * 0.1
    sel = torch.tensor(ids, dtype=torch.int32, device=dev).view(1, 8)
    w = torch.full((1, 8), 0.125, dtype=torch.float32, device=dev)
    return x, sel, w
from glm_xpu.tier import I
def bench(li, n=300):
    x, sel, w = layer_case(li)
    fn = lambda: X.moe_forward(x, sel, w, tier.ptrs[li], I, tier.K_of[li], tier.E)
    fn(); torch.xpu.synchronize()
    g = torch.xpu.XPUGraph()
    with torch.xpu.graph(g):
        fn()
    g.replay(); torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        g.replay()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / n * 1e6
lis = {k: next(l for l in range(tier.n_moe) if tier.K_of[l] == k) for k in (2, 3)}
recs = {k: float(tier.rec_of[lis[k]]) for k in lis}
print("layers", lis, "record MB", {k: round(v / 1e6, 2) for k, v in recs.items()})
best = {}
for gu in (4, 8, 16):
    for dn in (4, 16):
        try:
            X.moe_set_splits(gu, dn)
            r = {k: bench(li) for k, li in lis.items()}
        except Exception as e:
            print(f"gu {gu} dn {dn}: {type(e).__name__} {str(e)[:80]}"); continue
        print(f"gu {gu:2d} dn {dn:2d}: " + "  ".join(f"K{k} {us:6.1f} us ({8 * recs[k] / us / 1e3:5.0f} GB/s)" for k, us in r.items()), flush=True)
X.moe_set_splits(8, 4)
