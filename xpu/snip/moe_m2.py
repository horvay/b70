# moe_forward for M=2 (MTP verify): pair mode vs grouped (align) mode, with realistic picks (6 VRAM + 2 RAM experts per
# token, half shared between the tokens), graph-replayed.
import time, numpy as np, torch
X = tier.X; dev = tier.dev
from glm_xpu.tier import I
li = 0
vo = tier.vram_of.reshape(tier.L, tier.E)[li]; ro = tier.ram_of.reshape(tier.L, tier.E)[li]
tier.refresh_ptrs(); torch.xpu.synchronize()
pt = tier.ptrs_h[li]
# only experts with a live pointer: a RAM expert still being read in (fresh server) has pointer 0 -> NULL in the kernel
v = np.nonzero((vo >= 0) & (pt != 0))[0]; r = np.nonzero((vo < 0) & (ro >= 0) & (pt != 0))[0]
assert len(v) >= 9 and len(r) >= 3, (len(v), len(r))
rows = {1: [np.r_[v[:6], r[:2]]], 2: [np.r_[v[:6], r[:2]], np.r_[v[3:9], r[1:3]]]}
x = torch.randn(2, 4096, device=dev).half() * 0.1
def bench(M, n=100):
    sel = torch.tensor(np.stack(rows[M]), dtype=torch.int32, device=dev)
    w = torch.full((M, 8), 0.125, dtype=torch.float32, device=dev)
    fn = lambda: X.moe_forward(x[:M], sel, w, tier.ptrs[li], I, tier.K_of[li], tier.E)
    fn(); torch.xpu.synchronize()
    g = torch.xpu.XPUGraph()
    with torch.xpu.graph(g):
        fn()
    g.replay(); torch.xpu.synchronize(); t0 = time.perf_counter()
    for _ in range(n):
        g.replay()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / n * 1e6
print(f"M=1: {bench(1):7.1f} us")
for pm in (4, 1):
    X.moe_set_pair_max_m(pm)
    print(f"M=2 {'pair mode' if pm >= 2 else 'grouped  '}: {bench(2):7.1f} us", flush=True)
X.moe_set_pair_max_m(4)
sel = torch.tensor(np.stack(rows[2]), dtype=torch.int32, device=dev); w = torch.rand(2, 8, device=dev)
X.moe_set_pair_max_m(4); a = X.moe_forward(x, sel, w, tier.ptrs[li], I, tier.K_of[li], tier.E).float()
X.moe_set_pair_max_m(1); b = X.moe_forward(x, sel, w, tier.ptrs[li], I, tier.K_of[li], tier.E).float()
print(f"pair vs grouped M=2 rel diff {((a - b).norm() / a.norm()).item():.2e}")
