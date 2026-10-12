# Per-kernel cost inside an XPU graph replay (tiny kernels), and the replay cost of one block's real phase graphs.
import time, torch
from exllamav3.modules.transformer import TransformerBlock
from glm_xpu import block_graphs as bg
dev = torch.device("xpu")
def rep(g, n=300):
    g.replay(); torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        g.replay()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / n * 1e6
a = torch.zeros(4096, device=dev)
for nk in (1, 10, 40):
    g = torch.xpu.XPUGraph()
    with torch.xpu.graph(g):
        for _ in range(nk):
            a.add_(1.0)
    print(f"graph of {nk:2d} tiny kernels: {rep(g):7.1f} us  ({rep(g) / nk:5.1f} us per kernel)", flush=True)
blk = [b for b in model.modules if isinstance(b, TransformerBlock) and getattr(b.mlp, "tier_li", -1) == 5][0]
st = blk._gstate
for key, g in st.graphs.items():
    print(f"block {blk.key} phase {key}: {rep(g):7.1f} us per replay", flush=True)
