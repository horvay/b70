# Decode cost of one KDA layer's pieces, each captured in its own XPU graph and replayed (no launch overhead), against
# its weight bytes (memory-bound floor at ~500 GB/s).
import time, torch
from exllamav3.modules.transformer import TransformerBlock
blk = [b for b in model.modules if isinstance(b, TransformerBlock) and type(b.attn).__name__ == "GatedDeltaNet"][3]
m = blk.attn
dev = m.device
x = torch.randn(1, 1, m.hidden_size, device=dev).half() * 0.1
P = {}
def bench(label, fn, nbytes=0, n=200):
    fn(); torch.xpu.synchronize()
    g = torch.xpu.XPUGraph()
    with torch.xpu.graph(g):
        fn()
    g.replay(); torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        g.replay()
    torch.xpu.synchronize()
    us = (time.perf_counter() - t0) / n * 1e6
    floor = nbytes / 500e9 * 1e6
    print(f"  {label:28s} {us:8.1f} us   weights {nbytes / 2**20:6.1f} MiB  floor {floor:6.1f} us  x{us / floor if floor else 0:4.1f}", flush=True)
    return us
def sz(mod):
    try:
        return mod.storage_size()
    except Exception:
        return 0
tot = 0
tot += bench("qkv_proj", lambda: m.qkv_proj.forward(x, P), sz(m.qkv_proj))
tot += bench("o_proj", lambda: m.o_proj.forward(torch.zeros(1, 1, m.num_v_heads * m.v_head_dim, device=dev, dtype=torch.half) + x[..., :1], P), sz(m.o_proj))
tot += bench("g_a + g_b", lambda: m.g_b_proj.forward(m.g_a_proj.forward(x, P).half(), P), sz(m.g_a_proj) + sz(m.g_b_proj))
tot += bench("f_a + f_b", lambda: m.f_b_proj.forward(m.f_a_proj.forward(x, P).half(), P), sz(m.f_a_proj) + sz(m.f_b_proj))
tot += bench("b_proj", lambda: m.b_proj.forward(x, P), sz(m.b_proj))
print(f"  sum of the projections above: {tot:.0f} us; x 34 KDA layers = {tot * 34 / 1000:.1f} ms per token")
print("  linear kinds:", {n: type(getattr(m, n)).__name__ + "/" + type(getattr(getattr(m, n), "inner", None)).__name__
                         for n in ("qkv_proj", "o_proj", "g_a_proj", "b_proj")},
      "K:", {n: getattr(getattr(getattr(m, n), "inner", None), "K", None) for n in ("qkv_proj", "o_proj")})
