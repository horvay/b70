"""k_router.route (fused routing tail) vs the torch ops it replaces (block_graphs._route_subst / _route_scores).
  in_rootfs.sh --gpu python3 /w/xpu/tests/routertest.py
"""
from glm_xpu import compat

compat.install()
import torch

from glm_xpu import k_router

dev = torch.device("xpu", 0)
torch.manual_seed(0)
H, E, K, W1, scale = 4096, 288, 8, 16, 2.5
gate_c = (torch.randn(H, E + 8, device=dev) * 0.02).half()[:, :E]      # strided like gate_tensor[:, :E]
gate_n = (torch.randn(H, E + 8, device=dev) * 0.02).half()[:, :E]
esb_c, esb_n = torch.randn(E, device=dev) * 0.05, torch.randn(E, device=dev) * 0.05
bon_c = torch.where(torch.rand(E, device=dev) < 0.4, 0.1, torch.where(torch.rand(E, device=dev) < 0.3, 0.05, 0.0))
bon_n = torch.where(torch.rand(E, device=dev) < 0.4, 0.1, 0.0)
bad = 0
for Q in (1, 3):
    for trial in range(50):
        y = torch.randn(Q, H, device=dev).half()
        lc, ln = torch.matmul(y, gate_c), torch.matmul(y, gate_n)
        for bonus in (True, False):
            for nxt in (True, False):
                sel = torch.empty(Q, K, dtype=torch.int32, device=dev)
                w = torch.empty(Q, K, dtype=torch.float32, device=dev)
                picks = torch.full((2 * Q * K + Q * W1,), -1, dtype=torch.int64, device=dev)
                k_router.route(lc, ln if nxt else None, esb_c, esb_n, bon_c if bonus else None,
                               bon_n if bonus else None, scale, sel, w, picks, K, W1)
                s = torch.sigmoid(lc.float())
                key = s + esb_c
                exact = torch.topk(key, K, dim=-1, sorted=True).indices
                rsel = torch.topk(key + (bon_c if bonus else 0), K, dim=-1, sorted=True).indices
                rw = s.gather(1, rsel)
                rw = rw * scale / (rw.sum(-1, keepdim=True) + 1e-20)
                ok = (torch.equal(sel.long(), rsel) and torch.equal(picks[:Q * K].view(Q, K), rsel)
                      and torch.equal(picks[Q * K:2 * Q * K].view(Q, K), exact)
                      and torch.allclose(w, rw, rtol=1e-5, atol=1e-6))
                if nxt:
                    kn = torch.sigmoid(ln.float()) + esb_n + (bon_n if bonus else 0)
                    ok = ok and torch.equal(picks[2 * Q * K:].view(Q, W1), torch.topk(kn, W1, dim=-1).indices)
                if not ok:
                    bad += 1
                    if bad <= 3:
                        print(f"MISMATCH Q={Q} bonus={bonus} next={nxt}: sel {sel.tolist()} vs {rsel.tolist()}, "
                              f"w {w.tolist()} vs {rw.tolist()}", flush=True)
print(f"router: {bad} mismatches of {2 * 50 * 4} cases")
# speed: torch tail vs fused, Q = 1, captured in graphs as decode runs them
y = torch.randn(1, H, device=dev).half()
sel = torch.empty(1, K, dtype=torch.int32, device=dev); w = torch.empty(1, K, device=dev)
picks = torch.empty(2 * K + W1, dtype=torch.int64, device=dev)


def torch_tail():
    lc, ln = torch.matmul(y, gate_c), torch.matmul(y, gate_n)
    s = torch.sigmoid(lc.float()); key = s + esb_c
    ex = torch.topk(key, K, dim=-1, sorted=True).indices
    sl = torch.topk(key + bon_c, K, dim=-1, sorted=True).indices
    ww = s.gather(1, sl); ww = ww * scale / (ww.sum(-1, keepdim=True) + 1e-20)
    sel.copy_(sl); w.copy_(ww); picks[:K].copy_(sl.view(-1)); picks[K:2 * K].copy_(ex.view(-1))
    picks[2 * K:].copy_(torch.topk(torch.sigmoid(ln.float()) + esb_n + bon_n, W1, dim=-1).indices.view(-1))


def fused_tail():
    lc, ln = torch.matmul(y, gate_c), torch.matmul(y, gate_n)
    k_router.route(lc, ln, esb_c, esb_n, bon_c, bon_n, scale, sel, w, picks, K, W1)


for name, fn in (("torch tail", torch_tail), ("fused", fused_tail)):
    for _ in range(3):
        fn()
    torch.xpu.synchronize()
    g = torch.xpu.XPUGraph()
    with torch.xpu.graph(g):
        fn()
    g.replay(); torch.xpu.synchronize()
    e0, e1 = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
    e0.record()
    for _ in range(200):
        g.replay()
    e1.record(); torch.xpu.synchronize()
    print(f"{name}: {e0.elapsed_time(e1) / 200 * 1000:.1f} us per layer (graph replay, incl. the two gate GEMVs)",
          flush=True)
print("ROUTERTEST DONE")
