# Split hc_mix (decode) vs the one-program kernel: same outputs, and time per call in a graph.
import time, torch, importlib
from exllamav3.modules.transformer import TransformerBlock
import glm_xpu.k_hc as K
importlib.reload(K)
blk = [b for b in model.modules if isinstance(b, TransformerBlock) and getattr(b.mlp, "tier_li", -1) == 5][0]
hc = blk.mlp_hc
dev = torch.device("xpu")
torch.manual_seed(0)
st = torch.randn(1, 4, 4096, device=dev) * 0.3
fn = hc.fn_h if hc.fn_h is not None else hc.fn.half()
part = torch.zeros(1, 64, 25, device=dev)
def run(split):
    K.HC_SPLIT = split
    post = torch.empty(1, 4, device=dev); comb = torch.empty(1, 4, 4, device=dev); col = torch.empty(1, 4096, device=dev, dtype=torch.half)
    K.hc_mix(st, fn, hc.base, hc.scale, hc.rms_eps, hc.hc_eps, hc.sinkhorn_iters, part, post, comb, col)
    return post, comb, col, (lambda: K.hc_mix(st, fn, hc.base, hc.scale, hc.rms_eps, hc.hc_eps, hc.sinkhorn_iters, part, post, comb, col))
a = run(False); b = run(True)
rel = lambda x, y: ((x.float() - y.float()).norm() / (x.float().norm() + 1e-12)).item()
print(f"post {rel(a[0], b[0]):.1e}  comb {rel(a[1], b[1]):.1e}  collapsed {rel(a[2], b[2]):.1e}")
for split, r in ((False, a), (True, b)):
    K.HC_SPLIT = split
    g = torch.xpu.XPUGraph()
    with torch.xpu.graph(g):
        r[3]()
    g.replay(); torch.xpu.synchronize(); t0 = time.perf_counter()
    for _ in range(300):
        g.replay()
    torch.xpu.synchronize()
    print(f"{'split' if split else 'one program'}: {(time.perf_counter() - t0) / 300 * 1e6:.1f} us per call")
K.HC_SPLIT = True
