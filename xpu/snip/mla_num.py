# Numerical check of glm_xpu.mla_graph.decode (run eagerly) against MLAttention.forward on the same decode input and
# cache state, every MLA layer, short and 2.6K context (sparse). Appending the same row twice is idempotent.
import random
import torch
from exllamav3.modules.mla_attn import MLAttention
from glm_xpu import block_graphs as bg, mla_graph
exec(open("/w/xpu/snip/speed.py").read().split("run(False, 8)")[0])
errs = []
qs = set()
orig = MLAttention.forward
_S = {}
def fwd(self, x, params, out_dtype=None):
    y = orig(self, x, params, out_dtype) if out_dtype is not None else orig(self, x, params)
    if x.shape[0] == 1 and x.shape[1] <= 4 and mla_graph.supported(self):
        S = _S.get(0)
        if S is None:
            S = _S[0] = mla_graph.static_for(self, params["cache"], x.device)
        if S.refresh(params):
            y2 = mla_graph.decode(self, x.view(1, x.shape[1], -1).half(), params, S).view_as(y)
            qs.add(x.shape[1])
            errs.append(((y.float() - y2.float()).norm() / y.float().norm()).item())
    return y
MLAttention.forward = fwd
bg.MLA_GRAPH = False
words = "the river mountain keeper lantern dragon market silver harbor winter letter garden storm candle".split()
random.seed(7)
long_prompt = "Read this, then write a short story about a lighthouse keeper.\n" + " ".join(random.choice(words) for _ in range(2600))
try:
    for label, prompt in (("short", PROMPT), ("2.6K sparse", long_prompt)):
        errs.clear()
        qs.clear(); run(True, 8, prompt)
        print(f"{label}: query rows per call {sorted(qs)}; {len(errs)} MLA calls, rel err max {max(errs):.2e} mean {sum(errs)/len(errs):.2e}", flush=True)
finally:
    MLAttention.forward = orig
    bg.MLA_GRAPH = True
