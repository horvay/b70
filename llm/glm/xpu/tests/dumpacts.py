"""Per-layer activations of one token through the eager cache-less forward, for bisecting the C++ engine
(engines/glm.cpp): per layer [attn_in 4096 | attn_out 4096 | mlp_in 4096 | mlp_out 4096 | R 4x4096] fp32, then the
log-probabilities. Exact routing (GLM_XPU_RES_BONUS=0), no LoRA.
  in_rootfs.sh --gpu python3 /w/xpu/tests/dumpacts.py --token 11 --out /w/run/ref/acts
"""
import argparse
import json
import os

from glm_xpu.runner import load, log
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--store", default="/nvx/glm53_flash_exl3_experts_blob.bin")
ap.add_argument("--token", type=int, required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
model, config, cache, tok, tier = load("/models", a.store, cache_size=4096, ram_gb=8)
from exllamav3.modules.transformer import TransformerBlock

rows = {}


def wrap(obj, name, L, key_in, key_out):
    f = obj.forward

    def g(x, params, *args, **kw):
        y = f(x, params, *args, **kw)
        rows[(L, key_in)] = x.float().reshape(-1)[:4096].cpu()
        rows[(L, key_out)] = y.float().reshape(-1)[:4096].cpu()
        return y
    obj.forward = g


blocks = [m for m in model.modules if isinstance(m, TransformerBlock)]
for L, b in enumerate(blocks):
    wrap(b.attn, "attn", L, "attn_in", "attn_out")
    wrap(b.mlp, "mlp", L, "mlp_in", "mlp_out")
    bf = b.forward

    def bg(x, params, *args, _L=L, _f=bf, **kw):
        y = _f(x, params, *args, **kw)
        rows[(_L, "R")] = y.float().reshape(-1)[:4 * 4096].cpu()
        return y
    b.forward = bg
with torch.inference_mode():
    logits = model.forward(torch.tensor([[a.token]], dtype=torch.long), {"attn_mode": "flash_attn_nc"}).float()[0, -1]
lp = torch.log_softmax(logits[:tok.actual_vocab_size], -1).cpu()
out = []
for L in range(len(blocks)):
    out += [rows[(L, k)] for k in ("attn_in", "attn_out", "mlp_in", "mlp_out", "R")]
os.makedirs(os.path.dirname(a.out), exist_ok=True)
torch.cat(out).numpy().tofile(a.out + ".f32")
lp.numpy().tofile(a.out + ".lp.f32")
json.dump({"token": a.token, "layers": len(blocks), "vocab": lp.shape[0]}, open(a.out + ".json", "w"))
log(f"DUMPACTS {len(blocks)} layers -> {a.out}")
os._exit(0)
