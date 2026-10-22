"""Teacher-forced reference panel on the B70, in process (no server): top-1 agreement and mean KL over the reference
top-20 support against stock exllamav3 1.5.1 on CUDA (glm-flash-lite reference/glm-5.3-flash-exl3-ref-panel.json).

  in_rootfs.sh --gpu python3 /w/xpu/tests/panel.py [--n 2] [--ram-gb 24]
"""
import argparse
import json
import math
import time

from glm_xpu.runner import load, log
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--panel", default="/w/glm-flash-lite/reference/glm-5.3-flash-exl3-ref-panel.json")
ap.add_argument("--store", default="/nvx/glm53_flash_exl3_experts_blob.bin")
ap.add_argument("--n", type=int, default=2)
ap.add_argument("--ram-gb", type=float, default=24.0)
ap.add_argument("--max-vram-slots", type=int, default=None)
ap.add_argument("--out")
a = ap.parse_args()

model, config, cache, tok, tier = load("/models", a.store, cache_size=0, ram_gb=a.ram_gb,
                                       max_vram_slots=a.max_vram_slots)
panel = json.load(open(a.panel))[:a.n]
tot = dict(pos=0, top1=0, kl=0.0)
per = []
for i, item in enumerate(panel):
    toks, P = item["tokens"], item["prompt_len"]
    ref_ids, ref_lp = item["top20_ids"], item["top20_lp"]
    ids = torch.tensor([toks], dtype=torch.long)
    t0 = time.time()
    with torch.inference_mode():
        logits = model.forward(ids, {"attn_mode": "flash_attn_nc"}).float()[0]
    torch.xpu.synchronize()
    dt = time.time() - t0
    lp = torch.log_softmax(logits, -1).cpu()
    s = dict(pos=0, top1=0, kl=0.0)
    for j in range(len(ref_ids)):
        row = lp[P - 1 + j]
        srv_arg = int(row.argmax())
        pr = [math.exp(x) for x in ref_lp[j]]
        zr = sum(pr)
        qs = [math.exp(float(row[t])) for t in ref_ids[j]]
        zq = sum(qs)
        kl = sum((p / zr) * (math.log(p / zr) - math.log(max(q / zq, 1e-30))) for p, q in zip(pr, qs))
        s["pos"] += 1
        s["top1"] += int(srv_arg == ref_ids[j][0])
        s["kl"] += kl
    for k in tot:
        tot[k] += s[k]
    per.append(dict(i=i, tokens=len(toks), seconds=dt, top1=s["top1"] / s["pos"], kl=s["kl"] / s["pos"]))
    log(f"[{i}] {len(toks)} tokens in {dt:.1f} s: top1 {s['top1'] / s['pos']:.4f}  KL {s['kl'] / s['pos']:.5f}")
    log(f"tier {tier.summary()}")
res = dict(positions=tot["pos"], top1=tot["top1"] / tot["pos"], kl=tot["kl"] / tot["pos"], per=per)
log(json.dumps(res))
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
