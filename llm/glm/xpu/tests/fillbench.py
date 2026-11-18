"""The RAM-tier fill on its own (no model): Tier + warm() with the server's warm order, timed, with the CPU busy share
while it runs (LUKS decryption runs on the CPU).
  in_rootfs.sh --gpu python3 /w/xpu/tests/fillbench.py [--ram-gb 13] [--vram-gb 19.2]
"""
import argparse
import json
import os
import time

from glm_xpu import compat
compat.install()
import torch
from glm_xpu.tier import Tier

ap = argparse.ArgumentParser()
ap.add_argument("--store", default="/nvx/glm53_flash_exl3_experts_blob.bin")
ap.add_argument("--ram-gb", type=float, default=13.0)
ap.add_argument("--vram-gb", type=float, default=19.2)
ap.add_argument("--pretouch", action="store_true", help="write every RAM-tier page once before the fill")
a = ap.parse_args()
meta = json.load(open(a.store.rsplit(".", 1)[0] + ".json"))
li_of = {l["key"]: l["li"] for l in meta["layers"]}
stats = json.load(open("/w/xpu/data/route_stats_glm53.json"))
order = [(li, e) for _, li, e in sorted(((sc, li_of[k], e) for k, v in stats.items() if k in li_of
                                         for e, sc in enumerate(v)), reverse=True)]


def cpu():
    v = [int(x) for x in open("/proc/stat").readline().split()[1:]]
    return sum(v), v[3] + v[4]


tier = Tier(a.store, "xpu:0", int(a.vram_gb * 2**30), int(a.ram_gb * 2**30))
if a.pretouch:
    import numpy as np
    t = time.perf_counter()
    for ch in tier._ram_chunks:
        ch.numpy().fill(0)
    print(f"pre-touched {sum(ch.numel() for ch in tier._ram_chunks) / 1e9:.1f} GB in {time.perf_counter() - t:.1f} s")
c0 = cpu()
t0 = time.perf_counter()
tier.warm(order)                      # logs the VRAM warm and the RAM fill times
c1 = cpu()
busy = 1 - (c1[1] - c0[1]) / max(c1[0] - c0[0], 1)
print(f"warm {time.perf_counter() - t0:.1f} s; whole-machine CPU busy {busy:.0%} ({os.cpu_count()} CPUs)", flush=True)
os._exit(0)
