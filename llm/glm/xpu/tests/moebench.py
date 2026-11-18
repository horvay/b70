"""MoE kernel alone at decode shape (M = 1, top-8): 8 real experts from the store, placed in VRAM slots, pinned RAM
(read zero-copy over PCIe) or mixed; gate/up and down timed separately (moe_set_debug skip bits). No model.
  in_rootfs.sh --gpu python3 /w/xpu/tests/moebench.py [--layer-li 4]
"""
import argparse
import json
import os

from glm_xpu import compat

compat.install()
import torch

from glm_xpu import ops
from glm_xpu.tier import kcode

ap = argparse.ArgumentParser()
ap.add_argument("--store", default="/nvx/glm53_flash_exl3_experts_blob.bin")
ap.add_argument("--layer-li", type=int, default=4)
ap.add_argument("--iters", type=int, default=200)
a = ap.parse_args()
dev = torch.device("xpu", 0)
meta = json.load(open(a.store.rsplit(".", 1)[0] + ".json"))
lay = next(l for l in meta["layers"] if l["li"] == a.layer_li)
K, rec = kcode(lay["K"]), lay["record_bytes"]
E = meta["experts_per_layer"]
X = ops.moe_ops()
fd = os.open(a.store, os.O_RDONLY)
experts = [3, 17, 40, 77, 101, 150, 201, 260]
host = torch.empty((8, rec), dtype=torch.uint8, pin_memory=True)
for j, e in enumerate(experts):
    host[j].copy_(torch.frombuffer(bytearray(os.pread(fd, rec, lay["offset"] + e * rec)), dtype=torch.uint8))
vram = host.to(dev)


def ptrs_for(where):
    p = torch.zeros(E, dtype=torch.int64)
    for j, e in enumerate(experts):
        src = vram if where[j] == "v" else host
        p[e] = ops.s64(src.data_ptr() + j * rec)
    return p.to(dev)


x = (torch.randn(1, 4096) * 2).half().to(dev)
sel = torch.tensor([experts], dtype=torch.int32, device=dev)
w = torch.full((1, 8), 0.125, dtype=torch.float32, device=dev)
print(f"layer li {a.layer_li}, K code {K}, record {rec / 2**20:.2f} MiB, M=1 top-8", flush=True)
for name, where in (("all VRAM", "vvvvvvvv"), ("6 VRAM + 2 RAM", "vvvvvvrr"), ("all RAM", "rrrrrrrr")):
    ptrs = ptrs_for(where)
    res = {}
    for part, skip in (("gate/up", 2 | 4), ("down", 1 | 4), ("all", 0)):
        X.moe_set_debug(skip)
        for _ in range(5):
            X.moe_forward(x, sel, w, ptrs, 2048, K, E)
        torch.xpu.synchronize()
        e0, e1 = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
        e0.record()
        for _ in range(a.iters):
            X.moe_forward(x, sel, w, ptrs, 2048, K, E)
        e1.record()
        torch.xpu.synchronize()
        res[part] = e0.elapsed_time(e1) / a.iters
    X.moe_set_debug(0)
    nv, nr = where.count("v"), where.count("r")
    gu_bytes = 8 * rec * 2 / 3
    print(f"  {name}: gate/up {res['gate/up']:.3f} ms ({gu_bytes / res['gate/up'] / 1e6:.0f} GB/s), down "
          f"{res['down']:.3f} ms, whole moe_forward {res['all']:.3f} ms", flush=True)
print("MOEBENCH DONE")
