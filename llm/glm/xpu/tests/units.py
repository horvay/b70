"""Small B70 checks on real checkpoint tensors (seconds, no full model):
  1. BC_LinearEXL3 (exl3xpu ESIMD) vs a pure-torch EXL3 reference (trellis decode + Hadamard), K = 4, 5, 6, M = 1/4/300.
  2. The packed store + B70 MoE kernel vs the same expert computed with BC_LinearEXL3 linears + silu_mul.

  in_rootfs.sh --gpu python3 /w/xpu/tests/units.py
"""
import json
import math
import os
import struct

from glm_xpu import compat

compat.install()
import numpy as np
import torch

from glm_xpu import ops

dev = torch.device("xpu", 0)
MD = "/models"
idx = json.load(open(f"{MD}/model.safetensors.index.json"))["weight_map"]
_hdr = {}


def tensor(name):
    fn = idx[name]
    if fn not in _hdr:
        with open(f"{MD}/{fn}", "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            _hdr[fn] = (8 + n, json.loads(f.read(n)))
    base, h = _hdr[fn]
    v = h[name]
    a, b = v["data_offsets"]
    dt = {"I16": torch.int16, "F16": torch.float16, "I32": torch.int32, "BF16": torch.bfloat16, "F32": torch.float32}[v["dtype"]]
    with open(f"{MD}/{fn}", "rb") as f:
        f.seek(base + a)
        buf = bytearray(f.read(b - a))
    return torch.frombuffer(buf, dtype=dt).view(v["shape"]).clone()


MUL1 = 0x83DCD12D


def mul1_lut():
    st = torch.arange(65536, dtype=torch.int64)
    x = (st * MUL1) & 0xFFFFFFFF
    s = (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + ((x >> 24) & 255)
    return ((1024 + s).double() * 0.00676727294921875 - 10.3828125).float().half()


LUT = mul1_lut()


def w_hat(trellis, K):
    tk, tn, _ = trellis.shape
    words = trellis.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    t = torch.arange(256)
    e = (t + 1) * K
    ih = (e - 1) // 32
    sh = 32 * (ih + 1) - e
    il = (ih - 1) % (8 * K)
    row = 8 * ((t >> 1) & 1) + 2 * ((t >> 3) & 3) + (t & 1)
    col = (t >> 5) + 8 * ((t >> 2) & 1)
    st = (((words[..., il] << 32) | words[..., ih]) >> sh) & 0xFFFF
    tiles = torch.empty(tk, tn, 16, 16, dtype=torch.half)
    tiles[..., row, col] = LUT[st]
    return tiles.permute(0, 2, 1, 3).reshape(tk * 16, tn * 16)


def H128():
    h = torch.ones(1, 1)
    while h.shape[0] < 128:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h.double() / math.sqrt(128)


def ref_linear(x, trellis, suh, svh, K):
    """fp64 reference of the EXL3 linear on CPU."""
    H = H128()
    xh = (x.double() * suh.double()).view(-1, 128) @ H
    W = w_hat(trellis, K).double()
    acc = xh.view(x.shape[0], -1) @ W
    y = (acc.view(-1, 128) @ H).view(acc.shape) * svh.double()
    return y


def rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm())


def check_linear(key, Ms=(1, 4, 300), cols=None):
    tr, suh, svh = tensor(f"{key}.trellis"), tensor(f"{key}.suh"), tensor(f"{key}.svh")
    K = tr.shape[2] // 16
    if cols:
        tr, svh = tr[:, :cols // 16].contiguous(), svh[:cols].contiguous()
    bc = ops.BC_LinearEXL3(tr.to(dev), suh.to(dev), svh.to(dev), K, None, False, True, None)
    for M in Ms:
        x = torch.randn(M, tr.shape[0] * 16, dtype=torch.half)
        ref = ref_linear(x, tr, suh, svh, K)
        y16 = bc.run_alloc(x.to(dev), bc.out_features, False).cpu()
        y32 = bc.run_alloc(x.to(dev), bc.out_features, True).cpu()
        print(f"  {key} K={K} M={M}: rel err fp16 {rel(y16, ref):.2e}  fp32 {rel(y32, ref):.2e}", flush=True)
        assert rel(y32, ref) < 5e-3, "EXL3 linear mismatch"


import sys
ONLY_MOE = "--moe" in sys.argv
if os.environ.get("GLM_XPU_MOE_PREFILL_MIN_M"):
    ops.moe_ops().moe_set_prefill_min_m(int(os.environ["GLM_XPU_MOE_PREFILL_MIN_M"]))
print("1. EXL3 linear")
if not ONLY_MOE:
  check_linear("model.language_model.layers.0.mlp.gate_proj")                       # K=4
  check_linear("model.language_model.layers.0.self_attn.qkv_proj", Ms=(1, 300))     # K=5
  check_linear("lm_head", Ms=(1, 4), cols=16384)                                   # K=6 (column slice)

store = "/nvx/glm53_flash_exl3_experts_blob.bin"
meta = json.load(open(store.rsplit(".", 1)[0] + ".json"))
X = ops.moe_ops()
if os.environ.get("GLM_XPU_MR"):
    X.moe_set_mr(int(os.environ["GLM_XPU_MR"]))
fd = os.open(store, os.O_RDONLY)
experts = [0, 7, 100, 287]
seen_k = set()
for lay in meta["layers"]:
    if lay["K"] in seen_k or (os.environ.get("GLM_XPU_MOE_K") and lay["K"] != int(os.environ["GLM_XPU_MOE_K"])):
        continue
    seen_k.add(lay["K"])
    K, rec, ml = lay["K"], lay["record_bytes"], lay["model_layer"]
    print(f"2. store + MoE kernel vs EXL3 linears, layer {ml} (K={K})")
    slots = torch.empty((len(experts), rec), dtype=torch.uint8, device=dev)
    for j, e in enumerate(experts):
        slots[j].copy_(torch.frombuffer(bytearray(os.pread(fd, rec, lay["offset"] + e * rec)), dtype=torch.uint8))
    ptrs = torch.zeros(288, dtype=torch.int64)
    for j, e in enumerate(experts):
        ptrs[e] = ops.s64(slots.data_ptr() + j * rec)
    ptrs = ptrs.to(dev)
    for M in [int(m) for m in os.environ.get("GLM_XPU_MS", "1,4,300").split(",")]:
        x = (torch.randn(M, 4096, dtype=torch.half) * 2).to(dev)
        sel = torch.tensor(experts, dtype=torch.int32).repeat(M, 1)[:, torch.randperm(4)][:, :2].contiguous().to(dev)
        w = torch.rand(M, 2, dtype=torch.float32, device=dev)
        import time; t0 = time.time()
        out = X.moe_forward(x, sel, w, ptrs, 2048, K, 288).float(); torch.xpu.synchronize()
        print(f"  M={M}: moe_forward {time.time() - t0:.3f} s", flush=True)
        ref = torch.zeros(M, 4096, dtype=torch.float32, device=dev)
        for e in experts:
            p = f"model.language_model.layers.{ml}.mlp.experts.{e}"
            lin = {m: ops.BC_LinearEXL3(tensor(f"{p}.{m}.trellis").to(dev), tensor(f"{p}.{m}.suh").to(dev),
                                        tensor(f"{p}.{m}.svh").to(dev), K, None, False, True, None)
                   for m in ("gate_proj", "up_proj", "down_proj")}
            g = lin["gate_proj"].run_alloc(x, 2048, False)
            u = lin["up_proj"].run_alloc(x, 2048, False)
            ops.silu_mul(g, u, u, 10.0)
            d = lin["down_proj"].run_alloc(u, 4096, True)
            wt = (w * (sel == e)).sum(-1, keepdim=True)
            ref += d * wt
        print(f"  M={M}: MoE kernel vs linears rel err {rel(out.cpu(), ref.cpu()):.2e}", flush=True)
        assert rel(out.cpu(), ref.cpu()) < 1e-2, "MoE mismatch"
print("UNITS OK")
