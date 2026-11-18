#!/usr/bin/env python3
"""Turn the difference between edited and stock weights into a low-rank PEFT adapter that glm_xpu/lora.py applies.

  python3 -I make_delta_adapter.py <edited.safetensors> <stock.safetensors> <out dir> [--rank 2]

For each tensor, Δ = edited - stock ≈ U Uᵀ Δ with U the top `rank` left singular vectors: B = U, A = Uᵀ Δ, alpha = rank
(scale 1). Made for drowzeys' Dealign o_proj abliteration of GLM-5.3-Flash, where rank 1 already holds >99% of Δ.
Tensor names like model.language_model.layers.N.x become base_model.model.model.layers.N.x, the names exllamav3 uses.
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from delta_rank import load  # noqa: E402


def save(path, tensors, metadata):
    hdr, off, blobs = {}, 0, []
    for k, v in tensors.items():
        b = np.ascontiguousarray(v, dtype=np.float32).tobytes()
        hdr[k] = {"dtype": "F32", "shape": list(v.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    hdr["__metadata__"] = metadata
    h = json.dumps(hdr).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h)
        for b in blobs:
            f.write(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("edited")
    ap.add_argument("stock")
    ap.add_argument("out")
    ap.add_argument("--rank", type=int, default=2)
    a = ap.parse_args()
    edited, stock = load(a.edited), load(a.stock)
    out, targets = {}, set()
    for k in sorted(edited):
        d = (edited[k] - stock[k]).astype(np.float64)
        w, v = np.linalg.eigh(d @ d.T)
        u = v[:, ::-1][:, :a.rank]                                   # (out, r)
        kept = w[::-1][:a.rank].clip(0).sum() / max(w.clip(0).sum(), 1e-30)
        path = k.removesuffix(".weight").replace("model.language_model.", "model.")
        out[f"base_model.model.{path}.lora_A.weight"] = (u.T @ d).astype(np.float32)     # (r, in)
        out[f"base_model.model.{path}.lora_B.weight"] = u.astype(np.float32)             # (out, r)
        targets.add(path.rsplit(".", 1)[1])
        print(f"{path}: rank {a.rank} keeps {kept:.4f} of |Δ|²", flush=True)
    os.makedirs(a.out, exist_ok=True)
    save(os.path.join(a.out, "adapter_model.safetensors"), out, {"format": "pt"})
    json.dump({"peft_type": "LORA", "r": a.rank, "lora_alpha": a.rank, "target_modules": sorted(targets),
               "fan_in_fan_out": False, "use_dora": False, "use_rslora": False, "bias": "none",
               "base_model_name_or_path": "zai-org/GLM-5.3-Flash",
               "note": f"low-rank difference {os.path.basename(a.edited)} - {os.path.basename(a.stock)}"},
              open(os.path.join(a.out, "adapter_config.json"), "w"), indent=1)
    print(f"{len(out) // 2} targets -> {a.out}")


if __name__ == "__main__":
    main()
