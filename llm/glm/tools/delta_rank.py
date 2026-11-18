#!/usr/bin/env python3
"""How low-rank is the difference between two sets of matching weights (e.g. an abliterated o_proj and stock)?

  python3 -I delta_rank.py <edited.safetensors> <stock.safetensors>

Per tensor: |Δ| / |W| and the share of |Δ|² that the top 1 / 16 / 64 / 128 / 256 singular vectors hold.
"""
import json
import struct
import sys

import numpy as np


def load(path):
    f = open(path, "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
    base = 8 + n
    out = {}
    for k, t in hdr.items():
        if k == "__metadata__":
            continue
        s, e = t["data_offsets"]
        f.seek(base + s)
        raw = np.frombuffer(f.read(e - s), dtype=np.uint16 if t["dtype"] == "BF16" else np.float16)
        x = (raw.astype(np.uint32) << 16).view(np.float32) if t["dtype"] == "BF16" else raw.astype(np.float32)
        out[k] = x.reshape(t["shape"])
    return out


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    ranks = (1, 16, 64, 128, 256)
    print(f"{'tensor':58} {'shape':>12} {'|Δ|/|W|':>8} " + " ".join(f"r{r:>4}" for r in ranks))
    for k in sorted(a, key=lambda k: int(k.split(".layers.")[1].split(".")[0])):
        d = a[k] - b[k]
        rel = np.linalg.norm(d) / np.linalg.norm(b[k])
        s2 = np.clip(np.linalg.eigvalsh(d @ d.T)[::-1], 0, None)       # = squared singular values, faster than svd
        c = np.cumsum(s2) / s2.sum()
        print(f"{k:58} {str(d.shape):>12} {rel:8.4f} " + " ".join(f"{c[r - 1]:5.3f}" for r in ranks), flush=True)


if __name__ == "__main__":
    main()
