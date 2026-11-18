#!/usr/bin/env python3
"""Fetch named tensors out of a Hugging Face safetensors checkpoint by HTTP range, without downloading the shards.

  python3 -I fetch_tensors.py <repo> <out.safetensors> <regex> [--revision main]

Reads model.safetensors.index.json and each needed shard's header, then requests only the matching tensors' bytes.
Uses the token from `hf auth login` (gated repos need their terms accepted on the website first).
"""
import argparse
import json
import os
import re
import struct
import urllib.request


def token():
    for p in (os.environ.get("HF_TOKEN_PATH", ""), os.path.expanduser("~/.cache/huggingface/token")):
        if p and os.path.isfile(p):
            return open(p).read().strip()
    return os.environ.get("HF_TOKEN")


def get(url, start=None, end=None):
    req = urllib.request.Request(url)
    if (t := token()):
        req.add_header("Authorization", f"Bearer {t}")
    if start is not None:
        req.add_header("Range", f"bytes={start}-{end}")
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("out")
    ap.add_argument("pattern")
    ap.add_argument("--revision", default="main")
    a = ap.parse_args()
    base = f"https://huggingface.co/{a.repo}/resolve/{a.revision}/"
    wmap = json.loads(get(base + "model.safetensors.index.json"))["weight_map"]
    names = sorted(k for k in wmap if re.search(a.pattern, k))
    if not names:
        raise SystemExit(f"no tensor matches {a.pattern!r}")
    meta, blobs, off = {}, [], 0
    by_shard = {}
    for n in names:
        by_shard.setdefault(wmap[n], []).append(n)
    for shard, ns in sorted(by_shard.items()):
        n_hdr = struct.unpack("<Q", get(base + shard, 0, 7))[0]
        hdr = json.loads(get(base + shard, 8, 8 + n_hdr - 1))
        for n in ns:
            t = hdr[n]
            s, e = t["data_offsets"]
            data = get(base + shard, 8 + n_hdr + s, 8 + n_hdr + e - 1)
            assert len(data) == e - s, (n, len(data), e - s)
            meta[n] = {"dtype": t["dtype"], "shape": t["shape"], "data_offsets": [off, off + len(data)]}
            blobs.append(data)
            off += len(data)
            print(f"{n} {t['dtype']} {t['shape']} from {shard}", flush=True)
    meta["__metadata__"] = {"source": f"{a.repo}@{a.revision}"}
    h = json.dumps(meta).encode()
    h += b" " * (-len(h) % 8)
    with open(a.out + ".part", "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h)
        for b in blobs:
            f.write(b)
    os.replace(a.out + ".part", a.out)
    print(f"{len(names)} tensors, {off / 2**30:.2f} GiB -> {a.out}")


if __name__ == "__main__":
    main()
