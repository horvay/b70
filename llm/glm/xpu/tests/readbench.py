"""NVMe expert-store read speed, CPU only (no GPU, no torch): the tier's read pattern (O_DIRECT, each record in
pieces on a thread pool, through a thread-local bounce into a destination buffer) vs variants.
  python3 llm/glm/xpu/tests/readbench.py [store.bin] [--n 600]
Each case reads a different random set of records, so no case benefits from another's reads in the drive's cache.
"""
import argparse
import concurrent.futures as cf
import json
import mmap
import os
import random
import threading
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("store", nargs="?", default=os.path.expanduser("~/models/glm53-store/glm53_flash_exl3_experts_blob.bin"))
ap.add_argument("--n", type=int, default=600, help="records per case")
ap.add_argument("--pinned", action="store_true", help="also copy into pinned (USM host) memory as the RAM tier does; "
                "needs the card: in_rootfs.sh --gpu")
a = ap.parse_args()
meta = json.load(open(a.store.rsplit(".", 1)[0] + ".json"))
recs = [(l["offset"] + e * l["record_bytes"], l["record_bytes"]) for l in meta["layers"]
        for e in range(meta["experts_per_layer"])]
random.seed(1)
random.shuffle(recs)
fd = os.open(a.store, os.O_RDONLY | os.O_DIRECT)
rec_max = max(n for _, n in recs)
dst = np.zeros((64, rec_max), np.uint8)       # stand-in for the RAM slots
tl = threading.local()
used = [0]


def take(n):
    s = recs[used[0]:used[0] + n]
    used[0] += n
    return s


def piece(off, n, d, copy):
    if not hasattr(tl, "buf"):
        tl.buf = mmap.mmap(-1, rec_max, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        tl.np = np.frombuffer(tl.buf, dtype=np.uint8)
    mv = memoryview(tl.buf)
    got = 0
    while got < n:
        got += os.preadv(fd, [mv[got:n]], off + got)
    if copy:
        np.copyto(d[:n], tl.np[:n])


def case(label, readers, pieces, copy=True, into=None):
    d_all = dst if into is None else into
    batch = take(a.n)
    pool = cf.ThreadPoolExecutor(readers)
    t0 = time.perf_counter()
    fs = []
    for i, (off, n) in enumerate(batch):
        pages = n // 4096
        per = -(-pages // pieces)
        for p in range(0, pages, per):
            m = min(per, pages - p) * 4096
            fs.append(pool.submit(piece, off + p * 4096, m, d_all[i % 64, p * 4096:], copy))
    for f in fs:
        f.result()
    dt = time.perf_counter() - t0
    gb = sum(n for _, n in batch) / 1e9
    print(f"{label:42s} {gb / dt:5.2f} GB/s  ({gb:.1f} GB in {dt:.1f} s)", flush=True)
    pool.shutdown()


case("tier now: 16 readers, 8 pieces, copy", 16, 8)
case("16 readers, 8 pieces, no copy", 16, 8, copy=False)
case("16 readers, 1 piece", 16, 1)
case("32 readers, 1 piece", 32, 1)
case("32 readers, 2 pieces", 32, 2)
case("64 readers, 1 piece", 64, 1)
if a.pinned:
    from glm_xpu import compat
    compat.install()
    import torch
    pin = torch.empty((64, rec_max), dtype=torch.uint8, pin_memory=True)
    assert pin.is_pinned()
    pn = pin.numpy()
    case("pinned dst: 16 readers, 8 pieces", 16, 8, into=pn)
    t0 = time.perf_counter()
    for _ in range(20):
        np.copyto(pn, dst)
    print(f"{'memcpy into pinned, 1 thread':42s} {20 * dst.nbytes / 1e9 / (time.perf_counter() - t0):5.2f} GB/s")
    t0 = time.perf_counter()
    for _ in range(20):
        np.copyto(dst, pn)
    print(f"{'memcpy out of pinned, 1 thread':42s} {20 * dst.nbytes / 1e9 / (time.perf_counter() - t0):5.2f} GB/s")
    t0 = time.perf_counter()
    for _ in range(20):
        np.copyto(dst[:32], dst[32:])
    print(f"{'memcpy plain memory, 1 thread':42s} {10 * dst.nbytes / 1e9 / (time.perf_counter() - t0):5.2f} GB/s")
