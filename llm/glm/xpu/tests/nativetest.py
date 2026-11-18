"""CPU-only checks of exl3xpu.tier_native.Reader (no GPU, no torch): bytes, residency marking, errors, speed.
  in_rootfs.sh python3 /w/xpu/tests/nativetest.py [/nvx/glm53_flash_exl3_experts_blob.bin]
"""
import concurrent.futures as cf
import json
import mmap
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, "/w/xpu")
from exl3xpu import tier_native

store = sys.argv[1] if len(sys.argv) > 1 else "/nvx/glm53_flash_exl3_experts_blob.bin"
meta = json.load(open(store.rsplit(".", 1)[0] + ".json"))
E = meta["experts_per_layer"]
recs = [(l["offset"] + e * l["record_bytes"], l["record_bytes"]) for l in meta["layers"] for e in range(E)]
LE = len(recs)
fd = os.open(store, os.O_RDONLY | os.O_DIRECT)
res = np.full(LE, -1e4, np.float32)
bonus = np.zeros(LE, np.float32)
vram_of = np.full(LE, -1, np.int64)
ram_of = np.full(LE, -1, np.int64)
dirty = np.zeros(1, np.uint8)
r = tier_native.Reader(fd, 16, 4 << 20, res.ctypes.data, bonus.ctypes.data, vram_of.ctypes.data, ram_of.ctypes.data,
                       dirty.ctypes.data, 0.1, 0.05, True)
random.seed(3)
rec_max = max(n for _, n in recs)


def plain(off, n):
    m = mmap.mmap(-1, n)
    got = 0
    while got < n:
        got += os.preadv(fd, [memoryview(m)[got:n]], off + got)
    return bytes(m[:n])


# 1. bytes
keys = random.sample(range(LE), 300)
bufs = [np.empty(rec_max, np.uint8) for _ in keys]
ids = [r.submit(k, recs[k][0], recs[k][1], b.ctypes.data, 8) for k, b in zip(keys, bufs)]
for i in ids:
    r.wait(i); r.forget(i)
bad = sum(bufs[j][:recs[k][1]].tobytes() != plain(*recs[k]) for j, k in enumerate(keys))
print(f"1. bytes: {bad} of {len(keys)} records differ from plain pread", flush=True)
assert bad == 0

# 2. residency: marked only while the expert still owns the slot; VRAM-resident gets the VRAM bonus
k1, k2, k3 = keys[:3]
ram_of[k1] = 5; ram_of[k2] = 6; ram_of[k3] = 7; vram_of[k3] = 3
b = np.empty(rec_max, np.uint8)
i1 = r.submit(k1, recs[k1][0], recs[k1][1], b.ctypes.data, 8, 5)
i2 = r.submit(k2, recs[k2][0], recs[k2][1], b.ctypes.data, 8, 99)      # slot taken over meanwhile
i3 = r.submit(k3, recs[k3][0], recs[k3][1], b.ctypes.data, 8, 7)
for i in (i1, i2, i3):
    r.wait(i); r.forget(i)
print(f"2. residency: res {res[k1]}, {res[k2]}, {res[k3]}; bonus {bonus[k1]:.2f}, {bonus[k2]:.2f}, {bonus[k3]:.2f}; "
      f"dirty {dirty[0]}", flush=True)
assert res[k1] == 0 and res[k2] == -1e4 and res[k3] == 0 and dirty[0] == 1
assert abs(bonus[k1] - 0.05) < 1e-6 and bonus[k2] == 0 and abs(bonus[k3] - 0.1) < 1e-6

# 3. errors: a read past the end of the file raises on wait, and the reader keeps working
size = os.fstat(fd).st_size
i = r.submit(0, size, 4096 * 4, b.ctypes.data, 2)
try:
    r.wait(i)
    raise SystemExit("3. errors: no exception for a read past EOF")
except RuntimeError as e:
    print(f"3. errors: {e}", flush=True)
r.forget(i)
i = r.submit(keys[0], recs[keys[0]][0], recs[keys[0]][1], b.ctypes.data, 8)
r.wait(i); r.forget(i)

# 4. speed vs the Python pool (different records each, so neither benefits from the other's page cache... O_DIRECT)
n = 400
ks = random.sample(range(LE), 2 * n)
dst = np.empty((32, rec_max), np.uint8)
t0 = time.perf_counter()
ids = [r.submit(k, recs[k][0], recs[k][1], dst[j % 32].ctypes.data, 8) for j, k in enumerate(ks[:n])]
for i in ids:
    r.wait(i); r.forget(i)
dt_n = time.perf_counter() - t0
gb = sum(recs[k][1] for k in ks[:n]) / 1e9
pool = cf.ThreadPoolExecutor(16)
tl = __import__("threading").local()


def piece(off, cnt, d):
    if not hasattr(tl, "buf"):
        tl.buf = mmap.mmap(-1, rec_max); tl.np = np.frombuffer(tl.buf, np.uint8)
    got = 0
    while got < cnt:
        got += os.preadv(fd, [memoryview(tl.buf)[got:cnt]], off + got)
    np.copyto(d[:cnt], tl.np[:cnt])


t0 = time.perf_counter()
fs = []
for j, k in enumerate(ks[n:]):
    off, rec = recs[k]
    per = -(-rec // 4096 // 8) * 4096
    for o in range(0, rec, per):
        fs.append(pool.submit(piece, off + o, min(per, rec - o), dst[j % 32, o:]))
for f in fs:
    f.result()
dt_p = time.perf_counter() - t0
gb_p = sum(recs[k][1] for k in ks[n:]) / 1e9
print(f"4. speed: native {gb / dt_n:.2f} GB/s, Python pool {gb_p / dt_p:.2f} GB/s ({n} records each)", flush=True)
# 5. priority: behind 64 queued background reads, an urgent read (and a promoted one) waits far less than a plain one
def latency(kind):
    bg = random.sample(range(LE), 64)
    ids = [r.submit(k, recs[k][0], recs[k][1], dst[j % 32].ctypes.data, 8) for j, k in enumerate(bg)]
    k = random.randrange(LE)
    b2 = np.empty(rec_max, np.uint8)
    t0 = time.perf_counter()
    if kind == "promoted":
        i = ids[-1]
        r.promote(i)
    else:
        i = r.submit(k, recs[k][0], recs[k][1], b2.ctypes.data, 8, -1, kind == "urgent")
    r.wait(i)
    dt = time.perf_counter() - t0
    for j in ids:
        r.wait(j); r.forget(j)
    r.forget(i)
    return dt


lat = {kind: latency(kind) for kind in ("plain", "urgent", "promoted")}
print("5. priority behind 64 queued reads: " + ", ".join(f"{k} {1e3 * v:.1f} ms" for k, v in lat.items()), flush=True)
assert lat["urgent"] < 0.5 * lat["plain"] and lat["promoted"] < 0.5 * lat["plain"]
r.stop()
print("NATIVETEST OK")
