"""Stress of tier_native.Reader under the decode loop's call mix (CPU only): background reads that mark residency, urgent reads,
promotes, waits and out-of-order forgets. Run under ASan/UBSan with a debug build to catch lifetime bugs
(2026-10-30: promote() emptied the queue when the job had no queued pieces).
  in_rootfs.sh python3 /w/xpu/tests/nativestress.py /nvx/glm53_flash_exl3_experts_blob.bin /w/xpu/exl3xpu
"""
import json, os, random, sys, time
import numpy as np
sys.path.insert(0, sys.argv[2])
import tier_native as tn
store = sys.argv[1]
meta = json.load(open(store.rsplit(".", 1)[0] + ".json"))
E = meta["experts_per_layer"]
recs = [(l["offset"] + e * l["record_bytes"], l["record_bytes"]) for l in meta["layers"] for e in range(E)]
LE = len(recs); rec_max = max(n for _, n in recs)
fd = os.open(store, os.O_RDONLY | os.O_DIRECT)
res = np.full(LE, -1e4, np.float32); bonus = np.zeros(LE, np.float32)
vram_of = np.full(LE, -1, np.int64); ram_of = np.full(LE, -1, np.int64); dirty = np.zeros(1, np.uint8)
r = tn.Reader(fd, 16, 4 << 20, res.ctypes.data, bonus.ctypes.data, vram_of.ctypes.data, ram_of.ctypes.data,
              dirty.ctypes.data, 0.1, 0.05, True)
slots = np.empty((64, rec_max), np.uint8)
rnd = random.Random(1)
live = []
t0 = time.time()
for it in range(3000):
    op = rnd.random()
    k = rnd.randrange(LE)
    s = rnd.randrange(64)
    if op < 0.6:
        ram_of[k] = s
        live.append(r.submit(k, recs[k][0], recs[k][1], slots[s].ctypes.data, 8, s, False))
    elif op < 0.75:
        live.append(r.submit(k, recs[k][0], recs[k][1], slots[s].ctypes.data, 8, -1, True))
    elif op < 0.9 and live:
        r.promote(rnd.choice(live))
    elif live:
        i = live.pop(rnd.randrange(len(live)))
        if rnd.random() < 0.5:
            r.wait(i)
        r.forget(i)
    if it % 7 == 0 and live:      # waits on the most recent, as ensure does
        i = live.pop()
        r.wait(i); r.forget(i)
for i in live:
    r.wait(i); r.forget(i)
r.stop()
print(f"STRESS OK in {time.time() - t0:.1f} s")
