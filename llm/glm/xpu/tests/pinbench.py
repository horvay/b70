"""First-touch cost of pinned (USM host) memory, as the RAM tier allocates it (chunks just under 2 GiB):
  in_rootfs.sh --gpu python3 /w/xpu/tests/pinbench.py
"""
import time
import numpy as np
from glm_xpu import compat
compat.install()
import torch

src = np.ones(2**31 - 4096, np.uint8)
for label, mk in (("pageable (np.empty)", lambda n: np.empty(n, np.uint8)),
                  ("pinned (torch pin_memory)", lambda n: torch.empty(n, dtype=torch.uint8, pin_memory=True).numpy())):
    t0 = time.perf_counter()
    bufs = [mk(src.size) for _ in range(3)]
    ta = time.perf_counter() - t0
    out = []
    for i in range(2):
        t0 = time.perf_counter()
        for b in bufs:
            np.copyto(b, src)
        out.append(3 * src.nbytes / 1e9 / (time.perf_counter() - t0))
    print(f"{label:28s} alloc 6 GB {ta:5.2f} s; write: first touch {out[0]:5.2f} GB/s, again {out[1]:5.2f} GB/s", flush=True)
