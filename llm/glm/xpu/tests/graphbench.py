"""Does an XPU graph replay block the host until the GPU has caught up? (decode profile: replay calls took ~0.65 ms
each, about the GPU time of the phases they launch.) No model: graphs of known GPU time, replayed back to back.
  in_rootfs.sh --gpu python3 /w/xpu/tests/graphbench.py
"""
import time

from glm_xpu import compat

compat.install()
import torch

dev = torch.device("xpu", 0)
a = torch.randn(4096, 4096, device=dev, dtype=torch.half)
b = torch.randn(4096, 4096, device=dev, dtype=torch.half)
small = torch.randn(1, 4096, device=dev, dtype=torch.half)


def make(fn):
    for _ in range(2):
        fn()
    torch.xpu.synchronize()
    g = torch.xpu.XPUGraph()
    with torch.xpu.graph(g):
        fn()
    torch.xpu.synchronize()
    return g


heavy = make(lambda: [torch.matmul(a, b) for _ in range(4)])          # several ms of GPU work
tiny = make(lambda: [torch.matmul(small, b) for _ in range(20)])        # 20 small kernels, ~launch-bound
for name, g, n in (("heavy", heavy, 10), ("tiny", tiny, 200)):
    g.replay(); torch.xpu.synchronize()
    e0, e1 = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
    e0.record()
    t0 = time.perf_counter()
    for _ in range(n):
        g.replay()
    t_host = time.perf_counter() - t0
    e1.record()
    torch.xpu.synchronize()
    t_all = time.perf_counter() - t0
    gpu = e0.elapsed_time(e1) / 1000
    print(f"{name}: {n} replays: host time in replay() {1e3 * t_host / n:.3f} ms each, GPU {1e3 * gpu / n:.3f} ms each, "
          f"wall to sync {1e3 * t_all / n:.3f} ms each -> {'BLOCKING' if t_host > 0.7 * t_all else 'async'}", flush=True)
# 40 distinct graphs replayed once each per round, as decode replays one graph per layer and phase
many = [make(lambda: [torch.matmul(small, b) for _ in range(20)]) for _ in range(40)]
for g in many:
    g.replay()
torch.xpu.synchronize()
e0, e1 = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
e0.record()
t0 = time.perf_counter()
for _ in range(5):
    for g in many:
        g.replay()
t_host = time.perf_counter() - t0
e1.record()
torch.xpu.synchronize()
t_all = time.perf_counter() - t0
n = 5 * len(many)
print(f"40 distinct: host time in replay() {1e3 * t_host / n:.3f} ms each, GPU {e0.elapsed_time(e1) / n:.3f} ms each, "
      f"wall {1e3 * t_all / n:.3f} ms each -> {'BLOCKING' if t_host > 0.7 * t_all else 'async'}", flush=True)
# eager launches of the same work, for comparison
torch.xpu.synchronize()
t0 = time.perf_counter()
for _ in range(200):
    for _ in range(20):
        torch.matmul(small, b)
t_host = time.perf_counter() - t0
torch.xpu.synchronize()
print(f"eager tiny: host {1e3 * t_host / 200:.3f} ms per 20 launches, wall {1e3 * (time.perf_counter() - t0) / 200:.3f} ms",
      flush=True)
