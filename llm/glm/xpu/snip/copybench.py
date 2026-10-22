import time
from glm_xpu.ops import moe_ops, s64
X = moe_ops()
dev = tier.dev
src = tier.ram_t[:32]; dst = [tier.vring[j % tier.vring_n] for j in range(32)]
def t(f, n=32):
    torch.xpu.synchronize(); t0 = time.perf_counter(); f(); t1 = time.perf_counter(); torch.xpu.synchronize(); t2 = time.perf_counter()
    return (t1 - t0) / n * 1e6, (t2 - t0) / n * 1e6
print("torch copy_ pinned->dev non_blocking: issue %.0f us, total %.0f us per expert" % t(lambda: [d.copy_(s, non_blocking=True) for s, d in zip(src, dst)]))
print("X.memcpy_async pinned->dev:          issue %.0f us, total %.0f us per expert" % t(lambda: [X.memcpy_async(s64(d.data_ptr()), s64(s.data_ptr()), tier.rec) for s, d in zip(src, dst)]))
print("torch copy_ dev->dev:                issue %.0f us, total %.0f us" % t(lambda: [tier.vring[(j + 1) % tier.vring_n].copy_(tier.vram[j], non_blocking=True) for j in range(32)]))
def ev():
    for j in range(32):
        e = torch.xpu.Event(); e.record()
        with torch.xpu.stream(tier.side):
            tier.side.wait_event(e); e2 = torch.xpu.Event(); e2.record(tier.side)
print("event+side stream bookkeeping:       issue %.0f us" % t(ev)[0])
import numpy as np
print("argsort LRU 2290:                    %.0f us" % (lambda: (lambda t0: (np.argsort(tier.vram_tick, kind='stable'), (time.perf_counter() - t0) * 1e6)[1])(time.perf_counter()))())
