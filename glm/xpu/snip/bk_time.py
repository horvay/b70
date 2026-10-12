# Production no-sync decode: wall time per token vs CPU time spent in the per-token bookkeeping (step_nosync).
import time
exec(open("/w/xpu/snip/speed.py").read().split("run(False, 8)")[0])
acc = {"t": 0.0, "n": 0, "max": 0.0}
f0 = tier.step_nosync
def w(*a, **k):
    t = time.perf_counter(); r = f0(*a, **k); dt = time.perf_counter() - t
    acc["t"] += dt; acc["n"] += 1; acc["max"] = max(acc["max"], dt)
    return r
tier.step_nosync = w
try:
    t0 = time.perf_counter()
    run(True, 128)
finally:
    del tier.step_nosync
print(f"step_nosync: {acc['n']} calls, mean {1000 * acc['t'] / max(acc['n'], 1):.1f} ms, max {1000 * acc['max']:.1f} ms")
