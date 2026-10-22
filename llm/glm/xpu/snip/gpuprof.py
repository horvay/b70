# GPU kernel time per decode token, grouped by kernel name (torch.profiler, XPU activity), no-sync decode.
import time, collections
from torch.profiler import profile, ProfilerActivity
exec(open("/w/xpu/snip/speed.py").read().split("run(False, 8)")[0])
run(True, 8)
NT = 16
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]) as prof:
    run(True, NT)
ev = prof.key_averages()
rows = []
for e in ev:
    dt = getattr(e, "self_device_time_total", 0) or getattr(e, "self_xpu_time_total", 0)
    if dt > 0:
        rows.append((dt, e.count, e.key))
tot = sum(r[0] for r in rows)
print(f"GPU busy {tot / 1000 / NT:.1f} ms per token (includes the prompt pass)")
for dt, n, k in sorted(rows, reverse=True)[:25]:
    print(f"{dt / 1000 / NT:7.2f} ms/tok {100 * dt / tot:5.1f}%  x{n // NT:<5} {k[:90]}")
