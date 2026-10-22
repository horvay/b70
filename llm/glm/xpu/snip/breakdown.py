# Where does a no-sync decode token's time go? 48-token greedy decodes with pieces switched off (timing only; the
# output is meaningless when a piece is off) plus host time in the per-token tier bookkeeping.
import time
from glm_xpu import block_graphs as bg
exec(open("/w/xpu/snip/speed.py").read().split("run(False, 8)")[0])     # defines run()
t_step = [0.0, 0]
orig_step = bg._step_start
def timed_step():
    t = time.perf_counter(); orig_step(); t_step[0] += time.perf_counter() - t; t_step[1] += 1
bg._step_start = timed_step
X = tier.X
try:
    run(True, 8)
    for label, dbg in (("full no-sync step", 0), ("routed experts off (gate/up + down kernels)", 3), ("full no-sync step", 0)):
        X.moe_set_debug(dbg); t_step[:] = [0.0, 0]
        print(f"--- {label}", flush=True)
        run(True, 48)
        print(f"    tier bookkeeping (host, once per token): {1000 * t_step[0] / max(t_step[1], 1):.1f} ms/token", flush=True)
finally:
    X.moe_set_debug(0); bg._step_start = orig_step
