# Per-token time by decode phase (sync around each piece; adds ~0.03 ms per piece). no-sync decode.
import time, collections
from glm_xpu import block_graphs as bg
import exllamav3.modules.mla_attn as M
exec(open("/w/xpu/snip/speed.py").read().split("run(False, 8)")[0])
acc = collections.defaultdict(float)
sync = torch.xpu.synchronize
orig_run, orig_step, orig_mla = bg._run, bg._step_start, M.MLAttention.forward
def run_t(st, phase, slot, fn):
    sync(); t = time.perf_counter(); orig_run(st, phase, slot, fn); sync()
    acc[f"graph {phase}" + (" (MoE layer)" if st.moe else "")] += time.perf_counter() - t
def step_t():
    t = time.perf_counter(); orig_step(); acc["tier bookkeeping (step start, incl. its sync)"] += time.perf_counter() - t
def mla_t(self, x, params, *a, **k):
    sync(); t = time.perf_counter(); y = orig_mla(self, x, params, *a, **k); sync()
    acc["MLA attention (eager)"] += time.perf_counter() - t
    return y
bg._run, bg._step_start, M.MLAttention.forward = run_t, step_t, mla_t
try:
    run(True, 8)
    acc.clear()
    N = 32
    t0 = time.perf_counter()
    run(True, N)
    wall = time.perf_counter() - t0
finally:
    bg._run, bg._step_start, M.MLAttention.forward = orig_run, orig_step, orig_mla
tot = sum(acc.values())
print(f"wall {1000 * wall / N:.1f} ms/token (instrumented, incl. prompt), measured pieces {1000 * tot / N:.1f}")
for k, v in sorted(acc.items(), key=lambda kv: -kv[1]):
    print(f"  {1000 * v / N:6.1f} ms/token  {k}")
