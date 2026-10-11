# MLA attention in decode: host issue time (no sync) vs total time (sync after), per call.
import time
import exllamav3.modules.mla_attn as M
exec(open("/w/xpu/snip/speed.py").read().split("run(False, 8)")[0])
acc = [0.0, 0.0, 0]
orig = M.MLAttention.forward
def f(self, x, params, *a, **k):
    torch.xpu.synchronize(); t0 = time.perf_counter()
    y = orig(self, x, params, *a, **k)
    t1 = time.perf_counter(); torch.xpu.synchronize(); t2 = time.perf_counter()
    if x.shape[1] == 1:
        acc[0] += t1 - t0; acc[1] += t2 - t0; acc[2] += 1
    return y
M.MLAttention.forward = f
try:
    run(True, 8); acc[:] = [0.0, 0.0, 0]; run(True, 32)
finally:
    M.MLAttention.forward = orig
n = acc[2]
print(f"MLA decode calls {n} ({n // 32} per token): host issue {1000 * acc[0] / n:.2f} ms, total {1000 * acc[1] / n:.2f} ms per call")
