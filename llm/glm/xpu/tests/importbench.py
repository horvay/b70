"""Can the expert kernel read host memory that O_DIRECT filled directly? (the RAM tier's pinned USM memory refuses
O_DIRECT with EFAULT, so NVMe reads go through a bounce buffer and a memcpy). Ordinary anonymous memory, registered
with Level Zero (zexDriverImportExternalPointer), filled by O_DIRECT, then read zero-copy by moe_forward; compared
with the same 8 experts in VRAM and in pinned USM host memory.
  in_rootfs.sh --gpu python3 /w/xpu/tests/importbench.py
"""
import ctypes
import json
import mmap
import os
import time

from glm_xpu import compat

compat.install()
import torch

from glm_xpu import ops
from glm_xpu.tier import kcode

store = "/nvx/glm53_flash_exl3_experts_blob.bin"
dev = torch.device("xpu", 0)
meta = json.load(open(store.rsplit(".", 1)[0] + ".json"))
lay = next(l for l in meta["layers"] if l["li"] == 4)
K, rec, E = kcode(lay["K"]), lay["record_bytes"], meta["experts_per_layer"]
experts = [3, 17, 40, 77, 101, 150, 201, 260]
X = ops.moe_ops()
torch.zeros(1, device=dev)                                # bring up the runtime (and its Level Zero driver)

# --- Level Zero: the driver and the import extension
ze = ctypes.CDLL("libze_loader.so.1")
assert ze.zeInit(0) == 0
n = ctypes.c_uint32(0)
assert ze.zeDriverGet(ctypes.byref(n), None) == 0 and n.value > 0
drivers = (ctypes.c_void_p * n.value)()
assert ze.zeDriverGet(ctypes.byref(n), drivers) == 0
fn = ctypes.c_void_p()
r = ze.zeDriverGetExtensionFunctionAddress(ctypes.c_void_p(drivers[0]), b"zexDriverImportExternalPointer",
                                           ctypes.byref(fn))
print(f"zexDriverImportExternalPointer: result {r:#x}, address {fn.value}", flush=True)
assert r == 0 and fn.value, "import extension not available"
imp = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t)(fn.value)

# --- anonymous memory, imported, filled by O_DIRECT straight from the store (no bounce, no memcpy)
size = 8 * rec
mm = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
addr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
mm[:] = b"\0" * size                                       # touch
r = imp(ctypes.c_void_p(drivers[0]), ctypes.c_void_p(addr), ctypes.c_size_t(size))
print(f"import of {size / 2**20:.0f} MiB anonymous memory: result {r:#x}", flush=True)
assert r == 0
fd = os.open(store, os.O_RDONLY | os.O_DIRECT)
t0 = time.perf_counter()
for j, e in enumerate(experts):
    off, got = lay["offset"] + e * rec, 0
    mv = memoryview(mm)[j * rec:(j + 1) * rec]
    while got < rec:
        got += os.preadv(fd, [mv[got:]], off + got)
print(f"O_DIRECT straight into imported memory: {8 * rec / 1e9 / (time.perf_counter() - t0):.2f} GB/s", flush=True)

# reference copies: pinned USM host and VRAM
host = torch.empty((8, rec), dtype=torch.uint8, pin_memory=True)
host.numpy()[...] = torch.frombuffer(bytearray(mm[:size]), dtype=torch.uint8).view(8, rec).numpy()
vram = host.to(dev)
x = (torch.randn(1, 4096) * 2).half().to(dev)
sel = torch.tensor([experts], dtype=torch.int32, device=dev)
w = torch.full((1, 8), 0.125, dtype=torch.float32, device=dev)


def run(base_of):
    p = torch.zeros(E, dtype=torch.int64)
    for j, e in enumerate(experts):
        p[e] = ops.s64(base_of(j))
    p = p.to(dev)
    out = X.moe_forward(x, sel, w, p, 2048, K, E)
    torch.xpu.synchronize()
    e0, e1 = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
    e0.record()
    for _ in range(100):
        X.moe_forward(x, sel, w, p, 2048, K, E)
    e1.record()
    torch.xpu.synchronize()
    return out.float().cpu(), e0.elapsed_time(e1) / 100


# cold first access of the imported memory, VRAM use around it, and repeated accesses
p_imp = torch.zeros(E, dtype=torch.int64)
for j, e in enumerate(experts):
    p_imp[e] = ops.s64(addr + j * rec)
p_imp = p_imp.to(dev)
torch.xpu.synchronize()
free0 = torch.xpu.mem_get_info(0)[0]
t0 = time.perf_counter()
X.moe_forward(x, sel, w, p_imp, 2048, K, E)
torch.xpu.synchronize()
t_cold = time.perf_counter() - t0
free1 = torch.xpu.mem_get_info(0)[0]
print(f"imported, first (cold) call {1e3 * t_cold:.2f} ms; VRAM free before {free0 / 2**20:.0f} MiB, after "
      f"{free1 / 2**20:.0f} MiB", flush=True)
ref, t_v = run(lambda j: vram.data_ptr() + j * rec)
out_h, t_h = run(lambda j: host.data_ptr() + j * rec)
out_i, t_i = run(lambda j: addr + j * rec)
rel = lambda a: float((a - ref).norm() / ref.norm())
print(f"VRAM {t_v:.3f} ms; pinned USM host {t_h:.3f} ms (rel diff {rel(out_h):.1e}); imported anonymous memory "
      f"{t_i:.3f} ms (rel diff {rel(out_i):.1e})", flush=True)
# coherence: overwrite the imported memory from the CPU with other experts (as a reused RAM slot), O_DIRECT again
experts2 = [5, 19, 44, 80, 120, 170, 222, 280]
for j, e in enumerate(experts2):
    off, got = lay["offset"] + e * rec, 0
    mv = memoryview(mm)[j * rec:(j + 1) * rec]
    while got < rec:
        got += os.preadv(fd, [mv[got:]], off + got)
host2 = torch.empty((8, rec), dtype=torch.uint8, pin_memory=True)
host2.numpy()[...] = torch.frombuffer(bytearray(mm[:size]), dtype=torch.uint8).view(8, rec).numpy()
vram2 = host2.to(dev)
sel = torch.tensor([experts2], dtype=torch.int32, device=dev)
experts = experts2
ref2, _ = run(lambda j: vram2.data_ptr() + j * rec)
t0 = time.perf_counter()
out2, t_i2 = run(lambda j: addr + j * rec)
print(f"after a CPU-side overwrite: imported memory {t_i2:.3f} ms, rel diff vs the new experts in VRAM "
      f"{float((out2 - ref2).norm() / ref2.norm()):.1e} (0 = the GPU sees the new bytes); VRAM free now "
      f"{torch.xpu.mem_get_info(0)[0] / 2**20:.0f} MiB", flush=True)
print("IMPORTBENCH DONE")
os._exit(0)
