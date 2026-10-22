"""Make exllamav3's CUDA-flavoured torch calls run on Intel XPU.

install() aliases the torch.cuda entry points exllamav3 uses to their torch.xpu equivalents. Device strings and
`.is_cuda` / `device.type == "cuda"` checks are patched in the exllamav3 sources (branch `xpu`), not here.
Must run before exllamav3 is imported.
"""
import torch

_done = False


def _capability(device=None):
    # exllamav3 gates CUDA-only kernels on compute capability; report something modern but harmless.
    return (9, 0)


def install():
    global _done
    if _done:
        return
    _done = True
    x, c = torch.xpu, torch.cuda
    for name in ("synchronize", "device", "Event", "stream", "Stream", "get_device_properties", "empty_cache",
                 "device_count", "set_device", "mem_get_info", "current_stream", "memory_allocated",
                 "memory_reserved", "set_per_process_memory_fraction", "memory_stats", "reset_peak_memory_stats",
                 "max_memory_allocated", "max_memory_reserved", "is_available", "get_device_name", "current_device",
                 "set_stream", "graph", "graph_pool_handle", "make_graphed_callables", "is_current_stream_capturing",
                 "manual_seed", "manual_seed_all", "get_rng_state", "set_rng_state", "memory_snapshot"):
        if hasattr(x, name):
            setattr(c, name, getattr(x, name))
    c.CUDAGraph = x.XPUGraph
    c.OutOfMemoryError = torch.OutOfMemoryError
    c.get_device_capability = _capability
    c.get_arch_list = lambda: ["sm_90"]
    c.is_initialized = x.is_initialized
    c._lazy_init = lambda: None
    c.memory.mem_get_info = x.mem_get_info
    c.memory.empty_cache = x.empty_cache
    # torch.Tensor.is_cuda stays False for XPU tensors; the sources use glm_xpu.compat.on_gpu() instead.


def on_gpu(t) -> bool:
    return t.device.type in ("cuda", "xpu")


def is_gpu_device(d) -> bool:
    return torch.device(d).type in ("cuda", "xpu")
