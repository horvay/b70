"""XPU stand-in for exllamav3's compiled CUDA extension (exllamav3_ext).

Ops are implemented in glm_xpu.ops_* modules with torch, Triton (Intel XPU) and exl3xpu ESIMD kernels. Any op not
implemented yet resolves to a stub that raises when called, naming the op, so bring-up is driven by real call sites.
"""
import os
from glm_xpu import compat as _compat

_compat.install()

from glm_xpu.ops import *          # noqa: F401,F403  (implemented ops)

_TRACE = os.environ.get("GLM_XPU_TRACE_MISSING") == "1"
_missing_seen = set()


class _Missing:
    def __init__(self, name):
        self.__name__ = name

    def __call__(self, *a, **k):
        raise NotImplementedError(f"exllamav3_ext.{self.__name__} is not implemented on XPU")

    def __repr__(self):
        return f"<missing exllamav3_ext.{self.__name__}>"


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    if _TRACE and name not in _missing_seen:
        _missing_seen.add(name)
        print(f" -- exllamav3_ext: missing op {name}", flush=True)
    return _Missing(name)
