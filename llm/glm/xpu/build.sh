#!/usr/bin/env bash
# Build the XPU op libraries inside the rootfs (scripts/in_rootfs.sh bash /w/xpu/build.sh). No GPU needed.
#   exl3xpu/_C.so          EXL3 linear (ESIMD decode-in-GEMM, reconstruct, Hadamard), from trellis-serve csrc
#   exl3xpu/_moe_glm.so    grouped MoE over pointer tables, GLM-5.3-Flash swiglu clamp 10 (glm-flash-lite b70tier n128)
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
set -euo pipefail
cd "$(dirname "$0")"
T=$(python3 -c "import torch,os;print(os.path.dirname(torch.__file__))")
D=/opt/intel/oneapi/dnnl/latest
DNNL_FLAGS=""; DNNL_LIBS=""
if [ -f $D/include/oneapi/dnnl/dnnl_sycl.hpp ]; then DNNL_FLAGS="-DEXL3_DNNL -I$D/include"; DNNL_LIBS="-L$D/lib -Wl,-rpath,$D/lib -ldnnl"; fi
COMMON="-fsycl -fsycl-targets=spir64 -O3 -ffast-math -fPIC -std=c++17 -shared -fsycl-device-code-split=per_kernel -D_GLIBCXX_USE_CXX11_ABI=1 -I csrc -I$T/include -I$T/include/torch/csrc/api/include"
LIBS="-L$T/lib -Wl,-rpath,$T/lib -lc10 -ltorch -ltorch_cpu -lc10_xpu -ltorch_xpu"
what=${1:-all}
if [ "$what" = all ] || [ "$what" = linear ]; then
  icpx $COMMON ${EXL3_FLAGS:--DEXL3_ALL_CODEBOOKS} $DNNL_FLAGS -x c++ csrc/exl3_ops.sycl -x none -o exl3xpu/_C.so.tmp $LIBS $DNNL_LIBS
  mv -f exl3xpu/_C.so.tmp exl3xpu/_C.so
fi
if [ "$what" = all ] || [ "$what" = moe ]; then
  icpx $COMMON -DEXL3_ACT_LIMIT=10.0f -x c++ csrc/exl3_moe.n128.sycl -x none -o exl3xpu/_moe_glm.so.tmp $LIBS
  mv -f exl3xpu/_moe_glm.so.tmp exl3xpu/_moe_glm.so
fi
ls -la exl3xpu/*.so
