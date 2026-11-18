#!/usr/bin/env bash
# Build the XPU op libraries inside the rootfs (scripts/in_rootfs.sh bash /w/xpu/build.sh). No GPU needed.
#   exl3xpu/_C.so          EXL3 linear (ESIMD decode-in-GEMM, reconstruct, Hadamard), from trellis-serve csrc
#   exl3xpu/_moe_glm.so    grouped MoE over pointer tables, GLM-5.3-Flash swiglu clamp 10 (glm-flash-lite b70tier n128);
#                          ahead of time for the B70 (EXL3_AOT=0: JIT), with exl3xpu/_moe_prefill.so (its prompt
#                          path, large GRF) and exl3xpu/_moe_align.so (JIT, routing)
#   exl3xpu/tier_native*.so  the expert tier's native NVMe reader (C++ threads)
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
  if [ "${EXL3_AOT:-1}" = 1 ]; then
    # Ahead of time for the B70 (Battlemage G31): no driver JIT of ~270 kernels at the first use of each (startbench
    # "loaded and installed" 47 -> 21 s, first 4.9k-token prompt 68 -> 52 s), and the AOT gate/up kernel ran 0.216 ->
    # 0.140 ms per layer (tests/moebench.py, 2026-10-30). ESIMD needs -vc-codegen, which breaks ordinary SYCL kernels, so
    # the one GLM launches (routing alignment) is a separate JIT library. The prompt (DPAS) path needs large-GRF code,
    # the decode GEMV the default (large GRF for all: warm prompt 25.5 -> 14.1 s but decode MoE 0.71 -> 1.18 ms per
    # layer), so it is a second build of the same source (_moe_prefill.so, namespace moe_pf). spir64 images ride along
    # as JIT fallbacks.
    AOT="${COMMON/-fsycl-targets=spir64/-fsycl-targets=spir64_gen,spir64}"
    DEV="${EXL3_AOT_DEVICE:-bmg-g31}"
    icpx $COMMON -x c++ csrc/exl3_moe_align.sycl -x none -o exl3xpu/_moe_align.so.tmp $LIBS
    mv -f exl3xpu/_moe_align.so.tmp exl3xpu/_moe_align.so
    icpx $AOT -Xsycl-target-backend=spir64_gen "-device $DEV -options \"-vc-codegen -doubleGRF\"" \
      -DEXL3_ACT_LIMIT=10.0f -DEXL3_MOE_SPLIT_ALIGN -DEXL3_MOE_PREFILL_LIB -DEXL3_MOE_NS=moe_pf \
      -x c++ csrc/exl3_moe.n128.sycl -x none -o exl3xpu/_moe_prefill.so.tmp $LIBS \
      -Lexl3xpu -l:_moe_align.so -Wl,-rpath,'$ORIGIN'
    mv -f exl3xpu/_moe_prefill.so.tmp exl3xpu/_moe_prefill.so
    # (one icpx call: a separate -c then link lost the device images)
    icpx $AOT -Xsycl-target-backend=spir64_gen "-device $DEV -options -vc-codegen" \
      -DEXL3_ACT_LIMIT=10.0f -DEXL3_MOE_SPLIT_ALIGN -DEXL3_MOE_EXTERN_PREFILL -x c++ csrc/exl3_moe.n128.sycl -x none \
      -o exl3xpu/_moe_glm.so.tmp $LIBS -Lexl3xpu -l:_moe_prefill.so -l:_moe_align.so -Wl,-rpath,'$ORIGIN'
  else
    icpx $COMMON -DEXL3_ACT_LIMIT=10.0f -x c++ csrc/exl3_moe.n128.sycl -x none -o exl3xpu/_moe_glm.so.tmp $LIBS
  fi
  mv -f exl3xpu/_moe_glm.so.tmp exl3xpu/_moe_glm.so
fi
if [ "$what" = all ] || [ "$what" = native ]; then
  # tier_native: the expert tier's NVMe reader (plain C++ threads; pybind11 from torch's headers, no SYCL)
  PYINC=$(python3 -c "import sysconfig;print(sysconfig.get_paths()['include'])")
  SUF=$(python3 -c "import sysconfig;print(sysconfig.get_config_var('EXT_SUFFIX'))")
  ${CXX:-g++} -O3 -shared -fPIC -std=c++17 -Wall -I"$PYINC" -I"$T/include" csrc/tier_native.cpp -o exl3xpu/tier_native$SUF.tmp -lpthread
  mv -f exl3xpu/tier_native$SUF.tmp exl3xpu/tier_native$SUF
fi
ls -la exl3xpu/*.so
