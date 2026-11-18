#!/usr/bin/env bash
# Build exl3conv's SYCL binaries for the Arc Pro B70 (AOT for bmg-g31). CPU only; touches no GPU.
#
#   ./build.sh [target ...]     targets: exl3conv, exl3eval, test_gpu, test_quant, test_forward, test_ngram, test_qwen4exp_forward, test_qwen35_forward, test_wide, vit_variants (default: exl3conv exl3eval)
#
# Outputs go to build/. Environment overrides (defaults under ~/opt, see notes/toolchain-and-reuse.md):
#   ONEAPI   oneAPI 2025.3 root        MKLROOT   oneMKL 2025.3        ONEDNN   oneDNN libdnnl_h3
#   JSON_INC nlohmann/json include dir (default: the sd.cpp submodule's thirdparty/)
#   AOT=0    also skip AOT and JIT-compile on first run (faster builds while iterating)
#   BUILD    output folder (default build/; a dev build elsewhere leaves running jobs' binaries alone)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ONEAPI=${ONEAPI:-$HOME/opt/oneapi-2025.3/opt/intel/oneapi}
CMPLR=$ONEAPI/compiler/2025.3
UMF=$ONEAPI/umf/1.0/lib
JSON_INC=${JSON_INC:-$HERE/../../engines/sd.cpp/thirdparty}
export PATH=$CMPLR/bin:$PATH LD_LIBRARY_PATH=$CMPLR/lib:$UMF${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

if [[ ${AOT:-1} == 0 ]]; then
  TARGETS=(-fsycl-targets=spir64)
else
  TARGETS=(-fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device bmg-g31")
fi
CXXFLAGS=(-O2 -std=c++20 -fsycl "${TARGETS[@]}" -fsycl-device-code-split=per_kernel -fp-model=precise -fno-sycl-id-queries-fit-in-int
          -DSYCL_DISABLE_FSYCL_SYCLHPP_WARNING -Wall -Wno-unused-parameter -Wno-unknown-pragmas
          -I"$HERE/src" -I"$JSON_INC")
LDFLAGS=(-L"$UMF" -Wl,--no-as-needed -lumf -Wl,--as-needed -Wl,-rpath,"$CMPLR/lib:$UMF")
# oneMKL: SYCL BLAS on the card (ILP64), host LAPACK for the small Cholesky diagonal blocks.
MKLROOT=${MKLROOT:-$HOME/opt/onemkl-2025.3-deb/opt/intel/oneapi/mkl/2025.3}
MKL_CXX=(-DMKL_ILP64 -I"$MKLROOT/include")
MKL_LD=(-L"$MKLROOT/lib" -lmkl_sycl_blas -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl
        -Wl,-rpath,"$MKLROOT/lib")

BUILD=${BUILD:-$HERE/build}
build() {  # build <name> <source> [mkl]
  local out="$BUILD/$1" extra_cxx=() extra_ld=()
  [[ ${3:-} == mkl ]] && { extra_cxx=("${MKL_CXX[@]}"); extra_ld=("${MKL_LD[@]}"); }
  mkdir -p "$BUILD"
  echo "building $1"
  nice -n 10 icpx "${CXXFLAGS[@]}" "${extra_cxx[@]}" "$2" -o "$out.new" "${LDFLAGS[@]}" "${extra_ld[@]}"
  mv "$out.new" "$out"             # rename, so a running process keeps its mapped copy
  if ldd "$out" | awk '{print $1}' | grep -Ei 'vulkan|torch|c10'; then echo "unexpected Vulkan/torch dependency in $1" >&2; exit 1; fi
  if ldd "$out" | grep "not found"; then echo "missing shared libraries for $1" >&2; exit 1; fi
}

targets=("$@"); [[ ${#targets[@]} -gt 0 ]] || targets=(exl3conv exl3eval exl3measure exl3opt)
for t in "${targets[@]}"; do
  case $t in
    test_gpu) build test_gpu "$HERE/tests/test_gpu.cpp" ;;
    vit_variants) build vit_variants "$HERE/tests/vit_variants.cpp" ;;
    test_quant) build test_quant "$HERE/tests/test_quant.cpp" mkl ;;
    exl3conv) build exl3conv "$HERE/src/convert.cpp" mkl ;;
    test_forward) build test_forward "$HERE/tests/test_forward.cpp" mkl ;;
    exl3eval) build exl3eval "$HERE/src/eval.cpp" mkl ;;
    quant_dflash2) build quant_dflash2 "$HERE/src/quant_dflash2.cpp" mkl ;;
    exl3measure) build exl3measure "$HERE/src/measure.cpp" mkl ;;
    exl3opt) mkdir -p "$BUILD"; echo "building exl3opt"
             nice -n 10 g++ -O2 -std=c++20 -I"$JSON_INC" "$HERE/src/optimize.cpp" -o "$BUILD/exl3opt.new"
             mv "$BUILD/exl3opt.new" "$BUILD/exl3opt" ;;
    test_generate) build test_generate "$HERE/tests/test_generate.cpp" mkl ;;
    test_ngram) build test_ngram "$HERE/tests/test_ngram.cpp" ;;
    test_qwen4exp_forward) build test_qwen4exp_forward "$HERE/tests/test_qwen4exp_forward.cpp" mkl ;;
    test_qwen35_forward) build test_qwen35_forward "$HERE/tests/test_qwen35_forward.cpp" mkl ;;
    test_wide) build test_wide "$HERE/tests/test_wide.cpp" mkl ;;
    *) echo "unknown target $t" >&2; exit 1 ;;
  esac
done
