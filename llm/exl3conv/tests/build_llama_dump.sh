#!/usr/bin/env bash
# Build tests/llama_dump against engines/llama.cpp's SYCL build (build_sycl.sh there; its static libraries, nothing
# rebuilt in that tree). CPU only. Output: build/llama_dump.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
LC=${LLAMA_CPP:-$HERE/../../engines/llama.cpp}
B=$LC/build-sycl
ONEAPI=${ONEAPI:-$HOME/opt/oneapi-2025.3/opt/intel/oneapi}
CMPLR=$ONEAPI/compiler/2025.3
UMF=$ONEAPI/umf/1.0/lib
MKL=${MKLROOT:-$HOME/opt/onemkl-2025.3-deb/opt/intel/oneapi/mkl/2025.3}/lib
DNN=${ONEDNN:-$HOME/opt/onednn-sycl-graph}/lib
export PATH=$CMPLR/bin:$PATH
mkdir -p "$HERE/build"
nice -n 10 icpx -O2 -std=c++17 -I"$LC/include" -I"$LC/ggml/include" "$HERE/tests/llama_dump.cpp" -o "$HERE/build/llama_dump" \
  -static-intel -Wl,--disable-new-dtags -L"$UMF" -Wl,--no-as-needed -lumf -Wl,--as-needed \
  -fsycl -fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device bmg-g31" -fiopenmp \
  -Wl,-rpath,"$CMPLR/lib:$UMF:$MKL:$DNN" \
  "$B/src/libllama.a" "$B/ggml/src/libggml.a" "$B/ggml/src/libggml-cpu.a" "$B/ggml/src/ggml-sycl/libggml-sycl.a" \
  "$B/ggml/src/libggml-base.a" "$CMPLR/lib/libiomp5.so" -lm "$CMPLR/lib/libsycl.so" /usr/lib/libze_loader.so \
  "$DNN/libdnnl_h3.so.3.13" -ldl "$MKL/libmkl_sycl_blas.so" "$MKL/libmkl_intel_ilp64.so" "$MKL/libmkl_sequential.so" \
  "$MKL/libmkl_core.so" -lpthread -lsycl -lOpenCL
echo "built $HERE/build/llama_dump"
