#!/usr/bin/env bash
# Build libdnnl_h3: oneDNN v3.13.4 + the local SDPA patch, SYCL GPU runtime for Xe2 (Arc Pro B70), no CPU runtime,
# installed to ~/opt/onednn-sycl-graph. Every SYCL binary in this repo links it (sd-cli, the XPU libraries,
# xpu_music3, drama3_batch), so a process has one oneDNN. The patch adds two env switches for experiments:
# ONEDNN_SDPA_CONFIG (override the SDPA micro-kernel config) and ONEDNN_GRAPH_SDPA_PRIMITIVE_ONLY.
# Usage: ./build.sh [src-dir]   (default ~/opt/onednn-src-h3; cloned and patched if missing). CPU only, low parallelism.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC=${1:-$HOME/opt/onednn-src-h3}
PREFIX=${PREFIX:-$HOME/opt/onednn-sycl-graph}
ONEAPI=${ONEAPI:-$HOME/opt/oneapi-2025.3/opt/intel/oneapi}
CMPLR=$ONEAPI/compiler/2025.3
export PATH=$CMPLR/bin:$PATH LD_LIBRARY_PATH=$CMPLR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
if [[ ! -d $SRC ]]; then
  git clone -b v3.13.4 --depth 1 https://github.com/uxlfoundation/oneDNN.git "$SRC"
  git -C "$SRC" am "$HERE"/0001-*.patch
fi
cmake -S "$SRC" -B "$SRC/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" -DONEDNN_LIBRARY_NAME=dnnl_h3 -DONEDNN_CPU_RUNTIME=NONE \
  -DONEDNN_GPU_RUNTIME=SYCL -DONEDNN_GPU_VENDOR=INTEL -DONEDNN_ENABLE_PRIMITIVE_GPU_ISA=XE2 -DONEDNN_BUILD_GRAPH=ON \
  -DONEDNN_ENABLE_WORKLOAD=INFERENCE -DONEDNN_BUILD_TESTS=OFF -DONEDNN_BUILD_EXAMPLES=OFF -DONEDNN_BUILD_DOC=OFF
nice -n 10 cmake --build "$SRC/build" -j "${JOBS:-3}"
cmake --install "$SRC/build"
echo "installed libdnnl_h3 to $PREFIX/lib"
