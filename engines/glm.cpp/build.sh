#!/usr/bin/env bash
# Build glm.cpp on the host toolchain sd-cli uses (DPC++ 2025.3), no PyTorch.
#   ./build.sh [make targets...]     e.g. ./build.sh build/exl3_check
# JOBS defaults to one compile job per 2 GB of free memory (up to 28); builds run at nice 10.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ONEAPI=${ONEAPI:-$HOME/opt/oneapi-2025.3/opt/intel/oneapi}
export CMPLR=$ONEAPI/compiler/2025.3
export UMF=$ONEAPI/umf/1.0/lib
export PATH=$CMPLR/bin:$PATH LD_LIBRARY_PATH=$CMPLR/lib:$UMF${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
cd "$HERE"
if [[ -z "${JOBS:-}" ]]; then
  free_gb=$(awk '/MemAvailable/ { print int($2 / 1048576) }' /proc/meminfo)
  JOBS=$(( free_gb / 2 < 28 ? free_gb / 2 : 28 )); (( JOBS >= 3 )) || JOBS=3
fi
nice -n 10 make -j"$JOBS" "$@"
