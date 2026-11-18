#!/usr/bin/env bash
# Build and run the CPU tests (no GPU, no SYCL): tests/run_cpu.sh [f16|codebook|pack|reference|encode]
# The reference test decodes a tensor of an EXL3 model made by exllamav3 and compares it with a BF16 checkpoint of
# the same architecture; it runs when EXL3_REF and EXL3_BF16 are set (both are model directories).
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
JSON_INC=${JSON_INC:-$HERE/../../engines/sd.cpp/thirdparty}
mkdir -p "$HERE/build"
nice -n 10 g++ -O2 -std=c++20 -march=native -Wall -Wextra -Wno-unused-parameter \
  -I"$HERE/src" -I"$JSON_INC" "$HERE/tests/test_cpu.cpp" -o "$HERE/build/test_cpu.new"
mv "$HERE/build/test_cpu.new" "$HERE/build/test_cpu"
exec "$HERE/build/test_cpu" "$@"
