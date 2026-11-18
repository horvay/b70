#!/usr/bin/env bash
# Build and run tests/gguf_check.cpp (CPU only, no SYCL, runs no model): tests/gguf_check.sh <gguf dir> <ref dir>
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
JSON_INC=${JSON_INC:-$HERE/../../engines/sd.cpp/thirdparty}
mkdir -p "$HERE/build"
nice -n 10 g++ -O2 -std=c++20 -march=native -Wall -Wextra -Wno-unused-parameter -pthread \
  -I"$HERE/src" -I"$JSON_INC" "$HERE/tests/gguf_check.cpp" -o "$HERE/build/gguf_check.new"
mv "$HERE/build/gguf_check.new" "$HERE/build/gguf_check"
exec nice -n 10 "$HERE/build/gguf_check" "$@"
