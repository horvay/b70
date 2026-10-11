#!/usr/bin/env bash
# One-time setup (no Docker, no root, no GPU needed):
#   1. pull the pinned SGLang XPU image (torch 2.13 XPU, Triton XPU, oneAPI 2026.0 / icpx) and unpack it to image/rootfs
#   2. install the two extra Python packages exllamav3 needs into pylib/ (on PYTHONPATH inside the rootfs)
#   3. build the SYCL/ESIMD kernel libraries (xpu/build.sh)
# Needs: bubblewrap (bwrap), python3, ~25 GB free for the image.
set -euo pipefail
W=$(dirname "$(dirname "$(readlink -f "$0")")")
ROOT=${GLM_B70_ROOTFS:-$W/image/rootfs}
DIGEST=sha256:e4e10d66de374e6e8cc88136e8dbfc3182787070df8a48e581fb007cc7a51c36   # lmsysorg/sglang XPU build
if [ ! -x "$ROOT/usr/bin/python3" ] && [ ! -x "$ROOT/opt/venv/bin/python3" ]; then
    mkdir -p "$(dirname "$ROOT")"
    python3 "$W/scripts/oci_pull.py" registry-1.docker.io lmsysorg/sglang "$DIGEST" "$ROOT"
fi
[ -f "$W/exllamav3/exllamav3/__init__.py" ] || { echo "exllamav3/ missing: git submodule update --init"; exit 1; }
"$W/scripts/in_rootfs.sh" pip install -q --no-deps --target /w/pylib llguidance==1.9.1 marisa-trie==1.4.1
"$W/scripts/in_rootfs.sh" bash /w/xpu/build.sh
echo "setup done"
