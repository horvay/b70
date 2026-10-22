#!/usr/bin/env bash
# Run a command inside the unpacked lmsysorg/sglang:v0.5.20-xpu root filesystem with bubblewrap (no Docker, no root).
#   in_rootfs.sh [--gpu] <command...>
# Without --gpu no GPU device is passed in (setup, builds). With --gpu, /dev/dri is bound for serving.
# trellis-serve is bound at /opt/trellis-serve, the models folder read-only at /models, scripts/pyhooks at /opt/pyhooks
# (on PYTHONPATH only where the caller sets it, as serve_b70.sh does), and the cold expert store directory
# (QWEN_NVX, default ~/models/qwen38-store, on NVMe) at /nvx when it exists.
set -euo pipefail
T=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
ROOT=$T/image/rootfs
DEV=(--dev /dev)
if [ "${1:-}" = "--gpu" ]; then
    shift
    DEV=(--dev-bind /dev /dev)
fi
mkdir -p "$T/run" "$T/cache"
NVX=${QWEN_NVX:-$HOME/models/qwen38-store}
IMGENV=()  # the image's own environment (oneAPI paths, the venv), as Docker would set it
while IFS= read -r kv; do IMGENV+=(--setenv "${kv%%=*}" "${kv#*=}"); done < <(python3 -c '
import json, sys
for e in json.load(open(sys.argv[1]))["config"].get("Env", []): print(e)' "$T/image/image-config.json")
EXTRA=()   # extra bwrap arguments from the caller (serve_b70.sh: --setenv pairs), shell-quoted
if [ -n "${BWRAP_EXTRA:-}" ]; then eval "EXTRA=($BWRAP_EXTRA)"; fi
exec bwrap \
    --bind "$ROOT" / \
    "${DEV[@]}" \
    --proc /proc \
    --ro-bind /sys /sys \
    --tmpfs /tmp \
    --ro-bind /etc/resolv.conf /etc/resolv.conf \
    --bind "$T/trellis-serve" /opt/trellis-serve \
    --ro-bind "$T/models" /models \
    --bind "$T/run" /run/trellis \
    --ro-bind "$T/scripts/pyhooks" /opt/pyhooks \
    --bind "$T/cache" /root/.cache \
    --bind-try "$NVX" /nvx \
    --setenv HOME /root \
    "${IMGENV[@]}" \
    "${EXTRA[@]}" \
    --unshare-pid --die-with-parent \
    --chdir /opt/trellis-serve/xpu \
    "$@"
