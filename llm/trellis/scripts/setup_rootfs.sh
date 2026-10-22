#!/usr/bin/env bash
# One-time setup inside the unpacked SGLang XPU root filesystem (CPU only, no GPU): oneDNN, trellis-serve's three
# kernel libraries (built exactly as 0xSero's local-ai-images Dockerfile does), and the editable plugin install.
set -euo pipefail
S=$(dirname "$(readlink -f "$0")")
"$S/in_rootfs.sh" bash -euo pipefail -c '
X=/opt/trellis-serve/xpu/exl3xpu
python3 -c "import sys, torch, sglang; print(sys.version.split()[0], \"torch\", torch.__version__, \"sglang\", sglang.__version__)"
which icpx && icpx --version | head -1
pip install -q --no-deps onednn==2026.0.0 onednn-devel==2026.0.0
cd /opt/trellis-serve/xpu
EXL3_DNNL_DIR="$(python3 -c "import sys; print(sys.prefix)")" EXL3_OUT=$X/_C_sgl.so bash scripts/build_ext.sh
EXL3_MOE_OUT=$X/_moe_sgl.so bash scripts/build_moe.sh
sed -i "s/-march=native/-march=x86-64-v3 -mclflushopt/" scripts/build_ngram_nvme.sh
bash scripts/build_ngram_nvme.sh
test -s $X/_C_sgl.so && test -s $X/_moe_sgl.so && test -s $X/exl3xpu_ngram_nvme.so
readelf -d $X/_C_sgl.so | grep -q libdnnl.so || { echo "_C_sgl.so built without oneDNN"; exit 1; }
pip install -q --no-deps --no-build-isolation -e /opt/trellis-serve/core -e /opt/trellis-serve/xpu
python3 -c "from importlib.metadata import entry_points as e; \
assert any(x.name == \"exl3xpu\" for x in e(group=\"sglang.srt.plugins\")); print(\"exl3xpu sglang plugin registered\")"
EXL3_LIB=$X/_C_sgl.so EXL3_MOE_LIB=$X/_moe_sgl.so python3 -c "import os, torch, importlib.util as u; \
torch.ops.load_library(os.environ[\"EXL3_LIB\"]); torch.ops.load_library(os.environ[\"EXL3_MOE_LIB\"]); \
p = \"/opt/trellis-serve/xpu/exl3xpu/exl3xpu_ngram_nvme.so\"; s = u.spec_from_file_location(\"exl3xpu_ngram_nvme\", p); \
s.loader.exec_module(u.module_from_spec(s)); print(\"exl3xpu libraries load\")"
'
