#!/usr/bin/env bash
# Albedo (de-lit base colour) of pictures on the Arc Pro B70: the Spellcast3D albedo add-on for Qwen Image 2.1
# (LoRA + fixed prompt embedding + fine-tuned VAE decoder, one transformer pass per picture) through
# sd-cli --mode albedo, with the transformer blocks in the native XPU library (int8 base, the add-on's LoRA merged
# into the blocks at load) and the layers around them on ggml SYCL (LoRA at runtime).
#
# Usage: b70-albedo.sh IN [IN...] -o OUTDIR
# Writes OUTDIR/<stem>.png (albedo, gamma 2.2, at the input's size, the input's alpha kept) and
# OUTDIR/<stem>_lumkeep.png (the input's colour with the albedo's luminance, in linear light). RGBA inputs are
# composited on mid grey first; pictures are scaled down to fit 1024 px and padded to multiples of 32 for the model.
# Inputs sharing a file name get their folder's name in front (pa_v3/pre.png -> pa_v3_pre). Log: OUTDIR/albedo.log.
# Model card and licence (Qwen research licence, non-commercial): ~/models/image/qwen-image-2.1/albedo/.
set -euo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"   # ~/b70/tools
B70=$(dirname "$ROOT")
MODELS="${QWEN21_DIR:-$HOME/models/image/qwen-image-2.1}"
ADDON="${ALBEDO_DIR:-$MODELS/albedo}"
if [[ -z "${B70Q_TICKET:-}" && -z "${B70_ALBEDO_LEASED:-}" ]]; then
  export B70_ALBEDO_LEASED=1
  exec b70-lease run "${B70_OWNER:-albedo}" --note "albedo $#" -- "$0" "$@"
fi

SD_CLI="${SD_CLI:-$B70/engines/sd.cpp/build-sycl/bin/sd-cli}"
DIFFUSION="${DIFFUSION:-$MODELS/qwen_image_2.1_int8_convrot.safetensors}"
VAE="${VAE:-$MODELS/qwen_image_2.1_vae_bf16.safetensors}"
# The context needs its text encoder named, though the add-on's fixed prompt embedding means it never runs.
LLM="${LLM:-$MODELS/text-encoder/Qwen3VL-8B-Instruct-Q8_0.gguf}"
OUTDIR=""
INPUTS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--out) OUTDIR="$2"; shift 2 ;;
    -h|--help) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) INPUTS+=("$(readlink -f "$1")"); shift ;;
  esac
done
[[ -n "$OUTDIR" && ${#INPUTS[@]} -gt 0 ]] || { echo "usage: $0 IN [IN...] -o OUTDIR" >&2; exit 2; }
for f in "$SD_CLI" "$DIFFUSION" "$VAE" "$LLM" "$ADDON"/albedo_{lora,condition,vae}.safetensors "${INPUTS[@]}"; do
  [[ -e "$f" ]] || { echo "not found: $f" >&2; exit 1; }
done
mkdir -p "$OUTDIR"
OUTDIR=$(readlink -f "$OUTDIR")
LOG="$OUTDIR/albedo.log"
WORK=$(mktemp -d "${TMPDIR:-/tmp}/b70-albedo.XXXXXX")
trap 'rm -rf "$WORK"' EXIT

python3 -I "$ROOT/b70-albedo-img.py" prep "$WORK" "${INPUTS[@]}"

XPU_DIR=$B70/engines/sd.cpp/xpu_sdpa
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:0}" SYCL_CACHE_PERSISTENT=1
export SD_XPU_QWEN21_LIB="${SD_XPU_QWEN21_LIB:-$XPU_DIR/libxpu_qwen21.so}" SD_XPU_QWEN21_VERBOSE=1
export SD_XPU_QWEN21_WEIGHTS="$DIFFUSION"
# the blocks' half of the add-on's LoRA, merged by the XPU library when it loads (sd_albedo checks it)
export SD_XPU_QWEN21_LORA="$ADDON/albedo_lora.safetensors"

"$SD_CLI" -M albedo \
  --diffusion-model "$DIFFUSION" --vae "$VAE" --llm "$LLM" --diffusion-fa \
  --albedo-lora "$ADDON/albedo_lora.safetensors" \
  --albedo-condition "$ADDON/albedo_condition.safetensors" \
  --albedo-decoder "$ADDON/albedo_vae.safetensors" \
  -i "$WORK/in" -o "$WORK/out" > "$LOG" 2>&1 && status=0 || status=$?
grep -aE 'xpu_qwen21\] (loaded|LoRA)|albedo (LoRA|prompt|decoder|[0-9]+x)|albedo /|ERROR' "$LOG" | sed 's/^\[[A-Z]* *\] [^-]*- //' || true
[[ $status -eq 0 ]] || { echo "sd-cli failed ($status), see $LOG" >&2; exit $status; }

python3 -I "$ROOT/b70-albedo-img.py" post "$WORK" "$OUTDIR"
