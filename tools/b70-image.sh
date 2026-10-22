#!/usr/bin/env bash
# Qwen Image 2.1 (Noct Q Anime), Q8_0, on the Arc Pro B70 through sd-cli on ggml's SYCL backend (no Vulkan).
# Samples B70/CPU load as soon as sampling starts, so a CPU fallback shows up immediately.
set -euo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"   # ~/b70/tools
B70=$(dirname "$ROOT")
# Qwen Image 2.1 files in the model store (~/models/image/qwen-image-2.1)
MODELS="${QWEN21_DIR:-$HOME/models/image/qwen-image-2.1}"
# Every run takes the B70 through the b70q queue (b70-lease). Inside a queued job (B70Q_TICKET set: a batch run
# through b70q submit) it runs at once; otherwise it waits its turn. B70_OWNER sets the queue label.
if [[ -z "${B70Q_TICKET:-}" && -z "${B70_IMAGE_LEASED:-}" ]]; then
  export B70_IMAGE_LEASED=1
  exec b70-lease run "${B70_OWNER:-b70-image}" -- "$0" "$@"
fi
# --batch FILE [ARGS]: several stills in one sd-cli process (see b70-image-batch.sh).
if [[ "${1:-}" == --batch ]]; then shift; exec "$ROOT/b70-image-batch.sh" "$@"; fi
SD_CLI_SET="${SD_CLI:-}"
SD_CLI="${SD_CLI:-$B70/engines/sd.cpp/build-sycl/bin/sd-cli}"
# The one sd-cli has --cfg-steps and img_gen --multi-jobs; SD_FAST_BIN only overrides it for --cfg-steps runs.
SD_FAST_BIN="${SD_FAST_BIN:-$SD_CLI}"
DIFFUSION="${DIFFUSION:-}"
LLM="${LLM:-}"
VAE="${VAE:-}"
# Qwen3-VL vision tower; the GGUF text encoder has none, so -r refs need it to reach the prompt. It must match the
# text encoder (Qwen3-VL 8B).
LORA_DIR="${LORA_DIR:-$MODELS/loras}"
VISION="${VISION:-}"
# SYCL sees only the B70 (Level Zero device 0; the 3080 Ti has no Level Zero driver).
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:0}"

PROMPT=""
NEGATIVE=""
OUT="$PWD/b70-image.png"
SEED=""
MODEL=qwen
STEPS=""
CFG=""
WIDTH=""
HEIGHT=""
SAMPLER=euler
SCHEDULER=""
SHIFT=""
SIGMAS=""
INIT=""
STRENGTH=""
REFS=()
REF_PRESET=""
VERBOSE=0
XPU=0
CFG_STEPS=""
EMIT_ARGS=""

usage() {
  cat <<EOF
Usage: $0 -p PROMPT [-m qwen] [-n NEG] [-o OUT] [-r REF]... [--ref-preset NAME]
          [-i INIT --strength S] [--seed N] [--steps N] [--cfg N] [-W N] [-H N]
          [--sampler S] [--scheduler S] [--shift MU] [--sigmas S0,S1,..] [--xpu] [--cfg-steps N]
       $0 --batch JOBS [ARGS]   (several stills in one process, see b70-image-batch.sh)
Qwen Image 2.1 Q8_0 + Qwen3-VL Q8_0 on the B70, euler sampler. <lora:NAME:W> in -p loads loras/NAME.safetensors.
  qwen   (the only model and the default; -m qwen still works) Noct Q Anime (an anime fine-tune of Qwen Image 2.1) + Qwen3-VL-8B, 25 steps CFG 3, simple,
         1024x1536, workflow negative by default. Start prompts with "An anime illustration of".
         -r refs use its edit mode (refer to them as "image 1", "image 2"; sides multiples of 32).
--xpu: run the transformer blocks in the native XPU library (SYCL + oneDNN, no libtorch) with the int8 convrot
weights instead of ggml; prompt <lora:> tags go to it too.
--cfg-steps N: CFG only for the first N steps (or a fraction such as 0.6, or "all"), then one cond-only pass
per step (SD_FAST_BIN picks another sd-cli for it unless SD_CLI is set).
--shift is sd.cpp's mu (= ln of ComfyUI's AuraFlow shift).
Writes OUT and OUT-minus-extension.log. Override weights with DIFFUSION= / LLM= env vars.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--prompt) PROMPT="$2"; shift 2 ;;
    -n|--negative) NEGATIVE="$2"; shift 2 ;;
    -o|--out) OUT="$2"; shift 2 ;;
    -m|--model) MODEL="$2"; shift 2 ;;
    -r|--ref) REFS+=("$2"); shift 2 ;;
    -i|--init) INIT="$2"; shift 2 ;;
    --ref-preset) REF_PRESET="$2"; shift 2 ;;
    --strength) STRENGTH="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --cfg) CFG="$2"; shift 2 ;;
    -W|--width) WIDTH="$2"; shift 2 ;;
    -H|--height) HEIGHT="$2"; shift 2 ;;
    --sampler) SAMPLER="$2"; shift 2 ;;
    --scheduler) SCHEDULER="$2"; shift 2 ;;
    --shift) SHIFT="$2"; shift 2 ;;
    --sigmas) SIGMAS="$2"; shift 2 ;;
    -v|--verbose) VERBOSE=1; shift ;;
    --xpu) XPU=1; shift ;;
    --cfg-steps) CFG_STEPS="$2"; shift 2 ;;
    --emit-args) EMIT_ARGS="$2"; shift 2 ;;  # internal (b70-image-batch.sh): write the resolved run, don't run it
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -n "$PROMPT" ]] || { echo "missing -p PROMPT" >&2; usage; exit 2; }
def_shift="" te=text-encoder
case "$MODEL" in
  # Settings and negative from the Noct Q Anime workflow; the flow shift is picked from the resolution.
  qwen) model_file=noctQAnime_v10-Q8_0.gguf; def_steps=25; def_cfg=3
    def_w=1024 def_h=1536 def_sched=simple llm_file=$te/Qwen3VL-8B-Instruct-Q8_0.gguf
    vision_file=$te/mmproj-Qwen3VL-8B-Instruct-F16.gguf  # -r refs: Qwen 2.1 edit mode, sides multiples of 32
    vae_file=qwen_image_2.1_vae_bf16.safetensors
    def_vae_tile=384:0.125  # 4.8 s decode vs 8.5 s at sd.cpp's default 0.5 overlap, no visible seams
    def_neg="low quality, low resolution, blurry, jpeg artifacts, washed-out colors, sloppy lines, messy line art, extra fingers, missing fingers, badly drawn hands, deformed anatomy, 3d render, photograph, photorealistic" ;;
  *) echo "unknown model: $MODEL" >&2; usage; exit 2 ;;
esac
DIFFUSION="${DIFFUSION:-$MODELS/$model_file}"
LLM="${LLM:-$MODELS/$llm_file}"
VAE="${VAE:-$MODELS/$vae_file}"
VISION="${VISION:-$MODELS/$vision_file}"
NEGATIVE="${NEGATIVE:-$def_neg}"
VAE_TILE="${VAE_TILE-${def_vae_tile:-}}"
STEPS="${STEPS:-$def_steps}"
CFG="${CFG:-$def_cfg}"
WIDTH="${WIDTH:-$def_w}"
HEIGHT="${HEIGHT:-$def_h}"
SCHEDULER="${SCHEDULER:-$def_sched}"
SHIFT="${SHIFT:-$def_shift}"
[[ -n "$SEED" ]] || SEED=$RANDOM
[[ -n "$CFG_STEPS" && -z "$SD_CLI_SET" ]] && SD_CLI="$SD_FAST_BIN"
for f in "$SD_CLI" "$DIFFUSION" "$LLM" "$VAE" ${INIT:+"$INIT"} ${REFS[@]+"${REFS[@]}"}; do
  [[ -e "$f" ]] || { echo "not found: $f" >&2; exit 1; }
done
[[ "$OUT" == /* ]] || OUT="$PWD/$OUT"
mkdir -p "$(dirname "$OUT")"
LOG="${OUT%.*}.log"

extra=()
[[ "$VERBOSE" -eq 1 ]] && extra+=(-v)
[[ -n "$NEGATIVE" ]] && extra+=(-n "$NEGATIVE")
# VAE_TILING=0 decodes in one piece (faster when VRAM allows)
[[ "${VAE_TILING:-1}" == 1 ]] && extra+=(--vae-tiling)
# VAE_TILE=SIZE:OVERLAP, e.g. 768:0.25 (sd.cpp default tiles overlap by 0.5)
if [[ -n "${VAE_TILE:-}" ]]; then extra+=(--vae-tile-size "${VAE_TILE%%:*}" --vae-tile-overlap "${VAE_TILE##*:}"); fi
[[ -n "$INIT" ]] && extra+=(-i "$INIT")
[[ -n "$STRENGTH" ]] && extra+=(--strength "$STRENGTH")
[[ -n "$SHIFT" ]] && extra+=(--flow-shift "$SHIFT")
[[ -n "$SIGMAS" ]] && extra+=(--sigmas "$SIGMAS")
[[ -n "$REF_PRESET" ]] && extra+=(--ref-image-args "preset=$REF_PRESET")
for r in ${REFS[@]+"${REFS[@]}"}; do extra+=(-r "$r"); done
[[ ${#REFS[@]} -gt 0 && -f "$VISION" ]] && extra+=(--llm_vision "$VISION")
[[ -n "$CFG_STEPS" ]] && extra+=(--cfg-steps "$CFG_STEPS")
if [[ "$XPU" -eq 1 ]]; then
  XPU_DIR=$B70/engines/sd.cpp/xpu_sdpa
  export SD_XPU_QWEN21_LIB="${SD_XPU_QWEN21_LIB:-$XPU_DIR/libxpu_qwen21.so}" SD_XPU_QWEN21_VERBOSE=1
  # keep JIT-compiled SYCL kernels across runs
  export SYCL_CACHE_PERSISTENT=1
  export SD_XPU_QWEN21_WEIGHTS="${SD_XPU_QWEN21_WEIGHTS:-$MODELS/noctQAnime_v10_int8_convrot.safetensors}"
  # <lora:NAME:W> tags -> loras/NAME.safetensors@W for the XPU blocks
  SD_XPU_QWEN21_LORA=$(grep -oE '<lora:[^:>]+(:[0-9.]+)?>' <<<"$PROMPT" |
    sed -E "s#<lora:([^:>]+):?([0-9.]*)>#$LORA_DIR/\1.safetensors@\2#; s/@$/@1/" | paste -sd, -) || true
  export SD_XPU_QWEN21_LORA
fi

echo "b70 $MODEL$([[ $XPU -eq 1 ]] && echo +xpu) seed=$SEED steps=$STEPS cfg=$CFG${CFG_STEPS:+ cfg-steps=$CFG_STEPS} ${WIDTH}x${HEIGHT} $(basename "$DIFFUSION") -> $OUT (log $LOG)"

args=(
  --diffusion-model "$DIFFUSION"
  --llm "$LLM"
  --vae "$VAE"
  --lora-model-dir "$LORA_DIR"
  -p "$PROMPT"
  -o "$OUT"
  --seed "$SEED"
  --steps "$STEPS"
  --cfg-scale "$CFG"
  -W "$WIDTH"
  -H "$HEIGHT"
  --sampling-method "$SAMPLER"
  --scheduler "$SCHEDULER"
  --diffusion-fa
  "${extra[@]}"
)
if [[ -n "$EMIT_ARGS" ]]; then
  # NUL-separated: OUT, LOG, then KEY=VALUE env lines, "--", then the sd-cli arguments.
  {
    printf '%s\0' "$OUT" "$LOG"
    for v in ONEAPI_DEVICE_SELECTOR SD_XPU_QWEN21_LIB SD_XPU_QWEN21_VERBOSE SYCL_CACHE_PERSISTENT \
      SD_XPU_QWEN21_WEIGHTS SD_XPU_QWEN21_LORA; do
      [[ -n "${!v+x}" ]] && printf '%s\0' "$v=${!v}"
    done
    printf '%s\0' -- "${args[@]}"
  } > "$EMIT_ARGS"
  exit 0
fi

"$SD_CLI" "${args[@]}" > "$LOG" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null' INT TERM

# Confirm placement and load right away: weights must be in VRAM, the B70 compute engine busy.
until grep -aq 'total params memory size' "$LOG" 2>/dev/null || ! kill -0 $PID 2>/dev/null; do sleep 0.5; done
grep -aE 'level_zero:gpu|total params memory size|on XPU' "$LOG" | sed 's/^\[INFO *\] [^-]*- //'
# Watch from the first denoising step (after the lazy diffusion-weight upload).
until awk '/generating image/ { g = 1 } g && /s\/it|it\/s/ { f = 1 } END { exit !f }' "$LOG" 2>/dev/null || ! kill -0 $PID 2>/dev/null; do sleep 0.5; done
kill -0 $PID 2>/dev/null && "$ROOT/b70-watch.sh" $PID 3 2 || true

wait $PID && status=0 || status=$?
grep -aE 'completed, taking|generate_image completed|ERROR|xpu_qwen21\] (loaded|LoRA|error)' "$LOG" | grep -v 'loading tensors' | sed 's/^\[[A-Z]* *\] [^-]*- //'
[[ $status -eq 0 ]] && echo "saved $OUT" || echo "sd-cli failed ($status), see $LOG" >&2
exit $status
