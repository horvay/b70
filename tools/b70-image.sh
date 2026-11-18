#!/usr/bin/env bash
# Qwen Image 2.1 (Noct Q Anime), MOSS-Image (Base, Dirty-Moss, MOSS-FET, Photonix24) and MiniMax-H3-Image on the Arc
# Pro B70, through sd-cli on ggml's SYCL backend (no Vulkan). MOSS runs on the MOSS-capable build in engines/sd.cpp-moss,
# MiniMax-H3-Image on the upstream build in engines/sd.cpp-h3img.
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
LORA_DIR_SET="${LORA_DIR:-}"
VISION="${VISION:-}"
# SYCL sees only the B70 (Level Zero device 0; the 3080 Ti has no Level Zero driver).
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:0}"

PROMPT=""
NEGATIVE=""
OUT="$PWD/b70-image.png"
SEED=""
BATCH_COUNT=1
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
TURBO=0
HIRES=0

usage() {
  cat <<EOF
Usage: $0 -p PROMPT [-m qwen|moss|dirty-moss|moss-fet|photonix|h3] [-n NEG] [-o OUT] [-r REF]... [--ref-preset NAME]
          [-i INIT --strength S] [--seed N] [--steps N] [--cfg N] [-W N] [-H N]
          [--sampler S] [--scheduler S] [--shift MU] [--sigmas S0,S1,..] [--xpu] [--cfg-steps N]
          [--turbo] [--hires] [-b N]
       $0 --batch JOBS [ARGS]   (several stills in one process, see b70-image-batch.sh)
Qwen Image 2.1 Q8_0 + Qwen3-VL Q8_0 on the B70, euler sampler. <lora:NAME:W> in -p loads loras/NAME.safetensors.
  qwen   (the only model and the default; -m qwen still works) Noct Q Anime (an anime fine-tune of Qwen Image 2.1) + Qwen3-VL-8B, 25 steps CFG 3, simple,
         1024x1536, workflow negative by default. Start prompts with "An anime illustration of".
         -r refs use its edit mode (refer to them as "image 1", "image 2"; sides multiples of 32).
  moss        MOSS-Image Base Q8_0 + Qwen3.5-9B (Heretic) Q8_0, FLUX.2 VAE. 25 steps CFG 3, 1024x1536. Text-to-image
              only (no -r refs).
  dirty-moss  Dirty-Moss (photographic realism merge of MOSS-Image), same files otherwise. 25 steps CFG 2.
  moss-fet    MOSS-FET v0.6 (anime full fine-tune of MOSS-Image, civitai 3024984). 25 steps CFG 3. Its author suggests
              "from side, dynamic angle" in prompts.
  photonix    MOSS Photonix24 (versatile cinematic merge, civitai 3046358). Already step-distilled: 8 steps CFG 1, and
              --turbo adds nothing.
  The fine-tunes ship bf16 only, which sd-cli reads as is (--diffusion-model); with --xpu that supplies only the few
  non-block weights.
  h3          MiniMax-H3-Image (33B, Comfy-Org's pruned int8 convrot DiT with the six-step PDD heads) + the MiniMax-H3
              Qwen3-VL-32B text encoder (shared with the H3 video model), H3-Image 2D VAE. Always 6 steps, Euler, CFG 1,
              --sigmas minimax-h3-image; no negative prompt. Text-to-image and editing with up to 16 -r references,
              named <Picture 1>, <Picture 2>... in the prompt. Native 16 MP (4096x4096; 4736x3552, 5024x3360, 5472x3072,
              6272x2688 and their portraits); works from about 4 MP. Sides are multiples of 32. Default 2048x2048.
              The transformer blocks always run in libxpu_h3 (int8 W8A8, SD_XPU_H3_INT8=1); <lora:> tags go to it.
              --batch runs every still in one process (sd-cli --multi-jobs): the blocks and encoders load once.
--turbo (MOSS): the 8-step DMD2 LoRA (loras/moss_base_8step_dmd2), 8 steps, CFG 1.
--hires (MOSS): a second pass at 1.5x (Lanczos, 8 steps, denoise 0.15). HIRES_SCALE, HIRES_STEPS and
HIRES_STRENGTH override those.
--xpu: run the transformer blocks in the native XPU library (SYCL + oneDNN, no libtorch) with the int8 convrot
weights instead of ggml; prompt <lora:> tags go to it too. Qwen: libxpu_qwen21. MOSS: libxpu_moss with
MODEL-int8-convrot.safetensors (MOSS-Image-Base, Dirty-Moss, MOSS-FET, MOSS-Photonix24).
TE_OFFLOAD=1 with MOSS --xpu keeps the text encoder in RAM between uses (~2 GiB less peak VRAM, +6.5 s an image).
--cfg-steps N: CFG only for the first N steps (or a fraction such as 0.6, or "all"), then one cond-only pass
per step (SD_FAST_BIN picks another sd-cli for it unless SD_CLI is set).
-b N: N images with seeds SEED..SEED+N-1, saved as OUT-minus-extension_0.png .. _N-1.png, rendered in turn.
SD_BATCH_SAMPLING=1 samples MOSS images together (up to SD_MAX_BATCH=4 per pass); on the B70 that measured no faster.
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
    -b|--batch-count) BATCH_COUNT="$2"; shift 2 ;;
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
    --turbo) TURBO=1; shift ;;
    --hires) HIRES=1; shift ;;
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
  # MOSS-Image: settings from the Dirty-Moss page and the upstream pipeline (default discrete scheduler, shift 5).
  moss|dirty-moss|moss-fet|photonix)
    MODELS="${MOSS_DIR:-$HOME/models/image/moss-image}"
    [[ -z "$SD_CLI_SET" ]] && SD_CLI="$B70/engines/sd.cpp-moss/build-sycl/bin/sd-cli" SD_FAST_BIN="$SD_CLI"
    # moss_name: the file stem (NAME-Q8_0.gguf or NAME.safetensors for ggml, NAME-int8-convrot.safetensors for --xpu)
    def_steps=25 def_w=1024 def_h=1536 def_sched="" distilled=0
    case "$MODEL" in
      moss) moss_name=MOSS-Image-Base model_file=MOSS-Image-Base-Q8_0.gguf def_cfg=3 ;;
      dirty-moss) moss_name=Dirty-Moss model_file=Dirty-Moss-Q8_0.gguf def_cfg=2 ;;
      moss-fet) moss_name=MOSS-FET model_file=MOSS-FET.safetensors def_cfg=3 ;;
      photonix) moss_name=MOSS-Photonix24 model_file=MOSS-Photonix24.safetensors def_cfg=1 def_steps=8 distilled=1 ;;
    esac
    llm_file=$te/MOSS-Image-Text-Encoder-Heretic.Q8_0.gguf vision_file=""
    vae_file=vae/flux2-vae.safetensors tokenizer_file=tokenizer/tokenizer_qwen3.5.json
    def_neg="low quality, blurry, jpeg artifacts, watermark, signature"
    [[ ${#REFS[@]} -eq 0 ]] || { echo "$MODEL is text-to-image only: no -r refs" >&2; exit 2; }
    if [[ "$TURBO" -eq 1 && "$distilled" -eq 0 ]]; then
      def_steps=8 def_cfg=1
      PROMPT="$PROMPT<lora:moss_base_8step_dmd2:1>"
    fi ;;
  h3)
    MODELS="${H3_IMAGE_DIR:-$HOME/models/image/minimax-h3-image}"
    # the installed copy (build_sycl.sh renames it into place when a build is done), not the build tree's file
    [[ -z "$SD_CLI_SET" ]] && SD_CLI="$B70/engines/sd.cpp-h3img/bin/sd-cli" SD_FAST_BIN="$SD_CLI"
    model_file=diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors
    llm_file=$te/qwen3vl_32b_minimax_h3-Q4_K_M.gguf vision_file=""   # the GGUF holds the vision tower
    vae_file=vae/minimax_h3_image_vae_fp16.safetensors
    def_steps=6 def_cfg=1 def_w=2048 def_h=2048 def_sched="" def_neg=""
    # the PDD heads are trained for exactly this schedule; CFG is distilled away (a negative prompt does nothing)
    [[ -z "$STEPS" || "$STEPS" == 6 ]] || echo "h3: always 6 steps (asked for $STEPS)" >&2
    [[ -z "$CFG" || "$CFG" == 1 ]] || echo "h3: always CFG 1 (asked for $CFG)" >&2
    STEPS=6 CFG=1 NEGATIVE="" SAMPLER=euler SIGMAS="${SIGMAS:-minimax-h3-image}" XPU=1
    [[ ${#REFS[@]} -le 16 ]] || { echo "h3 takes at most 16 -r references" >&2; exit 2; } ;;
  *) echo "unknown model: $MODEL" >&2; usage; exit 2 ;;
esac
[[ -n "${moss_name:-}" || ( "$TURBO" -eq 0 && "$HIRES" -eq 0 ) ]] ||
  { echo "--turbo and --hires are for the MOSS models" >&2; exit 2; }
LORA_DIR="${LORA_DIR_SET:-$MODELS/loras}"
TOKENIZER="${tokenizer_file:+$MODELS/$tokenizer_file}"
DIFFUSION="${DIFFUSION:-$MODELS/$model_file}"
LLM="${LLM:-$MODELS/$llm_file}"
VAE="${VAE:-$MODELS/$vae_file}"
VISION="${VISION:-${vision_file:+$MODELS/$vision_file}}"
[[ "$MODEL" == h3 ]] || NEGATIVE="${NEGATIVE:-$def_neg}"
VAE_TILE="${VAE_TILE-${def_vae_tile:-}}"
STEPS="${STEPS:-$def_steps}"
CFG="${CFG:-$def_cfg}"
WIDTH="${WIDTH:-$def_w}"
HEIGHT="${HEIGHT:-$def_h}"
SCHEDULER="${SCHEDULER:-$def_sched}"
SHIFT="${SHIFT:-$def_shift}"
[[ -n "$SEED" ]] || SEED=$RANDOM
[[ -n "$CFG_STEPS" && -z "$SD_CLI_SET" ]] && SD_CLI="$SD_FAST_BIN"
for f in "$SD_CLI" "$DIFFUSION" "$LLM" "$VAE" ${TOKENIZER:+"$TOKENIZER"} ${INIT:+"$INIT"} ${REFS[@]+"${REFS[@]}"}; do
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
[[ ${#REFS[@]} -gt 0 && -n "$VISION" && -f "$VISION" ]] && extra+=(--llm_vision "$VISION")
[[ -n "$CFG_STEPS" ]] && extra+=(--cfg-steps "$CFG_STEPS")
[[ -n "$SCHEDULER" ]] && extra+=(--scheduler "$SCHEDULER")
# MOSS: the Qwen3.5 vocabulary comes from the external tokenizer.json; direct VAE convolutions (Oct 29 bench)
[[ -n "$TOKENIZER" ]] && extra+=(--tokenizer "$TOKENIZER" --vae-conv-direct)
[[ "$HIRES" -eq 1 ]] && extra+=(--hires --hires-upscaler Lanczos --hires-scale "${HIRES_SCALE:-1.5}"
  --hires-steps "${HIRES_STEPS:-8}" --hires-denoising-strength "${HIRES_STRENGTH:-0.15}")
if [[ "$XPU" -eq 1 ]]; then
  XPU_DIR=$B70/engines/sd.cpp/xpu_sdpa
  # keep JIT-compiled SYCL kernels across runs
  export SYCL_CACHE_PERSISTENT=1
  # <lora:NAME:W> tags -> loras/NAME.safetensors@W for the XPU blocks
  xpu_loras=$(grep -oE '<lora:[^:>]+(:-?[0-9.]+)?>' <<<"$PROMPT" |
    sed -E "s#<lora:([^:>]+):?(-?[0-9.]*)>#$LORA_DIR/\1.safetensors@\2#; s/@$/@1/" | paste -sd, -) || true
  if [[ "$MODEL" == h3 ]]; then
    # all 50 blocks in libxpu_h3 with the checkpoint's int8 weights; ggml keeps the embeddings and the final layer.
    # The text encoder's weights stay in RAM and visit the card only while it encodes (--offload-to-cpu), so the
    # 21 GB of blocks fit; the blocks are released again before the VAE decode.
    export SD_XPU_H3_LIB="${SD_XPU_H3_LIB:-$XPU_DIR/libxpu_h3.so}" SD_XPU_H3_VERBOSE=1
    export SD_XPU_H3_WEIGHTS="${SD_XPU_H3_WEIGHTS:-$DIFFUSION}" SD_XPU_H3_INT8="${SD_XPU_H3_INT8:-1}"
    export SD_XPU_H3_LORA="$xpu_loras"
    # the tags are the block library's alone (the ggml graph has no blocks to apply them to)
    PROMPT=$(sed -E 's/<lora:[^>]*>//g' <<<"$PROMPT")
    extra+=(--offload-to-cpu --rng cpu)
    # H3_VAE_RESIDENT=1: the 1.1 GB VAE stays on the card (not offloaded with the text encoder), so a batch does not
    # copy it over for every reference encode and decode
    [[ ${H3_VAE_RESIDENT:-0} == 1 ]] && extra+=(--params-backend "vae=SYCL0")
  elif [[ "$MODEL" == qwen ]]; then
    export SD_XPU_QWEN21_LIB="${SD_XPU_QWEN21_LIB:-$XPU_DIR/libxpu_qwen21.so}" SD_XPU_QWEN21_VERBOSE=1
    export SD_XPU_QWEN21_WEIGHTS="${SD_XPU_QWEN21_WEIGHTS:-$MODELS/noctQAnime_v10_int8_convrot.safetensors}"
    export SD_XPU_QWEN21_LORA="$xpu_loras"
  else
    # NAME-int8-convrot: ComfyUI's int8 convrot file where the model ships one (Base, MOSS-FET), else libxpu_moss's
    # one-time conversion of the bf16 file (xpu_moss_save); without it the library quantizes the bf16 file at every
    # load (~2x slower load).
    moss_xpu_file=$moss_name-int8-convrot.safetensors
    [[ -f "$MODELS/$moss_xpu_file" ]] || moss_xpu_file=$moss_name.safetensors
    export SD_XPU_MOSS_LIB="${SD_XPU_MOSS_LIB:-$XPU_DIR/libxpu_moss.so}" SD_XPU_MOSS_VERBOSE=1
    export SD_XPU_MOSS_WEIGHTS="${SD_XPU_MOSS_WEIGHTS:-$MODELS/$moss_xpu_file}"
    export SD_XPU_MOSS_LORA="$xpu_loras"
    [[ -f "$SD_XPU_MOSS_WEIGHTS" ]] || { echo "not found: $SD_XPU_MOSS_WEIGHTS" >&2; exit 1; }
    # TE_OFFLOAD=1: the text encoder's weights stay in RAM and go to the card only while it encodes (it still runs
    # on the B70), so they do not sit in VRAM through sampling.
    [[ "${TE_OFFLOAD:-0}" == 1 ]] && extra+=(--params-backend te=cpu)
  fi
fi

echo "b70 $MODEL$([[ $XPU -eq 1 ]] && echo +xpu)$([[ $TURBO -eq 1 ]] && echo +turbo)$([[ $HIRES -eq 1 ]] && echo +hires) seed=$SEED steps=$STEPS cfg=$CFG${CFG_STEPS:+ cfg-steps=$CFG_STEPS} ${WIDTH}x${HEIGHT} $(basename "$DIFFUSION") -> $OUT (log $LOG)"

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
  --diffusion-fa
  "${extra[@]}"
)
[[ "$BATCH_COUNT" =~ ^[1-8]$ ]] || { echo "-b takes 1 to 8" >&2; exit 2; }
[[ $BATCH_COUNT -gt 1 ]] && args+=(-b "$BATCH_COUNT")
if [[ -n "$EMIT_ARGS" ]]; then
  # NUL-separated: OUT, LOG, then KEY=VALUE env lines, "--", then the sd-cli arguments. B70_SD_CLI names the sd-cli
  # build the run needs (the batch runs it unless SD_FAST_BIN is set).
  {
    printf '%s\0' "$OUT" "$LOG" "B70_SD_CLI=$SD_CLI"
    for v in ONEAPI_DEVICE_SELECTOR SD_XPU_QWEN21_LIB SD_XPU_QWEN21_VERBOSE SYCL_CACHE_PERSISTENT \
      SD_XPU_QWEN21_WEIGHTS SD_XPU_QWEN21_LORA SD_XPU_MOSS_LIB SD_XPU_MOSS_VERBOSE SD_XPU_MOSS_WEIGHTS SD_XPU_MOSS_LORA \
      SD_XPU_H3_LIB SD_XPU_H3_VERBOSE SD_XPU_H3_WEIGHTS SD_XPU_H3_INT8 SD_XPU_H3_LORA; do
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
grep -aE 'completed, taking|generate_image completed|ERROR|xpu_(qwen21|moss|h3)\] (loaded|LoRA|error)' "$LOG" | grep -v 'loading tensors' | sed 's/^\[[A-Z]* *\] [^-]*- //'
[[ $status -eq 0 ]] && echo "saved $OUT" || echo "sd-cli failed ($status), see $LOG" >&2
exit $status
