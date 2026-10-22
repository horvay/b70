#!/usr/bin/env bash
# Several stills in one sd-cli process on the B70: the diffusion model, text encoder, VAE and (with --xpu) the
# XPU block weights load once, then the stills render one after another.
#
# Usage: b70-image-batch.sh JOBS [COMMON ARGS...]     (or: b70-image.sh --batch JOBS [COMMON ARGS...])
#   JOBS has one still per line, written as b70-image.sh arguments with shell quoting; blank lines and lines
#   starting with # are skipped. COMMON ARGS (e.g. -m qwen --xpu) are put before every line's arguments.
#     -p "An anime illustration of ..." -r refs/a.png --seed 7 -o out/a.png
#     -p "An anime illustration of ..." -r refs/a.png --seed 8 --cfg-steps 15 -o out/b.png
# Each line is resolved by b70-image.sh itself (--emit-args), so it gets exactly a single run's settings
# (per-model steps, CFG, size, scheduler, negative prompt, VAE tiling, vision tower for -r refs, XPU env and
# <lora:> tags). All jobs must use the same model files and, with --xpu, the same <lora:> tags (the XPU blocks
# load their LoRAs once per process).
#
# Writes every job's OUT and OUT-minus-extension.log (its part of the run log; the first job's log also holds the
# shared model load), plus BATCH_LOG (default: JOBS-minus-extension.batch.log) and the sd-cli jobs file
# (BATCH_LOG with .jobs.jsonl). SD_FAST_BIN picks the sd-cli build (needs img_gen --multi-jobs).
set -euo pipefail

ROOT="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"   # ~/b70/tools
SD_FAST_BIN="${SD_FAST_BIN:-$(dirname "$ROOT")/engines/sd.cpp/build-sycl/bin/sd-cli}"

[[ $# -ge 1 && "$1" != -h && "$1" != --help ]] || { sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
JOBS="$1"; shift
COMMON=("$@")
[[ -f "$JOBS" ]] || { echo "not found: $JOBS" >&2; exit 1; }
[[ -x "$SD_FAST_BIN" ]] || { echo "not found: $SD_FAST_BIN (set SD_FAST_BIN)" >&2; exit 1; }
BATCH_LOG="${BATCH_LOG:-${JOBS%.*}.batch.log}"
[[ "$BATCH_LOG" == /* ]] || BATCH_LOG="$PWD/$BATCH_LOG"
JOBS_JSONL="${BATCH_LOG%.log}.jobs.jsonl"
mkdir -p "$(dirname "$BATCH_LOG")"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# One NUL-separated argument file per job line (shell quoting rules, no expansion).
python3 - "$JOBS" "$tmp" <<'EOF'
import shlex, sys
n = 0
for line_no, line in enumerate(open(sys.argv[1], encoding="utf-8"), 1):
    if not line.strip() or line.lstrip().startswith("#"):
        continue
    try:
        args = shlex.split(line)
    except ValueError as e:
        sys.exit(f"{sys.argv[1]}:{line_no}: {e}")
    n += 1
    with open(f"{sys.argv[2]}/job-{n}.args", "wb") as f:
        f.write(b"".join(a.encode() + b"\0" for a in args))
if n == 0:
    sys.exit(f"{sys.argv[1]}: no jobs")
EOF

# Context (model) options of the sd-cli command line; everything else is per job.
ctx_with_value=" --diffusion-model --llm --vae --lora-model-dir --llm_vision "
ctx_flags=" --diffusion-fa -v "

base=() base_key="" base_env=() vision="" outs=() logs=()
n=$(find "$tmp" -name 'job-*.args' | wc -l)
for ((j = 1; j <= n; j++)); do
  mapfile -d '' jargs < "$tmp/job-$j.args"
  env -u SD_CLI "$ROOT/b70-image.sh" ${COMMON[@]+"${COMMON[@]}"} "${jargs[@]}" --emit-args "$tmp/job-$j.emit" ||
    { echo "job $j: b70-image.sh rejected its arguments" >&2; exit 2; }
  mapfile -d '' e < "$tmp/job-$j.emit"
  outs+=("${e[0]}") logs+=("${e[1]}")
  i=2 envs=()
  while [[ "${e[i]}" != -- ]]; do envs+=("${e[i]}"); ((i += 1)); done
  ((i += 1))
  ctx=() job=()
  while ((i < ${#e[@]})); do
    a="${e[i]}"
    if [[ "$ctx_with_value" == *" $a "* ]]; then
      if [[ "$a" == --llm_vision ]]; then
        [[ -z "$vision" || "$vision" == "${e[i + 1]}" ]] || { echo "job $j: different --llm_vision" >&2; exit 2; }
        vision="${e[i + 1]}"
      else
        ctx+=("$a" "${e[i + 1]}")
      fi
      ((i += 2))
    elif [[ "$ctx_flags" == *" $a "* ]]; then
      ctx+=("$a"); ((i += 1))
    else
      job+=("$a"); ((i += 1))
    fi
  done
  key=$(printf '%s\x1f' "${ctx[@]}" "|" "${envs[@]}")
  if [[ $j -eq 1 ]]; then
    base=("${ctx[@]}") base_key="$key" base_env=("${envs[@]}")
  elif [[ "$key" != "$base_key" ]]; then
    echo "job $j: model files, flags or XPU environment (e.g. <lora:> tags with --xpu) differ from job 1;" \
      "run it in a separate batch" >&2
    exit 2
  fi
  printf '%s\0' "${job[@]}" > "$tmp/job-$j.job"
done
[[ -n "$vision" ]] && base+=(--llm_vision "$vision")

python3 - "$tmp" "$n" > "$JOBS_JSONL" <<'EOF'
import json, sys
for j in range(1, int(sys.argv[2]) + 1):
    data = open(f"{sys.argv[1]}/job-{j}.job", "rb").read()
    print(json.dumps([a.decode() for a in data.split(b"\0")[:-1]], ensure_ascii=False))
EOF

echo "b70 batch: $n stills in one process ($(basename "$SD_FAST_BIN")), log $BATCH_LOG"
for ((j = 0; j < n; j++)); do echo "  $((j + 1)): ${outs[j]}"; done

env ${base_env[@]+"${base_env[@]}"} "$SD_FAST_BIN" "${base[@]}" --multi-jobs "$JOBS_JSONL" > "$BATCH_LOG" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null' INT TERM

until grep -aq 'total params memory size' "$BATCH_LOG" 2>/dev/null || ! kill -0 $PID 2>/dev/null; do sleep 0.5; done
grep -aE 'level_zero:gpu|total params memory size|on XPU' "$BATCH_LOG" | sed 's/^\[INFO *\] [^-]*- //'
until awk '/generating image/ { g = 1 } g && /s\/it|it\/s/ { f = 1 } END { exit !f }' "$BATCH_LOG" 2>/dev/null || ! kill -0 $PID 2>/dev/null; do sleep 0.5; done
kill -0 $PID 2>/dev/null && "$ROOT/b70-watch.sh" $PID 3 2 || true

wait $PID && status=0 || status=$?

# Split the run log per job at the "[multi] job N/M start" markers; the shared load goes to job 1's log.
awk -v files="$(printf '%s\n' "${logs[@]}")" '
  BEGIN { n = split(files, f, "\n"); cur = 1 }
  match($0, /\[multi\] job [0-9]+\/[0-9]+ start:/) { s = substr($0, RSTART + 12); cur = s + 0 }
  cur >= 1 && cur <= n { print > f[cur] }
' "$BATCH_LOG"

for ((j = 0; j < n; j++)); do
  echo "-- $((j + 1)): ${outs[j]}"
  [[ -f "${logs[j]}" ]] || { echo "   not started"; continue; }
  grep -aE 'completed, taking|generate_image completed|ERROR|xpu_qwen21\] (loaded|LoRA|error)|\[multi\] job [0-9]+/[0-9]+ (saved|\()' "${logs[j]}" |
    grep -v 'loading tensors' | sed 's/^\[[A-Z]* *\] [^-]*- //'
done
grep -aE '\[multi\] (context ready|[0-9]+/[0-9]+ jobs saved)' "$BATCH_LOG" | sed 's/^\[[A-Z]* *\] [^-]*- //'
[[ $status -eq 0 ]] && echo "saved $n stills" || echo "sd-cli failed ($status), see $BATCH_LOG" >&2
exit $status
