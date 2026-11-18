#!/bin/bash
# batchsize.sh: Drama 3 (drama3_batch, SYCL) speed and peak VRAM per in-process batch size, on the first N lines of
# bench256.tsv. One process per batch size, run inside one b70q job:
#   b70-lease run drama3-bench -- ~/b70/bench/drama3/batchsize.sh OUTDIR [N=128] [BATCHES="16 24 32"]
# Writes OUTDIR/batchsize.tsv (batch, capacity, lines, audio_s, gen_s, s_per_s, load_s, peak_vram_gib, exit) and
# OUTDIR/b<batch>.log. Peak VRAM is the process's drm-total-vram from /proc/<pid>/fdinfo, sampled every 0.25 s.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:?usage: batchsize.sh OUTDIR [N] [BATCHES]}; N=${2:-128}; BATCHES=${3:-"16 24 32"}
BIN=$HOME/b70/engines/audio.cpp/build/linux-sycl-release/bin/drama3_batch
MODEL=$HOME/models/voice/drama3/drama-3-backbone-8bit-experts-Q4_K.gguf
VOICES=${VOICES:-$HOME/cdk-work/romance/voice/designs}
export D3_BACKEND=sycl ONEAPI_DEVICE_SELECTOR=level_zero:0 CUDA_VISIBLE_DEVICES=""
mkdir -p "$OUT"; R=$OUT/batchsize.tsv
[[ -f $R ]] || echo -e "batch\tcapacity\tlines\taudio_s\tgen_s\ts_per_s\tload_s\tpeak_vram_gib\texit" > "$R"
vram() {  # sum of drm-total-vram* over the process's DRM fds, in KiB
  local t=0
  for f in /proc/$1/fdinfo/*; do
    v=$(awk '/^drm-total-vram/ {s+=$2} END {print s+0}' "$f" 2>/dev/null); t=$((t + ${v:-0}))
  done; echo $t
}
for b in $BATCHES; do
  cap=1024; [[ $b -ge 32 ]] && cap=${CAP32:-768}
  rm -rf "$OUT/wav$b"; mkdir -p "$OUT/wav$b"
  "$BIN" "$HERE/bench256.tsv" --model "$MODEL" --voices-dir "$VOICES" --out "$OUT/wav$b" --batch "$b" \
      --capacity "$cap" --limit "$N" > "$OUT/b$b.log" 2>&1 &
  pid=$!; peak=0
  while kill -0 $pid 2>/dev/null; do v=$(vram $pid); [[ $v -gt $peak ]] && peak=$v; sleep 0.25; done
  wait $pid; rc=$?
  line=$(grep -o 'DONE [0-9]* lines, [0-9.]* s audio. generate+decode [0-9.]* s ([0-9.]* s/s).*load [0-9.]* s' "$OUT/b$b.log")
  read -r lines audio gen sps load < <(echo "$line" | sed -E 's/DONE ([0-9]+) lines, ([0-9.]+) s audio. generate\+decode ([0-9.]+) s \(([0-9.]+) s\/s\).*load ([0-9.]+) s/\1 \2 \3 \4 \5/')
  echo -e "$b\t$cap\t${lines:-0}\t${audio:-0}\t${gen:-0}\t${sps:-0}\t${load:-0}\t$(awk -v k=$peak 'BEGIN{printf "%.1f", k/1048576}')\t$rc" | tee -a "$R"
  rm -rf "$OUT/wav$b"
done
