#!/usr/bin/env bash
# Sample B70 (xe) engine busy %, VRAM and CPU % for one process.
# Usage: b70-watch.sh PID [SAMPLES] [INTERVAL_S]
set -euo pipefail

PID="$1"
SAMPLES="${2:-5}"
INTERVAL="${3:-2}"
HZ=$(getconf CLK_TCK)

# Sum per-engine "drm-cycles-*" / "drm-total-cycles-*" and VRAM over every xe fd of PID.
read_xe() {
  local f
  for f in /proc/"$PID"/fdinfo/*; do
    grep -q '^drm-driver:[[:space:]]*xe' "$f" 2>/dev/null && cat "$f"
  done | awk '
    /^drm-client-id:/ { if (seen[$2]++) skip = 1; else skip = 0; next }
    skip { next }
    /^drm-cycles-/       { e = substr($1, 12); sub(":", "", e); cyc[e] += $2 }
    /^drm-total-cycles-/ { e = substr($1, 18); sub(":", "", e); tot[e] = $2 }
    /^drm-total-vram0:/  { m = $3 == "GiB" ? 1048576 : $3 == "MiB" ? 1024 : $3 == "KiB" ? 1 : 1 / 1024; vram += $2 * m }
    END {
      for (e in cyc) printf "%s %d %d\n", e, cyc[e], tot[e]
      printf "vram %d 0\n", vram
    }'
}

cpu_ticks() { awk '{ print $14 + $15 }' /proc/"$PID"/stat; }

declare -A c0 t0
while read -r e c t; do c0[$e]=$c; t0[$e]=$t; done < <(read_xe)
p0=$(cpu_ticks)
w0=$(date +%s.%N)

for ((i = 1; i <= SAMPLES; i++)); do
  sleep "$INTERVAL"
  [[ -d /proc/$PID ]] || { echo "pid $PID exited"; exit 0; }
  line=""
  vram=0
  while read -r e c t; do
    if [[ $e == vram ]]; then vram=$c; continue; fi
    dt=$((t - ${t0[$e]:-$t}))
    if ((dt > 0)); then
      pct=$(( (c - ${c0[$e]:-$c}) * 100 / dt )); line+=" $e=$(( pct > 100 ? 100 : pct ))%"
    fi
    c0[$e]=$c; t0[$e]=$t
  done < <(read_xe)
  p1=$(cpu_ticks); w1=$(date +%s.%N)
  cpu=$(awk -v a="$p0" -v b="$p1" -v x="$w0" -v y="$w1" -v hz="$HZ" 'BEGIN { printf "%.0f", (b - a) / hz / (y - x) * 100 }')
  p0=$p1; w0=$w1
  printf 'b70:%s vram=%.1fGiB | cpu=%s%%\n' "${line:- idle}" "$(awk -v k="$vram" 'BEGIN { print k / 1048576 }')" "$cpu"
done
