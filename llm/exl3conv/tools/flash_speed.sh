#!/usr/bin/env bash
# Decode speed of a Flash-Next EXL3 on the profile prompts (flashnext/tools/route_profile.py: synthetic game-master and
# general turns, nothing personal), for comparing b70 serve settings. One server run, then stops.
#   flash_speed.sh <model dir> [phases, default "A G"] [b70 serve options...]      e.g. --ram-list ram-8400 --margin 1
set -u
F=$(cd "$(dirname "$0")/../../flashnext" && pwd); L=$(dirname "$F"); B70=$L/b70
MODEL=$1; PH=${2:-"A G"}; shift 2 2>/dev/null || shift $#
PORT=30250; LOG=$(mktemp /tmp/flash_speed.XXXXXX.log)
"$B70" serve "$MODEL" --port $PORT --ctx 32768 --parallel 2 "$@" > "$LOG" 2>&1 &
G=$!
t0=$(date +%s)
until curl -s --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; do
    kill -0 $G 2>/dev/null || { echo "server ended before ready:"; grep -E "GUARD tripped|Error" "$LOG" | tail -3; exit 1; }
    [ $(( $(date +%s) - t0 )) -gt 1800 ] && { echo "not ready after 30 min"; "$B70" stop; exit 1; }
    sleep 3
done
echo "ready after $(( $(date +%s) - t0 )) s (serve options: $*)"
for ph in $PH; do
    case $ph in A|B) SYS=$L/exl3conv/data/gm-system-prompt.txt ;; *) SYS=- ;; esac
    python3 "$F/tools/route_profile.py" $ph "$SYS" "/tmp/flash_speed_$ph.json" 250 2>&1 | grep -E "mean decode"
done
"$B70" stop; wait $G 2>/dev/null; rm -f "$LOG"
