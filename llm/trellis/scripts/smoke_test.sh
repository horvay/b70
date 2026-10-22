#!/usr/bin/env bash
# Load the model, generate 100 tokens once, stop. Stops early if RAM runs short.
T=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd); LOG=$T/run/smoke.log; mkdir -p $T/run; : > $LOG
setsid env MEMFRAC=${MEMFRAC:-0.80} "$T/scripts/serve_b70.sh" >> $LOG 2>&1 &
SP=$!
stop() { kill -TERM -- -$SP 2>/dev/null; sleep 5; kill -KILL -- -$SP 2>/dev/null; }
t0=$(date +%s)
swap0=$(awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print int((t-f)/1048576)}' /proc/meminfo)
while true; do
    avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    swap=$(awk '/SwapTotal/{t=$2}/SwapFree/{f=$2}END{print int((t-f)/1048576)}' /proc/meminfo)
    if [ "$avail" -lt 4 ] || [ "$swap" -gt $((swap0 + 20)) ]; then echo "SMOKE: memory guard (available ${avail} GB, swap ${swap} GB), stopping"; stop; exit 2; fi
    kill -0 $SP 2>/dev/null || { echo "SMOKE: server exited during load"; exit 1; }
    curl -s --max-time 2 http://127.0.0.1:30250/v1/models >/dev/null 2>&1 && break
    [ $(( $(date +%s) - t0 )) -gt 1800 ] && { echo "SMOKE: not ready after 30 min"; stop; exit 3; }
    sleep 3
done
echo "SMOKE: ready after $(( $(date +%s) - t0 )) s; available ${avail} GB, swap ${swap} GB"
t1=$(date +%s.%N)
curl -s --max-time 600 http://127.0.0.1:30250/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"flashnext-uncensored","messages":[{"role":"user","content":"Describe a tavern at the edge of a haunted forest."}],"max_tokens":100,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
  > $T/run/smoke_response.json
t2=$(date +%s.%N)
python3 - "$T/run/smoke_response.json" "$t1" "$t2" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); dt = float(sys.argv[3]) - float(sys.argv[2])
c = d.get("choices")
if not c: print("SMOKE: no completion:", str(d)[:500]); sys.exit()
u = d.get("usage", {})
print("SMOKE: answer:", (c[0]["message"].get("content") or "")[:700])
print(f"SMOKE: {u.get('completion_tokens')} tokens in {dt:.1f} s total (incl. prompt) = {u.get('completion_tokens', 0) / dt:.1f} tok/s")
PY
free -g | head -2 | sed 's/^/SMOKE: /'
stop; echo "SMOKE: server stopped"
