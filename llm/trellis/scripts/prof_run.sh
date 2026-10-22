#!/usr/bin/env bash
# One guarded server run that records torch-profiler traces (XPU kernels + CPU ops) through SGLang's /start_profile:
# a warm-up, then a fresh long prompt (PROF_PROMPT, default run/long_16k_c.json) and separately ~60 decode steps of a
# short prompt. Traces go to run/prof/ (chrome .trace.json.gz); analyse with scripts/prof_top.py. Log run/prof.log.
set -u
T=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd); LOG=$T/run/prof.log; : > "$LOG"; R=$T/run
API=http://127.0.0.1:30250
rm -rf "$R/prof"; mkdir -p "$R/prof"
"$T/scripts/guarded.sh" "$LOG" "$T/scripts/serve_b70.sh" &
G=$!
t0=$(date +%s)
until curl -s --max-time 2 $API/v1/models >/dev/null 2>&1; do
    kill -0 $G 2>/dev/null || { echo "PROF: server ended before ready" | tee -a "$LOG"; exit 1; }
    [ $(( $(date +%s) - t0 )) -gt 900 ] && break
    sleep 3
done
echo "PROF: ready after $(( $(date +%s) - t0 )) s" | tee -a "$LOG"
ask() { curl -s --max-time 300 $API/v1/chat/completions -H 'Content-Type: application/json' -d "$1" >/dev/null; }
short() { echo '{"model":"flashnext-uncensored","messages":[{"role":"user","content":"'"$1"'"}],"max_tokens":'"$2"',"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'; }
ask "$(short 'Describe a blacksmith shop.' 40)"
curl -s --max-time 300 $API/v1/chat/completions -H 'Content-Type: application/json' -d @"$R/long_16k_b.json" >/dev/null
echo "PROF: warmed up" | tee -a "$LOG"
for what in prefill decode; do
    curl -s -X POST $API/start_profile -H 'Content-Type: application/json' \
      -d '{"output_dir":"/run/trellis/prof","activities":["CPU","GPU"],"with_stack":false,"record_shapes":false,"profile_id":"'$what'"}' | head -c 200; echo
    s=$(date +%s.%N)
    if [ $what = prefill ]; then
        curl -s --max-time 300 $API/v1/chat/completions -H 'Content-Type: application/json' -d @"${PROF_PROMPT:-$R/long_16k_c.json}" >/dev/null
    else
        ask "$(short 'Write a long story about a lighthouse keeper.' 60)"
    fi
    echo "PROF: $what request took $(awk -v a="$s" -v b="$(date +%s.%N)" 'BEGIN{printf "%.1f", b-a}') s" | tee -a "$LOG"
    curl -s -X POST $API/stop_profile | head -c 200; echo
    sleep 5
done
ls -la "$R/prof" | tee -a "$LOG"
for p in $(ps -eo pid,args | awk '/[s]glang\.launch_server/ {print $1}'); do kill "$p" 2>/dev/null; done
sleep 5
for p in $(ps -eo pid,stat,args | awk '$2 !~ /Z/ && /[s]glang|[b]wrap --bind \/home\/horvay\/work\/trellis-b70/ {print $1}'); do kill -9 "$p" 2>/dev/null; done
wait $G 2>/dev/null
echo "PROF: done" | tee -a "$LOG"
