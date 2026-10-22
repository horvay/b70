#!/usr/bin/env bash
# One guarded diagnostic run: the server under guarded.sh (killed at the first B70 fault in the kernel log), one short
# request, then stop. Logs: run/diag.log (+ .kernel). QUICK=1: no debug env (real speed), MAX_TOKENS (default 20), WARMUP=1: an untimed request first, LONG=1: then long prompts (LONG_FILES).
set -u
T=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd); LOG=$T/run/diag.log; : > "$LOG"
export MEMFRAC=${MEMFRAC:-0.75}
[ "${QUICK:-0}" = 1 ] || export UR_L0_SERIALIZE=2 EXL3_ADDR_LOG=1 EXL3_PTR_CHECK=1 EXL3_NGRAM_DEBUG=1
N=${MAX_TOKENS:-20}
"$T/scripts/guarded.sh" "$LOG" "$T/scripts/serve_b70.sh" &
G=$!
t0=$(date +%s)
until curl -s --max-time 2 http://127.0.0.1:30250/v1/models >/dev/null 2>&1; do
    kill -0 $G 2>/dev/null || { echo "DIAG: server ended before ready"; exit 1; }
    [ $(( $(date +%s) - t0 )) -gt 900 ] && { echo "DIAG: not ready after 15 min"; break; }
    sleep 3
done
echo "DIAG: ready after $(( $(date +%s) - t0 )) s" | tee -a "$LOG"
if [ "${WARMUP:-0}" = 1 ]; then      # one untimed request first (warms the expert cache)
    curl -s --max-time 300 http://127.0.0.1:30250/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"model":"flashnext-uncensored","messages":[{"role":"user","content":"Describe a blacksmith shop in a mountain village."}],"max_tokens":'"$N"',"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' >/dev/null
    echo "DIAG: warm-up request done" | tee -a "$LOG"
fi
r0=$(date +%s.%N)
curl -s --max-time 300 http://127.0.0.1:30250/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"flashnext-uncensored","messages":[{"role":"user","content":"Describe a tavern at the edge of a haunted forest."}],"max_tokens":'"$N"',"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
  > "$T/run/diag_response.json"
echo "DIAG: request took $(awk -v a="$r0" -v b="$(date +%s.%N)" 'BEGIN{printf "%.1f", b-a}') s" | tee -a "$LOG"
echo "DIAG: response: $(head -c 400 "$T/run/diag_response.json")" | tee -a "$LOG"
echo "DIAG: usage: $(grep -o '"usage":{[^}]*}' "$T/run/diag_response.json")" | tee -a "$LOG"
if [ "${VISION:-0}" = 1 ]; then        # one image (run/vision_test.png: red circle, blue square, "42") + a question
    python3 - "$T/run/vision_test.png" <<'PY' 2>&1 | tee -a "$LOG"
import base64, json, sys, time, urllib.request
url = "data:image/png;base64," + base64.b64encode(open(sys.argv[1], "rb").read()).decode()
body = {"model": "flashnext-uncensored", "temperature": 0, "max_tokens": 80, "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                     {"type": "text", "text": "What shapes and colours are in this image, and what number is written in it?"}]}]}
t0 = time.time()
try:
    req = urllib.request.Request("http://127.0.0.1:30250/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=300))
    print(f"DIAG: vision {time.time() - t0:.1f} s, usage {out.get('usage')}, answer: {out['choices'][0]['message']['content']!r}")
except Exception as e:
    print(f"DIAG: vision request failed after {time.time() - t0:.1f} s: {e} {getattr(e, 'read', lambda: b'')()[:300]!r}")
PY
fi
if [ "${LONG:-0}" = 1 ]; then          # long prompts (LONG_FILES, default run/long_prompt.json), 20 tokens out each
    for f in ${LONG_FILES:-$T/run/long_prompt.json}; do
        r1=$(date +%s.%N)
        curl -s --max-time 300 http://127.0.0.1:30250/v1/chat/completions -H 'Content-Type: application/json' \
          -d @"$f" > "$T/run/diag_long.json"
        dt=$(awk -v a="$r1" -v b="$(date +%s.%N)" 'BEGIN{printf "%.1f", b-a}')
        pt=$(grep -o '"prompt_tokens":[0-9]*' "$T/run/diag_long.json" | grep -o '[0-9]*$')
        # prefill rate: prompt tokens over the request time minus its 20 output tokens at ~30 tok/s
        echo "DIAG: long $(basename "$f"): ${pt:-no answer} tokens in $dt s, prefill ~$(awk -v p="${pt:-0}" -v t="$dt" 'BEGIN{printf "%.0f", p/(t-20/30)}') tok/s $(head -c 300 "$T/run/diag_long.json" | grep -o '"error[^}]*')" | tee -a "$LOG"
    done
    if [ -n "${EXL3_MOE_PROFILE:-}" ]; then   # a tiny prefill makes the server write its MoE timing; report it
        sleep 2.5
        curl -s --max-time 60 http://127.0.0.1:30250/v1/chat/completions -H 'Content-Type: application/json' \
          -d '{"model":"flashnext-uncensored","messages":[{"role":"user","content":"Say OK."}],"max_tokens":2}' >/dev/null
        python3 -c "import json; d=json.load(open('$T/run/route_profile.json')); print(f'DIAG: staged prefill MoE {d[\"prefill_moe_ms\"]/1e3:.2f} s for {d[\"prefill_moe_rows\"] // 48} tokens')" | tee -a "$LOG"
    fi
fi
for p in $(ps -eo pid,args | awk '/[s]glang\.launch_server/ {print $1}'); do kill "$p" 2>/dev/null; done
sleep 5
for p in $(ps -eo pid,stat,args | awk '$2 !~ /Z/ && /[s]glang|[b]wrap --bind \/home\/horvay\/work\/trellis-b70/ {print $1}'); do kill -9 "$p" 2>/dev/null; done
wait $G 2>/dev/null
echo "DIAG: done"
