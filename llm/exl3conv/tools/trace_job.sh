#!/usr/bin/env bash
# One queue turn of trace generation: serve <model> on the B70, run make_trace.py for about --minutes, stop the server.
# Exit 99 while the trace is short of its target (b70q submit --chunked queues the next turn), 0 when done.
#   trace_job.sh <exl3 model dir> <trace.json> [minutes [make_trace.py run options...]]
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
M=$1; TRACE=$2; MIN=${3:-10}; PORT=${TRACE_PORT:-8123}
LOG="${TRACE%.json}.serve.log"
PAR=${TRACE_PARALLEL:-8}
# b70 serve sizes the sliding-window pool for chat sessions (a window kept for every idle game); here nothing idles,
# so it gets one prefill chunk plus each running request's window, and the full-attention pool the rest.
export B70_SWA_TOKENS=${B70_SWA_TOKENS:-$((2048 + PAR * 1408 + 512))}
~/b70/llm/b70 serve "$M" --port "$PORT" --ctx "${TRACE_CTX:-6144}" --parallel "$PAR" ${TRACE_DRAFTER:+--drafter "$TRACE_DRAFTER"} > "$LOG" 2>&1 &
srv=$!
cleanup() { ~/b70/llm/b70 stop >/dev/null 2>&1; kill "$srv" 2>/dev/null; wait "$srv" 2>/dev/null; }
trap cleanup EXIT
for i in $(seq 1 240); do
  curl -s -m 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  kill -0 "$srv" 2>/dev/null || { echo "server died"; tail -30 "$LOG"; exit 1; }
  sleep 5
done
python3 -I "$HERE/make_trace.py" run --url "http://127.0.0.1:$PORT" --trace "$TRACE" --minutes "$MIN" "${@:4}"
