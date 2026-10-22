#!/usr/bin/env bash
T=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
# a server started by start_server.sh: its b70q ticket (the daemon stops every process of it)
if [ -s "$T/run/serve.ticket" ]; then
    b70q kill "$(cat "$T/run/serve.ticket")" >/dev/null 2>&1
    rm -f "$T/run/serve.ticket"
fi
# Stop the trellis-serve smoke test and server (patterns bracketed so they never match this script's own shell).
for p in $(ps -eo pid,args | awk '/[s]moke_test\.sh/ {print $1}'); do kill "$p" 2>/dev/null; done
for p in $(ps -eo pid,args | awk '/[s]glang\.launch_server|[b]wrap --bind \/home\/horvay\/work\/trellis-b70/ {print $1}'); do kill "$p" 2>/dev/null; done
sleep 5
for p in $(ps -eo pid,stat,args | awk '$2 !~ /Z/ && /[s]glang|[b]wrap --bind \/home\/horvay\/work\/trellis-b70/ {print $1}'); do kill -9 "$p" 2>/dev/null; done
# the guard wrapper of a detached server (start_server.sh); anchored so it never matches a shell that merely mentions it
for p in $(pgrep -f '^bash scripts/guarded\.sh run/serve\.log'); do kill "$p" 2>/dev/null; done
