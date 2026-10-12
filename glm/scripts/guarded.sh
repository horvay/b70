#!/usr/bin/env bash
# guarded.sh <log> <command...>: run a GPU command in its own process group, and kill the whole group at the first sign
# that it is hurting the machine:
#   - an xe fault / hang / reset line for the B70 in the kernel log (a faulting kernel wedges the xe driver if it keeps
#     going; see memory b70-gpu-faults-can-freeze-machine). Kernel lines from the run go to <log>.kernel;
#   - low RAM: MemAvailable under RAM_FLOOR_GB (default 4) or memory pressure (PSI "full" avg10) over PSI_MAX
#     (default 40; loading alone reaches ~20 while it streams 68 GB of shards, systemd-oomd acts at 50). The server pins ~35 GB of host RAM that cannot be swapped; on 2026-10-04 it and a video job
#     together ran the machine out of memory and systemd-oomd killed the whole terminal session;
#   - one of the user's own GPU jobs starting on the B70 (OTHER_JOBS, a regex over process command lines, and the
#     process has the B70 render node open; held 3 s): theirs wins.
# Before starting it refuses to run with less than RAM_START_GB (default 45) available or with such a job running.
# The command runs with oom_score_adj 1000, so the kernel OOM killer takes it before anything of the user's.
set -u
LOG=$1; shift
RAM_FLOOR_GB=${RAM_FLOOR_GB:-4}; PSI_MAX=${PSI_MAX:-40}; RAM_START_GB=${RAM_START_GB:-45}
OTHER_JOBS=${OTHER_JOBS:-'sd-cli|main\.py --listen|llama-server|ComfyUI'}
: > "$LOG.kernel"

avail_gb() { awk '/^MemAvailable:/ {printf "%d", $2 / 1048576}' /proc/meminfo; }
B70_RENDER=$(for r in /sys/class/drm/renderD*; do
    [ "$(basename "$(readlink -f "$r/device")")" = 0000:0a:00.0 ] && echo "/dev/dri/$(basename "$r")"; done)
# a job counts once it has the B70's render node open. One on the NVIDIA card (CUDA sd-server, ComfyUI) or with no GPU
# is left alone: a starting job's RAM is covered by the RAM checks, and CUDA init can take many seconds, so "no GPU
# yet" is not a sign it wants the B70. Shells whose command text merely mentions a job are skipped.
other_jobs() {
    ps -eo pid,comm,args | grep -E -- "$OTHER_JOBS" | while read -r pid comm args; do
        case "$comm" in bash|sh|zsh|dash|grep|guarded.sh|timeout|tail) continue ;; esac
        fds=$(ls -l /proc/"$pid"/fd 2>/dev/null | grep -oE '/dev/(dri/renderD[0-9]+|nvidia[0-9]+)' | sort -u)
        if echo "$fds" | grep -qx "$B70_RENDER"; then
            echo "$pid $args" | cut -c1-120
        fi
    done
}

a=$(avail_gb); o=$(other_jobs)
if [ "$a" -lt "$RAM_START_GB" ]; then
    echo "GUARD: not starting, only $a GB RAM available (need $RAM_START_GB)" | tee -a "$LOG"; exit 98
fi
if [ -n "$o" ]; then
    echo "GUARD: not starting, another GPU job is running: $o" | tee -a "$LOG"; exit 98
fi

START=$(stat -c %s "$LOG" 2>/dev/null || echo 0)     # only this run's GUARD lines count at the end
setsid bash -c 'echo 1000 > /proc/self/oom_score_adj; exec "$@"' guarded "$@" >> "$LOG" 2>&1 &
CMD=$!

stop_run() {      # $1: reason
    echo "GUARD: $1, stopping the run" >> "$LOG"
    {   # who was using the memory, for the post-mortem
        echo "GUARD top RAM users at $(date +%T):"
        ps -eo pid,rss,args --sort=-rss | head -9 | awk 'NR == 1 {print "  " $0; next} {printf "  %s %.1fG %s\n", $1, $2 / 1048576, substr($0, index($0, $3), 110)}'
        free -g | sed 's/^/  /'
    } >> "$LOG" 2>/dev/null
    kill -TERM -- -"$CMD" 2>/dev/null; sleep 3; kill -KILL -- -"$CMD" 2>/dev/null
}

GUARD_T0=$(date +%s)
(
    # -n 0 alone replayed old lines after a journal rotation (a 02:30 fault stopped a 02:33 start): also skip any line
    # stamped before this run started
    journalctl -k -f -n 0 --since "@$GUARD_T0" --no-pager 2>/dev/null | while IFS= read -r line; do
        ts=$(date -d "$(echo "$line" | cut -d' ' -f1-3)" +%s 2>/dev/null || echo "$GUARD_T0")
        [ "$ts" -lt "$GUARD_T0" ] && continue
        case "$line" in
            *"xe 0000:0a:00.0"*)
                echo "$line" >> "$LOG.kernel"
                case "$line" in
                    *Fault*|*"CAT error"*|*hang*|*Hang*|*reset*|*Reset*|*wedged*|*Timedout*)
                        stop_run "B70 fault in the kernel log: $line" ;;
                esac
                ;;
        esac
    done
) &
WATCH=$!

(
    while kill -0 "$CMD" 2>/dev/null; do
        a=$(avail_gb)
        p=$(awk '/^full/ {sub("avg10=", "", $2); printf "%d", $2}' /proc/pressure/memory 2>/dev/null)
        if [ "$a" -lt "$RAM_FLOOR_GB" ]; then stop_run "RAM low ($a GB available, floor $RAM_FLOOR_GB)"; break; fi
        if [ "${p:-0}" -gt "$PSI_MAX" ]; then stop_run "memory pressure ${p}% (full avg10, max $PSI_MAX), $a GB RAM available"; break; fi
        o=$(other_jobs)                 # 3 checks in a row: a starting CUDA job has opened its card by then
        if [ -n "$o" ]; then seen=$((${seen:-0} + 1)); else seen=0; fi
        if [ "$seen" -ge 3 ]; then stop_run "another GPU job started: $o"; break; fi
        sleep 1
    done
) &
MEMWATCH=$!

wait "$CMD"; RC=$?
sleep 1
pkill -P "$WATCH" journalctl 2>/dev/null; kill "$WATCH" "$MEMWATCH" 2>/dev/null
g=$(tail -c +$((START + 1)) "$LOG" | grep '^GUARD:' | tail -1)
[ -n "$g" ] && { echo "GUARD tripped: $g"; exit 99; }
exit $RC
