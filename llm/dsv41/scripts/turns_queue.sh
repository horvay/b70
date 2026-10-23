#!/bin/bash
# Prompt-path modes on a Game Master-shaped conversation (tests/t_turns.py): plain, short-prompt staging
# (DSV_PREFILL_TAIL), a 256-token first chunk (DSV_FIRST_CHUNK), set-aware prompt routing (DSV_PREFILL_SET_MIN), plain
# again (the cache warms within a run, so plain runs first and last). `turns_queue.sh confirm` alternates tail and plain
# twice, tail first, to check tail's lead isn't warm-up. One fresh server per mode, the same warm routing
# profile for each (DSV_SAVE_PROFILE=0). Queued through b70q; `b70q log <id> -f` follows it. Results land in run/turns_<mode>.out.
cd /home/horvay/b70/llm/dsv41
b70q submit dsv41-turns --ram-gib 37 --note "DSV prompt modes on GM-shaped turns" -- bash -c "SET=${1:-all}; $(cat <<'RUN'
PORT=8061
mode() {
    label=$1; shift
    echo "=== $label: $*"
    RAM_START_GB=37 scripts/guarded.sh run/turns_$label.log env DSV_BOOST=0.3 DSV_STAGE_GB=0.5 DSV_SAVE_PROFILE=0 "$@" \
        scripts/in_rootfs.sh --gpu python3 -X faulthandler /w/dsv41/xpu/serve.py --port $PORT &
    g=$!
    for _ in $(seq 200); do
        sleep 3
        grep -q "server ready" run/turns_$label.log 2>/dev/null && break
        kill -0 $g 2>/dev/null || break
    done
    if grep -q "server ready" run/turns_$label.log; then
        python3 tests/t_turns.py $PORT "$label" 2>&1 | tee run/turns_$label.out
    else
        echo "$label: server did not come up"; tail -5 run/turns_$label.log
    fi
    curl -s -m 5 -X POST http://127.0.0.1:$PORT/quit >/dev/null
    wait $g
    echo "$label exit $?"
    grep -E "GUARD|xe .*(fault|hang|reset)" run/turns_$label.log run/turns_$label.log.kernel && { echo "stopping: GPU or RAM trouble"; exit 1; }
}
if [ "$SET" = confirm ]; then
    mode tailA DSV_PREFILL_TAIL=32 DSV_STAGE_GB=1.9
    mode plainA
    mode tailB DSV_PREFILL_TAIL=32 DSV_STAGE_GB=1.9
    mode plainB
else
    mode plain1
    mode tail DSV_PREFILL_TAIL=32 DSV_STAGE_GB=1.9
    mode first256 DSV_FIRST_CHUNK=256
    mode set4 DSV_PREFILL_SET_MIN=4
    mode plain2
fi
grep -h SUMMARY run/turns_*.out
RUN
)"
