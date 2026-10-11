#!/usr/bin/env bash
# Restart the dev server (xpu/devserver.py) under the guard and wait until it is ready. Args go to devserver.py.
set -u
W=$HOME/work/glm-b70
cd "$W"
python3 xpu/dev.py --quit >/dev/null 2>&1
for _ in $(seq 60); do        # pinned RAM is released a few seconds after the old server exits
    [ "$(awk '/^MemAvailable:/ {printf "%d", $2 / 1048576}' /proc/meminfo)" -ge "${RAM_START_GB:-30}" ] && break
    sleep 2
done
# guards of earlier runs whose server is gone (a stopped run's guard can linger and log stale GUARD lines)
for g in $(pgrep -f "guarded.sh run/dev.log"); do
    pgrep -P "$g" >/dev/null 2>&1 || kill "$g" 2>/dev/null
done
pgrep -f "devserver.py" >/dev/null || pkill -f "guarded.sh run/dev.log" 2>/dev/null
off=$(stat -c %s run/dev.log 2>/dev/null || echo 0)
(RAM_START_GB=${RAM_START_GB:-30} setsid nohup scripts/guarded.sh run/dev.log scripts/in_rootfs.sh --gpu \
    python3 /w/xpu/devserver.py "$@" > run/dev.rc 2>&1 &)
for _ in $(seq 150); do
    sleep 2
    new=$(tail -c +$((off + 1)) run/dev.log)
    echo "$new" | grep -q "dev server ready" && { echo "$new" | grep "glm_xpu" | tail -3; exit 0; }
    echo "$new" | grep -qE "GUARD|Traceback" && { echo "$new" | tail -5; exit 1; }
done
echo "dev server did not come up"; exit 1
