#!/usr/bin/env bash
# Run one exl3conv command on the B70 with a fault guard: gpu_run.sh [-t SECONDS] -- <cmd ...>
#
# Selects the B70 (Level Zero device 0), keeps a time limit (default 600 s), and stops the command at the first
# xe fault, hang or reset for the B70 in the kernel log, like `b70 serve`'s guard (faulting kernels that keep running
# have wedged the driver). Does not take a queue turn itself: run it inside `b70-lease run` or a `b70q pause` window.
# The kernel-log follower runs in its own session with no inherited stdout, and is killed as a group on exit, so
# nothing outlives the command.
set -euo pipefail
LIMIT=600
if [[ ${1:-} == -t ]]; then LIMIT=$2; shift 2; fi
[[ ${1:-} == -- ]] && shift
PCI=${B70_PCI:-0000:0a:00.0}
export ONEAPI_DEVICE_SELECTOR=level_zero:0 ZE_AFFINITY_MASK=0 SYCL_CACHE_PERSISTENT=1

fault=$(mktemp)
setsid bash -c 'exec journalctl -k -f -n 0 -o cat | grep --line-buffered -m1 -E "$1" > "$2"' _ \
  "xe $PCI.*(Fault|CAT error|[Hh]ang|[Rr]eset|wedged|Timedout)" "$fault" </dev/null >/dev/null 2>&1 &
guard=$!
cleanup() { kill -- "-$guard" 2>/dev/null || true; rm -f "$fault"; }
trap cleanup EXIT

timeout --signal=TERM --kill-after=5 "$LIMIT" "$@" &
pid=$!
while kill -0 "$pid" 2>/dev/null; do
  if [[ -s $fault ]]; then
    echo "gpu_run: xe fault: $(cat "$fault")" >&2
    kill -TERM "$pid" 2>/dev/null; sleep 3; kill -KILL "$pid" 2>/dev/null || true
    break
  fi
  sleep 0.5
done
rc=0; wait "$pid" || rc=$?
[[ $rc == 124 ]] && echo "gpu_run: time limit ${LIMIT}s reached" >&2
exit "$rc"
