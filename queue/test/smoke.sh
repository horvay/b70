#!/usr/bin/env bash
# End-to-end check of b70q with a private daemon (own socket, port 7871, state in a temp dir, no legacy lock, 15 s turns,
# 4 s grace). Jobs are `sleep`/`sh`, never the GPU, so this is safe while the real queue runs.
# usage: test/smoke.sh
set -u
cd "$(dirname "$0")/.."
T=$(mktemp -d /tmp/b70q-smoke.XXXX)
export B70Q_STATE=$T/state B70Q_SOCKET=$T/b70q.sock B70Q_PORT=7871 B70Q_LEGACY_LOCK=off B70Q_TURN_MIN=0.25 B70Q_GRACE_S=4 B70_LEASE_POLL=600
L=$PWD/bin/b70-lease Q=$PWD/bin/b70q DB=$T/state/b70q.db
bun src/daemon.ts > $T/daemon.log 2>&1 < /dev/null & D=$!
trap 'kill $D 2>/dev/null; wait $D 2>/dev/null; [[ -n ${KEEP:-} ]] && echo "kept $T" || rm -rf $T' EXIT
sleep 2
fails=0
check() { if [[ "$2" == "$3" ]]; then echo "ok   $1"; else echo "FAIL $1: got [$2] want [$3]"; fails=$((fails + 1)); fi; }
q() { sqlite3 $DB "$1"; }

$L run t-a -- sh -c 'exit 7' 2>/dev/null; check "exit code passes through" $? 7
out=$($L run t-a -- sh -c 'echo $B70Q_TICKET' 2>/dev/null); check "job sees its ticket" "$out" "$(q 'SELECT MAX(id) FROM tickets')"
out=$($L run outer -- sh -c "$L run inner -- echo nested" 2>/dev/null); check "nested run doesn't deadlock" "$out" nested
tok=$($L acquire t-acq --ttl 1h 2>/dev/null | tail -1 | tr -dc '0-9')
check "acquire prints a numeric token" "$([[ -n $tok ]] && echo yes)" yes
check "status shows the acquire holder" "$($L status | head -1 | grep -c "t-acq .*mode=acquire ticket=$tok")" 1
out=$($L run t-acq -- echo same-owner 2>/dev/null); check "same-owner run nests in the acquire" "$out" same-owner
$L release $tok 2>/dev/null; check "release frees the card" "$($L status | head -1)" "lease: free"

# turns: loopA loops 3 s jobs; B, C arrive at 1 s, 2 s; nq-gm (next in line) at 8 s
first=$(( $(q 'SELECT MAX(id) FROM tickets') + 1 ))
for i in 1 2; do $L run loopA -- true 2>/dev/null; done   # history: loopA loops
first=$(( $(q 'SELECT MAX(id) FROM tickets') + 1 ))
( for i in 1 2 3 4 5 6 7; do $L run loopA -- sleep 3 2>/dev/null; done ) & P1=$!
sleep 1; $L run ownerB -- sleep 3 2>/dev/null & P2=$!
sleep 1; $L run ownerC -- sleep 3 2>/dev/null & P3=$!
sleep 6; $L run nq-gm -- sleep 2 2>/dev/null & P4=$!
wait $P1 $P2 $P3 $P4   # (a bare wait would also wait for the daemon)
order=$(q "SELECT group_concat(owner, ' ') FROM (SELECT owner FROM tickets WHERE id >= $first ORDER BY granted_at)")
check "turns + next in line" "$order" "loopA loopA loopA nq-gm loopA loopA ownerB ownerC loopA loopA"
gaps=$(q "SELECT MAX(b.granted_at - a.ended_at) FROM tickets a JOIN tickets b ON b.granted_at = (SELECT MIN(granted_at) FROM tickets WHERE granted_at > a.granted_at) WHERE a.id >= $first AND a.owner IN ('ownerB','ownerC','nq-gm')")
check "no idle gap after one-off jobs" "$(( gaps < 1000 ))" 1

$L run hold -- sleep 3 2>/dev/null & H=$!; sleep 0.5
$L run waiter -- true 2>/dev/null & W=$!; sleep 1; kill -TERM $W; wait $W; check "SIGTERM while waiting exits 143" $? 143
sleep 0.3; check "...and cancels the ticket" "$(q "SELECT state FROM tickets WHERE owner='waiter'")" cancelled
$L run waiter2 -- true 2>/dev/null & W=$!; sleep 1; kill -KILL $W; wait $W 2>/dev/null; sleep 2.5
check "killed waiter is dropped" "$(q "SELECT end_reason FROM tickets WHERE owner='waiter2'")" "client gone"
wait $H
$L run term -- sh -c 'trap "exit 143" TERM; sleep 30 & wait' 2>/dev/null & W=$!; sleep 1.5; kill -TERM $W; wait $W; check "SIGTERM reaches the job" $? 143
$L run killme -- sleep 60 2>/dev/null & W=$!; sleep 1; $Q kill "$(q "SELECT id FROM tickets WHERE owner='killme'")" >/dev/null; wait $W
check "b70q kill stops a running job" $? 143

id=$(cd /tmp && FOO=bar $Q submit t-sub -- sh -c 'echo "$(pwd) $FOO $B70Q_TICKET"; exit 3' 2>/dev/null); sleep 3
check "submit runs with cwd+env" "$($Q log $id | grep -v '^#')" "/tmp bar $id"
check "submit records the exit code" "$(q "SELECT exit_code FROM tickets WHERE id=$id")" 3
id=$($Q submit t-bad -- /nonexistent 2>/dev/null); sleep 12
check "a submit that can't start fails" "$(q "SELECT state FROM tickets WHERE id=$id")" failed

$Q pause testing >/dev/null; $L run during-pause -- true 2>/dev/null & W=$!; sleep 2
check "pause holds the queue" "$(q "SELECT state FROM tickets WHERE owner='during-pause'")" waiting
$Q resume >/dev/null; wait $W; check "resume releases it" "$(q "SELECT state FROM tickets WHERE owner='during-pause'")" done
check "status format" "$($L status | sed -n '1s/^lease: .*/lease/p;2s/^vram_gib: [0-9.]*$/vram/p' | tr '\n' ' ')" "lease vram "
check "second daemon refuses to start" "$(timeout 5 bun src/daemon.ts 2>&1 | grep -c 'already running')" 1

# --- job options (phase 2/3) ---
W2=$T/work; mkdir -p $W2
touch $W2/exists
out=$($L run t-skip --creates $W2/exists -- echo ran 2>/dev/null); check "run --creates skips when the file exists" "$out|$?" "|0"
id=$(cd $W2 && $Q submit t-skip --creates exists -- sh -c 'echo ran' 2>/dev/null); $Q wait $id 2>/dev/null
check "submit --creates (relative to cwd) skips" "$(q "SELECT state || ' ' || end_reason FROM tickets WHERE id=$id")" "done skipped: exists exists"

# chunked: exits 99 until it has run 3 times; exit file + on-exit hook only once, at the very end
cat > $W2/chunk.sh <<'EOS'
n=$(( $(cat count 2>/dev/null || echo 0) + 1 )); echo $n > count; echo "chunk $n"
[ $n -lt 3 ] && exit 99; exit 0
EOS
id=$(cd $W2 && $Q submit t-chunk --chunked --log $W2/chunk.log --exit-file $W2/chunk.log.exit --on-exit 'echo "$B70Q_TICKET $B70Q_EXIT $B70Q_STATE" >> hook.out' -- sh chunk.sh 2>/dev/null)
sleep 6
check "chunked job ran 3 chunks" "$(q "SELECT COUNT(*) || ' ' || GROUP_CONCAT(chunk) FROM tickets WHERE grp=$id")" "3 0,1,2"
check "chunks share one log" "$(grep -c '^chunk' $W2/chunk.log)" 3
check "exit file written once, at the end" "$(cat $W2/chunk.log.exit)" 0
rm -f $W2/count; id=$(cd $W2 && $Q submit t-chunk2 --chunked -- sh chunk.sh 2>/dev/null); $Q wait $id 2>/dev/null; rc=$?
check "wait follows a chunked job to its last chunk" "$rc $(q "SELECT COUNT(*) FROM tickets WHERE grp=$id AND state='done'")" "0 3"
check "on-exit hook ran once" "$(wc -l < $W2/hook.out | tr -d ' ') $(cut -d' ' -f2,3 $W2/hook.out)" "1 0 done"

# retries: fails once, then succeeds
cat > $W2/flaky.sh <<'EOS'
[ -f tried ] && exit 0; touch tried; exit 1
EOS
id=$(cd $W2 && $Q submit t-retry --retries 2 -- sh flaky.sh 2>/dev/null); $Q wait $id 2>/dev/null
check "retry after a failure" "$(q "SELECT GROUP_CONCAT(state || ':' || attempt) FROM tickets WHERE grp=$id")" "failed:0,done:1"
id=$($Q submit t-killed --retries 3 -- sleep 60 2>/dev/null); sleep 1.5; $Q kill $id >/dev/null; sleep 2
check "no retry after a kill" "$(q "SELECT COUNT(*) FROM tickets WHERE grp=$id")" 1
rm -f $W2/tried; id=$(cd $W2 && $Q submit t-retry2 --retries 2 -- sh flaky.sh 2>/dev/null); $Q wait $id 2>/dev/null; rc=$?
check "wait follows retries to the final result" "$rc $(q "SELECT COUNT(*) FROM tickets WHERE grp=$id")" "0 2"

# an app's contract: caller-chosen log and <log>.exit, and b70q wait returns the exit code
id=$($Q submit t-logs --log $W2/d.log --exit-file $W2/d.log.exit -- sh -c 'echo logged; exit 5' 2>/dev/null)
$Q wait $id 2>/dev/null; rc=$?
check "wait returns the exit code" $rc 5
check "log and exit file where the caller asked" "$(grep -cx logged $W2/d.log) $(cat $W2/d.log.exit)" "1 5"

# servers: an idle server keeps the card while nobody waits, and is stopped once someone does
id=$($Q serve t-srv --idle 3s -- sleep 120 2>/dev/null); sleep 6
check "idle server keeps the card when nobody waits" "$(q "SELECT state FROM tickets WHERE id=$id")" granted
$L run t-after-srv -- true 2>/dev/null; rc=$?
check "idle server stopped for a waiting job" "$rc $(q "SELECT state FROM tickets WHERE id=$id") $(q "SELECT end_reason FROM tickets WHERE id=$id" | cut -c1-19)" "0 done idle server stopped"

# host RAM gate
id=$($Q submit t-ram --ram-gib 100000 -- true 2>/dev/null); sleep 2
check "ram gate holds the job" "$(q "SELECT state FROM tickets WHERE id=$id") $(curl -s --unix-socket $B70Q_SOCKET http://x/api/state | grep -o '"kind":"ram"' | head -1)" 'waiting "kind":"ram"'
$Q cancel $id >/dev/null

echo "$fails failed"
exit $(( fails > 0 ))
