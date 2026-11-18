#!/usr/bin/env bash
# Standby servers end to end: a private daemon (like smoke.sh: own socket, port 7872, a card it sees as unused) and
# llm/b70's standby code around a fake model server (test/standby_harness.py), so no GPU. Settle and quiet times are
# 3 s here (settle 0 s and quiet 60 s for real; B70Q_STANDBY_SETTLE_S=0 test/standby.sh runs it as it is for real). Safe while the real queue runs.
# usage: test/standby.sh
set -u
cd "$(dirname "$0")/.."
T=$(mktemp -d /tmp/b70q-standby.XXXX)
export B70Q_STATE=$T/state B70Q_SOCKET=$T/b70q.sock B70Q_PORT=7872 B70Q_LEGACY_LOCK=off B70Q_TEST_EMPTY_CARD=1 \
  B70Q_STANDBY_SETTLE_S=${B70Q_STANDBY_SETTLE_S:-3} B70Q_STANDBY_QUIET_S=3 B70_LEASE_POLL=600 \
  B70_KEEP_LOG=1    # the checks below count a whole run's log lines; "the log starts over" has its own check
unset B70Q_TICKET
L=$PWD/bin/b70-lease Q=$PWD/bin/b70q DB=$T/state/b70q.db
PORT=$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
bun src/daemon.ts > $T/daemon.log 2>&1 < /dev/null & D=$!
H=
trap '[[ -n $H ]] && kill $H 2>/dev/null; kill $D 2>/dev/null; wait $D 2>/dev/null; [[ -n ${KEEP:-} ]] && echo "kept $T" || rm -rf $T' EXIT
sleep 2
fails=0
check() { if [[ "$2" == "$3" ]]; then echo "ok   $1"; else echo "FAIL $1: got [$2] want [$3]"; fails=$((fails + 1)); fi; }
q() { sqlite3 $DB "$1"; }
within() { local end=$((SECONDS + $1)); shift; until "$@"; do ((SECONDS >= end)) && return 1; sleep 0.2; done; }
newest() { q "SELECT $1 FROM tickets WHERE owner='nq-test' ORDER BY id DESC LIMIT 1"; }
logged() { grep -q "$1" "$2"; }
granted() { [[ $(q "SELECT state FROM tickets WHERE owner='$1' ORDER BY id DESC LIMIT 1") == granted ]]; }
loads() { [[ $(grep -c "loaded again" $T/serve.log) -ge $1 ]]; }
granted_none() { [[ $(q "SELECT COUNT(*) FROM tickets WHERE state='granted'") == 0 ]]; }
# a streamed completion asking for prompt_progress, through the server's port ($CHAT_PORT, default $PORT); the fake
# answers after <delay> s
chat() {
  curl -sN -m 120 "http://127.0.0.1:${CHAT_PORT:-$PORT}/v1/chat/completions" -H 'content-type: application/json' -d \
    "{\"model\":\"fake-model\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":true,\"return_progress\":true,\"tag\":\"$1\",\"delay\":$2}"
}
state() { curl -s -m 5 "http://127.0.0.1:7872/api/state" | python3 -c "import json, sys; s = json.load(sys.stdin); print($1)"; }
answered() { echo "$(grep -c '"hello"' <<<"$1") $(grep -c '\[DONE\]' <<<"$1")"; }

python3 test/standby_harness.py $PORT nq-test $T/fake.log > $T/serve.log 2>&1 & H=$!
within 30 logged "ready on" $T/serve.log; check "first load opens the port" $? 0
check "first load: a standby ticket in line like any (granted)" "$(newest "state || ' ' || (opts LIKE '%\"standby\":true%') || (opts LIKE '%when_free%')")" "granted 10"
check "a request goes through" "$(answered "$(chat turn1 0)")" "1 1"
sleep 4; check "keeps the card while nobody waits" "$(newest state)" granted

# a job comes; nothing has run on the server for 3 s: it steps aside, then loads again once the card has been free 3 s
sb=$(newest id)
$L run t-job -- sleep 2 2>/dev/null; check "the job runs" $? 0
check "the server stepped aside for it" "$(q "SELECT state || ' ' || (end_reason LIKE 'yielded to #% t-job') FROM tickets WHERE id=$sb")" "done 1"
[[ ${B70Q_STANDBY_SETTLE_S:-3} != 0 ]] && check "...and waits for a free card" "$(newest "state || ' ' || (opts LIKE '%when_free%')")" "waiting 1"
check "b70-lease waiters doesn't count it" "$($L waiters)" 0
# the standby model is not in the queue list: it is reported on its own, as unloaded while it waits and loaded after
if [[ ${B70Q_STANDBY_SETTLE_S:-3} != 0 ]]; then
  check "waiting standby: not in the queue list" "$(state "len(s['queue'])")" 0
  check "...reported as standby, unloaded" "$(state "len(s['standby']), s['standby'][0]['loaded']")" "1 False"
fi
within 20 loads 1; check "loads again once the card is free" $? 0
check "loaded standby: reported loaded, not in the queue list" "$(state "len(s['queue']), s['standby'][0]['loaded'], s['holder']['id'] == s['standby'][0]['id']")" "0 True True"
within 10 logged '"tag": "turn1", "max_tokens": 1' $T/fake.log; check "reads the last prompt again after the reload" $? 0

# a request in flight keeps the card: a job waits until it is over (and 3 s of quiet)
chat long 6 > $T/long.out & C=$!
sleep 1
$L run t-job2 -- true 2>/dev/null
wait $C
check "the request in flight finishes" "$(answered "$(cat $T/long.out)")" "1 1"
tl=$(grep '"long"' $T/fake.log | python3 -c 'import json, sys; print(json.loads(sys.stdin.readline())["t"])')
check "...and the job got the card after it" "$(q "SELECT granted_at FROM tickets WHERE owner='t-job2'" | awk -v t="$tl" '{ print ($1 / 1000 - t >= 6) }')" 1

# while it is unloaded: discovery answers, a request waits with keep-alives and puts it at the back of the line, with no
# priority for an nq… owner (Greg, 2026-11-17): after the job already waiting
within 20 loads 2; check "loads again after that job" $? 0
$L run t-long -- sleep 8 2>/dev/null & J=$!
within 15 granted t-long; check "a long job gets the card" $? 0
$L run t-other -- sleep 1 2>/dev/null & J2=$!
sleep 0.5
check "discovery while unloaded" "$(curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" | grep -c fake-model)" 1
chat held 0 > $T/held.out & C=$!
sleep 2
check "a waiting request puts it in line, no priority" "$(newest "state || ' ' || (opts LIKE '%when_free%') || ' ' || priority")" "waiting 0 0"
check "...behind the job that was already waiting" "$(state "[t['owner'] for t in s['queue']]")" "['t-other', 'nq-test']"
check "...with keep-alive progress on its stream" "$(grep -c '"prompt_progress".*"processed": 0' $T/held.out)" 1
wait $C $J $J2
check "the request is answered after the reload" "$(answered "$(cat $T/held.out)")" "1 1"
check "...which started right after that job (no settle wait)" "$(q "SELECT b.granted_at - a.ended_at BETWEEN 0 AND 2500 FROM tickets a, tickets b WHERE a.owner='t-other' AND b.id=(SELECT MAX(id) FROM tickets WHERE owner='nq-test')")" 1

# pausing the queue: it steps aside once quiet and doesn't load again until resumed
within 20 loads 3; check "loads again after the long job" $? 0
sleep 1; $Q pause testing >/dev/null
within 15 granted_none; check "a pause makes it step aside" $? 0
check "...with that reason" "$(q "SELECT end_reason FROM tickets WHERE owner='nq-test' AND end_reason LIKE 'yielded:%' ORDER BY id DESC LIMIT 1")" "yielded: the queue is paused"
sleep 5; check "...and it stays unloaded while paused" "$(newest state)" waiting
$Q resume >/dev/null; within 15 loads 4; check "loads again after resume" $? 0

kill -TERM $H; wait $H; check "SIGTERM stops it, exit 0" $? 0; H=
check "...leaving no ticket behind" "$(q "SELECT COUNT(*) FROM tickets WHERE state IN ('waiting','granted')")" 0

# several standby servers, one per owner (Greg, 2026-11-17): the one loaded last comes back whenever the card is free;
# another loads only for a request, at the back of the line, and is the one that comes back after that
freeport() { python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'; }
gone() { ! kill -0 "$1" 2>/dev/null; }
last() { q "SELECT $2 FROM tickets WHERE owner='$1' ORDER BY id DESC LIMIT 1"; }
models() { state "' '.join(f\"{b['owner']}:{'loaded' if b['loaded'] else 'line' if b.get('position') else 'current' if b['current'] else 'off'}\" for b in s['standby'])"; }
P1=$(freeport); P2=$(freeport); P3=$(freeport); P5=$(freeport)
python3 test/standby_harness.py $P1 nq-one $T/fake1.log > $T/one.log 2>&1 & H1=$!; H=$H1
within 30 logged "ready on" $T/one.log; check "several: the first standby loads" $? 0
python3 test/standby_harness.py $P2 nq-two $T/fake2.log > $T/two.log 2>&1 & H2=$!; H=$H2
sleep 1
check "...a second one (another owner) goes in line, no priority" "$(last nq-two "state || ' ' || priority")" "waiting 0"
within 30 logged "ready on" $T/two.log; check "...and loads once the first steps aside" $? 0
check "...the first one's server is still up" "$(kill -0 $H1 2>/dev/null; echo $?)" 0
check "...waiting for a free card" "$(last nq-one "state || ' ' || (opts LIKE '%when_free%')")" "waiting 1"
check "...the queue reports both, the second loaded" "$(models)" "nq-two:loaded nq-one:off"

$L run t-job4 -- sleep 2 2>/dev/null
within 20 granted nq-two; check "after a job the one loaded last comes back" $? 0
sleep 5; check "...and stays: the other doesn't load on its own" "$(last nq-one state)/$(last nq-two state)" "waiting/granted"

CHAT_PORT=$P1 chat wake 0 > $T/wake.out & C=$!
within 5 bash -c "[[ \$(sqlite3 $DB \"SELECT opts LIKE '%when_free%' FROM tickets WHERE owner='nq-one' ORDER BY id DESC LIMIT 1\") == 0 ]]"
check "a request for the other one puts it in line, no priority" "$(last nq-one "state || ' ' || priority")" "waiting 0"
wait $C
check "...the loaded one steps aside and the request is answered" "$(answered "$(cat $T/wake.out)")" "1 1"
check "...by the other one, now loaded" "$(models)" "nq-one:loaded nq-two:off"
$L run t-job5 -- sleep 2 2>/dev/null
within 20 granted nq-one; check "after a job the requested one comes back, not the first" $? 0
sleep 5; check "...and stays" "$(last nq-one state)/$(last nq-two state)" "granted/waiting"

# a standby server that starts replaces its own owner's old one only: a loaded one at once, a waiting one too
python3 test/standby_harness.py $P3 nq-one $T/fake3.log > $T/three.log 2>&1 & H3=$!; H=$H3
within 10 gone $H1; check "a newer standby of the same owner stops the loaded one" $? 0
check "...its ticket ended as replaced" "$(q "SELECT state || ' ' || (end_reason LIKE 'replaced by a newer standby server (#% nq-one)') FROM tickets WHERE owner='nq-one' AND id < (SELECT MAX(id) FROM tickets WHERE owner='nq-one') ORDER BY id DESC LIMIT 1")" "done 1"
check "...the other owner's one is left alone" "$(kill -0 $H2 2>/dev/null; echo $?)" 0
within 30 logged "ready on" $T/three.log; check "...and the newer one loads" $? 0
python3 test/standby_harness.py $P5 nq-two $T/fake5.log > $T/five.log 2>&1 & H5=$!; H=$H5
within 15 gone $H2; check "a newer standby stops its owner's waiting one" $? 0
check "...whose waiting ticket was cancelled as replaced" "$(q "SELECT COUNT(*) FROM tickets WHERE owner='nq-two' AND state='cancelled' AND end_reason LIKE 'replaced by a newer standby server (#% nq-two)'")" 1
within 30 logged "ready on" $T/five.log; check "...and the newer one loads (the other steps aside)" $? 0
check "...two left, one per owner" "$(q "SELECT group_concat(owner) FROM (SELECT owner FROM tickets WHERE state IN ('waiting','granted') ORDER BY owner)")" "nq-one,nq-two"
kill -TERM $H3 $H5; wait $H3; r3=$?; wait $H5; check "SIGTERM stops them, exit 0" "$r3 $?" "0 0"; H=
check "...leaving no ticket behind" "$(q "SELECT COUNT(*) FROM tickets WHERE state IN ('waiting','granted')")" 0

# the log starts over at each load (Greg, 2026-11-16): after a reload it holds that load, nothing of the one before
P4=$(freeport)
env -u B70_KEEP_LOG python3 test/standby_harness.py $P4 nq-log $T/fake4.log >> $T/log.log 2>&1 & H4=$!; H=$H4
within 30 logged "ready on" $T/log.log; check "log: the first load is in it" $? 0
$L run t-job5 -- sleep 2 2>/dev/null
within 30 logged "loaded again" $T/log.log; check "log: it loads again after a job" $? 0
check "...and the log starts at that load" "$(head -1 $T/log.log)" "b70: loading fake-model again"
check "...with nothing of the first load left" "$(grep -c "ready on" $T/log.log)" 0
kill -TERM $H4; wait $H4; H=

# a standby server the daemon runs (it doesn't report): stopped when the GPU has been idle 3 s and a job waits,
# queued again, started again once the card is free; a kill ends it
id=$($Q serve t-sbsrv --standby -- sleep 120 2>/dev/null); sleep 1.5
check "a daemon-run standby server starts" "$(q "SELECT state FROM tickets WHERE id=$id")" granted
$L run t-job3 -- true 2>/dev/null
check "it is stopped for a job" "$(q "SELECT state || ' ' || (end_reason LIKE 'yielded to #%') FROM tickets WHERE id=$id")" "done 1"
again() { [[ $(q "SELECT COUNT(*) FROM tickets WHERE grp=$id AND state='granted'") == 1 ]]; }
within 15 again; check "...and started again once the card is free" $? 0
sleep 1   # let the just-granted server's process start (a kill in the same millisecond finds nothing to stop)
$Q kill "$(q "SELECT MAX(id) FROM tickets WHERE grp=$id")" > /dev/null; sleep 3
check "a kill ends it for good" "$(q "SELECT COUNT(*) FROM tickets WHERE state IN ('waiting','granted')")" 0

echo "$fails failed"
exit $(( fails > 0 ))
