# b70q: the queue for the Intel Arc Pro B70

Every job on the B70 goes through this queue. One job at a time, strictly first in, first out (Greg, 2026-11-17): the
queue has no turns and no rules by owner. Only the submitter can move a job up, with `--next` or `b70q now`; those
jobs go ahead of the rest, in arrival order among themselves. A chunk or retry queues at the back like a new job, so a
series that submits its next batch after the last one ends waits behind everyone who came meanwhile. Nothing is ever
preempted. The queue waits for anything using the card outside it (Greg's manual runs, leftovers of a killed job). A
standby server is a model that stays loaded: loaded whenever nothing else wants the card, unloaded when a job comes and
it isn't in use. Several can be up; the one loaded last is the one that comes back (see "Standby" below).

- Daemon: `systemctl --user status b70q` (unit `~/.config/systemd/user/b70q.service`, code `~/b70/queue/src`)
- Dashboard: http://127.0.0.1:7870/: the job on the card, the queue, live card charts, alerts, and below them the
  output of the running job (or of the last one): the file its processes write to, found by the daemon, so a job whose
  real work logs to a file of its own shows that. Past jobs and learned job types are behind the History button.
- API: unix socket `$XDG_RUNTIME_DIR/b70q.sock` (`/run/user/1000/b70q.sock`); read-only and controls on the dashboard port
- State: `~/.local/state/b70q/` (SQLite history, logs of submitted jobs)
- Tests: `test/smoke.sh` (private daemon on port 7871, `sleep` jobs only) and `test/standby.sh` (port 7872, standby
  servers around a fake model server); both safe while the real queue runs

## Which command

| Work | Command |
|---|---|
| A job inside a script you're running | `b70-lease run <owner> [options] -- <cmd>` (blocks; returns the job's exit code) |
| A batch that should outlive your shell or turn | `b70q submit <owner> [options] -- <cmd>` (returns an id at once; `b70q wait <id>` blocks until the whole job, all chunks and retries, is over and exits with its code; `b70q log <id> -f`) |
| Long work (hours) | `b70q submit <owner> --chunked -- <cmd>`: the command does about 20–30 min of work and exits **99** if more remains, **0** when everything is done. Each chunk goes back in line. |
| A server (LLM, image server) | `b70q serve <owner> --idle 15m -- <cmd>`, or `b70-lease run <owner> --idle 15m -- <cmd>` if your own process must own it |
| The card's idle model (several can wait, one per owner) | `b70 serve … --standby` (LLMs); `b70q serve <owner> --standby -- <cmd>` for any other server |
| Holding the card across several tool calls | `b70-lease acquire <owner> --ttl 1h` → token; `export B70Q_TICKET=<token>`; `b70-lease release <token>` |
| Your own run, next in line | `b70q now -- <cmd>` |
| The card for yourself once the current job is done | `b70q pause [reason]` (dashboard: Pause queue): the job on the card finishes, idle and standby servers step aside once quiet, nothing new starts. Run your own thing, then `b70q resume`; the queue waits for anything of yours still on the card. |

Don't build queues on top: no per-project daemons, no `sleep 300` gaps between holds, no `pgrep` chains, no waiting for
log lines, no killing other jobs by process name. Submit everything and let the queue order it.

## Options

| Option | Meaning |
|---|---|
| `--note TEXT` | Label shown in the queue and history ("scene 12 H3 v3") |
| `--creates PATH` | Skip the job (done, exit 0) when PATH already exists. Relative to the job's directory. |
| `--retries N` | (submit) Run again after a failure (crash, OOM, GGML_ASSERT), up to N times, from the back of the line. Never after a kill. |
| `--chunked` / `--more-code N` | (submit) Exit code 99 (or N) means "more to do": the same command is queued again. |
| `--no-turn` | (old) No effect: every chunk or retry already waits behind everything queued before it. |
| `--log PATH` | (submit) Output goes here instead of `~/.local/state/b70q/logs/<id>.log`. |
| `--exit-file PATH` | The exit code (or the end reason, e.g. `cancelled`) is written here when the job is over. |
| `--on-exit 'shell cmd'` | Run when the job is over (after the last chunk/retry), in the job's directory, with `B70Q_TICKET`, `B70Q_EXIT`, `B70Q_STATE` (done/failed/cancelled), `B70Q_REASON`, `B70Q_OWNER`, `B70Q_LOG`. Use it to wake an agent. |
| `--idle DUR` / `--server` | A server: stopped (SIGTERM) after DUR (default 15m) with nothing running on the GPU, **only while another job waits** (or the queue is paused). Idle with an empty queue, it keeps the card. |
| `--standby` | A server that is the card's idle model (see "Standby"). |
| `--ram-gib N` | Start only when N GiB of host RAM is available (pinned-RAM jobs like DeepSeek). |
| `--vram-gib N` | Expected VRAM, shown in the queue (the daemon also learns it from past runs). |
| `--next` | Next in line: ahead of every job without `--next`. The only way to skip ahead. |
| `--min-gib N` | (old) Start only when less than N GiB of VRAM is in use (default 1). |

Durations: `90`, `90s`, `15m`, `4h`.

## What a job sees

- `B70Q_TICKET=<id>` in its environment. A `b70-lease run` or `b70q run` inside the job runs at once under the same
  ticket (no deadlock), so scripts that take the lease themselves can be called from anywhere.
- Killing `b70-lease` (SIGTERM/SIGHUP) passes the signal to the job. `b70q kill <id>` sends SIGTERM to every process of
  the ticket and SIGKILL 15 s later. Submitted jobs run in their own unit `b70q-<id>.service`.
- The daemon measures each job (VRAM, GTT, system memory, host RAM, GPU busy, energy) and learns per job type
  (`owner:program`) how long it takes and how much it needs; `b70q stats`, dashboard History → "Job types".

## HTTP API (Bun/Node)

```ts
const sock = `${process.env.XDG_RUNTIME_DIR}/b70q.sock`;
const b70q = (method: string, path: string, body?: unknown) =>
  fetch(`http://b70q${path}`, { unix: sock, method, headers: { "content-type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body) } as RequestInit).then((r) => r.json());

// queue a job the daemon runs (like `b70q submit`)
const { id } = await b70q("POST", "/api/tickets", {
  owner: "myapp-h3", kind: "submit",
  argv: ["bash", "-c", command], cwd, env: { PATH: process.env.PATH!, HOME: process.env.HOME! },
  log: "/abs/path/job.log", note: "scene 4 H3",
  opts: { exit_file: "/abs/path/job.log.exit", creates: "out/scene4.mp4", retries: 1, on_exit: ["bash", "-c", "…"] },
});
// long-poll: returns within 8 s; loop until the whole job (all chunks and retries) has ended
const w = await b70q("GET", `/api/tickets/${id}/wait?until=ended&timeout=8`);
// w.series.active: false once the job is over; w.ticket = its last chunk/retry: .state done|failed|cancelled, .exit_code
// while it runs or waits: w.ticket.state waiting|granted, w.position (in line), w.status
```

| Endpoint | |
|---|---|
| `POST /api/tickets` | `{owner, kind: run\|acquire\|submit, argv, cwd, env (submit), client_pid (run/acquire: the process that holds it), log, note, priority, min_gib, ttl_s (acquire), parent (B70Q_TICKET for nesting), opts: {creates, retries, more_code, exit_file, on_exit, idle_s, server, ram_gib, vram_gib}}` → `{id, nested?}` |
| `GET /api/tickets/:id` | `{ticket, position, series: {grp, ids, active, latest}}`. Chunks and retries are new tickets with the same `grp`; use `series.active` for "is the job still going" and `series.latest` for its result. |
| `GET /api/tickets/:id/wait?until=granted\|ended&timeout=8` | long-poll; `until=ended` follows the job's chunks and retries to its last ticket |
| `GET /api/tickets/:id/log?offset=N` | output of a submitted job (`offset=-4096` for the tail); `&src=output`: the file the job's processes write to (`ticket.output`, what the dashboard shows), else that log |
| `POST /api/tickets/:id/{done,started,release,cancel,kill,bump}` | `done` takes `{exit_code, reason?}` (run tickets report their own end) |
| `POST /api/tickets/:id/activity` | a standby holder's report every few seconds, `{busy, idle_s}` (requests in flight or waiting for it; seconds since the last one ended) → `{yield, reason?}`: step aside now, then end the ticket with that reason |
| `POST /api/tickets/:id/urgent` | a standby ticket waiting for a free card has a request waiting for it: it goes in line like any job |
| `GET /api/state` | card, holder with live usage, queue with ETAs, outside processes, alerts |
| `GET /api/history?hours=24&limit=500`, `/api/stats`, `/api/samples?hours=24` | history, learned job types, 30 s card samples |
| `POST /api/pause {reason}`, `/api/resume`, `/api/alerts/:id/ack` | |
| `ws://127.0.0.1:7870/ws` | `hello` (state + 1 h of samples), then `state` and `sample` messages |

On the dashboard port, POSTs need the header `x-b70q: 1`, and `submit` is refused (use the socket).

## A server on the queue

```sh
# a chat model server, stopped after 15 min idle on the GPU when someone else waits (add --next to skip the line).
b70-lease run nq-gm --idle 15m -- b70 serve …          # your process owns the server; exits when it's stopped
b70q serve nq-gm --idle 15m -- b70 serve …             # the daemon runs it in b70q-<id>.service
```

The caller knows the server has the card when its ticket is granted (`/api/tickets/:id`), then polls the server's own
health endpoint. When the daemon stops an idle server, the ticket ends as `done` with reason `idle server stopped …`;
start it again (same command) the next time it's needed. It waits in line again.

## Standby: the card's idle model

A standby server (option `standby`; a chat model server) holds the card only while nothing else wants
it (Greg, 2026-10-23):

- Its first ticket waits in line like any server's, at the back: a standby server never has priority, even with
  `--next` (Greg, 2026-11-17). After that it waits at the back (`when_free`) and, if it is the current standby model, is granted
  as soon as nothing else waits (`B70Q_STANDBY_SETTLE_S`, default 0, can add a wait). It is not part of the queue: `/api/state` reports it in `standby`
  (`loaded`, `busy`, `idle_ms`), `b70q` prints a "Standby" section, and the dashboard shows a "Standby model" strip.
- Several standby servers can be up, one per owner (Greg, 2026-11-17). The current one is the one that loaded last
  (kept in the `standby_order` meta row, so a daemon restart keeps it): it is the one that loads when the card is free.
  The others stay unloaded until a request comes for one; once loaded, that one is the current one. The card never
  switches between them on its own. A standby server that starts replaces its owner's old one, loaded or waiting,
  mid-request too (ended as `replaced by a newer standby server (#id owner)`); a server waiting to load again after
  stepping aside does not count as starting.
- When another job waits (or a process outside the queue starts on the card, or the queue is paused) and the server
  has had no request for 1 minute (`B70Q_STANDBY_QUIET_S`), it steps aside: the ticket ends as `done`, reason
  `yielded to #N owner` (`yielded: the queue is paused`), and it waits at the back again. `b70 serve --standby` reports whether it is in use (`activity`), so a request in flight is
  never cut off; a server that doesn't report (`b70q serve --standby`) is judged by the GPU and stopped (SIGTERM), and
  the daemon queues it again itself. A reporting server that doesn't step aside within a minute of being told is
  stopped too.
- A request for an unloaded `b70 serve --standby` makes its waiting ticket `urgent`: at the back of the line like any
  job, no priority. A loaded standby server steps aside for it like for any job (after its quiet minute).
- `b70-lease waiters` and `status` don't count a standby ticket waiting for a free card.

How `b70 serve --standby` keeps its port open while the model is unloaded: `../llm/README.md`. Tests:
`test/standby.sh` (a private daemon and `b70 serve`'s standby code around a fake model server; no GPU).
