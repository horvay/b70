# b70q: the queue for the Intel Arc Pro B70

Every job on the B70 goes through this queue. One job at a time, first come first served, with 30-minute turns: an
owner whose jobs come in a loop keeps the card for its next job while its turn lasts, then the next owner in line gets
it. Owners named `nq…` (Neverending Quest's game master) and `b70q now` go next in line. Nothing is ever preempted.
The queue waits for anything using the card outside it (Greg's manual runs, leftovers of a killed job).

- Daemon: `systemctl --user status b70q` (unit `~/.config/systemd/user/b70q.service`, code `~/b70/queue/src`)
- Dashboard: http://127.0.0.1:7870/
- API: unix socket `$XDG_RUNTIME_DIR/b70q.sock` (`/run/user/1000/b70q.sock`); read-only and controls on the dashboard port
- State: `~/.local/state/b70q/` (SQLite history, logs of submitted jobs)
- Tests: `test/smoke.sh` (private daemon on port 7871, `sleep` jobs only; safe while the real queue runs)

## Which command

| Work | Command |
|---|---|
| A job inside a script you're running | `b70-lease run <owner> [options] -- <cmd>` (blocks; returns the job's exit code) |
| A batch that should outlive your shell or turn | `b70q submit <owner> [options] -- <cmd>` (returns an id at once; `b70q wait <id>` blocks until the whole job, all chunks and retries, is over and exits with its code; `b70q log <id> -f`) |
| Long work (hours) | `b70q submit <owner> --chunked -- <cmd>`: the command does about 20–30 min of work and exits **99** if more remains, **0** when everything is done. Each chunk goes back in line. |
| A server (LLM, image server) | `b70q serve <owner> --idle 15m -- <cmd>`, or `b70-lease run <owner> --idle 15m -- <cmd>` if your own process must own it |
| Holding the card across several tool calls | `b70-lease acquire <owner> --ttl 1h` → token; `export B70Q_TICKET=<token>`; `b70-lease release <token>` |
| Your own run, next in line | `b70q now -- <cmd>` |

Don't build queues on top: no per-project daemons, no `sleep 300` gaps between holds, no `pgrep` chains, no waiting for
log lines, no killing other jobs by process name. Submit everything and let the queue order it.

## Options

| Option | Meaning |
|---|---|
| `--note TEXT` | Label shown in the queue and history ("scene 12 H3 v3") |
| `--creates PATH` | Skip the job (done, exit 0) when PATH already exists. Relative to the job's directory. |
| `--retries N` | (submit) Run again after a failure (crash, OOM, GGML_ASSERT), up to N times, from the back of the line. Never after a kill. |
| `--chunked` / `--more-code N` | (submit) Exit code 99 (or N) means "more to do": the same command is queued again. |
| `--log PATH` | (submit) Output goes here instead of `~/.local/state/b70q/logs/<id>.log`. |
| `--exit-file PATH` | The exit code (or the end reason, e.g. `cancelled`) is written here when the job is over. |
| `--on-exit 'shell cmd'` | Run when the job is over (after the last chunk/retry), in the job's directory, with `B70Q_TICKET`, `B70Q_EXIT`, `B70Q_STATE` (done/failed/cancelled), `B70Q_REASON`, `B70Q_OWNER`, `B70Q_LOG`. Use it to wake an agent. |
| `--idle DUR` / `--server` | A server: stopped (SIGTERM) after DUR (default 15m) with nothing running on the GPU, **only while another job waits**. Idle with an empty queue, it keeps the card. |
| `--ram-gib N` | Start only when N GiB of host RAM is available (pinned-RAM jobs like DeepSeek). |
| `--vram-gib N` | Expected VRAM, shown in the queue (the daemon also learns it from past runs). |
| `--next` | Next in line. |
| `--min-gib N` | (old) Start only when less than N GiB of VRAM is in use (default 1). |

Durations: `90`, `90s`, `15m`, `4h`.

## What a job sees

- `B70Q_TICKET=<id>` in its environment. A `b70-lease run` or `b70q run` inside the job runs at once under the same
  ticket (no deadlock), so scripts that take the lease themselves can be called from anywhere.
- Killing `b70-lease` (SIGTERM/SIGHUP) passes the signal to the job. `b70q kill <id>` sends SIGTERM to every process of
  the ticket and SIGKILL 15 s later. Submitted jobs run in their own unit `b70q-<id>.service`.
- The daemon measures each job (VRAM, GTT, system memory, host RAM, GPU busy, energy) and learns per job type
  (`owner:program`) how long it takes and how much it needs; `b70q stats`, dashboard "Job types".

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
| `GET /api/tickets/:id/log?offset=N` | output of a submitted job (`offset=-4096` for the tail) |
| `POST /api/tickets/:id/{done,started,release,cancel,kill,bump}` | `done` takes `{exit_code}` (run tickets report their own end) |
| `GET /api/state` | card, holder with live usage, queue with ETAs, outside processes, alerts |
| `GET /api/history?hours=24&limit=500`, `/api/stats`, `/api/samples?hours=24` | history, learned job types, 30 s card samples |
| `POST /api/pause {reason}`, `/api/resume`, `/api/alerts/:id/ack` | |
| `ws://127.0.0.1:7870/ws` | `hello` (state + 1 h of samples), then `state` and `sample` messages |

On the dashboard port, POSTs need the header `x-b70q: 1`, and `submit` is refused (use the socket).

## A server on the queue

```sh
# NQ's game master: next in line (owner nq-…), stopped after 15 min idle on the GPU when someone else waits.
b70-lease run nq-gm --idle 15m -- b70 serve …          # your process owns the server; exits when it's stopped
b70q serve nq-gm --idle 15m -- b70 serve …             # the daemon runs it in b70q-<id>.service
```

The caller knows the server has the card when its ticket is granted (`/api/tickets/:id`), then polls the server's own
health endpoint. When the daemon stops an idle server, the ticket ends as `done` with reason `idle server stopped …`;
start it again (same command) the next time it's needed. It goes next in line again.
