// b70q: look at and steer the B70 queue.
//
//   b70q [ls]                            what the card is doing, the queue, alerts
//   b70q run <owner> [options] -- <cmd>  queue a job and run it here (same as b70-lease run)
//   b70q now [--owner greg] -- <cmd>     run next in line (ahead of the queue, after the current job)
//   b70q submit <owner> [options] -- <cmd>   queue a job the daemon runs on its own (survives this shell); prints its id
//       --creates PATH    skip when PATH exists          --retries N        run again after a failure (crash, OOM)
//       --chunked         exit 99 = more to do: run again from the back of the line (--more-code N for another code)
//       --log PATH        output here (default ~/.local/state/b70q/logs/<id>.log)   --exit-file PATH  exit code here
//       --on-exit 'cmd'   run when the job (all chunks/retries) is over; env B70Q_TICKET B70Q_EXIT B70Q_STATE B70Q_LOG
//       --ram-gib N       start only when N GiB host RAM is free     --vram-gib N   expected VRAM (shown)
//       --note TEXT       label in the queue      --next   next in line      --min-gib N   (old) max VRAM in use at start
//   b70q serve <owner> [--idle 15m] [options] -- <cmd>   a server: stopped after --idle on the GPU while others wait
//   b70q wait <id>                       block until a ticket ends; exits with its exit code
//   b70q log <id> [-f]                   output of a submitted job
//   b70q acquire <owner> [--ttl 4h] | release <id>
//   b70q kill <id> | cancel <id> | bump <id>
//   b70q pause [reason] | resume | ack <alert-id>
//   b70q history [n] | stats | web
import { PORT } from "./config";
import { api, dur, gib, up } from "./client";
import { acquire, ACQUIRE_FLAGS, parseFlags, run, RUN_FLAGS, SUBMIT_FLAGS } from "./lease";

const C = { dim: "\x1b[2m", b: "\x1b[1m", red: "\x1b[31m", yel: "\x1b[33m", grn: "\x1b[32m", cyan: "\x1b[36m", off: "\x1b[0m" };
const tty = process.stdout.isTTY;
const c = (k: keyof typeof C, s: string) => (tty ? C[k] + s + C.off : s);
const clock = (t: number | null | undefined) => (t ? new Date(t).toLocaleTimeString("en-GB", { hour12: false }) : "-");
const cmdText = (argv: string[]) => argv.join(" ").replace(/\s+/g, " ").slice(0, 90);

function table(rows: string[][]) {
  // eslint-disable-next-line no-control-regex
  const w = rows[0].map((_, i) => Math.max(...rows.map((r) => (r[i] ?? "").replace(/\x1b\[\d+m/g, "").length)));
  // eslint-disable-next-line no-control-regex
  for (const r of rows) console.log(r.map((x, i) => x + " ".repeat(w[i] - x.replace(/\x1b\[\d+m/g, "").length)).join("  ").trimEnd());
}

async function ls() {
  const s = await api("GET", "/api/state");
  const k = s.card;
  const kind = s.status.kind;
  const color = kind === "running" ? "grn" : kind === "idle" ? "dim" : kind === "fault" || kind === "paused" ? "red" : "yel";
  console.log(`${c("b", "B70")}  ${c(color, kind.toUpperCase())}  ${s.status.text}`);
  if (k) console.log(c("dim", `VRAM ${gib(k.vramUsed)}/${gib(k.vramTotal)} GiB  ${k.power?.toFixed(0) ?? "?"} W  ${k.temp ?? "?"}°C  busy ${((k.util ?? 0) * 100).toFixed(0)}%  host RAM free ${gib(k.memAvail)} GiB`));
  const h = s.holder;
  if (h) {
    const u = h.usage ?? {};
    console.log(`\n${c("b", "Now")}  #${h.id} ${c("cyan", h.owner)} (${h.kind}) for ${dur(s.t - h.granted_at)}${h.est_ms ? c("dim", ` of ~${dur(h.est_ms)}`) : ""}`);
    if (h.note) console.log(`  ${h.note}`);
    if (h.argv.length) console.log(`  ${c("dim", cmdText(h.argv))}`);
    console.log(`  VRAM ${gib(u.vram)} GiB  GTT ${gib(u.gtt)}  sys ${gib(u.sys)}  RAM ${gib(u.rss)} GiB  GPU ${((u.busy ?? 0) * 100).toFixed(0)}%` +
      (s.turn && s.turn.owner === h.owner ? c("dim", `  turn left ${dur(s.turn.left_ms)}`) : ""));
  }
  if (s.queue.length) {
    console.log(`\n${c("b", "Queue")}`);
    table([["", "#", "owner", "kind", "waiting", "starts ~", "takes ~", "command"].map((x) => c("dim", x)),
      ...s.queue.map((t: any) => [String(t.position), String(t.id), (t.priority ? c("yel", "▲ ") : "") + t.owner,
        t.kind + (t.opts?.server ? " server" : "") + (t.chunk ? ` chunk ${t.chunk + 1}` : "") + (t.attempt ? ` retry ${t.attempt}` : ""),
        dur(s.t - t.created_at), clock(t.eta_start), dur(t.est_ms), (t.note ? `${t.note}: ` : "") + cmdText(t.argv)])]);
  }
  if (s.others.length) {
    console.log(`\n${c("b", "Also on the card")}`);
    for (const o of s.others) console.log(`  ${c("yel", o.why === "outside" ? "outside the queue" : `leftover of #${o.ticket}`)}  pid ${o.pid} ${o.comm}  ${gib(o.vram)} GiB  ${c("dim", o.cmd.slice(0, 80))}`);
  }
  if (s.alerts.length) {
    console.log(`\n${c("b", "Alerts")}`);
    for (const a of s.alerts.slice(0, 8)) console.log(`  ${c(a.level === "error" ? "red" : a.level === "warn" ? "yel" : "dim", a.level.padEnd(5))} ${clock(a.t)}  ${a.msg}  ${c("dim", `(ack ${a.id})`)}`);
  }
  console.log(c("dim", `\ndashboard http://127.0.0.1:${PORT}/`));
}

async function history(n: number) {
  const rows = (await api("GET", `/api/history?hours=168&limit=${n}`)) as any[];
  table([["#", "owner", "state", "queued", "waited", "ran", "VRAM", "RAM", "kWh", "result", "command"].map((x) => c("dim", x)),
    ...rows.map((t) => [String(t.id), t.owner, t.state, clock(t.created_at), dur((t.granted_at ?? t.ended_at ?? Date.now()) - t.created_at),
      t.granted_at ? dur((t.ended_at ?? Date.now()) - t.granted_at) : "-", gib(t.peak_vram), gib(t.peak_rss), (t.energy_j / 3.6e6).toFixed(2),
      t.state === "granted" || t.state === "waiting" ? "" : `${t.exit_code ?? ""} ${t.end_reason ?? ""}`.trim(), cmdText(t.argv)])]);
}

async function stats() {
  const rows = (await api("GET", "/api/stats")) as any[];
  table([["job type", "runs", "failed", "median", "p90", "VRAM p90", "VRAM max", "RAM max", "last"].map((x) => c("dim", x)),
    ...rows.map((s) => [s.jobtype, String(s.n), String(s.failed), dur(s.median_ms), dur(s.p90_ms), gib(s.p90_vram), gib(s.max_vram), gib(s.max_rss), new Date(s.last).toLocaleString("sv-SE").slice(5, 16)])]);
}

async function log(id: number, follow: boolean) {
  let off = 0;
  for (;;) {
    const r = await api("GET", `/api/tickets/${id}/log?offset=${off}`);
    process.stdout.write(r.text); off = r.offset;
    if (!follow) return;
    const t = (await api("GET", `/api/tickets/${id}`)).ticket;
    if (t.state !== "waiting" && t.state !== "granted" && off >= r.size) return;
    await Bun.sleep(1000);
  }
}

const flags = (rest: string[], allow: Set<string>) => { const f = parseFlags(rest, allow); if (!f) process.exit(2); return f; };

async function waitEnded(id: number): Promise<number> {
  for (;;) {
    const w = await api("GET", `/api/tickets/${id}/wait?until=ended&timeout=8`, undefined, 20_000).catch(() => null);
    if (!w) { await Bun.sleep(2000); continue; }
    const t = w.ticket; // the last ticket of the job (its last chunk or retry)
    if (!w.series?.active && t.state !== "waiting" && t.state !== "granted") {
      const n = w.series?.ids?.length ?? 1;
      console.error(`#${id} ${t.owner}: ${t.state} (${t.end_reason ?? ""})${n > 1 ? ` after ${n} runs, last #${t.id}` : ""}`);
      return t.exit_code ?? (t.state === "done" ? 0 : 1);
    }
  }
}

const [cmd = "ls", ...rest] = process.argv.slice(2);
if (cmd === "web") { console.log(`http://127.0.0.1:${PORT}/`); process.exit(0); }
if (!(await up())) { console.error("b70q: the daemon isn't running (systemctl --user start b70q)"); process.exit(1); }
const id = Number(rest[0]);
const post = async (path: string, body?: unknown) => { try { await api("POST", path, body ?? {}); } catch (e) { console.error(`b70q: ${(e as Error).message}`); process.exit(1); } };

switch (cmd) {
  case "ls": case "status": await ls(); break;
  case "run": { const owner = rest.shift()!; const f = flags(rest, RUN_FLAGS); process.exit(await run(owner, rest, f)); }
  case "now": { const f = flags(rest, new Set([...RUN_FLAGS, "--owner"])); process.exit(await run(f.owner ?? "greg", rest, { ...f, priority: 1 })); }
  case "submit": case "serve": {
    const owner = rest.shift()!;
    const f = flags(rest, SUBMIT_FLAGS);
    if (cmd === "serve") { f.opts.server = true; f.opts.idle_s ??= 15 * 60; }
    if (!owner || owner.startsWith("-") || !rest.length) { console.error(`b70q ${cmd} <owner> [options] -- <cmd...>`); process.exit(2); }
    const env = Object.fromEntries(Object.entries(process.env).filter(([k]) => k !== "B70Q_TICKET")) as Record<string, string>;
    const r = await api("POST", "/api/tickets", {
      owner, kind: "submit", argv: rest, cwd: process.cwd(), env, priority: f.priority, min_gib: f.minGib, opts: f.opts, note: f.note, log: f.log,
    });
    console.log(r.id);
    console.error(`queued #${r.id}; output: b70q log ${r.id} -f; wait for it: b70q wait ${r.id}`);
    break;
  }
  case "wait": process.exit(await waitEnded(id));
  case "acquire": { const owner = rest.shift()!; const f = flags(rest, ACQUIRE_FLAGS); console.log(await acquire(owner, f.ttl ?? 4 * 3600, f)); break; }
  case "release": await post(`/api/tickets/${id}/release`); break;
  case "kill": case "cancel": case "bump": await post(`/api/tickets/${id}/${cmd}`); await ls(); break;
  case "pause": await post("/api/pause", { reason: rest.join(" ") || undefined }); await ls(); break;
  case "resume": await post("/api/resume"); await ls(); break;
  case "ack": await post(`/api/alerts/${id}/ack`); break;
  case "log": await log(id, rest.includes("-f")); break;
  case "history": await history(Number(rest[0] ?? 30)); break;
  case "stats": await stats(); break;
  default: console.error(`b70q: unknown command ${cmd}`); process.exit(2);
}
