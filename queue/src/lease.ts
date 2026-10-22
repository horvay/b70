// b70-lease, now a client of the b70q daemon. Same commands, arguments, output and exit codes as the old flock script
// (kept as legacy/b70-lease, which this falls back to when the daemon isn't running):
//
//   b70-lease run <owner> [options] -- <cmd...>   wait in the queue, run <cmd>, release when it exits
//   b70-lease acquire <owner> [--ttl 4h] [options] wait, then hold the card until released; prints the token
//   b70-lease release <token>                     end an acquired lease
//   b70-lease status | vram | waiters | yield [max_s]
//
// options: --min-gib N (old), --next, --note TEXT, --creates PATH (skip if it exists), --ram-gib N (wait for free host
// RAM), --vram-gib N (expected), --idle DUR (server: stopped after DUR idle on the GPU while others wait), --server
// (same, 15 min), --on-exit 'shell cmd', --exit-file PATH. Durations: 90, 90s, 15m, 4h.
//
// <cmd> gets B70Q_TICKET=<id> in its environment: the daemon uses it to measure the job and to spot leftovers, and a
// b70-lease run inside it (a script that takes the lease itself) runs at once instead of deadlocking. After `acquire`,
// `export B70Q_TICKET=<token>` (or the same owner name) lets later `b70-lease run` calls use the acquired card.
import { LEGACY_SCRIPT } from "./config";
import { api, ApiError, ts, up } from "./client";
import type { Opts } from "./db";

const POLL = Number(process.env.B70_LEASE_POLL ?? 60) * 1000;
const say = (msg: string) => process.stderr.write(`${ts()} b70-lease: ${msg}\n`);
const SIGNO: Record<string, number> = { SIGHUP: 1, SIGINT: 2, SIGQUIT: 3, SIGKILL: 9, SIGTERM: 15, SIGABRT: 6, SIGSEGV: 11, SIGPIPE: 13 };

export function parseTtl(s: string): number {
  const m = s.match(/^(\d+(?:\.\d+)?)([smhd]?)$/);
  if (!m) throw new Error(`bad duration ${s} (use 90, 90s, 15m, 4h)`);
  return Number(m[1]) * { "": 1, s: 1, m: 60, h: 3600, d: 86400 }[m[2] as "s"];
}

export interface JobFlags {
  minGib?: number; priority?: number; ttl?: number; note?: string; owner?: string; log?: string;
  opts: Opts;
}

// Parses options up to `--` (or, for old-style `run`, the first word that isn't an option). Returns null on error.
export function parseFlags(rest: string[], allow: Set<string>, oldStyle = false): JobFlags | null {
  const f: JobFlags = { opts: {} };
  const val = (a: string) => { const v = rest.shift(); if (v === undefined) throw new Error(`${a} needs a value`); return v; };
  try {
    while (rest.length && rest[0] !== "--") {
      const a = rest[0];
      if (!a.startsWith("--") || !allow.has(a)) {
        if (oldStyle && !a.startsWith("--")) break;
        console.error(`unknown option ${a}`); return null;
      }
      rest.shift();
      switch (a) {
        case "--min-gib": f.minGib = Number(val(a)); break;
        case "--ttl": f.ttl = parseTtl(val(a)); break;
        case "--next": f.priority = 1; break;
        case "--note": f.note = val(a); break;
        case "--owner": f.owner = val(a); break;
        case "--log": f.log = val(a); break;
        case "--creates": f.opts.creates = val(a); break;
        case "--ram-gib": f.opts.ram_gib = Number(val(a)); break;
        case "--vram-gib": f.opts.vram_gib = Number(val(a)); break;
        case "--idle": f.opts.idle_s = parseTtl(val(a)); f.opts.server = true; break;
        case "--server": f.opts.server = true; f.opts.idle_s ??= 15 * 60; break;
        case "--on-exit": f.opts.on_exit = ["bash", "-c", val(a)]; break;
        case "--exit-file": f.opts.exit_file = val(a); break;
        case "--retries": f.opts.retries = Number(val(a)); break;
        case "--chunked": f.opts.more_code = 99; break;
        case "--more-code": f.opts.more_code = Number(val(a)); break;
      }
    }
  } catch (e) { console.error((e as Error).message); return null; }
  if (rest[0] === "--") rest.shift();
  return f;
}

export const RUN_FLAGS = new Set(["--min-gib", "--next", "--note", "--creates", "--ram-gib", "--vram-gib", "--idle", "--server", "--on-exit", "--exit-file"]);
export const ACQUIRE_FLAGS = new Set([...RUN_FLAGS, "--ttl"]);
export const SUBMIT_FLAGS = new Set([...RUN_FLAGS, "--retries", "--chunked", "--more-code", "--log"]);

async function legacy(args: string[], why = "queue daemon not running; using the old lock"): Promise<never> {
  say(why);
  const p = Bun.spawn([LEGACY_SCRIPT, ...args], { stdio: ["inherit", "inherit", "inherit"] });
  process.exit(await p.exited);
}
const legacyRun = (owner: string, argv: string[], f: JobFlags) => ["run", owner, ...(f.minGib ? ["--min-gib", String(f.minGib)] : []), "--", ...argv];
const legacyAcquire = (owner: string, ttl: number, f: JobFlags) => ["acquire", owner, "--ttl", `${ttl}s`, ...(f.minGib ? ["--min-gib", String(f.minGib)] : [])];

// Waits until ticket `id` is granted ("granted"), or skipped because its --creates path exists ("skipped").
// Exits 75 if it was cancelled or killed while waiting; 128+signal if this process was signalled.
async function waitGranted(id: number, owner: string, fallback: string[]): Promise<"granted" | "skipped"> {
  let lastSay = -Infinity, down = false, downSince = 0, timeout = 0, signalled = "";
  const cancel = async (sig: string) => { signalled = sig; try { await api("POST", `/api/tickets/${id}/cancel`); } catch {} process.exit(128 + SIGNO[sig]); };
  const handlers = (["SIGINT", "SIGTERM", "SIGHUP"] as const).map((s) => { const h = () => cancel(s); process.on(s, h); return [s, h] as const; });
  try {
    for (;;) {
      let w;
      try { w = await api("GET", `/api/tickets/${id}/wait?timeout=${timeout}`, undefined, (timeout + 15) * 1000); down = false; }
      catch (e) {
        if (e instanceof ApiError && e.status === 404) { say(`ticket #${id} vanished from the queue`); process.exit(75); }
        if (!down) { say("lost the queue daemon; retrying (the ticket keeps its place)"); downSince = Date.now(); }
        down = true;
        if (Date.now() - downSince > 5 * 60_000) await legacy(fallback, "queue daemon gone for 5 min; falling back to the old lock");
        await Bun.sleep(2000); continue;
      }
      timeout = 8;
      const t = w.ticket;
      if (t.state === "granted") return "granted";
      if (t.state === "done" && String(t.end_reason).startsWith("skipped")) { say(`#${id} ${t.end_reason}`); return "skipped"; }
      if (t.state !== "waiting") { if (signalled) process.exit(128 + SIGNO[signalled]); say(`#${id} ${t.state}: ${t.end_reason}`); process.exit(75); }
      if (Date.now() - lastSay >= POLL) {
        say(`B70 ${w.status.kind === "running" ? `leased by ${w.status.text}` : w.status.text}, waiting (#${id} ${owner}, ${w.position} of ${w.queued} in line)`);
        lastSay = Date.now();
      }
    }
  } finally { for (const [s, h] of handlers) process.off(s, h); }
}

export async function run(owner: string, argv: string[], f: JobFlags = { opts: {} }): Promise<number> {
  const parent = Number(process.env.B70Q_TICKET) || undefined;
  const { id, nested } = await api("POST", "/api/tickets", {
    owner, kind: "run", argv, cwd: process.cwd(), client_pid: process.pid, min_gib: f.minGib, priority: f.priority, parent, opts: f.opts, note: f.note,
  });
  if (nested) say(`B70 already held by #${id}; running inside it`);
  else {
    if ((await waitGranted(id, owner, legacyRun(owner, argv, f))) === "skipped") return 0;
    say(`B70 leased to ${owner} (#${id})`);
  }

  const child = Bun.spawn(argv, { stdio: ["inherit", "inherit", "inherit"], env: { ...process.env, B70Q_TICKET: String(id) } });
  if (!nested) api("POST", `/api/tickets/${id}/started`, { child_pid: child.pid }).catch(() => {});
  // Pass SIGTERM/SIGHUP on so killing this wrapper stops the job too (the old lease let the job run on unleased).
  // SIGINT from a terminal already reaches the child through the process group.
  for (const s of ["SIGTERM", "SIGHUP"] as const) process.on(s, () => child.kill(s));
  process.on("SIGINT", () => {});
  const code = await child.exited;
  const rc = child.signalCode ? 128 + (SIGNO[child.signalCode] ?? 1) : code;
  if (!nested) for (let i = 0; i < 30; i++) {
    try { await api("POST", `/api/tickets/${id}/done`, { exit_code: rc }); break; } catch { await Bun.sleep(1000); }
  }
  return rc;
}

export async function acquire(owner: string, ttlS: number, f: JobFlags = { opts: {} }): Promise<number> {
  const { id } = await api("POST", "/api/tickets", {
    owner, kind: "acquire", client_pid: process.pid, ttl_s: ttlS, min_gib: f.minGib, priority: f.priority, opts: f.opts, note: f.note,
  });
  await waitGranted(id, owner, legacyAcquire(owner, ttlS, f));
  return id;
}

function holderLine(st: any): string {
  const h = st.holder;
  if (h) return `${h.owner} pid=${h.client_pid ?? 0} since=${new Date(h.granted_at).toLocaleString("sv-SE")} mode=${h.kind} ticket=${h.id}`;
  const k = st.status.kind;
  if (k === "idle" && !st.queue.length) return "free";
  if (k === "idle") return `queued: ${st.queue.length} waiting`;
  return `${k}: ${st.status.text}`;
}

async function main(args: string[]): Promise<number> {
  const [cmd, ...rest] = args;
  if (!cmd || !["run", "acquire", "release", "status", "vram", "waiters", "yield"].includes(cmd)) {
    console.error("usage: b70-lease run <owner> [options] -- <cmd...> | acquire <owner> [--ttl 4h] [options] | release <token> | status | vram | waiters | yield [max_s]");
    return 2;
  }

  if (cmd === "run" || cmd === "acquire") {
    const owner = rest.shift();
    if (!owner || owner.startsWith("-")) { console.error(`b70-lease ${cmd} <owner> ...`); return 2; }
    const f = parseFlags(rest, cmd === "run" ? RUN_FLAGS : ACQUIRE_FLAGS, cmd === "run"); // old usage allowed run without --
    if (!f) return 2;
    const ttl = f.ttl ?? 4 * 3600;
    if (cmd === "run" && !rest.length) { console.error("b70-lease run <owner> -- <cmd...>"); return 2; }
    if (!(await up())) await legacy(cmd === "run" ? legacyRun(owner, rest, f) : legacyAcquire(owner, ttl, f));
    if (cmd === "run") return run(owner, rest, f);
    const id = await acquire(owner, ttl, f);
    say(`B70 leased to ${owner} (token ${id}, ttl ${ttl}s); end it with: b70-lease release ${id}  (export B70Q_TICKET=${id} so b70-lease run uses this lease)`);
    console.log(id);
    return 0;
  }
  if (!(await up())) await legacy(args);
  if (cmd === "release") {
    const tok = rest[0];
    if (!tok) { console.error("b70-lease release <token>"); return 2; }
    try { await api("POST", `/api/tickets/${Number(tok)}/release`); say("released"); return 0; }
    catch (e) {
      if (e instanceof ApiError && e.status === 404) await legacy(args, `#${tok} is not a queue ticket; trying the old lease`);
      console.error(`b70-lease: token ${tok}: ${(e as Error).message}`); return 1;
    }
  }
  const st = await api("GET", "/api/state");
  if (cmd === "status") {
    console.log(`lease: ${holderLine(st)}`);
    console.log(`vram_gib: ${st.card ? (st.card.vramUsed / 2 ** 30).toFixed(2) : ""}`);
    console.log(`queue: ${st.queue.length} waiting`);
    return 0;
  }
  if (cmd === "vram") { console.log(st.card ? (st.card.vramUsed / 2 ** 30).toFixed(2) : ""); return 0; }
  if (cmd === "waiters") { console.log(st.queue.length); return 0; }
  if (cmd === "yield") {
    // Between two holds of a chunked job: if anyone is waiting, wait until one of them has the card.
    const max = Number(rest[0] ?? 120) * 1000, t0 = Date.now();
    await Bun.sleep(3000);
    let s = await api("GET", "/api/state");
    while (Date.now() - t0 < max && s.queue.length && !s.holder) { await Bun.sleep(2000); s = await api("GET", "/api/state"); }
    if (Date.now() - t0 > 3500) say(`yielded ${Math.round((Date.now() - t0) / 1000)}s, now ${holderLine(s)}`);
    return 0;
  }
  return 2;
}

if (import.meta.main) process.exit(await main(process.argv.slice(2)));
