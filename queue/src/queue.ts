// The B70 queue: who has the card, who waits, and what the card is doing.
//
// One holder at a time. Waiting tickets are served FIFO with 30-minute turns: an owner whose jobs come in a loop
// (an H3 batch, Drama 3 chunks) keeps the card for its next job while its turn lasts, then the oldest ticket of another
// owner goes next. Tickets with priority (NQ's game master, `b70q now`) go next in line, ahead of turns. Nothing is
// ever preempted. The card is only granted when nothing outside the queue is using it: processes without a ticket
// (Greg's manual runs), leftovers of an ended ticket, or a holder of the old flock-based lease.
import * as fs from "node:fs";
import * as path from "node:path";
import type { Database } from "bun:sqlite";
import * as C from "./config";
import { Card, procStart, type Client, type Proc, type Sample } from "./card";
import { jobtype, type Kind, type Opts, type Ticket } from "./db";
import { flock, LOCK_EX, LOCK_NB, LOCK_UN } from "./libc";

export interface Alert { id: number; key: string; t: number; level: "info" | "warn" | "error"; kind: string; ticket?: number; msg: string }
export interface Usage { vram: number; gtt: number; sys: number; rss: number; busy: number; cpu: number; pids: number[] }
export interface Other { pid: number; comm: string; cmd: string; vram: number; gtt: number; sys: number; busy: number; why: "outside" | "leftover"; ticket?: number; owner?: string }
export interface RingPoint { t: number; vram: number; power?: number; util?: number; temp?: number; vramTemp?: number; memAvail: number; holder?: number; hvram?: number; hgtt?: number; hsys?: number; hrss?: number }
export type CardStatus = { kind: "idle" | "running" | "paused" | "outside" | "leftover" | "legacy" | "fault" | "reserved" | "vram" | "ram"; text: string };

const now = () => Date.now();
const gib = (b: number) => (b / 2 ** 30).toFixed(1);
const alive = (pid: number | null, start: number | null) => !!pid && procStart(pid) !== undefined && (start == null || procStart(pid) === start);

export class Queue {
  card = new Card();
  active = new Map<number, Ticket>(); // waiting + granted
  last?: Sample;
  ring: RingPoint[] = [];
  usage?: Usage; // the holder's live usage
  others: Other[] = [];
  status: CardStatus = { kind: "idle", text: "idle" };
  paused = false; pauseReason = "";
  faultUntil = 0; faults: number[] = [];
  turn?: { owner: string; since: number; lastEnd?: number };
  reserve?: { owner: string; until: number };
  alerts = new Map<string, Alert>();
  private waiters = new Map<number, Set<() => void>>();
  private listeners = new Set<(ev: string, data: unknown) => void>();
  private legacyFd = -1;
  private legacyHeld = false;
  private health = new Map<number, { lastActive: number; cpu: number; cpuRate: number; vram: number; idleSince?: number; gpuIdleSince?: number }>();
  private aggFrom = now();
  private killTimers = new Map<number, Timer>();
  private killed = new Map<number, string>(); // ticket -> why we stopped it (no retry after a kill)
  private stats?: { at: number; byType: Map<string, Stat> };

  constructor(public db: Database) {
    for (const t of db.query("SELECT * FROM tickets WHERE state IN ('waiting','granted') ORDER BY seq").all() as Ticket[]) this.active.set(t.id, t);
    for (const a of db.query("SELECT * FROM alerts WHERE cleared_at IS NULL").all() as (Alert & { key?: string })[])
      this.alerts.set(`db:${a.id}`, { ...a, key: `db:${a.id}`, ticket: a.ticket ?? undefined });
    const meta = (k: string) => (db.query("SELECT value FROM meta WHERE key=?").get(k) as { value: string } | null)?.value;
    this.paused = meta("paused") === "1"; this.pauseReason = meta("pause_reason") ?? "";
    const turn = meta("turn"); if (turn) this.turn = JSON.parse(turn);
    if (C.LEGACY_LOCK !== "off") {
      fs.mkdirSync(path.dirname(C.LEGACY_LOCK), { recursive: true });
      this.legacyFd = fs.openSync(C.LEGACY_LOCK, "a");
    }
    // A daemon restart keeps running jobs: re-take the old lock for the current holder.
    const h = this.holder();
    if (h) { this.takeLegacy(); this.writeHolder(h); }
    for (const p of this.ringFromDb()) this.ring.push(p);
  }

  on(fn: (ev: string, data: unknown) => void) { this.listeners.add(fn); return () => this.listeners.delete(fn); }
  private emit(ev: string, data: unknown) { for (const fn of this.listeners) fn(ev, data); }

  holder(): Ticket | undefined { for (const t of this.active.values()) if (t.state === "granted") return t; }
  waiting(): Ticket[] { return [...this.active.values()].filter((t) => t.state === "waiting").sort((a, b) => b.priority - a.priority || a.seq - b.seq); }
  get(id: number): Ticket | undefined { return this.active.get(id) ?? (this.db.query("SELECT * FROM tickets WHERE id=?").get(id) as Ticket | null) ?? undefined; }

  private static SAVED = ["state", "seq", "priority", "client_pid", "client_start", "child_pid", "granted_at", "ended_at", "exit_code",
    "end_reason", "peak_vram", "peak_gtt", "peak_sys", "peak_rss", "energy_j", "busy_s", "log"] as const;
  private save(t: Ticket) {
    const cols = Queue.SAVED;
    this.db.query(`UPDATE tickets SET ${cols.map((c) => `${c}=?`).join(", ")} WHERE id=?`).run(...cols.map((c) => t[c] ?? null), t.id);
  }
  private setMeta(k: string, v: string) { this.db.query("INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value").run(k, v); }

  // --- API --------------------------------------------------------------------------------------------------------

  enqueue(r: EnqueueRequest): { id: number; nested?: boolean } {
    const owner = String(r.owner ?? "").trim() || "?";
    const h = this.holder();
    // Nested: a command started inside the holder's job (it inherits B70Q_TICKET), or a `run` by the owner of an
    // acquired lease (an agent holding the card across tool calls). Runs at once, under the holder's ticket.
    if (h && r.kind === "run" && (r.parent === h.id || (h.kind === "acquire" && h.owner === owner))) return { id: h.id, nested: true };
    const argv = r.argv ?? [];
    const opts = cleanOpts(r.opts ?? {});
    const maxSeq = (this.db.query("SELECT MAX(seq) AS m FROM tickets").get() as { m: number | null }).m ?? 0;
    const priority = r.priority ?? (C.NEXT_OWNERS.test(owner) ? 1 : 0);
    const res = this.db.query(`INSERT INTO tickets (owner, kind, state, seq, priority, argv, cwd, jobtype, client_pid, client_start, min_gib, ttl_s,
      created_at, opts, note, grp, chunk, attempt) VALUES (?, ?, 'waiting', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`).run(
      owner, r.kind, maxSeq + 1, priority, JSON.stringify(argv), r.cwd ?? "", jobtype(owner, r.kind, argv),
      r.client_pid ?? null, r.client_pid ? procStart(r.client_pid) ?? null : null, r.min_gib ?? C.DEFAULT_MIN_GIB, r.ttl_s ?? null, now(),
      JSON.stringify(opts), r.note ?? null, r.grp ?? null, r.chunk ?? 0, r.attempt ?? 0);
    const id = Number(res.lastInsertRowid);
    if (r.kind === "submit") {
      const log = r.log ? path.resolve(r.cwd ?? "/", r.log) : path.join(C.LOG_DIR, `${id}.log`);
      fs.mkdirSync(C.SPEC_DIR, { recursive: true, mode: 0o700 });
      fs.writeFileSync(path.join(C.SPEC_DIR, `${id}.json`), JSON.stringify({ argv, cwd: r.cwd, env: r.env ?? {}, log }), { mode: 0o600 });
      this.db.query("UPDATE tickets SET log=?, grp=COALESCE(grp, ?) WHERE id=?").run(log, id, id);
    }
    this.active.set(id, this.db.query("SELECT * FROM tickets WHERE id=?").get(id) as Ticket);
    this.tick();
    return { id };
  }

  // The next run of a submitted job: another chunk, or a retry after a failure. Same spec, back of the line.
  private again(t: Ticket, what: "chunk" | "attempt") {
    let spec: { argv: string[]; cwd?: string; env?: Record<string, string>; log?: string };
    try { spec = JSON.parse(fs.readFileSync(path.join(C.SPEC_DIR, `${t.id}.json`), "utf8")); } catch { return; }
    const { id } = this.enqueue({
      owner: t.owner, kind: "submit", argv: spec.argv, cwd: spec.cwd, env: spec.env, log: spec.log, priority: t.priority,
      min_gib: t.min_gib, opts: JSON.parse(t.opts || "{}"), note: t.note ?? undefined, grp: t.grp ?? t.id,
      chunk: t.chunk + (what === "chunk" ? 1 : 0), attempt: what === "attempt" ? t.attempt + 1 : 0,
    });
    return id;
  }

  started(id: number, b: { client_pid?: number; child_pid?: number }) {
    const t = this.active.get(id); if (!t) return;
    if (b.client_pid) { t.client_pid = b.client_pid; t.client_start = procStart(b.client_pid) ?? null; }
    if (b.child_pid) t.child_pid = b.child_pid;
    this.save(t);
  }

  done(id: number, exit_code: number | null, reason = "exited") {
    const t = this.active.get(id); if (!t) return;
    const why = this.killed.get(id);
    if (why) this.end(t, why.startsWith("idle") ? "done" : "failed", why, exit_code);
    else this.end(t, exit_code === 0 || t.kind === "acquire" ? "done" : "failed", reason, exit_code);
  }

  cancel(id: number, reason = "cancelled") {
    const t = this.active.get(id); if (!t) return false;
    if (t.state === "granted") return false;
    this.end(t, "cancelled", reason, null);
    return true;
  }

  bump(id: number) {
    const t = this.active.get(id); if (!t || t.state !== "waiting") return false;
    const min = Math.min(...this.waiting().map((w) => w.seq));
    t.seq = min - 1; this.save(t); this.tick();
    return true;
  }

  // SIGTERM the ticket's processes (the shim forwards it to its command), SIGKILL whatever is left after 15 s.
  kill(id: number, why = "killed") {
    const t = this.active.get(id); if (!t) return false;
    if (t.state === "waiting") return this.cancel(id, "killed while waiting");
    this.killed.set(id, why);
    if (t.kind === "submit") Bun.spawn(["systemctl", "--user", "stop", `b70q-${id}.service`]);
    const pids = this.ticketPids(id);
    if (alive(t.client_pid, t.client_start)) pids.add(t.client_pid!);
    for (const p of pids) try { process.kill(p, "SIGTERM"); } catch {}
    if (t.kind === "acquire") this.end(t, "done", why, null);
    clearTimeout(this.killTimers.get(id));
    this.killTimers.set(id, setTimeout(() => {
      for (const p of this.ticketPids(id)) try { process.kill(p, "SIGKILL"); } catch {}
      const tt = this.active.get(id);
      if (tt && !alive(tt.client_pid, tt.client_start)) this.end(tt, "failed", why, null);
    }, 15_000));
    this.alert(`kill:${id}`, "info", "kill", `${why}: #${id} ${t.owner}`, id);
    return true;
  }

  release(id: number) {
    const t = this.active.get(id); if (!t) return false;
    if (t.state === "waiting") return this.cancel(id, "released while waiting");
    this.end(t, "done", "released", null);
    return true;
  }

  pause(reason = "paused by hand") { this.paused = true; this.pauseReason = reason; this.setMeta("paused", "1"); this.setMeta("pause_reason", reason); this.tick(); }
  resume() { this.paused = false; this.pauseReason = ""; this.faultUntil = 0; this.faults = []; this.setMeta("paused", "0"); this.setMeta("pause_reason", ""); this.tick(); }

  // A submitted job's whole run: its chunks and retries are separate tickets sharing grp (the first ticket's id).
  series(id: number): { grp: number; ids: number[]; latest: Ticket; active: boolean } | undefined {
    const t = this.get(id); if (!t) return;
    const grp = t.grp ?? t.id;
    const rows = this.db.query("SELECT id FROM tickets WHERE grp=? OR id=? ORDER BY id").all(grp, grp) as { id: number }[];
    const ids = rows.map((r) => r.id);
    const latest = this.get(ids[ids.length - 1] ?? t.id)!;
    return { grp, ids, latest, active: ids.some((i) => this.active.has(i)) };
  }

  // Resolves when the ticket leaves the waiting state (until=granted) or ends (until=ended), or after timeoutMs.
  wait(id: number, timeoutMs: number, until: "granted" | "ended" = "granted"): Promise<void> {
    const t = this.active.get(id);
    if (!t || (until === "granted" && t.state !== "waiting")) return Promise.resolve();
    return new Promise((resolve) => {
      const set = this.waiters.get(id) ?? new Set();
      const done = () => { clearTimeout(timer); set.delete(done); resolve(); };
      const timer = setTimeout(done, timeoutMs);
      set.add(done); this.waiters.set(id, set);
    });
  }
  private notify(id: number) { for (const fn of this.waiters.get(id) ?? []) fn(); this.waiters.delete(id); }

  // --- scheduling ---------------------------------------------------------------------------------------------------

  tick() {
    const t = now();
    // Clients that went away: a waiting shim that was killed, a job whose shim/runner exited without reporting.
    for (const k of [...this.active.values()]) {
      if (k.state === "waiting" && k.client_pid && !alive(k.client_pid, k.client_start)) this.end(k, "cancelled", "client gone", null);
      else if (k.state === "granted") {
        if (k.kind === "acquire" && k.ttl_s && t - k.granted_at! > k.ttl_s * 1000) this.end(k, "done", "ttl expired", null);
        else if (k.kind === "run" && !alive(k.client_pid, k.client_start)) this.end(k, "failed", "b70-lease process died", null);
        else if (k.kind === "submit" && k.client_pid && !alive(k.client_pid, k.client_start)) this.end(k, "failed", "runner died", null);
        else if (k.kind === "submit" && !k.client_pid && t - k.granted_at! > 120_000) this.end(k, "failed", "runner never started", null);
      }
    }
    if (this.reserve && t >= this.reserve.until) this.reserve = undefined;
    this.status = this.computeStatus();
    if (this.status.kind === "idle" || this.status.kind === "reserved") {
      const pick = this.pick();
      if (pick && this.takeLegacy()) { this.grant(pick); if (this.holder()) this.status = this.computeStatus(); }
      else if (pick) this.status = { kind: "legacy", text: `old b70-lease holder: ${this.legacyHolderText()}` };
    }
    this.emit("state", null);
  }

  private pick(): Ticket | undefined {
    const waiting = this.waiting();
    if (!waiting.length) return;
    if (waiting[0].priority > 0) return waiting[0];
    const t = now();
    if (this.reserve && t < this.reserve.until) return waiting.find((w) => w.owner === this.reserve!.owner);
    const live = this.turn && t - this.turn.since < C.TURN_MS;
    if (live) return waiting.find((w) => w.owner === this.turn!.owner) ?? waiting[0];
    return waiting.find((w) => w.owner !== this.turn?.owner) ?? waiting[0];
  }

  private grant(k: Ticket) {
    const o = opt(k);
    if (o.creates && fs.existsSync(path.resolve(k.cwd || "/", o.creates))) {
      this.releaseLegacy();
      this.end(k, "done", `skipped: ${o.creates} exists`, 0);
      return;
    }
    if (o.ram_gib && this.last && this.last.memAvail < o.ram_gib * 2 ** 30) {
      this.releaseLegacy();
      this.status = { kind: "ram", text: `#${k.id} ${k.owner} needs ${o.ram_gib} GiB host RAM free; ${gib(this.last.memAvail)} GiB is` };
      return;
    }
    // The min-VRAM check of the old lease: device VRAM in use must be under the ticket's min_gib.
    if (this.last && this.last.vramUsed >= k.min_gib * 2 ** 30) {
      this.releaseLegacy();
      this.status = { kind: "vram", text: `${gib(this.last.vramUsed)} GiB VRAM in use by nothing we can see; waiting for it to drop under ${k.min_gib} GiB` };
      return;
    }
    k.state = "granted"; k.granted_at = now();
    this.save(k);
    // A turn starts when another owner gets the card, or when the same owner comes back after a break (not a loop).
    // Next-in-line jobs slip in between turns without starting one of their own.
    const t = this.turn;
    if (!k.priority && (!t || t.owner !== k.owner || (!t.lastEnd || k.granted_at - t.lastEnd > C.GRACE_MS + 5000))) {
      this.turn = { owner: k.owner, since: k.granted_at };
      this.setMeta("turn", JSON.stringify(this.turn));
    }
    this.reserve = undefined;
    this.health.set(k.id, { lastActive: now(), cpu: -1, cpuRate: 0, vram: 0 });
    this.writeHolder(k);
    if (k.kind === "submit") this.launch(k);
    this.notify(k.id);
    this.emit("granted", k);
  }

  private end(k: Ticket, state: Ticket["state"], reason: string, rc: number | null) {
    const wasGranted = k.state === "granted";
    const o = opt(k);
    const killedWhy = this.killed.get(k.id);
    this.killed.delete(k.id);
    // A chunked job that says "more to do" is a finished chunk; a failed job with retries left runs again.
    let follow: "chunk" | "attempt" | undefined;
    if (k.kind === "submit" && wasGranted && !killedWhy) {
      if (o.more_code && rc === o.more_code) { state = "done"; reason = `chunk ${k.chunk + 1} done, more to do`; follow = "chunk"; }
      else if (state === "failed" && k.attempt < (o.retries ?? 0)) { reason = `${reason}; retrying (${k.attempt + 1}/${o.retries})`; follow = "attempt"; }
    }
    k.state = state; k.ended_at = now(); k.end_reason = reason; k.exit_code = rc;
    this.save(k);
    this.active.delete(k.id);
    this.health.delete(k.id);
    for (const key of [`stall:${k.id}`, `cpu:${k.id}`, `spill:${k.id}`, `idle:${k.id}`]) this.clear(key);
    if (wasGranted) {
      this.releaseLegacy();
      this.clearHolder();
      this.loopers.delete(k.owner);
      if (this.turn?.owner === k.owner) { this.turn.lastEnd = now(); this.setMeta("turn", JSON.stringify(this.turn)); }
      if (!k.priority && this.turn?.owner === k.owner && now() - this.turn.since < C.TURN_MS && (follow === "chunk" || this.loops(k.owner)))
        this.reserve = { owner: k.owner, until: now() + C.GRACE_MS };
      this.stats = undefined;
    }
    // Queued only now, after the card is released, so the follow-up can be granted cleanly.
    const next = follow ? this.again(k, follow) : undefined;
    if (k.kind === "submit") fs.rmSync(path.join(C.SPEC_DIR, `${k.id}.json`), { force: true });
    if (!next) this.finished(k, o);
    this.notify(k.id);
    this.emit("ended", k);
    queueMicrotask(() => this.tick());
  }

  // Does this owner run its jobs in a loop (usually asks again within the grace time after a job ends)? Only then is
  // the card kept for its next job; a one-off job's owner isn't coming back, so the next owner starts at once.
  private loopers = new Map<string, boolean>();
  private loops(owner: string): boolean {
    let v = this.loopers.get(owner);
    if (v === undefined) {
      const rows = (this.db.query("SELECT created_at, ended_at FROM tickets WHERE owner=? AND kind!='acquire' ORDER BY id DESC LIMIT 12").all(owner) as
        { created_at: number; ended_at: number | null }[]).reverse();
      let pairs = 0, quick = 0;
      for (let i = 1; i < rows.length; i++) {
        if (rows[i - 1].ended_at == null) continue;
        pairs++;
        const gap = rows[i].created_at - rows[i - 1].ended_at!;
        if (gap >= -1000 && gap <= C.GRACE_MS) quick++;
      }
      v = pairs > 0 && quick / pairs >= 0.5;
      this.loopers.set(owner, v);
    }
    return v;
  }

  // The whole job (all chunks/retries) is over: write the exit file and run the on-exit hook.
  private finished(k: Ticket, o: Opts) {
    const code = k.exit_code != null ? String(k.exit_code) : k.end_reason ?? k.state;
    if (o.exit_file) try { fs.writeFileSync(path.resolve(k.cwd || "/", o.exit_file), code + "\n"); } catch (e) { console.error(`exit file for #${k.id}:`, e); }
    if (o.on_exit) try {
      Bun.spawn(o.on_exit, {
        cwd: k.cwd && fs.existsSync(k.cwd) ? k.cwd : process.env.HOME, stdio: ["ignore", "ignore", "ignore"],
        env: { ...process.env, B70Q_TICKET: String(k.id), B70Q_EXIT: code, B70Q_STATE: k.state, B70Q_REASON: k.end_reason ?? "", B70Q_OWNER: k.owner, B70Q_LOG: k.log ?? "" },
      });
    } catch (e) { this.alert(`hook:${k.id}`, "warn", "hook", `#${k.id} ${k.owner}: on-exit hook failed to start: ${(e as Error).message}`, k.id); }
  }

  private computeStatus(): CardStatus {
    const h = this.holder();
    if (h) return { kind: "running", text: `#${h.id} ${h.owner}` };
    if (this.paused) return { kind: "paused", text: this.pauseReason || "paused" };
    if (now() < this.faultUntil) return { kind: "fault", text: `xe fault; holding until ${new Date(this.faultUntil).toLocaleTimeString()}` };
    const outside = this.others.filter((o) => o.why === "outside");
    if (outside.length) return { kind: "outside", text: `outside the queue: ${outside.map((o) => `${o.comm} (pid ${o.pid}, ${gib(o.vram)} GiB)`).join(", ")}` };
    const left = this.others.filter((o) => o.why === "leftover");
    if (left.length) return { kind: "leftover", text: `leftover of #${left[0].ticket} ${left[0].owner ?? ""}: ${left[0].comm} (pid ${left[0].pid}) still on the card` };
    if (!this.legacyFree()) return { kind: "legacy", text: `old b70-lease holder: ${this.legacyHolderText()}` };
    if (this.reserve && this.waiting().some((w) => w.owner !== this.reserve!.owner))
      return { kind: "reserved", text: `kept ${Math.ceil((this.reserve.until - now()) / 1000)} s for ${this.reserve.owner}'s next job (turn)` };
    return { kind: "idle", text: "idle" };
  }

  // --- the old flock lease -------------------------------------------------------------------------------------------

  private takeLegacy(): boolean {
    if (this.legacyFd < 0 || this.legacyHeld) return true;
    this.legacyHeld = flock(this.legacyFd, LOCK_EX | LOCK_NB);
    return this.legacyHeld;
  }
  private releaseLegacy() { if (this.legacyHeld) { flock(this.legacyFd, LOCK_UN); this.legacyHeld = false; } }
  private legacyFree(): boolean {
    if (this.legacyFd < 0 || this.legacyHeld) return true;
    if (!flock(this.legacyFd, LOCK_EX | LOCK_NB)) return false;
    flock(this.legacyFd, LOCK_UN);
    return true;
  }
  private legacyHolderText() { try { return fs.readFileSync(C.LEGACY_HOLDER, "utf8").trim() || "unknown"; } catch { return "unknown"; } }
  private writeHolder(k: Ticket) {
    if (this.legacyFd < 0) return;
    const since = new Date(k.granted_at!).toLocaleString("sv-SE").replace("T", " ");
    fs.writeFileSync(C.LEGACY_HOLDER, `${k.owner} pid=${k.client_pid ?? 0} since=${since} mode=${k.kind} ticket=${k.id}\n`);
  }
  private clearHolder() { if (this.legacyFd >= 0) fs.writeFileSync(C.LEGACY_HOLDER, ""); }

  // --- submitted jobs: the daemon starts them as systemd user units -----------------------------------------------

  private launch(k: Ticket) {
    fs.mkdirSync(C.LOG_DIR, { recursive: true });
    const runner = path.join(import.meta.dir, "runner.ts");
    const p = Bun.spawn(["systemd-run", "--user", "--quiet", "--collect", `--unit=b70q-${k.id}`, `--description=b70q #${k.id} ${k.owner}`,
      "-p", "KillMode=mixed", `--setenv=B70Q_SOCKET=${C.SOCKET}`, `--setenv=B70Q_STATE=${C.STATE}`,
      process.execPath, runner, String(k.id)], { stdout: "ignore", stderr: "pipe" });
    p.exited.then(async (rc) => {
      if (rc !== 0) this.done(k.id, null, `systemd-run failed: ${(await new Response(p.stderr).text()).trim()}`);
    });
    // A runner that dies before reporting in (bad spec, missing program) ends the ticket instead of holding the card.
    setTimeout(async () => {
      const t = this.active.get(k.id);
      if (!t || t.client_pid) return;
      const st = Bun.spawnSync(["systemctl", "--user", "show", "-p", "ActiveState", "--value", `b70q-${k.id}.service`]).stdout.toString().trim();
      if (st !== "active" && st !== "activating") this.done(k.id, null, `runner failed (journalctl --user -u b70q-${k.id})`);
    }, 10_000);
  }

  // --- sampling and health --------------------------------------------------------------------------------------------

  // The ticket a process belongs to: its own B70Q_TICKET, or an ancestor's (bwrap and friends may clear the env),
  // or the shim/runner that started it.
  private ticketOf(pid: number, procs: Map<number, Proc>, clients: Map<number, number>): number | undefined {
    for (let p = procs.get(pid), hops = 0; p && hops < 64; p = procs.get(p.ppid), hops++) {
      if (p.ticket) return p.ticket;
      const c = clients.get(p.pid); if (c) return c;
    }
  }

  private ticketPids(id: number): Set<number> {
    const procs = this.last?.procs ?? new Map();
    const clients = this.clientPids();
    const out = new Set<number>();
    for (const p of procs.values()) if (p.pid !== process.pid && this.ticketOf(p.pid, procs, clients) === id) out.add(p.pid);
    return out;
  }

  private clientPids(): Map<number, number> {
    const m = new Map<number, number>();
    for (const t of this.active.values()) if (t.state === "granted" && t.client_pid) m.set(t.client_pid, t.id);
    return m;
  }

  sample() {
    const s = this.card.sample();
    const prevT = this.last?.t ?? s.t;
    const dt = (s.t - prevT) / 1000;
    this.last = s;
    const h = this.holder();
    const clients = this.clientPids();
    const others: Other[] = [];
    const u: Usage = { vram: 0, gtt: 0, sys: 0, rss: 0, busy: 0, cpu: 0, pids: [] };
    for (const c of s.clients) {
      const tid = this.ticketOf(c.pid, s.procs, clients);
      const mine = h && (tid === h.id || (tid === undefined && h.kind === "acquire"));
      if (mine) { u.vram += c.vram; u.gtt += c.gtt; u.sys += c.sys; u.busy = Math.max(u.busy, c.busy); continue; }
      const significant = c.vram + c.gtt + c.sys >= C.FOREIGN_MIN_BYTES || c.busy > 0.05;
      if (!significant) continue;
      if (tid !== undefined) {
        const t = this.get(tid);
        others.push({ ...c, why: "leftover", ticket: tid, owner: t?.owner });
      } else others.push({ ...c, why: "outside" });
    }
    if (h) for (const p of s.procs.values()) if (this.ticketOf(p.pid, s.procs, clients) === h.id) { u.rss += p.rss; u.cpu += p.cpu; u.pids.push(p.pid); }
    this.others = others;
    this.usage = h ? u : undefined;

    if (h) {
      h.peak_vram = Math.max(h.peak_vram, u.vram); h.peak_gtt = Math.max(h.peak_gtt, u.gtt);
      h.peak_sys = Math.max(h.peak_sys, u.sys); h.peak_rss = Math.max(h.peak_rss, u.rss);
      if (s.power !== undefined) h.energy_j += s.power * dt;
      h.busy_s += u.busy * dt;
      this.save(h);
      this.checkHolder(h, u, s, dt);
    }
    this.checkCard(s, h);

    const point: RingPoint = { t: s.t, vram: s.vramUsed, power: s.power, util: s.util, temp: s.temp, vramTemp: s.vramTemp, memAvail: s.memAvail,
      holder: h?.id, hvram: h ? u.vram : undefined, hgtt: h ? u.gtt : undefined, hsys: h ? u.sys : undefined, hrss: h ? u.rss : undefined };
    this.ring.push(point);
    if (this.ring.length > C.RING_SAMPLES) this.ring.shift();
    if (s.t - this.aggFrom >= C.AGG_MS) this.aggregate(s.t);
    this.emit("sample", point);
    this.tick();
  }

  private checkHolder(h: Ticket, u: Usage, s: Sample, dt: number) {
    const hs = this.health.get(h.id) ?? { lastActive: now(), cpu: -1, cpuRate: 0, vram: 0 };
    const cpuRate = hs.cpu >= 0 && dt > 0 ? Math.max(0, (u.cpu - hs.cpu) / dt) : 0;
    hs.cpuRate = hs.cpuRate * 0.9 + cpuRate * 0.1; // ~20 s average
    const active = u.busy > 0.02 || cpuRate > 0.2 || Math.abs(u.vram - hs.vram) > 64 * 2 ** 20;
    if (active) hs.lastActive = now();
    hs.cpu = u.cpu; hs.vram = u.vram;
    this.health.set(h.id, hs);
    const held = now() - h.granted_at!;
    const others = this.waiting().length;

    // Servers (idle_s): stop one that has had nothing to do on the GPU for idle_s while others wait.
    const o = opt(h);
    if (o.idle_s) {
      hs.gpuIdleSince = u.busy < 0.02 ? hs.gpuIdleSince ?? now() : undefined;
      if (hs.gpuIdleSince && now() - hs.gpuIdleSince > o.idle_s * 1000 && others) {
        this.kill(h.id, `idle server stopped after ${Math.round(o.idle_s / 60)} min idle (${others} waiting)`);
        hs.gpuIdleSince = undefined;
      }
      return;
    }
    if (h.kind === "acquire") {
      // A lease held across tool calls is often idle on purpose; say so only when it keeps others waiting.
      const idle = u.vram < C.FOREIGN_MIN_BYTES && u.busy < 0.02;
      hs.idleSince = idle ? hs.idleSince ?? now() : undefined;
      if (idle && others && now() - hs.idleSince! > 20 * 60_000)
        this.alert(`idle:${h.id}`, "warn", "idle-lease", `#${h.id} ${h.owner} holds an acquired lease with nothing on the card for ${Math.round((now() - hs.idleSince!) / 60_000)} min while ${others} wait`, h.id);
      else this.clear(`idle:${h.id}`);
      return;
    }
    if (now() - hs.lastActive > C.STALL_MS)
      this.alert(`stall:${h.id}`, "warn", "stall", `#${h.id} ${h.owner}: no GPU work, no CPU time and no VRAM change for ${Math.round((now() - hs.lastActive) / 60_000)} min`, h.id);
    else this.clear(`stall:${h.id}`);

    const cpuBound = held > C.CPU_FALLBACK_MS && u.busy < 0.02 && (s.power ?? 999) < 70 && hs.cpuRate > 4;
    if (cpuBound) this.alert(`cpu:${h.id}`, "warn", "cpu-fallback", `#${h.id} ${h.owner} is using ${hs.cpuRate.toFixed(1)} CPU cores while the B70 is idle (${(s.power ?? 0).toFixed(0)} W): running on the CPU?`, h.id);
    else if (u.busy > 0.1) this.clear(`cpu:${h.id}`);

    const spill = s.vramUsed > 0.97 * s.vramTotal && u.gtt + u.sys > 2 ** 30;
    if (spill) this.alert(`spill:${h.id}`, "warn", "spill", `#${h.id} ${h.owner}: VRAM full (${gib(s.vramUsed)}/${gib(s.vramTotal)} GiB) and ${gib(u.gtt + u.sys)} GiB in system memory; it may be spilling`, h.id);
    else if (s.vramUsed < 0.9 * s.vramTotal) this.clear(`spill:${h.id}`);
  }

  private checkCard(s: Sample, h?: Ticket) {
    const outside = this.others.filter((o) => o.why === "outside");
    const ids = new Set(outside.map((o) => o.pid));
    for (const key of [...this.alerts.keys()]) if (key.startsWith("outside:") && !ids.has(Number(key.slice(8)))) this.clear(key);
    for (const o of outside)
      this.alert(`outside:${o.pid}`, h ? "warn" : "info", "outside", `${h ? `running next to #${h.id} ${h.owner}: ` : "queue waits for "}${o.comm} (pid ${o.pid}, ${gib(o.vram)} GiB VRAM) outside the queue: ${o.cmd.slice(0, 160)}`);
    const left = this.others.filter((o) => o.why === "leftover");
    const lids = new Set(left.map((o) => o.pid));
    for (const key of [...this.alerts.keys()]) if (key.startsWith("leftover:") && !lids.has(Number(key.slice(9)))) this.clear(key);
    for (const o of left)
      this.alert(`leftover:${o.pid}`, "warn", "leftover", `#${o.ticket} ${o.owner ?? ""} ended but ${o.comm} (pid ${o.pid}) still holds ${gib(o.vram)} GiB on the card; the queue waits for it (b70q kill ${o.ticket})`, o.ticket);
    if (s.memAvail < C.LOW_RAM_BYTES) this.alert("lowram", "warn", "low-ram", `host RAM low: ${gib(s.memAvail)} GiB available`);
    else if (s.memAvail > C.LOW_RAM_BYTES * 1.5) this.clear("lowram");
  }

  // xe fault from the kernel log: hold the queue briefly; repeated faults hold it until resumed by hand.
  fault(line: string) {
    const t = now();
    const h = this.holder();
    this.faults = [...this.faults.filter((f) => t - f < C.FAULT_WINDOW_MS), t];
    this.alert(`fault:${t}`, "error", "xe-fault", `xe fault${h ? ` during #${h.id} ${h.owner}` : ""}: ${line}`, h?.id);
    if (this.faults.length >= C.FAULT_HOLD_COUNT && !this.paused) this.pause(`${this.faults.length} xe faults within an hour; check the card, then resume`);
    else this.faultUntil = t + C.FAULT_PAUSE_MS;
    this.tick();
  }

  alert(key: string, level: Alert["level"], kind: string, msg: string, ticket?: number) {
    const a = this.alerts.get(key);
    if (a) { if (a.msg !== msg) a.msg = msg; return; }
    const res = this.db.query("INSERT INTO alerts (t, level, kind, ticket, msg) VALUES (?,?,?,?,?)").run(now(), level, kind, ticket ?? null, msg);
    const al: Alert = { id: Number(res.lastInsertRowid), key, t: now(), level, kind, ticket, msg };
    this.alerts.set(key, al);
    this.emit("alert", al);
  }
  clear(key: string) {
    const a = this.alerts.get(key); if (!a) return;
    this.db.query("UPDATE alerts SET cleared_at=? WHERE id=?").run(now(), a.id);
    this.alerts.delete(key);
  }
  ack(id: number) { for (const [k, a] of this.alerts) if (a.id === id) this.clear(k); }

  private aggregate(t: number) {
    const pts = this.ring.filter((p) => p.t >= this.aggFrom && p.t < t);
    this.aggFrom = t;
    if (!pts.length) return;
    const avg = (f: (p: RingPoint) => number | undefined) => { const v = pts.map(f).filter((x): x is number => x !== undefined); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null; };
    this.db.query("INSERT OR REPLACE INTO samples (t, vram, power, util, temp, mem_avail, holder) VALUES (?,?,?,?,?,?,?)").run(
      pts[0].t, Math.max(...pts.map((p) => p.vram)), avg((p) => p.power), avg((p) => p.util), avg((p) => p.temp),
      Math.min(...pts.map((p) => p.memAvail)), pts[pts.length - 1].holder ?? null);
  }
  // After a restart the live charts start from the 30 s aggregates of the last hour.
  private ringFromDb(): RingPoint[] {
    return (this.db.query("SELECT * FROM samples WHERE t > ? ORDER BY t").all(now() - 3600_000) as
      { t: number; vram: number; power: number | null; util: number | null; temp: number | null; mem_avail: number; holder: number | null }[])
      .map((r) => ({ t: r.t, vram: r.vram, power: r.power ?? undefined, util: r.util ?? undefined, temp: r.temp ?? undefined, memAvail: r.mem_avail, holder: r.holder ?? undefined }));
  }

  // --- views --------------------------------------------------------------------------------------------------------

  statsByType(): Map<string, Stat> {
    if (this.stats && now() - this.stats.at < 60_000) return this.stats.byType;
    const rows = this.db.query(`SELECT jobtype, owner, granted_at, ended_at, peak_vram, peak_rss, state FROM tickets
      WHERE granted_at IS NOT NULL AND ended_at IS NOT NULL AND kind != 'acquire' ORDER BY ended_at DESC LIMIT 5000`).all() as
      { jobtype: string; owner: string; granted_at: number; ended_at: number; peak_vram: number; peak_rss: number; state: string }[];
    const by = new Map<string, typeof rows>();
    for (const r of rows) (by.get(r.jobtype) ?? by.set(r.jobtype, []).get(r.jobtype)!).push(r);
    const out = new Map<string, Stat>();
    for (const [k, rs] of by) {
      const dur = rs.map((r) => r.ended_at - r.granted_at).sort((a, b) => a - b);
      const vram = rs.map((r) => r.peak_vram).sort((a, b) => a - b);
      out.set(k, {
        jobtype: k, owner: rs[0].owner, n: rs.length, failed: rs.filter((r) => r.state === "failed").length,
        median_ms: dur[Math.floor(dur.length / 2)], p90_ms: dur[Math.floor(dur.length * 0.9)],
        p90_vram: vram[Math.floor(vram.length * 0.9)], max_vram: vram[vram.length - 1],
        max_rss: Math.max(...rs.map((r) => r.peak_rss)), last: rs[0].ended_at,
      });
    }
    this.stats = { at: now(), byType: out };
    return out;
  }

  // Expected run time of a ticket: its job type's median, else its owner's, else 10 min.
  estimate(k: Ticket): number {
    const by = this.statsByType();
    const s = by.get(k.jobtype);
    if (s && s.n >= 1) return s.median_ms;
    const own = [...by.values()].filter((x) => x.owner === k.owner);
    if (own.length) return own.sort((a, b) => b.n - a.n)[0].median_ms;
    return k.kind === "acquire" ? (k.ttl_s ?? 3600) * 1000 / 2 : 10 * 60_000;
  }

  view() {
    const s = this.last;
    const h = this.holder();
    const t = now();
    let at = t;
    if (h) at += Math.max(60_000, this.estimate(h) - (t - h.granted_at!));
    const queue = this.waiting().map((k, i) => {
      const est = this.estimate(k);
      const v = { ...this.ticketView(k), position: i + 1, eta_start: at, est_ms: est };
      at += est;
      return v;
    });
    return {
      t,
      card: s && { vramUsed: s.vramUsed, vramTotal: s.vramTotal, power: s.power, temp: s.temp, vramTemp: s.vramTemp, util: s.util, freq: s.freq, memAvail: s.memAvail, memTotal: s.memTotal },
      status: this.status,
      holder: h ? { ...this.ticketView(h), usage: this.usage, est_ms: this.estimate(h) } : null,
      turn: this.turn && { ...this.turn, left_ms: Math.max(0, C.TURN_MS - (t - this.turn.since)) },
      reserve: this.reserve,
      queue,
      others: this.others,
      alerts: [...this.alerts.values()].sort((a, b) => b.t - a.t),
      paused: this.paused, pauseReason: this.pauseReason, faultUntil: this.faultUntil,
      config: { turn_ms: C.TURN_MS, grace_ms: C.GRACE_MS },
    };
  }

  ticketView(k: Ticket) {
    const { argv, opts, ...rest } = k;
    return { ...rest, argv: JSON.parse(argv) as string[], opts: JSON.parse(opts || "{}") as Opts };
  }
}

export interface Stat { jobtype: string; owner: string; n: number; failed: number; median_ms: number; p90_ms: number; p90_vram: number; max_vram: number; max_rss: number; last: number }

export interface EnqueueRequest {
  owner: string; kind: Kind; argv?: string[]; cwd?: string; client_pid?: number; min_gib?: number; ttl_s?: number;
  priority?: number; parent?: number; env?: Record<string, string>;
  opts?: Opts; note?: string; log?: string; grp?: number; chunk?: number; attempt?: number;
}

const opt = (t: Ticket): Opts => { try { return JSON.parse(t.opts || "{}"); } catch { return {}; } };

function cleanOpts(o: Opts): Opts {
  const out: Opts = {};
  const pos = (v: unknown) => (typeof v === "number" && isFinite(v) && v > 0 ? v : undefined);
  if (typeof o.creates === "string" && o.creates) out.creates = o.creates;
  if (Array.isArray(o.on_exit) && o.on_exit.length) out.on_exit = o.on_exit.map(String);
  if (typeof o.exit_file === "string" && o.exit_file) out.exit_file = o.exit_file;
  if (pos(o.retries)) out.retries = Math.floor(o.retries!);
  if (typeof o.more_code === "number" && o.more_code > 0 && o.more_code < 256) out.more_code = o.more_code;
  if (pos(o.idle_s)) out.idle_s = o.idle_s;
  if (pos(o.ram_gib)) out.ram_gib = o.ram_gib;
  if (pos(o.vram_gib)) out.vram_gib = o.vram_gib;
  if (o.server) out.server = true;
  return out;
}
