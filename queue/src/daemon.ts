// b70q daemon: the queue for the Arc Pro B70, its HTTP API (unix socket for tools, localhost TCP for the dashboard)
// and the live dashboard over WebSocket.
import * as fs from "node:fs";
import * as path from "node:path";
import type { Server } from "bun";
import dashboard from "../web/index.html";
import * as C from "./config";
import { open } from "./db";
import { watchKernel } from "./kernel";
import { Queue } from "./queue";
import { flock, LOCK_EX, LOCK_NB } from "./libc";

// One daemon per state dir: a second copy would schedule the same tickets twice.
fs.mkdirSync(C.STATE, { recursive: true });
const lockFd = fs.openSync(path.join(C.STATE, "daemon.lock"), "a");
if (!flock(lockFd, LOCK_EX | LOCK_NB)) { console.error(`b70q: another daemon is already running for ${C.STATE}`); process.exit(1); }

const db = open();
const q = new Queue(db);
const startedAt = Date.now();

const json = (v: unknown, status = 200) => Response.json(v, { status });
const bad = (msg: string, status = 400) => new Response(msg, { status });

async function handle(req: Request, tcp: boolean): Promise<Response> {
  const url = new URL(req.url);
  const p = url.pathname;
  // The dashboard port is reachable by any web page in the browser; state changes need a header a cross-site page
  // can't send without a CORS preflight, which we never answer.
  if (tcp && req.method !== "GET" && req.headers.get("x-b70q") !== "1") return bad("missing x-b70q header", 403);
  const body = req.method === "POST" ? await req.json().catch(() => ({})) : {};

  if (p === "/api/ping") return json({ ok: true, pid: process.pid, since: startedAt });
  if (p === "/api/state") return json(q.view());
  if (p === "/api/ring") return json(q.ring);
  if (p === "/api/stats") return json([...q.statsByType().values()].sort((a, b) => b.last - a.last));
  if (p === "/api/samples") {
    const hours = Number(url.searchParams.get("hours") ?? 24);
    return json(db.query("SELECT * FROM samples WHERE t > ? ORDER BY t").all(Date.now() - hours * 3600_000));
  }
  if (p === "/api/history") {
    const hours = Number(url.searchParams.get("hours") ?? 24);
    const limit = Number(url.searchParams.get("limit") ?? 500);
    const rows = db.query(`SELECT * FROM tickets WHERE created_at > ? OR ended_at > ? OR state IN ('waiting','granted')
      ORDER BY id DESC LIMIT ?`).all(Date.now() - hours * 3600_000, Date.now() - hours * 3600_000, limit) as any[];
    return json(rows.map((t) => q.ticketView(t)));
  }
  if (p === "/api/pause" && req.method === "POST") { q.pause(body.reason || "paused by hand"); return json(q.view()); }
  if (p === "/api/resume" && req.method === "POST") { q.resume(); return json(q.view()); }
  let m = p.match(/^\/api\/alerts\/(\d+)\/ack$/);
  if (m && req.method === "POST") { q.ack(Number(m[1])); return json({ ok: true }); }

  if (p === "/api/tickets" && req.method === "POST") {
    if (!body.owner || !["run", "acquire", "submit"].includes(body.kind)) return bad("owner and kind (run|acquire|submit) required");
    if (body.kind !== "acquire" && !(Array.isArray(body.argv) && body.argv.length)) return bad("argv required");
    if (body.kind === "submit" && tcp) return bad("submit only over the unix socket", 403);
    return json(q.enqueue(body));
  }
  m = p.match(/^\/api\/tickets\/(\d+)(?:\/(\w+))?$/);
  if (m) {
    const id = Number(m[1]), action = m[2];
    const t = q.get(id);
    if (!t) return bad(`no ticket ${id}`, 404);
    const position = () => q.waiting().findIndex((w) => w.id === id) + 1;
    const seriesView = () => {
      const sr = q.series(id)!;
      return { grp: sr.grp, ids: sr.ids, active: sr.active, latest: q.ticketView(sr.latest) };
    };
    if (!action) return json({ ticket: q.ticketView(t), position: position(), series: seriesView() });
    if (action === "wait") {
      const ms = Math.min(8, Number(url.searchParams.get("timeout") ?? 8)) * 1000;
      if (url.searchParams.get("until") === "ended") {
        // The whole job: follow its chunks and retries to the last ticket. `ticket` is that last ticket.
        const deadline = Date.now() + ms;
        for (let sr = q.series(id)!; sr.active && Date.now() < deadline; sr = q.series(id)!)
          await q.wait(sr.latest.id, deadline - Date.now(), "ended");
        const sr = q.series(id)!;
        const pos = q.waiting().findIndex((w) => w.id === sr.latest.id) + 1;
        return json({ ticket: q.ticketView(sr.latest), position: pos, queued: q.waiting().length, status: q.status, series: seriesView() });
      }
      await q.wait(id, ms, "granted");
      return json({ ticket: q.ticketView(q.get(id)!), position: position(), queued: q.waiting().length, status: q.status });
    }
    if (action === "spec") {
      if (tcp) return bad("spec only over the unix socket", 403);
      try { return new Response(fs.readFileSync(path.join(C.SPEC_DIR, `${id}.json`))); } catch { return bad("no spec", 404); }
    }
    if (action === "log") {
      if (!t.log) return bad("no log for this ticket (only submitted jobs have one)", 404);
      const off = Number(url.searchParams.get("offset") ?? 0);
      try {
        const fd = fs.openSync(t.log, "r");
        const size = fs.fstatSync(fd).size;
        const start = off < 0 ? Math.max(0, size + off) : Math.min(off, size);
        const buf = Buffer.alloc(Math.min(size - start, 4 * 2 ** 20));
        fs.readSync(fd, buf, 0, buf.length, start); fs.closeSync(fd);
        return json({ offset: start + buf.length, size, text: buf.toString("utf8") });
      } catch { return json({ offset: 0, size: 0, text: "" }); }
    }
    if (req.method !== "POST") return bad("POST required", 405);
    if (action === "started") { q.started(id, body); return json({ ok: true }); }
    if (action === "done") { q.done(id, body.exit_code ?? null, body.reason); return json({ ok: true }); }
    const ok = action === "release" ? q.release(id) : action === "cancel" ? q.cancel(id) : action === "kill" ? q.kill(id) : action === "bump" ? q.bump(id) : null;
    if (ok === null) return bad(`unknown action ${action}`, 404);
    return ok ? json({ ok: true }) : bad(`#${id} is ${t.state}; can't ${action} it`, 409);
  }
  return bad("not found", 404);
}

// Unix socket for b70-lease, b70q and project tools.
fs.mkdirSync(path.dirname(C.SOCKET), { recursive: true });
fs.rmSync(C.SOCKET, { force: true });
Bun.serve({ unix: C.SOCKET, fetch: (req) => handle(req, false) }); // idle timeout 10 s: clients long-poll for 8 s
fs.chmodSync(C.SOCKET, 0o600);

// Dashboard on localhost.
const web: Server<unknown> = Bun.serve({
  hostname: "127.0.0.1", port: C.PORT, reusePort: false, idleTimeout: 120, development: false,
  routes: { "/": dashboard },
  fetch(req, server) {
    if (new URL(req.url).pathname === "/ws") return server.upgrade(req, { data: undefined }) ? undefined : bad("upgrade failed");
    return handle(req, true);
  },
  websocket: {
    open(ws) { ws.subscribe("live"); ws.send(JSON.stringify({ type: "hello", state: q.view(), ring: q.ring })); },
    message() {},
  },
});

let lastPush = 0, pending: Timer | undefined;
const push = () => { lastPush = Date.now(); pending = undefined; web.publish("live", JSON.stringify({ type: "state", state: q.view() })); };
q.on((ev, data) => {
  if (ev === "sample") web.publish("live", JSON.stringify({ type: "sample", point: data }));
  if (Date.now() - lastPush > 500) push(); else pending ??= setTimeout(push, 500);
});

setInterval(() => { try { q.sample(); } catch (e) { console.error("sample:", e); } }, C.SAMPLE_MS);
setInterval(() => { try { q.tick(); } catch (e) { console.error("tick:", e); } }, 1000);
watchKernel(q.card.slot, (line) => { console.error("xe fault:", line); q.fault(line); });
q.sample();
console.error(`b70q: card ${q.card.slot} (${q.card.render}); socket ${C.SOCKET}; dashboard http://127.0.0.1:${C.PORT}/; ${q.active.size} active tickets`);
