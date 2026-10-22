import { Database } from "bun:sqlite";
import * as fs from "node:fs";
import { DB_PATH, KEEP_DAYS, STATE } from "./config";

export type Kind = "run" | "acquire" | "submit";
export type State = "waiting" | "granted" | "done" | "cancelled" | "failed";

export interface Ticket {
  id: number;
  owner: string;
  kind: Kind;
  state: State;
  seq: number; // queue order (bump moves it to the front)
  priority: number; // 1 = next in line (NQ game master, `b70q now`)
  argv: string; // JSON array
  cwd: string;
  jobtype: string; // owner:program, the key for learned stats
  client_pid: number | null; // the shim / runner process that owns the ticket
  client_start: number | null;
  child_pid: number | null;
  min_gib: number;
  ttl_s: number | null;
  created_at: number;
  granted_at: number | null;
  ended_at: number | null;
  exit_code: number | null;
  end_reason: string | null;
  peak_vram: number; peak_gtt: number; peak_sys: number; peak_rss: number;
  energy_j: number; busy_s: number;
  log: string | null;
  opts: string; // JSON Opts
  note: string | null; // human label shown in the queue
  grp: number | null; // first ticket of a chunked/retried series
  chunk: number; // 0 = first run of the series
  attempt: number; // 0 = first try
}

// Optional job settings (all kinds unless noted).
export interface Opts {
  creates?: string; // skip the job (done, exit 0) when this path already exists
  on_exit?: string[]; // argv the daemon runs when the ticket ends: env B70Q_TICKET/B70Q_EXIT/B70Q_STATE/B70Q_REASON/B70Q_LOG
  exit_file?: string; // the exit code (or the end reason) is written here when the ticket ends
  retries?: number; // submit: re-queue after a failure this many times
  more_code?: number; // submit, chunked: exiting with this code means "more to do"; the job is queued again at the back
  idle_s?: number; // server: stopped when the GPU has been idle this long while others wait
  ram_gib?: number; // wait until this much host RAM is available before starting
  vram_gib?: number; // expected VRAM, shown in the queue
  server?: boolean;
}

export function open(): Database {
  fs.mkdirSync(STATE, { recursive: true });
  const db = new Database(DB_PATH, { create: true });
  db.exec("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA busy_timeout=5000;");
  db.exec(`
    CREATE TABLE IF NOT EXISTS tickets (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      owner TEXT NOT NULL, kind TEXT NOT NULL, state TEXT NOT NULL,
      seq REAL NOT NULL, priority INTEGER NOT NULL DEFAULT 0,
      argv TEXT NOT NULL DEFAULT '[]', cwd TEXT NOT NULL DEFAULT '', jobtype TEXT NOT NULL,
      client_pid INTEGER, client_start INTEGER, child_pid INTEGER,
      min_gib REAL NOT NULL DEFAULT 1, ttl_s INTEGER,
      created_at INTEGER NOT NULL, granted_at INTEGER, ended_at INTEGER,
      exit_code INTEGER, end_reason TEXT,
      peak_vram INTEGER NOT NULL DEFAULT 0, peak_gtt INTEGER NOT NULL DEFAULT 0,
      peak_sys INTEGER NOT NULL DEFAULT 0, peak_rss INTEGER NOT NULL DEFAULT 0,
      energy_j REAL NOT NULL DEFAULT 0, busy_s REAL NOT NULL DEFAULT 0,
      log TEXT
    );
    CREATE INDEX IF NOT EXISTS tickets_state ON tickets(state);
    CREATE INDEX IF NOT EXISTS tickets_owner ON tickets(owner);
    CREATE INDEX IF NOT EXISTS tickets_ended ON tickets(ended_at);
    CREATE TABLE IF NOT EXISTS samples (
      t INTEGER PRIMARY KEY, vram INTEGER, power REAL, util REAL, temp REAL, mem_avail INTEGER, holder INTEGER
    );
    CREATE TABLE IF NOT EXISTS alerts (
      id INTEGER PRIMARY KEY AUTOINCREMENT, t INTEGER NOT NULL, level TEXT NOT NULL, kind TEXT NOT NULL,
      ticket INTEGER, msg TEXT NOT NULL, cleared_at INTEGER
    );
    CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
  `);
  const cols = new Set((db.query("PRAGMA table_info(tickets)").all() as { name: string }[]).map((c) => c.name));
  for (const [c, def] of [["opts", "TEXT NOT NULL DEFAULT '{}'"], ["note", "TEXT"], ["grp", "INTEGER"], ["chunk", "INTEGER NOT NULL DEFAULT 0"], ["attempt", "INTEGER NOT NULL DEFAULT 0"]])
    if (!cols.has(c)) db.exec(`ALTER TABLE tickets ADD COLUMN ${c} ${def}`);
  const cutoff = Date.now() - KEEP_DAYS * 86_400_000;
  db.query("DELETE FROM samples WHERE t < ?").run(cutoff);
  db.query("DELETE FROM alerts WHERE t < ? AND cleared_at IS NOT NULL").run(cutoff);
  return db;
}

// owner + the program that does the work: "myapp-h3:sd-cli", "dsv41-q5:in_rootfs.sh", "drama3-opt:acquire".
export function jobtype(owner: string, kind: Kind, argv: string[]): string {
  if (kind === "acquire") return `${owner}:acquire`;
  const wrappers = new Set(["bash", "sh", "env", "python", "python3", "bun", "node", "setsid", "nice", "timeout", "exec"]);
  let prog = "";
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const base = a.split("/").pop()!;
    if (wrappers.has(base)) {
      if (base === "bash" || base === "sh") { if (argv[i + 1] === "-c") { prog = (argv[i + 2] ?? "").trim().split(/\s+/)[0]?.split("/").pop() ?? ""; break; } }
      continue;
    }
    if (a.startsWith("-") || /^\w+=/.test(a) || /^\d+[smhd]?$/.test(a)) continue;
    prog = base; break;
  }
  return `${owner}:${prog || "?"}`;
}
