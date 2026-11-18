// Settings for the B70 queue daemon. Everything can be overridden by env for a test instance.
import * as path from "node:path";

const HOME = process.env.HOME!;
const env = (k: string, d: string) => process.env[k] ?? d;
const num = (k: string, d: number) => (process.env[k] ? Number(process.env[k]) : d);

export const STATE = env("B70Q_STATE", path.join(HOME, ".local/state/b70q"));
export const SOCKET = env("B70Q_SOCKET", path.join(process.env.XDG_RUNTIME_DIR ?? `/run/user/${process.getuid?.()}`, "b70q.sock"));
export const PORT = num("B70Q_PORT", 7870);
export const DB_PATH = path.join(STATE, "b70q.db");
export const LOG_DIR = path.join(STATE, "logs");
export const SPEC_DIR = path.join(STATE, "specs");

// The old flock-based b70-lease. The daemon holds this lock while it grants the card, so scripts still on the old
// lease (and the shim's fallback when the daemon is down) stay mutually exclusive with the queue.
export const LEGACY_DIR = path.join(HOME, ".local/state/b70");
export const LEGACY_LOCK = env("B70Q_LEGACY_LOCK", path.join(LEGACY_DIR, "lock")); // "off" disables (tests)
export const LEGACY_HOLDER = path.join(LEGACY_DIR, "holder");
export const LEGACY_SCRIPT = path.join(import.meta.dir, "..", "legacy", "b70-lease");

// Scheduling (Greg, 2026-11-17): first in, first out; only the submitter's --next / `b70q now` moves a ticket up.
// Manual runs outside the queue are detected and waited for.
// Standby servers, the card's idle model (Greg, 2026-10-23): one loads once nothing is queued (Greg, 2026-10-25: no
// wait after the card frees; B70Q_STANDBY_SETTLE_S brings one back), and steps aside for another job once no request has run on it for STANDBY_QUIET.
export const STANDBY_SETTLE_MS = num("B70Q_STANDBY_SETTLE_S", 0) * 1000;
export const STANDBY_QUIET_MS = num("B70Q_STANDBY_QUIET_S", 60) * 1000;
export const STANDBY_REPORT_MS = 15_000; // a standby server that hasn't reported for this long is judged by the GPU
export const STANDBY_YIELD_MS = 60_000; // told to step aside this long ago and still on the card: stopped

// Card. Outside-queue processes count as using the card above this much VRAM.
export const FOREIGN_MIN_BYTES = num("B70Q_FOREIGN_MIB", 256) * 2 ** 20;
export const DEFAULT_MIN_GIB = 1; // grant only when device VRAM in use is below this (old b70-lease --min-gib)

// Health
export const FAULT_PAUSE_MS = 2 * 60_000; // queue holds this long after an xe fault
export const FAULT_HOLD_COUNT = 3; // this many faults within FAULT_WINDOW_MS: hold until resumed by hand
export const FAULT_WINDOW_MS = 60 * 60_000;
export const STALL_MS = 15 * 60_000; // holder with no GPU work, no CPU time and no VRAM change this long
export const CPU_FALLBACK_MS = 3 * 60_000;
export const LOW_RAM_BYTES = 8 * 2 ** 30;

export const SAMPLE_MS = 2000;
export const RING_SAMPLES = 1800; // 1 h of 2 s samples in memory
export const AGG_MS = 30_000; // 30 s aggregates persisted for the 24 h charts
export const KEEP_DAYS = 30;
