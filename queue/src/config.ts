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

// Scheduling (Greg, 2026-10-21): FIFO with 30-minute turns; NQ's game master goes next in line; manual runs outside
// the queue are detected and waited for.
export const TURN_MS = num("B70Q_TURN_MIN", 30) * 60_000;
export const GRACE_MS = num("B70Q_GRACE_S", 20) * 1000; // card kept for the turn owner's next job in a loop
export const NEXT_OWNERS = new RegExp(env("B70Q_NEXT_OWNERS", "^(nq|neverending)"), "i");

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
