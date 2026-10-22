// Talks to the daemon over its unix socket.
import { SOCKET } from "./config";

export class ApiError extends Error { constructor(public status: number, msg: string) { super(msg); } }

export async function api<T = any>(method: string, path: string, body?: unknown, timeoutMs = 10_000): Promise<T> {
  const res = await fetch(`http://b70q${path}`, {
    unix: SOCKET, method,
    body: body === undefined ? undefined : JSON.stringify(body),
    headers: { "content-type": "application/json" },
    signal: AbortSignal.timeout(timeoutMs),
  } as RequestInit);
  if (!res.ok) throw new ApiError(res.status, (await res.text()) || res.statusText);
  return res.json() as Promise<T>;
}

export async function up(): Promise<boolean> {
  try { await api("GET", "/api/ping", undefined, 3000); return true; } catch { return false; }
}

export const ts = () => new Date().toLocaleTimeString("en-GB", { hour12: false });
export const gib = (b: number | null | undefined, d = 1) => (b == null ? "?" : (b / 2 ** 30).toFixed(d));
export function dur(ms: number | null | undefined): string {
  if (ms == null || !isFinite(ms)) return "?";
  const s = Math.round(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m${String(s % 60).padStart(2, "0")}s`;
  return `${Math.floor(m / 60)}h${String(m % 60).padStart(2, "0")}m`;
}
