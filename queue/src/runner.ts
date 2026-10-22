// Runs one submitted job inside its systemd user unit (b70q-<id>.service): reads the job spec from the daemon, runs
// the command with its saved environment and directory, logs to ~/.local/state/b70q/logs/<id>.log, reports the exit.
import * as fs from "node:fs";
import * as path from "node:path";
import { LOG_DIR } from "./config";
import { api } from "./client";

const id = Number(process.argv[2]);
const spec = await api<{ argv: string[]; cwd?: string; env: Record<string, string>; log?: string }>("GET", `/api/tickets/${id}/spec`);
const logPath = spec.log ?? path.join(LOG_DIR, `${id}.log`);
fs.mkdirSync(path.dirname(logPath), { recursive: true });
const log = fs.openSync(logPath, "a");
fs.writeSync(log, `# b70q #${id} started ${new Date().toISOString()} in ${spec.cwd}\n# ${JSON.stringify(spec.argv)}\n`);
const child = Bun.spawn(spec.argv, {
  cwd: spec.cwd || process.env.HOME, env: { ...spec.env, B70Q_TICKET: String(id) },
  stdin: "ignore", stdout: log, stderr: log,
});
await api("POST", `/api/tickets/${id}/started`, { client_pid: process.pid, child_pid: child.pid });
for (const s of ["SIGTERM", "SIGHUP", "SIGINT"] as const) process.on(s, () => child.kill(s));
const code = await child.exited;
const rc = child.signalCode ? 128 + ({ SIGTERM: 15, SIGKILL: 9, SIGINT: 2, SIGHUP: 1, SIGABRT: 6, SIGSEGV: 11 }[child.signalCode as string] ?? 1) : code;
fs.writeSync(log, `# b70q #${id} exited ${rc} at ${new Date().toISOString()}\n`);
for (let i = 0; i < 30; i++) { try { await api("POST", `/api/tickets/${id}/done`, { exit_code: rc }); break; } catch { await Bun.sleep(1000); } }
process.exit(rc);
