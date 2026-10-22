// Reads the Arc Pro B70: device VRAM (xe memory-region query ioctl), power, temperatures and GT busy from sysfs/hwmon,
// and per-process VRAM/GTT/system memory and engine busy from DRM fdinfo (the same sources as ~/work/rigtop).
// Opening the render node for the query loads no GPU libraries and allocates nothing on the card.
import * as fs from "node:fs";
import * as path from "node:path";
import { ioctl, ptr } from "./libc";

const XE_DEVICE_QUERY = 0xc0286440;
const XE_QUERY_MEM_REGIONS = 1;
const XE_MEM_CLASS_VRAM = 1;
const CLK_TCK = 100;

const read = (p: string) => { try { return fs.readFileSync(p, "utf8").trim(); } catch { return undefined; } };
const readInt = (p: string) => { const v = read(p); return v === undefined ? undefined : Number(v); };

export interface Proc {
  pid: number; ppid: number; comm: string; start: number; // start: clock ticks since boot (detects pid reuse)
  ticket?: number; // B70Q_TICKET from the environment
  rss: number; cpu: number; // bytes; utime+stime in seconds
}

export interface Client {
  pid: number; comm: string; cmd: string; ticket?: number;
  vram: number; gtt: number; sys: number; busy: number; // bytes; busy 0..1 (max engine)
}

export interface Sample {
  t: number;
  vramUsed: number; vramTotal: number;
  power?: number; temp?: number; vramTemp?: number; util?: number; freq?: number;
  memAvail: number; memTotal: number;
  clients: Client[];
  procs: Map<number, Proc>;
}

export class Card {
  slot = ""; sysdev = ""; render = ""; fd = -1;
  private hwmon?: string;
  private gt?: string;
  private prev?: { t: number; energy?: number; idle?: number };
  private prevCycles = new Map<string, [number, number]>();

  constructor() {
    for (const node of fs.readdirSync("/sys/class/drm").filter((n) => n.startsWith("renderD")).sort()) {
      const dev = fs.realpathSync(`/sys/class/drm/${node}/device`);
      if (path.basename(fs.realpathSync(`${dev}/driver`)) !== "xe") continue;
      this.slot = path.basename(dev); this.sysdev = dev; this.render = `/dev/dri/${node}`;
      break;
    }
    if (!this.slot) throw new Error("no xe GPU found");
    this.fd = fs.openSync(this.render, "r+");
    const hw = fs.readdirSync(`${this.sysdev}/hwmon`);
    this.hwmon = hw.length ? `${this.sysdev}/hwmon/${hw[0]}` : undefined;
    const tile = `${this.sysdev}/tile0`;
    for (const gt of fs.existsSync(tile) ? fs.readdirSync(tile).sort() : [])
      if (read(`${tile}/${gt}/gtidle/name`)?.endsWith("-rc")) { this.gt = `${tile}/${gt}`; break; }
  }

  private hwmonBy(label: string, kind: string): string | undefined {
    if (!this.hwmon) return;
    for (const f of fs.readdirSync(this.hwmon))
      if (f.startsWith(kind) && f.endsWith("_label") && read(`${this.hwmon}/${f}`) === label) return `${this.hwmon}/${f.slice(0, -6)}_input`;
  }
  private energyPath = () => this.hwmonBy("card", "energy") ?? this.hwmonBy("pkg", "energy");
  private paths?: { energy?: string; temp?: string; vramTemp?: string };

  memRegions(): { used: number; total: number } {
    const q = new BigUint64Array(5);
    const dv = new DataView(q.buffer);
    dv.setUint32(8, XE_QUERY_MEM_REGIONS, true);
    if (ioctl(this.fd, XE_DEVICE_QUERY, q) !== 0) return { used: 0, total: 0 };
    const buf = new Uint8Array(dv.getUint32(12, true));
    dv.setBigUint64(16, BigInt(ptr(buf)), true);
    if (ioctl(this.fd, XE_DEVICE_QUERY, q) !== 0) return { used: 0, total: 0 };
    const b = new DataView(buf.buffer);
    let used = 0, total = 0;
    for (let i = 0; i < b.getUint32(0, true); i++) {
      const o = 8 + i * 88;
      if (b.getUint16(o, true) !== XE_MEM_CLASS_VRAM) continue;
      total += Number(b.getBigUint64(o + 8, true));
      used += Number(b.getBigUint64(o + 16, true));
    }
    return { used, total };
  }

  sample(): Sample {
    const t = Date.now();
    this.paths ??= { energy: this.energyPath(), temp: this.hwmonBy("pkg", "temp"), vramTemp: this.hwmonBy("vram", "temp") };
    const energy = this.paths.energy ? readInt(this.paths.energy) : undefined;
    const idle = this.gt ? readInt(`${this.gt}/gtidle/idle_residency_ms`) : undefined;
    let power: number | undefined, util: number | undefined;
    if (this.prev && t > this.prev.t) {
      const dt = (t - this.prev.t) / 1000;
      if (energy !== undefined && this.prev.energy !== undefined) power = (energy - this.prev.energy) / 1e6 / dt;
      if (idle !== undefined && this.prev.idle !== undefined) util = Math.max(0, Math.min(1, 1 - (idle - this.prev.idle) / (dt * 1000)));
    }
    this.prev = { t, energy, idle };
    const { used, total } = this.memRegions();
    const mem = meminfo();
    const procs = scanProcs();
    return {
      t, vramUsed: used, vramTotal: total, power, util,
      temp: this.paths.temp ? (readInt(this.paths.temp) ?? 0) / 1000 : undefined,
      vramTemp: this.paths.vramTemp ? (readInt(this.paths.vramTemp) ?? 0) / 1000 : undefined,
      freq: this.gt ? readInt(`${this.gt}/freq0/act_freq`) : undefined,
      memAvail: mem.MemAvailable ?? 0, memTotal: mem.MemTotal ?? 0,
      clients: this.clients(procs), procs,
    };
  }

  // Every process with this card open: VRAM/GTT/system resident and engine busy, one entry per pid.
  private clients(procs: Map<number, Proc>): Client[] {
    const seen = new Set<string>();
    const per = new Map<number, Client>();
    const cycles = new Map<string, [number, number]>();
    const me = process.pid;
    for (const pid of procs.keys()) {
      if (pid === me) continue;
      let fds: string[];
      try { fds = fs.readdirSync(`/proc/${pid}/fd`); } catch { continue; }
      for (const fd of fds) {
        let info: Record<string, string>;
        try {
          if (!fs.readlinkSync(`/proc/${pid}/fd/${fd}`).startsWith("/dev/dri/")) continue;
          info = Object.fromEntries(fs.readFileSync(`/proc/${pid}/fdinfo/${fd}`, "utf8").split("\n")
            .filter((l) => l.includes(":")).map((l) => { const i = l.indexOf(":"); return [l.slice(0, i), l.slice(i + 1).trim()]; }));
        } catch { continue; }
        if (info["drm-pdev"] !== this.slot) continue;
        const cid = info["drm-client-id"];
        if (!cid || seen.has(cid)) continue;
        seen.add(cid);
        const p = procs.get(pid)!;
        const c = per.get(pid) ?? { pid, comm: p.comm, cmd: cmdline(pid) || p.comm, ticket: p.ticket, vram: 0, gtt: 0, sys: 0, busy: 0 };
        for (const [k, v] of Object.entries(info)) {
          if (!k.startsWith("drm-resident-")) continue;
          const region = k.slice(13);
          const bytes = parseMem(v);
          if (region.startsWith("vram")) c.vram += bytes;
          else if (region.startsWith("gtt")) c.gtt += bytes;
          else if (region.startsWith("system")) c.sys += bytes;
        }
        for (const k of Object.keys(info)) {
          if (!k.startsWith("drm-cycles-")) continue;
          const eng = k.slice(11);
          const cyc = Number(info[k]), tot = Number(info[`drm-total-cycles-${eng}`]);
          const cap = Number(info[`drm-engine-capacity-${eng}`] ?? 1) || 1;
          cycles.set(`${cid}/${eng}`, [cyc, tot]);
          const prev = this.prevCycles.get(`${cid}/${eng}`);
          if (prev && tot > prev[1]) c.busy = Math.max(c.busy, Math.min(1, (cyc - prev[0]) / (tot - prev[1]) / cap));
        }
        per.set(pid, c);
      }
    }
    this.prevCycles = cycles;
    return [...per.values()].sort((a, b) => b.vram - a.vram);
  }
}

function parseMem(v: string): number {
  const [n, unit] = v.split(/\s+/);
  const mul = unit === "KiB" ? 1024 : unit === "MiB" ? 2 ** 20 : unit === "GiB" ? 2 ** 30 : 1;
  return Number(n) * mul;
}

function meminfo(): Record<string, number> {
  const out: Record<string, number> = {};
  for (const l of (read("/proc/meminfo") ?? "").split("\n")) {
    const m = l.match(/^(\w+):\s+(\d+)\s*kB/);
    if (m) out[m[1]] = Number(m[2]) * 1024;
  }
  return out;
}

export function cmdline(pid: number): string {
  return (read(`/proc/${pid}/cmdline`) ?? "").replace(/\0/g, " ").trim();
}

export function procStart(pid: number): number | undefined {
  const s = read(`/proc/${pid}/stat`);
  if (!s) return;
  return Number(s.slice(s.lastIndexOf(")") + 2).split(" ")[19]);
}

// One pass over /proc: our processes with their ticket tag, memory and CPU time.
export function scanProcs(): Map<number, Proc> {
  const out = new Map<number, Proc>();
  const page = 4096;
  for (const name of fs.readdirSync("/proc")) {
    if (!/^\d+$/.test(name)) continue;
    const pid = Number(name);
    const stat = read(`/proc/${pid}/stat`);
    if (!stat) continue;
    const r = stat.lastIndexOf(")");
    const comm = stat.slice(stat.indexOf("(") + 1, r);
    const f = stat.slice(r + 2).split(" ");
    let ticket: number | undefined;
    try {
      const env = fs.readFileSync(`/proc/${pid}/environ`, "latin1");
      const i = env.indexOf("B70Q_TICKET=");
      if (i >= 0 && (i === 0 || env[i - 1] === "\0")) ticket = Number(env.slice(i + 12, env.indexOf("\0", i)));
    } catch {}
    out.set(pid, {
      pid, ppid: Number(f[1]), comm, start: Number(f[19]), ticket,
      rss: Number(f[21]) * page, cpu: (Number(f[11]) + Number(f[12])) / CLK_TCK,
    });
  }
  return out;
}
