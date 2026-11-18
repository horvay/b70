import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";

type Any = any;
let state: Any = null;
let ring: Any[] = [];
let history: Any[] = [];
let stats: Any[] = [];
let rangeMin = 15;
const ownerById = new Map<number, string>();

const $ = <T extends HTMLElement = HTMLElement>(s: string) => document.querySelector(s) as T;
// The page stays open for days and these sections change every second, so patch the nodes that are there instead of
// replacing them. Anything watching the DOM can keep what we remove: Video Speed Controller's MutationObserver holds it
// on requestIdleCallback, which piled up GBs of detached nodes in a tab left in the background.
const parsed = document.createElement("template");
function morph(el: Node, to: Node) {
  let a = el.firstChild, b = to.firstChild;
  while (b) {
    const next = b.nextSibling;
    if (!a) el.appendChild(b);
    else {
      const after: ChildNode | null = a.nextSibling;
      if (a.nodeName !== b.nodeName) el.replaceChild(b, a);
      else if (a instanceof Element) {
        for (const { name } of [...a.attributes]) if (!(b as Element).hasAttribute(name)) a.removeAttribute(name);
        for (const { name, value } of (b as Element).attributes) if (a.getAttribute(name) !== value) a.setAttribute(name, value);
        morph(a, b);
      } else if (a.nodeValue !== b.nodeValue) a.nodeValue = b.nodeValue;
      a = after;
    }
    b = next;
  }
  while (a) { const after: ChildNode | null = a.nextSibling; el.removeChild(a); a = after; }
}
function setHTML(el: Element, html: string) { parsed.innerHTML = html; morph(el, parsed.content); }
function setText(el: Element, text: string) { if (el.textContent !== text) el.textContent = text; }
function setClass(el: Element, name: string) { if (el.className !== name) el.className = name; }
const esc = (s: unknown) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!);
const gib = (b: number | null | undefined, d = 1) => (b == null ? "–" : (b / 2 ** 30).toFixed(d));
const clock = (t?: number | null) => (t ? new Date(t).toLocaleTimeString("en-GB", { hour12: false, hour: "2-digit", minute: "2-digit" }) : "–");
const dateClock = (t: number) => {
  const d = new Date(t), today = new Date();
  return d.toDateString() === today.toDateString() ? clock(t) : `${d.toLocaleDateString("en-GB", { weekday: "short" })} ${clock(t)}`;
};
function dur(ms?: number | null): string {
  if (ms == null || !isFinite(ms)) return "–";
  const s = Math.max(0, Math.round(ms / 1000));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${String(s % 60).padStart(2, "0")}s`;
  return `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, "0")}m`;
}
function color(owner: string, a = 1) {
  let h = 0;
  for (const ch of owner.replace(/-(h3|stills|sharpen|q\d+|clone)$/, "")) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  return `hsla(${h % 360}, 62%, ${owner.endsWith("stills") ? 70 : 60}%, ${a})`;
}
const cmd = (argv: string[]) => esc((argv ?? []).join(" ").replace(/\s+/g, " "));
// Labels for a ticket: its kind (but in the history), server or standby, chunk n, retry n.
function tags(t: Any, kind = true): string {
  const o = t.opts ?? {};
  const out: string[] = [];
  if (kind && t.kind !== "run") out.push(`<span class="tag">${t.kind}</span>`);
  if (o.standby) out.push(`<span class="tag">standby${o.when_free ? " · when free" : ""}</span>`);
  else if (o.server) out.push(`<span class="tag">server${o.idle_s ? ` · idle ${dur(o.idle_s * 1000)}` : ""}</span>`);
  if (o.more_code || t.chunk) out.push(`<span class="tag">chunk ${t.chunk + 1}</span>`);
  if (t.attempt) out.push(`<span class="tag next">retry ${t.attempt}/${o.retries ?? "?"}</span>`);
  if (t.priority) out.push('<span class="tag next">next</span>');
  return out.join("");
}
const note = (t: Any) => (t.note ? `<b style="color:var(--text);font-family:system-ui">${esc(t.note)}</b> · ` : "");

async function post(path: string, body?: unknown) {
  const r = await fetch(path, { method: "POST", headers: { "x-b70q": "1", "content-type": "application/json" }, body: JSON.stringify(body ?? {}) });
  if (!r.ok) alert(await r.text());
}
(window as Any).b70q = {
  kill: (id: number, owner: string) => confirm(`Kill #${id} ${owner}? Its processes get SIGTERM, then SIGKILL after 15 s.`) && post(`/api/tickets/${id}/kill`),
  release: (id: number) => post(`/api/tickets/${id}/release`),
  cancel: (id: number, owner: string) => confirm(`Take #${id} ${owner} out of the queue?`) && post(`/api/tickets/${id}/cancel`),
  bump: (id: number) => post(`/api/tickets/${id}/bump`),
  ack: (id: number) => post(`/api/alerts/${id}/ack`),
  ackAll: () => Promise.all(state.alerts.map((a: Any) => post(`/api/alerts/${a.id}/ack`))),
};

// --- header -------------------------------------------------------------------------------------------------------------

function renderHeader() {
  const s = state;
  const pill = $("#pill");
  setClass(pill, `pill ${s.status.kind}`);
  setText(pill, s.status.kind.toUpperCase());
  $("#pausePill").hidden = !s.paused || s.status.kind === "paused"; // paused with a job still finishing
  setText($("#statusText"), s.status.kind === "running" && !s.paused ? "" : s.status.text);
  const k = s.card;
  if (k) {
    const kpi = (label: string, val: string, frac?: number) =>
      `<div class="kpi"><b>${val}</b><span>${label}</span>${frac != null ? `<div class="bar"><i style="width:${Math.min(100, frac * 100).toFixed(1)}%"></i></div>` : ""}</div>`;
    setHTML($("#kpis"), kpi("VRAM GiB", `${gib(k.vramUsed)} / ${gib(k.vramTotal, 0)}`, k.vramUsed / k.vramTotal) +
      kpi("power", `${k.power?.toFixed(0) ?? "–"} W`) + kpi("GPU busy", `${((k.util ?? 0) * 100).toFixed(0)}%`) +
      kpi("temp", `${k.temp?.toFixed(0) ?? "–"}°C`) + kpi("host RAM free", `${gib(k.memAvail)} GiB`, 1 - k.memAvail / k.memTotal));
  }
  const btn = $("#pauseBtn");
  setText(btn, s.paused ? "Resume queue" : "Pause queue");
  setClass(btn, s.paused ? "" : "danger");
  btn.onclick = () => {
    if (s.paused) return post("/api/resume");
    const reason = prompt("Pause the queue? The job on the card finishes, idle servers step aside, and nothing new starts until you resume.\n\nReason (shown in the queue):", "paused from the dashboard");
    if (reason !== null) post("/api/pause", { reason: reason || undefined });
  };
}

// --- now / queue ---------------------------------------------------------------------------------------------------------

function renderNow() {
  const el = $("#now .body");
  const s = state, h = s.holder;
  if (!h) {
    const last = history.find((t) => t.ended_at && t.granted_at);
    const next = s.queue[0];
    setHTML(el, `<div class="title"><span class="owner" style="color:var(--dim)">${s.status.kind === "idle" ? "Idle" : esc(s.status.kind)}</span></div>
      <div class="meta">${esc(s.status.text)}</div>
      ${last ? `<div class="meta">Last job: <span class="dot" style="background:${color(last.owner)}"></span>${esc(last.owner)} #${last.id}, ended ${dateClock(last.ended_at)} (${dur(Date.now() - last.ended_at)} ago)</div>` : ""}
      ${next ? `<div class="meta">Next: <b>${esc(next.owner)}</b> #${next.id}</div>` : ""}`);
    return;
  }
  const u = h.usage ?? {};
  const el2 = Date.now() - h.granted_at;
  const frac = h.est_ms ? el2 / h.est_ms : 0;
  const card = s.card ?? {};
  const gauge = (label: string, val: string, f?: number, sub = "") =>
    `<div class="gauge"><span>${label}</span><b>${val}</b>${f != null ? `<div class="bar"><i style="width:${Math.min(100, f * 100).toFixed(1)}%"></i></div>` : ""}${sub ? `<span>${sub}</span>` : ""}</div>`;
  setHTML(el, `
    <div class="title"><span class="dot" style="background:${color(h.owner)}"></span><span class="owner">${esc(h.owner)}</span>
      <span class="mono" style="color:var(--dim)">#${h.id}</span>${tags(h)}</div>
    <div class="meta">since ${dateClock(h.granted_at)} · running ${dur(el2)}${h.est_ms ? ` of ~${dur(h.est_ms)} (typical)` : ""} · waited ${dur(h.granted_at - h.created_at)}</div>
    ${h.est_ms ? `<div class="progress"><i style="width:${Math.min(100, frac * 100).toFixed(1)}%;${frac > 1 ? "background:var(--warn)" : ""}"></i></div>` : ""}
    <div class="cmd">${note(h)}${cmd(h.argv)}</div>
    <div class="gauges">
      ${gauge("VRAM", `${gib(u.vram)} GiB`, (u.vram ?? 0) / (card.vramTotal || 1), `peak ${gib(h.peak_vram)}`)}
      ${gauge("GTT + system", `${gib((u.gtt ?? 0) + (u.sys ?? 0))} GiB`, undefined, `peak ${gib(h.peak_gtt + h.peak_sys)}`)}
      ${gauge("host RAM", `${gib(u.rss)} GiB`, (u.rss ?? 0) / (card.memTotal || 1), `peak ${gib(h.peak_rss)}`)}
      ${gauge("GPU busy", `${((u.busy ?? 0) * 100).toFixed(0)}%`, u.busy ?? 0)}
      ${gauge("power", `${card.power?.toFixed(0) ?? "–"} W`, (card.power ?? 0) / 300)}
      ${gauge("energy", `${(h.energy_j / 3600).toFixed(0)} Wh`, undefined, `${(u.pids ?? []).length} processes`)}
    </div>
    <div class="row">
      <button class="danger" onclick="b70q.kill(${h.id}, '${esc(h.owner)}')">Kill</button>
      ${h.kind === "acquire" ? `<button onclick="b70q.release(${h.id})">Release</button><span class="cmd">ttl ${dur((h.ttl_s ?? 0) * 1000)}</span>` : ""}
      ${h.client_pid ? `<span class="cmd">pid ${h.client_pid}${h.child_pid ? ` → ${h.child_pid}` : ""}</span>` : ""}
    </div>`);
}

function learned(jobtype: string) { return stats.find((s) => s.jobtype === jobtype); }

function renderQueue() {
  const el = $("#queue .body");
  const q = state.queue;
  setHTML($("#queue h2"), `Queue <small>${q.length ? `${q.length} waiting · first in, first out` : ""}</small>` +
    (state.paused ? `<small style="color:var(--err)">paused: nothing starts until you resume</small>` : ""));
  if (!q.length) { setHTML(el, `<div class="empty">Nothing waiting.</div>`); return; }
  setHTML(el, `<table><tr><th></th><th>owner</th><th>#</th><th class="num">waiting</th><th class="num">starts ~</th><th class="num">takes ~</th><th class="num">VRAM ~</th><th>command</th><th></th></tr>
    ${q.map((t: Any) => {
      const l = learned(t.jobtype);
      return `<tr class="${t.priority ? "prio" : ""}"><td class="num">${t.position}</td>
        <td style="white-space:nowrap"><span class="dot" style="background:${color(t.owner)}"></span><span class="owner">${esc(t.owner)}</span>${tags(t)}</td>
        <td class="num">${t.id}</td><td class="num">${dur(Date.now() - t.created_at)}</td><td class="num">${state.paused ? "paused" : clock(t.eta_start)}</td>
        <td class="num">${l ? dur(t.est_ms) : `<span style="color:var(--faint)">${dur(t.est_ms)}?</span>`}</td>
        <td class="num">${l ? gib(l.p90_vram) : "–"}</td>
        <td class="cmd line" title="${cmd(t.argv)}">${note(t)}${cmd(t.argv)}</td>
        <td style="white-space:nowrap"><button class="small" title="move to the front" onclick="b70q.bump(${t.id})">▲</button>
          <button class="small danger" title="take out of the queue" onclick="b70q.cancel(${t.id}, '${esc(t.owner)}')">✕</button></td></tr>`;
    }).join("")}</table>`);
}

// The standby models stay loaded until the card is wanted for something else. The current one (loaded last) loads when
// nothing is queued; another one loads only for a request, in line like any job, and is the current one after that.
function renderStandby() {
  const list: Any[] = state.standby ?? [];
  $("#standby").hidden = !list.length;
  setHTML($("#standby h2"), list.length > 1 ? "Standby models" : "Standby model");
  setHTML($("#standby .body"), list.map((b) => {
    const led = b.loaded ? (b.busy ? "use" : "on") : b.position || b.current ? "wait" : "off";
    const what = b.loaded
      ? (b.busy ? "loaded, answering a request" : `loaded, idle${b.idle_ms != null ? ` for ${dur(b.idle_ms)}` : ""}; steps aside when a job is queued`)
      : b.position ? `in line to load (${b.position} in the queue)`
      : b.current ? "unloaded; the current model, loads as soon as nothing is queued"
      : "unloaded; loads when a request comes in";
    return `<div class="row"><span class="led ${led}"></span><span class="dot" style="background:${color(b.owner)}"></span><span class="owner">${esc(b.owner)}</span>
      <span class="mono" style="color:var(--dim)">#${b.id}</span><span class="what">${esc(b.note ?? "")} · ${what}</span></div>`;
  }).join(""));
}

// Alerts, and processes on the card outside the queue (the queue waits for them) or left over from a job.
function renderAlerts() {
  const a = state.alerts, o = state.others;
  $("#alerts").hidden = !a.length && !o.length;
  setHTML($("#alerts h2"), `Alerts${a.length > 1 ? '<button class="small close" onclick="b70q.ackAll()">dismiss all</button>' : ""}`);
  setHTML($("#alerts .body"), o.map((x: Any) =>
    `<div class="alert"><span class="lvl warn">${x.why === "outside" ? "outside the queue" : `leftover of #${x.ticket}`}</span>
      <span class="msg"><span class="mono">${esc(x.comm)} ${x.pid}</span> · ${gib(x.vram)} GiB · ${(x.busy * 100).toFixed(0)}% busy${x.owner ? ` · ${esc(x.owner)}` : ""}
      <div class="cmd">${esc(x.cmd).slice(0, 160)}</div></span>
      ${x.ticket ? `<button class="small danger" onclick="b70q.kill(${x.ticket}, '${esc(x.owner ?? "")}')">kill</button>` : ""}</div>`).join("") +
    a.slice(0, 20).map((x: Any) =>
    `<div class="alert"><span class="lvl ${x.level}">${x.level}</span><span class="msg">${esc(x.msg)}</span><time>${dateClock(x.t)}</time>
      <button class="small" title="dismiss" onclick="b70q.ack(${x.id})">✓</button></div>`).join(""));
}

// How a past job ended: "exit 0", "exit 1", or the reason it was stopped.
function result(t: Any): string {
  if (t.state === "granted") return "running";
  if (t.end_reason === "exited") return `exit ${t.exit_code ?? "?"}`;
  return `${t.end_reason ?? ""}${t.exit_code != null ? ` (exit ${t.exit_code})` : ""}`;
}

function renderHistory() {
  const rows = history.filter((t) => t.state !== "waiting");
  setHTML($("#history .body"), rows.length ? `<table class="past"><tr><th class="num">#</th><th>owner</th><th>job</th><th>result</th><th class="num">queued</th>
    <th class="num">waited</th><th class="num">ran</th><th class="num" title="peak VRAM, GiB">VRAM</th><th class="num opt" title="peak GTT + system memory, GiB">GTT+sys</th>
    <th class="num opt" title="peak host RAM, GiB">RAM</th><th class="num opt">Wh</th><th>command</th></tr>
    ${rows.map((t) => `<tr><td class="num">${t.id}</td><td class="clip" title="${esc(t.owner)}"><span class="dot" style="background:${color(t.owner)}"></span>${esc(t.owner)}</td>
      <td class="clip job" title="${esc(t.note)}">${tags(t, false)} ${esc(t.note)}</td>
      <td class="clip" title="${esc(result(t))}"><span class="state-${t.state}">${t.state === "granted" ? "running" : t.state}</span>
        ${t.state === "granted" ? "" : `<span class="dim">${esc(result(t))}</span>`}</td>
      <td class="num">${dateClock(t.created_at)}</td>
      <td class="num">${dur((t.granted_at ?? t.ended_at ?? Date.now()) - t.created_at)}</td>
      <td class="num">${t.granted_at ? dur((t.ended_at ?? Date.now()) - t.granted_at) : "–"}</td>
      <td class="num">${gib(t.peak_vram)}</td><td class="num opt">${gib(t.peak_gtt + t.peak_sys)}</td><td class="num opt">${gib(t.peak_rss)}</td>
      <td class="num opt">${(t.energy_j / 3600).toFixed(0)}</td>
      <td class="cmd line" title="${cmd(t.argv)}">${cmd(t.argv)}</td></tr>`).join("")}</table>`
    : `<div class="empty">No jobs in the last 3 days.</div>`);
}

function renderStats() {
  setHTML($("#stats .body"), stats.length ? `<table class="past"><tr><th>job type</th><th class="num">runs</th><th class="num">failed</th><th class="num">median</th><th class="num">p90</th>
    <th class="num">VRAM p90</th><th class="num">VRAM max</th><th class="num">RAM max</th><th class="num">last run</th></tr>
    ${stats.map((s) => `<tr><td><span class="dot" style="background:${color(s.owner)}"></span>${esc(s.jobtype)}</td><td class="num">${s.n}</td>
      <td class="num">${s.failed || ""}</td><td class="num">${dur(s.median_ms)}</td><td class="num">${dur(s.p90_ms)}</td><td class="num">${gib(s.p90_vram)}</td>
      <td class="num">${gib(s.max_vram)}</td><td class="num">${gib(s.max_rss)}</td><td class="num">${dateClock(s.last)}</td></tr>`).join("")}</table>`
    : `<div class="empty">No finished jobs yet. Every job's run time, VRAM and RAM are recorded here by job type.</div>`);
}

// --- live charts --------------------------------------------------------------------------------------------------------

const charts: Record<string, uPlot> = {};
const axis = { stroke: "#7d8797", grid: { stroke: "#1f2530", width: 1 }, ticks: { stroke: "#232a35", width: 1 }, font: "11px ui-monospace, monospace" };

// Shade the chart background by who held the card.
const holderBands = (u: uPlot) => {
  const xs = u.data[0] as number[];
  const pts = windowed();
  const ctx = u.ctx;
  let start = 0;
  for (let i = 1; i <= pts.length; i++) {
    if (i < pts.length && pts[i].holder === pts[start].holder) continue;
    const id = pts[start].holder;
    if (id) {
      const x0 = u.valToPos(xs[start], "x", true), x1 = u.valToPos(xs[Math.min(i, pts.length - 1)], "x", true);
      ctx.fillStyle = color(ownerById.get(id) ?? String(id), 0.1);
      ctx.fillRect(x0, u.bbox.top, Math.max(1, x1 - x0), u.bbox.height);
    }
    start = i;
  }
};

function makeChart(el: string, series: { label: string; stroke: string; fill?: string; dash?: number[] }[], range?: [number, number]): uPlot {
  const host = $(el);
  return new uPlot({
    width: host.clientWidth || 300, height: 150,
    scales: { x: { time: true }, y: range ? { range } : { range: (_u, _min, max) => [0, Math.max(max * 1.1, 1)] } },
    axes: [{ ...axis, values: (_u: uPlot, ticks: number[]) => ticks.map((t) => new Date(t * 1000).toLocaleTimeString("en-GB", { hour12: false, hour: "2-digit", minute: "2-digit" })) }, { ...axis, size: 42 }],
    legend: { show: true, live: true },
    cursor: { points: { size: 5 } },
    hooks: { drawClear: [holderBands] },
    series: [{}, ...series.map((s) => ({ label: s.label, stroke: s.stroke, fill: s.fill, width: 1.5, dash: s.dash, points: { show: false } }))],
  }, [[]], host);
}

function windowed() { const from = Date.now() - rangeMin * 60_000; return ring.filter((p) => p.t >= from); }

function updateCharts() {
  const pts = windowed();
  const x = pts.map((p) => p.t / 1000);
  const total = state?.card?.vramTotal ?? 32 * 2 ** 30;
  charts.vram ??= makeChart("#chVram", [{ label: "card", stroke: "#5fb3ff", fill: "#5fb3ff22" }, { label: "job", stroke: "#4fd18b" }], [0, total / 2 ** 30]);
  charts.power ??= makeChart("#chPower", [{ label: "W", stroke: "#f5b942", fill: "#f5b94222" }]);
  charts.util ??= makeChart("#chUtil", [{ label: "%", stroke: "#c18fff", fill: "#c18fff22" }], [0, 100]);
  charts.temp ??= makeChart("#chTemp", [{ label: "GPU", stroke: "#ff8a65" }, { label: "VRAM", stroke: "#ffcc80", dash: [4, 3] }]);
  charts.vram.setData([x, pts.map((p) => p.vram / 2 ** 30), pts.map((p) => (p.hvram != null ? p.hvram / 2 ** 30 : null))] as Any);
  charts.power.setData([x, pts.map((p) => p.power ?? null)] as Any);
  charts.util.setData([x, pts.map((p) => (p.util != null ? p.util * 100 : null))] as Any);
  charts.temp.setData([x, pts.map((p) => p.temp ?? null), pts.map((p) => p.vramTemp ?? null)] as Any);
}

function resizeCharts() {
  for (const [k, c] of Object.entries(charts)) {
    const host = $({ vram: "#chVram", power: "#chPower", util: "#chUtil", temp: "#chTemp" }[k]!);
    c.setSize({ width: host.clientWidth, height: 150 });
  }
}

// --- log ----------------------------------------------------------------------------------------------------------------
// The output of the job on the card, or of the last job while the card is idle: the file the job's processes write to
// (the daemon finds it), else a submitted job's own log. Progress bars redraw their line with \r; colours are dropped.

const ESCAPES = /\x1b\[[0-9;?]*[ -\/]*[@-~]|\x1b[\]P^_][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[@-Z\\-_]|[\x00-\x08\x0b\x0c\x0e-\x1a\x1c-\x1f\x7f]/g;
const MAX_LINES = 4000, MAX_CHARS = 16 * 2 ** 20, MAX_PARTIAL = 2 ** 20;
const pre = $("#logText");
const done = document.createTextNode(""), cur = document.createTextNode("");
const lg = { key: "", offset: 0, lines: 0, partial: "", ended: false, busy: false };
// what a line shows after its \r redraws: the last non-empty pass
const shown = (l: string) => { const p = l.split("\r"); for (let i = p.length - 1; i > 0; i--) if (p[i]) return p[i]; return p[0]; };
const home = (f: string) => f.replace(/^\/home\/[^/]+\//, "~/");

function logMessage(text: string) {
  setClass(pre, "msg"); setText(pre, text);
}

let lastHolder = 0;
function logSource(): { t: Any; live: boolean } | "stale" | undefined {
  if (state.holder) return { t: (lastHolder = state.holder.id, state.holder), live: true };
  if (lastHolder && !history.some((t) => t.id === lastHolder && t.ended_at)) return "stale"; // until history has its end
  const last = history.find((t) => t.ended_at && t.granted_at);
  return last && { t: last, live: false };
}

async function pollLog() {
  if (lg.busy || !state) return;
  const src = logSource();
  if (src === "stale") return;
  if (!src) { lg.key = ""; setText($("#logWhat"), ""); logMessage("Nothing has run on the card yet."); return; }
  const { t, live } = src, file = t.output ?? t.log;
  const key = `${t.id}|${file}|${t.ended_at ?? ""}`;
  if (key !== lg.key) {
    Object.assign(lg, { key, offset: -256 * 1024, lines: 0, partial: "", ended: false });
    const what = live ? `#${t.id} ${t.owner}` : `last job #${t.id} ${t.owner}, ended ${dateClock(t.ended_at)} (${`${t.exit_code ?? ""} ${t.end_reason ?? ""}`.trim()})`;
    setHTML($("#logWhat"), `${esc(what)}${file ? ` · <span title="${esc(file)}">${esc(home(file))}</span>` : ""}`);
    if (!file) return logMessage(`No log to show: the output of #${t.id} goes to the terminal or program that started it, not to a file.`);
    pre.className = ""; pre.replaceChildren(done, cur); done.data = ""; cur.data = "";
  }
  if (!file || lg.ended) return;
  lg.busy = true;
  try {
    const r = await fetch(`/api/tickets/${t.id}/log?src=output&offset=${lg.offset}`);
    if (!r.ok || key !== lg.key) return;
    const j = await r.json();
    if (lg.offset >= 0 && j.size < lg.offset) { lg.key = ""; return; } // the file was truncated or replaced: read it again
    let text = lg.partial + j.text;
    if (lg.offset < 0 && j.start > 0) text = text.slice(text.indexOf("\n") + 1); // started mid-line
    lg.offset = j.offset;
    lg.ended = !live;
    const lines = text.replace(ESCAPES, "").split("\n");
    lg.partial = lines.pop()!.slice(-MAX_PARTIAL); // a progress bar may never end its line
    if (!lines.length && cur.data === shown(lg.partial)) return;
    const end = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 40;
    if (lines.length) {
      done.appendData(lines.map(shown).join("\n") + "\n");
      lg.lines += lines.length;
      if (lg.lines > MAX_LINES * 1.25) { // keep the last MAX_LINES
        const all = done.data.split("\n"); done.data = all.slice(-MAX_LINES - 1).join("\n"); lg.lines = MAX_LINES;
      }
      if (done.length > MAX_CHARS) done.deleteData(0, done.length - MAX_CHARS / 2);
    }
    cur.data = shown(lg.partial);
    if (end) pre.scrollTop = pre.scrollHeight;
  } finally { lg.busy = false; }
}

// --- data ---------------------------------------------------------------------------------------------------------------

async function refresh() {
  const [h, s] = await Promise.all([
    fetch("/api/history?hours=72&limit=300").then((r) => r.json()),
    fetch("/api/stats").then((r) => r.json()),
  ]);
  history = h; stats = s;
  for (const t of history) ownerById.set(t.id, t.owner);
  renderHistory(); renderStats();
  if (state) renderNow();
}

let lastKey = "";
function onState(s: Any) {
  state = s;
  if (s.holder) ownerById.set(s.holder.id, s.holder.owner);
  for (const q of s.queue) ownerById.set(q.id, q.owner);
  renderHeader(); renderNow(); renderQueue(); renderStandby(); renderAlerts();
  const key = `${s.holder?.id}|${s.queue.map((q: Any) => q.id).join(",")}`;
  if (key !== lastKey) { lastKey = key; refresh(); }
}

function connect() {
  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.type === "hello") { ring = m.ring; onState(m.state); updateCharts(); }
    else if (m.type === "state") onState(m.state);
    else if (m.type === "sample") { ring.push(m.point); if (ring.length > 1800) ring.shift(); updateCharts(); }
  };
  ws.onclose = () => {
    const pill = $("#pill"); pill.className = "pill paused"; pill.textContent = "DISCONNECTED";
    $("#statusText").textContent = "lost the daemon; reconnecting…";
    setTimeout(connect, 2000);
  };
}

for (const b of document.querySelectorAll<HTMLButtonElement>("#range button")) b.onclick = () => {
  rangeMin = Number(b.dataset.min); document.querySelectorAll("#range button").forEach((x) => x.classList.toggle("on", x === b)); updateCharts();
};
// History and job types, in a dialog
const past = $<HTMLDialogElement>("#past");
$("#historyBtn").onclick = () => { refresh(); past.showModal(); };
$("#pastClose").onclick = () => past.close();
past.onclick = (ev) => { if (ev.target === past) past.close(); }; // a click on the backdrop
for (const b of document.querySelectorAll<HTMLButtonElement>("#pastTabs button")) b.onclick = () => {
  document.querySelectorAll("#pastTabs button").forEach((x) => x.classList.toggle("on", x === b));
  $("#history").hidden = b.dataset.tab !== "history"; $("#stats").hidden = b.dataset.tab !== "stats";
  $("#pastNote").textContent = b.dataset.tab === "history" ? HISTORY_NOTE : "learned from past runs; queue estimates come from these";
};
const HISTORY_NOTE = "last 3 days · VRAM, GTT+sys and RAM are peaks in GiB";
$("#pastNote").textContent = HISTORY_NOTE;
addEventListener("resize", () => { clearTimeout((window as Any)._rs); (window as Any)._rs = setTimeout(resizeCharts, 150); });
setInterval(refresh, 60_000);
setInterval(() => { if (state) { renderNow(); renderQueue(); renderStandby(); } }, 1000); // tick the clocks
setInterval(pollLog, 1000);
connect();
