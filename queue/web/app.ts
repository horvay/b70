import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";

type Any = any;
let state: Any = null;
let ring: Any[] = [];
let history: Any[] = [];
let stats: Any[] = [];
let samples: Any[] = [];
let rangeMin = 15;
let tlHours = 24;
const ownerById = new Map<number, string>();

const $ = <T extends HTMLElement = HTMLElement>(s: string) => document.querySelector(s) as T;
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
// Labels for a ticket: server, chunk n, retry n; plus its note.
function tags(t: Any): string {
  const o = t.opts ?? {};
  const out: string[] = [];
  if (t.kind !== "run") out.push(`<span class="tag">${t.kind}</span>`);
  if (o.server) out.push(`<span class="tag">server${o.idle_s ? ` · idle ${dur(o.idle_s * 1000)}` : ""}</span>`);
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
};

// --- header -------------------------------------------------------------------------------------------------------------

function renderHeader() {
  const s = state;
  const pill = $("#pill");
  pill.className = `pill ${s.status.kind}`;
  pill.textContent = s.status.kind.toUpperCase();
  $("#statusText").textContent = s.status.kind === "running" ? "" : s.status.text;
  const k = s.card;
  if (k) {
    const kpi = (label: string, val: string, frac?: number) =>
      `<div class="kpi"><b>${val}</b><span>${label}</span>${frac != null ? `<div class="bar"><i style="width:${Math.min(100, frac * 100).toFixed(1)}%"></i></div>` : ""}</div>`;
    $("#kpis").innerHTML = kpi("VRAM GiB", `${gib(k.vramUsed)} / ${gib(k.vramTotal, 0)}`, k.vramUsed / k.vramTotal) +
      kpi("power", `${k.power?.toFixed(0) ?? "–"} W`) + kpi("GPU busy", `${((k.util ?? 0) * 100).toFixed(0)}%`) +
      kpi("temp", `${k.temp?.toFixed(0) ?? "–"}°C`) + kpi("host RAM free", `${gib(k.memAvail)} GiB`, 1 - k.memAvail / k.memTotal);
  }
  const btn = $("#pauseBtn");
  btn.textContent = s.paused ? "Resume queue" : "Pause queue";
  btn.className = s.paused ? "" : "danger";
  btn.onclick = () => s.paused ? post("/api/resume") : post("/api/pause", { reason: prompt("Why pause the queue?", "paused from the dashboard") || undefined });
}

// --- now / queue ---------------------------------------------------------------------------------------------------------

function renderNow() {
  const el = $("#now .body");
  const s = state, h = s.holder;
  if (!h) {
    const last = history.find((t) => t.ended_at && t.granted_at);
    const next = s.queue[0];
    el.innerHTML = `<div class="title"><span class="owner" style="color:var(--dim)">${s.status.kind === "idle" ? "Idle" : esc(s.status.kind)}</span></div>
      <div class="meta">${esc(s.status.text)}</div>
      ${last ? `<div class="meta">Last job: <span class="dot" style="background:${color(last.owner)}"></span>${esc(last.owner)} #${last.id}, ended ${dateClock(last.ended_at)} (${dur(Date.now() - last.ended_at)} ago)</div>` : ""}
      ${next ? `<div class="meta">Next: <b>${esc(next.owner)}</b> #${next.id}</div>` : ""}`;
    return;
  }
  const u = h.usage ?? {};
  const el2 = Date.now() - h.granted_at;
  const frac = h.est_ms ? el2 / h.est_ms : 0;
  const turn = s.turn && s.turn.owner === h.owner && s.queue.some((q: Any) => q.owner !== h.owner)
    ? `turn: ${dur(s.turn.left_ms)} left, then ${esc(s.queue.find((q: Any) => q.owner !== h.owner)?.owner)}` : "";
  const card = s.card ?? {};
  const gauge = (label: string, val: string, f?: number, sub = "") =>
    `<div class="gauge"><span>${label}</span><b>${val}</b>${f != null ? `<div class="bar"><i style="width:${Math.min(100, f * 100).toFixed(1)}%"></i></div>` : ""}${sub ? `<span>${sub}</span>` : ""}</div>`;
  el.innerHTML = `
    <div class="title"><span class="dot" style="background:${color(h.owner)}"></span><span class="owner">${esc(h.owner)}</span>
      <span class="mono" style="color:var(--dim)">#${h.id}</span>${tags(h)}</div>
    <div class="meta">since ${dateClock(h.granted_at)} · running ${dur(el2)}${h.est_ms ? ` of ~${dur(h.est_ms)} (typical)` : ""} · waited ${dur(h.granted_at - h.created_at)}${turn ? ` · ${turn}` : ""}</div>
    ${h.est_ms ? `<div class="progress"><i style="width:${Math.min(100, frac * 100)}%;${frac > 1 ? "background:var(--warn)" : ""}"></i></div>` : ""}
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
    </div>`;
}

function learned(jobtype: string) { return stats.find((s) => s.jobtype === jobtype); }

function renderQueue() {
  const el = $("#queue .body");
  const q = state.queue;
  $("#queue h2").innerHTML = `Queue <small>${q.length ? `${q.length} waiting · FIFO, ${dur(state.config.turn_ms)} turns` : ""}</small>`;
  if (!q.length) { el.innerHTML = `<div class="empty">Nothing waiting.</div>`; return; }
  el.innerHTML = `<table><tr><th></th><th>owner</th><th>#</th><th class="num">waiting</th><th class="num">starts ~</th><th class="num">takes ~</th><th class="num">VRAM ~</th><th>command</th><th></th></tr>
    ${q.map((t: Any) => {
      const l = learned(t.jobtype);
      return `<tr class="${t.priority ? "prio" : ""}"><td class="num">${t.position}</td>
        <td><span class="dot" style="background:${color(t.owner)}"></span><span class="owner">${esc(t.owner)}</span>${tags(t)}</td>
        <td class="num">${t.id}</td><td class="num">${dur(Date.now() - t.created_at)}</td><td class="num">${clock(t.eta_start)}</td>
        <td class="num">${l ? dur(t.est_ms) : `<span style="color:var(--faint)">${dur(t.est_ms)}?</span>`}</td>
        <td class="num">${l ? gib(l.p90_vram) : "–"}</td>
        <td class="cmd">${note(t)}${cmd(t.argv).slice(0, 140)}</td>
        <td style="white-space:nowrap"><button class="small" title="move to the front" onclick="b70q.bump(${t.id})">▲</button>
          <button class="small danger" title="take out of the queue" onclick="b70q.cancel(${t.id}, '${esc(t.owner)}')">✕</button></td></tr>`;
    }).join("")}</table>`;
}

function renderAlerts() {
  const a = state.alerts;
  $("#alerts .body").innerHTML = a.length ? a.slice(0, 20).map((x: Any) =>
    `<div class="alert"><span class="lvl ${x.level}">${x.level}</span><span class="msg">${esc(x.msg)}</span><time>${dateClock(x.t)}</time>
      <button class="small" title="dismiss" onclick="b70q.ack(${x.id})">✓</button></div>`).join("") : `<div class="empty">No alerts.</div>`;
}

function renderOthers() {
  const o = state.others;
  $("#others .body").innerHTML = o.length ? `<table><tr><th>what</th><th>process</th><th class="num">VRAM</th><th class="num">busy</th><th>command</th><th></th></tr>
    ${o.map((x: Any) => `<tr><td>${x.why === "outside" ? '<span class="lvl warn">outside the queue</span>' : `<span class="lvl warn">leftover of #${x.ticket}</span> ${esc(x.owner ?? "")}`}</td>
      <td class="mono">${esc(x.comm)} ${x.pid}</td><td class="num">${gib(x.vram)}</td><td class="num">${(x.busy * 100).toFixed(0)}%</td><td class="cmd">${esc(x.cmd).slice(0, 120)}</td>
      <td>${x.ticket ? `<button class="small danger" onclick="b70q.kill(${x.ticket}, '${esc(x.owner ?? "")}')">kill</button>` : ""}</td></tr>`).join("")}</table>`
    : `<div class="empty">Only queue jobs are using the card. Anything else using it (your manual runs) shows up here, and the queue waits for it.</div>`;
}

function renderHistory() {
  const rows = history.filter((t) => t.state !== "waiting").slice(0, 40);
  $("#history .body").innerHTML = `<table><tr><th>#</th><th>owner</th><th>state</th><th class="num">queued</th><th class="num">waited</th><th class="num">ran</th>
    <th class="num">VRAM</th><th class="num">GTT+sys</th><th class="num">RAM</th><th class="num">Wh</th><th>result</th><th>command</th></tr>
    ${rows.map((t) => `<tr><td class="num">${t.id}</td><td><span class="dot" style="background:${color(t.owner)}"></span>${esc(t.owner)}${tags(t)}</td>
      <td class="state-${t.state}">${t.state}</td><td class="num">${dateClock(t.created_at)}</td>
      <td class="num">${dur((t.granted_at ?? t.ended_at ?? Date.now()) - t.created_at)}</td>
      <td class="num">${t.granted_at ? dur((t.ended_at ?? Date.now()) - t.granted_at) : "–"}</td>
      <td class="num">${gib(t.peak_vram)}</td><td class="num">${gib(t.peak_gtt + t.peak_sys)}</td><td class="num">${gib(t.peak_rss)}</td>
      <td class="num">${(t.energy_j / 3600).toFixed(0)}</td>
      <td>${t.state === "granted" ? "running" : esc(`${t.exit_code ?? ""} ${t.end_reason ?? ""}`.trim())}</td>
      <td class="cmd">${note(t)}${cmd(t.argv).slice(0, 120)}</td></tr>`).join("")}</table>`;
}

function renderStats() {
  $("#stats .body").innerHTML = stats.length ? `<table><tr><th>job type</th><th class="num">runs</th><th class="num">failed</th><th class="num">median</th><th class="num">p90</th>
    <th class="num">VRAM p90</th><th class="num">VRAM max</th><th class="num">RAM max</th><th class="num">last run</th></tr>
    ${stats.map((s) => `<tr><td><span class="dot" style="background:${color(s.owner)}"></span>${esc(s.jobtype)}</td><td class="num">${s.n}</td>
      <td class="num">${s.failed || ""}</td><td class="num">${dur(s.median_ms)}</td><td class="num">${dur(s.p90_ms)}</td><td class="num">${gib(s.p90_vram)}</td>
      <td class="num">${gib(s.max_vram)}</td><td class="num">${gib(s.max_rss)}</td><td class="num">${dateClock(s.last)}</td></tr>`).join("")}</table>`
    : `<div class="empty">No finished jobs yet. Every job's run time, VRAM and RAM are recorded here by job type.</div>`;
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
  renderTimeline();
}

// --- timeline (Gantt) ---------------------------------------------------------------------------------------------------

function renderTimeline() {
  const host = $("#timeline");
  const W = host.clientWidth || 1000;
  const t1 = Date.now(), t0 = t1 - tlHours * 3600_000;
  const tickets = history.filter((t) => (t.ended_at ?? t1) >= t0 && t.created_at <= t1 && (t.granted_at || t.state === "waiting"));
  const lanes: string[] = [];
  for (const t of [...tickets].sort((a, b) => a.created_at - b.created_at)) if (!lanes.includes(t.owner)) lanes.push(t.owner);
  const L = 180, R = 12, top = 58, rowH = 24;
  const H = top + Math.max(1, lanes.length) * rowH + 8;
  const x = (t: number) => L + ((Math.max(t0, Math.min(t1, t)) - t0) / (t1 - t0)) * (W - L - R);
  let svg = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">`;
  // power strip: how hard the card worked
  const ps = samples.filter((s) => s.t >= t0);
  if (ps.length) {
    const maxP = 320, sh = 26, base = 30;
    svg += `<text x="${L - 8}" y="${base - 8}" text-anchor="end">power</text>`;
    svg += `<path d="M${x(ps[0].t)},${base} ${ps.map((s) => `L${x(s.t).toFixed(1)},${(base - Math.min(1, (s.power ?? 0) / maxP) * sh).toFixed(1)}`).join(" ")} L${x(ps[ps.length - 1].t)},${base} Z" fill="#f5b94233" stroke="#f5b942" stroke-width="1"/>`;
  }
  // hour grid
  const step = tlHours <= 6 ? 3600_000 / 2 : tlHours <= 24 ? 2 * 3600_000 : 6 * 3600_000;
  for (let t = Math.ceil(t0 / step) * step; t <= t1; t += step) {
    svg += `<line x1="${x(t)}" x2="${x(t)}" y1="${top - 14}" y2="${H}" stroke="#1f2530"/><text x="${x(t)}" y="${top - 18}" text-anchor="middle">${dateClock(t)}</text>`;
  }
  lanes.forEach((o, i) => {
    const y = top + i * rowH;
    svg += `<rect x="0" y="${y}" width="${W}" height="${rowH}" fill="${i % 2 ? "#ffffff05" : "transparent"}"/>`;
    svg += `<rect x="8" y="${y + 8}" width="9" height="9" rx="2" fill="${color(o)}"/><text class="lane-label" x="24" y="${y + 16}">${esc(o.length > 24 ? o.slice(0, 23) + "…" : o)}</text>`;
  });
  for (const t of tickets) {
    const y = top + lanes.indexOf(t.owner) * rowH;
    const g = t.granted_at, e = t.ended_at ?? t1;
    // waiting: thin line from queued to start
    const w1 = g ?? e;
    if (w1 - t.created_at > 5000) svg += `<line x1="${x(t.created_at)}" x2="${x(w1)}" y1="${y + 12}" y2="${y + 12}" stroke="${color(t.owner, 0.45)}" stroke-width="2" stroke-dasharray="2 3"/>`;
    if (g) {
      const fail = t.state === "failed";
      svg += `<rect data-id="${t.id}" x="${x(g)}" y="${y + 4}" width="${Math.max(2, x(e) - x(g))}" height="${rowH - 8}" rx="3"
        fill="${color(t.owner, t.state === "granted" ? 0.95 : 0.75)}" stroke="${fail ? "#ff6b6b" : "none"}" stroke-width="${fail ? 1.5 : 0}"/>`;
    }
  }
  svg += `<line x1="${x(t1)}" x2="${x(t1)}" y1="${top - 14}" y2="${H}" stroke="#5fb3ff" stroke-dasharray="3 3"/>`;
  svg += `</svg>`;
  host.innerHTML = lanes.length ? svg : `<div class="empty">No jobs in the last ${tlHours} h.</div>${svg}`;
  const tip = $("#tip");
  host.querySelectorAll("rect[data-id]").forEach((r) => {
    r.addEventListener("mousemove", (ev) => {
      const t = history.find((h) => h.id === Number((r as SVGElement).dataset.id));
      if (!t) return;
      const me = ev as MouseEvent;
      tip.innerHTML = `<b>#${t.id} ${esc(t.owner)}</b> <span class="tag">${t.kind}</span><div>${dateClock(t.granted_at)} – ${t.ended_at ? dateClock(t.ended_at) : "now"}
        · ran ${dur((t.ended_at ?? Date.now()) - t.granted_at)} · waited ${dur(t.granted_at - t.created_at)}</div>
        <div>VRAM ${gib(t.peak_vram)} GiB · GTT+sys ${gib(t.peak_gtt + t.peak_sys)} · RAM ${gib(t.peak_rss)} GiB · ${(t.energy_j / 3600).toFixed(0)} Wh</div>
        <div>${esc(t.state)} ${esc(t.exit_code ?? "")} ${esc(t.end_reason ?? "")}</div><div class="cmd">${cmd(t.argv).slice(0, 300)}</div>`;
      tip.style.display = "block";
      tip.style.left = `${Math.min(me.clientX + 14, innerWidth - 480)}px`;
      tip.style.top = `${me.clientY + 14}px`;
    });
    r.addEventListener("mouseleave", () => (tip.style.display = "none"));
  });
}

// --- data ---------------------------------------------------------------------------------------------------------------

async function refresh() {
  const [h, s, sm] = await Promise.all([
    fetch(`/api/history?hours=${Math.max(tlHours, 24)}&limit=3000`).then((r) => r.json()),
    fetch("/api/stats").then((r) => r.json()),
    fetch(`/api/samples?hours=${tlHours}`).then((r) => r.json()),
  ]);
  history = h; stats = s; samples = sm;
  for (const t of history) ownerById.set(t.id, t.owner);
  renderHistory(); renderStats(); renderTimeline();
  if (state) renderNow();
}

let lastKey = "";
function onState(s: Any) {
  state = s;
  if (s.holder) ownerById.set(s.holder.id, s.holder.owner);
  for (const q of s.queue) ownerById.set(q.id, q.owner);
  renderHeader(); renderNow(); renderQueue(); renderAlerts(); renderOthers();
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
for (const b of document.querySelectorAll<HTMLButtonElement>("#tlRange button")) b.onclick = () => {
  tlHours = Number(b.dataset.h); document.querySelectorAll("#tlRange button").forEach((x) => x.classList.toggle("on", x === b)); refresh();
};
addEventListener("resize", () => { clearTimeout((window as Any)._rs); (window as Any)._rs = setTimeout(resizeCharts, 150); });
setInterval(refresh, 60_000);
setInterval(() => { if (state) { renderNow(); renderQueue(); } }, 1000); // tick the clocks
connect();
