'use strict';
// The chat page: chats live in this browser's localStorage (and in exported files), nothing goes to the server except
// the prompt of the reply being generated. prompt.js builds the prompt; replies come from /v1/completions so that a
// reply can be stopped, edited anywhere (thinking or answer) and continued.
const $ = id => document.getElementById(id);
const KEY = 'chatapp.v1';
const uid = () => Math.random().toString(36).slice(2, 10);
const DEFAULTS = { temp: 0.8, minp: 0.05, topp: 1, maxt: 4096, think: 'on', effort: 'xhigh', model: '' };

let db;
try { db = JSON.parse(localStorage.getItem(KEY)); } catch (e) { db = null; }
if (!db || !db.chats) db = { cur: null, chats: [], settings: {} };
db.settings = Object.assign({}, DEFAULTS, db.settings);

let busy = null;          // AbortController of the reply being generated
let editing = null;       // message id being edited
const thinkOpen = {};     // message id -> the user opened or closed its thinking box
const undo = [];          // {chat id, JSON of {system, messages}}
let modelName = null;

document.head.insertAdjacentHTML('beforeend', '<style>body.busy .acts, body.busy .insert, body.busy .vers { pointer-events:none; opacity:.35 }</style>');

// ---------------------------------------------------------------------------------------------------------- state
let saveTimer = null;
function save(now) {
  if (now) { clearTimeout(saveTimer); saveTimer = null; }
  else if (saveTimer) return;
  else { saveTimer = setTimeout(() => save(true), 800); return; }
  try { localStorage.setItem(KEY, JSON.stringify(db)); } catch (e) { $('stat').textContent = 'could not save: ' + e; }
}
const chat = () => db.chats.find(c => c.id === db.cur);
const cv = m => m.versions[m.v];
const msgIndex = id => chat().messages.findIndex(m => m.id === id);
function newChat() {
  const c = { id: uid(), title: '', system: '', messages: [], updated: Date.now() };
  db.chats.unshift(c); db.cur = c.id; editing = null; save(true); renderAll();
}
function snap() {
  const c = chat(); if (!c) return;
  undo.push({ id: c.id, json: JSON.stringify({ system: c.system, messages: c.messages }) });
  if (undo.length > 60) undo.shift();
}
function touch() { chat().updated = Date.now(); save(); }
const userMsg = text => ({ id: uid(), role: 'user', text });
const modelMsg = (think, text, closed) => ({ id: uid(), role: 'assistant', v: 0, versions: [{ think: think || '', text: text || '', closed: !!closed }] });

// -------------------------------------------------------------------------------------------------------- rendering
const esc = s => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const fmt = s => esc(s).replace(/`([^`\n]+)`/g, '<code>$1</code>').replace(/\*\*([^*\n]+)\*\*/g, '<b>$1</b>').replace(/\*([^*\n]+)\*/g, '<em>$1</em>');
const nearBottom = () => { const l = $('log'); return l.scrollHeight - l.scrollTop - l.clientHeight < 160; };

function button(label, title, fn, cls) {
  const b = document.createElement('button');
  b.textContent = label; b.title = title || ''; if (cls) b.className = cls;
  b.type = 'button'; b.onclick = e => { e.stopPropagation(); fn(); };
  return b;
}

function renderChats() {
  const box = $('chats'); box.innerHTML = '';
  for (const c of db.chats.slice().sort((a, b) => b.updated - a.updated)) {
    const d = document.createElement('div'); d.className = 'chat' + (c.id === db.cur ? ' on' : '');
    const t = document.createElement('span');
    t.textContent = c.title || (c.messages.find(m => m.role === 'user') || {}).text || 'New chat';
    d.appendChild(t);
    const r = document.createElement('i'); r.textContent = '✎'; r.title = 'rename';
    r.onclick = e => { e.stopPropagation(); const n = prompt('Chat name', t.textContent); if (n !== null) { c.title = n.trim(); save(true); renderChats(); } };
    const x = document.createElement('i'); x.textContent = '✕'; x.title = 'delete this chat';
    x.onclick = e => {
      e.stopPropagation(); if (!confirm('Delete this chat for good?')) return;
      db.chats = db.chats.filter(k => k !== c);
      if (db.cur === c.id) db.cur = db.chats.length ? db.chats[0].id : null;
      if (!db.cur) return newChat();
      save(true); renderAll();
    };
    d.append(r, x);
    d.onclick = () => { if (busy || db.cur === c.id) return; db.cur = c.id; editing = null; save(true); renderAll(); };
    box.appendChild(d);
  }
}

function renderAll() {
  renderChats();
  const c = chat();
  $('sys').value = c.system;
  for (const [k, id] of [['temp', 'temp'], ['minp', 'minp'], ['topp', 'topp'], ['maxt', 'maxt'], ['think', 'think'], ['effort', 'effort'], ['model', 'model']]) $(id).value = db.settings[k];
  const log = $('log'); log.innerHTML = '';
  c.messages.forEach((m, i) => {
    const el = document.createElement('div'); el.dataset.id = m.id;
    log.appendChild(el); fillMsg(el, m);
    if (i < c.messages.length - 1) log.appendChild(insertBar(i + 1));
  });
  log.scrollTop = log.scrollHeight;
}

function insertBar(at) {
  const d = document.createElement('div'); d.className = 'insert';
  d.append(button('+ you', 'insert a message from you here', () => insertAt(at, 'user')), ' ', button('+ model', 'insert a model message here', () => insertAt(at, 'assistant')));
  return d;
}

function refreshMsg(m) {
  const el = $('log').querySelector(`[data-id="${m.id}"]`);
  if (el) fillMsg(el, m);
}

function fillMsg(el, m) {
  el.innerHTML = ''; el.className = 'msg ' + m.role + (editing === m.id ? ' editing' : '');
  const head = document.createElement('div'); head.className = 'head';
  const who = document.createElement('span'); who.textContent = m.role === 'user' ? 'you' : 'model'; head.appendChild(who);
  if (m.role === 'assistant' && m.versions.length > 1) {
    const vs = document.createElement('span'); vs.className = 'vers';
    vs.append(button('‹', 'previous version', () => { m.v = (m.v + m.versions.length - 1) % m.versions.length; save(); refreshMsg(m); }),
              `${m.v + 1}/${m.versions.length}`,
              button('›', 'next version', () => { m.v = (m.v + 1) % m.versions.length; save(); refreshMsg(m); }));
    head.appendChild(vs);
  }
  const sp = document.createElement('span'); sp.className = 'sp'; head.appendChild(sp);
  const acts = document.createElement('span'); acts.className = 'acts';
  if (editing !== m.id) {
    acts.appendChild(button('Edit', 'edit this message', () => { editing = m.id; refreshMsg(m); }));
    if (m.role === 'assistant') {
      acts.appendChild(button('Regenerate', 'a new version of this reply (the old one stays under ‹ ›)', () => regenerate(m)));
      if (cv(m).closed && cv(m).think) acts.appendChild(button('Re-answer', 'a new answer from the same thinking', () => reanswer(m)));
      acts.appendChild(button('Continue', cv(m).closed ? 'keep writing the answer' : 'keep thinking', () => continueMsg(m)));
    }
    acts.appendChild(button('Delete', 'delete this message', () => del(m, false), 'del'));
    acts.appendChild(button('Delete ↓', 'delete this message and everything after it', () => del(m, true), 'del'));
  }
  head.appendChild(acts); el.appendChild(head);

  if (editing === m.id) return fillEdit(el, m);

  if (m.role === 'user') {
    const b = document.createElement('div'); b.className = 'body'; b.innerHTML = fmt(m.text); el.appendChild(b);
    return;
  }
  const v = cv(m);
  if (v.think || !v.closed) {
    const d = document.createElement('details');
    d.open = m.id in thinkOpen ? thinkOpen[m.id] : (!v.closed && busy && busy.id === m.id);
    d.innerHTML = '<summary>thinking</summary><pre></pre>';
    d.querySelector('pre').textContent = v.think || (v.closed ? '' : '…');
    d.querySelector('summary').onclick = () => { thinkOpen[m.id] = !d.open; };   // only a click of yours counts, not the box opening itself while it streams
    el.appendChild(d);
  }
  const b = document.createElement('div'); b.className = 'body'; b.innerHTML = fmt(v.text); el.appendChild(b);
  if (v.cut) { const t = document.createElement('div'); t.className = 'tag'; t.textContent = 'cut off at the token limit — Continue picks it up'; el.appendChild(t); }
  if (v.err) { const t = document.createElement('div'); t.className = 'err'; t.textContent = v.err; el.appendChild(t); }
}

function area(value, rows) {
  const t = document.createElement('textarea'); t.value = value; t.rows = rows || 3;
  const fit = () => { t.style.height = 'auto'; t.style.height = Math.min(t.scrollHeight + 4, window.innerHeight * 0.55) + 'px'; };
  t.addEventListener('input', fit); setTimeout(fit, 0);
  return t;
}

function fillEdit(el, m) {
  const box = document.createElement('div'); box.className = 'edit';
  let tt = null;
  const lab = s => { const d = document.createElement('div'); d.className = 'lab'; d.textContent = s; return d; };
  if (m.role === 'assistant') { box.appendChild(lab('thinking')); tt = area(cv(m).think, 3); box.appendChild(tt); box.appendChild(lab('answer')); }
  const at = area(m.role === 'user' ? m.text : cv(m).text, 4); box.appendChild(at);
  const btns = document.createElement('div'); btns.className = 'btns';
  const commit = () => {
    snap();
    if (m.role === 'user') m.text = at.value;
    else { const v = cv(m); v.think = tt.value; v.text = at.value; v.cut = false; v.err = null; }
    editing = null; touch();
  };
  const b = (label, title, fn, cls) => { const x = button(label, title, fn); x.className = 'b sm' + (cls ? ' ' + cls : ''); btns.appendChild(x); };
  b('Save', 'keep the edit (Ctrl+Enter)', () => { commit(); refreshMsg(m); renderChats(); });
  if (m.role === 'user') b('Save & regenerate', 'drop everything after this message and write a new reply', () => { commit(); sendFrom(m); });
  else {
    b('Save & continue thinking', 'the model goes on from the end of the thinking; the answer is written again after it', () => {
      commit(); const v = cv(m); v.text = ''; v.closed = false; refreshMsg(m); run(m, 'think');
    });
    b('Save & continue answer', 'the model goes on from the end of the answer (empty: writes the answer from the thinking)', () => {
      commit(); const v = cv(m); v.closed = true; refreshMsg(m); run(m, 'answer');
    });
  }
  b('Cancel', 'drop the edit (Esc)', () => { editing = null; refreshMsg(m); }, 'alt');
  box.appendChild(btns);
  box.addEventListener('keydown', e => {
    if (e.key === 'Escape') { editing = null; refreshMsg(m); }
    else if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); commit(); refreshMsg(m); renderChats(); }
  });
  el.appendChild(box);
  setTimeout(() => (at.value ? at : tt || at).focus(), 0);
}

// ---------------------------------------------------------------------------------------------------- operations
function del(m, from) {
  const c = chat(), i = msgIndex(m.id);
  if (busy) return;
  snap();
  c.messages.splice(i, from ? c.messages.length : 1);
  if (editing && msgIndex(editing) < 0) editing = null;
  touch(); renderAll();
}
function insertAt(at, role) {
  if (busy) return;
  snap();
  const m = role === 'user' ? userMsg('') : modelMsg('', '', true);
  chat().messages.splice(at, 0, m); editing = m.id; touch(); renderAll();
  $('log').querySelector(`[data-id="${m.id}"]`).scrollIntoView({ block: 'center' });
}
function regenerate(m) {
  if (busy) return;
  m.versions.push({ think: '', text: '', closed: false }); m.v = m.versions.length - 1; run(m, 'fresh');
}
function reanswer(m) {
  if (busy) return;
  const t = cv(m).think; m.versions.push({ think: t, text: '', closed: true }); m.v = m.versions.length - 1; run(m, 'answer');
}
function continueMsg(m) { if (!busy) run(m, cv(m).closed ? 'answer' : 'think'); }

// A new reply after the message `m` (user): everything after it is dropped.
function sendFrom(m) {
  const c = chat(), i = msgIndex(m.id);
  c.messages.splice(i + 1);
  const a = modelMsg('', '', false); c.messages.push(a); touch(); renderAll(); run(a, 'fresh');
}
function send() {
  if (busy) return;
  const c = chat(), text = $('in').value.trim();
  const last = c.messages[c.messages.length - 1];
  if (!text) {   // empty box: answer a trailing message of yours, or continue a trailing reply
    if (!last) return;
    if (last.role === 'user') return sendFrom(last);
    return continueMsg(last);
  }
  $('in').value = '';
  const u = userMsg(text); c.messages.push(u);
  const a = modelMsg('', '', false); c.messages.push(a);
  touch(); renderAll(); run(a, 'fresh');
}

// ------------------------------------------------------------------------------------------------------ generating
async function run(m, mode) {
  const c = chat(), i = msgIndex(m.id), v = cv(m), s = db.settings;
  const thinking = s.think !== 'off';
  const base = { think: v.think, text: v.text };
  const prompt = Prompt.build({
    system: c.system, thinking, effort: s.effort, tail: mode === 'fresh' ? { mode } : { mode, think: v.think, text: v.text },
    messages: c.messages.slice(0, i).map(k => k.role === 'user' ? { role: 'user', text: k.text } : { role: 'assistant', think: cv(k).think, text: cv(k).text }),
  });
  const ctl = new AbortController(); ctl.id = m.id; busy = ctl;
  document.body.classList.add('busy'); $('send').hidden = true; $('stop').hidden = false;
  v.cut = false; v.err = null;
  let gen = '', n = 0, finish = null, tFirst = 0; const t0 = performance.now();
  refreshMsg(m);
  try {
    if (!s.model.trim() && !modelName) modelName = (await (await fetch('/v1/models')).json()).data[0].id;
    const r = await fetch('/v1/completions', {
      method: 'POST', signal: ctl.signal,
      body: JSON.stringify({ model: s.model.trim() || modelName, prompt, stream: true, max_tokens: s.maxt, temperature: s.temp, min_p: s.minp, top_p: s.topp, stop: ['<|im_end|>'] }),
    });
    if (!r.ok) throw new Error(r.status + ' ' + (await r.text()).slice(0, 300));
    const rd = r.body.getReader(), dec = new TextDecoder(); let buf = '';
    for (;;) {
      const { done, value } = await rd.read(); if (done) break;
      buf += dec.decode(value, { stream: true });
      let k;
      while ((k = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, k).trim(); buf = buf.slice(k + 1);
        if (!line.startsWith('data:') || line.slice(5).trim() === '[DONE]') continue;
        const ch = JSON.parse(line.slice(5)).choices?.[0]; if (!ch) continue;
        if (ch.text) { gen += ch.text; n++; if (n === 1) tFirst = performance.now(); }
        if (ch.finish_reason) finish = ch.finish_reason;
        Object.assign(v, Prompt.parse(mode, base, gen, thinking));
        const keep = nearBottom(); refreshMsg(m); if (keep) $('log').scrollTop = $('log').scrollHeight;
        if (n > 1) $('stat').textContent = ((n - 1) / ((performance.now() - tFirst) / 1000)).toFixed(1) + ' tokens/s, first token after ' + ((tFirst - t0) / 1000).toFixed(1) + ' s, ' + n + ' tokens';
        save();
      }
    }
    if (finish === 'length') v.cut = true;
  } catch (e) {
    if (e.name !== 'AbortError') v.err = String(e);
  } finally {
    busy = null; document.body.classList.remove('busy'); $('send').hidden = false; $('stop').hidden = true;
    if (mode === 'fresh' && thinking === false) v.closed = true;
    touch(); save(true); renderChats(); refreshMsg(m);
  }
}

// ------------------------------------------------------------------------------------------------------- controls
$('f').addEventListener('submit', e => { e.preventDefault(); send(); });
$('in').addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } });
$('stop').onclick = () => busy && busy.abort();
$('new').onclick = () => { if (!busy) newChat(); };
$('undo').onclick = () => {
  if (busy) return;
  const u = undo.pop(); if (!u) { $('stat').textContent = 'nothing to undo'; return; }
  const c = db.chats.find(k => k.id === u.id); if (!c) return;
  Object.assign(c, JSON.parse(u.json)); db.cur = c.id; editing = null; save(true); renderAll();
};
$('sys').oninput = () => { chat().system = $('sys').value; save(); };
for (const [id, k, num] of [['temp', 'temp', 1], ['minp', 'minp', 1], ['topp', 'topp', 1], ['maxt', 'maxt', 1], ['think', 'think', 0], ['effort', 'effort', 0], ['model', 'model', 0]])
  $(id).onchange = () => { db.settings[k] = num ? +$(id).value : $(id).value; save(true); };
$('export').onclick = () => {
  const c = chat(), a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(c, null, 1)], { type: 'application/json' }));
  a.download = ((c.title || 'chat').replace(/[^\w.-]+/g, '_').slice(0, 40) || 'chat') + '.json'; a.click(); URL.revokeObjectURL(a.href);
};
$('import').onclick = () => $('file').click();
$('file').onchange = async () => {
  const f = $('file').files[0]; $('file').value = ''; if (!f) return;
  try {
    const c = JSON.parse(await f.text());
    if (!Array.isArray(c.messages)) throw new Error('not a chat file');
    c.id = uid(); c.updated = Date.now(); c.system = c.system || ''; c.title = c.title || f.name.replace(/\.json$/, '');
    db.chats.unshift(c); db.cur = c.id; save(true); renderAll();
  } catch (e) { $('stat').textContent = 'import failed: ' + e; }
};

if (!db.chats.length) newChat(); else { if (!chat()) db.cur = db.chats[0].id; renderAll(); }
