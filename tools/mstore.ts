#!/usr/bin/env bun
// mstore: move model files/dirs into ~/models and leave a symlink at the old path.
//
// Every swap is atomic, so a job that opens the old path at any moment sees either the
// real file or the symlink, never nothing:
//   file: hardlink src -> dest, then rename a temp symlink over src
//   dir:  cp -al src -> dest (hardlink tree), then `mv --exchange` a temp symlink with src
// Open file handles keep working (same inode). Every move is journaled; `undo` reverses it.
//
//   mstore move <src> <dest>      move one file or dir (dest is the full new path)
//   mstore plan <manifest.tsv>    dry run: src<TAB>dest per line
//   mstore apply <manifest.tsv>   run the manifest
//   mstore undo <src>             put the real file/dir back at src
//   mstore check                  verify every journaled symlink still resolves
//   mstore index                  write ~/models/INDEX.md (every model, size, old paths)

import * as fs from "node:fs";
import * as path from "node:path";

const HOME = process.env.HOME!;
const STORE = path.join(HOME, "models");
const JOURNAL = path.join(STORE, "_tools", "moves.tsv");

const expand = (p: string) => path.resolve(p.replace(/^~(?=\/|$)/, HOME));
const die = (msg: string): never => { console.error(`mstore: ${msg}`); process.exit(1); };

function sh(cmd: string[]) {
  const r = Bun.spawnSync(cmd, { stderr: "pipe" });
  if (r.exitCode !== 0) die(`${cmd.join(" ")}: ${r.stderr.toString().trim()}`);
}

function journal(op: string, src: string, dest: string, kind: string, bytes: number) {
  fs.mkdirSync(path.dirname(JOURNAL), { recursive: true });
  fs.appendFileSync(JOURNAL, [new Date().toISOString(), op, kind, bytes, src, dest].join("\t") + "\n");
}

function treeBytes(p: string): number {
  const st = fs.lstatSync(p);
  if (!st.isDirectory()) return st.isFile() ? st.size : 0;
  return fs.readdirSync(p).reduce((n, e) => n + treeBytes(path.join(p, e)), 0);
}

// Files in a tree with only one link: created after `cp -al`, so they'd be lost on delete.
function loneFiles(p: string): string[] {
  const st = fs.lstatSync(p);
  if (st.isDirectory()) return fs.readdirSync(p).flatMap((e) => loneFiles(path.join(p, e)));
  return st.isFile() && st.nlink < 2 ? [p] : [];
}

const under = (p: string, root: string) => p === root || p.startsWith(root + "/");

// Processes using src: cwd/root/open fds/mmaps under it, or a sandbox (bwrap) bind mount of
// src or something inside it (the sandbox keeps the old folder, which we then empty). A mount
// of a folder above src only matters if the sandbox can't see the store: then the new symlink
// dangles inside it (Steam's runtime mounts all of ~/.cache but also sees ~/models).
function users(src: string): string[] {
  const found: string[] = [];
  for (const pid of fs.readdirSync("/proc").filter((d) => /^\d+$/.test(d))) {
    const p = `/proc/${pid}`;
    const why: string[] = [];
    try {
      for (const l of ["cwd", "root"]) if (under(fs.readlinkSync(`${p}/${l}`), src)) why.push(l);
      for (const fd of fs.readdirSync(`${p}/fd`)) {
        try { if (under(fs.readlinkSync(`${p}/fd/${fd}`), src)) { why.push("open"); break; } } catch {}
      }
      if (fs.readFileSync(`${p}/maps`, "utf8").split("\n").some((l) => under(l.slice(l.indexOf("/")), src))) why.push("mmap");
      for (const l of fs.readFileSync(`${p}/mountinfo`, "utf8").split("\n")) {
        const root = l.split(" ")[3] ?? "";
        const i = root.indexOf(HOME);
        if (i < 0) continue;
        const r = root.slice(i);
        if (r.length <= HOME.length) continue;
        if (under(r, src) || (under(src, r) && !fs.existsSync(`${p}/root${STORE}/_tools`))) { why.push(`mount ${r}`); break; }
      }
    } catch { continue; }
    if (why.length) {
      const cmd = fs.readFileSync(`${p}/cmdline`, "utf8").replace(/\0/g, " ").slice(0, 80);
      found.push(`${pid} (${[...new Set(why)].join(", ")}): ${cmd}`);
    }
  }
  return found;
}

function check(src: string, dest: string) {
  if (!fs.existsSync(src)) die(`${src} does not exist`);
  const busy = users(src);
  if (busy.length) die(`${src} is in use; try again when idle:\n  ${busy.join("\n  ")}`);
  const st = fs.lstatSync(src);
  if (st.isSymbolicLink()) die(`${src} is already a symlink (-> ${fs.readlinkSync(src)})`);
  if (fs.existsSync(dest) || fs.lstatSync(dest, { throwIfNoEntry: false })) die(`${dest} already exists`);
  if (!dest.startsWith(STORE + "/")) die(`${dest} is outside ${STORE}`);
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  if (fs.statSync(path.dirname(dest)).dev !== st.dev) die(`${src} is on another filesystem; catalog it instead`);
  return st;
}

function move(srcArg: string, destArg: string) {
  const src = expand(srcArg), dest = expand(destArg);
  const st = check(src, dest);
  const tmp = path.join(path.dirname(src), `.mstore-${path.basename(src)}.${process.pid}`);
  const bytes = treeBytes(src);

  if (st.isFile()) {
    fs.linkSync(src, dest);
    fs.symlinkSync(dest, tmp);
    fs.renameSync(tmp, src);
    if (fs.statSync(src).ino !== st.ino) die(`verify failed for ${src}`);
    journal("move", src, dest, "file", bytes);
  } else if (st.isDirectory()) {
    sh(["cp", "-al", "--", src, dest]);
    fs.symlinkSync(dest, tmp);
    sh(["mv", "--exchange", "--no-target-directory", "--", tmp, src]);
    // tmp is now the old real dir; its files are hardlinked into dest.
    const lone = loneFiles(tmp);
    if (lone.length) {
      console.error(`mstore: ${lone.length} file(s) appeared during the move; kept ${tmp}:\n  ${lone.join("\n  ")}`);
    } else {
      fs.rmSync(tmp, { recursive: true });
    }
    journal("move", src, dest, "dir", bytes);
  } else die(`${src} is not a file or dir`);
  console.log(`moved ${(bytes / 1e9).toFixed(2)} GB  ${src} -> ${dest}`);
}

function undo(srcArg: string) {
  const src = expand(srcArg);
  if (!fs.lstatSync(src).isSymbolicLink()) die(`${src} is not a symlink`);
  const dest = fs.readlinkSync(src);
  const st = fs.lstatSync(dest);
  if (st.isFile()) {
    fs.renameSync(dest, src); // atomically replaces the symlink with the real file
  } else {
    const tmp = path.join(path.dirname(src), `.mstore-undo-${path.basename(src)}.${process.pid}`);
    sh(["cp", "-al", "--", dest, tmp]);
    sh(["mv", "--exchange", "--no-target-directory", "--", tmp, src]);
    fs.unlinkSync(tmp); // tmp is now the symlink
    fs.rmSync(dest, { recursive: true });
  }
  journal("undo", src, dest, st.isFile() ? "file" : "dir", treeBytes(src));
  console.log(`restored ${src}`);
}

function readManifest(file: string): [string, string][] {
  return fs.readFileSync(expand(file), "utf8").split("\n")
    .map((l) => l.replace(/#.*/, "").trim()).filter(Boolean)
    .map((l) => { const [s, d] = l.split("\t"); if (!d) die(`bad manifest line: ${l}`); return [s, d]; });
}

function plan(file: string) {
  let total = 0, bad = 0;
  for (const [s, d] of readManifest(file)) {
    const src = expand(s), dest = expand(d);
    const st = fs.lstatSync(src, { throwIfNoEntry: false });
    const why = !st ? "missing" : st.isSymbolicLink() ? "already a symlink"
      : fs.lstatSync(dest, { throwIfNoEntry: false }) ? "dest exists"
      : users(src).length ? "in use" : "";
    const bytes = st && !why ? treeBytes(src) : 0;
    total += bytes; if (why) bad++;
    console.log(`${why ? "SKIP " + why.padEnd(18) : (bytes / 1e9).toFixed(2).padStart(7) + " GB" + " ".repeat(10)}  ${s} -> ${d}`);
  }
  console.log(`${(total / 1e9).toFixed(1)} GB to move, ${bad} skipped`);
}

function verify() {
  if (!fs.existsSync(JOURNAL)) return console.log("no moves yet");
  const live = new Map<string, string>();
  for (const l of fs.readFileSync(JOURNAL, "utf8").trim().split("\n")) {
    const [, op, , , src, dest] = l.split("\t");
    op === "move" ? live.set(src, dest) : live.delete(src);
  }
  let bad = 0;
  for (const [src, dest] of live) {
    const ok = fs.lstatSync(src, { throwIfNoEntry: false })?.isSymbolicLink() && fs.existsSync(src);
    if (!ok) { bad++; console.log(`BROKEN  ${src} -> ${dest}`); }
  }
  console.log(`${live.size} moved paths, ${bad} broken`);
  if (bad) process.exit(1);
}

// INDEX.md: every model folder with its size, big files, and the old paths that symlink to it.
const CATEGORIES = ["video", "image", "music", "voice", "llm"];
function index() {
  const old = new Map<string, string>();
  if (fs.existsSync(JOURNAL)) for (const l of fs.readFileSync(JOURNAL, "utf8").trim().split("\n")) {
    const [, op, , , src, dest] = l.split("\t");
    op === "move" ? old.set(dest, src) : old.delete(dest);
  }
  const gb = (n: number) => `${(n / 1e9).toFixed(1)} GB`;
  const tilde = (p: string) => p.replace(HOME, "~");
  // HF repos: skip blobs/ and list the snapshot links instead, so nothing is counted twice.
  let broken: string[] = [];
  const big = (p: string): string[] => {
    const st = fs.statSync(p, { throwIfNoEntry: false });
    if (!st) { broken.push(`- broken link: ${path.basename(p)} -> ${tilde(fs.readlinkSync(p))}`); return []; }
    if (st.isDirectory()) return fs.readdirSync(p).filter((e) => e !== ".cache" && e !== "blobs").sort().flatMap((e) => big(path.join(p, e)));
    return st.size > 50e6 ? [p] : [];
  };
  const entries: [string, string][] = [];
  for (const top of fs.readdirSync(STORE).sort()) {
    if (top.startsWith("_") || top.startsWith(".") || top === "logs" || top.endsWith(".md")) continue;
    const p = path.join(STORE, top);
    if (CATEGORIES.includes(top)) for (const m of fs.readdirSync(p).sort()) entries.push([top, path.join(p, m)]);
    else entries.push(["llm", p]);
  }
  entries.sort((x, y) => x[0].localeCompare(y[0]));
  let out = `# ~/models\n\nB70 models, one folder per model. Old paths are symlinks to these, so existing scripts keep working; use these paths in new scripts.\nGenerated by \`mstore index\` (~/b70/tools/mstore.ts) on ${new Date().toISOString().slice(0, 10)}. Moves are logged in \`_tools/moves.tsv\`.\n`;
  let cat = "";
  for (const [c, p] of entries) {
    if (c !== cat) { out += `\n## ${c}\n`; cat = c; }
    broken = [];
    const files = big(p);
    const size = files.reduce((n, f) => n + fs.statSync(f).size, 0);
    out += `\n### ${path.basename(p)} — ${gb(size)}\n\`${tilde(p)}\`\n`;
    for (const f of files) {
      const real = fs.realpathSync(f);
      const ext = real.startsWith(STORE + "/") ? "" : `  (on ${tilde(path.dirname(real))})`;
      out += `- ${path.relative(p, f)} — ${gb(fs.statSync(f).size)}${ext}\n`;
    }
    const links = [...old].filter(([d]) => under(d, p)).map(([d, s]) => `\`${tilde(s)}\`${d === p ? "" : ` → ${path.relative(p, d)}`}`);
    if (links.length) out += `- old paths: ${links.join(", ")}\n`;
    if (broken.length) out += broken.join("\n") + "\n";
  }
  fs.writeFileSync(path.join(STORE, "INDEX.md"), out);
  console.log(`wrote ${path.join(STORE, "INDEX.md")} (${entries.length} models)`);
}

const [cmd, a, b] = process.argv.slice(2);
switch (cmd) {
  case "move": a && b ? move(a, b) : die("usage: mstore move <src> <dest>"); break;
  case "plan": a ? plan(a) : die("usage: mstore plan <manifest.tsv>"); break;
  case "apply": a ? readManifest(a).forEach(([s, d]) => move(s, d)) : die("usage: mstore apply <manifest.tsv>"); break;
  case "undo": a ? undo(a) : die("usage: mstore undo <src>"); break;
  case "check": verify(); break;
  case "index": index(); break;
  default: die("commands: move plan apply undo check index");
}
