#!/usr/bin/env python3
"""music3 batch: many MiniMax Music 3 songs through xpu_music3 on the B70, as short chunks of the b70q queue.

  ~/b70/tools/music3/batch JOBS.jsonl --out DIR [--group 16] [--one-chunk]

JOBS.jsonl, one song per line:
  {"id": "cafe_017", "caption": "...", "lyrics": "[intro]\\n\\n[outro]", "seed": 7, "seconds": 180,
   "negative": "...", "min_seconds": 80, "steps": 30, "cfg": 1.7, "ar_cfg": 1.5, "top_k": 50}
   (negative .. top_k optional; negative: a caption for the AR stage's CFG row, see
   ~/b70/engines/sd.cpp/xpu_sdpa/MUSIC3_PROMPTING.md)

Songs go to `xpu_music3 --jobs` in groups of --group (default 16): the AR stage runs the whole group at once (one read
of the 8B LM's weights per frame for all of them; int8 KV cache, hidden frames kept on the host), then the DiT (all
windows and both CFG branches of a step batched) and the DAV one song after another. A group is filled with songs of
the same length ceiling first, so they tend to end together (the AR stage runs until the longest one ends).

Writes DIR/<id>.wav and DIR/<id>.log (the job, then the group's engine log). A song is skipped when DIR/<id>.wav,
.skip or .done exists (write .skip to drop a job, .done when its wav has been taken away, e.g. mastered elsewhere). The
jobs file is read again at the start of every chunk, so lines appended meanwhile (re-rolls, new songs) are picked up
by the next chunk; lines are done in file order, re-rolls ("take" > 0) first.

--one-chunk: one group, then exit 99 if songs are left and 0 when all are done: the shape `b70q submit --chunked`
wants, so each chunk goes back in line behind everyone else. Without it, every group in one go.

Must run under the queue (b70q submit / b70-lease run): it uses the B70.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

BIN = os.path.expanduser("~/b70/engines/sd.cpp/xpu_sdpa/xpu_music3")
FIELDS = ("caption", "lyrics", "negative", "seconds", "min_seconds", "seed", "steps", "cfg", "ar_cfg", "top_k", "dit_variants", "ban_codes")


def load_jobs(path):
    jobs = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if ln and not ln.startswith("#"):
                jobs.append(json.loads(ln))
    return jobs


def pending(jobs, out):
    return [j for j in jobs if not any(os.path.exists(os.path.join(out, j["id"] + ext)) for ext in (".wav", ".skip", ".done"))]


def run_group(group, out):
    """One engine process for the group; returns the engine's exit code."""
    with tempfile.TemporaryDirectory(prefix="music3_") as td:
        jf = os.path.join(td, "jobs.jsonl")
        with open(jf, "w") as f:
            for j in group:
                e = {k: j[k] for k in FIELDS if k in j}
                e["out"] = os.path.join(td, j["id"] + ".wav")   # moved into DIR when complete
                f.write(json.dumps(e) + "\n")
        log = os.path.join(td, "engine.log")
        with open(log, "w") as lf:
            rc = subprocess.run([BIN, "--jobs", jf, "--batch", str(len(group))], stdout=lf, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL).returncode
        with open(log) as lf:
            text = "".join(ln for ln in lf if "DiT step" not in ln)
        for j in group:
            with open(os.path.join(out, j["id"] + ".log"), "w") as f:
                f.write(json.dumps(j) + "\n" + text)
            src = os.path.join(td, j["id"] + ".wav")
            if os.path.exists(src):
                # across file systems a move is a copy: land it as .part, then rename, so readers never see half a wav
                dst = os.path.join(out, j["id"] + ".wav")
                shutil.move(src, dst + ".part")
                os.replace(dst + ".part", dst)
            c0 = os.path.join(td, j["id"] + ".c0.i32")   # the semantic codes per frame -> DIR/codes/
            if os.path.exists(c0):
                os.makedirs(os.path.join(out, "codes"), exist_ok=True)
                shutil.move(c0, os.path.join(out, "codes", j["id"] + ".c0.i32"))
            for f in os.listdir(td):   # "dit_variants" test renders: <id>_s20.wav ... -> DIR/variants/
                if f.startswith(j["id"] + "_") and f.endswith(".wav"):
                    os.makedirs(os.path.join(out, "variants"), exist_ok=True)
                    shutil.move(os.path.join(td, f), os.path.join(out, "variants", f))
        return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--group", type=int, default=16)
    ap.add_argument("--one-chunk", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    failed = set()
    while True:
        todo = [j for j in pending(load_jobs(a.jobs), a.out) if j["id"] not in failed]
        if not todo:
            print("[music3-batch] all done", flush=True)
            return 0
        todo.sort(key=lambda j: 0 if j.get("take", 0) > 0 else 1)   # re-rolls first (stable: file order otherwise)
        # the first pending song, then others with its length ceiling, then the rest in order
        same = [j for j in todo if j.get("seconds") == todo[0].get("seconds")]
        group = (same + [j for j in todo if j not in same])[: a.group]
        t0 = time.time()
        rc = run_group(group, a.out)
        dt = time.time() - t0
        got = 0.0
        for j in group:
            wav = os.path.join(a.out, j["id"] + ".wav")
            if os.path.exists(wav):
                secs = (os.path.getsize(wav) - 44) / (44100 * 4)
                got += secs
                print(f"[music3-batch] {j['id']}: {secs:.1f} s", flush=True)
            else:
                failed.add(j["id"])
                print(f"[music3-batch] {j['id']} FAILED (engine rc={rc}, see {j['id']}.log)", flush=True)
        print(f"[music3-batch] group of {len(group)}: {got:.0f} s of audio in {dt:.0f} s ({dt / max(got, 1):.2f} s/s)", flush=True)
        if rc in (-9, 137):  # killed
            return 1
        if a.one_chunk:
            left = [j for j in pending(load_jobs(a.jobs), a.out) if j["id"] not in failed]
            return 99 if left else 0


if __name__ == "__main__":
    sys.exit(main())
