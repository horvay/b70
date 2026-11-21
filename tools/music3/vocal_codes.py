#!/usr/bin/env python3
"""Which of Music 3's semantic codes (c0, 16,384 of them) come with a voice?

  python3 vocal_codes.py CODES_DIR QC_DIR [--out ban.json] [--min-count 6] [--min-ratio 0.75]

CODES_DIR/<id>.c0.i32: the c0 code of every frame of a take (25 frames/s; xpu_music3 --jobs writes them, the batch
tool collects them). QC_DIR/<id>.json: qc.py's metrics with "vocal_frames" (0.5 s frames where the demucs vocal stem
was within 12 dB of the mix). Each frame is labelled voiced or not; a code that shows up mostly in voiced frames, over
enough frames and in more than one take, goes on the list. xpu_music3 can mask such codes out of sampling
("ban_codes" in a --jobs line).
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("codes")
    ap.add_argument("qc")
    ap.add_argument("--out")
    ap.add_argument("--min-count", type=int, default=6)
    ap.add_argument("--min-ratio", type=float, default=0.75)
    a = ap.parse_args()
    V = np.zeros(16384)
    T = np.zeros(16384)
    takes = {}
    nv = nt = 0
    for f in sorted(os.listdir(a.codes)):
        if not f.endswith(".c0.i32"):
            continue
        tid = f[: -len(".c0.i32")]
        q = os.path.join(a.qc, tid + ".json")
        if not os.path.exists(q):
            continue
        with open(q) as fh:
            m = json.load(fh)
        vf = m.get("vocal_frames")
        if vf is None:
            continue
        c = np.fromfile(os.path.join(a.codes, f), dtype=np.int32)
        # AR frame k plays at k / 25 s (the hidden frames start at the second AR step; within a frame of the audio)
        idx = np.minimum((np.arange(len(c)) / 25.0 / 0.5).astype(int), len(vf) - 1)
        lab = np.array([vf[i] == "1" for i in idx])
        np.add.at(T, c, 1)
        np.add.at(V, c[lab], 1)
        for code in set(c[lab].tolist()):
            takes.setdefault(code, set()).add(tid)
        nv += int(lab.sum())
        nt += len(c)
    base = nv / max(nt, 1)
    ratio = (V + 0.5) / (T + 1.0)
    cand = [int(k) for k in np.where((V >= a.min_count) & (ratio >= a.min_ratio))[0] if len(takes.get(int(k), ())) >= 2]
    cover = V[cand].sum() / max(nv, 1)
    cost = (T[cand] - V[cand]).sum() / max(nt - nv, 1)
    print(f"{nt} frames, {nv} voiced ({100 * base:.1f} %), {int((T > 0).sum())} codes used")
    print(f"{len(cand)} codes voiced >= {a.min_ratio:.0%} of the time (>= {a.min_count} voiced frames, 2+ takes): "
          f"they cover {100 * cover:.1f} % of voiced frames and {100 * cost:.2f} % of unvoiced ones")
    for k in sorted(cand, key=lambda k: -V[k])[:25]:
        print(f"  code {k:5d}: {int(V[k])}/{int(T[k])} voiced, {len(takes[k])} takes")
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"codes": cand, "voiced_frames": nv, "frames": nt, "cover": cover, "cost": cost}, f)


if __name__ == "__main__":
    main()
