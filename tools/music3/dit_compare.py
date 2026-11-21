#!/usr/bin/env python3
"""Does the DiT need all 30 steps? Compares renders of the same AR output (xpu_music3 "dit_variants").

  python3 dit_compare.py BASE.wav VARIANT.wav...      (e.g. raw/x.wav raw/variants/x_s20.wav x_s12.wav x_seed_s30.wav)

For each variant: log-mel distance to the 30-step render (mean absolute difference in dB over 40 mel bands and 46 ms
frames, after aligning levels), and what fewer steps tend to do to a flow model's output: less high end (the share of
energy above 6 kHz), more noise (spectral flatness), less transient detail (spectral flux). The "_seed" render (same
steps, another noise seed) is the yardstick: a variant closer to the base than that is within the model's own spread.
"""
import subprocess
import sys

import numpy as np


def load(path, sr=22050):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(sr), "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def feats(y, sr=22050):
    n, hop = 2048, 1024
    nf = 1 + (len(y) - n) // hop
    idx = np.arange(n)[None, :] + hop * np.arange(nf)[:, None]
    S = np.abs(np.fft.rfft(y[idx] * np.hanning(n), axis=1)) ** 2
    f = np.fft.rfftfreq(n, 1 / sr)
    mel = lambda h: 2595 * np.log10(1 + h / 700)  # noqa: E731
    pts = 700 * (10 ** (np.linspace(mel(40), mel(10500), 42) / 2595) - 1)
    fb = np.zeros((40, len(f)))
    for i in range(40):
        l, c, r = pts[i], pts[i + 1], pts[i + 2]
        fb[i] = np.clip(np.minimum((f - l) / (c - l), (r - f) / (r - c)), 0, None)
    M = 10 * np.log10(S @ fb.T + 1e-10)
    hf = S[:, f > 6000].sum() / S.sum()
    flat = np.exp(np.mean(np.log(S[:, 5:] + 1e-12), axis=1)) / (np.mean(S[:, 5:], axis=1) + 1e-12)
    flux = np.mean(np.clip(np.diff(M, axis=0), 0, None))
    return M, hf, float(np.median(flat)), float(flux)


def main():
    base, *vars_ = sys.argv[1:]
    yb = load(base)
    Mb, hfb, flb, fxb = feats(yb)
    print(f"{'render':40s} {'mel dist dB':>11s} {'HF>6k %':>8s} {'flatness':>9s} {'flux':>6s}")
    print(f"{base.split('/')[-1]:40s} {'-':>11s} {100 * hfb:8.2f} {flb:9.4f} {fxb:6.3f}")
    for v in vars_:
        y = load(v)
        M, hf, fl, fx = feats(y)
        m = min(len(M), len(Mb))
        a, b = M[:m], Mb[:m]
        a = a - (a.mean() - b.mean())
        d = float(np.mean(np.abs(a - b)))
        print(f"{v.split('/')[-1]:40s} {d:11.2f} {100 * hf:8.2f} {fl:9.4f} {fx:6.3f}")


if __name__ == "__main__":
    main()
