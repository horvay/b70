#!/usr/bin/env python3
"""Quality checks for generated music (MiniMax Music 3 / xpu_music3 output), on the RTX 3080 (CUDA) and the CPU.

Run with WhisperX's venv (torch + CUDA + faster-whisper) and the cached demucs package:
  ~/b70/tools/music3/qc WAV... [--cache DIR] [--no-vocals] [--json]

Per track it writes <cache>/<stem>.json (metrics) and <stem>.npz (fingerprint) and skips tracks whose cache is newer
than the audio. Nothing here uses the B70.

Checks
  vocals    htdemucs (CUDA) splits out the vocal stem; `vocal_frac` is the share of 0.5 s frames where that stem is
            within 12 dB of the mix and above -45 dBFS. faster-whisper large-v3 (CUDA, int8_float16) transcribes the
            stem; `sung_words` counts words with probability >= 0.5 outside Whisper's stock hallucinations.
  clipping  samples at |x| >= 0.999 (xpu_music3 clamps to [-1, 1] when it writes the wav)
  silence   runs under -50 dBFS (RMS, 100 ms frames) inside the track; leading and trailing silence
  glitches  single 100 ms frames more than 18 dB above both neighbours and the local median (pops, bursts), and
            dropouts (a frame 30 dB under its neighbours while they are loud)
  static    stretches of 8 s or more where the spectrum barely changes (a stuck AR loop or a drone)
  ending    level of the last second against the track median (`end_drop_db`; small = abrupt)
  fingerprint  MFCC mean/std, chroma mean, a 1 frame/s chroma sequence, tempo and key estimates (npz), for the
            near-duplicate check in `similar`.

  ~/b70/tools/music3/qc similar NPZ... [--top 20]   rank the closest pairs (timbre x harmony sequence)
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

HALLUCINATIONS = {
    "thank you", "thanks for watching", "thank you for watching", "you", "bye", "subscribe", "please subscribe",
    "music", "applause", "silence", "amém", "amen",
}

HALLUCINATION_SUBSTRINGS = (
    "thank you for watching", "thanks for watching", "please subscribe", "subscribe to", "amara.org",
    "subtitles by", "transcribed by", "captioned by", "dimatorzok", "субтитры", "подпишитесь",
    "спасибо за просмотр", "продолжение следует", "ご視聴ありがとう", "チャンネル登録",
    "谢谢观看", "感谢观看", "请不吝点赞", "字幕", "시청해주셔서", "구독과 좋아요", "mbc 뉴스",
)


def _is_hallucination(text):
    low = text.strip().lower()
    clean = low.strip(".!?,。！？、~♪ \t\n")
    if not clean or clean in HALLUCINATIONS:
        return True
    return any(sub in low for sub in HALLUCINATION_SUBSTRINGS)


def load_audio(path, sr=44100):
    """float32 [2, N] at sr via ffmpeg (any format)."""
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "2", "-ar", str(sr), "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).T.copy()


def raw_peak(path):
    """Peak and clipped-sample count of the file as written (no resampling)."""
    x = load_audio(path, 44100)
    a = np.abs(x)
    return float(a.max()), int((a >= 0.999).sum())


def frame_db(mono, sr, hop_s=0.1):
    hop = int(sr * hop_s)
    n = len(mono) // hop
    f = mono[: n * hop].reshape(n, hop)
    rms = np.sqrt((f.astype(np.float64) ** 2).mean(1) + 1e-12)
    return 20 * np.log10(rms + 1e-9)


def runs(mask):
    """[(start, length)] of True runs."""
    out, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append((start, i - start))
            start = None
    if start is not None:
        out.append((start, len(mask) - start))
    return out


def audio_checks(x, sr):
    mono = x.mean(0)
    db = frame_db(mono, sr)  # 100 ms frames
    n = len(db)
    loud = db > -50
    lead = 0
    while lead < n and not loud[lead]:
        lead += 1
    trail = 0
    while trail < n - lead and not loud[n - 1 - trail]:
        trail += 1
    body = db[lead : n - trail] if n - trail > lead else db
    med = float(np.median(body)) if len(body) else -120.0
    internal_sil = [(s + lead, l) for s, l in runs(body < -50) if l >= 15]
    # pops / bursts: a frame far above both neighbours and the local median; dropouts: a frame far under loud neighbours
    pops, drops = [], []
    for i in range(2, len(body) - 2):
        nb = max(body[i - 1], body[i + 1])
        loc = np.median(body[max(0, i - 30) : i + 30])
        if body[i] - nb > 18 and body[i] - loc > 18:
            pops.append(round((i + lead) * 0.1, 1))
        if min(body[i - 1], body[i + 1]) > med - 10 and body[i] < min(body[i - 1], body[i + 1]) - 30:
            drops.append(round((i + lead) * 0.1, 1))
    # static: spectral change per 0.5 s frame
    hop = sr // 2
    nf = len(mono) // hop
    spec = np.abs(np.fft.rfft(mono[: nf * hop].reshape(nf, hop) * np.hanning(hop), axis=1))[:, : hop // 4]
    spec = np.log1p(spec * 100)
    spec /= np.linalg.norm(spec, axis=1, keepdims=True) + 1e-9
    change = 1 - (spec[1:] * spec[:-1]).sum(1)
    static = [(round(s * 0.5, 1), round(l * 0.5, 1)) for s, l in runs(change < 0.004) if l >= 16]
    end_db = float(np.mean(db[max(0, n - trail - 10) : n - trail])) if n - trail > 10 else -120.0
    return {
        "seconds": round(x.shape[1] / sr, 2),
        "lead_silence_s": round(lead * 0.1, 1),
        "trail_silence_s": round(trail * 0.1, 1),
        "median_db": round(med, 1),
        "internal_silence": [(round(s * 0.1, 1), round(l * 0.1, 1)) for s, l in internal_sil],
        "pops": pops[:20],
        "dropouts": drops[:20],
        "static": static,
        "end_drop_db": round(med - end_db, 1),
    }


# ---- noise (numpy only): hiss, crackle and other noise beds under the music ----

def noise_metrics(x, sr):
    """A noise bed shows in the quiet moments and in the top octave: hiss is a steady 8-16 kHz band that stays up when
    the music is quiet; crackle is isolated clicks above 4 kHz; rain, room tone and other beds keep the quiet frames'
    spectrum flat (noise-like) instead of tonal.
      hiss_db       8-16 kHz band level (dBFS) in the quietest 15 % of frames
      hiss_rel_db   the same against those frames' full-band level
      hiss_steady   std (dB) of the 8-16 kHz band over all frames (a steady bed: low)
      crackle_rate  clicks per second above 5 kHz (spikes 10x the local median that die within 1-3 ms)
      quiet_flat    median spectral flatness (200 Hz-16 kHz) of the quietest frames (noise: high)"""
    y = x.mean(0).astype(np.float64)
    n, hop = 2048, 1024
    nf = 1 + (len(y) - n) // hop
    idx = np.arange(n)[None, :] + hop * np.arange(nf)[:, None]
    S = np.abs(np.fft.rfft(y[idx] * np.hanning(n), axis=1)) ** 2 + 1e-20
    f = np.fft.rfftfreq(n, 1 / sr)
    full = S[:, (f > 30) & (f < 16000)].sum(1)
    band = S[:, (f > 8000) & (f < 16000)].sum(1)
    full_db = 10 * np.log10(full)
    live = full_db > full_db.max() - 60            # not silence
    q = live & (full_db <= np.percentile(full_db[live], 15))
    band_db = 10 * np.log10(band)
    mid = S[:, (f > 200) & (f < 16000)]
    flat = np.exp(np.mean(np.log(mid), axis=1)) / np.mean(mid, axis=1)
    # crackle: high-pass at 5 kHz; a click is a spike 10x the local median level (50 ms) that is gone 1-3 ms later
    # (vinyl crackle), unlike a hi-hat or a shaker, whose burst rings on
    Y = np.fft.rfft(y)
    Y[np.fft.rfftfreq(len(y), 1 / sr) < 5000] = 0
    hp = np.abs(np.fft.irfft(Y, len(y)))
    w = int(sr * 0.05)
    nb = len(hp) // w
    med = np.repeat(np.median(hp[: nb * w].reshape(nb, w), axis=1), w)
    med = np.concatenate([med, np.full(len(hp) - len(med), med[-1] if len(med) else 1e-9)]) + 1e-9
    cand = np.where(hp > 10 * med)[0]
    a1, a3 = int(sr * 0.001), int(sr * 0.003)
    clicks, last = 0, -sr
    for i in cand:
        if i - last < sr * 0.005 or i + a3 >= len(hp):
            continue
        if hp[i + a1 : i + a3].max() < 0.3 * hp[i]:
            clicks += 1
            last = i
    secs = len(y) / sr
    norm = 10 * np.log10(n * np.sum(np.hanning(n) ** 2) / 2)   # dB of a full-scale sine's frame, so levels are ~dBFS
    return {
        "hiss_db": round(float(np.median(band_db[q]) - norm), 1),
        "hiss_rel_db": round(float(np.median(band_db[q] - full_db[q])), 1),
        "hiss_steady": round(float(np.std(band_db[live])), 1),
        "crackle_rate": round(clicks / secs, 2),
        "quiet_flat": round(float(np.median(flat[q])), 4),
    }


# ---- fingerprint (numpy only) ----

def mel_fb(sr, n_fft, n_mels=40, fmin=40.0, fmax=11000.0):
    def hz2mel(h):
        return 2595 * np.log10(1 + h / 700.0)

    def mel2hz(m):
        return 700 * (10 ** (m / 2595.0) - 1)

    pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2))
    bins = np.fft.rfftfreq(n_fft, 1 / sr)
    fb = np.zeros((n_mels, len(bins)))
    for i in range(n_mels):
        l, c, r = pts[i], pts[i + 1], pts[i + 2]
        fb[i] = np.clip(np.minimum((bins - l) / (c - l), (r - bins) / (r - c)), 0, None)
    return fb


KRUMHANSL_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KRUMHANSL_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NOTES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


def fingerprint(x, sr):
    y = x.mean(0)
    # 22.05 kHz by decimation (the engine writes 44.1 kHz)
    if sr == 44100:
        y = (y[0::2][: len(y) // 2] + y[1::2][: len(y) // 2]) * 0.5
        sr = 22050
    n_fft, hop = 4096, 2048
    nf = 1 + (len(y) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(nf)[:, None]
    S = np.abs(np.fft.rfft(y[idx] * np.hanning(n_fft), axis=1)) ** 2  # [frames, bins]
    mel = np.log(S @ mel_fb(sr, n_fft).T + 1e-10)
    k = np.arange(20)[:, None]
    dct = np.cos(np.pi * k * (np.arange(40)[None, :] + 0.5) / 40)
    mfcc = mel @ dct.T
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    sel = (freqs > 55) & (freqs < 5000)
    pc = (np.round(12 * np.log2(freqs[sel] / 440.0)) + 9) % 12
    chroma = np.zeros((nf, 12))
    for p in range(12):
        chroma[:, p] = np.sqrt(S[:, sel][:, pc == p]).sum(1)
    chroma /= chroma.sum(1, keepdims=True) + 1e-9
    # 1 frame/s sequence (10.77 frames/s -> average groups of 11)
    g = 11
    seq = chroma[: (nf // g) * g].reshape(-1, g, 12).mean(1)
    # tempo from the onset envelope (mel flux), autocorrelation over 50-115 BPM (lofi: avoids the double-time octave)
    flux = np.clip(np.diff(mel, axis=0), 0, None).sum(1)
    flux = flux - flux.mean()
    ac = np.correlate(flux, flux, "full")[len(flux) - 1 :]
    fps = sr / hop
    lags = np.arange(len(ac))
    bpm_of = lambda lag: 60 * fps / lag  # noqa: E731
    ok = (lags > 0) & (bpm_of(np.maximum(lags, 1)) >= 50) & (bpm_of(np.maximum(lags, 1)) <= 115)
    best = lags[ok][np.argmax(ac[ok])] if ok.any() else 1
    cm = chroma.mean(0)
    keys = []
    for r in range(12):
        keys.append((np.corrcoef(cm, np.roll(KRUMHANSL_MAJ, r))[0, 1], NOTES[r] + " major"))
        keys.append((np.corrcoef(cm, np.roll(KRUMHANSL_MIN, r))[0, 1], NOTES[r] + " minor"))
    key = max(keys)[1]
    return {
        "mfcc_mean": mfcc.mean(0),
        "mfcc_std": mfcc.std(0),
        "chroma_mean": cm,
        "chroma_seq": seq,
        "tempo": np.array(round(float(bpm_of(best)), 1)),
        "key": np.array(key),
    }


# ---- vocals (CUDA) ----

_demucs = None
_whisper = None


def vocal_stem(x, sr):
    global _demucs
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    if _demucs is None:
        _demucs = get_model("htdemucs").cuda().eval()
    assert sr == _demucs.samplerate
    t = torch.from_numpy(x).float()
    ref = t.mean(0)
    t = (t - ref.mean()) / (ref.std() + 1e-8)
    with torch.no_grad():
        out = apply_model(_demucs, t[None].cuda(), shifts=0, split=True, overlap=0.25, progress=False)[0]
    out = out * (ref.std() + 1e-8) + ref.mean()
    v = out[_demucs.sources.index("vocals")].cpu().numpy()
    torch.cuda.empty_cache()
    return v


def whisper_words(v, sr, voc_db=None, mix_db=None, return_vad=False):
    global _whisper
    from faster_whisper import WhisperModel
    from faster_whisper.vad import get_speech_timestamps, VadOptions
    from scipy.signal import resample_poly

    if _whisper is None:
        _whisper = WhisperModel("large-v3", device="cuda", compute_type="int8_float16")
    mono = v.mean(0)
    # 44.1 kHz -> 16 kHz (polyphase)
    mono16 = resample_poly(mono, 160, 441).astype(np.float32)
    dur = len(mono16) / 16000.0
    peak = float(np.abs(mono16).max())
    if peak <= 0:
        return ([], []) if return_vad else []
    a_glob = mono16 / peak * 0.9

    raw_vad = get_speech_timestamps(a_glob, VadOptions(threshold=0.35, min_speech_duration_ms=250))
    raw_spans = [(d["start"] / 16000.0, d["end"] / 16000.0) for d in raw_vad]
    merged = []
    for s0, s1 in raw_spans:
        if merged and s0 - merged[-1][1] <= 1.5:
            merged[-1] = (merged[-1][0], max(merged[-1][1], s1))
        else:
            merged.append((s0, s1))

    kept_spans = []
    for s0, s1 in merged:
        if voc_db is not None and mix_db is not None:
            i0 = max(0, int((s0 - 0.5) * 2))
            i1 = min(len(voc_db), int((s1 + 0.5) * 2) + 1)
            if not any(voc_db[i] > -45 and voc_db[i] - mix_db[i] > -15 for i in range(i0, i1)):
                continue
        kept_spans.append((s0, s1))

    words = []
    for s0, s1 in kept_spans:
        t0 = max(0.0, s0 - 1.5)
        t1 = min(dur, s1 + 1.5)
        seg_a = mono16[int(t0 * 16000) : int(t1 * 16000)].copy()
        sp = float(np.abs(seg_a).max())
        if sp <= 0:
            continue
        seg_a = seg_a / sp * 0.9
        segs, _ = _whisper.transcribe(seg_a, language=None, vad_filter=False, word_timestamps=True, beam_size=5,
                                      condition_on_previous_text=False)
        for s in segs:
            if s.no_speech_prob > 0.6 or _is_hallucination(s.text):
                continue
            for w in s.words or []:
                if w.probability >= 0.5:
                    wt = round(float(max(s0, min(s1, t0 + w.start))), 1)
                    w_txt = w.word.strip()
                    if w_txt:
                        words.append((wt, w_txt, round(float(w.probability), 2)))

    vad_spans = [[round(float(a), 1), round(float(b), 1)] for a, b in kept_spans]
    return (words, vad_spans) if return_vad else words


def vocal_metrics(x, sr):
    v = vocal_stem(x, sr)
    mix_db = frame_db(x.mean(0), sr, 0.5)
    voc_db = frame_db(v.mean(0), sr, 0.5)
    active = (voc_db > -45) & (voc_db - mix_db > -12)
    # worst 10 s window
    w = 20
    win = np.convolve(active.astype(float), np.ones(w) / w, "valid") if len(active) >= w else active.astype(float)
    words, vad_spans = whisper_words(v, sr, voc_db=voc_db, mix_db=mix_db, return_vad=True)
    return {
        "vocal_frac": round(float(active.mean()), 3),
        "vocal_win_max": round(float(win.max()) if len(win) else 0.0, 3),
        "vocal_rel_db": round(float(np.median(voc_db - mix_db)), 1),
        "sung_words": len(words),
        "words": words[:40],
        "vocal_vad": vad_spans,
        "vocal_frames": "".join("1" if a else "0" for a in active),   # 0.5 s frames, the stem within 12 dB of the mix
    }


def analyze(path, cache, do_vocals=True):
    stem = os.path.splitext(os.path.basename(path))[0]
    jpath = os.path.join(cache, stem + ".json")
    if os.path.exists(jpath) and os.path.getmtime(jpath) >= os.path.getmtime(path):
        with open(jpath) as f:
            r = json.load(f)
        if not do_vocals or "vocal_frac" in r:
            return r
    sr = 44100
    x = load_audio(path, sr)
    peak, clipped = raw_peak(path)
    r = {"file": os.path.abspath(path), "peak": round(peak, 4), "clipped": clipped}
    r.update(audio_checks(x, sr))
    fp = fingerprint(x, sr)
    r.update(noise_metrics(x, sr))
    r["tempo_est"] = float(fp["tempo"])
    r["key_est"] = str(fp["key"])
    np.savez(os.path.join(cache, stem + ".npz"), **fp)
    if do_vocals:
        r.update(vocal_metrics(x, sr))
    with open(jpath + ".tmp", "w") as f:
        json.dump(r, f, indent=1)
    os.replace(jpath + ".tmp", jpath)
    return r


def similar(npzs, top=20, quiet=False):
    """Pairs ranked by similarity: timbre (z-scored MFCC stats cosine) x harmony (best chroma-sequence correlation)."""
    names = [os.path.splitext(os.path.basename(p))[0] for p in npzs]
    F = [np.load(p) for p in npzs]
    tim = np.stack([np.concatenate([f["mfcc_mean"], f["mfcc_std"]]) for f in F])
    tim = (tim - tim.mean(0)) / (tim.std(0) + 1e-9)
    tim /= np.linalg.norm(tim, axis=1, keepdims=True) + 1e-9
    T = tim @ tim.T
    seqs = []
    for f in F:
        s = f["chroma_seq"] - f["chroma_seq"].mean(0)
        s /= np.linalg.norm(s, axis=1, keepdims=True) + 1e-9
        seqs.append(s)
    pairs = []
    for i in range(len(F)):
        for j in range(i + 1, len(F)):
            a, b = seqs[i], seqs[j]
            # mean framewise cosine at the best lag within +-30 s (same key: a transposed copy is a different track)
            best = -1.0
            for lag in range(-30, 31, 2):
                if lag >= 0:
                    aa, bb = a[lag:], b[: len(a) - lag]
                else:
                    aa, bb = a[: len(a) + lag], b[-lag:]
                m = min(len(aa), len(bb))
                if m < 30:
                    continue
                best = max(best, float((aa[:m] * bb[:m]).sum(1).mean()))
            hc = float(np.dot(F[i]["chroma_mean"], F[j]["chroma_mean"]) /
                       (np.linalg.norm(F[i]["chroma_mean"]) * np.linalg.norm(F[j]["chroma_mean"]) + 1e-9))
            pairs.append((float(T[i, j]), best, hc, names[i], names[j]))
    pairs.sort(key=lambda p: -(p[0] + p[1]))
    if not quiet:
        for t, s, hc, a, b in pairs[:top]:
            print(f"{t:6.3f} timbre  {s:6.3f} seq  {hc:6.3f} chroma  {a}  {b}")
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--no-vocals", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args()
    if a.files[0] == "similar":
        similar(a.files[1:], a.top)
        return
    if a.files[0] == "noise":   # qc noise FILE...: the noise measures only, one JSON line per file
        for p in a.files[1:]:
            print(json.dumps({"file": p, **noise_metrics(load_audio(p, 44100), 44100)}), flush=True)
        return
    for p in a.files:
        cache = a.cache or os.path.join(os.path.dirname(os.path.abspath(p)), ".qc")
        os.makedirs(cache, exist_ok=True)
        r = analyze(p, cache, not a.no_vocals)
        if a.json:
            print(json.dumps(r))
        else:
            print(f"{os.path.basename(p)}: {r['seconds']} s, peak {r['peak']}, clipped {r['clipped']}, "
                  f"vocal_frac {r.get('vocal_frac')}, win {r.get('vocal_win_max')}, rel {r.get('vocal_rel_db')} dB, "
                  f"words {r.get('sung_words')} {[w[1] for w in r.get('words', [])][:12]}, "
                  f"sil {r['internal_silence']}, pops {len(r['pops'])}, drops {len(r['dropouts'])}, static {r['static']}, "
                  f"end_drop {r['end_drop_db']} dB, tempo {r['tempo_est']}, key {r['key_est']}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
