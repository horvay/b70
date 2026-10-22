"""Quality check for a folder of generated lines (CPU only, never touches a GPU).

usage: CUDA_VISIBLE_DEVICES="" ~/.local/share/pipx/venvs/whisperx/bin/python score.py <jobs.tsv> <wav_dir> <tag> \
           [--voices-dir DIR (default ./voices)] [--report scores.tsv]

For every row of the jobs TSV (out_name<TAB>voice<TAB>text) with a wav in <wav_dir>:
  WhisperX large-v3 (CPU int8) word error rate against the text with [tags] and <tags> removed,
  WeSpeaker ResNet34 cosine similarity to <voices-dir>/<voice>.wav,
  duration, lead/trail silence and the longest inner pause.
Appends one row per clip to the report (default scores.tsv next to this script) and prints the set means.
"""
import os, re, sys, wave, warnings
os.environ["CUDA_VISIBLE_DEVICES"] = ""
warnings.filterwarnings("ignore")
import numpy as np, torch, whisperx
from pyannote.audio import Model, Inference

HERE = os.path.dirname(os.path.abspath(__file__))
args = sys.argv[1:]
def opt(name, default):
    if name in args:
        i = args.index(name); v = args[i + 1]; del args[i:i + 2]; return v
    return default
voices_dir = os.path.expanduser(opt("--voices-dir", "voices"))
report = opt("--report", os.path.join(HERE, "scores.tsv"))
jobs, wav_dir, tag = args[:3]


def words(t):
    return re.sub(r"[^a-z0-9' ]", " ", t.lower().replace("’", "'")).split()


def wer(r, h):
    r, h = words(r), words(h); d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
    return d[len(h)] / max(1, len(r))


def signal(path):
    w = wave.open(path)
    sr, n, sw, ch = w.getframerate(), w.getnframes(), w.getsampwidth(), w.getnchannels()
    x = np.frombuffer(w.readframes(n), dtype=np.int16 if sw == 2 else np.int32).astype(np.float32)
    x = x.reshape(-1, ch).mean(1) / (32768.0 if sw == 2 else 2**31)
    hop = int(sr * 0.02)
    fr = x[: len(x) // hop * hop].reshape(-1, hop)
    db = 20 * np.log10(np.sqrt((fr ** 2).mean(1)) + 1e-9)
    voiced = db > max(db.max() - 40, -60)
    idx = np.flatnonzero(voiced)
    lead = idx[0] * 0.02 if len(idx) else len(x) / sr
    trail = (len(db) - 1 - idx[-1]) * 0.02 if len(idx) else 0
    gap = best = 0
    if len(idx):
        for v in voiced[idx[0]: idx[-1] + 1]:
            gap = 0 if v else gap + 1; best = max(best, gap)
    return len(x) / sr, lead, trail, best * 0.02


torch.set_num_threads(int(os.environ.get("SCORE_THREADS", "16")))
asr = whisperx.load_model("large-v3", "cpu", compute_type="int8", language="en")
spk = Inference(Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM"), window="whole", device=torch.device("cpu"))
emb = lambda p: spk({"waveform": torch.from_numpy(whisperx.load_audio(p))[None], "sample_rate": 16000})
cos = lambda a, b: float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
ref_emb = {}
rows = []
new = not os.path.exists(report)
with open(report, "a") as f:
    if new:
        f.write("tag\tclip\tvoice\tdur_s\tlead_s\ttrail_s\tmax_pause_s\twer\tspeaker_sim\ttext\ttranscript\n")
    for ln in open(jobs, encoding="utf-8"):
        if not ln.strip() or ln.startswith("#"):
            continue
        name, voice, text = ln.rstrip("\n").split("\t")[:3]
        path = os.path.join(wav_dir, name + ".wav")
        if not os.path.exists(path):
            print("missing", path); continue
        if voice not in ref_emb:
            ref_emb[voice] = emb(os.path.join(voices_dir, voice + ".wav"))
        plain = re.sub(r"<[^>]*>", " ", re.sub(r"\[[^\]]*\]", " ", text))
        hyp = " ".join(s["text"].strip() for s in asr.transcribe(whisperx.load_audio(path), batch_size=4)["segments"])
        dur, lead, trail, pause = signal(path)
        r = (tag, name, voice, dur, lead, trail, pause, wer(plain, hyp), cos(ref_emb[voice], emb(path)), text, hyp)
        rows.append(r)
        f.write("%s\t%s\t%s\t%.2f\t%.2f\t%.2f\t%.2f\t%.3f\t%.3f\t%s\t%s\n" % r); f.flush()
        print("%-40s dur %5.2f wer %.3f sim %.3f pause %.2f" % (name[:40], dur, r[7], r[8], pause), flush=True)
if rows:
    a = np.array([[r[3], r[7], r[8], r[6]] for r in rows])
    print(f"SUMMARY {tag}: n={len(rows)} audio={a[:,0].sum():.1f}s meanWER={a[:,1].mean():.3f} "
          f"clips_WER>0.2={int((a[:,1] > 0.2).sum())} meanSIM={a[:,2].mean():.3f} minSIM={a[:,2].min():.3f} "
          f"maxPause={a[:,3].max():.2f}")
