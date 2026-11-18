#!/usr/bin/env python3
"""Image prep and finishing for b70-albedo.sh (CPU image math only; the model runs in sd-cli on the B70).

prep WORK IN [IN...]   RGBA inputs composited on mid grey, scaled down to fit 1024 px and padded to multiples of 32
                       (edge pixels) into WORK/in/<stem>.png; WORK/manifest.json records how to undo it.
post WORK OUTDIR       WORK/out/<stem>.png (albedo, gamma 2.2) cropped and scaled back to the input's size, the input's
                       alpha put back -> OUTDIR/<stem>.png; plus OUTDIR/<stem>_lumkeep.png: the input's colour with
                       the albedo's luminance, per pixel in linear light (input * lum(albedo) / lum(input)).
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

MAX_SIDE = 1024
GREY = 0.5  # in the picture's own (sRGB) encoding, like a studio backdrop
GAMMA = 2.2
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def stems_for(paths):
    stems = [p.stem for p in paths]
    out = []
    for p, s in zip(paths, stems):
        out.append(f"{p.parent.name}_{s}" if stems.count(s) > 1 else s)
    return out


def prep(work, inputs):
    work = Path(work)
    (work / "in").mkdir(parents=True, exist_ok=True)
    paths = [Path(p).resolve() for p in inputs]
    manifest = []
    for path, stem in zip(paths, stems_for(paths)):
        im = Image.open(path)
        has_alpha = im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info)
        rgba = np.asarray(im.convert("RGBA"), dtype=np.float32) / 255.0
        rgb, a = rgba[..., :3], rgba[..., 3:4]
        if has_alpha:
            rgb = rgb * a + GREY * (1.0 - a)
        h, w = rgb.shape[:2]
        scale = min(1.0, MAX_SIDE / max(w, h))
        sw, sh = max(32, round(w * scale)), max(32, round(h * scale))
        pic = Image.fromarray((rgb * 255.0 + 0.5).astype(np.uint8))
        if (sw, sh) != (w, h):
            pic = pic.resize((sw, sh), Image.LANCZOS)
        pw, ph = -(-sw // 32) * 32, -(-sh // 32) * 32
        arr = np.asarray(pic)
        left, top = (pw - sw) // 2, (ph - sh) // 2
        arr = np.pad(arr, ((top, ph - sh - top), (left, pw - sw - left), (0, 0)), mode="edge")
        Image.fromarray(arr).save(work / "in" / f"{stem}.png")
        manifest.append({"input": str(path), "stem": stem, "size": [w, h], "scaled": [sw, sh], "offset": [left, top],
                         "alpha": bool(has_alpha)})
        print(f"{path} -> {stem}.png {pw}x{ph}" + (" (composited on grey)" if has_alpha else ""))
    (work / "manifest.json").write_text(json.dumps(manifest, indent=1))


def post(work, outdir):
    work, outdir = Path(work), Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for m in json.loads((work / "manifest.json").read_text()):
        w, h = m["size"]
        sw, sh = m["scaled"]
        left, top = m["offset"]
        alb = Image.open(work / "out" / f"{m['stem']}.png").convert("RGB")
        alb = alb.crop((left, top, left + sw, top + sh))
        if (sw, sh) != (w, h):
            alb = alb.resize((w, h), Image.LANCZOS)
        alb = np.asarray(alb, dtype=np.float32) / 255.0

        rgba = np.asarray(Image.open(m["input"]).convert("RGBA"), dtype=np.float32) / 255.0
        rgb, a = rgba[..., :3], rgba[..., 3:4]
        if m["alpha"]:
            rgb = rgb * a + GREY * (1.0 - a)

        alb_lin = alb ** GAMMA
        in_lin = rgb ** GAMMA
        ratio = (alb_lin @ LUMA) / np.maximum(in_lin @ LUMA, 1e-4)
        keep = np.clip(in_lin * ratio[..., None], 0.0, 1.0) ** (1.0 / GAMMA)

        for name, img in ((f"{m['stem']}.png", alb), (f"{m['stem']}_lumkeep.png", keep)):
            out = img if not m["alpha"] else np.concatenate([img, a], axis=-1)
            Image.fromarray((out * 255.0 + 0.5).astype(np.uint8)).save(outdir / name)
        print(f"{outdir / (m['stem'] + '.png')}, {m['stem']}_lumkeep.png")


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "prep":
        prep(sys.argv[2], sys.argv[3:])
    elif len(sys.argv) == 4 and sys.argv[1] == "post":
        post(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)
