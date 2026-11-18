"""Reference logits for the C++ engine (engines/glm.cpp): a one-token prompt, then a forced continuation, every step on
the decode graph path (block graphs, exact routing: run with GLM_XPU_RES_BONUS=0). Writes <out>.json (token ids) and
<out>.f32 (each step's log-probabilities, [steps, vocab] fp32).
  in_rootfs.sh --gpu python3 /w/xpu/tests/dumpref.py --text /w/run/nqfake/force.txt --steps 64 --out /w/run/ref/base
"""
import argparse
import json
import os

from glm_xpu.runner import load, log
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--store", default="/nvx/glm53_flash_exl3_experts_blob.bin")
ap.add_argument("--ram-gb", type=float, default=14.0)
ap.add_argument("--cache", type=int, default=8192)
ap.add_argument("--lora", default=None)
ap.add_argument("--text", required=True, help="the forced continuation")
ap.add_argument("--prompt-file", help="a fully rendered prompt before it (default: the text's first token)")
ap.add_argument("--steps", type=int, default=64)
ap.add_argument("--out", required=True)
ap.add_argument("--allow-bonus", action="store_true", help="record the quality-mode (residency bonus) path instead")
ap.add_argument("--time-only", action="store_true",
                help="no logits kept (no per-step log_softmax and copy): the forced decode's speed alone; writes nothing")
a = ap.parse_args()
assert a.allow_bonus or float(os.environ.get("GLM_XPU_RES_BONUS", "0")) == 0, \
    "reference runs need exact routing (GLM_XPU_RES_BONUS=0)"

meta = json.load(open(a.store.rsplit(".", 1)[0] + ".json"))
li_of = {l["key"]: l["li"] for l in meta["layers"]}
stats = json.load(open("/w/xpu/data/route_stats_glm53.json"))
order = [(li, e) for _, li, e in sorted(((sc, li_of[k], e) for k, v in stats.items() if k in li_of
                                         for e, sc in enumerate(v)), reverse=True)]
model, config, cache, tok, tier = load("/models", a.store, cache_size=a.cache, ram_gb=a.ram_gb, warm=order,
                                       lora=a.lora)
from exllamav3 import Generator
from exllamav3.generator import Job
from exllamav3.generator.sampler import ArgmaxSampler
from glm_xpu import block_graphs
gen = Generator(model=model, cache=cache, tokenizer=tok)
block_graphs.install(tier)

ids = tok.encode(open(a.text).read(), add_bos=False)[0].tolist()
if a.prompt_file:
    prompt = tok.encode(open(a.prompt_file).read(), add_bos=False, encode_special_tokens=True)[0].tolist()
    forced = ids[:a.steps]
else:
    prompt, forced = ids[:1], ids[1:a.steps + 1]


class Forcing(ArgmaxSampler):
    def __init__(self, ids):
        super().__init__()
        self.ids, self.rows = ids, []

    def forward(self, logits, *args, **kw):
        import time as _t
        if not hasattr(self, "times"):
            self.times = []
        self.times.append(_t.perf_counter())
        out = super().forward(logits, *args, **kw)
        self.rows.append(None if a.time_only else torch.log_softmax(logits.float().view(-1, logits.shape[-1])[-1], -1).cpu())
        out.fill_(self.ids[min(len(self.rows) - 1, len(self.ids) - 1)])
        return out


fs = Forcing(forced)
gen.enqueue(Job(input_ids=torch.tensor([prompt]), max_new_tokens=len(forced), sampler=fs, stop_conditions=[]))
with torch.inference_mode():
    while gen.num_remaining_jobs():
        gen.iterate()
n = min(len(fs.rows), len(forced))
if a.time_only:
    tt = fs.times
    log(f"DUMPREF decode: {(tt[-1] - tt[1]) * 1e3 / (len(tt) - 2):.1f} ms per token (steps 1..{len(tt) - 1}), time only")
    os._exit(0)
lp = torch.stack(fs.rows[:n])[:, :tok.actual_vocab_size].contiguous()
os.makedirs(os.path.dirname(a.out), exist_ok=True)
lp.numpy().tofile(a.out + ".f32")
json.dump({"tokens": prompt + forced[:n], "prompt_len": len(prompt), "steps": n, "vocab": lp.shape[1], "lora": a.lora,
           "graphs": dict(block_graphs.stats)}, open(a.out + ".json", "w"))
from glm_xpu import block_graphs as _bg
log(f"DUMPREF tier {dict((k, v) for k, v in tier.stats.items() if k in ('picks', 'vram_hit', 'ram_hit', 'nvme', 'admit'))}, "
    f"router swapped {_bg.subst_stats}")
if len(getattr(fs, "times", [])) > 2:
    tt = fs.times
    log(f"DUMPREF decode: {(tt[-1] - tt[1]) * 1e3 / (len(tt) - 2):.1f} ms per token (steps 1..{len(tt) - 1})")
log(f"DUMPREF {n} steps, vocab {lp.shape[1]}, block graphs {dict(block_graphs.stats)} -> {a.out}")
os._exit(0)
