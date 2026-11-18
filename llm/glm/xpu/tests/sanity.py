"""End-to-end sanity on the B70, in process (no server): is the model's output right at all?
  in_rootfs.sh --gpu python3 /w/xpu/tests/sanity.py [--ram-gb 18] [--tokens 120]
1. Perplexity of a counting sequence (any working model: ~1.0) and of English prose (a 3 bpw GLM: single digits),
   teacher forced through the cache-less forward.
2. Greedy generation through the Generator (paged cache, prefill + decode, graphs), then the same tokens teacher
   forced through the cache-less forward: the share of generated tokens that forward also ranks first. A correct decode
   path agrees on nearly all of them; a broken one generates text the model itself finds unlikely.
Set EXL3_SRC to put another exllamav3 tree first on the path (bisecting a change).
"""
import argparse
import json
import math
import os
import sys
import time

if os.environ.get("EXL3_SRC"):
    sys.path.insert(0, os.environ["EXL3_SRC"])
from glm_xpu.runner import load, log
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--store", default="/nvx/glm53_flash_exl3_experts_blob.bin")
ap.add_argument("--ram-gb", type=float, default=18.0)
ap.add_argument("--cache", type=int, default=8192)
ap.add_argument("--tokens", type=int, default=120)
ap.add_argument("--lora")
ap.add_argument("--prompt-file", nargs="+", help="fully rendered prompts (e.g. ones serve.py built), run in order, instead of the two "
                "questions; one file runs twice")
ap.add_argument("--skip-ppl", action="store_true")
ap.add_argument("--block-graphs", action="store_true", help="install glm_xpu.block_graphs as serve.py does")
ap.add_argument("--force", help="teacher-forced decode: a text file continuing --prompt-file; each decode step's distribution vs the cache-less forward's (KL, top-1)")
ap.add_argument("--mtp", type=int, default=0, help="speculative decoding with N MTP drafts (set GLM_XPU_MTP=N too)")
ap.add_argument("--temp", type=float, default=0.0, help="sample at this temperature (top_p 0.95) instead of greedy")
a = ap.parse_args()

import exllamav3
log(f"exllamav3 from {os.path.dirname(exllamav3.__file__)}")
meta = json.load(open(a.store.rsplit(".", 1)[0] + ".json"))
li_of = {l["key"]: l["li"] for l in meta["layers"]}
stats = json.load(open("/w/xpu/data/route_stats_glm53.json"))
order = [(li, e) for _, li, e in sorted(((sc, li_of[k], e) for k, v in stats.items() if k in li_of
                                         for e, sc in enumerate(v)), reverse=True)]
model, config, cache, tok, tier = load("/models", a.store, cache_size=a.cache, ram_gb=a.ram_gb, warm=order,
                                       lora=a.lora, mtp=a.mtp)


def nc_logprobs(ids):
    with torch.inference_mode():
        logits = model.forward(torch.tensor([ids], dtype=torch.long), {"attn_mode": "flash_attn_nc"}).float()[0]
    return torch.log_softmax(logits, -1).cpu()


def ppl(text, label):
    ids = tok.encode(text)[0].tolist()
    lp = nc_logprobs(ids)
    nll = -sum(float(lp[i - 1, ids[i]]) for i in range(1, len(ids))) / (len(ids) - 1)
    top1 = sum(int(lp[i - 1].argmax()) == ids[i] for i in range(1, len(ids))) / (len(ids) - 1)
    log(f"PPL {label}: {math.exp(nll):.3f} over {len(ids)} tokens, top-1 {top1:.3f}")


if not a.skip_ppl:
    ppl(", ".join(str(i) for i in range(1, 301)), "counting 1..300")
prose = ("The history of the lighthouse begins with fires lit on hilltops to guide ships at night. Later, towers were "
         "built so the light could be seen from farther away. The most famous of the ancient lighthouses stood at "
         "Alexandria in Egypt, on the island of Pharos. It was completed in the third century BC and was one of the "
         "Seven Wonders of the Ancient World. For centuries, lighthouses burned wood, coal or oil, and keepers had to "
         "tend the flame through the night. In the nineteenth century the Fresnel lens made the beams far brighter, and "
         "electricity eventually replaced the flame. Today most lighthouses are automated, and satellite navigation has "
         "made many of them unnecessary, but they remain beloved landmarks along the coasts of the world.")
if not a.skip_ppl:
    ppl(prose, "English prose")

from exllamav3 import Generator
from exllamav3.generator.sampler import ArgmaxSampler, ComboSampler
gen = Generator(model=model, cache=cache, tokenizer=tok, max_chunk_size=int(os.environ.get("GLM_XPU_CHUNK", "8192")),
                **(dict(draft_model=model.mtp[0], draft_cache=model.mtp[1], num_draft_tokens=a.mtp,
                        dynamic_draft_tokens=True) if a.mtp else {}))
if a.block_graphs:
    from glm_xpu import block_graphs
    block_graphs.install(tier)
    log("block graphs installed")
if a.force:
    class Forcing(ArgmaxSampler):
        """Records each step's log-probabilities and returns the next token of the forced continuation."""
        def __init__(self, ids):
            super().__init__()
            self.ids, self.rows = ids, []

        def forward(self, logits, *args, **kw):
            out = super().forward(logits, *args, **kw)
            self.rows.append(torch.log_softmax(logits.float().view(-1, logits.shape[-1])[-1], -1).cpu())
            out.fill_(self.ids[min(len(self.rows) - 1, len(self.ids) - 1)])
            return out

    from exllamav3.generator import Job
    prompt = open(a.prompt_file[0]).read()
    p_ids = tok.encode(prompt, add_bos=False, encode_special_tokens=True)[0].tolist()
    f_ids = tok.encode(open(a.force).read(), add_bos=False)[0].tolist()[:a.tokens]
    ref = nc_logprobs(p_ids + f_ids)[len(p_ids) - 1:len(p_ids) - 1 + len(f_ids)]
    for rep in range(2):
        fs = Forcing(f_ids)
        s0, t0 = dict(tier.stats), time.time()
        from glm_xpu import block_graphs as _bg
        ss0 = dict(_bg.subst_stats)
        job = Job(input_ids=torch.tensor([p_ids]), max_new_tokens=len(f_ids), sampler=fs, stop_conditions=[])
        gen.enqueue(job)
        while gen.num_remaining_jobs():
            gen.iterate()
        n = min(len(fs.rows), len(f_ids))
        dt = time.time() - t0
        d = {k: tier.stats.get(k, 0) - s0.get(k, 0) for k in ("picks", "nosync_miss", "nosync_inflight", "nosync_noslot", "vram_hit", "ram_hit", "nvme", "admit", "ptr_fix")}
        log(f"tier over the pass: {d}; substituted picks {d['nosync_miss'] / max(d['picks'], 1):.1%}; "
            f"{dt:.1f} s for the prompt and {n} steps; router swapped "
            f"{(_bg.subst_stats['substituted'] - ss0['substituted']) / max(_bg.subst_stats['picks'] - ss0['picks'], 1):.1%}"
            f" of decode picks")
        V = min(ref.shape[-1], fs.rows[0].shape[-1], tok.actual_vocab_size)   # the padded vocabulary is -inf
        kl = [float(torch.nan_to_num(ref[j, :V].exp() * (ref[j, :V] - fs.rows[j][:V].clamp(min=-1e4))).sum())
              for j in range(n)]
        top1 = [int(fs.rows[j].argmax()) == int(ref[j].argmax()) for j in range(n)]
        h = n // 2
        log(f"FORCED pass {rep}: {n} decode steps, KL vs cache-less mean {sum(kl) / n:.4f} (first half "
            f"{sum(kl[:h]) / max(h, 1):.4f}, second {sum(kl[h:]) / max(n - h, 1):.4f}, max {max(kl):.3f} at "
            f"{kl.index(max(kl))}), top-1 agreement {sum(top1) / n:.3f}, steps with KL > 0.5: {sum(k > 0.5 for k in kl)}"
            + (f"; drafts accepted {job.accepted_draft_tokens}/{job.accepted_draft_tokens + job.rejected_draft_tokens}"
               f" ({job.accepted_draft_tokens / n:.0%} of the tokens), {dt:.1f} s" if a.mtp else ""))
    log("SANITY DONE")
    raise SystemExit
prompts = [open(f).read() for f in (a.prompt_file * 2 if len(a.prompt_file or []) == 1 else a.prompt_file)] if a.prompt_file else [model.default_chat_prompt(q) for q in (
    "List the eight planets of the solar system in order from the Sun, one per line.",
    "Write the numbers from 1 to 30 separated by commas.")]
for prompt in prompts:
    t0 = time.time()
    out = gen.generate(prompt=prompt, max_new_tokens=a.tokens, sampler=ComboSampler(temperature=a.temp, top_p=0.95) if a.temp else ArgmaxSampler(),
                       completion_only=True, add_bos=False, stop_conditions=config.eos_token_id_list
                       if hasattr(config, "eos_token_id_list") else None)
    dt = time.time() - t0
    p_ids = tok.encode(prompt, add_bos=False, encode_special_tokens=True)[0].tolist()
    o_ids = tok.encode(out, add_bos=False)[0].tolist()
    if len(p_ids) > 3000:       # the cache-less forward of a long prompt needs more VRAM than the tier leaves
        log(f"GEN {len(o_ids)} tokens after a {len(p_ids)}-token prompt in {dt:.1f} s")
        log("OUTPUT: " + out[:1500].replace("\n", " | "))
        continue
    lp = nc_logprobs(p_ids + o_ids)
    agree = [int(lp[len(p_ids) - 1 + j].argmax()) == o_ids[j] for j in range(len(o_ids))]
    nll = -sum(float(lp[len(p_ids) - 1 + j, o_ids[j]]) for j in range(len(o_ids))) / max(len(o_ids), 1)
    log(f"mean NLL of the generated tokens under the cache-less forward {nll:.3f}")
    log(f"GEN {len(o_ids)} tokens in {dt:.1f} s; cache-less forward agrees on {sum(agree)}/{len(agree)}"
        f" (first disagreement at {agree.index(False) if False in agree else '-'})")
    log("OUTPUT: " + out[:1500].replace("\n", " | "))
log("SANITY DONE")
