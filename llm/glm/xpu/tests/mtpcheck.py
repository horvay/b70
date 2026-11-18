"""MTP verify steps on the synced decode path: greedy generation with and without MTP drafts should agree (both
compute the same model; only the step shapes differ), and the drafts' acceptance rate shows whether MTP can pay.
  in_rootfs.sh --gpu python3 /w/xpu/tests/mtpcheck.py [--mtp 2] [--tokens 128]
Run with GLM_XPU_RES_BONUS=0 for the exact comparison (with the bonus, routing depends on what is cached).
"""
import argparse
import json
import os

from glm_xpu.runner import load, log
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--store", default="/nvx/glm53_flash_exl3_experts_blob.bin")
ap.add_argument("--ram-gb", type=float, default=14.0)
ap.add_argument("--cache", type=int, default=50176)
ap.add_argument("--mtp", type=int, default=2)
ap.add_argument("--tokens", type=int, default=128)
ap.add_argument("--lora", default="/models/lora" if os.path.exists("/models/lora") else None)
a = ap.parse_args()

meta = json.load(open(a.store.rsplit(".", 1)[0] + ".json"))
li_of = {l["key"]: l["li"] for l in meta["layers"]}
stats = json.load(open("/w/xpu/data/route_stats_glm53.json"))
order = [(li, e) for _, li, e in sorted(((sc, li_of[k], e) for k, v in stats.items() if k in li_of
                                         for e, sc in enumerate(v)), reverse=True)]
model, config, cache, tok, tier = load("/models", a.store, cache_size=a.cache, ram_gb=a.ram_gb, warm=order,
                                       lora=a.lora, mtp=a.mtp)
from exllamav3 import Generator
from exllamav3.generator.job import Job
from exllamav3.generator.sampler import ArgmaxSampler
from glm_xpu import block_graphs
block_graphs.install(tier)
q = "Write a long, detailed story about a lighthouse keeper who finds a message in a bottle."
ids = tok.encode(model.default_chat_prompt(q), add_bos=False)


def run(gen):
    job = Job(input_ids=ids, max_new_tokens=a.tokens, sampler=ArgmaxSampler(), stop_conditions=[])
    gen.enqueue(job)
    out = []
    with torch.inference_mode():
        while gen.num_remaining_jobs():
            for r in gen.iterate():
                if r.get("stage") == "streaming" and "token_ids" in r:
                    out += r["token_ids"].view(-1).tolist()
    return out, job


# one Generator owns the cache at a time: the plain one runs first, the speculative one is created after it
base, _ = run(Generator(model=model, cache=cache, tokenizer=tok))
spec = Generator(model=model, cache=cache, tokenizer=tok, draft_model=model.mtp[0], draft_cache=model.mtp[1],
                 num_draft_tokens=a.mtp, dynamic_draft_tokens=True)
for rep in range(2):
    got, job = run(spec)
    n = min(len(base), len(got))
    same = next((i for i in range(n) if base[i] != got[i]), n)
    acc, rej = job.accepted_draft_tokens, job.rejected_draft_tokens
    log(f"MTP {a.mtp} run {rep}: identical to plain greedy for the first {same} of {n} tokens; drafts accepted "
        f"{acc}/{acc + rej} ({acc / max(acc + rej, 1):.0%})")
    if rep == 0:
        log(f"  plain: {tok.decode(torch.tensor(base[:40]))!r}")
        log(f"  MTP:   {tok.decode(torch.tensor(got[:40]))!r}")
        if same < n:
            # margin of the plain model's own choice at the first difference (a near-tie flips on rounding)
            seq = torch.cat([ids, torch.tensor([base[:same]], dtype=ids.dtype)], 1)
            with torch.inference_mode():
                lg = model.forward(seq, {"attn_mode": "flash_attn_nc"}).float()[0, -1]
            top = torch.topk(lg, 2)
            log(f"  at {same}: plain {base[same]} ({lg[base[same]]:.2f}), MTP {got[same]} ({lg[got[same]]:.2f}), "
                f"top-2 {top.indices.tolist()} {[round(v, 2) for v in top.values.tolist()]}")
log(f"block graphs: {dict(block_graphs.stats)}")
log("MTPCHECK DONE")
os._exit(0)
