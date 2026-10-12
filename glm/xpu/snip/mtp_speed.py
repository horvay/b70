# MTP speculative decode: tok/s and draft acceptance over a 128-token greedy generation (no-sync main model).
import time
from exllamav3.generator import Job
from glm_xpu import block_graphs as bg
bg.set_nosync(True)
for prompt in ("Write a short story about a lighthouse keeper.", "Explain how a refrigerator works, step by step."):
    gen.pagetable.reset_page_table()
    ids = tok.encode(model.default_chat_prompt(prompt), add_bos=False, encode_special_tokens=True)
    job = Job(input_ids=ids, max_new_tokens=128, sampler=ArgmaxSampler())
    gen.enqueue(job)
    t0 = time.perf_counter(); t1 = None; got = 0; text = ""
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            ids_ = r.get("token_ids")
            k = int(ids_.numel()) if ids_ is not None else 0
            if k and t1 is None:
                t1 = time.perf_counter()
            got += k; text += r.get("text", "") or ""
    t2 = time.perf_counter()
    acc = getattr(job, "accepted_draft_tokens", None)
    print(f"MTP {gen.num_draft_tokens}: decode {(got - 1) / (t2 - t1):5.2f} tok/s ({got} tokens), prompt {t1 - t0:.2f} s, "
          f"accepted drafts {acc} of ~{got} tokens\n    {text[:90]!r}", flush=True)
