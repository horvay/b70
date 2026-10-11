# Repro of the 2.6K-context MTP fault, with index checks (GLM_XPU_DEBUG_IDX=1, GLM_XPU_GRAPH_EAGER=1 at server start):
# step 1 the stock eager path for the 2-token verify (block graphs off), step 2 the block-graph path (MLA graph Q=2).
import random, time
from exllamav3.generator import Job
from glm_xpu import block_graphs as bg
words = "the river mountain keeper lantern dragon market silver harbor winter letter garden storm candle".split()
random.seed(7)
long_prompt = "Read this, then write a short story about a lighthouse keeper.\n" + " ".join(random.choice(words) for _ in range(2600))
def go(label, n=8):
    gen.pagetable.reset_page_table()
    ids = tok.encode(model.default_chat_prompt(long_prompt), add_bos=False, encode_special_tokens=True)
    job = Job(input_ids=ids, max_new_tokens=n, sampler=ArgmaxSampler())
    gen.enqueue(job); out = []
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("stage") == "error":
                print(f"{label}: ERROR {r.get('error')!r}"[:400], flush=True)
            out.append(r.get("text", "") or "")
    print(f"{label}: ok, accepted drafts {job.accepted_draft_tokens}, text {''.join(out)!r}", flush=True)
bg.set_nosync(True)
step = globals().get("STEP", 1)
if step == 1:
    bg.ENABLED = False
    try:
        go("stock eager verify")
    finally:
        bg.ENABLED = True
else:
    go("block-graph verify (MLA graph, eager phases)")
