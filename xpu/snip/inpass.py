# In-pass recurrent checkpoint (job.py _INPASS_CKPT) vs the old separate last-page forward: same prompt both ways,
# compare the stashed checkpoint and prompt time; then a follow-up prompt (prompt + more text) must reuse it.
import random, time
import exllamav3.generator.job as J
from exllamav3.generator import Job
from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
words = "the river mountain keeper lantern dragon market silver harbor winter letter garden storm candle".split()
random.seed(11)
body = f"({random.randint(0, 10**9)}) Summarize: " + " ".join(random.choice(words) for _ in range(2000))
n_moe = [0]
fb = BlockSparseMLP.forward
def wb(self, x, params, *a, **k):
    if x.shape[-2] > 1:
        n_moe[0] += 1
    return fb(self, x, params, *a, **k)
def prompt_run(text, reset=True, n=1):
    if reset:
        gen.pagetable.reset_page_table()
    ids = tok.encode(model.default_chat_prompt(text), add_bos=False, encode_special_tokens=True)
    n_moe[0] = 0
    out = []
    job = Job(input_ids=ids, max_new_tokens=n, sampler=ArgmaxSampler())
    gen.enqueue(job)
    t0 = time.perf_counter()
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            out.append(r.get("text", ""))
    return ids.shape[-1], time.perf_counter() - t0, n_moe[0], "".join(out), job
BlockSparseMLP.forward = wb
rc = gen.recurrent_cache
try:
    res = {}
    for inpass in (False, True):
        J._INPASS_CKPT = inpass
        rc.clear()
        T, dt, nm, text, job = prompt_run(body, n=8)
        key, st = list(rc.items())[-1]
        res[inpass] = st
        print(f"inpass={inpass}: {T} tokens, {dt:.1f} s, large MoE calls {nm}, checkpoint at {st['position']}, "
              f"stashes {len(rc)}; text {text!r}", flush=True)
    a, b = res[False], res[True]
    assert a["position"] == b["position"]
    worst = 0.0
    for k in a:
        if k in ("position", "checkpoint_size"):
            continue
        for x, y in zip(a[k], b[k]):
            worst = max(worst, ((x.float() - y.float()).norm() / (x.float().norm() + 1e-12)).item())
    print(f"checkpoint tensors: worst rel diff {worst:.2e} (KDA prefill noise ~1e-4)", flush=True)
    # follow-up turn: same prompt + more text, page table kept -> must restart from the checkpoint
    T2, dt2, nm2, text2, job2 = prompt_run(body + " And then the keeper climbed the stairs.", reset=False, n=8)
    print(f"follow-up: {T2} tokens, {dt2:.1f} s, large MoE calls {nm2}, cached pages {job2.cached_pages}; text {text2!r}")
finally:
    BlockSparseMLP.forward = fb
    J._INPASS_CKPT = True
