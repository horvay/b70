# Decode speed only (prompt processing timed separately): exact vs no-sync, N tokens, greedy.
import time
from exllamav3.generator import Job
from glm_xpu import block_graphs as bg
N = globals().get("N_DEC", 64)
PROMPT = globals().get("PROMPT", "Write a short story about a lighthouse keeper.")

def run(nosync, n=N, prompt=PROMPT):
    bg.set_nosync(nosync)
    gen.pagetable.reset_page_table()
    ids = tok.encode(model.default_chat_prompt(prompt), add_bos=False, encode_special_tokens=True)
    s0 = dict(tier.stats)
    job = Job(input_ids=ids, max_new_tokens=n, sampler=ArgmaxSampler())
    gen.enqueue(job)
    t0 = time.perf_counter(); t1 = None; got = 0; text = ""; slow = (0.0, -1)
    while gen.num_remaining_jobs():
        ti = time.perf_counter()
        rs = gen.iterate()
        dt = time.perf_counter() - ti
        if t1 is not None and dt > slow[0]:
            slow = (dt, got)
        for r in rs:
            k = len(r.get("token_ids", torch.empty(0)).view(-1)) if r.get("token_ids") is not None else 0
            if k and t1 is None:
                t1 = time.perf_counter()
            got += k
            text += r.get("text", "") or ""
    t2 = time.perf_counter()
    d = {k: round(tier.stats[k] - s0.get(k, 0)) for k in ("vram_hit", "ram_hit", "nvme", "nosync_miss", "admit", "zero_copy") if k in tier.stats}
    print(f"{'no-sync' if nosync else 'exact  '}: decode {(got - 1) / (t2 - t1):5.2f} tok/s ({got} tokens), prompt {t1 - t0:.2f} s, slowest step {1000 * slow[0]:.0f} ms at token {slow[1]}  {d}\n    {text[:90]!r}", flush=True)

run(False, 8)                     # warm-up (graph capture)
for _ in range(2):
    run(False)
    run(True)
