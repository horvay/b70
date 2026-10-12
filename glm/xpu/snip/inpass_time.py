# Prompt time, old last-page pass vs in-pass checkpoint, alternating, warm kernels, fresh prompts each time.
import random, time
import exllamav3.generator.job as J
from exllamav3.generator import Job
words = "the river mountain keeper lantern dragon market silver harbor winter letter garden storm candle".split()
NW = globals().get("IT_N", 2000)
try:
    for inpass in (False, True, False, True):
        J._INPASS_CKPT = inpass
        gen.pagetable.reset_page_table()
        ids = tok.encode(model.default_chat_prompt(f"({random.randint(0, 10**9)}) Summarize: " + " ".join(random.choice(words) for _ in range(NW))), add_bos=False, encode_special_tokens=True)
        s0 = dict(tier.stats)
        gen.enqueue(Job(input_ids=ids, max_new_tokens=1, sampler=ArgmaxSampler()))
        t0 = time.perf_counter()
        while gen.num_remaining_jobs():
            gen.iterate()
        dt = time.perf_counter() - t0
        d = {k: tier.stats[k] - s0.get(k, 0) for k in ("prefill_stage", "prefill_nvme")}
        print(f"{'in-pass' if inpass else 'old    '}: {ids.shape[-1]} tokens {dt:5.1f} s ({ids.shape[-1] / dt:4.0f} tok/s)  {d}", flush=True)
finally:
    J._INPASS_CKPT = True
