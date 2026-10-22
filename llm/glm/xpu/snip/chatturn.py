# A chat-like second turn: turn 1 = ~2K-token history + 48 generated tokens; turn 2 = turn 1's prompt + its reply + a
# short new message (page table kept, so the cached prefix and its recurrent checkpoint can be reused).
import random, time
from exllamav3.generator import Job
words = "the river mountain keeper lantern dragon market silver harbor winter letter garden storm candle".split()
history = f"({random.randint(0, 10**9)}) Here are my notes: " + " ".join(random.choice(words) for _ in range(2000))
def turn(prompt_text, n):
    ids = tok.encode(prompt_text, add_bos=False, encode_special_tokens=True)
    s0 = dict(tier.stats)
    job = Job(input_ids=ids, max_new_tokens=n, sampler=ArgmaxSampler())
    gen.enqueue(job)
    t0 = time.perf_counter(); t_first = None; out = []
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("stage") == "streaming" and t_first is None:
                t_first = time.perf_counter() - t0
            out.append(r.get("text", ""))
    d = {k: tier.stats[k] - s0.get(k, 0) for k in ("prefill_stage", "prefill_nvme")}
    print(f"  {ids.shape[-1]} tokens, cached {job.cached_pages * 256 + job.cached_tokens}, first token {t_first:.1f} s, "
          f"total {time.perf_counter() - t0:.1f} s {d}", flush=True)
    return "".join(out)
gen.pagetable.reset_page_table()
p1 = model.default_chat_prompt(history + "\nSummarize the notes in one sentence.")
print("turn 1:"); r1 = turn(p1, 48)
p2 = p1 + r1 + "<|user|>\nNow make it shorter.<|assistant|>\n"
print("turn 2:"); r2 = turn(p2, 16)
print("  reply:", repr(r2[:80]))
