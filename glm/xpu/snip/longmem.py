# GPU memory around a long prompt (no decode after it).
import random
from exllamav3.generator import Job
def mem(tag):
    torch.xpu.synchronize()
    free, tot = torch.xpu.mem_get_info(0)
    print(f"{tag}: device free {free / 2**30:.2f} GiB; torch allocated {torch.xpu.memory_allocated() / 2**30:.2f}, "
          f"reserved {torch.xpu.memory_reserved() / 2**30:.2f}, peak allocated {torch.xpu.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
torch.xpu.reset_peak_memory_stats()
mem("before")
for N in globals().get("NS", (1000, 2000)):
    words = "the river mountain keeper lantern dragon market silver harbor winter letter garden storm candle".split()
    p = model.default_chat_prompt(f"({random.randint(0, 10**9)}) Summarize: " + " ".join(random.choice(words) for _ in range(N)))
    ids = tok.encode(p, add_bos=False, encode_special_tokens=True)
    gen.enqueue(Job(input_ids=ids, max_new_tokens=1, sampler=ArgmaxSampler()))
    while gen.num_remaining_jobs():
        gen.iterate()
    mem(f"after {ids.shape[-1]}-token prompt")
