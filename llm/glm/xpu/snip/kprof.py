# Device-time breakdown of decode with the torch profiler: python3 xpu/dev.py xpu/snip/kprof.py
import random
from torch.profiler import profile, ProfilerActivity
from exllamav3.generator import Job
p = model.default_chat_prompt(f"({random.randint(0, 10**9)}) Write a poem about rivers.")
ids = tok.encode(p, add_bos=False, encode_special_tokens=True)
job = Job(input_ids=ids, max_new_tokens=40, sampler=ArgmaxSampler())
gen.enqueue(job)
steps = 0
while gen.num_remaining_jobs() and steps < 20:       # past prefill and first steps
    gen.iterate(); steps += 1
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]) as prof:
    n = 0
    while gen.num_remaining_jobs() and n < 12:
        gen.iterate(); n += 1
    torch.xpu.synchronize()
while gen.num_remaining_jobs():
    gen.iterate()
print(prof.key_averages().table(sort_by="self_xpu_time_total", row_limit=25, max_name_column_width=60))
