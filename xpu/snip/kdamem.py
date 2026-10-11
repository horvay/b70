# KDA prefill memory split: allocated at chunk_kda entry (what the module keeps alive) and the peak inside it, relative
# to GatedDeltaNet.forward entry; one N-token prompt, worst layer.
import random
from exllamav3.generator import Job
from exllamav3.modules.gated_delta_net import GatedDeltaNet
import exllamav3.vendor.fla as fla
N = globals().get("KM_N", 1000)
rec = {}
of, ok = GatedDeltaNet.forward, fla.chunk_kda
def f(self, x, params, *a, **k):
    T = x.shape[1]
    if T >= 64:
        torch.xpu.synchronize(); rec["base"] = torch.xpu.memory_allocated(); torch.xpu.reset_peak_memory_stats()
    y = of(self, x, params, *a, **k)
    if T >= 64:
        torch.xpu.synchronize()
        p = torch.xpu.max_memory_allocated() - rec["base"]
        if p > rec.get("peak", (0,))[0]:
            rec["peak"] = (p, T, rec.get("enter"), rec.get("inner"))
    return y
def c(*a, **k):
    torch.xpu.synchronize(); e = torch.xpu.memory_allocated() - rec["base"]; torch.xpu.reset_peak_memory_stats()
    m0 = torch.xpu.memory_allocated()
    r = ok(*a, **k)
    torch.xpu.synchronize(); rec["enter"] = e; rec["inner"] = torch.xpu.max_memory_allocated() - m0
    return r
GatedDeltaNet.forward, fla.chunk_kda = f, c
words = "the river mountain keeper lantern dragon market silver harbor winter letter garden storm candle".split()
try:
    gen.pagetable.reset_page_table()
    ids = tok.encode(model.default_chat_prompt(f"({random.randint(0, 10**9)}) Summarize: " + " ".join(random.choice(words) for _ in range(N))), add_bos=False, encode_special_tokens=True)
    gen.enqueue(Job(input_ids=ids, max_new_tokens=1, sampler=ArgmaxSampler()))
    while gen.num_remaining_jobs():
        gen.iterate()
finally:
    GatedDeltaNet.forward, fla.chunk_kda = of, ok
p, T, e, i = rec["peak"]
K = lambda b: f"{b / 2**20:6.1f} MiB ({b / T / 1024:5.1f} KiB/token)"
print(f"T={T}: forward peak {K(p)}; alive at chunk_kda entry {K(e)}; chunk_kda own peak {K(i)}")
