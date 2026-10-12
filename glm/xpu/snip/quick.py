# One quick pass over the current changes (~1 min on a warm server):
#   1. decode tok/s, exact routing (one host sync per MoE layer) vs no-sync routing
#   2. no-sync quality: how many greedy tokens match exact before the text diverges
import time
from glm_xpu import block_graphs as bg
N = globals().get("N_DEC", 48)

def speed(on, prompt):
    bg.set_nosync(on)
    r, text, d = decode(N, prompt=prompt)
    keep = ("vram_hit", "ram_hit", "nvme", "nosync_miss", "admit", "zero_copy")
    print(f"decode {'no-sync' if on else 'exact  '}: {r:.2f} tok/s  " +
          " ".join(f"{k}={d[k]}" for k in keep if k in d), flush=True)
    return text

def same_prefix(a, b):
    ta, tb = (tok.encode(t, add_bos=False)[0].tolist() for t in (a, b))
    n = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
    return n, min(len(ta), len(tb))

for p in ["Write a short story about a lighthouse keeper."]:
    n, m = same_prefix(speed(False, p), speed(True, p))
    print(f"no-sync matches exact for {n} of {m} tokens", flush=True)
gen.pagetable.reset_page_table()

