# Same prompt, exact routing, greedy, prompt cache cleared each run: which recurrent-state slot did each run use?
from glm_xpu import block_graphs as bg
bg.set_nosync(False)
print("slots:", cache.num_slots, "free:", list(cache.free_list), flush=True)
for i in range(4):
    gen.pagetable.reset_page_table()
    before = set(bg._slot_dev)
    free0 = list(cache.free_list)
    r, text, d = decode(12, warm=False)
    print(f"run {i}: free before {free0} -> after {list(cache.free_list)}; graph slots {sorted(k[1] for k in bg._slot_dev)}: {text!r}", flush=True)
