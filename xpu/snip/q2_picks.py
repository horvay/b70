# What did the last 2-token verify pick per MoE layer: unique experts, how many in VRAM / RAM (from the static sel).
import numpy as np
from exllamav3.modules.transformer import TransformerBlock
from glm_xpu import block_graphs as bg
vo = tier.vram_of.reshape(tier.L, tier.E)
tot = dict(uniq=0, vram=0, ram=0, other=0, n=0, shared=0)
for b in model.modules:
    if not isinstance(b, TransformerBlock) or not hasattr(b, "_gstates"):
        continue
    st = b._gstates.get((2, True))
    if st is None or not st.moe:
        continue
    li = b.mlp.tier_li
    sel = st.sel.cpu().numpy()
    u = np.unique(sel)
    tot["uniq"] += len(u); tot["n"] += 1
    tot["shared"] += len(set(sel[0]) & set(sel[1]))
    for e in u:
        if vo[li, e] >= 0:
            tot["vram"] += 1
        elif tier.ptrs_h[li, e] != 0:
            tot["ram"] += 1
        else:
            tot["other"] += 1
n = tot["n"]
print(f"{n} layers; per layer: unique experts {tot['uniq'] / n:.1f} (shared by both tokens {tot['shared'] / n:.1f}), "
      f"VRAM {tot['vram'] / n:.1f}, RAM {tot['ram'] / n:.1f}, other {tot['other'] / n:.1f}")
