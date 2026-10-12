# Timing probe: no-sync with RAM-tier experts hidden from the router (only VRAM experts usable) -> phase C without PCIe.
import numpy as np
orig_refresh = tier.refresh_ptrs
def vram_only():
    orig_refresh()
    vo = tier.vram_of.reshape(tier.L, tier.E)
    tier.ptrs_h[vo < 0] = 0
    tier.min_resident = int((tier.ptrs_h[:tier.n_moe] != 0).sum(1).min())
    tier.ptrs.copy_(tier.ptrs_pin, non_blocking=True)
tier.refresh_ptrs = vram_only
try:
    print("min VRAM experts in a layer:", int((tier.vram_of.reshape(tier.L, tier.E)[:tier.n_moe] >= 0).sum(1).min()))
    exec(open("/w/xpu/snip/phases.py").read())
finally:
    tier.refresh_ptrs = orig_refresh
