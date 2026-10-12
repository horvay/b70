"""Load GLM-5.3-Flash on the B70 (exllamav3 + the XPU shim + the expert tier).

  from glm_xpu.runner import load
  model, config, cache, tokenizer, tier = load("/models", "/nvx/glm53_flash_exl3_experts_blob.bin")
"""
import os

for k, v in (("EXL3_XPU", "1"), ("EXL3_BC_GDN", "0"), ("EXL3_BC_ATTN", "0"), ("TRITON_F32_DEFAULT", "ieee")):
    os.environ.setdefault(k, v)

from glm_xpu import compat  # noqa: E402

compat.install()

import time  # noqa: E402

import torch  # noqa: E402


def log(*a):
    print(" -- glm_xpu:", *a, flush=True)


def load(model_dir, store_bin, cache_size=32768, max_batch_size=1, ram_gb=24.0, vram_reserve_gb=3.0,
         max_vram_slots=None, stage_slots=288, warm=None, mtp=0):
    from exllamav3 import Config, Model, Cache, Tokenizer
    from exllamav3.cache import CacheLayer_fp16
    from glm_xpu import moe_patch
    from glm_xpu.tier import Tier
    import json

    dev = torch.device("xpu", 0)
    meta = json.load(open(store_bin.rsplit(".", 1)[0] + ".json"))
    layer_li = {l["model_layer"]: l["li"] for l in meta["layers"]}
    moe_patch.install(None, layer_li)

    t0 = time.time()
    config = Config.from_directory(model_dir, load_method="python")
    config.infer_params.no_reconstruct = True
    model = Model.from_config(config)
    # mtp > 0: also load the MTP head (model layer 45) as a draft model for speculative decoding; the main cache keeps
    # mtp tokens of recurrent-state history so a partly rejected draft can be rolled back. Its routed experts are the
    # store's MTP layer, served by the same tier.
    cache = Cache(model, max_num_tokens=cache_size, layer_type=CacheLayer_fp16, max_batch_size=max_batch_size,
                  max_history=mtp) if cache_size else None
    model.load(device=dev, progressbar=True)
    model.mtp = None
    if mtp:
        draft = Model.from_config(config, component="mtp")
        dcache = Cache(draft, max_num_tokens=cache_size, layer_type=CacheLayer_fp16, max_batch_size=max_batch_size)
        draft.load(device=dev, progressbar=True)
        model.mtp = (draft, dcache)
    torch.xpu.synchronize()
    free, total = torch.xpu.mem_get_info(0)
    log(f"model loaded in {time.time() - t0:.0f} s; VRAM free {free / 2**30:.2f} of {total / 2**30:.2f} GiB, "
        f"allocated {torch.xpu.memory_allocated(0) / 2**30:.2f} GiB")

    rec = max(int(l["record_bytes"]) for l in meta["layers"])    # staging rows are sized for the largest record
    budget = free - vram_reserve_gb * 2**30 - 48 * rec      # 48: VRAM victim ring; prefill staging borrows pool slots
    vram_bytes = max(64 * rec, int(budget))
    if max_vram_slots:
        vram_bytes = min(vram_bytes, max_vram_slots * rec)
    tier = Tier(store_bin, dev, vram_bytes, int(ram_gb * 2**30), n_stage=stage_slots)
    moe_patch.TIER = tier
    log(f"{moe_patch.link_layers(model)} MoE layers linked for layer-ahead prefetch")
    if warm:
        tier.warm(warm)
    tokenizer = Tokenizer.from_config(config)
    return model, config, cache, tokenizer, tier
