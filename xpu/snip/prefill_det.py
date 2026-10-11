# Prompt processing alone, 4x on the same prompt: are the final logits identical? (tier stats deltas per run)
am = globals().get("ADMIT_MAX"); tier.admit_max = am or tier.admit_max; print("admit_max", tier.admit_max)
from exllamav3.constants import PAGE_SIZE
p = model.default_chat_prompt("Write a short story about a lighthouse keeper.")
ids = tok.encode(p, add_bos=False, encode_special_tokens=True)
T = ids.shape[-1]
cap = -(-T // PAGE_SIZE) * PAGE_SIZE
ref = None
for i in range(4):
    gen.pagetable.reset_page_table()
    s0 = dict(tier.stats)
    state = cache.get_new_state()
    lg = model.forward(ids, {"attn_mode": "flash_attn", "cache": cache, "batch_shape": (1, cap), "past_len": 0,
                             "recurrent_states": [state]})[0, -1].float().cpu()
    cache.release_state(state)
    ref = lg if ref is None else ref
    d = {k: tier.stats[k] - s0[k] for k in ("prefill_stage", "prefill_nvme", "vram_hit", "ram_hit", "nvme") if k in s0}
    top = torch.topk(lg, 3)
    print(f"run {i}: max|diff vs run0| {float((lg - ref).abs().max()):.4f}  top3 {[tok.decode(torch.tensor([[t]]))[0] if False else int(t) for t in top.indices]}  {d}", flush=True)
