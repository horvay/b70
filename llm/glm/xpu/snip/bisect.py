# Two prompt passes; record every block's output and every MoE / attention module output; report the first divergence.
from exllamav3.constants import PAGE_SIZE
from exllamav3.modules.transformer import TransformerBlock
from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
from glm_xpu import block_graphs as bg
tier.admit_max = 100000
p = model.default_chat_prompt("Write a short story about a lighthouse keeper.")
ids = tok.encode(p, add_bos=False, encode_special_tokens=True)
cap = -(-ids.shape[-1] // PAGE_SIZE) * PAGE_SIZE
rec = []
orig_block = bg._orig["block"]
orig_moe = BlockSparseMLP.forward
attn_cls = {type(b.attn) for b in model.modules if isinstance(b, TransformerBlock)}
orig_attn = {c: c.forward for c in attn_cls}

def wrap(name_fn, f):
    def g(self, x, params, *a, **k):
        y = f(self, x, params, *a, **k)
        t = y[0] if isinstance(y, tuple) else y
        rec.append((name_fn(self), x.float().cpu().clone(), t.float().cpu().clone()))
        return y
    return g
bg._orig["block"] = wrap(lambda s: "block " + s.key, orig_block)
BlockSparseMLP.forward = wrap(lambda s: "moe " + s.key, orig_moe)
for c in attn_cls:
    c.forward = wrap(lambda s, c=c: c.__name__ + " " + s.key, orig_attn[c])
runs = []
try:
    for i in range(3):
        rec = []
        state = cache.get_new_state()
        model.forward(ids, {"attn_mode": "flash_attn", "cache": cache, "batch_shape": (1, cap), "past_len": 0,
                            "recurrent_states": [state]})
        cache.release_state(state)
        runs.append(rec)
finally:
    bg._orig["block"] = orig_block
    BlockSparseMLP.forward = orig_moe
    for c in attn_cls:
        c.forward = orig_attn[c]
    tier.admit_max = 512
for a, b in ((0, 1), (1, 2), (0, 2)):
    shown = 0
    for (n, xa, ya), (_, xb, yb) in zip(runs[a], runs[b]):
        dx, dy = float((xa - xb).abs().max()), float((ya - yb).abs().max())
        if dy > 0 and shown < 3:
            print(f"runs {a}/{b}: {n}: input diff {dx:.3g}, output diff {dy:.3g} (output scale {float(ya.abs().max()):.3g})")
            shown += 1
    if not shown:
        print(f"runs {a}/{b}: identical")
