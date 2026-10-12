# How many forward passes does one prompt make, and with how many tokens each? (the profiler counted every module x2)
import random
from exllamav3.generator import Job
from exllamav3.modules.mla_attn import MLAttention
from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
calls = []
fm, fb = MLAttention.forward, BlockSparseMLP.forward
def wm(self, x, params, *a, **k):
    calls.append(("mla", self.layer_idx, tuple(x.shape), bool(params.get("prefill")), "cache_seqlens" in params))
    return fm(self, x, params, *a, **k)
def wb(self, x, params, *a, **k):
    calls.append(("moe", getattr(self, "tier_li", -1), tuple(x.shape)))
    return fb(self, x, params, *a, **k)
MLAttention.forward, BlockSparseMLP.forward = wm, wb
words = "the river mountain keeper lantern dragon market silver harbor winter letter garden storm candle".split()
try:
    gen.pagetable.reset_page_table()
    ids = tok.encode(model.default_chat_prompt(f"({random.randint(0, 10**9)}) Summarize: " + " ".join(random.choice(words) for _ in range(2000))), add_bos=False, encode_special_tokens=True)
    gen.enqueue(Job(input_ids=ids, max_new_tokens=1, sampler=ArgmaxSampler()))
    while gen.num_remaining_jobs():
        gen.iterate()
finally:
    MLAttention.forward, BlockSparseMLP.forward = fm, fb
print("prompt", ids.shape[-1], "tokens; recurrent_checkpoint_interval", getattr(gen, "recurrent_checkpoint_interval", None),
      "pp", getattr(gen, "recurrent_checkpoint_interval_pp", None))
for c in calls:
    if c[1] in (3, 0) or c[0] == "mla" and c[1] == 7:
        print(c)
