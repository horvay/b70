# RAM margin sweep (no-sync router), measured only through the generator: a forcing sampler records the model's
# log-probs at each step and emits the reference token, so every setting decodes the same text. Quality = top-1
# agreement and KL vs exact routing; then decode speed per margin. Margin 0 = today's no-sync; 100 = VRAM experts only.
import time
from exllamav3.generator import Job
from glm_xpu import block_graphs as bg
exec(open("/w/xpu/snip/speed.py").read().split("run(False, 8)")[0])          # run()
MARGINS = globals().get("MARGINS", [0.0, 0.05, 0.1, 0.2, 100.0])
N = globals().get("N_TF", 64)
prompts = ["Explain how a refrigerator works, step by step.",
           "Write a short scene where a detective questions a nervous innkeeper.",
           "What were the main causes of the French Revolution?"]

class Forced(ArgmaxSampler):
    def __init__(self, ref):
        super().__init__(); self.ref = ref; self.i = 0; self.lp = []
    def forward(self, logits, *a, **k):
        self.lp.append(torch.log_softmax(logits.view(-1, logits.shape[-1])[0].float(), -1))
        r = super().forward(logits, *a, **k)
        t = r[0] if isinstance(r, tuple) else r
        t.fill_(self.ref[min(self.i, len(self.ref) - 1)]); self.i += 1
        return r

def forced_run(ids, ref):
    gen.pagetable.reset_page_table()
    s = Forced(ref)
    gen.enqueue(Job(input_ids=ids, max_new_tokens=len(ref), sampler=s, stop_conditions=[]))
    while gen.num_remaining_jobs():
        gen.iterate()
    return torch.stack(s.lp)

import os
REF = "/w/run/margin_refs.pt"                      # exact-routing references, computed once
if os.path.exists(REF):
    refs, base = torch.load(REF)
    base = [b.to(tier.dev) for b in base]
else:
  bg.set_nosync(False)                             # references: exact routing, greedy
  refs = []
  for p in prompts:
      ids = tok.encode(model.default_chat_prompt(p), add_bos=False, encode_special_tokens=True)
      gen.pagetable.reset_page_table()
      job = Job(input_ids=ids, max_new_tokens=N, sampler=ArgmaxSampler(), stop_conditions=[])
      gen.enqueue(job); out = []
      while gen.num_remaining_jobs():
          for r in gen.iterate():
              if r.get("token_ids") is not None:
                  out += r["token_ids"].view(-1).tolist()
      refs.append((ids, out))
  base = [forced_run(ids, ref) for ids, ref in refs]
  torch.save((refs, [b.cpu() for b in base]), REF)
print("references:", [len(r) for _, r in refs], "tokens", flush=True)
bg.set_nosync(True)
try:
    for mg in MARGINS:
        bg.set_margin(mg)
        kl = top1 = n = 0
        for (ids, ref), b in zip(refs, base):
            lp = forced_run(ids, ref)
            m = min(len(lp), len(b))
            kl += float((b[:m].exp() * (b[:m] - lp[:m])).sum())
            top1 += int((lp[:m].argmax(-1) == b[:m].argmax(-1)).sum())
            n += m
        print(f"margin {mg:6.2f}: top-1 agreement {100 * top1 / n:5.1f}%  KL {kl / n:.4f}  ({n} positions)", flush=True)
        run(True, 128); run(True, 128)
finally:
    bg.set_margin(0.0)
    gen.pagetable.reset_page_table()
