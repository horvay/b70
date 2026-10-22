# Time per MTP step part: draft (MTP layer, 1 token), target verify forward (2 tokens), draft prefill (accepted), rewind.
import time, collections
from exllamav3.generator import Job
from glm_xpu import block_graphs as bg
bg.set_nosync(True)
acc = collections.defaultdict(float); cnt = collections.Counter()
def wrap(obj, name, label):
    f = getattr(obj, name)
    def g(*a, **k):
        torch.xpu.synchronize(); t = time.perf_counter()
        r = f(*a, **k)
        torch.xpu.synchronize(); acc[label] += time.perf_counter() - t; cnt[label] += 1
        return r
    setattr(obj, name, g)
    return f
saved = [(gen, "iterate_draftmodel_mtp_gen", wrap(gen, "iterate_draftmodel_mtp_gen", "draft step (MTP layer)")),
         (model, "forward", wrap(model, "forward", "target forward")),
         (gen.draft_model, "prefill", wrap(gen.draft_model, "prefill", "draft prefill (accepted)"))]
from exllamav3.modules.gated_delta_net import GDNState
fr = GDNState.rewind
def rw(self, n):
    torch.xpu.synchronize(); t = time.perf_counter(); r = fr(self, n); torch.xpu.synchronize()
    acc["rewind"] += time.perf_counter() - t; cnt["rewind"] += 1; return r
GDNState.rewind = rw
try:
    gen.pagetable.reset_page_table()
    ids = tok.encode(model.default_chat_prompt("Explain how a refrigerator works, step by step."), add_bos=False, encode_special_tokens=True)
    job = Job(input_ids=ids, max_new_tokens=96, sampler=ArgmaxSampler())
    gen.enqueue(job)
    while gen.num_remaining_jobs():
        ti = time.perf_counter(); gen.iterate(); acc["iterate total"] += time.perf_counter() - ti; cnt["iterate total"] += 1
finally:
    for obj, name, f in saved:
        delattr(obj, name) if name in obj.__dict__ else None
    GDNState.rewind = fr
print("accepted drafts", job.accepted_draft_tokens)
for k, v in sorted(acc.items(), key=lambda kv: -kv[1]):
    print(f"  {1000 * v / cnt[k]:7.1f} ms x{cnt[k]:<4} {k}")
