# Which checkpoint tensors differ between the in-pass and the old last-page checkpoint, against old-vs-old noise.
import random
import exllamav3.generator.job as J
from exllamav3.generator import Job
words = "the river mountain keeper lantern dragon market silver harbor winter letter garden storm candle".split()
random.seed(11)
body = f"({random.randint(0, 10**9)}) Summarize: " + " ".join(random.choice(words) for _ in range(2000))
rc = gen.recurrent_cache
def ckpt(inpass):
    J._INPASS_CKPT = inpass
    rc.clear(); gen.pagetable.reset_page_table()
    ids = tok.encode(model.default_chat_prompt(body), add_bos=False, encode_special_tokens=True)
    gen.enqueue(Job(input_ids=ids, max_new_tokens=1, sampler=ArgmaxSampler()))
    while gen.num_remaining_jobs():
        gen.iterate()
    return list(rc.items())[-1][1]
def cmp(a, b, label):
    rows = []
    for k in a:
        if k in ("position", "checkpoint_size"):
            continue
        r = [((x.float() - y.float()).norm() / (x.float().norm() + 1e-12)).item() for x, y in zip(a[k], b[k])]
        rows.append((k, r))
    rec = sorted(r[0] for _, r in rows); conv = sorted(r[1] for _, r in rows)
    wk = max(rows, key=lambda kr: max(kr[1]))
    print(f"{label}: recurrent median {rec[len(rec)//2]:.1e} max {rec[-1]:.1e}; conv median {conv[len(conv)//2]:.1e} "
          f"max {conv[-1]:.1e}; worst layer {wk[0]} {['%.1e' % v for v in wk[1]]}", flush=True)
try:
    o1 = ckpt(False); n1 = ckpt(True); o2 = o1
    cmp(o1, o2, "old vs old")
    cmp(o1, n1, "old vs in-pass")
    k0 = [k for k in o1 if k not in ("position", "checkpoint_size")][0]
    print("shapes", [tuple(t.shape) for t in o1[k0]], "norms", [round(t.float().norm().item(), 3) for t in o1[k0]])
finally:
    J._INPASS_CKPT = True
ks = [k for k in o1 if k not in ("position", "checkpoint_size")]
for k in ks[:4] + ks[-2:]:
    print(k, ["%.1e" % ((x.float() - y.float()).norm() / (x.float().norm() + 1e-12)).item() for x, y in zip(o1[k], n1[k])])
