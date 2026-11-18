"""The sampler on the B70 vs the distribution it should draw from (seconds, no model):
  in_rootfs.sh --gpu python3 /w/xpu/tests/sampler.py
A fixed logits row over GLM's vocabulary, sampled N times per sampler setting; prints the drawn frequency of the top
tokens next to the expected probability (softmax at the temperature, renormalised over the top-p set) and the total
variation distance. A correct sampler stays within sampling noise (TV ~0.02 at N=4000).
"""
import math
import os
os.environ.setdefault("EXL3_XPU", "1")
from glm_xpu import compat
compat.install()
import torch
from exllamav3.generator.sampler import ComboSampler

dev = torch.device("xpu", 0)
V, N = 154880, 4000
torch.manual_seed(0)
logits = torch.randn(V) * 2 - 8
head = torch.tensor([6.0, 5.5, 5.0, 4.0, 3.0, 2.5, 2.0, 1.0, 0.0, -1.0])
logits[:10] = head
perm = torch.randperm(V)                   # the likely tokens spread over the vocabulary, not at its start
row = torch.empty(V)
row[perm] = logits


def expected(t, top_p):
    p = torch.softmax(row.double() / t, -1)
    sp, si = p.sort(descending=True)
    keep = (sp.cumsum(0) - sp) < top_p
    q = torch.zeros_like(p)
    q[si[keep]] = sp[keep]
    return q / q.sum()


for dtype in (torch.float, torch.half):
    for t, top_p, top_k, min_p in ((1.0, 1.0, 0, 0.0), (1.0, 0.95, 0, 0.0), (0.7, 0.95, 0, 0.0), (1.0, 0.95, 40, 0.05)):
        s = ComboSampler(temperature=t, top_p=top_p, top_k=top_k, min_p=min_p)
        x = row.to(dtype).to(dev).view(1, 1, V)
        counts = torch.zeros(V, dtype=torch.long)
        for i in range(N):
            tok = int(s.forward(x.clone(), rand_u32=(i * 2654435761 + 12345) & 0xFFFFFFFF).flatten()[0])
            counts[tok] += 1
        f = counts.double() / N
        q = expected(t, top_p)
        if top_k:
            q[q.argsort(descending=True)[top_k:]] = 0
        if min_p:
            q[q < q.max() * min_p] = 0
        q = q / q.sum()
        tv = 0.5 * float((f - q).abs().sum())
        top = q.argsort(descending=True)[:6]
        cols = "  ".join(f"{float(q[j]):.3f}/{float(f[j]):.3f}" for j in top)
        print(f"{str(dtype)[6:]:5} T={t} top_p={top_p} top_k={top_k} min_p={min_p}: TV {tv:.3f}  "
              f"outside the expected set {float(f[q == 0].sum()):.3f}   expected/drawn {cols}", flush=True)
print("SAMPLER DONE")
