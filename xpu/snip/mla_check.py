# MLA attention in the phase-A graph (bg.MLA_GRAPH) vs eager: greedy text + decode speed, on a short context and on
# one past index_topk (2048) so the sparse selection is exercised. Then per-phase timing with the graph on.
import random
from glm_xpu import block_graphs as bg
exec(open("/w/xpu/snip/speed.py").read().split("run(False, 8)")[0])
words = "the river mountain keeper lantern dragon market silver harbor winter letter garden storm candle".split()
random.seed(7)
long_prompt = "Read this, then write a short story about a lighthouse keeper.\n" + " ".join(random.choice(words) for _ in range(2600))
for label, prompt in (("short context", PROMPT), ("2.6K context (sparse)", long_prompt)):
    print(f"--- {label}", flush=True)
    for on in (False, True, True):
        bg.MLA_GRAPH = on
        print(f"MLA {'graph' if on else 'eager'}:", end=" ", flush=True)
        run(True, 32, prompt)
bg.MLA_GRAPH = True
exec(open("/w/xpu/snip/phases.py").read())
