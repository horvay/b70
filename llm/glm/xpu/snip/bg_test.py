# Block-level decode graphs: correctness vs eager, then speed. Needs a fresh server (no per-layer KDA graphs).
from glm_xpu import block_graphs as bg
import sys
bg.install(tier)
bg.ENABLED = False
P = "Explain how a refrigerator works, step by step."
_, a, _ = decode(40, prompt=P, warm=False)
_, b, _ = decode(40, prompt=P, warm=False)
bg.ENABLED = True
_, c, _ = decode(40, prompt=P, warm=False)          # warm-up + capture
_, d, _ = decode(40, prompt=P, warm=False)          # replays
print("graph stats", bg.stats)
print("eager A==B", a == b, " graph(capture)==B", c == b, " graph(replay)==B", d == b)
print("B:", b[:160].replace("\n", " | "))
print("D:", d[:160].replace("\n", " | "))
