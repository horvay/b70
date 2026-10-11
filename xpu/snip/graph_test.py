# Compare greedy text eager vs with KDA decode graphs, then time decode with graphs.
from glm_xpu import graphs
graphs.ENABLED = False
graphs.install()
_, ref, _ = decode(48, prompt="Explain how a refrigerator works, step by step.")
graphs.ENABLED = True
_, out1, _ = decode(48, prompt="Explain how a refrigerator works, step by step.", warm=False)   # captures
_, out2, _ = decode(48, prompt="Explain how a refrigerator works, step by step.", warm=False)   # replays
print("graph stats", graphs.stats)
print("eager == graph run1:", ref == out1, " eager == graph run2:", ref == out2)
if ref != out2:
    import difflib
    print("eager:", ref[:300].replace("\n", " | "))
    print("graph:", out2[:300].replace("\n", " | "))
