# Graph-fault bisection: only PHASES may capture (others eager), 2.6K prompt + 64 MTP tokens.
from glm_xpu import block_graphs as bg
bg.GRAPH_EAGER = False
bg.GRAPH_PHASES = PHASES
bg.GRAPH_BLOCKS = globals().get("BLOCKS")
for st in bg._states:                 # drop graphs captured under earlier settings
    st.graphs.clear()
    for k in [k for k in list(st.__dict__) if k.startswith("warm_")]:
        del st.__dict__[k]
STEP = 2
exec(open("/w/xpu/snip/mtp_long.py").read().replace("n=8", "n=64"))
print("PHASES", PHASES, "stats", bg.stats)
