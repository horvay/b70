# Per-module decode breakdown on the live dev server: python3 xpu/dev.py xpu/snip/prof.py
import collections, time
from exllamav3.modules import hyperconnections as hcm, rmsnorm, gated_delta_net, mla_attn, block_sparse_mlp, mlp, linear
if "_PROF" not in globals():
    _PROF = dict(T=collections.Counter(), N=collections.Counter(), on=[False])

    def _wrap(cls, name, label):
        f = getattr(cls, name)
        def g(*a, **k):
            if not _PROF["on"][0]:
                return f(*a, **k)
            torch.xpu.synchronize(); t0 = time.perf_counter()
            r = f(*a, **k)
            torch.xpu.synchronize()
            _PROF["T"][label] += time.perf_counter() - t0; _PROF["N"][label] += 1
            return r
        setattr(cls, name, g)
    for cls, name, label in ((hcm.HyperConnection, "mix", "hc_mix"), (hcm.HyperConnection, "apply_", "hc_apply"),
                             (rmsnorm.RMSNorm, "forward", "rmsnorm"), (gated_delta_net.GatedDeltaNet, "forward", "kda"),
                             (mla_attn.MLAttention, "forward", "mla"), (block_sparse_mlp.BlockSparseMLP, "forward", "moe"),
                             (mlp.GatedMLP, "forward", "dense/shared mlp"), (type(tier), "ensure", "tier.ensure")):
        _wrap(cls, name, label)
N_TOK = globals().get("N_TOK", 16)
decode(4)
_PROF["T"].clear(); _PROF["N"].clear()
_PROF["on"][0] = True
tps_synced, _, d = decode(N_TOK, warm=False)
_PROF["on"][0] = False
for k, v in _PROF["T"].most_common():
    print(f"{k:18s} {1000 * v / N_TOK:7.1f} ms/token  ({_PROF['N'][k]} calls)")
print("tier", d)
tps, out, d2 = decode(N_TOK, warm=False)
print(f"decode {tps:.2f} tok/s unsynced ({tps_synced:.2f} with timers)")
