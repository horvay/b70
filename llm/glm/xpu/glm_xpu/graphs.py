"""XPU graph capture of decode steps (bsz 1, one token) for layers with static shapes, to cut launch overhead.

KDA layers (GatedDeltaNet): one graph per (layer, recurrent slot). The graph replays the whole torch-path layer
forward (qkv/o EXL3 linears, the five fp16 gate linears, conv, recurrent step, gated norm) from a static input buffer.
The slot index is a static device tensor captured into the graph (the generator's per-step host copy is not captured).
Any capture failure leaves that key on the eager path.
"""
import os

import torch

from exllamav3.modules.gated_delta_net import GatedDeltaNet

ENABLED = os.environ.get("GLM_XPU_GRAPHS", "1") == "1"
_orig = {}
_graphs = {}
_failed = set()
_pool = None
stats = dict(replay=0, capture=0, eager=0, failed=0)


def _slot_of(params):
    rs = params.get("recurrent_slots")
    if rs is None or rs.numel() != 1:
        return None
    return int(rs.view(-1)[0]) if rs.device.type == "cpu" else None


def _kda_forward(self, x, params, out_dtype=None):
    orig = _orig["kda"]
    if not ENABLED or x.dim() != 3 or x.shape[0] != 1 or x.shape[1] != 1 or not params.get("recurrent_states") \
            or params.get("recurrent_history") or self.tp_reduce:
        stats["eager"] += 1
        return orig(self, x, params, out_dtype)
    slot = _slot_of(params)
    key = (id(self), slot, x.dtype, out_dtype)
    if slot is None or key in _failed:
        stats["eager"] += 1
        return orig(self, x, params, out_dtype)
    g = _graphs.get(key)
    if g is None:
        global _pool
        y = orig(self, x, params, out_dtype)                 # eager warm-up step (also lazy init on first use)
        try:
            p2 = {k: v for k, v in params.items() if k != "dev_cache"}
            sl = torch.tensor([slot], dtype=torch.int32, device=x.device)
            p2["recurrent_slots"] = sl
            sx = x.clone()
            torch.xpu.synchronize()
            gr = torch.xpu.XPUGraph()
            if _pool is None:
                _pool = torch.xpu.graph_pool_handle()
            # the warm-up already advanced the state by this token: capture on a scratch copy of the slot's state
            rsl = params["recurrent_states"][0].cache.get_recurrent_layer((self.layer_idx, params.get("layer_instance", 0)))
            conv_state, rec_state = rsl.get_state_tensors()
            c0, r0 = conv_state[slot].clone(), rec_state[slot].clone()
            with torch.xpu.graph(gr, pool=_pool):
                sy = orig(self, sx, p2, out_dtype)
            torch.xpu.synchronize()
            conv_state[slot].copy_(c0)                       # capture does not run kernels, but be safe
            rec_state[slot].copy_(r0)
            _graphs[key] = (gr, sx, sy, sl)
            stats["capture"] += 1
        except Exception as e:                               # noqa: BLE001
            _failed.add(key)
            stats["failed"] += 1
            print(f" -- glm_xpu.graphs: KDA capture failed ({type(e).__name__}: {e}); eager for this layer/slot",
                  flush=True)
        return y
    gr, sx, sy, _ = g
    sx.copy_(x)
    gr.replay()
    stats["replay"] += 1
    return sy


def install():
    if "kda" not in _orig:
        _orig["kda"] = GatedDeltaNet.forward
        GatedDeltaNet.forward = _kda_forward


def reset():
    _graphs.clear()
    _failed.clear()
