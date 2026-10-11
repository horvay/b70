"""Decode-step XPU graphs per transformer block (bsz 1, one token, one recurrent slot).

Each block runs as up to three graph replays around the single host sync it needs (reading the router's picks so the
tier can make the experts resident):
  A   attention-site mix + input norm (+ the KDA attention itself; MLA attention runs eagerly after A because its
      kernel grids depend on the context length)
  B   attention-site apply, MLP-site mix + norm, and for MoE layers the router and the next layer's predicted picks
  --  host: read picks, prefetch the next layer's predicted experts, make this layer's picks resident (tier.ensure)
  C   routed experts (B70 MoE kernel through the layer's pointer table) + shared expert (or the dense MLP), MLP apply
The residual stream stack lives in one static buffer. Every value crossing a phase boundary is a static tensor
allocated outside capture. Prefill and anything not matching the decode shape run the stock forward.
"""
import os

import numpy as np
import torch

from exllamav3.modules.transformer import TransformerBlock
from exllamav3.modules.gated_delta_net import GatedDeltaNet
from exllamav3.modules.block_sparse_mlp import BlockSparseMLP

ENABLED = os.environ.get("GLM_XPU_BLOCK_GRAPHS", "1") == "1"
RAM_MARGIN = float(os.environ.get("GLM_XPU_RAM_MARGIN", "0.2"))   # see _B_nosync / set_margin; 2026-10-11 sweep: 0.2 = same quality as 0 (top-1 vs exact 91.7 vs 92.7%, KL 0.085 vs 0.072), 12.9 -> 17.8 tok/s; 0.3+ loses quality
GRAPH_EAGER = os.environ.get("GLM_XPU_GRAPH_EAGER", "0") == "1"   # debug: run every phase eagerly, never capture
GRAPH_PHASES = None   # debug: set of phases (or (phase, Q) pairs) allowed to capture; None = all
GRAPH_BLOCKS = None   # debug: predicate on the block key; blocks it rejects run eagerly
QMAX = 8            # tokens per step on the graph path: decode (1) or MTP verify (the token + up to 7 drafts)
# Router substitution ("dynamic REAP"): the top SUBST_KEEP picks are exact; the remaining slots take the best experts
# that are already in VRAM or RAM (residency bias from the tier), weights renormalised over the chosen set. The exact
# picks that were passed over are loaded in the background so later tokens can use them. 8 = exact (off).
SUBST_KEEP = int(os.environ.get("GLM_XPU_SUBST_KEEP", "8"))
# "replace": fill a passed-over slot with the next-best resident expert; "drop": give it weight 0 and renormalise the
# remaining picks (its slot points at the top pick, so the kernel only reads resident memory)
SUBST_MODE = os.environ.get("GLM_XPU_SUBST_MODE", "replace")
# No-sync decode: the router only picks experts that are in VRAM or RAM (pointer-table row != 0; an NVMe-only pick is
# replaced by the next-best resident expert), so a token runs with no host sync inside it; the tier's bookkeeping runs
# once per token (tier.step_nosync) on the previous token's exact picks. 0 = exact (one host sync per MoE layer).
NOSYNC = os.environ.get("GLM_XPU_NOSYNC", "1") == "1"
# MLA (DSA) attention inside the phase-A graph (glm_xpu/mla_graph.py: device-side lengths, fixed-size selection).
# 0 = eager MLA attention between phases A and B.
MLA_GRAPH = os.environ.get("GLM_XPU_MLA_GRAPH", "1") == "1"
_mla = {}           # device -> mla_graph.Static (this step's lengths and block table)
TIER = None
_orig = {}
_np = {}            # no-sync: device / pinned buffers of every MoE layer's exact picks, and whether they are valid
_pool = None
_resid = {}
_slot_dev = {}
_states = []
subst_stats = dict(picks=0, substituted=0)
stats = dict(replay=0, capture=0, eager=0, failed=0)


def _slot_of(params):
    rs = params.get("recurrent_slots")
    if rs is None or rs.numel() != 1 or rs.device.type != "cpu":
        return None
    return int(rs.view(-1)[0])


class _BlockState:
    """Static buffers and graphs of one block for Q tokens per step (Q = 1 decode; Q > 1 MTP verify: the token and
    its drafts, with recurrent history)."""
    def __init__(self, blk, dev, Q=1):
        H, D = 4, blk.attn_hc.hidden_size if hasattr(blk.attn_hc, "hidden_size") else 4096
        self.Q = Q
        self.key = blk.key
        self.y_attn = torch.empty((1, Q, D), dtype=torch.half, device=dev)
        self.attn_out = torch.empty((1, Q, D), dtype=torch.float, device=dev)
        self.post_a = torch.empty((1, Q, H), dtype=torch.float, device=dev)
        self.comb_a = torch.empty((1, Q, H, H), dtype=torch.float, device=dev)
        self.post_f = torch.empty((1, Q, H), dtype=torch.float, device=dev)
        self.comb_f = torch.empty((1, Q, H, H), dtype=torch.float, device=dev)
        self.y_mlp = torch.empty((1, Q, D), dtype=torch.half, device=dev)
        self.moe = isinstance(blk.mlp, BlockSparseMLP)
        if self.moe:
            k = blk.mlp.num_experts_per_tok
            self.sel = torch.empty((Q, k), dtype=torch.int32, device=dev)
            self.w = torch.empty((Q, k), dtype=torch.float, device=dev)
            # [sel | predicted next | exact (wanted) sel]
            self.picks = torch.empty((3 * k,), dtype=torch.int64, device=dev)
            self.picks_h = torch.empty((3 * k,), dtype=torch.int64, pin_memory=True)
        self.graphs = {}          # (phase, slot) -> XPUGraph
        self.failed = set()


def _A(blk, st, R, params, with_attn, attn_fn=None):
    post, comb, y = blk.attn_hc.mix(R, params)
    st.post_a.copy_(post)
    st.comb_a.copy_(comb)
    y = blk.attn_norm.forward(y.half(), params, out_dtype=torch.half)
    st.y_attn.copy_(y)
    if with_attn:
        st.attn_out.copy_(attn_fn(st.y_attn) if attn_fn else blk.attn.forward(st.y_attn, params))


def _mla_ready(blk, params, dev):
    """MLA block whose attention can run in the phase-A graph this step."""
    ok = blk.__dict__.get("_mla_ok")
    if ok is None:
        from exllamav3.modules.mla_attn import MLAttention
        from glm_xpu import mla_graph
        ok = blk._mla_ok = isinstance(blk.attn, MLAttention) and mla_graph.supported(blk.attn)
    if not ok or params.get("cache") is None:
        return False
    S = _mla.get(dev)
    if S is None:
        from glm_xpu import mla_graph
        S = _mla[dev] = mla_graph.static_for(blk.attn, params["cache"], dev)
        S.refresh(params)
    return S.ok and params.get("cache") is S.cache


def _B(blk, st, R, params):
    blk.attn_hc.apply_(R, st.attn_out, st.post_a, st.comb_a, params)
    post, comb, y = blk.mlp_hc.mix(R, params)
    st.post_f.copy_(post)
    st.comb_f.copy_(comb)
    y = blk.mlp_norm.forward(y.half(), params, out_dtype=torch.half)
    st.y_mlp.copy_(y)
    if st.moe:
        m = blk.mlp
        y2 = st.y_mlp.view(1, -1)
        k = m.num_experts_per_tok
        if SUBST_KEEP < k:
            sel, w, exact = _route_subst(m, y2, SUBST_KEEP)
        else:
            sel, w = m.routing_fn(1, m.routing_cfg, y2, params)
            exact = sel
        st.sel.copy_(sel)
        st.w.copy_(w)
        st.picks[:k].copy_(sel.view(-1))
        st.picks[2 * k:].copy_(exact.view(-1))
        nxt = getattr(m, "_next_moe", None)
        if nxt is not None:
            logits = torch.matmul(y2, nxt.routing_cfg.gate_tensor[:, :nxt.num_experts])
            s = torch.sigmoid(logits.float())
            esb = nxt.routing_cfg.e_score_correction_bias
            if esb is not None:
                s = s + esb.float()
            st.picks[k:2 * k].copy_(torch.topk(s, nxt.num_experts_per_tok, dim=-1).indices.view(-1))
        else:
            st.picks[k:2 * k].copy_(sel.view(-1))


def _B_nosync(blk, st, R, params):
    blk.attn_hc.apply_(R, st.attn_out, st.post_a, st.comb_a, params)
    post, comb, y = blk.mlp_hc.mix(R, params)
    st.post_f.copy_(post)
    st.comb_f.copy_(comb)
    y = blk.mlp_norm.forward(y.half(), params, out_dtype=torch.half)
    st.y_mlp.copy_(y)
    if st.moe:
        m = blk.mlp
        cfg = m.routing_cfg
        E, k, li, Q = m.num_experts, m.num_experts_per_tok, m.tier_li, st.Q
        logits = torch.matmul(st.y_mlp.view(Q, -1), cfg.gate_tensor[:, :E])
        sc = torch.sigmoid(logits.float())
        key = sc + cfg.e_score_correction_bias.float() if cfg.e_score_correction_bias is not None else sc
        # exact picks of every row; rows past Q repeat row 0 (the tier takes the unique set)
        # top-k one row at a time: XPU topk over several rows touched the pinned host allocator during graph capture
        # (CachingHostAllocator use_count assert), which left the graph pool recording and later faulted the GPU
        dv = _np["dev"][li].view(QMAX, k)
        # RAM margin: a RAM-tier expert competes with its score lowered by the margin (0 = plain resident top-k, large =
        # VRAM experts only): RAM experts cross PCIe (~27 GB/s) on every use, VRAM ones don't
        pen = _margin_dev(key.device) * (1.0 - TIER.vram_dev[li])
        key2 = torch.where(TIER.ptrs[li].view(1, -1) != 0, key - pen.view(1, -1), float("-inf"))
        sels = []
        for t in range(Q):
            dv[t:t + 1].copy_(torch.topk(key[t:t + 1], k, dim=-1, sorted=False).indices)
            # top-k among resident experts = the exact picks that are resident + the next-best resident ones
            sels.append(torch.topk(key2[t:t + 1], k, dim=-1, sorted=False).indices)
        for t in range(Q, QMAX):
            dv[t:t + 1].copy_(dv[0:1])
        sel = sels[0] if Q == 1 else torch.cat(sels, 0)
        w = sc.gather(1, sel)
        w = w * cfg.routed_scaling_factor / (w.sum(-1, keepdim=True) + 1e-20)
        st.sel.copy_(sel)
        st.w.copy_(w)


_margin = {}


def _margin_dev(dev):
    m = _margin.get(dev)
    if m is None:
        m = _margin[dev] = torch.full((1,), RAM_MARGIN, dtype=torch.float32, device=dev)
    return m


def set_margin(v):
    """No-sync router RAM margin (static device scalar: captured graphs pick it up, no recapture)."""
    global RAM_MARGIN
    RAM_MARGIN = float(v)
    for m in _margin.values():
        m.fill_(RAM_MARGIN)


def set_nosync(on):
    global NOSYNC
    NOSYNC = bool(on)
    _np["valid"] = False
    _np.pop("snap", None)


def _snapshot_picks():
    """First block of a decode step: the previous token's work is complete (the generator synced on its sampled
    token), so its exact picks can be read back without waiting."""
    if _np.get("valid"):
        _np["host"].copy_(_np["dev"])
        _np["snap"] = True
    _np["valid"] = False


def _step_start():
    """First MoE layer of a no-sync decode step: the tier's per-token bookkeeping on the previous token's exact picks.
    No sync: the GPU meanwhile runs the dense layers before this one, which read no routed experts, and the previous
    token's copies have all run (see tier.step_nosync)."""
    picks = _np["host"].numpy()[:TIER.n_moe] if _np.pop("snap", False) else None
    TIER.step_nosync(picks)


def _route_subst(m, y2, keep):
    """sigmoid / noaux_tc routing with substitution of the low-ranked non-resident picks (see SUBST_KEEP)."""
    cfg = m.routing_cfg
    E, k = m.num_experts, m.num_experts_per_tok
    logits = torch.matmul(y2, cfg.gate_tensor[:, :E])
    s = torch.sigmoid(logits.float())
    key = s + cfg.e_score_correction_bias.float() if cfg.e_score_correction_bias is not None else s
    exact = torch.topk(key, k, dim=-1, sorted=True).indices
    kept = exact[:, :keep]
    if SUBST_MODE == "drop":
        tail = exact[:, keep:]
        ok = TIER.res_dev[m.tier_li].gather(0, tail.view(-1)).view(tail.shape) > -1.0
        sel = torch.cat([kept, torch.where(ok, tail, exact[:, :1].expand_as(tail))], dim=1)
        w = s.gather(1, exact)
        w = w * torch.cat([torch.ones_like(kept, dtype=w.dtype), ok.to(w.dtype)], dim=1)
    else:
        key2 = key + TIER.res_dev[m.tier_li]
        key2 = key2.scatter(1, kept, float("-inf"))
        rest = torch.topk(key2, k - keep, dim=-1, sorted=True).indices
        sel = torch.cat([kept, rest], dim=1)
        w = s.gather(1, sel)
    w = w * cfg.routed_scaling_factor / (w.sum(-1, keepdim=True) + 1e-20)
    return sel, w, exact


def set_subst(keep, mode=None):
    """Change SUBST_KEEP (and SUBST_MODE); drops captured phase-B graphs so they are recaptured with the new routing."""
    global SUBST_KEEP, SUBST_MODE
    SUBST_KEEP = keep
    if mode:
        SUBST_MODE = mode
    for st in _states:
        for key in [key for key in st.graphs if key[0] == "B"]:
            del st.graphs[key]
        for key in [key for key in list(st.__dict__) if key.startswith("warm_('B'")]:
            del st.__dict__[key]


def _C(blk, st, R, params):
    m = blk.mlp
    if st.moe:
        Q = st.Q
        out = TIER.X.moe_forward(st.y_mlp.view(Q, -1), st.sel, st.w, TIER.ptrs[m.tier_li], 2048, TIER.K_of[m.tier_li], TIER.E).float()
        if m.shared_experts is not None:
            out += m.shared_experts.forward(st.y_mlp, params).view(out.shape)
        y = out.view(1, Q, -1)
    else:
        y = m.forward(st.y_mlp, params)
    blk.mlp_hc.apply_(R, y, st.post_f, st.comb_f, params)


def _run(st, phase, slot, fn):
    """Replay the (phase, slot) graph, capturing it on first use (the first call runs eagerly)."""
    global _pool, GRAPH_EAGER
    key = (phase, slot)
    if GRAPH_EAGER or (GRAPH_PHASES is not None and (phase, st.Q) not in GRAPH_PHASES and phase not in GRAPH_PHASES) \
            or (GRAPH_BLOCKS is not None and not GRAPH_BLOCKS(st.key)):
        fn()
        return
    g = st.graphs.get(key)
    if g is not None:
        g.replay()
        stats["replay"] += 1
        return
    if key in st.failed or st.__dict__.get("warm_" + str(key)) is None:
        st.__dict__["warm_" + str(key)] = True
        fn()
        stats["eager"] += 1
        return
    try:
        torch.xpu.synchronize()
        gr = torch.xpu.XPUGraph()
        if _pool is None:
            _pool = torch.xpu.graph_pool_handle()
        with torch.xpu.graph(gr, pool=_pool):
            fn()
        st.graphs[key] = gr
        stats["capture"] += 1
        gr.replay()                    # capture records only; run the step for real
        stats["replay"] += 1
    except Exception as e:             # noqa: BLE001
        st.failed.add(key)
        stats["failed"] += 1
        # a failed capture can leave the pool in recording mode ("already recording to mempool_id" afterwards); replays
        # in that state faulted the GPU (2026-10-11). Run everything eagerly from here on
        GRAPH_EAGER = True
        print(f" -- glm_xpu.block_graphs: capture failed for phase {phase} ({type(e).__name__}: {e}); graphs disabled "
              f"for this session", flush=True)
        fn()


def _forward(self, x, params, out_dtype=None):
    orig = _orig["block"]
    slot = _slot_of(params)
    Q = x.shape[1] if x.dim() == 4 else 0
    hist = bool(params.get("recurrent_history"))
    if Q > 1 and TIER is not None and TIER.min_resident == 0 and NOSYNC:
        TIER.refresh_ptrs()             # residency table before any per-token bookkeeping ran (MTP: every step has Q > 1)
    if (not ENABLED or slot is None or x.dim() != 4 or x.shape[0] != 1 or not 1 <= Q <= QMAX or not self.attn_hc
            or not self.mlp_hc or params.get("prefill") or (hist and Q == 1)
            or params.get("export_state_layers") or self.attn_resid_scalar is not None
            or self.mlp_resid_scalar is not None or self.layer_scalar_f is not None
            # several tokens per step (MTP verify) only on the no-sync path: the synced path reads one token's picks
            or (Q > 1 and isinstance(self.mlp, BlockSparseMLP)
                and not (NOSYNC and self.mlp.tier_li < TIER.n_moe
                         and TIER.min_resident >= self.mlp.num_experts_per_tok))):
        stats["eager"] += 1
        return orig(self, x, params, out_dtype)
    dev = x.device
    R = _resid.get((dev, Q))                         # per Q: captured graphs keep pointing at their own buffer
    if R is None:
        R = _resid[(dev, Q)] = torch.empty_like(x)
    if x.data_ptr() != R.data_ptr():
        R.copy_(x)
    sts = self.__dict__.get("_gstates")
    if sts is None:
        sts = self._gstates = {}
    st = sts.get((Q, hist))
    if st is None:
        st = sts[(Q, hist)] = _BlockState(self, dev, Q)
        _states.append(st)
        if Q == 1 and not hist:
            self._gstate = st                       # (snippets read the decode state here)
    kda = isinstance(self.attn, GatedDeltaNet)

    # KDA reads its state slot through params["recurrent_slots"]; the generator uploads a fresh host tensor every step,
    # which must not be captured. Give the attention a static device copy of the slot index instead.
    pa = params
    if kda:
        sd = _slot_dev.get((dev, slot))
        if sd is None:
            sd = _slot_dev[(dev, slot)] = torch.tensor([slot], dtype=torch.int32, device=dev)
        pa = {k: v for k, v in params.items() if k != "dev_cache"}
        pa["recurrent_slots"] = sd

    first = self.__dict__.get("_first_block")
    if first is None:
        first = self._first_block = self.key.endswith(".layers.0")
    if first:
        if TIER is not None and TIER.stage_borrowed:
            TIER._end_large_forward()               # before any graph replays: frees the prompt's cached blocks
        if NOSYNC and "dev" in _np:
            _snapshot_picks()
        if MLA_GRAPH and dev in _mla:
            _mla[dev].refresh(params)               # this step's lengths / block table, before any graph replays

    if not kda and MLA_GRAPH and _mla_ready(self, params, dev):
        from glm_xpu import mla_graph
        S = _mla[dev]
        _run(st, "Am", -1, lambda: _A(self, st, R, params, True, lambda y: mla_graph.decode(self.attn, y, params, S)))
    else:
        def pA():
            _A(self, st, R, pa, kda)
        _run(st, "A", slot if kda else -1, pA)
        if not kda:
            st.attn_out.copy_(self.attn.forward(st.y_attn, params))
    if st.moe and NOSYNC and self.mlp.tier_li < TIER.n_moe:     # the MTP draft layer always runs exact
        li = self.mlp.tier_li
        if "dev" not in _np:
            k = self.mlp.num_experts_per_tok
            _np["dev"] = torch.zeros((TIER.L, QMAX * k), dtype=torch.int64, device=dev)
            _np["host"] = torch.zeros((TIER.L, QMAX * k), dtype=torch.int64, pin_memory=True)
        if li == 0:
            _step_start()
        if TIER.min_resident >= self.mlp.num_experts_per_tok:
            _run(st, "Bn", -1, lambda: _B_nosync(self, st, R, params))
            _run(st, "C", -1, lambda: _C(self, st, R, params))
            if li == TIER.n_moe - 1:
                _np["valid"] = True
            return R
    if st.moe:
        TIER.sync_res()
    _run(st, "B", -1, lambda: _B(self, st, R, params))
    if st.moe:
        st.picks_h.copy_(st.picks)                 # sync: the host needs this layer's picks
        torch.xpu.current_stream().synchronize()
        p = st.picks_h.numpy()
        k = st.sel.shape[-1]
        m = self.mlp
        TIER.stats["calls"] += 1
        nxt = getattr(m, "_next_moe", None)
        protect = (p[:k] + m.tier_li * TIER.E).tolist()
        if nxt is not None:
            TIER.prefetch(nxt.tier_li, p[k:2 * k], protect=protect)
        if SUBST_KEEP < k:
            missed = set(p[2 * k:].tolist()) - set(p[:k].tolist())
            subst_stats["picks"] += k
            subst_stats["substituted"] += len(missed)
            if missed:
                TIER.prefetch(m.tier_li, np.array(sorted(missed)), protect=protect)   # for later tokens
        TIER.ensure(m.tier_li, p[:k].reshape(1, k))
    _run(st, "C", -1, lambda: _C(self, st, R, params))
    return R


def install(tier):
    global TIER
    TIER = tier
    if "block" not in _orig:
        _orig["block"] = TransformerBlock.forward
        TransformerBlock.forward = _forward


def uninstall():
    if "block" in _orig:
        TransformerBlock.forward = _orig.pop("block")
