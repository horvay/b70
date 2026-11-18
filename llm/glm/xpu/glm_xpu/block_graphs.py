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
import time

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
# exact decode prefetch: the next MoE layer's router run on this layer's MLP input predicts its picks; its top
# PF_WIDTH (>= k) non-resident experts are read ahead, and the top PF_WIDTH2 of the layer after it (0 = none)
PF_WIDTH = int(os.environ.get("GLM_XPU_PF_WIDTH", "12"))   # 2026-10-30 forced 200-token pass: 8 28.4 s, 12 26.2, 16 27.2, 24 30.1
PF_WIDTH2 = int(os.environ.get("GLM_XPU_PF_WIDTH2", "0"))
# "replace": fill a passed-over slot with the next-best resident expert; "drop": give it weight 0 and renormalise the
# remaining picks (its slot points at the top pick, so the kernel only reads resident memory)
SUBST_MODE = os.environ.get("GLM_XPU_SUBST_MODE", "replace")
# Residency bonus (exact decode): experts in VRAM or RAM compete with RES_BONUS added to their selection score
# (sigmoid + correction bias), so a near-tie goes to the cached expert and a clear preference still reads from NVMe.
# Weights stay the un-biased scores; the passed-over exact picks are read in the background and count for VRAM
# admission. 0 = off.
RES_BONUS = float(os.environ.get("GLM_XPU_RES_BONUS", "0"))
# with the bonus: how many passed-over exact picks per layer call are read for later tokens (highest ranked first);
# -1 = all of them
MISS_PF = int(os.environ.get("GLM_XPU_MISS_PF", "-1"))
# the bonus / multi-row router's tail (sigmoid, bias, bonus, top-ks, weights, picks) as one Triton kernel (k_router)
FUSED_ROUTER = os.environ.get("GLM_XPU_FUSED_ROUTER", "1") == "1"
STAGE_TOPK = int(os.environ.get("GLM_XPU_STAGE_TOPK", "8"))   # layer-ahead staging: predicted ranks considered


def _subst_on(k):
    return SUBST_KEEP < k or RES_BONUS > 0
# No-sync decode: the router only picks experts that are in VRAM or RAM (pointer-table row != 0; an NVMe-only pick is
# replaced by the next-best resident expert), so a token runs with no host sync inside it; the tier's bookkeeping runs
# once per token (tier.step_nosync) on the previous token's exact picks. 0 = exact (one host sync per MoE layer).
# Off by default: with the 3.10 bpw quant (K=3 experts) about a third of the experts fit in VRAM + RAM, and no-sync
# replaced 37% of all picks (19% with RAM margin 0); teacher forced against the cache-less forward, KL 0.52 and top-1
# 66% vs exact 0.006-0.03 and 93-96%, and the model wrote word salad and loops (tests/sanity.py --force,
# 2026-10-28). Dropping missing picks instead of replacing them was no better (KL 0.22-0.34).
NOSYNC = os.environ.get("GLM_XPU_NOSYNC", "0") == "1"
# MTP verify steps with exact routing for the token only: row 0 (the token after the last accepted one) keeps the exact
# picks, so the first draft is accepted only if the full-quality model agrees with it; the draft rows (1..Q-1) pick
# among experts already in VRAM or RAM, so they add no NVMe reads. The tokens accepted past the first draft are
# confirmed by the VRAM/RAM-only model (quality between quality and fast mode). 0 = every row exact.
SPEC_RO = os.environ.get("GLM_XPU_SPEC_RO", "0") == "1"
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
# GLM_XPU_PROF=1: where decode time goes. GPU time of every graph replay by phase (XPU timing events), host seconds
# waiting at the per-layer routing sync, in TIER.ensure (its NVMe wait is TIER.stats t_nvme) and in prefetch; prof_take()
# GLM_XPU_PROF=2: host timers only (no GPU events, which slow decode by ~50%): the GPU idles from each sync's return
# until the expert phase is submitted (pre_c_s), so that is the host glue to cut
PROF_LEVEL = int(os.environ.get("GLM_XPU_PROF", "0") or 0)
PROF = PROF_LEVEL > 0
PROF_EV = PROF_LEVEL == 1
_prof_ev = []
prof = dict(sync_s=0.0, ensure_s=0.0, prefetch_s=0.0, gap_s=0.0, replay_s=0.0, n_replay=0, pre_c_s=0.0,
            ram_picks=0, ram_predicted=0)
_pred = {}          # PROF: tier layer -> experts the previous MoE layer predicted for it
PROF_SPLIT_B = PROF_EV and os.environ.get("GLM_XPU_PROF_SPLIT_B", "1") == "1"   # B as two graphs, B1 and B2


def prof_take():
    """Totals since the last call: GPU ms per phase (A attention, Am MLA attention, B router, C experts + shared expert,
    ...) and the host seconds above. Synchronises."""
    torch.xpu.synchronize()
    gpu = {}
    for ph, e0, e1 in _prof_ev:
        gpu[ph] = gpu.get(ph, 0.0) + e0.elapsed_time(e1)
    _prof_ev.clear()
    out = dict(prof, gpu_ms=gpu)
    for k in prof:
        prof[k] = 0.0
    return out


def _slot_of(params):
    rs = params.get("recurrent_slots")
    if rs is None or rs.numel() != 1 or rs.device.type != "cpu":
        return None
    return int(rs.view(-1)[0])


def picks_layout(Q, k):
    """(w1, sizes of the picks buffer sections): per row of Q tokens, k selected, k exact (best first),
    w1 = max(PF_WIDTH, k) predicted for the next MoE layer, PF_WIDTH2 for the one after."""
    w1 = max(PF_WIDTH, k)
    return w1, Q * k, Q * k, Q * w1, Q * PF_WIDTH2


def _topk_rows(x, k, sorted=True):
    """top-k indices row by row: a batched XPU topk faulted under graph capture (see _B_nosync) and its sbtopk kernel
    hits intel/llvm#21972 with the SYCL cache on"""
    if x.shape[0] == 1:
        return torch.topk(x, k, dim=-1, sorted=sorted).indices
    return torch.cat([torch.topk(x[t:t + 1], k, dim=-1, sorted=sorted).indices for t in range(x.shape[0])], 0)


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
            # picks_layout(): [sel Q*k | exact (wanted) Q*k | predicted next layer Q*w1 | the layer after Q*PF_WIDTH2]
            n = sum(picks_layout(Q, k)[1:])
            self.picks = torch.empty((n,), dtype=torch.int64, device=dev)
            self.picks_h = torch.empty((n,), dtype=torch.int64, pin_memory=True)
        self.graphs = {}          # (phase, slot) -> XPUGraph
        self.failed = set()


MIX_INTO = os.environ.get("GLM_XPU_MIX_INTO", "1") == "1"


def _mix_norm_into(hc, norm, R, params, post_out, comb_out, y_out):
    """hc.mix(R) then norm(collapsed) with the results written straight into the block state (post_out, comb_out,
    y_out) instead of workspaces copied over afterwards: three fewer kernels per mix (2 per block, 46 blocks). Same
    kernels and arithmetic as HyperConnection.mix's fused path and RMSNorm.forward; falls back to them otherwise."""
    b, s_, H, D = R.shape
    ok = (MIX_INTO and hc.hc_mult == 4 and R.dtype == torch.float and D % 4 == 0 and R.is_contiguous()
          and b * s_ <= 32 and not norm.span_heads and norm.groups == 1 and not norm.unweighted
          and norm.key not in params.get("export_state_norm_keys", ())
          and post_out.is_contiguous() and comb_out.is_contiguous() and y_out.is_contiguous())
    if not ok:
        post, comb, y = hc.mix(R, params)
        post_out.copy_(post)
        comb_out.copy_(comb)
        y_out.copy_(norm.forward(y.half(), params, out_dtype=torch.half))
        return
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.util.tensor import g_tensor_cache
    n = b * s_
    st = R.view(n, H, D)
    chunks = ext.hc_mix_num_chunks(n, H * D)
    M1 = 2 * H + H * H + 1
    partials = g_tensor_cache.get_bucketed(R.device, n * chunks * M1, torch.float, "hc_mix_partials").view(n, chunks, M1)
    collapsed = g_tensor_cache.get_bucketed(R.device, n * D, torch.half, "hc_coll").view(n, D)
    if hc.fn_h is None:
        hc.fn_h = hc.fn.half()
    ext.hc_mix(st, hc.fn_h, hc.base, hc.scale, hc.rms_eps, hc.hc_eps, hc.sinkhorn_iters, partials,
               post_out.view(n, H), comb_out.view(n, H, H), collapsed)
    ext.rms_norm(collapsed, norm.weight, y_out.view(n, D), norm.rms_norm_eps, norm.constant_bias,
                 norm.constant_scale, False, False)


def _A(blk, st, R, params, with_attn, attn_fn=None):
    _mix_norm_into(blk.attn_hc, blk.attn_norm, R, params, st.post_a, st.comb_a, st.y_attn)
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


def _route_scores(m, y2):
    """MoE layer m's router scores as its top-k ranks them for hidden state y2: sigmoid + correction bias, plus
    RES_BONUS for resident experts (so a prefetch prediction reads what the bonus router will actually pick)."""
    s = torch.sigmoid(torch.matmul(y2, m.routing_cfg.gate_tensor[:, :m.num_experts]).float())
    esb = m.routing_cfg.e_score_correction_bias
    s = s + esb.float() if esb is not None else s
    if RES_BONUS > 0 and getattr(m, "tier_li", None) is not None:
        s = s + TIER.bonus_dev[m.tier_li]
    return s


def _B(blk, st, R, params):
    _B1(blk, st, R, params)
    _B2(blk, st, params)


def _B1(blk, st, R, params):
    """attention residual (mHC apply), MLP mix and norm"""
    blk.attn_hc.apply_(R, st.attn_out, st.post_a, st.comb_a, params)
    _mix_norm_into(blk.mlp_hc, blk.mlp_norm, R, params, st.post_f, st.comb_f, st.y_mlp)


def _B2(blk, st, params):
    """routing (exact, bonus) and the next layers' prefetch predictions"""
    if st.moe:
        m = blk.mlp
        Q = st.Q
        y2 = st.y_mlp.view(Q, -1)            # Q > 1: an MTP verify step (the token and its drafts)
        k = m.num_experts_per_tok
        if FUSED_ROUTER and (RES_BONUS > 0 or Q > 1) and SUBST_KEEP >= k:
            from glm_xpu import k_router
            cfg, E = m.routing_cfg, m.num_experts
            nxt = getattr(m, "_next_moe", None)
            lc = torch.matmul(y2, cfg.gate_tensor[:, :E])
            ln = torch.matmul(y2, nxt.routing_cfg.gate_tensor[:, :nxt.num_experts]) if nxt is not None else None
            bon = RES_BONUS > 0
            k_router.route(lc, ln, cfg.e_score_correction_bias,
                           nxt.routing_cfg.e_score_correction_bias if nxt is not None else None,
                           TIER.bonus_dev[m.tier_li] if bon else None,
                           TIER.bonus_dev[nxt.tier_li] if bon and nxt is not None else None,
                           cfg.routed_scaling_factor, st.sel, st.w, st.picks, k, picks_layout(Q, k)[0])
            if PF_WIDTH2:
                # the layer after next (two-ahead prefetch): its own top-k after the fused kernel's sections
                nxt2 = getattr(nxt, "_next_moe", None) if nxt is not None else None
                w1, n_sel, n_ex, n_pn, n_pn2 = picks_layout(Q, k)
                if nxt2 is not None:
                    st.picks[n_sel + n_ex + n_pn:].copy_(_topk_rows(_route_scores(nxt2, y2), PF_WIDTH2).view(-1))
            if SPEC_RO and Q > 1 and m.tier_li < TIER.n_moe:
                _ro_rows(m, st, y2, lc)
            return
        if _subst_on(k):
            sel, w, exact = _route_subst(m, y2, SUBST_KEEP)
        elif Q == 1:
            sel, w = m.routing_fn(1, m.routing_cfg, y2, params)
            exact = sel
        else:
            sel, w, exact = _route_subst(m, y2, k)
        st.sel.copy_(sel)
        st.w.copy_(w)
        w1, n_sel, n_ex, n_pn, n_pn2 = picks_layout(Q, k)
        st.picks[:n_sel].copy_(sel.view(-1))
        st.picks[n_sel:n_sel + n_ex].copy_(exact.view(-1))
        nxt = getattr(m, "_next_moe", None)
        if nxt is not None:
            o = n_sel + n_ex
            st.picks[o:o + n_pn].copy_(_topk_rows(_route_scores(nxt, y2), w1).view(-1))
            nxt2 = getattr(nxt, "_next_moe", None)
            if PF_WIDTH2:
                st.picks[o + n_pn:].copy_(_topk_rows(_route_scores(nxt2 or nxt, y2), PF_WIDTH2).view(-1))
        if SPEC_RO and Q > 1 and m.tier_li < TIER.n_moe:
            _ro_rows(m, st, y2)


def _ro_rows(m, st, y2, lc=None):
    """SPEC_RO: re-route the draft rows (1..Q-1) of a verify step among resident experts (pointer row != 0, with the
    residency bonus); the host's ensure() still sees every row's picks, so a stale pointer row only costs a read."""
    cfg, E, k, li, Q = m.routing_cfg, m.num_experts, m.num_experts_per_tok, m.tier_li, st.Q
    lc = lc[1:] if lc is not None else torch.matmul(y2[1:], cfg.gate_tensor[:, :E])
    sc = torch.sigmoid(lc.float())
    key = sc + cfg.e_score_correction_bias.float() if cfg.e_score_correction_bias is not None else sc
    if RES_BONUS > 0:
        key = key + TIER.bonus_dev[li]
    key = torch.where(TIER.ptrs[li].view(1, -1) != 0, key, float("-inf"))
    # one row at a time (see _B_nosync: multi-row XPU topk during graph capture)
    sel = torch.cat([torch.topk(key[t:t + 1], k, dim=-1, sorted=False).indices for t in range(Q - 1)], 0)
    w = sc.gather(1, sel)
    w = w * cfg.routed_scaling_factor / (w.sum(-1, keepdim=True) + 1e-20)
    st.sel[1:].copy_(sel)
    st.w[1:].copy_(w)
    st.picks[k:Q * k].copy_(sel.view(-1))


def _B_nosync(blk, st, R, params):
    blk.attn_hc.apply_(R, st.attn_out, st.post_a, st.comb_a, params)
    _mix_norm_into(blk.mlp_hc, blk.mlp_norm, R, params, st.post_f, st.comb_f, st.y_mlp)
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
    exact = _topk_rows(key, k)
    if RES_BONUS > 0:
        sel = _topk_rows(key + TIER.bonus_dev[m.tier_li], k)     # VRAM / RAM residency bonus (tier.bonus_v / r)
        w = s.gather(1, sel)
        w = w * cfg.routed_scaling_factor / (w.sum(-1, keepdim=True) + 1e-20)
        return sel, w, exact
    if keep >= k:
        sel = exact
        w = s.gather(1, sel)
        w = w * cfg.routed_scaling_factor / (w.sum(-1, keepdim=True) + 1e-20)
        return sel, w, exact
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
        rest = _topk_rows(key2, k - keep)
        sel = torch.cat([kept, rest], dim=1)
        w = s.gather(1, sel)
    w = w * cfg.routed_scaling_factor / (w.sum(-1, keepdim=True) + 1e-20)
    return sel, w, exact


def set_subst(keep, mode=None):
    """Change SUBST_KEEP (and SUBST_MODE); drops captured phase-B graphs so they are recaptured with the new routing."""
    global SUBST_KEEP, SUBST_MODE
    SUBST_KEEP = keep
    if TIER is not None and keep < 8:
        TIER.upload_res = TIER.res_dirty = True
    if mode:
        SUBST_MODE = mode
    for st in _states:
        for key in [key for key in st.graphs if key[0] == "B"]:
            del st.graphs[key]
        for key in [key for key in list(st.__dict__) if key.startswith("warm_('B'")]:
            del st.__dict__[key]


# The shared expert does not depend on the routing picks: as its own graph (phase S) it runs on the GPU while the host
# does the layer's cache step, which waits only for the picks' copy (an event) instead of a full stream sync.
SPLIT_SHARED = os.environ.get("GLM_XPU_SPLIT_SHARED", "1") == "1"


def _S(blk, st, params):
    st.sh = blk.mlp.shared_experts.forward(st.y_mlp, params)   # graph-pool output, kept alive by this reference


def _C(blk, st, R, params):
    m = blk.mlp
    if st.moe:
        Q = st.Q
        out = TIER.X.moe_forward(st.y_mlp.view(Q, -1), st.sel, st.w, TIER.ptrs[m.tier_li], 2048, TIER.K_of[m.tier_li], TIER.E).float()
        if m.shared_experts is not None:
            sh = st.__dict__.get("sh") if st.__dict__.get("split") else None
            out += (sh if sh is not None else m.shared_experts.forward(st.y_mlp, params)).view(out.shape)
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
        if PROF_EV:
            e0, e1 = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
            e0.record()
            h0 = time.perf_counter()
            g.replay()
            prof["replay_s"] += time.perf_counter() - h0      # host cost of handing the graph to the driver
            prof["n_replay"] += 1
            e1.record()
            _prof_ev.append((phase, e0, e1))
        else:
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
            # several tokens per step (MTP verify): the synced path reads every row's picks; the no-sync path needs
            # enough resident experts per layer
            or (Q > 1 and NOSYNC and isinstance(self.mlp, BlockSparseMLP)
                and not (self.mlp.tier_li < TIER.n_moe and TIER.min_resident >= self.mlp.num_experts_per_tok))):
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
    if PROF_SPLIT_B:
        _run(st, "B1", -1, lambda: _B1(self, st, R, params))
        _run(st, "B2", -1, lambda: _B2(self, st, params))
    else:
        _run(st, "B", -1, lambda: _B(self, st, R, params))
    t_sync = None
    if st.moe:
        split = SPLIT_SHARED and self.mlp.shared_experts is not None
        if split != st.__dict__.get("split"):
            st.split = split
            st.graphs.pop(("C", -1), None)          # C captured without / with the shared expert inside
        if split:
            st.picks_h.copy_(st.picks, non_blocking=True)
            ev = st.__dict__.get("pev")
            if ev is None:
                ev = st.pev = torch.xpu.Event()
            ev.record()
            _run(st, "S", -1, lambda: _S(self, st, params))
        else:
            st.picks_h.copy_(st.picks)             # sync: the host needs this layer's picks
        if PROF:
            t0 = time.perf_counter()
        if split:
            ev.synchronize()                       # the picks only; the shared expert keeps running
        else:
            torch.xpu.current_stream().synchronize()
        if PROF:
            t_sync = time.perf_counter()
            prof["sync_s"] += t_sync - t0
            t0 = t_sync
        p = st.picks_h.numpy()
        Q, k = st.sel.shape
        m = self.mlp
        TIER.stats["calls"] += 1
        w1, n_sel, n_ex, n_pn, n_pn2 = picks_layout(Q, k)
        sel_h = p[:n_sel].reshape(Q, k)
        ex_h = p[n_sel:n_sel + n_ex].reshape(Q, k)
        if PROF:
            # would a layer-ahead copy of predicted RAM-resident experts into VRAM have covered this layer's RAM picks?
            pr = _pred.get(m.tier_li, ())
            for e in set(sel_h.ravel().tolist()):
                kk = m.tier_li * TIER.E + e
                if TIER.vram_of[kk] < 0 and TIER.ram_of[kk] >= 0:
                    prof["ram_picks"] += 1
                    prof["ram_predicted"] += e in pr
            nx = getattr(self.mlp, "_next_moe", None)
            if nx is not None:
                w1_, n_sel_, n_ex_, n_pn_, _ = picks_layout(st.sel.shape[0], st.sel.shape[1])
                _pred[nx.tier_li] = set(p[n_sel_ + n_ex_:n_sel_ + n_ex_ + n_pn_].tolist())
        TIER.begin_demand(m.tier_li, sel_h)        # this layer's NVMe reads first, then the next layer's prefetches
        nxt = getattr(m, "_next_moe", None)
        base = m.tier_li * TIER.E
        protect = [e + base for e in set(sel_h.ravel().tolist())]
        if TIER.core is not None:
            # substitution bookkeeping and prefetch filtering in C++ (tier_native.Core.after_sync)
            pf_next, pf_this, n_sub, n_pk = TIER.core.after_sync(m.tier_li, nxt.tier_li if nxt is not None else -1, p,
                                                                 Q, k, w1, MISS_PF)
            subst_stats["picks"] += n_pk
            subst_stats["substituted"] += n_sub
            if pf_next:
                TIER.prefetch(nxt.tier_li, np.array(pf_next, np.int64), protect=protect)
            if pf_this:
                TIER.prefetch(m.tier_li, np.array(pf_this, np.int64), protect=protect)   # for later tokens
            nxt2 = getattr(nxt, "_next_moe", None) if nxt is not None else None
            if PF_WIDTH2 and nxt2 is not None:
                TIER.prefetch(nxt2.tier_li, p[n_sel + n_ex + n_pn:], protect=protect)    # two layers ahead
        elif nxt is not None:
            o = n_sel + n_ex
            TIER.prefetch(nxt.tier_li, p[o:o + n_pn], protect=protect)
            nxt2 = getattr(nxt, "_next_moe", None)
            if PF_WIDTH2 and nxt2 is not None:
                TIER.prefetch(nxt2.tier_li, p[o + n_pn:], protect=protect)
        if TIER.core is None and (_subst_on(k) or Q > 1):
            misses = []
            for t in range(Q):
                chosen = set(sel_h[t].tolist())
                ranked = [e for e in ex_h[t].tolist() if e not in chosen]     # exact picks, best first
                subst_stats["picks"] += k
                subst_stats["substituted"] += len(ranked)
                misses += ranked[:MISS_PF] if MISS_PF >= 0 else ranked
                if ranked:
                    TIER.freq[np.array(ranked) + m.tier_li * TIER.E] += 1.0   # admission follows what the router wanted
            if misses:
                TIER.prefetch(m.tier_li, np.array(misses), protect=protect)   # for later tokens
        if nxt is not None and Q == 1 and TIER.sa_on:
            o = n_sel + n_ex
            # next layer's top predicted RAM experts -> VRAM, overlapped (the list is in rank order)
            TIER.stage_ahead(nxt.tier_li, p[o:o + min(n_pn, STAGE_TOPK)])
        if PROF:
            t1 = time.perf_counter()
            prof["prefetch_s"] += t1 - t0
        TIER.ensure(m.tier_li, sel_h)
        if PROF:
            prof["ensure_s"] += time.perf_counter() - t1
    if t_sync is not None:
        prof["pre_c_s"] += time.perf_counter() - t_sync    # GPU idle: sync return -> expert phase submitted
    _run(st, "C", -1, lambda: _C(self, st, R, params))
    if st.moe and TIER is not None and TIER.sa_on:
        TIER.stage_done(self.mlp.tier_li)
    if t_sync is not None:
        prof["gap_s"] += time.perf_counter() - t_sync      # sync return -> expert phase handed to the GPU
    return R


def install(tier):
    global TIER
    TIER = tier
    tier.upload_res = SUBST_KEEP < 8           # res_dev is read only by the legacy substitution router (_route_subst)
    if "block" not in _orig:
        _orig["block"] = TransformerBlock.forward
        TransformerBlock.forward = _forward


def uninstall():
    if "block" in _orig:
        TransformerBlock.forward = _orig.pop("block")
