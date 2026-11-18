"""Fused sigmoid / noaux_tc routing tail for the synced decode router (block_graphs._B2), one Triton program per row.

Replaces ~18 small kernels per MoE layer (sigmoid, bias, bonus add, three top-ks, gather, normalise, five picks copies)
with one launch after the two gate GEMVs (oneDNN). Per row r, from fp16 logits lc (this layer) and ln (next MoE layer):
  s    = sigmoid(lc),  key = s + esb_c
  exact = top-K(key)                      (best first; the router's wanted picks, for admission bookkeeping)
  sel   = top-K(key + bonus_c)            (VRAM / RAM residency bonus, tier.bonus_dev)
  w     = s[sel] * scale / (sum s[sel] + 1e-20)
  pred  = top-W1(sigmoid(ln) + esb_n + bonus_n)    (prefetch prediction for the next MoE layer)
written to sel [Q, K] int32, w [Q, K] fp32 and the picks buffer (block_graphs.picks_layout, int64):
  [sel Q*K | exact Q*K | pred Q*W1].
Top-k is K (or W1) rounds of argmax-and-mask over the E scores; ties go to the lower index (torch.topk's choice on
exact ties may differ; real router scores do not tie).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _route_kernel(lc_ptr, ln_ptr, esbc_ptr, esbn_ptr, bonc_ptr, bonn_ptr, sel_ptr, w_ptr, picks_ptr, E, Q, scale,
                  K: tl.constexpr, W1: tl.constexpr, BLOCK: tl.constexpr, HAS_ESB_C: tl.constexpr,
                  HAS_ESB_N: tl.constexpr, HAS_NEXT: tl.constexpr, HAS_BONUS: tl.constexpr):
    r = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    m = offs < E
    neg = float("-inf")
    lc = tl.load(lc_ptr + r * E + offs, mask=m, other=0.0).to(tl.float32)
    s = 1.0 / (1.0 + tl.exp(-lc))
    key = s
    if HAS_ESB_C:
        key = key + tl.load(esbc_ptr + offs, mask=m, other=0.0).to(tl.float32)
    key = tl.where(m, key, neg)

    # exact top-K, best first
    tk = tl.arange(0, K)
    kk = key
    for t in tl.static_range(K):
        i = tl.argmax(kk, axis=0)
        tl.store(picks_ptr + Q * K + r * K + t, i.to(tl.int64))
        kk = tl.where(offs == i, neg, kk)

    # selected top-K with the residency bonus, and the routing weights from the unbiased sigmoid
    kb = key
    if HAS_BONUS:
        kb = kb + tl.load(bonc_ptr + offs, mask=m, other=0.0)
        kb = tl.where(m, kb, neg)
    wv = tl.zeros((K,), dtype=tl.float32)
    for t in tl.static_range(K):
        i = tl.argmax(kb, axis=0)
        wi = tl.sum(tl.where(offs == i, s, 0.0), axis=0)
        wv = tl.where(tk == t, wi, wv)
        tl.store(sel_ptr + r * K + t, i.to(tl.int32))
        tl.store(picks_ptr + r * K + t, i.to(tl.int64))
        kb = tl.where(offs == i, neg, kb)
    wv = wv * scale / (tl.sum(wv, axis=0) + 1e-20)
    tl.store(w_ptr + r * K + tk, wv)

    # next MoE layer's prefetch prediction
    if HAS_NEXT:
        ln = tl.load(ln_ptr + r * E + offs, mask=m, other=0.0).to(tl.float32)
        kn = 1.0 / (1.0 + tl.exp(-ln))
        if HAS_ESB_N:
            kn = kn + tl.load(esbn_ptr + offs, mask=m, other=0.0).to(tl.float32)
        if HAS_BONUS:
            kn = kn + tl.load(bonn_ptr + offs, mask=m, other=0.0)
        kn = tl.where(m, kn, neg)
        for t in tl.static_range(W1):
            i = tl.argmax(kn, axis=0)
            tl.store(picks_ptr + 2 * Q * K + r * W1 + t, i.to(tl.int64))
            kn = tl.where(offs == i, neg, kn)


def route(lc, ln, esb_c, esb_n, bonus_c, bonus_n, scale, sel, w, picks, K, W1):
    """lc, ln: [Q, E] fp16 logits (ln None for the last MoE layer); esb_*: [E] or None; bonus_*: [E] fp32 or None.
    sel [Q, K] int32, w [Q, K] fp32, picks int64 (picks_layout(Q, K) with PF_WIDTH2 = 0) are written in place."""
    Q, E = lc.shape
    assert lc.is_contiguous() and (ln is None or ln.is_contiguous()) and K & (K - 1) == 0
    BLOCK = triton.next_power_of_2(E)
    _route_kernel[(Q,)](lc, ln if ln is not None else lc, esb_c if esb_c is not None else lc,
                        esb_n if esb_n is not None else lc, bonus_c if bonus_c is not None else lc,
                        bonus_n if bonus_n is not None else lc, sel, w, picks, E, Q, float(scale),
                        K=K, W1=W1, BLOCK=BLOCK, HAS_ESB_C=esb_c is not None, HAS_ESB_N=esb_n is not None,
                        HAS_NEXT=ln is not None, HAS_BONUS=bonus_c is not None, num_warps=4)
