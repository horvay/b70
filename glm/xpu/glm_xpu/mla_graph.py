"""MLA (DSA) attention for one decode token with no host-side lengths, so the whole layer can be captured in an XPU
graph (block_graphs phase "Am").

exllamav3's eager decode (MLAttention._attend) branches on host lengths: the pooled-key update runs only when a pool
of index_kpool tokens completes, the indexer scores as many pools as the context holds, and below index_topk tokens
it switches to dense attention. Here every one of those reads the position from a static device tensor instead:
  - the current token's pool key is recomputed every step (an incomplete pool is invisible to the scorer, and it is
    rewritten with its final members on the step that completes it)
  - the indexer scores every pool slot the block table can address (fixed count) and masks those past the causal
    bound min((pos + 1) // P, ...) with -inf
  - the selection is a fixed-size stable sort (top index_topk // P pools, lowest index wins ties, as ext.dsa_topk),
    expanded to tokens, plus the incomplete tail pool's tokens
  - attention always runs the gathered (sparse) form over that fixed index list; with the whole context selected
    (context <= index_topk) it equals the dense path
Supported: GLM-5.3 shape (full indexers, index_kpool > 1, no rope dims, no Llama-4 scale, fp16 MLA cache, bsz 1).
"""
import torch

from exllamav3.modules.attention_fn.mla_triton import mla_absorb, mla_unfold, mla_kv_append, mla_plane_append

from glm_xpu.sparse_mla import sparse_mla_latent


QMAX = 8          # query rows per step: the decode token and up to QMAX - 1 MTP drafts


class Static:
    """Per-step device inputs shared by every MLA layer of one decode step (refreshed before the graphs replay)."""

    def __init__(self, dev, n_pages, page_size, k_pad):
        self.dev = dev
        self.page = page_size
        self.cs = torch.zeros((1,), dtype=torch.int32, device=dev)               # tokens before this one
        self.bt = torch.zeros((1, n_pages), dtype=torch.int32, device=dev)       # job block table, zero-padded
        self.indices = torch.full((QMAX, k_pad), -1, dtype=torch.int32, device=dev)   # per query row
        self.cache = None
        self.ok = False

    def refresh(self, params):
        """Copy this step's lengths and block table in; False when this forward cannot use the graphs."""
        from exllamav3.util.tensor import get_for_device
        self.ok = False
        cs = get_for_device(params, "cache_seqlens", self.dev, None)
        bt = get_for_device(params, "block_table", self.dev, None)
        if cs is None or bt is None or cs.numel() != 1 or bt.dim() != 2 or bt.shape[0] != 1 \
                or bt.shape[1] > self.bt.shape[1]:
            return False
        self.cs.copy_(cs.view(-1).to(torch.int32), non_blocking=True)
        if bt.shape[1] < self.bt.shape[1]:
            self.bt.zero_()
        self.bt[:, :bt.shape[1]].copy_(bt.to(torch.int32), non_blocking=True)
        self.cache = params.get("cache")
        self.ok = True
        return True


def supported(m):
    return (m.indexer_mode == "full" and m.index_kpool > 1 and m.qk_rope_head_dim == 0 and not m.l4_beta
            and m.q_a_proj is not None and m.idx_kpool_ape is not None and not m.has_split_cache)


def static_for(m, cache, dev):
    """Static buffers sized for `cache` (pages, page size) and the module's selection width."""
    layer = cache.layers[m.layer_idx, 0] if hasattr(cache, "layers") else cache
    n_pages, page = layer.k.shape[0], layer.k.shape[1]
    P = m.index_kpool
    k_pad = -(-((m.index_topk // P) * P + (P - 1 if m.index_kpool_tail else 0)) // 32) * 32
    return Static(dev, n_pages, page, k_pad)


def decode(m, x, params, S):
    """m: MLAttention; x (1, Q, hidden) fp16 static, Q = 1 (decode) or a few (MTP verify: the token and its drafts);
    S: Static (refreshed this step). Query t sits at position cs + t and sees the cache up to it. Returns (1, Q, hidden)."""
    layer = S.cache.layers[m.layer_idx, 0] if hasattr(S.cache, "layers") else S.cache
    Q = x.shape[1]
    H = m.num_q_heads
    P, D_i, H_i = m.index_kpool, m.index_head_dim, m.index_n_heads
    page = S.page
    epp_pool = page // P
    dev = x.device

    # queries and the new latent rows (as MLAttention._attend)
    q, q_resid = m.project_q(x, params, return_resid=True)
    q = q.view(Q, H, m.qk_head_dim)
    ckv_kpe = m.kv_a_proj_with_mqa.forward(x, params)
    ckv = m.kv_a_layernorm.forward(ckv_kpe[..., :m.kv_lora_rank].contiguous(), params, out_dtype=torch.half)
    k_pe = ckv_kpe[..., m.kv_lora_rank:].reshape(1, Q, 0)
    q_lat = mla_absorb(q, m.w_uk_flat, H, m.qk_nope_head_dim)
    mla_kv_append(ckv.reshape(1, Q, m.kv_lora_rank), k_pe, layer.k, layer.v, S.bt, S.cs)

    # indexer keys of these tokens ([k || pool gate] rows) into the per-token plane
    k_idx = m._indexer_keys(x, params, 0, None, None, None)
    mla_plane_append(k_idx.reshape(1, Q, layer.k_idx.shape[-1]), layer.k_idx, S.bt, S.cs)

    # pooled key of the pool holding each token, from its members' rows (members not written yet are masked by the
    # scorer's causal bound until the pool completes; a pool shared by two of the tokens is written twice, same value)
    pos = S.cs.long() + torch.arange(Q, device=dev)                       # (Q,)
    pool = torch.div(pos, P, rounding_mode="floor")
    tok = (pool.view(Q, 1) * P + torch.arange(P, device=dev).view(1, P)).view(-1)
    rows = S.bt[0].long()[torch.div(tok, page, rounding_mode="floor")] * page + tok % page
    members = layer.k_idx.view(-1, 2 * D_i).index_select(0, rows).float().view(Q, P, 2 * D_i)
    probs = torch.softmax(members[..., D_i:] + m.idx_kpool_ape.unsqueeze(0), dim=1)
    pool_key = (probs * members[..., :D_i]).sum(1).to(torch.half)        # (Q, D_i)
    pool32 = pool.to(torch.int32)
    for t in range(Q):
        mla_plane_append(pool_key[t].view(1, 1, D_i), layer.k_pool, S.bt, pool32[t:t + 1])

    # indexer scores over every pool slot the block table addresses, per-query causal bound from the device position
    n_pools = S.bt.shape[1] * epp_pool
    j = torch.arange(n_pools, device=dev)
    prow = S.bt[0].long()[torch.div(j, epp_pool, rounding_mode="floor")] * epp_pool + j % epp_pool
    kp = layer.k_pool.view(-1, D_i).index_select(0, prow)                  # (n_pools, D_i) fp16
    q_idx = m.idx_wq_b.forward(q_resid, params).view(Q * H_i, D_i)
    w = m.idx_weights.forward(x, params).view(Q, H_i)
    # fp32 dots and reduction like dsa_indexer_scores (fp16 dots flip near-ties in the top-k past index_topk tokens)
    sc = torch.relu(torch.matmul(q_idx.float(), kp.float().t()) * (D_i ** -0.5)).view(Q, H_i, n_pools)
    sc = torch.matmul((w.float() * (H_i ** -0.5)).view(Q, 1, H_i), sc).view(Q, n_pools)
    visible = torch.div(pos + 1, P, rounding_mode="floor").view(Q, 1)
    sc = torch.where(j.view(1, -1) < visible, sc, torch.full_like(sc, float("-inf"))).to(torch.half)

    # per query: top index_topk // P pools (stable: lowest index wins ties), expanded to tokens, plus its incomplete
    # tail pool
    k_sel = m.index_topk // P
    # one row at a time (multi-row XPU sort/topk touched the pinned host allocator during graph capture)
    srt = [torch.sort(sc[t], descending=True, stable=True) for t in range(Q)]
    vals = srt[0][0].view(1, -1) if Q == 1 else torch.stack([v for v, _ in srt])
    order = srt[0][1].view(1, -1) if Q == 1 else torch.stack([o for _, o in srt])
    top, ok = order[:, :k_sel], vals[:, :k_sel] > float("-inf")
    ar = torch.arange(P, device=dev)
    exp = (top.unsqueeze(-1) * P + ar).view(Q, -1)
    exp = torch.where(ok.unsqueeze(-1).expand(-1, -1, P).reshape(Q, -1), exp, torch.full_like(exp, -1))
    S.indices[:Q, :k_sel * P].copy_(exp.to(torch.int32))
    if m.index_kpool_tail:
        vis = (pos + 1).view(Q, 1)
        tail_count = vis % P
        toff = torch.arange(P - 1, device=dev).view(1, -1)
        tail = torch.where(toff < tail_count, vis - tail_count + toff, torch.full_like(toff, -1).expand(Q, -1))
        S.indices[:Q, k_sel * P:k_sel * P + P - 1].copy_(tail.to(torch.int32))

    from glm_xpu import sparse_mla as _sm
    if _sm.DEBUG_IDX:
        # the selection must stay inside each query's causal range and the pools / member rows inside the planes
        idx = S.indices[:Q].long().cpu()
        p_h = pos.cpu()
        for t in range(Q):
            v = idx[t][idx[t] >= 0]
            if v.numel() and int(v.max()) > int(p_h[t]):
                raise RuntimeError(f"mla_graph: row {t} selects token {int(v.max())} past its position {int(p_h[t])}")
        if int(rows.max()) >= layer.k_idx.view(-1, 2 * D_i).shape[0] or int(prow.max()) >= layer.k_pool.view(-1, D_i).shape[0]:
            raise RuntimeError("mla_graph: plane row out of range")
    o_lat = sparse_mla_latent(q_lat, layer.k, S.bt, S.indices[:Q], m.sm_scale)
    o = mla_unfold(o_lat, m.w_uv_flat, m.v_head_dim).reshape(1, Q, H * m.v_head_dim)
    return m.o_proj.forward(o, params)
