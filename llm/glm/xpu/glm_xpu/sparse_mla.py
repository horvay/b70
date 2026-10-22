"""Sparse (DSA-selected) MLA attention for Intel XPU with oneDNN batched GEMMs instead of exllamav3's Triton kernels.

For query row r and head h over its selected token positions idx[r, :] (-1 padded):
  s = (q_lat[h, r] . ckv[tok]) * scale,  p = softmax(s),  o_lat[h, r] = sum p * ckv[tok]
with tok = block_table[row, idx // page] * page + idx % page. NoPE (no rope part), latent output, fp16 cache only.
Rows are processed in chunks: gather the selected latent rows (chunk, K, 512), then two bmm's with a fp32 softmax.
"""
import os

import torch

CHUNK = int(os.environ.get("GLM_XPU_SPARSE_CHUNK", "128"))


def sparse_mla_latent(q_lat, ckv_cache, block_table, indices, scale, out=None):
    """q_lat (H, R, D) fp16 head-major; ckv_cache (pages, page, 1, D) fp16; block_table (1 or R, n_pages) int32;
    indices (R, K) int32 token positions (-1 = none). Returns o_lat (H, R, D) fp16 head-major."""
    H, R, D = q_lat.shape
    page = ckv_cache.shape[1]
    flat = ckv_cache.view(-1, D)
    if DEBUG_IDX:
        _check_idx("sparse_mla_latent", indices[:R], block_table, page, flat.shape[0])
    if out is None:
        out = torch.empty((H, R, D), dtype=q_lat.dtype, device=q_lat.device)
    shared_bt = block_table.shape[0] == 1
    for r0 in range(0, R, CHUNK):
        r1 = min(R, r0 + CHUNK)
        idx = indices[r0:r1].long()
        valid = idx >= 0
        idx0 = idx.clamp_min(0)
        bt = block_table.expand(r1 - r0, -1) if shared_bt else block_table[r0:r1]
        rows = bt.long().gather(1, idx0 // page) * page + idx0 % page          # (c, K)
        kv = flat.index_select(0, rows.view(-1)).view(r1 - r0, -1, D)          # (c, K, D)
        q = q_lat[:, r0:r1].transpose(0, 1)                                    # (c, H, D)
        s = torch.bmm(q, kv.transpose(1, 2)).float() * scale                   # (c, H, K)
        s = s.masked_fill(~valid[:, None, :], float("-inf"))
        p = torch.softmax(s, -1).nan_to_num_(0.0).to(q.dtype)
        o = torch.bmm(p, kv)                                                   # (c, H, D)
        out[:, r0:r1].copy_(o.transpose(0, 1))
    return out


DEBUG_IDX = os.environ.get("GLM_XPU_DEBUG_IDX", "0") == "1"


def _check_idx(where, indices, block_table, page, n_rows):
    """Debug (GLM_XPU_DEBUG_IDX=1, host sync): every gather index in range before the gather runs."""
    idx = indices.long().cpu()
    bt = block_table.long().cpu()
    valid = idx[idx >= 0]
    if (idx < -1).any():
        raise RuntimeError(f"{where}: index < -1: {idx.min().item()}")
    if valid.numel():
        mx = int(valid.max())
        if mx // page >= bt.shape[-1]:
            raise RuntimeError(f"{where}: token {mx} -> page column {mx // page} >= block table width {bt.shape[-1]}")
        pages = bt.view(-1, bt.shape[-1])[:, : mx // page + 1]
        if int(pages.max()) * page + page > n_rows or int(pages.min()) < 0:
            raise RuntimeError(f"{where}: page index {int(pages.max())} out of range ({n_rows // page} pages)")


PREFILL_HEAD_GROUP = int(os.environ.get("GLM_XPU_MLA_HEAD_GROUP", "16"))
PREFILL_Q_BLOCK = int(os.environ.get("GLM_XPU_MLA_Q_BLOCK", "512"))


def mla_prefill_mha_sdpa(q, w_uk_flat, w_uv_flat, ckv_cache, block_table, host_seqlens, bsz, q_len, v_head_dim,
                         qk_nope_head_dim, scale):
    """Dense MHA-form MLA prefill with torch's fused (oneDNN) SDPA: up-project every cached latent row of the
    sequence (past + this chunk, already appended) to per-head K/V, then causal attention aligned to the end
    (query i of the chunk sees cache rows <= past + i). q (R, H, D_nope) fp16 token-major; returns (R, H, D_v)."""
    import torch.nn.functional as F
    R, H, Dn = q.shape
    page = ckv_cache.shape[1]
    D = ckv_cache.shape[-1]
    pages_all = ckv_cache.view(-1, page, D)
    out = torch.empty((R, H, v_head_dim), dtype=q.dtype, device=q.device)
    for b in range(bsz):
        past = int(host_seqlens[b])
        total = past + q_len
        npg = -(-total // page)
        pg = block_table[b, :npg].long()
        ckv = pages_all.index_select(0, pg).view(-1, D)[:total]
        # Heads in groups (independent, so the result is the same): the up-projected per-head K/V of the whole
        # context (64 KB per token for all 64 heads, plus SDPA's contiguous copies) bounded the prompt chunk size.
        # With cached context (past > 0) the masked SDPA is not the fused kernel, so queries also go in blocks to
        # bound its (queries x context) scores
        # XPU's SDPA for this shape (head dim 256) materializes the (queries x keys) scores, so the causal case is
        # quadratic in the chunk too (183 MiB at 1K tokens, 1.3 GiB at 4K): queries always go in blocks
        G = max(1, min(H, PREFILL_HEAD_GROUP))
        QB = PREFILL_Q_BLOCK if PREFILL_Q_BLOCK > 0 else q_len
        for h0 in range(0, H, G):
            h1 = min(H, h0 + G)
            k = torch.matmul(ckv, w_uk_flat[:, h0 * Dn:h1 * Dn]).view(total, h1 - h0, Dn).transpose(0, 1).contiguous()
            v = torch.matmul(ckv, w_uv_flat[:, h0 * v_head_dim:h1 * v_head_dim]) \
                .view(total, h1 - h0, v_head_dim).transpose(0, 1).contiguous()
            for i0 in range(0, q_len, QB):
                i1 = min(q_len, i0 + QB)
                qb = q[b * q_len + i0:b * q_len + i1, h0:h1].transpose(0, 1)
                if past == 0 and i0 == 0:
                    o = F.scaled_dot_product_attention(qb[None], k[None, :, :i1], v[None, :, :i1], is_causal=True,
                                                       scale=scale)[0]
                else:
                    i = torch.arange(i0, i1, device=q.device).view(-1, 1)
                    j = torch.arange(past + i1, device=q.device).view(1, -1)
                    o = F.scaled_dot_product_attention(qb[None], k[None, :, :past + i1], v[None, :, :past + i1],
                                                       attn_mask=(j <= past + i), scale=scale)[0]
                out[b * q_len + i0:b * q_len + i1, h0:h1].copy_(o.transpose(0, 1))
                del o
            del k, v
    return out
