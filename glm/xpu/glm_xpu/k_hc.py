"""Fused mHC hyper-connection mix (exllamav3 ext.hc_mix semantics) as one Triton kernel per row, for Intel XPU.

Per row r (streams (R, 4, D) fp32, fn (24, 4D)):
  mix[m] = sum(s * fn[m]),  rmr = rsqrt(sum(s^2) / 4D + rms_eps)
  pre  = sigmoid(mix[0:4] * rmr * scale0 + base[0:4]) + hc_eps
  post = 2 sigmoid(mix[4:8] * rmr * scale1 + base[4:8])
  comb = sinkhorn(softmax_j(mix[8:24] * rmr * scale2 + base[8:24]) + hc_eps)   (1 column pass + (iters-1) row/col pairs)
  collapsed[d] = sum_h pre[h] * s[h, d]
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _hc_mix_kernel(s_ptr, fn_ptr, base_ptr, scale_ptr, post_ptr, comb_ptr, out_ptr, D, rms_eps, hc_eps,
                   ITERS: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    HD = 4 * D
    s_row = s_ptr + r * HD
    m = tl.arange(0, 32)
    mmask = m < 24
    acc = tl.zeros((32, BLOCK), dtype=tl.float32)
    ss = tl.zeros((BLOCK,), dtype=tl.float32)
    for c0 in range(0, HD, BLOCK):
        c = c0 + tl.arange(0, BLOCK)
        sv = tl.load(s_row + c)
        ss += sv * sv
        f = tl.load(fn_ptr + m[:, None] * HD + c[None, :], mask=mmask[:, None], other=0.0).to(tl.float32)
        acc += f * sv[None, :]
    mix = tl.sum(acc, axis=1)                                  # (32,)
    rmr = 1.0 / tl.sqrt(tl.sum(ss, axis=0) / HD + rms_eps)
    sidx = tl.where(m < 4, 0, tl.where(m < 8, 1, 2))
    sc = tl.load(scale_ptr + sidx, mask=mmask, other=0.0)
    bs = tl.load(base_ptr + m, mask=mmask, other=0.0)
    z = mix * rmr * sc + bs                                    # (32,)
    sig = 1.0 / (1.0 + tl.exp(-z))
    # post (m 4..7)
    tl.store(post_ptr + r * 4 + (m - 4), 2.0 * sig, mask=(m >= 4) & (m < 8))
    # comb (m 8..23) as a 4x4 matrix
    i4 = tl.arange(0, 4)
    zc = tl.reshape(tl.where((m >= 8) & (m < 24), z, 0.0), (8, 4))  # rows 2..5 hold comb rows 0..3
    rowsel = tl.arange(0, 8)
    L = tl.zeros((4, 4), dtype=tl.float32)
    for i in tl.static_range(4):
        rowv = tl.sum(tl.where(rowsel[:, None] == (2 + i), zc, 0.0), axis=0)     # (4,)
        L = tl.where(i4[:, None] == i, rowv[None, :], L)
    mx = tl.max(L, axis=1)
    e = tl.exp(L - mx[:, None])
    C = e / tl.sum(e, axis=1)[:, None] + hc_eps
    C = C / (tl.sum(C, axis=0)[None, :] + hc_eps)
    for _ in tl.static_range(ITERS - 1):
        C = C / (tl.sum(C, axis=1)[:, None] + hc_eps)
        C = C / (tl.sum(C, axis=0)[None, :] + hc_eps)
    tl.store(comb_ptr + r * 16 + i4[:, None] * 4 + i4[None, :], C)
    # pre (m 0..3) -> collapsed
    pre = tl.sum(tl.where(i4[:, None] == m[None, :], sig[None, :], 0.0), axis=1) + hc_eps    # (4,)
    for d0 in range(0, D, BLOCK):
        d = d0 + tl.arange(0, BLOCK)
        sv = tl.load(s_row + i4[:, None] * D + d[None, :])     # (4, BLOCK)
        col = tl.sum(sv * pre[:, None], axis=0)
        tl.store(out_ptr + r * D + d, col.to(out_ptr.dtype.element_ty))


@triton.jit
def _hc_mix_partial_kernel(s_ptr, fn_ptr, part_ptr, HD, CC, NC, BLOCK: tl.constexpr):
    """partials[r, c, 0:24] = fn[:, cols] . s[r, cols], partials[r, c, 24] = sum s[r, cols]^2 over chunk c's cols."""
    r = tl.program_id(0)
    ch = tl.program_id(1)
    m = tl.arange(0, 32)
    mmask = m < 24
    acc = tl.zeros((32, BLOCK), dtype=tl.float32)
    ss = tl.zeros((BLOCK,), dtype=tl.float32)
    for c0 in range(ch * CC, ch * CC + CC, BLOCK):
        c = c0 + tl.arange(0, BLOCK)
        cm = c < HD
        sv = tl.load(s_ptr + r * HD + c, mask=cm, other=0.0)
        ss += sv * sv
        f = tl.load(fn_ptr + m[:, None] * HD + c[None, :], mask=mmask[:, None] & cm[None, :], other=0.0).to(tl.float32)
        acc += f * sv[None, :]
    mix = tl.sum(acc, axis=1)
    out = tl.where(m == 24, tl.sum(ss, axis=0), mix)
    tl.store(part_ptr + (r * NC + ch) * 25 + m, out, mask=m < 25)


@triton.jit
def _hc_mix_final_kernel(s_ptr, part_ptr, base_ptr, scale_ptr, post_ptr, comb_ptr, out_ptr, D, NC, rms_eps, hc_eps,
                         ITERS: tl.constexpr, NCP: tl.constexpr, BLOCK: tl.constexpr):
    """Reduce the chunk partials, then as _hc_mix_kernel; program (r, b) writes collapsed[r, b-th BLOCK of D] (post
    and comb by b == 0)."""
    r = tl.program_id(0)
    db = tl.program_id(1)
    HD = 4 * D
    m = tl.arange(0, 32)
    mmask = m < 24
    ci = tl.arange(0, NCP)
    pt = tl.load(part_ptr + (r * NC + ci[:, None]) * 25 + m[None, :], mask=(ci[:, None] < NC) & (m[None, :] < 25),
                 other=0.0)
    tot = tl.sum(pt, axis=0)                                   # (32,): mix[0:24], ss at 24
    mix = tl.where(mmask, tot, 0.0)
    ssum = tl.sum(tl.where(m == 24, tot, 0.0), axis=0)
    rmr = 1.0 / tl.sqrt(ssum / HD + rms_eps)
    sidx = tl.where(m < 4, 0, tl.where(m < 8, 1, 2))
    sc = tl.load(scale_ptr + sidx, mask=mmask, other=0.0)
    bs = tl.load(base_ptr + m, mask=mmask, other=0.0)
    z = mix * rmr * sc + bs
    sig = 1.0 / (1.0 + tl.exp(-z))
    i4 = tl.arange(0, 4)
    if db == 0:
        tl.store(post_ptr + r * 4 + (m - 4), 2.0 * sig, mask=(m >= 4) & (m < 8))
        zc = tl.reshape(tl.where((m >= 8) & (m < 24), z, 0.0), (8, 4))
        rowsel = tl.arange(0, 8)
        L = tl.zeros((4, 4), dtype=tl.float32)
        for i in tl.static_range(4):
            rowv = tl.sum(tl.where(rowsel[:, None] == (2 + i), zc, 0.0), axis=0)
            L = tl.where(i4[:, None] == i, rowv[None, :], L)
        mx = tl.max(L, axis=1)
        e = tl.exp(L - mx[:, None])
        C = e / tl.sum(e, axis=1)[:, None] + hc_eps
        C = C / (tl.sum(C, axis=0)[None, :] + hc_eps)
        for _ in tl.static_range(ITERS - 1):
            C = C / (tl.sum(C, axis=1)[:, None] + hc_eps)
            C = C / (tl.sum(C, axis=0)[None, :] + hc_eps)
        tl.store(comb_ptr + r * 16 + i4[:, None] * 4 + i4[None, :], C)
    pre = tl.sum(tl.where(i4[:, None] == m[None, :], sig[None, :], 0.0), axis=1) + hc_eps
    d = db * BLOCK + tl.arange(0, BLOCK)
    sv = tl.load(s_ptr + r * HD + i4[:, None] * D + d[None, :])
    col = tl.sum(sv * pre[:, None], axis=0)
    tl.store(out_ptr + r * D + d, col.to(out_ptr.dtype.element_ty))


# glm-b70: one program per row left a decode step (R = 1) reading the 24 x 4D mixing matrix (1.5 MB) from a single
# work-group, ~170 us per call and two calls per block (~15 ms per token). Small R splits the reduction over the
# caller's partials buffer (R, chunks, 25) and finishes in a second kernel; GLM_XPU_HC_SPLIT=0 restores the old one
import os as _os
HC_SPLIT = _os.environ.get("GLM_XPU_HC_SPLIT", "1") == "1"


def hc_mix(streams, fn, base, scale, rms_eps, hc_eps, sinkhorn_iters, partials, post, comb, collapsed):
    R, H, D = streams.shape
    assert H == 4 and D % 256 == 0 and streams.is_contiguous() and fn.is_contiguous()
    if R == 0:
        return
    if HC_SPLIT and partials is not None and partials.dim() == 3 and partials.shape[2] == 25 \
            and partials.shape[0] >= R and partials.is_contiguous():
        NC = partials.shape[1]
        HD = 4 * D
        CC = -(-(-(-HD // NC)) // 256) * 256
        _hc_mix_partial_kernel[(R, NC)](streams, fn, partials, HD, CC, NC, BLOCK=256, num_warps=4)
        _hc_mix_final_kernel[(R, D // 512)](streams, partials, base, scale, post, comb, collapsed, D, NC,
                                            float(rms_eps), float(hc_eps), ITERS=int(sinkhorn_iters),
                                            NCP=triton.next_power_of_2(NC), BLOCK=512, num_warps=4)
        return
    _hc_mix_kernel[(R,)](streams, fn, base, scale, post, comb, collapsed, D, float(rms_eps), float(hc_eps),
                         ITERS=int(sinkhorn_iters), BLOCK=128, num_warps=4)


@triton.jit
def _hc_apply_kernel(x_ptr, y_ptr, post_ptr, comb_ptr, D, BLOCK: tl.constexpr):
    """x[r, h, :] = post[r, h] * y[r, :] + sum_g comb[r, g, h] * x[r, g, :]  (in place, fp32)."""
    r = tl.program_id(0)
    db = tl.program_id(1)
    d = db * BLOCK + tl.arange(0, BLOCK)
    i4 = tl.arange(0, 4)
    xs = tl.load(x_ptr + r * 4 * D + i4[:, None] * D + d[None, :])          # (4, BLOCK) [g, d]
    y = tl.load(y_ptr + r * D + d).to(tl.float32)
    post = tl.load(post_ptr + r * 4 + i4)                                  # (4,) [h]
    comb = tl.load(comb_ptr + r * 16 + i4[:, None] * 4 + i4[None, :])      # (4, 4) [g, h]
    for h in tl.static_range(4):
        ch = tl.sum(tl.where(i4[None, :] == h, comb, 0.0), axis=1)         # comb[:, h] -> (4,) over g
        ph = tl.sum(tl.where(i4 == h, post, 0.0), axis=0)
        out = ph * y + tl.sum(ch[:, None] * xs, axis=0)
        tl.store(x_ptr + r * 4 * D + h * D + d, out)


def hc_apply(x, y, post, comb):
    R, H, D = x.shape
    assert H == 4 and x.is_contiguous() and y.is_contiguous() and comb is not None
    BLOCK = 512
    _hc_apply_kernel[(R, D // BLOCK)](x, y, post.contiguous(), comb.contiguous(), D, BLOCK=BLOCK, num_warps=4)


@triton.jit
def _rms_norm_kernel(x_ptr, w_ptr, y_ptr, D, eps, cbias, cscale, HAS_W: tl.constexpr, CLAMP: tl.constexpr,
                     BLOCK: tl.constexpr):
    r = tl.program_id(0)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for c0 in range(0, D, BLOCK):
        c = c0 + tl.arange(0, BLOCK)
        m = c < D
        xv = tl.load(x_ptr + r * D + c, mask=m, other=0.0).to(tl.float32)
        if CLAMP:
            xv = tl.minimum(tl.maximum(xv, -65504.0), 65504.0)
        acc += xv * xv
    rmf = 1.0 / tl.sqrt(tl.sum(acc, axis=0) / D + eps) * cscale
    for c0 in range(0, D, BLOCK):
        c = c0 + tl.arange(0, BLOCK)
        m = c < D
        xv = tl.load(x_ptr + r * D + c, mask=m, other=0.0).to(tl.float32)
        if CLAMP:
            xv = tl.minimum(tl.maximum(xv, -65504.0), 65504.0)
        if HAS_W:
            xv = xv * (tl.load(w_ptr + c, mask=m, other=0.0).to(tl.float32) + cbias)
        tl.store(y_ptr + r * D + c, (xv * rmf).to(y_ptr.dtype.element_ty), mask=m)


def rms_norm(x, w, y, eps, cbias, cscale):
    D = x.shape[-1]
    R = x.numel() // D
    _rms_norm_kernel[(R,)](x, w if w is not None else x, y, D, float(eps), float(cbias), float(cscale),
                           HAS_W=w is not None, CLAMP=x.dtype == torch.half, BLOCK=1024, num_warps=4)
