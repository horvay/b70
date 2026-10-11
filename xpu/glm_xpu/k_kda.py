"""KDA (channelwise gated delta rule) recurrent step as a Triton kernel for Intel XPU (exllamav3
ext.cuda_recurrent_gated_delta_rule semantics, no history). One program per (row, head, BV columns of v), looping over the
row's tokens:
  q, k L2-normalised (eps 1e-6);  S = diag(exp g) S;  u = v - S^T k;  S += beta k u^T;  o = S^T q / sqrt(Dk)
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _kda_step_kernel(qkv_ptr, g_ptr, beta_ptr, state_ptr, out_ptr, slots_ptr, S, H, F, scale, NHIST,
                     DK: tl.constexpr, DV: tl.constexpr, BV: tl.constexpr, HAS_SLOTS: tl.constexpr,
                     HIST: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    vb = tl.program_id(2)
    slot = tl.load(slots_ptr + b).to(tl.int64) if HAS_SLOTS else b.to(tl.int64)
    kk = tl.arange(0, DK)
    vv = vb * BV + tl.arange(0, BV)
    st_ptr = state_ptr + ((slot * NHIST * H + h) * DK) * DV  # state [slot, j, h, k, v]; j = 0 current, 1.. history
    St = tl.load(st_ptr + kk[:, None] * DV + vv[None, :])  # (DK, BV) fp32
    for t in range(S):
        row = qkv_ptr + (b * S + t) * F
        q = tl.load(row + h * DK + kk).to(tl.float32)
        k = tl.load(row + (H + h) * DK + kk).to(tl.float32)
        v = tl.load(row + 2 * H * DK + h * DV + vv).to(tl.float32)
        q = q / tl.sqrt(tl.sum(q * q, axis=0) + 1e-6)
        k = k / tl.sqrt(tl.sum(k * k, axis=0) + 1e-6)
        a = tl.exp(tl.load(g_ptr + ((b * S + t) * H + h) * DK + kk))
        beta = tl.load(beta_ptr + (b * S + t) * H + h).to(tl.float32)
        St = St * a[:, None]
        u = v - tl.sum(St * k[:, None], axis=0)
        St = St + beta * k[:, None] * u[None, :]
        o = tl.sum(St * q[:, None], axis=0) * scale
        tl.store(out_ptr + ((b * S + t) * H + h) * DV + vv, o.to(out_ptr.dtype.element_ty))
        if HIST:
            if t < S - 1:                                  # state after token t -> history slot t + 1 (rewind source)
                tl.store(st_ptr + (t + 1) * H * DK * DV + kk[:, None] * DV + vv[None, :], St)
    tl.store(st_ptr + kk[:, None] * DV + vv[None, :], St)


def recurrent_kda(mixed_qkv, g, beta, recurrent_state, core_attn_out, H, Dk, Dv, slots, history=False):
    B, S, F = mixed_qkv.shape
    nhist = recurrent_state.shape[1]
    assert recurrent_state.is_contiguous() and (not history or nhist >= S)
    assert mixed_qkv.is_contiguous() and g.is_contiguous() and beta.is_contiguous() and core_attn_out.is_contiguous()
    BV = 32
    has = slots is not None
    sl = slots.to(device=mixed_qkv.device, dtype=torch.int32) if has else mixed_qkv
    _kda_step_kernel[(B, H, Dv // BV)](mixed_qkv, g, beta, recurrent_state, core_attn_out, sl, S, H, F, Dk ** -0.5,
                                       nhist, DK=Dk, DV=Dv, BV=BV, HAS_SLOTS=has, HIST=bool(history), num_warps=4)
