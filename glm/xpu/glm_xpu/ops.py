"""exllamav3_ext ops implemented for Intel XPU (re-exported by the exllamav3_ext shim).

Semantics follow docs/ext-op-inventory.md (the exllamav3 v1.5.1 CUDA kernels). EXL3 linears run on exl3xpu's ESIMD
kernels (same trellis layout, bit-exact mul1 decode); everything else is torch. Ops write their preallocated outputs
in place, as the CUDA ops do, because callers read the output tensors rather than return values.
"""
import math
import os

import torch
import torch.nn.functional as F

# ----------------------------------------------------------------------------------------------- constants

FUSED_SAMPLER_MAX_BLOCKS = 128
FUSED_SAMPLER_HIST_STRIDE = 2 * 1024 * 12 + 64

# ----------------------------------------------------------------------------------------------- EXL3 linear

_esimd = None
SMALL_M_MAX = int(os.environ.get("EXL3_SMALL_M_MAX", "128"))
RECON_SLICE_N = int(os.environ.get("EXL3_RECON_SLICE_N", "16384"))


def _E():
    global _esimd
    if _esimd is None:
        here = os.path.dirname(os.path.abspath(__file__))
        torch.ops.load_library(os.environ.get("EXL3_LIB") or os.path.join(here, "..", "exl3xpu", "_C.so"))
        _esimd = torch.ops.exl3xpu_C
    return _esimd


class BC_LinearEXL3:
    """EXL3 linear: y = Had(Had(x * suh) @ W_hat) * svh, on exl3xpu ESIMD kernels (mul1 codebook, cb = 2)."""

    def __init__(self, trellis, suh, svh, K, bias, mcg, mul1, xh):
        self.trellis = trellis
        self.suh = suh
        self.svh = svh
        self.K = K
        self.bias = bias
        self.mcg = mcg
        self.mul1 = mul1
        self.xh = xh
        assert mul1 and not mcg, "XPU port: only the mul1 codebook is supported"
        assert float(K) == int(K), "XPU port: half-integer bitrates are not supported"
        self.Ki = int(K)
        self.cb = 2
        self.in_features = trellis.shape[0] * 16
        self.out_features = trellis.shape[1] * 16
        self._suh2 = None
        self._sonb = None

    def _prep(self):
        if self._suh2 is None or self._suh2.device != self.trellis.device:
            self._suh2 = self.suh.view(1, -1).contiguous()
            self._sonb = torch.zeros(self.out_features // 128, dtype=torch.int32, device=self.trellis.device)

    def run(self, x, y):
        """x fp16 [..., k] -> y (fp16 or fp32, preallocated) [..., n]."""
        self._prep()
        E = _E()
        x2 = x.reshape(-1, self.in_features)
        if x2.dtype != torch.half:
            x2 = x2.half()
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        y2 = y.view(-1, self.out_features)
        M = x2.shape[0]
        if M == 0:
            return
        if M <= SMALL_M_MAX and y2.is_contiguous():
            E.exl3_gemm_small(x2, self.trellis, self._suh2, self.svh, self._sonb, y2, self.Ki, self.cb)
        else:
            out = E.linear(x2, self.trellis, self._suh2, self.svh, self._sonb, [0, self.out_features], self.Ki,
                           self.cb, SMALL_M_MAX, RECON_SLICE_N)
            y2.copy_(out)
        if self.bias is not None:
            y2.add_(self.bias)

    def run_alloc(self, x, out_features, output_fp32):
        y = torch.empty(x.shape[:-1] + (out_features,), dtype=torch.float if output_fp32 else torch.half,
                        device=x.device)
        if out_features == 0:
            return y
        self.run(x.reshape(-1, x.shape[-1]), y.view(-1, out_features))
        return y


class BC_LinearFP16:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    def run(self, x, y):
        if x.dtype == y.dtype:
            torch.matmul(x, self.weight, out=y)
        else:
            hgemm(x, self.weight, y)
        if self.bias is not None:
            y.add_(self.bias)


def hgemm(a, b, c):
    """c = a @ b with fp32 accumulation, written into c (c may be a strided column slice)."""
    if a.dtype == b.dtype:
        # oneDNN accumulates fp16 GEMMs in fp32; only the output is rounded to fp16 before the widening copy
        r = torch.matmul(a, b)
    else:
        r = torch.matmul(a.float(), b.float())
    c.copy_(r.view(c.shape))


hgemm_recon = hgemm


_H128 = {}


def _h128(dev):
    H = _H128.get(dev)
    if H is None:
        h = torch.ones(1, 1)
        while h.shape[0] < 128:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        H = (h / math.sqrt(128)).to(dev)
        _H128[dev] = H
    return H


def had_r_128(input, output, pre_scale, post_scale, scale):
    """Blockwise 128-point Hadamard (natural Sylvester order), * scale / sqrt(128); pre_scale wins over post_scale."""
    x = input
    if pre_scale is not None:
        x = x * pre_scale
    H = _h128(x.device)
    y = (x.float().reshape(-1, 128) @ H).reshape(x.shape) * scale
    y = y.to(output.dtype)
    if pre_scale is None and post_scale is not None:
        y = y * post_scale
    output.copy_(y)


# ----------------------------------------------------------------------------------------------- activations / MLP

def silu_mul(x, y, z, act_limit):
    """z = min(silu(x), L) * clamp(y, -L, L) (L = act_limit, 0 disables), computed in fp32, stored to z's dtype."""
    s = F.silu(x.float())
    u = y.float()
    if act_limit:
        s = s.clamp_max(act_limit)
        u = u.clamp(-act_limit, act_limit)
    z.copy_(s * u)


class BC_GatedMLP:
    """Dense / shared-expert gated MLP for rows <= 8: g, u = gate(x), up(x); a = silu_mul(g, u); d = down(a)."""

    def __init__(self, guh, gu, a, down_xh, gu_ptrs_trellis, gu_ptrs_suh, gu_ptrs_svh, gu_K, gu_mcg, gu_mul1,
                 act_silu, act_gelu, act_relu2, gate, up, down, act_limit):
        assert gate is not None and up is not None, "XPU port: fused gate/up pointer tables are not supported"
        assert act_silu, "XPU port: only SiLU gated MLPs"
        self.gate, self.up, self.down = gate, up, down
        self.act_limit = act_limit

    def run_bszN(self, x, d):
        x2 = x.reshape(-1, x.shape[-1])
        g = self.gate.run_alloc(x2, self.gate.out_features, False)
        u = self.up.run_alloc(x2, self.up.out_features, False)
        silu_mul(g, u, u, self.act_limit)
        self.down.run(u, d.view(-1, d.shape[-1]))


# ----------------------------------------------------------------------------------------------- routing

def routing_ds3_nogroup(hidden, gate, scores, bias, topk_indices, topk_weights, scaling_factor, gate_t, act_fn,
                        gate_i8, gate_sb):
    """Sigmoid / noaux_tc top-k routing (no groups). Selection by sigmoid + bias, weights from the un-biased sigmoid,
    normalised over the selected and scaled. Writes scores, topk_indices, topk_weights in place."""
    assert act_fn == 0, "XPU port: sigmoid routing only"
    E = scores.shape[-1]
    k = topk_indices.shape[-1]
    logits = torch.matmul(hidden, gate[:, :E]) if hidden.dtype == gate.dtype == torch.half else \
        torch.matmul(hidden.float(), gate[:, :E].float()).half()
    scores.copy_(logits)
    s = torch.sigmoid(logits.float())
    key = s + bias.float() if bias is not None else logits.float()
    idx = torch.topk(key, k, dim=-1, sorted=True).indices
    w = s.gather(-1, idx)
    w = w * scaling_factor / (w.sum(-1, keepdim=True) + 1e-20)
    topk_indices.copy_(idx)
    topk_weights.copy_(w)


def det_quant_weight(w, w_i8, w_s):
    pass        # only feeds the CUDA deterministic routing GEMM


# ----------------------------------------------------------------------------------------------- hyper-connections / norms

def hc_mix_num_chunks(R, row_len):
    chunks_a = min(128, max(1, 512 // max(R, 1)))
    chunk_cols = -(-(row_len // chunks_a) // 256) * 256
    return -(-row_len // chunk_cols)


HC_FUSED_MAX_R = int(os.environ.get("GLM_XPU_HC_FUSED_MAX_R", "64"))


def hc_mix(streams, fn, base, scale, rms_eps, hc_eps, sinkhorn_iters, partials, post, comb, collapsed):
    R, H, D = streams.shape
    if R <= HC_FUSED_MAX_R and H == 4 and streams.device.type == "xpu":
        from glm_xpu import k_hc
        return k_hc.hc_mix(streams, fn, base, scale, rms_eps, hc_eps, sinkhorn_iters, partials, post, comb, collapsed)
    hc_mix_torch(streams, fn, base, scale, rms_eps, hc_eps, sinkhorn_iters, partials, post, comb, collapsed)


def hc_mix_torch(streams, fn, base, scale, rms_eps, hc_eps, sinkhorn_iters, partials, post, comb, collapsed):
    R, H, D = streams.shape
    s = streams.reshape(R, H * D)
    mix = s @ fn.float().t()
    rmr = torch.rsqrt(s.pow(2).sum(-1, keepdim=True) / (H * D) + rms_eps)
    mix = mix * rmr
    pre = torch.sigmoid(mix[:, :H] * scale[0] + base[:H]) + hc_eps
    post.copy_(2 * torch.sigmoid(mix[:, H:2 * H] * scale[1] + base[H:2 * H]))
    c = (mix[:, 2 * H:] * scale[2] + base[2 * H:]).view(R, H, H)
    c = torch.softmax(c, -1) + hc_eps
    c = c / (c.sum(-2, keepdim=True) + hc_eps)
    for _ in range(sinkhorn_iters - 1):
        c = c / (c.sum(-1, keepdim=True) + hc_eps)
        c = c / (c.sum(-2, keepdim=True) + hc_eps)
    comb.copy_(c)
    collapsed.copy_((pre.unsqueeze(-1) * streams).sum(1))


def hc_apply(x, y, post, comb):
    if comb is not None and x.device.type == "xpu" and x.shape[1] == 4 and x.shape[2] % 512 == 0 \
            and x.is_contiguous() and y.is_contiguous():
        from glm_xpu import k_hc
        return k_hc.hc_apply(x, y, post, comb)
    hc_apply_torch(x, y, post, comb)


def hc_apply_torch(x, y, post, comb):
    yf = y.float().unsqueeze(1)
    if comb is None:
        x.add_(post.unsqueeze(-1) * yf)
    else:
        x.copy_(post.unsqueeze(-1) * yf + torch.matmul(comb.transpose(-1, -2), x))


def rms_norm(x, w, y, epsilon, constant_bias, constant_scale, span_heads, add_residual, w_groups=1):
    if (not span_heads and not add_residual and w_groups == 1 and x.device.type == "xpu" and x.is_contiguous()
            and y.is_contiguous() and y.dtype in (torch.half, torch.float)):
        from glm_xpu import k_hc
        return k_hc.rms_norm(x, w, y, epsilon, constant_bias, constant_scale)
    rms_norm_torch(x, w, y, epsilon, constant_bias, constant_scale, span_heads, add_residual, w_groups)


def rms_norm_res_in(x, w, y, r, epsilon, constant_bias, constant_scale):
    """Fused pre-norm residual (CUDA norm.cu RES_IN): r += x in place, then y = rms_norm(r) * w."""
    r.add_(x.to(r.dtype))
    rms_norm(r, w, y, epsilon, constant_bias, constant_scale, False, False)


def rms_norm_torch(x, w, y, epsilon, constant_bias, constant_scale, span_heads, add_residual, w_groups=1):
    xs, ys = (x.flatten(-2), y.flatten(-2)) if span_heads else (x, y)
    xf = xs.float()
    if xs.dtype == torch.half:
        xf = xf.clamp(-65504, 65504)
    rmf = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + epsilon) * constant_scale
    if w is not None:
        wf = w.float()
        if w_groups > 1:
            rows = xf.reshape(-1, xf.shape[-1]).shape[0]
            wf = wf.view(w_groups, -1).repeat(rows // w_groups, 1).view(xf.shape)
        xf = xf * (wf + constant_bias)
    out = xf * rmf
    if add_residual:
        out = out + ys.float()
    ys.copy_(out)


def gated_rms_norm(x, w, y, g, epsilon, constant_bias, w_groups, gate_first, gate_act):
    assert not gate_first and w_groups == 1, "XPU port: gated_rms_norm variant not implemented"
    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + epsilon)
    h = h * (w.float() + constant_bias)
    gf = g.float().view(h.shape)
    h = h * (torch.sigmoid(gf) if gate_act == 1 else F.silu(gf))
    y.copy_(h.view(y.shape))


class BC_GatedRMSNorm:
    def __init__(self, weight, rms_norm_eps, constant_bias, w_groups=1, gate_first=False, gate_act=0):
        self.weight, self.eps, self.constant_bias = weight, rms_norm_eps, constant_bias
        self.w_groups, self.gate_first, self.gate_act = w_groups, gate_first, gate_act

    def run(self, x, y, gate):
        gated_rms_norm(x, self.weight, y, gate, self.eps, self.constant_bias, self.w_groups, self.gate_first,
                       self.gate_act)


# ----------------------------------------------------------------------------------------------- KDA

def cuda_recurrent_gated_delta_rule(mixed_qkv, g, beta, recurrent_state, core_attn_out, num_k_heads, num_v_heads,
                                    k_head_dim, v_head_dim, slots, history=False):
    S = mixed_qkv.shape[1]
    if (g.dim() == 4 and (recurrent_state.shape[1] == 1 or history) and mixed_qkv.device.type == "xpu"
            and (not history or recurrent_state.shape[1] >= S) and recurrent_state.is_contiguous()
            and num_k_heads == num_v_heads and v_head_dim % 32 == 0 and os.environ.get("GLM_XPU_KDA_TORCH") != "1"):
        # history (MTP verify): the Triton step also writes the per-token states the rewind reads (was a torch loop)
        from glm_xpu import k_kda
        return k_kda.recurrent_kda(mixed_qkv.contiguous(), g.float().contiguous(), beta.contiguous(), recurrent_state,
                                   core_attn_out, num_k_heads, k_head_dim, v_head_dim, slots, history)
    cuda_recurrent_gated_delta_rule_torch(mixed_qkv, g, beta, recurrent_state, core_attn_out, num_k_heads, num_v_heads,
                                          k_head_dim, v_head_dim, slots, history)


def cuda_recurrent_gated_delta_rule_torch(mixed_qkv, g, beta, recurrent_state, core_attn_out, num_k_heads, num_v_heads,
                                          k_head_dim, v_head_dim, slots, history=False):
    """Channelwise (KDA) gated delta rule, token-sequential. State [slots, hist+1, Hv, Dk, Dv] fp32, in place."""
    assert g.dim() == 4, "XPU port: only the channelwise (KDA) variant"
    assert num_k_heads == num_v_heads
    B, S, _ = mixed_qkv.shape
    H, Dk, Dv = num_k_heads, k_head_dim, v_head_dim
    q, k, v = mixed_qkv.float().split([H * Dk, H * Dk, H * Dv], -1)
    q = q.reshape(B, S, H, Dk)
    k = k.reshape(B, S, H, Dk)
    v = v.reshape(B, S, H, Dv)
    q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    a = g.float().exp()
    bt = beta.float()
    idx = slots.long() if slots is not None else torch.arange(B, device=recurrent_state.device)
    if idx.device != recurrent_state.device:
        idx = idx.to(recurrent_state.device)
    scale = Dk ** -0.5
    St = recurrent_state[idx, 0]                                    # [B, H, Dk, Dv]
    for t in range(S):
        St = St * a[:, t].unsqueeze(-1)
        u = v[:, t] - torch.einsum("bhk,bhkv->bhv", k[:, t], St)
        St = St + bt[:, t, :, None, None] * k[:, t].unsqueeze(-1) * u.unsqueeze(-2)
        o = torch.einsum("bhk,bhkv->bhv", q[:, t], St) * scale
        core_attn_out[:, t] = o.to(core_attn_out.dtype)
        if history and t < S - 1:
            recurrent_state[idx, t + 1] = St
    recurrent_state[idx, 0] = St


# ----------------------------------------------------------------------------------------------- MLA indexer

def dsa_topk(scores, indices, k, t_ptr=None, t_seq=0):
    """k largest finite fp16 scores per row, lowest index wins ties, emitted in ascending index order, -1 padded."""
    assert t_ptr is None
    R, T = scores.shape
    s = scores.float()
    k_eff = min(k, T)
    v, i = torch.sort(s, dim=1, descending=True, stable=True)
    v, i = v[:, :k_eff], i[:, :k_eff]
    bad = (v == float("-inf")) | torch.isnan(v)
    i = torch.where(bad, torch.full_like(i, T + 1), i)
    i, _ = torch.sort(i, dim=1)
    i = torch.where(i > T, torch.full_like(i, -1), i)
    indices.fill_(-1)
    indices[:, :k_eff].copy_(i.to(indices.dtype))


# ----------------------------------------------------------------------------------------------- sampling

def _gumbel_like(x, seed):
    gen = torch.Generator(device=x.device)
    gen.manual_seed(int(seed) & 0xFFFFFFFFFFFFFFFF)
    u = torch.rand(x.shape, generator=gen, device=x.device, dtype=torch.float32)
    u = u.clamp(1e-20, 0.99999994)
    return -torch.log((-torch.log(u)).clamp_min(1e-20))


def fused_sampler(logits, logit_mask, logit_bitmask, out, workspace, size, inv_temp, minp_log, random, mode, filters,
                  top_k, top_p, inv_temp_filter, histogram):
    """Modes: 0 greedy, 1 Gumbel, 2 Gumbel + min-P, 3 Gumbel + filters (1 top-K, 2 top-P, 4 min-P)."""
    x = logits[:, :size].float()
    if logit_mask is not None:
        x = x + logit_mask[:, :size].float()
    if logit_bitmask is not None:
        words = logit_bitmask.to(torch.int64) & 0xFFFFFFFF
        bit = torch.arange(size, device=x.device)
        keep = ((words[:, bit // 32] >> (bit % 32)) & 1).bool()
        x = x.masked_fill(~keep, float("-inf"))
    x = torch.nan_to_num(x, nan=float("-inf"), posinf=float("inf"), neginf=float("-inf"))
    if mode == 0:
        out.copy_(x.argmax(-1, keepdim=True))
        return
    keep = x > float("-inf")
    M = x.max(-1, keepdim=True).values
    if mode == 2 or (mode == 3 and filters & 4):
        keep &= x >= M + minp_log
    if mode == 3 and filters & 3:
        xs = x.masked_fill(~keep, float("-inf"))
        v, order = torch.sort(xs, dim=-1, descending=True, stable=True)
        kept = v > float("-inf")
        if filters & 1 and top_k > 0:
            kk = min(top_k, v.shape[-1])
            kept &= v >= v[:, kk - 1:kk]
        if filters & 2:
            w = torch.exp((v - M) * inv_temp_filter) * kept
            prev = torch.cumsum(w, -1) - w
            kept &= prev <= top_p * w.sum(-1, keepdim=True)
            kept[:, 0] = True
        k2 = torch.zeros_like(keep)
        k2.scatter_(-1, order, kept)
        keep &= k2
    v = torch.where(keep, x * inv_temp + _gumbel_like(x, random), torch.full_like(x, float("-inf")))
    out.copy_(v.argmax(-1, keepdim=True))


def partial_strings_match(match, offsets, strings):
    q = bytes(match).decode("utf-32-le")
    offs = memoryview(bytes(offsets)).cast("i").tolist()
    sb = bytes(strings)
    needles = [sb[offs[i]:offs[i + 1]].decode("utf-32-le") for i in range(len(offs) - 1)]
    best_full = best_partial = len(q) + 1
    for nd in needles:
        for a0 in range(min(best_full, best_partial, len(q))):
            n = 0
            while n < len(nd) and a0 + n < len(q) and q[a0 + n] == nd[n]:
                n += 1
            if n == len(nd):
                best_full = a0
                break
            if a0 + n == len(q):
                best_partial = a0
                break
    if best_partial < best_full:
        return -2
    return best_full if best_full <= len(q) else -1


def cache_rotate(cache, order, temp):
    flat = cache.view(cache.shape[0], -1)
    t = temp.view(-1)
    if flat.shape[1] == 0:
        return
    o = order.cpu().tolist()
    for i in range(0, len(o), 2):
        a, b = o[i], o[i + 1]
        (flat[a] if a >= 0 else t).copy_(flat[b] if b >= 0 else t)


# ----------------------------------------------------------------------------------------------- B70 MoE kernel library

_moe = None


def s64(p: int) -> int:
    """Device address as a signed int64 (XPU USM addresses can be >= 2**63)."""
    return p - (1 << 64) if p >= (1 << 63) else p


def moe_ops():
    """torch.ops.exl3xpu_moe from exl3xpu/_moe_glm.so (grouped MoE over pointer tables, GLM clamp)."""
    global _moe
    if _moe is None:
        here = os.path.dirname(os.path.abspath(__file__))
        torch.ops.load_library(os.environ.get("GLM_XPU_MOE_LIB") or os.path.join(here, "..", "exl3xpu", "_moe_glm.so"))
        _moe = torch.ops.exl3xpu_moe
        # GLM shapes (H 4096, I 2048) need power-of-two k-splits: gate|up 8, down 4 (glm-flash-lite b70srv defaults)
        _moe.moe_set_splits(int(os.environ.get("GLM_XPU_MOE_GU", "8")), int(os.environ.get("GLM_XPU_MOE_DN", "4")))
        # M >= 2 (MTP verify) grouped: pair mode reads an expert once per (token, pick), twice when both tokens pick it,
        # and RAM experts cross PCIe twice (M=2, 3 RAM experts: pair 1508 us, grouped 1115 us per layer)
        _moe.moe_set_pair_max_m(int(os.environ.get("GLM_XPU_MOE_PAIR_MAX_M", "1")))
        # DPAS prefill path from M >= 256 (the kernel's measured crossover). It hung the B70 on GLM shapes until the
        # gate/up work-group was split to <= 32 threads for large-GRF mode (2026-10-10, csrc/exl3_moe.n128.sycl).
        _moe.moe_set_prefill_min_m(int(os.environ.get("GLM_XPU_MOE_PREFILL_MIN_M", "256")))
        # K values with the DPAS prefill path (bit K). K=2 / K=4 (build4_any) added 2026-10-10 for the 2.51 bpw model;
        # GLM_XPU_MOE_PREFILL_K_MASK=8 keeps them on the grouped GEMV (K=3 only)
        _moe.moe_set_prefill_k_mask(int(os.environ.get("GLM_XPU_MOE_PREFILL_K_MASK", str(4 | 8 | 16))))
    return _moe
