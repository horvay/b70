"""Fused XPU kernels vs the torch reference ops (random data, seconds).

  in_rootfs.sh --gpu python3 /w/xpu/tests/kernels.py
"""
import sys
import time

import os
os.environ.setdefault("EXL3_XPU", "1")
from glm_xpu import compat

compat.install()
import torch

from glm_xpu import ops

dev = torch.device("xpu", 0)
torch.manual_seed(0)


def rel(a, b):
    return float((a.double() - b.double()).norm() / max(b.double().norm(), 1e-30))


def bench(f, n=50):
    f(); torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        f()
    torch.xpu.synchronize()
    return (time.perf_counter() - t0) / n * 1e6


which = sys.argv[1:] or ["hc"]
if "hc" in which:
    from glm_xpu import k_hc
    D = 4096
    fn32 = torch.randn(24, 4 * D, device=dev) * 0.02
    base = torch.randn(24, device=dev)
    scale = torch.rand(3, device=dev) + 0.5
    for R in (1, 4, 8, 64):
        st = torch.randn(R, 4, D, device=dev)
        for fn in (fn32, fn32.half()):
            outs = []
            for impl in (ops.hc_mix_torch, k_hc.hc_mix):
                post = torch.empty(R, 4, device=dev); comb = torch.empty(R, 4, 4, device=dev)
                col = torch.empty(R, D, dtype=torch.half, device=dev)
                impl(st, fn, base, scale, 1e-5, 1e-6, 20, None, post, comb, col)
                outs.append((post, comb, col))
            (p0, c0, x0), (p1, c1, x1) = outs
            us_t = bench(lambda: ops.hc_mix_torch(st, fn, base, scale, 1e-5, 1e-6, 20, None, p0, c0, x0))
            us_k = bench(lambda: k_hc.hc_mix(st, fn, base, scale, 1e-5, 1e-6, 20, None, p1, c1, x1))
            print(f"hc_mix R={R} fn={fn.dtype}: post {rel(p1, p0):.1e} comb {rel(c1, c0):.1e} col {rel(x1, x0):.1e}"
                  f"  torch {us_t:.0f} us  fused {us_k:.0f} us", flush=True)
            assert rel(p1, p0) < 1e-4 and rel(c1, c0) < 1e-4 and rel(x1, x0) < 1e-3
if "norm" in which:
    for R, D in ((1, 4096), (8, 4096), (300, 4096), (4, 1536), (4, 512)):
        for dt in (torch.half, torch.float):
            x = (torch.randn(R, D, device=dev) * 3).to(dt)
            w = torch.randn(D, device=dev).half()
            y0 = torch.empty(R, D, dtype=torch.half, device=dev); y1 = torch.empty_like(y0)
            ops.rms_norm_torch(x, w, y0, 1e-5, 0.0, 1.0, False, False)
            ops.rms_norm(x, w, y1, 1e-5, 0.0, 1.0, False, False)
            print(f"rms_norm R={R} D={D} {dt}: {rel(y1, y0):.1e}  torch {bench(lambda: ops.rms_norm_torch(x, w, y0, 1e-5, 0.0, 1.0, False, False)):.0f} us  "
                  f"fused {bench(lambda: ops.rms_norm(x, w, y1, 1e-5, 0.0, 1.0, False, False)):.0f} us", flush=True)
            assert rel(y1, y0) < 2e-3
    for R in (1, 8, 300):
        x0 = torch.randn(R, 4, 4096, device=dev); y = torch.randn(R, 4096, device=dev)
        post = torch.rand(R, 4, device=dev); comb = torch.rand(R, 4, 4, device=dev)
        a, b = x0.clone(), x0.clone()
        ops.hc_apply_torch(a, y, post, comb); ops.hc_apply(b, y, post, comb)
        print(f"hc_apply R={R}: {rel(b, a):.1e}  torch {bench(lambda: ops.hc_apply_torch(x0.clone(), y, post, comb)):.0f} us  "
              f"fused {bench(lambda: ops.hc_apply(x0.clone(), y, post, comb)):.0f} us", flush=True)
        assert rel(b, a) < 1e-5
if "dsa" in which:
    from glm_xpu.sparse_mla import sparse_mla_latent
    from exllamav3.modules.attention_fn.dsa_triton import dsa_attn
    H, D, page = 64, 512, 256
    for R, ctx in ((1, 4096), (4, 4096), (512, 4096), (2048, 16384)):
        pages = ctx // page
        ckv = (torch.randn(pages, page, 1, D, device=dev) * 0.5).half()
        kpe = torch.empty(pages, page, 1, 0, dtype=torch.half, device=dev)
        bt = torch.randperm(pages, device=dev).to(torch.int32).view(1, -1)
        K = 2080
        sel = torch.stack([torch.sort(torch.randperm(ctx, device=dev)[:2051]).values for _ in range(R)])
        idx = torch.full((R, K), -1, dtype=torch.int32, device=dev)
        idx[:, :2051] = sel.to(torch.int32)
        q = (torch.randn(H, R, D, device=dev) * 0.1).half()
        qpe = torch.empty(R, H, 0, dtype=torch.half, device=dev)
        sc = 1 / 16
        f_t = lambda: dsa_attn(q, ckv, kpe, bt, indices=idx, k_len=K, scale=sc, page_size=page, q_pe=qpe,
                               out_latent=True, qc=None, pool_len=ctx)
        f_x = lambda: sparse_mla_latent(q, ckv, bt, idx, sc)
        o_x = f_x()
        o_t = f_t()
        o_t = o_t.view(H, R, D) if o_t.numel() == H * R * D else o_t
        n = 20 if R < 100 else 3
        print(f"sparse MLA R={R} ctx={ctx}: rel {rel(o_x.float(), o_t.float()):.1e}  triton {bench(f_t, n) / 1e3:.2f} ms  "
              f"oneDNN {bench(f_x, n) / 1e3:.2f} ms", flush=True)
if "mha" in which:
    from glm_xpu.sparse_mla import mla_prefill_mha_sdpa
    from exllamav3.modules.attention_fn.mla_triton import mla_attn_triton_prefill_mha
    H, D, Dn, page = 64, 512, 256, 256
    w_uk = (torch.randn(D, H * Dn, device=dev) * 0.04).half()
    w_uv = (torch.randn(D, H * Dn, device=dev) * 0.04).half()
    for past, S in ((0, 512), (0, 2048), (1024, 1024)):
        T = past + S
        pages = -(-T // page) + 1
        ckv = (torch.randn(pages, page, 1, D, device=dev) * 0.5).half()
        kpe = torch.empty(pages, page, 1, 0, dtype=torch.half, device=dev)
        bt = torch.randperm(pages, device=dev).to(torch.int32).view(1, -1)
        q = (torch.randn(S, H, Dn, device=dev) * 0.1).half()
        f_t = lambda: mla_attn_triton_prefill_mha(q.clone(), w_uk, w_uv, ckv, kpe, bt, [past], 1, S, Dn, Dn, 1 / 16,
                                                  pre_appended_len=S)
        f_x = lambda: mla_prefill_mha_sdpa(q, w_uk, w_uv, ckv, bt, [past], 1, S, Dn, Dn, 1 / 16)
        o_t, o_x = f_t(), f_x()
        print(f"MLA prefill past={past} S={S}: rel {rel(o_x.float(), o_t.float()):.1e}  triton {bench(f_t, 3) / 1e3:.1f} ms  "
              f"sdpa {bench(f_x, 3) / 1e3:.1f} ms", flush=True)
if "kda" in which:
    H, Dk, Dv = 64, 128, 128
    F = 3 * H * Dk
    for B, S in ((1, 1), (4, 1), (1, 8)):
        qkv = torch.randn(B, S, F, device=dev).to(torch.bfloat16)
        g = (-5 * torch.sigmoid(torch.randn(B, S, H, Dk, device=dev))).contiguous()
        beta = torch.sigmoid(torch.randn(B, S, H, device=dev)).to(torch.bfloat16)
        st0 = torch.randn(B + 2, 1, H, Dk, Dv, device=dev) * 0.1
        slots = torch.tensor(list(range(B, 0, -1)), dtype=torch.int32, device=dev)
        outs = []
        for impl in (ops.cuda_recurrent_gated_delta_rule_torch, ops.cuda_recurrent_gated_delta_rule):
            st = st0.clone()
            o = torch.empty(B, S, H, Dv, dtype=torch.bfloat16, device=dev)
            impl(qkv, g, beta, st, o, H, H, Dk, Dv, slots, False)
            outs.append((st, o))
        (s0, o0), (s1, o1) = outs
        us_t = bench(lambda: ops.cuda_recurrent_gated_delta_rule_torch(qkv, g, beta, s0, o0, H, H, Dk, Dv, slots, False), 20)
        us_k = bench(lambda: ops.cuda_recurrent_gated_delta_rule(qkv, g, beta, s1, o1, H, H, Dk, Dv, slots, False), 20)
        print(f"kda B={B} S={S}: state {rel(s1, s0):.1e} out {rel(o1, o0):.1e}  torch {us_t:.0f} us  fused {us_k:.0f} us",
              flush=True)
        assert rel(s1, s0) < 1e-4 and rel(o1, o0) < 1e-2
print("KERNELS OK")
