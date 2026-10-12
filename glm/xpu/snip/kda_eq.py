# The KDA prefill memory changes keep the values: (a) forget-gate decay computed in place vs the old expression,
# (b) chunk_kda with q/k pre-normalized and in-place cumsum vs the old call, same inputs (strided views as in the model).
import torch
from exllamav3.modules.transformer import TransformerBlock
import exllamav3.vendor.fla as fla
m = [b.attn for b in model.modules if isinstance(b, TransformerBlock) and type(b.attn).__name__ == "GatedDeltaNet"][0]
dev = m.device
torch.manual_seed(0)
T, nv, hk = 777, m.num_v_heads, m.k_head_dim
f = (torch.randn(1, T, nv * hk, device=dev) * 4).half()
gf = (f.float() + m.dt_bias.float().view(1, 1, -1)).view(1, T, nv, hk)
decay = torch.exp(m.a_log.float()).view(1, 1, nv, 1)
old = m.gate_lower_bound * torch.sigmoid(decay * gf) if m.gate_lower_bound is not None else \
    -decay * torch.where(gf > 20.0, gf, torch.log1p(torch.exp(gf)))
g = f.float().add_(m.dt_bias.float().view(1, 1, -1)).view(1, T, nv, hk)
new = g.mul_(decay).sigmoid_().mul_(m.gate_lower_bound) if m.gate_lower_bound is not None else \
    torch.nn.functional.softplus(g, beta=1.0, threshold=20.0).mul_(-decay)
print(f"(a) gate: max abs diff {(old - new).abs().max().item():.3e}  (lower bound {m.gate_lower_bound})")
mixed = torch.randn(1, T, 3 * nv * hk, device=dev).bfloat16()
q, k, v = [t.view(1, T, nv, hk) for t in torch.split(mixed, [nv * hk] * 3, dim=-1)]
beta = torch.rand(1, T, nv, device=dev).bfloat16()
st0 = torch.randn(1, nv, hk, hk, device=dev) * 0.1
o1, s1 = fla.chunk_kda(q, k, v, g=old.clone(), beta=beta, initial_state=st0.clone(), output_final_state=True,
                       use_qk_l2norm_in_kernel=True)
from exllamav3.vendor.fla.l2norm import l2norm_fwd
g2 = old.clone()
o2, s2 = fla.chunk_kda(l2norm_fwd(q.contiguous())[0], l2norm_fwd(k.contiguous())[0], v, g=g2, beta=beta,
                       initial_state=st0.clone(), output_final_state=True, use_qk_l2norm_in_kernel=False, g_in_place=True)
rel = lambda a, b: ((a.float() - b.float()).norm() / a.float().norm()).item()
print(f"(b) chunk_kda: out rel diff {rel(o1, o2):.3e}, state rel diff {rel(s1, s2):.3e}, g overwritten {not torch.equal(g2, old)}")
