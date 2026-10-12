# Pieces of the no-sync routing phase (Bn) and the rest of phase C, each in its own graph, one MoE block.
import time, torch
from exllamav3.modules.transformer import TransformerBlock
from glm_xpu import block_graphs as bg
blk = [b for b in model.modules if isinstance(b, TransformerBlock) and getattr(b.mlp, "tier_li", -1) == 5][0]
st = blk._gstate; m = blk.mlp; cfg = m.routing_cfg; E, k, li = m.num_experts, m.num_experts_per_tok, m.tier_li
R = bg._resid[st.y_mlp.device] if hasattr(bg, "_resid") else None
P = {}
def bench(label, fn, n=300):
    fn(); torch.xpu.synchronize()
    g = torch.xpu.XPUGraph()
    with torch.xpu.graph(g):
        fn()
    g.replay(); torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        g.replay()
    torch.xpu.synchronize()
    us = (time.perf_counter() - t0) / n * 1e6
    print(f"  {label:34s} {us:7.1f} us", flush=True)
y = st.y_mlp.view(1, -1)
logits = torch.matmul(y, cfg.gate_tensor[:, :E]); sc = torch.sigmoid(logits.float())
key = sc + cfg.e_score_correction_bias.float()
bench("hc apply (attn)", lambda: blk.attn_hc.apply_(R, st.attn_out, st.post_a, st.comb_a, P))
bench("hc mix (mlp)", lambda: blk.mlp_hc.mix(R, P))
bench("mlp_norm", lambda: blk.mlp_norm.forward(st.y_mlp, P, out_dtype=torch.half))
bench("router matmul 4096x288", lambda: torch.matmul(y, cfg.gate_tensor[:, :E]))
bench("sigmoid + bias", lambda: torch.sigmoid(logits.float()) + cfg.e_score_correction_bias.float())
bench("topk 8 of 288", lambda: torch.topk(key, k, dim=-1, sorted=False))
bench("where + topk (resident)", lambda: torch.topk(torch.where(tier.ptrs[li] != 0, key.view(-1), float("-inf")).view(1, -1), k, dim=-1, sorted=False))
bench("gather + normalize", lambda: (lambda w: w * cfg.routed_scaling_factor / (w.sum(-1, keepdim=True) + 1e-20))(sc.gather(1, st.sel)))
bench("whole Bn (eager in graph)", lambda: bg._B_nosync(blk, st, R, P))
bench("shared expert", lambda: m.shared_experts.forward(st.y_mlp, P))
bench("hc apply (mlp)", lambda: blk.mlp_hc.apply_(R, st.y_mlp.view(1, 1, -1).float().half() if False else torch.zeros(1, 1, 4096, device=y.device, dtype=torch.float), st.post_f, st.comb_f, P))
