# LoRA check (devserver, adapter bound at /lora): every kind of target must give
#   out(with adapter) - out(without) == x @ (s B A)^T     (fp32 reference from the adapter file)
# then decode speed without and with it. Run on a fresh devserver (the first decode() is the baseline).
import torch
from glm_xpu import lora

D = "/lora"
P = "model.language_model.layers"
cfg, s, pairs = lora._peft_tensors(D)
if "LORA_AD" not in globals():
    tps, _, _ = decode(128)
    print(f"decode without LoRA: {tps:.2f} tok/s")
    LORA_AD = lora.apply(model, D, 1.0)
AD = LORA_AD
mods = {m.key: m for m in model}


def ref(path, x):
    A, B = pairs[path]["A"].float(), pairs[path]["B"].float() * s
    return x.float().cpu().view(-1, x.shape[-1]) @ A.t() @ B.t()


def delta(lin, x):
    with_ = lin.forward(x, {}).float()
    a, b = lin.lora_a_tensors.pop(AD), lin.lora_b_tensors.pop(AD)
    without = lin.forward(x, {}).float()
    lin.lora_a_tensors[AD], lin.lora_b_tensors[AD] = a, b
    return (with_ - without).cpu().view(-1, with_.shape[-1])


def report(name, got, want):
    got = got[:, :want.shape[1]]
    e = float((got - want).norm() / want.norm())
    print(f"  {'ok ' if e < 2e-2 else 'BAD'} {name}: rel err {e:.2e}, |delta| {float(want.norm()):.3g}")
    return e < 2e-2


torch.manual_seed(0)
ok = True
with torch.inference_mode():
    for path in [f"{P}.3.self_attn.q_a_proj", f"{P}.3.self_attn.q_b_proj", f"{P}.3.self_attn.kv_a_proj_with_mqa",
                 f"{P}.3.self_attn.o_proj", f"{P}.0.self_attn.o_proj", f"{P}.0.self_attn.f_a_proj",
                 f"{P}.0.self_attn.f_b_proj", f"{P}.0.self_attn.g_a_proj", f"{P}.0.self_attn.g_b_proj",
                 f"{P}.0.self_attn.b_proj", f"{P}.1.mlp.gate_proj", f"{P}.1.mlp.down_proj",
                 f"{P}.10.mlp.shared_experts.up_proj", f"{P}.44.mlp.shared_experts.down_proj"]:
        lin = mods[path]
        x = (torch.randn(1, 4, lin.in_features_unpadded) * 0.5).half().to(lin.device)
        ok &= report(path.split("layers.")[1], delta(lin, x), ref(path, x))
    # fused KDA q/k/v
    a = f"{P}.0.self_attn"
    lin = mods[a].qkv_proj
    x = (torch.randn(1, 4, lin.in_features_unpadded) * 0.5).half().to(lin.device)
    ok &= report("0.self_attn.{q,k,v}_proj (fused qkv_proj)", delta(lin, x),
                 torch.cat([ref(f"{a}.{c}_proj", x) for c in "qkv"], 1))
    # kv_b merged into the MLA flats: rebuild the checkpoint-layout weight both ways (the module's own inverse)
    m = mods[f"{P}.7.self_attn"]
    k = f"{P}.7.self_attn.kv_b_proj.weight"
    w1 = m.get_tensors()[k].float().cpu()
    mm = [e for e in AD.merged if e[0] is m][0]
    H, nope, v, D_c = m.num_q_heads, m.qk_nope_head_dim, m.v_head_dim, m.kv_lora_rank
    m.w_uk_flat.view(D_c, H, nope).sub_(mm[1]); m.w_uv_flat.view(D_c, H, v).sub_(mm[2])
    w0 = m.get_tensors()[k].float().cpu()
    m.w_uk_flat.view(D_c, H, nope).add_(mm[1]); m.w_uv_flat.view(D_c, H, v).add_(mm[2])
    pk = f"{P}.7.self_attn.kv_b_proj"
    ok &= report("7.self_attn.kv_b_proj (merged)", w1 - w0, pairs[pk]["B"].float() * s @ pairs[pk]["A"].float())
print("ALL OK" if ok else "SOME BAD")
tps, txt, _ = decode(128)
print(f"decode with LoRA: {tps:.2f} tok/s")
