"""Apply a PEFT LoRA adapter (e.g. cgg507/GLM-5.3-Flash-GM-LoRA) to the loaded GLM-5.3-Flash.

  from glm_xpu import lora
  lora.apply(model, "/lora", scale=1.0)        # after runner.load(), before the first decode graph is captured

Quantized linears get exllamav3's runtime LoRA (Linear.lora_a_tensors / lora_b_tensors: out += x @ A @ B). Three
targets need more:
  - KDA q/k/v_proj are one fused qkv_proj here: their A matrices are stacked along the rank and B is block-diagonal.
  - kv_b_proj is not a Linear (it lives as the w_uk / w_uv flats the MLA kernels read); the delta is merged into them.
  - GatedMLP's fused decode paths (bc, multi_gu) skip Linear.forward, so they are turned off where an adapter applies.
`scale` multiplies the adapter's own alpha / r.
"""
import json
import os
import re

import torch


class Adapter:
    """Key for the Linear lora dicts (exllamav3 keys them by adapter object)."""

    def __init__(self, name):
        self.name = name
        self.merged = []                                        # (MLAttention, w_uk delta, w_uv delta)
        self.b = []                                             # (B as Linear reads it, a copy of it) for set_scale
        self.scale = 1.0


def _peft_tensors(directory):
    from safetensors.torch import load_file
    cfg = json.load(open(os.path.join(directory, "adapter_config.json")))
    if cfg.get("fan_in_fan_out") or cfg.get("use_dora"):
        raise ValueError("fan_in_fan_out and DoRA adapters are not supported")
    alpha = float(cfg["lora_alpha"]) * (cfg["r"] ** 0.5 if cfg.get("use_rslora") else 1.0)
    raw = load_file(os.path.join(directory, "adapter_model.safetensors"), device="cpu")
    pairs = {}
    for k, t in raw.items():
        m = re.match(r"(?:base_model\.model\.)?(.+)\.lora_([AB])\.weight$", k)
        if not m:
            continue
        path = m.group(1).replace(".forget_gate.", ".")         # HF nests f_a/f_b under forget_gate; exllamav3 doesn't
        pairs.setdefault(path, {})[m.group(2)] = t          # bf16 here; fp32 one module at a time in apply()
    return cfg, alpha / cfg["r"], pairs


def _set(linear, adapter, A, B):
    """A: (r, in), B: (out, r), already scaled, fp32 -> the padded, transposed fp16 pair Linear.apply_lora reads."""
    a = torch.zeros(linear.in_features, A.shape[0])
    a[:A.shape[1]] = A.t()
    b = torch.zeros(B.shape[1], linear.out_features)
    b[:, :B.shape[0]] = B.t()
    dev = linear.device
    linear.lora_a_tensors[adapter] = a.half().to(dev)
    linear.lora_b_tensors[adapter] = b.half().to(dev)
    adapter.b.append((linear.lora_b_tensors[adapter], linear.lora_b_tensors[adapter].clone()))


def _apply_lora_addmm(self, lora_input, x):
    """Linear.apply_lora with the B product and the residual add as one addmm_ (x += (in @ A) @ B): 2 kernels per
    adapted linear per step instead of 3 (485 linears in the GM adapter). The add happens inside the GEMM's fp32
    accumulation, one rounding instead of two. Falls back to the stock path for other dtypes / layouts."""
    orig = lora_input.shape
    flat = lora_input.view(-1, orig[-1])
    x2 = x.view(-1, x.shape[-1]) if x.is_contiguous() else None
    for lora, a in self.lora_a_tensors.items():
        b = self.lora_b_tensors.get(lora)
        if b is None:
            continue
        t = (flat if flat.dtype == a.dtype else flat.to(a.dtype)) @ a
        if x2 is not None and x2.dtype == b.dtype:
            x2.addmm_(t, b)
        else:
            x += (t @ b).view(*orig[:-1], -1).to(x.dtype)


def _install_addmm():
    from exllamav3.modules.linear import Linear
    if os.environ.get("GLM_XPU_LORA_ADDMM", "1") == "1" and Linear.apply_lora is not _apply_lora_addmm:
        Linear.apply_lora = _apply_lora_addmm


@torch.inference_mode()
def apply(model, directory, scale=1.0, log=print):
    _install_addmm()
    from exllamav3.modules.linear import Linear
    from exllamav3.modules.mla_attn import MLAttention
    from exllamav3.modules.mlp import GatedMLP

    cfg, s, pairs = _peft_tensors(directory)
    s *= scale
    adapter = Adapter(os.path.basename(os.path.normpath(directory)))
    mods = {m.key: m for m in model}
    if getattr(model, "mtp", None):                             # the MTP draft model holds layer 45
        mods.update({m.key: m for m in model.mtp[0]})
    # HF adapters may omit language_model even when this checkpoint keeps that prefix.
    # Match both spellings against the actual loaded modules before deciding a layer is absent.
    def layer_id(path):
        match = re.match(r"model\.(?:language_model\.)?layers\.(\d+)\.", path)
        return match.group(1) if match else None

    layers = {layer_id(k) for k in mods} - {None}
    done, merged, fused_off, skipped = 0, 0, 0, 0
    unmatched = []
    qkv = {}                                                    # KDA attention key -> {q|k|v: (A, B)}

    for path, ab in sorted(pairs.items()):
        if path.startswith("model.layers."):
            alternate = path.replace("model.layers.", "model.language_model.layers.", 1)
        else:
            alternate = path.replace("model.language_model.layers.", "model.layers.", 1)
        if path not in mods and path.rsplit(".", 1)[0] not in mods:
            if alternate in mods or alternate.rsplit(".", 1)[0] in mods:
                path = alternate
        A, B = ab["A"].float(), ab["B"].float() * s
        parent, leaf = path.rsplit(".", 1)
        target = mods.get(path)
        if isinstance(target, Linear):
            _set(target, adapter, A, B)
            done += 1
            continue
        if leaf in ("q_proj", "k_proj", "v_proj") and isinstance(getattr(mods.get(parent), "qkv_proj", None), Linear):
            qkv.setdefault(parent, {})[leaf[0]] = (A, B)
            continue
        if leaf == "kv_b_proj" and isinstance(mods.get(parent), MLAttention):
            m = mods[parent]
            H, nope, v, D_c = m.num_q_heads, m.qk_nope_head_dim, m.v_head_dim, m.kv_lora_rank
            d = (B @ A).view(H, nope + v, D_c)                  # same layout as the checkpoint weight
            duk = d[:, :nope, :].permute(2, 0, 1).to(m.w_uk_flat)
            duv = d[:, nope:, :].permute(2, 0, 1).to(m.w_uv_flat)
            m.w_uk_flat.view(D_c, H, nope).add_(duk)
            m.w_uv_flat.view(D_c, H, v).add_(duv)
            adapter.merged.append((m, duk, duv))
            merged += 1
            continue
        if layer_id(path) is not None and layer_id(path) not in layers:
            skipped += 1                                        # a layer not loaded here, e.g. MTP with drafting off
            continue
        unmatched.append(path)

    for parent, parts in qkv.items():
        if set(parts) != {"q", "k", "v"}:
            raise ValueError(f"{parent}: LoRA covers only {sorted(parts)} of the fused q/k/v projection")
        lin = mods[parent].qkv_proj
        As = [parts[c][0] for c in "qkv"]
        Bs = [parts[c][1] for c in "qkv"]
        A = torch.cat(As, 0)                                    # (3r, in)
        B = torch.zeros(sum(b.shape[0] for b in Bs), A.shape[0])
        o = r = 0
        for b in Bs:
            B[o:o + b.shape[0], r:r + b.shape[1]] = b
            o, r = o + b.shape[0], r + b.shape[1]
        _set(lin, adapter, A, B)
        done += 3

    for m in mods.values():
        if isinstance(m, GatedMLP) and any(l.lora_a_tensors for l in (*m.gates, *m.ups, *m.downs)):
            if m.bc is not None or any(x is not None for x in m.multi_gu):
                m.bc = None
                m.multi_gu = [None] * len(m.multi_gu)
                fused_off += 1

    # the adapter file and the fp32 copies (~1.1 GB) go back to the OS: the RAM tier leaves little host RAM
    del pairs, qkv
    import ctypes
    import gc
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    if unmatched:
        raise ValueError(f"LoRA {adapter.name}: {len(unmatched)} targets match no module, e.g. {unmatched[:3]}")
    if not done and not merged:
        raise ValueError(f"LoRA {adapter.name}: no targets applied ({skipped} in layers not loaded)")
    log(f"LoRA {adapter.name}: {done} linears at runtime, {merged} kv_b merged, fused MLP path off in {fused_off} "
        f"MLPs (r={cfg['r']}, alpha={cfg['lora_alpha']}, scale {s:.3f})" + (f", {skipped} in layers not loaded" if skipped else ""))
    return adapter



@torch.inference_mode()
def set_scale(adapter, k):
    """Scale an applied adapter by k relative to how it was applied (0 = off, 1 = as applied), in place: the decode
    graphs hold the addresses of these tensors, so they must never be replaced or freed while graphs exist."""
    if k == adapter.scale:
        return
    for b, b0 in adapter.b:
        b.copy_(b0).mul_(k)
    for m, duk, duv in adapter.merged:
        H, nope, v, D_c = m.num_q_heads, m.qk_nope_head_dim, m.v_head_dim, m.kv_lora_rank
        m.w_uk_flat.view(D_c, H, nope).add_(duk, alpha=k - adapter.scale)
        m.w_uv_flat.view(D_c, H, v).add_(duv, alpha=k - adapter.scale)
    adapter.scale = k
