"""Replace exllamav3's BlockSparseMLP routed-expert path with the B70 tier engine (glm_xpu.tier).

load(): loads the router, shared experts and score-correction bias, never the routed-expert Linears (114 GB); the
experts come from the packed NVMe store through the tier. forward(): routing (exllamav3's own routing_fn), the routed
sum on the B70 MoE kernel, plus the shared expert.
"""
import torch

from exllamav3.modules.block_sparse_mlp import BlockSparseMLP

TIER = None
_layer_li = {}


def install(tier, layer_li: dict):
    """tier: glm_xpu.tier.Tier; layer_li: model layer index -> store layer index."""
    global TIER
    TIER = tier
    _layer_li.update(layer_li)
    BlockSparseMLP.load = _load
    BlockSparseMLP.forward = _forward


def _model_layer(key: str) -> int:
    # "model.language_model.layers.12.mlp"
    return int(key.split(".layers.")[1].split(".")[0])


def _load(self, device, **kwargs):
    self.device = device
    experts = {id(m) for m in self.gates + self.ups + self.downs}
    for m in self.modules:
        if id(m) not in experts:
            m.load(device, **kwargs)
    self.e_score_correction_bias = None
    if self.e_score_correction_bias_key:
        for k in [self.e_score_correction_bias_key, "gate.e_score_correction_bias"]:
            esb = self.config.stc.get_tensor(f"{self.key}.{k}", self.device, optional=True, allow_bf16=True,
                                             no_defer=True)
            if esb is not None:
                self.e_score_correction_bias = esb if esb.dtype == torch.half else esb.float()
                break
    self.load_routing(**kwargs)
    self.tier_li = _layer_li[_model_layer(self.key)]


PREFETCH = True


def link_layers(model):
    """Chain the MoE modules in order so each can predict the next layer's picks (layer-ahead prefetch)."""
    mods = []

    def walk(m):
        if isinstance(m, BlockSparseMLP):
            mods.append(m)
        for c in getattr(m, "modules", []):
            walk(c)
    for m in model.modules:
        walk(m)
    for a, b in zip(mods, mods[1:]):
        a._next_moe = b
    if mods:
        mods[-1]._next_moe = None
    return len(mods)


def _forward(self, x, params, out_dtype=None):
    y = x.view(-1, self.hidden_size)
    if y.dtype != torch.half:
        y = y.half()
    bsz = y.shape[0]
    sel, w = self.routing_fn(bsz, self.routing_cfg, y, params)
    nxt = getattr(self, "_next_moe", None)
    if PREFETCH and nxt is not None and sel.numel() <= TIER.admit_max:
        # layer-ahead prediction: the next MoE layer's router on this layer's MoE input (a residency hint only)
        logits = torch.matmul(y, nxt.routing_cfg.gate_tensor[:, :nxt.num_experts])
        s = torch.sigmoid(logits.float())
        esb = nxt.routing_cfg.e_score_correction_bias
        if esb is not None:
            s = s + esb.float()
        pred = torch.topk(s, nxt.num_experts_per_tok, dim=-1).indices
        both = torch.cat([sel.view(-1), pred.view(-1)]).cpu().numpy()
        k = sel.numel()
        TIER.prefetch(nxt.tier_li, both[k:], protect=(both[:k] + self.tier_li * TIER.E).tolist())
        out = TIER.moe(self.tier_li, y, sel, w, ids_np=both[:k].reshape(sel.shape)).float()
    else:
        out = TIER.moe(self.tier_li, y, sel, w).float()
    if self.shared_experts is not None:
        out += self.shared_experts.forward(x, params).view(out.shape)
    out = out.view(x.shape[:-1] + (self.hidden_size,))
    if out_dtype is not None:
        out = out.to(out_dtype)
    return out
