# Triton KDA step with history vs the torch fallback: outputs and every history slot equal.
import torch
from glm_xpu import ops
dev = torch.device("xpu"); torch.manual_seed(0)
H, Dk, Dv, S, NH = 64, 128, 128, 2, 4
F = 2 * H * Dk + H * Dv
qkv = torch.randn(1, S, F, device=dev).bfloat16()
g = -torch.rand(1, S, H, Dk, device=dev) * 0.5
beta = torch.rand(1, S, H, device=dev).bfloat16()
st0 = torch.randn(2, NH, H, Dk, Dv, device=dev) * 0.1
slots = torch.tensor([1], dtype=torch.int32, device=dev)
outs = []
for torch_path in (True, False):
    st = st0.clone(); out = torch.empty(1, S, H, Dv, device=dev, dtype=torch.bfloat16)
    if torch_path:
        ops.cuda_recurrent_gated_delta_rule_torch(qkv, g, beta, st, out, H, H, Dk, Dv, slots, True)
    else:
        ops.cuda_recurrent_gated_delta_rule(qkv, g, beta, st, out, H, H, Dk, Dv, slots, True)
    outs.append((out.float(), st))
rel = lambda a, b: ((a - b).norm() / (a.norm() + 1e-12)).item()
print(f"out {rel(outs[0][0], outs[1][0]):.1e}  state[slot,0] {rel(outs[0][1][1, 0], outs[1][1][1, 0]):.1e}  "
      f"history[slot,1] {rel(outs[0][1][1, 1], outs[1][1][1, 1]):.1e}  other slot untouched {torch.equal(outs[1][1][0], st0[0])}")
