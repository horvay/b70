# Blocked MLA prefill attention (head groups, query blocks) vs one SDPA call, random data, past = 0 and past > 0.
import importlib, torch
import glm_xpu.sparse_mla as SM
importlib.reload(SM)
dev = torch.device("xpu")
torch.manual_seed(0)
H, Dn, Dv, D, page = 64, 256, 256, 512, 256
pages = 12
ckv = (torch.randn(pages, page, 1, D, device=dev) * 0.5).half()
bt = torch.randperm(pages, device=dev).int().view(1, -1)
wk = (torch.randn(D, H * Dn, device=dev) * 0.05).half()
wv = (torch.randn(D, H * Dv, device=dev) * 0.05).half()
for past, q_len in ((0, 1300), (700, 900)):
    q = (torch.randn(q_len, H, Dn, device=dev)).half()
    args = (q, wk, wv, ckv, bt, [past], 1, q_len, Dv, Dn, Dn ** -0.5)
    SM.PREFILL_HEAD_GROUP, SM.PREFILL_Q_BLOCK = 64, 0
    ref = SM.mla_prefill_mha_sdpa(*args)
    SM.PREFILL_HEAD_GROUP, SM.PREFILL_Q_BLOCK = 16, 512
    new = SM.mla_prefill_mha_sdpa(*args)
    print(f"past {past} q_len {q_len}: rel diff {((ref.float() - new.float()).norm() / ref.float().norm()).item():.2e}")
