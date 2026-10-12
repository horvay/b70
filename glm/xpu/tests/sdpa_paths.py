# Which XPU SDPA call shape is slow? fp16, head dim 256, small sizes, each timed (synchronized).
import time, torch
import torch.nn.functional as F
dev = torch.device("xpu")
torch.manual_seed(0)
G, D = 16, 256
def t(label, fn):
    torch.xpu.synchronize(); t0 = time.perf_counter(); fn(); torch.xpu.synchronize()
    print(f"{label:55s} {1000 * (time.perf_counter() - t0):9.1f} ms", flush=True)
for T in (256, 1024):
    q = torch.randn(1, G, T, D, device=dev).half()
    k = torch.randn(1, G, 2 * T, D, device=dev).half()
    v = torch.randn(1, G, 2 * T, D, device=dev).half()
    for rep in range(2):
        t(f"T={T} causal, contiguous k/v", lambda: F.scaled_dot_product_attention(q, k[:, :, :T].contiguous(), v[:, :, :T].contiguous(), is_causal=True))
        t(f"T={T} causal, sliced k/v view", lambda: F.scaled_dot_product_attention(q, k[:, :, :T], v[:, :, :T], is_causal=True))
        m = torch.arange(2 * T, device=dev).view(1, -1) <= (T + torch.arange(T, device=dev)).view(-1, 1)
        t(f"T={T} bool mask, keys 2T, contiguous", lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=m))
        t(f"T={T} bool mask, keys 2T-1 sliced view", lambda: F.scaled_dot_product_attention(q, k[:, :, :2 * T - 1], v[:, :, :2 * T - 1], attn_mask=m[:, :2 * T - 1]))
