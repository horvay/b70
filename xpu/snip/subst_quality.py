# Paired decode-quality check for router substitution: generate reference continuations with exact routing, then
# teacher-force the same tokens through one-token decode steps at several SUBST_KEEP values and compare logits with
# an exact teacher-forced pass (KL on the full vocabulary, top-1 agreement). Resets the generator's page table after.
import math, time
from glm_xpu import block_graphs as bg
from exllamav3.constants import PAGE_SIZE
N = globals().get("N_TF", 64)
KEEPS = globals().get("KEEPS", [(8, None), (6, "replace"), (6, "drop"), (4, "drop")])
prompts = ["Explain how a refrigerator works, step by step.",
           "Write a short scene where a detective questions a nervous innkeeper.",
           "What were the main causes of the French Revolution?"]

def tf_decode(tokens, P):
    """Prefill tokens[:P-1], then feed tokens[P-1..] one at a time; returns log-probs [T-P+1, V] (fp32, GPU)."""
    state = cache.get_new_state()
    T = len(tokens)
    cap = -(-(T + 1) // PAGE_SIZE) * PAGE_SIZE
    ids = torch.tensor([tokens], dtype=torch.long)
    base = {"attn_mode": "flash_attn", "cache": cache, "batch_shape": (1, cap)}
    model.forward(ids[:, :P - 1], dict(base, past_len=0, recurrent_states=[state]))
    out = []
    for t in range(P - 1, T):
        lg = model.forward(ids[:, t:t + 1], dict(base, past_len=t, recurrent_states=[state]))
        out.append(torch.log_softmax(lg[0, -1].float(), -1))
    cache.release_state(state)
    return torch.stack(out)

refs = []
bg.set_subst(8)
for p in prompts:
    _, text, _ = decode(N, prompt=p, warm=False)
    pt = model.default_chat_prompt(p)
    ptoks = tok.encode(pt, add_bos=False, encode_special_tokens=True)[0].tolist()
    ctoks = tok.encode(text, add_bos=False, encode_special_tokens=True)[0].tolist()
    refs.append((ptoks + ctoks, len(ptoks)))
base = [tf_decode(t, P) for t, P in refs]
for keep, mode in KEEPS:
    bg.set_subst(keep, mode)
    s0 = dict(bg.subst_stats)
    tz = dict(tier.stats)
    t0 = time.perf_counter()
    kl = top1 = n = 0
    for (t, P), b in zip(refs, base):
        lp = tf_decode(t, P)
        m = min(len(lp), len(b))
        pb = b[:m].exp()
        kl += float((pb * (b[:m] - lp[:m])).sum())
        top1 += int((lp[:m].argmax(-1) == b[:m].argmax(-1)).sum())
        n += m
    dt = time.perf_counter() - t0
    sub = bg.subst_stats["substituted"] - s0["substituted"]; pk = bg.subst_stats["picks"] - s0["picks"]
    nv = (tier.stats["nvme"] - tz["nvme"]) + (tier.stats.get("prefetch_used", 0) - tz.get("prefetch_used", 0))
    print(f"keep {keep} {mode or ''}: KL {kl / n:.5f}  top1 {top1 / n:.4f}  substituted {100 * sub / max(pk, 1):.1f}% of picks  "
          f"NVMe-served picks {nv}  ({n} positions, {dt:.0f} s)", flush=True)
bg.set_subst(8)
gen.pagetable.reset_page_table()
