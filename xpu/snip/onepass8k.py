# Item 5: per-layer memory bounds (KDA split every 2048, MLA heads in groups of 16, dense MLP row slices).
# (1) 3K prompt, old settings vs new: checkpoint diff + greedy text. (2) one-pass prompts of 4K/6K/8K: time, peak memory.
import random, time
import exllamav3.generator.job as J
import exllamav3.modules.gated_delta_net as GD, exllamav3.modules.mlp as MLPm, glm_xpu.sparse_mla as SM
from exllamav3.generator import Job
words = "the river mountain keeper lantern dragon market silver harbor winter letter garden storm candle".split()
rc = gen.recurrent_cache
def run(body, n=24):
    rc.clear(); gen.pagetable.reset_page_table()
    ids = tok.encode(model.default_chat_prompt(body), add_bos=False, encode_special_tokens=True)
    torch.xpu.synchronize(); torch.xpu.reset_peak_memory_stats(); base = torch.xpu.memory_allocated()
    out = []; t0 = time.perf_counter(); tp = None
    gen.enqueue(Job(input_ids=ids, max_new_tokens=n, sampler=ArgmaxSampler()))
    while gen.num_remaining_jobs():
        for r in gen.iterate():
            if r.get("stage") == "streaming" and tp is None:
                tp = time.perf_counter() - t0
            out.append(r.get("text", ""))
    peak = (torch.xpu.max_memory_allocated() - base) / 2**30
    st = list(rc.items())[-1][1] if len(rc) else None
    return ids.shape[-1], tp, peak, "".join(out), st
def settings(new):
    GD._KDA_SUB = 2048 if new else 0
    MLPm._ROW_CHUNK = 2048 if new else 0
    SM.PREFILL_HEAD_GROUP = 16 if new else 64
random.seed(5)
body3 = "Summarize: " + " ".join(random.choice(words) for _ in range(3000))
res = {}
try:
    gen.max_chunk_size = 4096
    for new in (() if globals().get("SKIP_EQ") else (False, True)):
        settings(new)
        T, tp, peak, text, st = run(body3)
        res[new] = st
        print(f"3K {'new' if new else 'old'}: {T} tokens, prompt {tp:.1f} s, peak extra {peak:.2f} GiB, text {text!r}", flush=True)
    a, b = res.get(False), res.get(True)
    ks = [] if a is None else [k for k in a if k not in ("position", "checkpoint_size")]
    for k in ((ks[0], ks[len(ks) // 2], ks[-1]) if ks else ()):
        print("  ckpt", k, ["%.1e" % ((x.float() - y.float()).norm() / (x.float().norm() + 1e-12)).item() for x, y in zip(a[k], b[k])])
    gen.max_chunk_size = 8192
    for nw in (4000, 6000, 8000):
        T, tp, peak, text, _ = run("Summarize: " + " ".join(random.choice(words) for _ in range(nw)), n=8)
        if tp is None:
            print(f"one pass {T} tokens: no tokens generated (failed)", flush=True); break
        print(f"one pass {T} tokens: prompt {tp:.1f} s ({T / tp:.0f} tok/s), peak extra {peak:.2f} GiB, "
              f"free now {torch.xpu.mem_get_info()[0] / 2**30:.2f} GiB, text {text!r}", flush=True)
finally:
    settings(True)
