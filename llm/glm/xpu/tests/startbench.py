"""Startup timing, in process, loaded as serve.py loads: the background RAM-tier fill vs the first request.
  in_rootfs.sh --gpu python3 /w/xpu/tests/startbench.py [--wait-fill] [--ram-gb 13]
Without --wait-fill the first request ("Hello.", one token) starts as soon as the model is loaded, as the launcher's
warm-up does; with it, the request waits until the fill is done. Then two more requests time a warm server.
"""
import argparse
import json
import os
import time

from glm_xpu.runner import load, log
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--store", default="/nvx/glm53_flash_exl3_experts_blob.bin")
ap.add_argument("--ram-gb", type=float, default=13.0)
ap.add_argument("--cache", type=int, default=49152)
ap.add_argument("--wait-fill", action="store_true")
ap.add_argument("--sleep", type=float, default=0, help="idle this long between load and the first request")
ap.add_argument("--decode", type=int, default=0, help="then generate this many tokens and report decode speed")
ap.add_argument("--mtp", type=int, default=0, help="draft tokens from the MTP head (speculative decoding), as serve.py")
ap.add_argument("--greedy", action="store_true", help="argmax sampling (MTP acceptance ceiling)")
ap.add_argument("--prefill", type=int, default=0, help="time fresh prompts of about this many tokens (twice)")
ap.add_argument("--turns", type=int, default=0, help="a conversation of this many turns (--decode tokens each), with "
                                                    "the server's between-turn cache curation unless GLM_XPU_CURATE=0")
ap.add_argument("--bg", action="store_true", help="measure GPU capacity a second queue gets during decode "
                                                    "(--decode tokens, with and without a background stream)")
ap.add_argument("--bg-prio", type=int, default=0, help="--bg: background stream priority (lower = higher; 1 = low)")
ap.add_argument("--main-prio", type=int, default=0, help="--bg: run decode on a stream of this priority (-1 = high)")
ap.add_argument("--kprof", type=int, default=0, help="then profile this many decode tokens per kernel (eager)")
ap.add_argument("--window", type=int, default=0, help="with --decode: one generation, tok/s per this many tokens")
ap.add_argument("--skip-hello", action="store_true")
ap.add_argument("--dump", type=float, default=0, help="dump every thread's stack this often during the first request")
ap.add_argument("--lora", default=",".join(f"/models/{n}" for n in sorted(os.listdir("/models"))
                                           if n == "lora" or n.startswith("lora-")) or None)
a = ap.parse_args()

meta = json.load(open(a.store.rsplit(".", 1)[0] + ".json"))
li_of = {l["key"]: l["li"] for l in meta["layers"]}
stats = json.load(open("/w/xpu/data/route_stats_glm53.json"))
order = [(li, e) for _, li, e in sorted(((sc, li_of[k], e) for k, v in stats.items() if k in li_of
                                         for e, sc in enumerate(v)), reverse=True)]
t_start = time.perf_counter()
model, config, cache, tok, tier = load("/models", a.store, cache_size=a.cache, ram_gb=a.ram_gb, warm=order,
                                       lora=a.lora, mtp=a.mtp)
from exllamav3 import Generator
from exllamav3.generator.sampler import ComboSampler
from glm_xpu import block_graphs
gkw = dict(draft_model=model.mtp[0], draft_cache=model.mtp[1], num_draft_tokens=a.mtp,
           dynamic_draft_tokens=True) if a.mtp else {}
gen = Generator(model=model, cache=cache, tokenizer=tok, max_chunk_size=int(os.environ.get("GLM_XPU_CHUNK", "8192")),
                **gkw)
block_graphs.install(tier)
log(f"loaded and installed in {time.perf_counter() - t_start:.1f} s")


def fill_left():
    return sum(1 for fs in tier.inflight.values() for f in fs if not f.done())


if a.wait_fill:
    t0 = time.perf_counter()
    while fill_left():
        time.sleep(0.2)
    log(f"waited {time.perf_counter() - t0:.1f} s for the RAM fill")
if a.sleep:
    time.sleep(a.sleep)
    log(f"slept {a.sleep:.0f} s")
if a.dump:
    import faulthandler
    import sys
    faulthandler.dump_traceback_later(a.dump, repeat=True, file=sys.stderr)
KEYS = ("picks", "vram_hit", "ram_hit", "nvme", "prefetch", "prefetch_used", "t_nvme", "t_pread", "t_copy", "t_h2d",
        "admit", "writeback", "staged", "staged_used")
for i, q in enumerate(() if a.skip_hello else ("Hello.", "Hello.", "What is the capital of France? Answer briefly.")):
    s0 = dict(tier.stats)
    left = fill_left()
    t0 = time.perf_counter()
    with torch.inference_mode():
        gen.generate(prompt=model.default_chat_prompt(q), max_new_tokens=1, completion_only=True, add_bos=False,
                     sampler=ComboSampler(temperature=1.0, top_p=0.95))
    torch.xpu.synchronize()
    d = {k: round(tier.stats.get(k, 0) - s0.get(k, 0), 2) for k in KEYS}
    if i == 0 and a.dump:
        faulthandler.cancel_dump_traceback_later()
    log(f"request {i} ({q!r}): {time.perf_counter() - t0:.1f} s, {left} fill pieces still queued at its start; "
        f"tier {d}")
if a.decode and a.window:
    # one long generation, tok/s per window of a.window tokens: does decode speed up as the cache adapts?
    from exllamav3.generator.job import Job
    q = "Write a long, detailed story about a lighthouse keeper who finds a message in a bottle."
    ids = tok.encode(model.default_chat_prompt(q), add_bos=False)
    from exllamav3.generator.sampler import ArgmaxSampler
    job = Job(input_ids=ids, max_new_tokens=a.decode, stop_conditions=[], seed=1234,
              sampler=ArgmaxSampler() if a.greedy else ComboSampler(temperature=1.0, top_p=0.95))
    gen.enqueue(job)
    n = n0 = 0
    s0 = dict(tier.stats)
    t0 = None
    cpr = None
    if os.environ.get("GLM_XPU_CPROFILE"):
        import cProfile
        cpr = cProfile.Profile()
    with torch.inference_mode():
        while gen.num_remaining_jobs():
            if cpr is not None and n == a.window:
                cpr.enable()                            # after the first window (graph capture, warm-up)
            for r in gen.iterate():
                if r.get("stage") != "streaming" or "token_ids" not in r:
                    continue
                if t0 is None:                          # first token: the prompt is done
                    if block_graphs.PROF:
                        block_graphs.prof_take()
                    t0 = time.perf_counter()
                    s0 = dict(tier.stats)
                    continue
                n += r["token_ids"].shape[-1]
                if n - n0 >= a.window or r.get("eos"):
                    dt = time.perf_counter() - t0
                    d = {k: tier.stats.get(k, 0) - s0.get(k, 0) for k in KEYS}
                    m = max(n - n0, 1)
                    acc = (f", drafts accepted {job.accepted_draft_tokens}/"
                           f"{job.accepted_draft_tokens + job.rejected_draft_tokens}") if a.mtp else ""
                    log(f"tokens {n0:4d}-{n:4d}: {m / dt:.2f} tok/s{acc}; per token nvme {d['nvme'] / m:.1f}, "
                        f"prefetch {d['prefetch'] / m:.1f} (used {d['prefetch_used'] / m:.1f}), "
                        f"vram {d['vram_hit'] / m:.0f}, ram {d['ram_hit'] / m:.0f}, admit {d['admit'] / m:.1f}, "
                        f"writeback {d['writeback'] / m:.1f}, staged {d['staged'] / m:.1f} (used {d['staged_used'] / m:.1f})")
                    if block_graphs.PROF:
                        pr = block_graphs.prof_take()
                        ms = lambda v: 1000 * v / m
                        gpu = pr["gpu_ms"]
                        g_all = sum(gpu.values()) / m
                        host = dt - pr["sync_s"] - pr["ensure_s"] - pr["prefetch_s"]
                        log(f"   ms/token: wall {ms(dt):.1f} = GPU in decode graphs {g_all:.1f} ("
                            + ", ".join(f"{k} {v / m:.1f}" for k, v in sorted(gpu.items()))
                            + f") + GPU idle {ms(dt) - g_all:.1f}; host: waiting at syncs {ms(pr['sync_s']):.1f}, "
                            f"ensure {ms(pr['ensure_s']):.1f} (NVMe wait {ms(d['t_nvme']):.1f}), prefetch "
                            f"{ms(pr['prefetch_s']):.1f}, sync return -> experts submitted {ms(pr['pre_c_s']):.1f}, graph replay calls "
                            f"{ms(pr['replay_s']):.1f} ({pr['n_replay'] / m:.0f} per token), everything else "
                            f"{ms(host):.1f}; RAM picks {pr['ram_picks'] / m:.1f}/token, of them predicted a layer "
                            f"ahead {pr['ram_predicted'] / max(pr['ram_picks'], 1):.0%}")
                    n0, t0, s0 = n, time.perf_counter(), dict(tier.stats)
    if cpr is not None:
        cpr.disable()
        import pstats
        pstats.Stats(cpr).sort_stats("tottime").print_stats(30)
elif a.decode:
    q ="Write a long, detailed story about a lighthouse keeper who finds a message in a bottle."
    for rep in range(2):
        s0 = dict(tier.stats)
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = gen.generate(prompt=model.default_chat_prompt(q), max_new_tokens=a.decode, completion_only=True,
                               add_bos=False, sampler=ComboSampler(temperature=1.0, top_p=0.95), stop_conditions=[],
                               seed=1234 + rep)    # same tokens in every run, so prefetch settings compare fairly
        torch.xpu.synchronize()
        dt = time.perf_counter() - t0
        n = len(tok.encode(out)[0])
        d = {k: tier.stats.get(k, 0) - s0.get(k, 0) for k in KEYS}
        log(f"decode {rep}: {n} tokens in {dt:.1f} s = {n / dt:.2f} tok/s (incl. a short prefill); per token: "
            + ", ".join(f"{k} {v / max(n, 1):.3f}" for k, v in d.items())
            + f"; t_nvme share of wall {d['t_nvme'] / dt:.0%}")
if a.prefill:
    para = ("The old lighthouse stood on a cliff above the harbor, its lamp turning through the long winter nights. "
            "Ships passed below, their crews trusting the beam to guide them past the rocks. ")
    PK = ("prefill_stage", "prefill_nvme", "t_nvme", "t_pread", "t_copy", "nvme", "picks")
    for rep in range(2):
        words = f"Variant {rep}: " + para * max(1, a.prefill // 30)
        tier.stats.update(tier.native.stats() if tier.native is not None else {})
        s0 = dict(tier.stats)
        torch.xpu.synchronize()
        kp = None
        if rep == 1 and os.environ.get("GLM_XPU_KPROF_PREFILL"):
            from torch.profiler import ProfilerActivity, profile
            kp = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU])
            kp.__enter__()
        t0 = time.perf_counter()
        with torch.inference_mode():
            gen.generate(prompt=model.default_chat_prompt(words + " Summarize."), max_new_tokens=1, add_bos=False,
                         completion_only=True, sampler=ComboSampler(temperature=1.0, top_p=0.95))
        torch.xpu.synchronize()
        dt = time.perf_counter() - t0
        if kp is not None:
            kp.__exit__(None, None, None)
            ev = sorted([e for e in kp.key_averages() if getattr(e, "device_time_total", 0) > 0],
                        key=lambda e: -e.device_time_total)
            log(f"prefill kernels (device ms): total {sum(e.device_time_total for e in ev) / 1e3:.0f}")
            for e in ev[:25]:
                log(f"  {e.device_time_total / 1e3:9.1f} ms {e.count:7d}x  {e.key[:100]}")
        tier.stats.update(tier.native.stats() if tier.native is not None else {})
        d = {k: tier.stats.get(k, 0) - s0.get(k, 0) for k in PK}
        n = len(tok.encode(words)[0])
        gb = d["prefill_nvme"] * tier.rec_max / 1e9
        log(f"prefill {rep}: ~{n} tokens in {dt:.1f} s ({n / dt:.0f} tok/s); experts staged {d['prefill_stage']}, from "
            f"NVMe {d['prefill_nvme']} (~{gb:.1f} GB, {gb / max(dt, 1e-9):.1f} GB/s over the whole prefill); host waited "
            f"on NVMe {d['t_nvme']:.1f} s; reader pread {d['t_pread']:.1f} s, copy {d['t_copy']:.1f} s (thread-seconds)")
if a.turns:
    from exllamav3.generator.job import Job
    asks = ["Write a long, detailed story about a lighthouse keeper who finds a message in a bottle.",
            "Continue the story: the keeper decides to answer the message. What happens next?",
            "Now tell it from the point of view of the person who wrote the message.",
            "Describe the storm that finally brings them together."]
    hist = []
    for t in range(a.turns):
        hist.append({"role": "user", "content": asks[t % len(asks)]})
        try:
            ids = tok.hf_chat_template(hist, add_generation_prompt=True)      # the model's own chat template
        except Exception:                                                  # noqa: BLE001  (no transformers)
            ids = tok.encode(model.default_chat_prompt("\n\n".join(m["content"] for m in hist)), add_bos=False)
        if not torch.is_tensor(ids):
            ids = torch.tensor([ids], dtype=torch.long)
        job = Job(input_ids=ids, max_new_tokens=a.decode, stop_conditions=[], seed=4321 + t,
                  sampler=ComboSampler(temperature=1.0, top_p=0.95))
        gen.enqueue(job)
        s0, t0, n, out = dict(tier.stats), None, 0, []
        with torch.inference_mode():
            while gen.num_remaining_jobs():
                for r in gen.iterate():
                    if r.get("stage") == "streaming" and "token_ids" in r:
                        if t0 is None:
                            t0, s0 = time.perf_counter(), dict(tier.stats)
                            continue
                        n += r["token_ids"].shape[-1]
                        out.append(r.get("text", ""))
        dt = time.perf_counter() - t0
        d = {k: tier.stats.get(k, 0) - s0.get(k, 0) for k in KEYS}
        m = max(n, 1)
        log(f"turn {t}: {n} tokens at {n / dt:.2f} tok/s; per token vram {d['vram_hit'] / m:.0f}, ram "
            f"{d['ram_hit'] / m:.0f}, nvme {d['nvme'] / m:.1f}, admit {d['admit'] / m:.1f}")
        hist.append({"role": "assistant", "content": "".join(out)})
        if os.environ.get("GLM_XPU_CURATE", "1") == "1" and t + 1 < a.turns:
            with torch.inference_mode():
                nv, nr = tier.curate()
            torch.xpu.synchronize()
            time.sleep(2)                                    # NVMe reads it started land, as during a user's think time
            log(f"  curated: {nv} RAM -> VRAM, {nr} NVMe -> RAM")
if a.bg:
    import threading
    from exllamav3.generator.job import Job
    A = torch.randn(2048, 2048, device=torch.device("xpu", 0), dtype=torch.half)
    B = torch.randn(2048, 2048, device=torch.device("xpu", 0), dtype=torch.half)
    bs = torch.xpu.Stream(torch.device("xpu", 0), priority=a.bg_prio)
    ms = torch.xpu.Stream(torch.device("xpu", 0), priority=a.main_prio) if a.main_prio else None
    log(f"bg stream priority {bs.priority}, decode stream priority {ms.priority if ms else 'default'}")
    state = {"run": False, "n": 0}

    def bg():
        # small kernels (~ms or less) on their own stream, at most 8 in flight
        while state["run"]:
            with torch.xpu.stream(bs):
                for _ in range(8):
                    torch.matmul(A, B)
            bs.synchronize()
            state["n"] += 8

    def bg_for(sec):
        state.update(run=True, n=0)
        th = threading.Thread(target=bg, daemon=True)
        th.start()
        t0 = time.perf_counter()
        time.sleep(sec)
        state["run"] = False
        th.join()
        return state["n"] / (time.perf_counter() - t0)

    with torch.inference_mode():
        for _ in range(3):
            torch.matmul(A, B)
        torch.xpu.synchronize()
        alone = bg_for(10)
    log(f"bg alone: {alone:.0f} matmuls/s (2048^3 fp16, {alone * 2 * 2048 ** 3 / 1e12:.1f} TFLOP/s)")

    def decode(n_tok, seed):
        q = "Write a long, detailed story about a lighthouse keeper who finds a message in a bottle."
        job = Job(input_ids=tok.encode(model.default_chat_prompt(q), add_bos=False), max_new_tokens=n_tok,
                  stop_conditions=[], seed=seed, sampler=ComboSampler(temperature=1.0, top_p=0.95))
        gen.enqueue(job)
        t0, n = None, 0
        import contextlib
        with torch.inference_mode(), (torch.xpu.stream(ms) if ms else contextlib.nullcontext()):
            while gen.num_remaining_jobs():
                for r in gen.iterate():
                    if r.get("stage") == "streaming" and "token_ids" in r:
                        if t0 is None:
                            t0 = time.perf_counter()
                            continue
                        n += r["token_ids"].shape[-1]
        return n / (time.perf_counter() - t0)

    decode(100, 1)                                            # warm-up (graph capture, cache)
    base = decode(a.decode, 7)
    state.update(run=True, n=0)
    th = threading.Thread(target=bg, daemon=True)
    th.start()
    t0 = time.perf_counter()
    with_bg = decode(a.decode, 7)
    state["run"] = False
    th.join()
    got = state["n"] / (time.perf_counter() - t0)
    base2 = decode(a.decode, 7)
    log(f"decode alone {base:.2f} / {base2:.2f} tok/s; with the background stream {with_bg:.2f} tok/s "
        f"({with_bg / ((base + base2) / 2):.0%}); background got {got:.0f} matmuls/s = {got / alone:.0%} of the GPU")
if a.kprof:
    # device time per kernel over a.kprof decode tokens (run with GLM_XPU_BLOCK_GRAPHS=0: kernels inside captured graphs
    # are invisible to the profiler; replaying graphs outside a decode step faulted the GPU, 2026-10-30)
    from torch.profiler import ProfilerActivity, profile
    q = "Continue the story with a storm at sea."
    torch.xpu.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.XPU]) as prof, torch.inference_mode():
        out = gen.generate(prompt=model.default_chat_prompt(q), max_new_tokens=a.kprof, completion_only=True,
                           add_bos=False, sampler=ComboSampler(temperature=1.0, top_p=0.95), stop_conditions=[],
                           seed=99)
        torch.xpu.synchronize()
    m = max(len(tok.encode(out)[0]), 1)
    ev = [e for e in prof.key_averages() if getattr(e, "device_time_total", 0) > 0]
    ev.sort(key=lambda e: -e.device_time_total)
    tot = sum(e.device_time_total for e in ev)
    log(f"kprof: {m} tokens, device time {tot / 1e3 / m:.1f} ms per token (kernels below, per token)")
    for e in ev[:45]:
        log(f"  {e.device_time_total / 1e3 / m:7.2f} ms {e.count / m:7.1f}x  {e.key[:110]}")
log("STARTBENCH DONE")
os._exit(0)
