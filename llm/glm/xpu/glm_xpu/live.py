"""Live speed meter: wraps Generator.iterate and appends one line per second to a log while a job runs (prompt
progress and tok/s during prefill, tok/s during decode), plus a summary line per job. Watch with tail -f.
"""
import time
from typing import Any


def install(gen, path):
    orig = gen.iterate
    orig_enqueue = gen.enqueue
    st: dict[int, dict[str, Any]] = {}
    t_enq: dict[int, float] = {}

    def enqueue(job, *a, **k):
        for j in job if isinstance(job, list) else [job]:
            t_enq[id(j)] = time.perf_counter()
        return orig_enqueue(job, *a, **k)

    def write(line):
        try:
            with open(path, "a") as f:
                f.write(time.strftime("%H:%M:%S ") + line + "\n")
        except OSError:
            pass

    def iterate(*a, **k):
        results = orig(*a, **k)
        now = time.perf_counter()
        for r in results:
            j = id(r.get("job"))
            s: Any = st.get(j)
            if s is None:
                t0 = t_enq.pop(j, now)
                s = st[j] = {"t0": t0, "t_last": t0, "pf": 0, "pf_max": 0, "pf_last": 0, "t_pf_end": None,
                             "tok": 0, "tok_last": 0}
            stage = r.get("stage")
            if stage == "prefill":
                s["pf"], s["pf_max"] = r.get("curr_progress", 0), r.get("max_progress", 0)
            elif stage == "streaming":
                if s["t_pf_end"] is None:
                    s["t_pf_end"] = now
                    s["t_last"], s["tok_last"] = now, 0
                    dt = now - s["t0"]
                    write(f"prompt done: {s['pf_max']} tokens in {dt:.1f} s ({s['pf_max'] / max(dt, 1e-9):.0f} tok/s)")
                ids = r.get("token_ids")
                s["tok"] += int(ids.numel()) if ids is not None else 0
            if now - s["t_last"] >= 1.0 and not r.get("eos"):
                if s["t_pf_end"] is None:
                    rate = (s["pf"] - s["pf_last"]) / (now - s["t_last"])
                    write(f"prefill: {s['pf']}/{s['pf_max']} tokens, {rate:.0f} tok/s")
                    s["pf_last"] = s["pf"]
                else:
                    rate = (s["tok"] - s["tok_last"]) / (now - s["t_last"])
                    write(f"decode: {rate:5.1f} tok/s ({s['tok']} tokens)")
                    s["tok_last"] = s["tok"]
                s["t_last"] = now
            if r.get("eos"):
                if stage == "error":
                    write(f"job failed: {r.get('error')!r}"[:200])
                elif s["t_pf_end"] is not None:
                    dt = now - s["t_pf_end"]
                    write(f"job done: decode {s['tok']} tokens in {dt:.1f} s ({max(s['tok'] - 1, 0) / max(dt, 1e-9):.1f} tok/s)")
                st.pop(j, None)
        return results

    gen.iterate = iterate
    gen.enqueue = enqueue
