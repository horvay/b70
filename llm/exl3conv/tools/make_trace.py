#!/usr/bin/env python3
# Self-sampled calibration trace (exllamav3's sc_trace.py, notes/optimize-recipes.md §1), generated through a
# running B70 server (`b70 serve <a 6 bpw EXL3 of the model>`): the model answers exllamav3's 157 seed conversations
# (data/trace-conversations.json, turn by turn, follow-ups after its own answer), and every (context, reply) becomes
# one calibration row starting at BOS. Standard library only.
#
#   make_trace.py run  --url http://127.0.0.1:8123 --trace T.json [--target-tokens 512000] [--parallel 8] [--minutes M]
#   make_trace.py pack --trace T.json --out cal.safetensors [--rows 250] [--cols 2048] [--heldout eval.json]
#
# run: sampling as sc_trace: min_p 0.08 then temperature 0.8 (no top-k / top-p), at most 3072 new tokens, the
# template's enable_thinking on. The server's own chat template renders the context (prompt_token_ids) and its
# reasoning parser splits the answer, so a follow-up's context holds the previous answer without its thinking, as a
# chat client sends it (sc_trace leaves Gemma 4's thinking in, see the notes). Epochs repeat the conversations with
# new seeds until the trace holds --target-tokens context + reply tokens. Each reply is saved at once (T.json, via
# a temp file), so a run resumes where it stopped; with --minutes it stops starting conversations after M minutes,
# finishes the ones running and exits 99 if the target is not reached (b70q submit --chunked runs it again).
# pack: shuffle the rows (seed 0), the first --rows of them become int64 input_ids [rows, cols] (right-padded with 0)
# and lengths [rows], the tensors exl3conv --cal-data reads; the rest can go to --heldout, a held-out set for
# exl3eval --trace (same format as T.json).
import argparse, json, os, random, struct, sys, threading, time, urllib.request, zlib
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def save(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def post(url, body, timeout=1800):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def cmd_run(a):
    convs = json.load(open(a.conversations, encoding="utf-8"))["conversations"]
    tr = load(a.trace) or {"source": "exl3conv tools/make_trace.py", "url": a.url, "sampling": {
        "min_p": 0.08, "temperature": 0.8, "max_new_tokens": 3072, "enable_thinking": True, "seed": a.seed}, "rows": []}
    model = json.load(urllib.request.urlopen(a.url + "/v1/models"))["data"][0]["id"]
    tr["model"] = model
    lock = threading.Lock()
    done = {(r["epoch"], r["conversation"]) for r in tr["rows"] if r.get("last_turn")}
    total = lambda: sum(len(r["input_ids"]) + len(r["response_ids"]) for r in tr["rows"])
    t0 = time.time()
    stop = {"flag": False}

    def converse(epoch, ci):
        c = convs[ci]
        turns = [c["prompt"]] + list(c["followups"])
        msgs = []
        for ti, user in enumerate(turns):
            msgs.append({"role": "user", "content": user})
            seed = zlib.crc32(f"{a.seed}|{epoch}|{ci}|{ti}".encode()) & 0x7FFFFFFF
            r = post(a.url + "/v1/chat/completions", {
                "model": model, "messages": msgs, "max_tokens": 3072, "temperature": 0.8, "min_p": 0.08, "top_p": 1.0,
                "top_k": -1, "seed": seed, "chat_template_kwargs": {"enable_thinking": True},
                "return_prompt_token_ids": True, "return_token_ids": True, "skip_special_tokens": False})
            ch = r["choices"][0]
            resp = ch.get("response_token_ids") or []
            answer = (ch["message"].get("content") or "").strip()
            last = ti == len(turns) - 1 or not resp or not answer
            with lock:
                tr["rows"].append({"conversation": ci, "turn": ti, "epoch": epoch, "finish_reason": ch.get("finish_reason"),
                                   "input_ids": ch["prompt_token_ids"], "response_ids": resp, "last_turn": last})
                save(a.trace, tr)
            if last:
                return
            msgs.append({"role": "assistant", "content": answer})

    for epoch in range(a.max_epochs):
        todo = [ci for ci in range(len(convs)) if (epoch, ci) not in done]
        if not todo:
            continue
        # A conversation cut short by a stop is redone from its first turn: drop its partial rows.
        with lock:
            tr["rows"] = [r for r in tr["rows"] if r["epoch"] != epoch or r["conversation"] not in todo]
        failed = []

        def job(e, ci):
            # Checked when a conversation would start: past the turn's minutes or the target, it doesn't.
            if stop["flag"] or total() >= a.target_tokens or (a.minutes and time.time() - t0 > a.minutes * 60):
                stop["flag"] = True
                return
            try:
                converse(e, ci)
            except Exception as ex:   # a failed request: the conversation is redone next turn
                print(f"conversation {ci} (epoch {e}) failed: {ex}", flush=True)
                failed.append(ci)

        with ThreadPoolExecutor(a.parallel) as ex:
            for f in [ex.submit(job, epoch, ci) for ci in todo]:
                f.result()
        if failed:
            stop["flag"] = True
        n = total()
        print(f"epoch {epoch}: {len(tr['rows'])} rows, {n} tokens", flush=True)
        if n >= a.target_tokens:
            print("target reached", flush=True)
            return 0
        if stop["flag"]:
            return 99
    return 0


def cmd_pack(a):
    tr = load(a.trace)
    if tr is None:
        sys.exit(f"no trace at {a.trace}")
    rows = [r["input_ids"] + r["response_ids"] for r in tr["rows"] if r["response_ids"]]
    order = list(range(len(rows)))
    random.Random(0).shuffle(order)
    cal, rest = order[:a.rows], order[a.rows:]
    if len(cal) < a.rows:
        print(f"warning: only {len(cal)} rows", file=sys.stderr)
    n = len(cal)
    ids = bytearray(8 * n * a.cols)
    lens = bytearray(8 * n)
    for i, o in enumerate(cal):
        row = rows[o][:a.cols]
        struct.pack_into(f"<{len(row)}q", ids, 8 * i * a.cols, *row)
        struct.pack_into("<q", lens, 8 * i, len(row))
    hdr = {"input_ids": {"dtype": "I64", "shape": [n, a.cols], "data_offsets": [0, len(ids)]},
           "lengths": {"dtype": "I64", "shape": [n], "data_offsets": [len(ids), len(ids) + len(lens)]},
           "__metadata__": {"source": "exl3conv tools/make_trace.py", "model": str(tr.get("model"))}}
    h = json.dumps(hdr).encode()
    h += b" " * (-len(h) % 8)
    with open(a.out, "wb") as f:
        f.write(struct.pack("<Q", len(h))); f.write(h); f.write(ids); f.write(lens)
    tot = sum(min(len(rows[o]), a.cols) for o in cal)
    print(f"{a.out}: {n} rows x {a.cols}, {tot} tokens (mean {tot / max(n, 1):.0f}); {len(rest)} rows left over")
    if a.heldout:
        kept = [r for r in tr["rows"] if r["response_ids"]]
        save(a.heldout, {**{k: v for k, v in tr.items() if k != "rows"}, "rows": [kept[o] for o in rest]})
        print(f"{a.heldout}: {len(rest)} held-out rows")
    return 0


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--url", default="http://127.0.0.1:8123")
    r.add_argument("--trace", required=True)
    r.add_argument("--conversations", default=os.path.join(HERE, "../data/trace-conversations.json"))
    r.add_argument("--target-tokens", type=int, default=250 * 2048)
    r.add_argument("--parallel", type=int, default=8)
    r.add_argument("--max-epochs", type=int, default=4)
    r.add_argument("--minutes", type=float, default=0)
    r.add_argument("--seed", type=int, default=0)
    k = sub.add_parser("pack")
    k.add_argument("--trace", required=True)
    k.add_argument("--out", required=True)
    k.add_argument("--rows", type=int, default=250)
    k.add_argument("--cols", type=int, default=2048)
    k.add_argument("--heldout")
    a = p.parse_args()
    sys.exit(cmd_run(a) if a.cmd == "run" else cmd_pack(a))


if __name__ == "__main__":
    main()
