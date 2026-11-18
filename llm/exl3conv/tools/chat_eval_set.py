#!/usr/bin/env python3
# Build exl3eval's chat set from OpenAssistant oasst2 (human-written conversations, Apache-2.0):
#
#   hf download OpenAssistant/oasst2 2023-11-05_oasst2_ready.trees.jsonl.gz --repo-type dataset --local-dir D
#   python3 -I tools/chat_eval_set.py D/2023-11-05_oasst2_ready.trees.jsonl.gz data/oasst2-eval.jsonl [N]
#
# Each tree becomes one conversation by following its best-ranked reply at every turn, ending on an assistant turn.
# Conversations of 1,500-7,000 characters are kept (enough tokens to score, short enough for 2,048-token rows), and N
# of them (default 100) are picked by the sha256 of the tree id, so the set is fixed for a given dump. All languages
# stay in, as in the dump. Output: one {"id", "lang", "messages": [{"role", "content"}]} per line.
import gzip, hashlib, json, sys

src, dst = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 100


def best(replies):
    ok = [r for r in replies if not r.get("deleted") and r.get("review_result", True) is not False]
    return min(ok, key=lambda r: (r["rank"] if r.get("rank") is not None else 1 << 30)) if ok else None


picked = []
with gzip.open(src, "rt", encoding="utf-8") as f:
    for line in f:
        t = json.loads(line)
        node, msgs = t["prompt"], []
        while node is not None:
            msgs.append({"role": "user" if node["role"] == "prompter" else "assistant", "content": node["text"].strip()})
            node = best(node.get("replies") or [])
        while msgs and msgs[-1]["role"] != "assistant":
            msgs.pop()
        chars = sum(len(m["content"]) for m in msgs)
        if len(msgs) < 2 or not 1500 <= chars <= 7000:
            continue
        key = hashlib.sha256(t["message_tree_id"].encode()).hexdigest()
        picked.append((key, {"id": t["message_tree_id"], "lang": t["prompt"].get("lang"), "messages": msgs}))

picked.sort(key=lambda p: p[0])
with open(dst, "w", encoding="utf-8", newline="\n") as f:
    for _, c in picked[:n]:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")
langs = {}
for _, c in picked[:n]:
    langs[c["lang"]] = langs.get(c["lang"], 0) + 1
print(f"{min(n, len(picked))} of {len(picked)} eligible conversations -> {dst}; languages {dict(sorted(langs.items(), key=lambda kv: -kv[1]))}")
