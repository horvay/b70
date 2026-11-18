# chat_check.py <model dir> <conversations.jsonl> <chat_dump output>: compare exl3eval's rendered token ids with
# Hugging Face's apply_chat_template, and check that the scored tokens are exactly the assistant replies plus <turn|>.
# Run with a Python that has transformers (CPU only).
import json, sys
from transformers import AutoTokenizer
tk = AutoTokenizer.from_pretrained(sys.argv[1])
convs = [json.loads(l) for l in open(sys.argv[2], encoding="utf-8") if l.strip()]
rows = [[tuple(map(int, p.split(":"))) for p in l.split()] for l in open(sys.argv[3])]
bad = 0
for c, r in zip(convs, rows):
    ids = tk.apply_chat_template(c["messages"], tokenize=True)
    ids = ids["input_ids"] if hasattr(ids, "keys") else ids
    ours = [i for i, _ in r]
    if ours != list(ids):
        bad += 1
        k = next((i for i in range(min(len(ours), len(ids))) if ours[i] != ids[i]), min(len(ours), len(ids)))
        print(f"{c['id']}: differs at {k}: ours {ours[k:k+6]} hf {list(ids)[k:k+6]}")
        continue
    scored = tk.decode([i for i, s in r if s])
    want = "".join(m["content"] + "<turn|>" for m in c["messages"] if m["role"] == "assistant")
    if scored != want:
        bad += 1; print(f"{c['id']}: scored text differs: {scored[:120]!r} vs {want[:120]!r}")
print(f"{len(rows)} conversations, {bad} mismatches")
