#!/usr/bin/env python3
# tok_check.py <tokenizers package dir> <tokenizer.json> <tok_dump binary> <text file>...
# Compares exl3conv's native BPE with Hugging Face tokenizers (special-token text encoded as plain text, as
# exllamav3 calibrates), file by file. CPU only.
import subprocess, sys
sys.path.insert(0, sys.argv[1])
import tokenizers
t = tokenizers.Tokenizer.from_file(sys.argv[2])
t.encode_special_tokens = True
bad = 0
for path in sys.argv[4:]:
    text = open(path, encoding="utf8", newline="").read()   # keep \r\n as is, like the C++ side
    ref = t.encode(text, add_special_tokens=False).ids
    got = [int(x) for x in subprocess.run([sys.argv[3], sys.argv[2], path], capture_output=True, text=True, check=True).stdout.split()]
    if got == ref:
        print(f"  same   {path}: {len(ref)} tokens")
    else:
        bad += 1
        i = next((i for i, (a, b) in enumerate(zip(got, ref)) if a != b), min(len(got), len(ref)))
        print(f"  DIFF   {path}: native {len(got)} vs HF {len(ref)} tokens, first difference at {i}: {got[i:i+5]} vs {ref[i:i+5]}")
sys.exit(1 if bad else 0)
