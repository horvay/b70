#!/usr/bin/env python3
"""Pure-Python port of HF Qwen4ExpTextNgramEmbedding's id computation (modeling_qwen4_exp.py, _shift_right_ignore_eos
and forward), line for line, for checking the C++ hashing. Reads rows of int32 tokens from a binary file
(T int32 per row), prints the 16 ids of every token as text. CPU, stdlib only.
    ngram_hash_ref.py <tokens.bin> <T> <eos> m0 m1 m2 <sizes,comma> <offsets,comma>"""
import struct, sys

def shift_right_ignore_eos(tok, shift, eos):
    if shift == 0:
        return tok
    n = len(tok)
    eos_pos = [i if tok[i] == eos else -1 for i in range(n)]
    cm, run = [], -1
    for v in eos_pos:
        run = max(run, v); cm.append(run)
    prev = [-1] + cm[:-1]
    out = []
    for i in range(n):
        seg_start = prev[i] + 1
        pis = i - seg_start
        src = i - shift
        shifted = tok[max(src, 0)]
        out.append(shifted if (pis >= shift and src >= 0) else eos)
    return out

def main():
    path, T, eos = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    m = [int(x) for x in sys.argv[4:7]]
    sizes = [int(x) for x in sys.argv[7].split(',')]
    offs = [int(x) for x in sys.argv[8].split(',')]
    data = open(path, 'rb').read()
    rows = len(data) // (4 * T)
    out = sys.stdout
    for r in range(rows):
        ids = list(struct.unpack_from('<%di' % T, data, r * 4 * T))
        hist = [eos, eos] + ids
        sh = [shift_right_ignore_eos(hist, s, eos) for s in range(3)]
        for p in range(2, len(hist)):
            mix2 = (sh[0][p] * m[0]) ^ (sh[1][p] * m[1])
            mix3 = mix2 ^ (sh[2][p] * m[2])
            out.write(' '.join(str((mix2 if h < 8 else mix3) % sizes[h] + offs[h]) for h in range(16)) + '\n')

main()
