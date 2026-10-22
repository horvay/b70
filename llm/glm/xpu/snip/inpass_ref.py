# Is the in-pass checkpoint difference ordinary batch-size rounding? Old method with chunk 4096 vs chunk 1024 (both
# legitimate runs of the unchanged code path) for the same prompt.
exec(open("/w/xpu/snip/inpass_diff.py").read().split("try:")[0])
try:
    gen.max_chunk_size = 4096; a = ckpt(False)
    gen.max_chunk_size = 1024; b = ckpt(False)
    cmp(a, b, "old chunk 4096 vs old chunk 1024")
finally:
    gen.max_chunk_size = 4096
    J._INPASS_CKPT = True
