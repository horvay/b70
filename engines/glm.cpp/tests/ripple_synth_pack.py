#!/usr/bin/env python3
"""A Ripple-65B-shaped EXL3 pack of pseudo-random trellises, to measure load, VRAM and speed before a real pack exists. Its logits are
noise. SAGE-like K: gate / up K_MLP, down K_DOWN, attention K_ATTN, head 16-bit fp16 unless --head-k.
   ripple_synth_pack.py <out dir> <config dir> <parts.safetensors> [K_MLP K_DOWN K_ATTN]
The norm weights, lambdas and the embedding are copied from the first layers of the real checkpoint (cycled)."""
import json, os, re, shutil, struct, sys
import numpy as np

out, cfgd, parts = sys.argv[1:4]
K_MLP, K_DOWN, K_ATTN = (float(x) for x in (sys.argv[4:7] or (2.5, 3, 4)))
os.makedirs(out, exist_ok=True)
for f in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja"):
    shutil.copy(os.path.join(cfgd, f), out)
with open(parts, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    src = json.loads(f.read(n))
base = 8 + n
def raw(name):
    m = src[name]
    with open(parts, "rb") as f:
        f.seek(base + m["data_offsets"][0])
        return f.read(m["data_offsets"][1] - m["data_offsets"][0])
rng = np.random.default_rng(1)
shards, cur, size = [], {}, 0
def add(name, dtype, shape, data):
    cur[name] = (dtype, shape, data)
def flush(i):
    head, off, fn = {}, 0, f"model-{i:05d}.safetensors"
    for k, (dt, sh, d) in cur.items():
        z = d if isinstance(d, int) else len(d)
        head[k] = {"dtype": dt, "shape": sh, "data_offsets": [off, off + z]}
        off += z
    hb = json.dumps(head).encode(); hb += b" " * (-(8 + len(hb)) % 8)
    with open(os.path.join(out, fn), "wb") as o:
        o.write(struct.pack("<Q", len(hb))); o.write(hb)
        for k, (dt, sh, d) in cur.items():
            if isinstance(d, int):
                left = d
                while left:
                    c = min(left, 1 << 28)
                    o.write(rng.integers(0, 65536, c // 2, dtype=np.uint16).tobytes()); left -= c
            else:
                o.write(d)
    shards.append((fn, list(head)))
    cur.clear()
def exl3(prefix, i_, o_, K):
    words = int(16 * K)
    add(prefix + ".trellis", "I16", [i_ // 16, o_ // 16, words], (i_ // 16) * (o_ // 16) * words * 2)
    # output scales near what real packs hold, so the activations stay finite
    add(prefix + ".suh", "F16", [i_], np.full(i_, 0.5, dtype=np.float16).tobytes())
    add(prefix + ".svh", "F16", [o_], np.full(o_, 0.002, dtype=np.float16).tobytes())
    add(prefix + ".mul1", "I32", [], struct.pack("<I", 0x83DCD12D))
add("model.embed_tokens.weight", "BF16", [100352, 5120], raw("model.embed_tokens.weight"))
add("model.norm.weight", "BF16", [5120], raw("model.norm.weight"))
add("lm_head.weight", "F16", [100352, 5120], (rng.standard_normal(100352 * 5120, dtype=np.float32) * 0.02).astype(np.float16).tobytes())
flush(1)
for li in range(80):
    s = f"model.layers.{li % 5}."; p = f"model.layers.{li}."
    for t in ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"):
        add(p + t + ".weight", "BF16", [5120], raw(s + t + ".weight"))
    for t in ("q_norm", "k_norm"):
        add(p + "self_attn." + t + ".weight", "BF16", [128], raw(s + "self_attn." + t + ".weight"))
    add(p + "self_attn.value_res_lambda", "BF16", [1], raw(s + "self_attn.value_res_lambda"))
    a = p + "self_attn."
    exl3(a + "q_proj", 5120, 10240, K_ATTN); exl3(a + "g_proj", 5120, 10240, K_ATTN)
    exl3(a + "k_proj", 5120, 1024, K_ATTN); exl3(a + "v_proj", 5120, 1024, K_ATTN)
    exl3(a + "o_proj", 10240, 5120, K_ATTN)
    exl3(p + "mlp.gate_proj", 5120, 40960, K_MLP); exl3(p + "mlp.up_proj", 5120, 40960, K_MLP)
    exl3(p + "mlp.down_proj", 40960, 5120, K_DOWN)
    flush(li + 2)
    print("layer", li, flush=True)
json.dump({"weight_map": {k: fn for fn, ks in shards for k in ks}}, open(os.path.join(out, "model.safetensors.index.json"), "w"))
