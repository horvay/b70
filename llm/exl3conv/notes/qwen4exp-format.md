# EXL3 Qwen3.8-Flash-Next (`qwen4_exp`): the on-disk format

What an EXL3 conversion of Qwen3.8-Flash-Next must look like on disk, so that it loads in exllamav3 and in the B70
server (SGLang 0.5.20-xpu + trellis-serve `ft-xpu-vision` @ 72fd2d4 + `../../trellis-serve.patch`). It is read off
real exllamav3-made models (safetensors headers fetched with HTTP range requests, small JSON files in full) and
checked against the exllamav3 sources (`../dsv41/exllamav3`, `architecture/qwen4_exp.py`, `conversion/ngram.py`,
`modules/block_sparse_mlp.py`) and the B70 server sources (`../state/trellis-serve/xpu/exl3xpu/*.py`, `csrc/*.sycl`).
The per-linear EXL3 format itself (trellis/suh/svh/mul1, packing, tile order) is in `exl3-algorithm.md` section 5;
this file only adds what is specific to this model.

## 0. Models inspected

| repo (branch) | exllamav3 | bits / head / mtp / vision / n-gram | total tensor bytes |
|---|---|---|---|
| **Lygodactylus/Qwen3.8-Flash-Next-Uncensored-exl3-3bpw** (the B70 model, "reference" below) | 1.4.6 | 3.05 / 5 / 3 / 5 / 3 | 72.18 GB (67.2 GiB) |
| turboderp/Qwen3.8-Flash-Next-exl3 (`3.05bpw_h5_ng5`) | 1.4.4 | 3.05 / 5 / 3 / 5 / 5 | 84.95 GB |
| turboderp (`2.05bpw_h4_ng4`) | 1.4.4 | 2.05 / 4 / 2 / 4 / 4 | 62.62 GB |
| turboderp (`4.05bpw_h6_ng6`) | 1.4.4 | 4.05 / 6 / 4 / 6 (side file) / 6 | 107.27 GB |
| turboderp (`1.66bpw_h4_ng3`, self-calibrated) | 1.6.3 | 1.66 / 4 / 4 / 4 / 3, **embed 4** | 50.02 GB |
| turboderp (`2.18bpw_h6_ng8`, self-calibrated) | 1.6.3 | 2.18 / 6 / 2 / 6 / 8 | 90.50 GB |
| Lygodactylus ...-exl3-2bpw | 1.6.3 | 2.05 / 4 / 2 / 4 / 4 | 62.65 GB |
| Lygodactylus ...-exl3-4bpw | 1.4.6 | 4.05 / 6 / 8 / **16** / 6 | 108.93 GB |
| groxaxo/Qwen3.8-Flash-Next-exl3-3.50bpw_hq_h6_ng6 | 1.5.0 | 3.52 / 6 / 4 / 6 / 6 | 99.04 GB |
| r0b0tlab/Qwen3.8-Flash-Next-EXL3-2.50bpw | 1.5.0 | 2.52 / 6 / 4 / 6 / 3 | 64.38 GB |

turboderp's other branches (`2.68/3.18/3.68/4.18bpw_h6_ng8`, `5.05/6.05bpw_h6_ng6`) and ~40 more community quants
exist (HF search `Qwen3.8-Flash-Next exl3`); not inspected. BF16 source for the name mapping:
dongfangshuo/Dongfangshuo-Qwen3.8-Flash-Next (131 shards, 360.5 GB, index identical to the local copy in
`~/models/llm/Dongfangshuo-Qwen3.8-Flash-Next/`).

All quants share one tensor naming scheme. They differ only in K per tensor, in the n-gram K, and in a few
version-dependent details (section 7).

Model facts used below (`text_config`): hidden 2560, 48 layers (`layer_types`: every 4th, i.e. layers 3, 7, ..., 47,
is `full_attention`, the other 36 are `linear_attention`), 512 routed experts (10 per token), moe and shared-expert
intermediate 640, 24 q heads / 2 kv heads of 256 (q_proj output doubled by the interleaved output gate), GDN 16 key
heads / 48 value heads of 128, `hc_count` 4 (hyper-connection streams of 2560, so 10240 wide), `hc_lowrank` 320,
vocab 248320, untied head, PLE at `ple_layer_ids = [2]` (1-based, so module key `layers.1.ple`), n-gram rows of 160,
16 hash heads, one MTP layer, a 27-block vision tower (hidden 1152, intermediate 4304).

## 1. File inventory (reference)

```
          1696  .gitattributes             (HF)
          3235  LICENSE
          3814  README.md
          8952  chat_template.jinja
          5081  config.json
           202  generation_config.json
       3353259  merges.txt
    7761633442  model-00001-of-00007.safetensors   embed_tokens, layers 0-5 (incl. PLE + layer 1)
    8128444878  model-00002-of-00007.safetensors   layers 6-13
    8128469620  model-00003-of-00007.safetensors   layers 14-21
    8128469620  model-00004-of-00007.safetensors   layers 22-29
    8128469620  model-00005-of-00007.safetensors   layers 30-37
    8128469620  model-00006-of-00007.safetensors   layers 38-45
    3972763005  model-00007-of-00007.safetensors   layers 46-47, mixer, lm_head, mtp.*, model.visual.*
      32763270  model.safetensors.index.json
   19840121920  ngram_embedding.safetensors        the n-gram table (not in the index)
           390  preprocessor_config.json
      96643134  quantization_config.json
      12809320  tokenizer.json
         17928  tokenizer_config.json
           385  video_preprocessor_config.json
       6722759  vocab.json
```

- **Main shards**: 8192 MiB budget, whole modules per shard (a shard can start in the middle of a layer: shard 2
  starts with layer 6's experts, and layer 5's `attn_hyper_connection` closes shard 1). Headers are minified JSON,
  no `__metadata__`, not padded (lengths 4,913,730 / 6,561,910 / ... bytes, not multiples of 8). Tensor order inside a
  shard is irrelevant to every loader; the reference lists a layer's experts in string order (`experts.10` first).
- **`model.safetensors.index.json`** (`json.dumps(indent=4)`): 304,108 entries = every main-shard tensor; the 132
  n-gram tensors are **not** listed. `metadata.total_size` = 52,402,113,436, which is the main-shard tensor bytes
  (52,336,434,076) plus the PLE dense tensors (65,679,360) counted a second time: an exllamav3 quirk, informational
  only. Write the true sum.
- **`ngram_embedding.safetensors`**: its own safetensors file, found by name (section 4). 132 tensors, header padded
  to a multiple of 8 with spaces, **with** `__metadata__`.
- **Copied files**: tokenizer, chat template, generation/preprocessor configs, LICENSE, README. The converter rule
  (`exl3-algorithm.md` 6.5) copies every top-level regular file except safetensors/config/index. The Dongfangshuo
  source also has `modeling_qwen4_exp.py` (harmless), `LICENSE-QWEN`, `NOTICE` and a 907 MB
  `mmproj-Qwen3.8-Flash-Next-BF16.gguf` at the top level: **skip `*.gguf`** (and the `lora_checkpoints/`,
  `IQ*/`, `UD-*/` subdirectories are not copied anyway).
- `preprocessor_config.json` must be present: exllamav3 enables the vision model only if it exists
  (`qwen4_exp.py`, `has_vision`).

### 1.1 config.json

The source `config.json` (`indent=4`), unchanged, plus `quantization_config` appended at the end (diffing BF16 vs
EXL3 config shows no other change):

```json
"quantization_config": {
    "quant_method": "exl3",
    "version": "1.4.6",
    "bits": 3.05,
    "head_bits": 5,
    "calibration": {"rows": 250, "cols": 2048},
    "out_scales": "always",
    "codebook": "mul1",
    "vision_bits": 5,
    "mtp_bits": 3
}
```

- There is **no `ngram_bits` and no `embed_bits` key** in any inspected config, even turboderp's ng5/ng8 quants and
  the 1.66 bpw quant whose embedding is quantized. The n-gram K lives only in the n-gram file's metadata; the
  embedding K only in the tensor shape.
- `vision_bits` is absent when the tower is unquantized (Lygodactylus 4bpw: vision 16) and also in turboderp's 4.05,
  whose quantized tower sits in a side file.
- `out_scales` is `"always"` up to 1.5.0 and `"auto"` in the 1.6.3 quants; `calibration.rows` is 217 in turboderp's
  self-calibrated quants. Nothing reads this block at load time (exllamav3 and the B70 server detect EXL3 per tensor).

### 1.2 quantization_config.json

`{quant_method, version, bits, head_bits, calibration, out_scales, codebook, vision_bits, mtp_bits, tensor_storage}`
(`indent=4`, 96.6 MB). `tensor_storage` has **74,398 entries, main text model only** (no `mtp.*`, no `model.visual.*`),
in module order:

```
model.language_model.embed_tokens
hc_expand                                               {"stored_tensors": {}}
per layer L:
  [L == 1 only, before the block:]
  model.language_model.layers.1.ple.ple_embedding.ngram_embedding   (132 stored tensors, the n-gram file's)
  model.language_model.layers.1.ple.key_proj / value_proj / norm_key / norm_query / norm_conv   (ple.conv1d is NOT listed)
  ...layers.L.attn_hyper_connection                     (4 tensors)
  linear layers:  linear_attn.in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, out_proj, norm
  full layers:    self_attn.indexer.index_qk_proj, indexer.q_layernorm, indexer.k_layernorm,
                  self_attn.q_proj, k_proj, v_proj, o_proj, q_norm, k_norm
  ...layers.L.mlp_hyper_connection
  ...layers.L.mlp.gate, mlp.shared_expert_gate
  ...layers.L.mlp.experts.{0..511}.{up_proj, gate_proj, down_proj}     (up before gate)
  ...layers.L.mlp.shared_expert.{up_proj, gate_proj, down_proj}
model.language_model.hyper_connection_mixer             (3 tensors)
lm_head
```

Entry format (EXL3 linear):

```json
"model.language_model.layers.5.mlp.experts.3.gate_proj": {
    "stored_tensors": {
        "...gate_proj.suh":     {"shape": [2560], "n_bytes": 5120, "dtype": "torch.float16"},
        "...gate_proj.svh":     {"shape": [640], "n_bytes": 1280, "dtype": "torch.float16"},
        "...gate_proj.mul1":    {"shape": [], "n_bytes": 4, "dtype": "torch.int32"},
        "...gate_proj.trellis": {"shape": [160, 40, 48], "n_bytes": 614400, "dtype": "torch.int16"}
    },
    "quant_format": "exl3", "bits_per_weight": 3, "mul1_multiplier": 2212286765
}
```

Non-EXL3 modules have only `stored_tensors`. Informational only; write it the same way.

## 2. K distribution (bits per weight per tensor)

K = `trellis.shape[-1] / 16`. Integer K gives a last dim of 16, 32, 48, 64, 80, 96, 128. Fractional K is
**visible in the shape**: K = 1.5 -> 24, K = 2.5 -> 40 (K = 3.5 -> 56). The reference has **no fractional K**.

### 2.1 Reference, 3.05 bpw (1.4.6)

| tensors | K |
|---|---|
| routed experts, layer 0 (1,536 = 512 x gate/up/down) | **4** |
| routed experts, layers 1-47 (72,192) | **3** |
| GDN `in_proj_qkv`, `in_proj_z`, `out_proj`, layers 0 and 1 | 5 |
| GDN `in_proj_qkv`, `in_proj_z`, `out_proj`, the other 34 linear layers | 4 |
| full attention `q/k/v/o_proj` and `indexer.index_qk_proj`, all 12 full layers | 5 |
| shared expert gate/up/down, layer 0 | 5 |
| shared expert, layers 1-46, and layer 47 gate/up | 4 |
| shared expert down, layer 47 | 5 |
| `lm_head` | 5 (`head_bits`) |
| all MTP linears incl. its 512 experts | 3 (`mtp_bits`) |
| all vision linears | 5 (`vision_bits`) |
| n-gram table | 3 (default `round(bits)`) |

Budgeted ("bits") trellis bits / padded weights = exactly 3.0500. The mechanism (1.4.6 `create_q_strategy`): every
budgeted linear starts at `floor(bits)` = 3 (integer only in 1.4.x, no half steps), then repeated passes bump whole
qgroups by +1 while the budget allows, in order of priority (full attention 4, GDN and shared expert 3, routed
experts lower), then distance from the nearer end of the stack. The 0.05 bpw over 3 is about 6 Gbit, enough for
attention and shared experts (+1 or +2) and one layer of experts (layer 0, +1); layer 47's experts didn't fit.
`select_hq_bits = 2` on attention and shared experts applies only with `--hq`.

### 2.2 Other quants

| quant | routed experts | GDN | full attn | indexer | shared | head | mtp experts / mtp attn | vision | ngram |
|---|---|---|---|---|---|---|---|---|---|
| td 3.05 h5 ng5 (1.4.4) | 3 all layers | 5 | 5 | 3 | 5 | 5 | 3 / 5 | 5 | 5 |
| td 2.05 h4 ng4 | 2 all | 4 | 4 | 2 | 4 | 4 | 2 / 4 | 4 | 4 |
| td 4.05 h6 ng6 | 4 all | 6 | 6 | 4 | 6 | 6 | 4 / 6 | 6 | 6 |
| td 1.66 h4 ng3 (1.6.3) | **1.5** (L4-43), 2 (L0-3, 44-47) | 5 | 5 | 5 | 5 | 4 | 4 / 8 | 4 | 3, embed 4 |
| td 2.18 h6 ng8 (1.6.3) | **2** (L4-43), **2.5** (L0-3, 44-47) | 6 | 6 | 6 | 6 | 6 | 2 / 6 | 6 | 8 |
| Lygo 2bpw (1.6.3) | 2, **2.5** in L0, L1, L47 | 2.5 / 3 | 3 | 3 | 2.5 / 3 | 4 | 2 / 2 | 4 | 4 |
| Lygo 4bpw (1.4.6) | 4, 5 in L0 | 5, 6 in L0-1 | 6 | 6 | 5 / 6 | 6 | 8 / 8 | 16 | 6 |
| groxaxo 3.52 hq (1.5.0) | 4 (L0-11, 37-47), 3 (L12-36) | 5 | 5 | 5 | 5 | 6 | 4 / 6 | 6 | 6 |
| r0b0tlab 2.52 (1.5.0) | 3 (L0-11, 37-47), 2 (L12-36) | 4 | 4 | 4 | 4 | 6 | 4 / 6 | 6 | 3 |

Always true: gate, up and down of one routed expert have the same K, and all 512 experts of a layer have the same K
(K changes only per layer). The router `mlp.gate`, `shared_expert_gate`, GDN `in_proj_a/b` and the PLE projections
are never quantized.

## 3. Per-layer tensors

Below, `L.` stands for `model.language_model.layers.L.`. Every EXL3 linear `P` is the quartet `P.suh` F16 `[in]`,
`P.svh` F16 `[out]`, `P.mul1` I32 `[]` (= -2082680531), `P.trellis` I16 `[in/16, out/16, 16K]`; written below as
"EXL3 in x out". Unquantized linears are `P.weight` in HF orientation `[out, in]`.

### 3.1 Linear-attention (GatedDeltaNet) layer, e.g. layer 4 (reference K)

| tensor | dtype | shape | bytes |
|---|---|---|---|
| `L.attn_hyper_connection.input_mix_weight_down.weight` | F16 | [320, 10240] | 6,553,600 |
| `L.attn_hyper_connection.input_mix_weight_up.weight` | F16 | [10240, 320] | 6,553,600 |
| `L.attn_hyper_connection.block_inject_weight.weight` | F16 | [4, 10240] | 81,920 |
| `L.attn_hyper_connection.hc_norm.weight` | F16 | [10240] | 20,480 |
| `L.linear_attn.in_proj_qkv` | EXL3 2560 x 10240 | trellis [160, 640, 64] | 13,107,200 + scales |
| `L.linear_attn.in_proj_z` | EXL3 2560 x 6144 | trellis [160, 384, 64] | 7,864,320 + scales |
| `L.linear_attn.out_proj` | EXL3 6144 x 2560 | trellis [384, 160, 64] | 7,864,320 + scales |
| `L.linear_attn.in_proj_a.weight` | F16 | [48, 2560] | 245,760 |
| `L.linear_attn.in_proj_b.weight` | F16 | [48, 2560] | 245,760 |
| `L.linear_attn.conv1d.weight` | BF16 | [10240, 1, 4] | 81,920 |
| `L.linear_attn.A_log` | BF16 | [48] | 96 |
| `L.linear_attn.dt_bias` | BF16 | [48] | 96 |
| `L.linear_attn.norm.weight` | BF16 | [128] | 256 |
| `L.mlp_hyper_connection.*` | F16 | same four as attn_hyper_connection | 13,209,600 |
| `L.mlp.gate.weight` (router) | F16 | [512, 2560] | 2,621,440 |
| `L.mlp.shared_expert_gate.weight` | F16 | [1, 2560] | 5,120 |
| `L.mlp.shared_expert.gate_proj` / `up_proj` | EXL3 2560 x 640 | trellis [160, 40, 64] | 819,200 each |
| `L.mlp.shared_expert.down_proj` | EXL3 640 x 2560 | trellis [40, 160, 64] | 819,200 |
| `L.mlp.experts.<e>.gate_proj` / `up_proj`, e = 0..511 | EXL3 2560 x 640 | trellis [160, 40, 48] | 614,400 each |
| `L.mlp.experts.<e>.down_proj` | EXL3 640 x 2560 | trellis [40, 160, 48] | 614,400 |

Scales: `in_proj_qkv` suh [2560] svh [10240]; `in_proj_z` suh [2560] svh [6144]; `out_proj` suh [6144] svh [2560];
expert/shared gate and up suh [2560] svh [640]; down suh [640] svh [2560].

There is no `input_layernorm`/`post_attention_layernorm`: the hyper-connection sites replace them, and `hc_norm` is
their norm.

### 3.2 Full-attention (QSA) layer, e.g. layer 3

Same hyper-connection, router, shared-expert and expert tensors as 3.1; the attention part is:

| tensor | dtype | shape | bytes |
|---|---|---|---|
| `L.self_attn.q_proj` | EXL3 2560 x 12288 | trellis [160, 768, 80] | 19,660,800 |
| `L.self_attn.k_proj` | EXL3 2560 x 512 | trellis [160, 32, 80] | 819,200 |
| `L.self_attn.v_proj` | EXL3 2560 x 512 | trellis [160, 32, 80] | 819,200 |
| `L.self_attn.o_proj` | EXL3 6144 x 2560 | trellis [384, 160, 80] | 9,830,400 |
| `L.self_attn.q_norm.weight` / `k_norm.weight` | BF16 | [256] | 512 each |
| `L.self_attn.indexer.index_qk_proj` | EXL3 2560 x 640 | trellis [160, 40, 80] | 1,024,000 |
| `L.self_attn.indexer.q_layernorm.weight` / `k_layernorm.weight` | BF16 | [128] | 256 each |

q, k, v stay separate tensors (no fused qkv on disk); `q_proj`'s 12288 outputs are 24 heads x 256 x (query, gate)
interleaved, exactly as in the BF16 weight.

### 3.3 Routed experts

**One EXL3 quartet per expert and matrix**, `L.mlp.experts.<e>.{gate,up,down}_proj.{suh,svh,mul1,trellis}`,
e = 0..511: 12 tensors per expert, 6,144 per layer, 294,912 in the main model (+6,144 in MTP). No stacked or fused
tensors. Per expert at K: trellis 3 x 204,800 K bytes, plus 19,212 bytes of suh/svh/mul1 (K = 3: 1,862,412 B; K = 4:
2,476,812 B).

Measured on the reference (layer 2): the gate and up `suh` of one expert have identical signs, and so do the `suh` of
different experts of the layer (all experts' gate/up share one input Hessian and therefore one random sign vector);
the magnitudes differ (rms 0.0143 vs 0.0141). `svh` rms 1.0. Any sign choice is valid; this is just what exllamav3
does.

### 3.4 PLE layer (layer 1 only)

| tensor | dtype | shape | file |
|---|---|---|---|
| `layers.1.ple.key_proj.weight` | F16 | [10240, 2560] | shard 1 |
| `layers.1.ple.value_proj.weight` | F16 | [2560, 2560] | shard 1 |
| `layers.1.ple.conv1d.weight` | F16 | [10240, 1, 4] | shard 1 |
| `layers.1.ple.norm_key.weight` / `norm_query.weight` / `norm_conv.weight` | BF16 | [10240] | shard 1 |
| `layers.1.ple.ple_embedding.ngram_embedding.*` | | section 4 | ngram file |

The PLE projections are plain fp16 (not EXL3). turboderp's 4.05 stores the three PLE norms as F16 instead of BF16.

## 4. The n-gram table: `ngram_embedding.safetensors`

Prefix `P = model.language_model.layers.1.ple.ple_embedding.ngram_embedding`. Header `__metadata__` (all strings):

```json
{"format": "exl3_ngram_trellis", "version": "1", "K": "3", "codebook": "mul1",
 "codebook_scale": "heuristic(gamma=3.20, hi=0.98)", "row_dim": "160", "rows": "320001536",
 "shard_rows": "2500012", "source": "/data/models/qwen38-flash-next-uncensored-bf16"}
```

Tensors, in file order (small ones first, then the shards back to back, contiguous):

| tensor | dtype | shape | value / note |
|---|---|---|---|
| `P.head_bias` | F16 | [16, 160] | per-hash-head channel mean, subtracted before quantizing |
| `P.head_offsets` | I64 | [16] | copy of BF16 `ngram_heads_offsets`: 0, 20000003, 40000026, ... 300001275 |
| `P.head_vocab_sizes` | I64 | [16] | copy of BF16 `ngram_heads_vocab_sizes`: 20000003, 20000023, ... 20000171 |
| `P.layer_multipliers` | I64 | [3] | copy of BF16 `layer_multipliers`: 23703573157769, 20109073645365, 8052911324071 |
| `P.shard_0.trellis` ... `P.shard_127.trellis` | I16 | [2500012, 1 + 10K] each | 128 shards = 320,001,536 rows (the BF16 shard row count, kept) |

Row record = `1 + 10K` uint16 words (31 at K = 3): word 0 is the row's fp16 scale, words 1..10K are one 160-position
tail-biting mul1 trellis ring (160K bits, same MSB-first u32-word, little-endian-halves packing as the 256 rings).
Decode: `row[i] = mul1(state_i) * scale + head_bias[head(row)]`; `head(row)` from `head_offsets`. No Hadamard, no
suh/svh. Quantization (`conversion/ngram.py`): subtract the head bias, pre-scale each row to rms `cs` with the per-row
heuristic `cs = clamp(gamma / (absmax/rms), 0.55, hi)` (`(gamma, hi)` per K: 3 -> (3.2, 0.98), 4 -> (3.0, 0.98),
5 -> (3.0, 0.95), 6 -> (3.0, 0.92), 8 -> (3.0, 0.86)), run the length-160 Viterbi, then store the least-squares
scale. Measured on the reference: row 0 scale 0.01598, row 1 0.00935; `head_bias[0][:3]` = 0.00117, 0.0000998,
0.000387.

Format notes:

- The file is not in `model.safetensors.index.json`; exllamav3 and the B70 server open it by name.
- It is swappable between quants of the same base weights (turboderp says so). It depends only on the BF16 table
  (calibration-free), so we can reuse an existing one for a re-quantization of the **same** checkpoint; a fine-tune
  that changed the table needs its own.
- turboderp's 4.05 (1.4.4) still has the older single-tensor layout `P.trellis [320001536, 61]` and no `shard_rows`:
  exllamav3 reads both, **the B70 server only the `shard_N.trellis` form**.
- Header: `json.dumps(separators=(",", ":"))` padded with spaces to a multiple of 8.

## 5. Global tensors

| tensor | dtype | shape | note |
|---|---|---|---|
| `model.language_model.embed_tokens.weight` | BF16 | [248320, 2560] | copied unchanged (1.27 GB) |
| `model.language_model.hyper_connection_mixer.input_mix_weight_down.weight` | F16 | [320, 10240] | the final stream collapse; **there is no final `norm`** |
| `...hyper_connection_mixer.input_mix_weight_up.weight` | F16 | [10240, 320] | |
| `...hyper_connection_mixer.hc_norm.weight` | F16 | [10240] | (no block_inject: `use_combine = False`) |
| `lm_head` | EXL3 2560 x 248320 | trellis [160, 15520, 16 x head_bits] | K = 5: 397 MB. HF name `lm_head.weight` (BF16, untied) |

With `head_bits = 16` the head would be `lm_head.weight` F16 [248320, 2560] (1.27 GB).

**Quantized embedding** (1.6.3 `--embed_bits`, only in turboderp's 1.66): `embed_tokens.trellis` I16
[248320, 650] (10 groups of 256 columns x (1 + 16 x 4) words: per group an fp16 scale word + a 256-ring) and
`embed_tokens.signs` F16 [10, 256]; no `.weight`. **The B70 server has no support for it**; keep the BF16 copy.

### 5.1 MTP (all under `mtp.`, in the last shard)

| tensor | dtype | shape |
|---|---|---|
| `mtp.fc_embedding`, `mtp.fc_hidden` | EXL3 2560 x 2560 | trellis [160, 160, 16K] |
| `mtp.pre_fc_norm_embedding.weight` | BF16 | [2560] |
| `mtp.pre_fc_norm_hidden.weight` | **F16** | [10240] |
| `mtp.hyper_connection_mixer.{input_mix_weight_down, input_mix_weight_up, hc_norm}.weight` | F16 | as the main mixer |
| `mtp.layers.0.*` | | a complete full-attention block, same names and shapes as 3.2 + 3.3, prefix `mtp.layers.0.` |

All MTP linears are EXL3 at `mtp_bits` (512 experts too: 0.95 GB at K = 3); quantized uncalibrated. 1.4.4 forgot the
MTP mixer: turboderp's 3.05/2.05 ship it in an extra `mtp_hyper_connection_mixer_patch.safetensors` (3 tensors, not
in the index). Write it in the main shards and the index, as 1.4.6+ does.

### 5.2 Vision tower (`model.visual.*`, last shard)

Quantized at `vision_bits` (reference K = 5), uncalibrated:

| tensor | dtype | shape |
|---|---|---|
| `model.visual.blocks.<b>.attn.{q_proj,k_proj,v_proj}` | EXL3 1152 x 1152 + `.bias` F16 [1152] | trellis [72, 72, 16K] |
| `model.visual.blocks.<b>.attn.proj` | EXL3 1152 x 1152 + `.bias` F16 [1152] | |
| `model.visual.blocks.<b>.attn.qkv.weight` / `.bias` | **BF16** [3456, 1152] / [3456] | stray copy of the source |
| `model.visual.blocks.<b>.mlp.linear_fc1` | EXL3 1152 x **4352** + `.bias` F16 [4352] | 4304 padded to 4352, trellis [72, 272, 16K] |
| `model.visual.blocks.<b>.mlp.linear_fc2` | EXL3 **4352** x 1152 + `.bias` F16 [1152] | suh [4352] |
| `model.visual.blocks.<b>.norm1/norm2.{weight,bias}` | F16 | [1152] |
| `model.visual.merger.linear_fc1` | EXL3 4608 x 4608 + bias F16 [4608] | trellis [288, 288, 16K] |
| `model.visual.merger.linear_fc2` | EXL3 4608 x 2560 + bias F16 [2560] | trellis [288, 160, 16K] |
| `model.visual.merger.norm.{weight,bias}` | F16 | [1152] |
| `model.visual.patch_embed.proj.weight` / `.bias` | F16 | [1152, 3, 2, 16, 16] / [1152] |
| `model.visual.pos_embed.weight` | BF16 | [2304, 1152] |

The fused source `attn.qkv` is split into q/k/v by rows (0:1152, 1152:2304, 2304:3456, biases likewise), and the
original BF16 qkv is also written (exllamav3's `get_additional_compiled_tensors` copies every `model.visual*` source
tensor whose name isn't produced otherwise). exllamav3 ignores the stray copy; the B70 server drops it
(`vision_xpu.py`). It costs 215 MB; keeping it reproduces the reference exactly.

With `vision_bits = 16` (Lygodactylus 4bpw) the tower is the BF16 source under the source names (fused `qkv`, fc1
4304 unpadded). Untested on the B70 server, whose vision path builds the ViT with the EXL3 method: quantize the tower.

## 6. BF16 source name -> EXL3 name

`L.` = `model.language_model.layers.L.`; "EXL3" = `.weight` replaced by the `{suh, svh, mul1, trellis}` quartet of
the transposed (in, out) matrix; "fp16" = same name, converted BF16 -> F16; "same" = copied unchanged (BF16).

| BF16 source | EXL3 |
|---|---|
| `model.language_model.embed_tokens.weight` [248320, 2560] | same |
| `lm_head.weight` [248320, 2560] | `lm_head` EXL3 (head_bits) |
| `model.language_model.hyper_connection_mixer.*.weight` | fp16 |
| `L.{attn,mlp}_hyper_connection.{input_mix_weight_down, input_mix_weight_up, block_inject_weight, hc_norm}.weight` | fp16 |
| `L.linear_attn.in_proj_qkv.weight` [10240, 2560], `in_proj_z.weight` [6144, 2560], `out_proj.weight` [2560, 6144] | EXL3 |
| `L.linear_attn.in_proj_a.weight`, `in_proj_b.weight` [48, 2560] | fp16 |
| `L.linear_attn.conv1d.weight`, `A_log`, `dt_bias`, `norm.weight` | same |
| `L.self_attn.{q,k,v,o}_proj.weight`, `L.self_attn.indexer.index_qk_proj.weight` | EXL3 |
| `L.self_attn.{q_norm,k_norm}.weight`, `L.self_attn.indexer.{q,k}_layernorm.weight` | same |
| `L.mlp.gate.weight` [512, 2560], `L.mlp.shared_expert_gate.weight` [1, 2560] | fp16 |
| `L.mlp.shared_expert.{gate,up,down}_proj.weight` | EXL3 |
| `L.mlp.experts.gate_up_proj` [512, 1280, 2560] (no `.weight` suffix) | `L.mlp.experts.<e>.gate_proj` EXL3 from `gate_up_proj[e, 0:640, :]` and `L.mlp.experts.<e>.up_proj` EXL3 from `gate_up_proj[e, 640:1280, :]` (each [640, 2560] = HF [out, in], not interleaved; HF does `linear(x, gate_up_proj[e]).chunk(2)`) |
| `L.mlp.experts.down_proj` [512, 2560, 640] | `L.mlp.experts.<e>.down_proj` EXL3 from `down_proj[e]` ([2560, 640] = [out, in]) |
| `layers.1.ple.key_proj.weight`, `value_proj.weight`, `conv1d.weight` | fp16 |
| `layers.1.ple.norm_{key,query,conv}.weight` | same |
| `layers.1.ple.ple_embedding.ngram_embedding.shard_<s>.weight` [2500012, 160] x 128 | `P.shard_<s>.trellis` [2500012, 1+10K] in `ngram_embedding.safetensors` |
| `layers.1.ple.ple_embedding.ngram_heads_offsets` [16] | `P.head_offsets` (renamed, moved into the n-gram file) |
| `layers.1.ple.ple_embedding.ngram_heads_vocab_sizes` [16] | `P.head_vocab_sizes` |
| `layers.1.ple.ple_embedding.layer_multipliers` [3] | `P.layer_multipliers` (note the added `.ngram_embedding`) |
| (none) | `P.head_bias` F16 [16, 160], computed |
| `mtp.fc_embedding.weight`, `mtp.fc_hidden.weight` | EXL3 (mtp_bits) |
| `mtp.pre_fc_norm_embedding.weight` | same |
| `mtp.pre_fc_norm_hidden.weight` | fp16 |
| `mtp.hyper_connection_mixer.*`, `mtp.layers.0.*` | as the main-model rows above (experts split the same way) |
| `model.visual.blocks.<b>.attn.qkv.weight` [3456, 1152] / `.bias` | `attn.{q,k,v}_proj` EXL3 (row thirds) + `.bias` fp16 thirds, **plus** the BF16 originals unchanged |
| `model.visual.blocks.<b>.attn.proj.weight` / `.bias` | EXL3 + bias fp16 |
| `model.visual.blocks.<b>.mlp.linear_fc1.weight` [4304, 1152] / `.bias` [4304] | EXL3 1152 x 4352 (zero-padded) + bias fp16 [4352] (zero-padded) |
| `model.visual.blocks.<b>.mlp.linear_fc2.weight` [1152, 4304] / `.bias` | EXL3 4352 x 1152 (input zero-padded) + bias fp16 |
| `model.visual.blocks.<b>.norm{1,2}.*`, `model.visual.merger.norm.*`, `model.visual.patch_embed.proj.*` | fp16 |
| `model.visual.merger.linear_fc{1,2}.weight` / `.bias` | EXL3 + bias fp16 |
| `model.visual.pos_embed.weight` | same |
| `base_model.model.*.lora_A/B.weight` (only in `lora_checkpoints/*.safetensors`) | not part of the model, ignore |

Tensor counts: BF16 index 1,658 (text 1,166 + n-gram shards 128 + vision 333 + mtp 31; total_size 360.0 GB);
EXL3 304,240 (main shards 304,108, n-gram file 132).
Dtype rule of thumb from exllamav3: unquantized `Linear` modules, hyper-connection weights, convs inside PLE, vision
norms and MTP's 10240-wide norm become F16; RMSNorm weights of the text model, GDN conv1d/A_log/dt_bias, the token
embedding and `pos_embed` stay BF16. Both loaders cast on load, but match the list exactly.

## 7. Version-dependent details we must choose

| detail | 1.4.4 (turboderp plain) | 1.4.6 (reference) | 1.5.0 | 1.6.3 |
|---|---|---|---|---|
| fractional K (1.5 / 2.5 / 3.5) | no | no | no | **yes** (below 4 bpw) |
| `mtp.hyper_connection_mixer` | missing (patch file) | in main shards | yes | yes |
| n-gram layout | sharded (`3.05`, `2.05`) or single `.trellis` (`4.05`) | sharded | sharded | sharded |
| quantized vision | main shards, or a side file `vision_k6.safetensors` not in the index | main shards | main shards | main shards |
| embedding | BF16 | BF16 | BF16 | BF16, or EXL3 table (`--embed_bits`) |
| `out_scales` | always | always | always | auto |

Match the reference column: integer K, everything except the n-gram file in the main shards and the index.

## 8. What the B70 server accepts (from its sources)

Read from `../state/trellis-serve/xpu` (72fd2d4 + `trellis-serve.patch`, as `b70 setup sglang` builds it):

- **Tensor discovery**: `scan_checkpoint` reads only the files named in `model.safetensors.index.json` (all
  `*.safetensors` only when there is no index). Every model tensor except the n-gram table must be in the index; a
  side file like turboderp's `vision_k6.safetensors` would be invisible.
- **K is read as `trellis.shape[2] // 16`**: integer division, so a fractional-K tensor (last dim 24/40/56) would be
  taken as K = 1/2/3 and decoded wrong. **No fractional K anywhere for the B70.**
- **Routed experts**: mul1 only, the same K for gate/up/down of an expert, **K = 3 or 4 only** (`exl3_moe.sycl`:
  `TORCH_CHECK(K == 3 || K == 4)`), one geometry for every MoE layer. K = 3 is the main store (LRU slots, hot list,
  RAM list, NVMe cold store). Any other K gets its own store whose layers are **all-static in VRAM**: one K = 4
  layer = 512 x 2.48 MB = 1.27 GB of device memory (the reference's layer 0). More K = 4 layers cost 1.27 GB of VRAM
  each, or need patch work. The MTP layer's experts go into the store of their K.
- **Dense linears**: native ESIMD/oneDNN kernels for mul1 K = 3, 4, 5, 6 (K = 2 and 3INST/MCG only with
  `EXL3_ALL_CODEBOOKS`); other K fall back to the slow Triton prototype path.
- **N-gram file**: name `ngram_embedding.safetensors`, `P.shard_N.trellis` tensors contiguous in the file, row width
  `1 + 160K/16` matching `__metadata__.K`, the four aux tensors under `P.`. The gather kernel stages a row in a
  64-word local buffer, so **K <= 6** (K = 8 rows are 81 words). The hashing needs `ngram_size` 3.
- **Embedding**: `embed_tokens.weight` only (no `.signs`/`.trellis` embedding support).
- **Vision**: expects the EXL3 q/k/v/proj/fc1/fc2/merger tensors, fc1 width taken from the checkpoint (4352); drops
  the stray BF16 qkv.
- `EXL3_MOE_HOT`/`EXL3_MOE_RAM` lists in `../flashnext/hot/` are (layer, expert id) routing statistics of the
  Uncensored fine-tune; for another fine-tune they still load but rank experts by the wrong statistics.

So a B70 target is: routed experts K in {3, 4} (4 in at most a layer or two), dense linears K 3-6, head K 3-6, MTP
K 3-4 for its experts, integer K everywhere, n-gram K 3-6, BF16 embedding.

## 9. Size budget

### 9.1 Reference by category (tensor bytes; files add ~38 MB of headers)

| category | GB | share | tensors |
|---|---|---|---|
| routed experts (47 layers K3 + layer 0 K4) | 46.085 | 63.9% | 294,912 |
| n-gram table (K3) | 19.840 | 27.5% | 132 |
| hyper-connections (97 sites, F16) | 1.281 | 1.8% | 387 |
| `embed_tokens` (BF16) | 1.271 | 1.8% | 1 |
| GDN linears (K4/K5) | 1.055 | 1.5% | 432 |
| MTP experts (K3) | 0.954 | 1.3% | 6,144 |
| vision (K5; incl. 0.215 stray BF16 qkv) | 0.505 | 0.7% | 987 |
| `lm_head` (K5) | 0.398 | 0.6% | 4 |
| full attention q/k/v/o + norms (K5) | 0.374 | 0.5% | 216 |
| routers + shared-expert gates (F16) | 0.126 | 0.2% | 96 |
| shared experts (K4/K5) | 0.120 | 0.2% | 576 |
| MTP rest (attention, fc, mixer, norms) | 0.068 | 0.1% | 59 |
| PLE dense (key/value/conv/norms) | 0.066 | 0.1% | 6 |
| QSA indexers (K5) | 0.012 | 0.0% | 72 |
| GDN small (a, b, conv1d, A_log, dt_bias, norm) | 0.021 | 0.0% | 216 |
| **total** | **72.177** | | 304,240 |

### 9.2 Formulas for planning

| item | bytes |
|---|---|
| one routed expert (3 matrices 2560 x 640) | 614,400 K + 19,212 (K3 1.862 MB, K4 2.477 MB, K2 1.248 MB) |
| one MoE layer (512 experts) | K3 0.954 GB, K4 1.268 GB, K2 0.639 GB |
| all 48 MoE layers | K3 45.77 GB, K4 60.87 GB, K2 30.67 GB (+1 layer at K4: +0.315 GB) |
| n-gram table | 640,003,072 x (1 + 10K): K3 19.84, K4 26.24, K5 32.64, K6 39.04, K8 51.84 GB |
| GDN linears, one layer (57.67 M weights) | 7.21 MB x K |
| full-attention layer incl. indexer (51.45 M weights) | 6.43 MB x K |
| shared expert, one layer (4.92 M weights) | 0.61 MB x K |
| `lm_head` (635.7 M weights) | 79.5 MB x K (fp16: 1.27 GB) |
| MTP layer | experts as one MoE layer at mtp_bits + ~60-110 MB |
| fixed: embed 1.271, hyper-connections 1.281, routers 0.126, PLE dense 0.066 | 2.74 GB |

The n-gram table is the only big lever besides the experts: at the reference's K3 it is 19.8 GB, at turboderp's K5
32.6 GB, for 51.2 B of the model's ~176 B parameters. Disk for a conversion: output ~72 GB at the reference's
settings, plus the BF16 source (360 GB, 131 shards; its 128 n-gram shards are 102 GB of that).
