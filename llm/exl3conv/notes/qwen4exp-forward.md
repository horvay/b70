# Qwen3.8-Flash-Next (`qwen4_exp`) text forward, and how exllamav3 converts it

Spec for the native C++/SYCL EXL3 converter's second architecture. Target model:
`Qwen4ExpForConditionalGeneration`, `model_type: qwen4_exp` (Dongfangshuo-Qwen3.8-Flash-Next, BF16, 131 shards,
360 GB, about 180B parameters of which 51.2B are the hashed n-gram table). Read `exl3-algorithm.md` first for the
per-tensor quantizer and the file format, and `gemma4-forward.md` for the conventions this file reuses (module loop,
capture/advance, fp16 activations).

Sources and the short names used below:

| Name | Path |
|---|---|
| `HF` | `~/models/llm/Dongfangshuo-Qwen3.8-Flash-Next/modeling_qwen4_exp.py` (remote code shipped with the model, generated from `modular_qwen4_exp.py`, transformers 5.8.0.dev0) |
| `CFG` | `.../Dongfangshuo-Qwen3.8-Flash-Next/config.json` |
| `X/...` | `~/b70/llm/dsv41/exllamav3/exllamav3/...` (fork HEAD d968d96d). Stock 1.5.3 = `git show d3739fd3:<path>`; differences are flagged as **[fork]** |
| `S/...` | `~/b70/llm/trellis/image/rootfs/opt/venv/lib/python3.12/site-packages/sglang/srt/...` (SGLang 0.5.20, cross-check only) |

Tensor shapes and dtypes come from the safetensors headers of the shards (local partial downloads plus HTTP range
requests for shards 1, 37, 41-131), and agree with `model.safetensors.index.json` (1658 tensors, 359,999,963,128 bytes).
Facts marked "checked" were verified on checkpoint bytes with stdlib Python.

---

## 0. Config, as the forward uses it (`CFG` `text_config`)

| Key | Value | Note |
|---|---|---|
| `hidden_size` | 2560 | = 20 x 128 |
| `num_hidden_layers` | 48 | |
| `layer_types` | `linear_attention` x3, `full_attention` x1, repeated 12 times | full layers: 3, 7, 11, ..., 47 (`L % 4 == 3`); 36 GDN layers, 12 attention layers |
| `num_attention_heads` / `num_key_value_heads` / `head_dim` | 24 / 2 / 256 | full layers; GQA group 12 |
| `partial_rotary_factor`, `rope_parameters` | 0.25; `rope_type default`, `rope_theta 1e7`, `mrope_section [11,11,10]`, `mrope_interleaved true` | rotary dim 64 |
| `attention_bias` | false | no biases in the text model |
| `linear_num_key_heads` / `linear_num_value_heads` | 16 / 48 | GDN; each k-head serves 3 v-heads |
| `linear_key_head_dim` / `linear_value_head_dim` | 128 / 128 | key_dim 2048, value_dim 6144, conv_dim 10240 |
| `linear_conv_kernel_dim` | 4 | |
| `output_gate_type` | `sigmoid` | GDN output norm gate is **sigmoid**, not SiLU (`HF:490-492`) |
| `mamba_ssm_dtype` | float32 | GDN state |
| `num_experts` / `num_experts_per_tok` | 512 / 10 | |
| `moe_intermediate_size` / `shared_expert_intermediate_size` | 640 / 640 | |
| `norm_topk_prob` | absent; default **true** | exllamav3 asserts true (`X/architecture/qwen3_5.py:209`); SGLang defaults true (`S/configs/qwen3_next.py:214`) |
| `hidden_act` | silu | experts and shared expert; GDN conv activation |
| `hc_count` / `hc_lowrank` | 4 / 320 | 4 residual streams, rank-320 mixing gate |
| `indexer_n_heads` / `indexer_kv_heads` / `indexer_head_dim` | 4 / 1 / 128 | QSA indexer |
| `indexer_budget` / `indexer_compress_ratio` | 2048 / 4 | block_topk = 512 |
| `ple_layer_ids` | `[2]` | **1-based**: the PLE layer sits in front of 0-based layer **1** (`HF:1259`, `X/architecture/qwen4_exp.py:250`) |
| `ple_embed_dim` / `ple_conv_kernel_size` | 2560 / 4 | conv dilation = `ngram_size` = 3 |
| `ngram_size` / `heads_per_ngram` | 3 / 8 | 16 hash heads (8 bigram + 8 trigram), 160 dims each |
| `ngram_vocab_size_base` / `make_ngram_vocab_size_divisible_by` / `split_ngram_parts` | 20,000,000 / 128 / 128 | table 320,001,536 rows in 128 shards |
| `seed` (hash) | absent; **1234** | SGLang default `getattr(config, "seed", 1234)` (`S/models/qwen4_exp.py:533`); checked: seed 1234 reproduces the checkpoint's `layer_multipliers` exactly. Read the buffers from the checkpoint anyway |
| `rms_norm_eps` | 1e-6 | every norm (and l2norm uses 1e-6) |
| `vocab_size` | 248320 | 248077 ids used, rows 248077..248319 are padding |
| `tie_word_embeddings` | false | separate `lm_head.weight` |
| `bos_token_id` / `eos_token_id` | 248044 / 248044 (`<|endoftext|>`) | the n-gram hashing uses 248044 as its segment token |
| `mtp_num_hidden_layers` | 1 | MTP not used by the trunk forward |
| `max_position_embeddings` | 262144 | |

---

## 1. Tensors

All tensors are BF16 except the three int64 hashing buffers. Shapes are `[out, in]` (nn.Linear). `L` = layer index.
"Q" = quantized to EXL3 (tensors `P.trellis I16 [in/16, out/16, 16K]`, `P.suh F16 [in]`, `P.svh F16 [out]`,
`P.mul1 I32 []`, exl3-algorithm.md section 5.1). "fp16" = stored by exllamav3 as an F16 copy (unquantized Linear:
`P.weight` F16 `[out, in]`, `X/modules/quant/fp16.py:62-67`). "copy" = stored as read (BF16).

### 1.1 Embedding, final mixer, head

| Tensor | Shape | exllamav3 |
|---|---|---|
| `model.language_model.embed_tokens.weight` | `[248320, 2560]` | copy BF16 (`embed_bits` default 16) |
| `model.language_model.hyper_connection_mixer.hc_norm.weight` | `[10240]` | copy BF16 |
| `model.language_model.hyper_connection_mixer.input_mix_weight_down.weight` | `[320, 10240]` | fp16 |
| `model.language_model.hyper_connection_mixer.input_mix_weight_up.weight` | `[10240, 320]` | fp16 |
| `lm_head.weight` | `[248320, 2560]` | Q at `head_bits` (default **6**, `X/conversion/convert_model.py:278`), Hessian = mixer output; or `-hb 16`: `lm_head.weight` F16 |

There is **no final RMSNorm** (`model.language_model.norm` does not exist). The combine-less mixer replaces it
(`HF:1384, 1484`; `X/architecture/qwen4_exp.py:281-291`).

### 1.2 Per layer, both types: hyper-connection sites and MoE

| Tensor (`model.language_model.layers.L.` prefix) | Shape | exllamav3 |
|---|---|---|
| `attn_hyper_connection.hc_norm.weight` | `[10240]` | copy BF16 |
| `attn_hyper_connection.input_mix_weight_down.weight` | `[320, 10240]` | fp16 |
| `attn_hyper_connection.input_mix_weight_up.weight` | `[10240, 320]` | fp16 |
| `attn_hyper_connection.block_inject_weight.weight` | `[4, 10240]` | fp16 |
| `mlp_hyper_connection.*` | same four | same |
| `mlp.gate.weight` (router) | `[512, 2560]` | fp16 (qmap None, `X/modules/block_sparse_mlp.py:306-318`) |
| `mlp.experts.gate_up_proj` | `[512, 1280, 2560]` **fused, 3-D** | split per expert: rows 0..639 = gate, 640..1279 = up (`frange`, `X/modules/block_sparse_mlp.py:410-445`, `transpose_fused_weights = False`). Q as `mlp.experts.E.gate_proj.*` and `mlp.experts.E.up_proj.*` (in 2560, out 640), E = 0..511 |
| `mlp.experts.down_proj` | `[512, 2560, 640]` fused | Q per expert as `mlp.experts.E.down_proj.*` (in 640, out 2560) |
| `mlp.shared_expert.gate_proj.weight`, `up_proj.weight` | `[640, 2560]` | Q |
| `mlp.shared_expert.down_proj.weight` | `[2560, 640]` | Q |
| `mlp.shared_expert_gate.weight` | `[1, 2560]` | fp16 (qmap None) |

The HF 3-D expert tensors are used as `F.linear(x, gate_up_proj[e])` then `.chunk(2)` (`HF:942`): gate is the first
640 output rows. SGLang loads the same split (`w13`, rows 0-639 gate). 1,536 routed-expert EXL3 linears per layer.

### 1.3 Linear-attention (Gated DeltaNet) layers (36 of them)

| Tensor (`...layers.L.linear_attn.`) | Shape | exllamav3 |
|---|---|---|
| `in_proj_qkv.weight` | `[10240, 2560]` | Q. Output rows: `[q: 16 heads x 128 | k: 16 x 128 | v: 48 x 128]`, plain contiguous, no per-head interleave (unlike Qwen3-Next's `in_proj_qkvz`) |
| `in_proj_z.weight` | `[6144, 2560]` | Q |
| `in_proj_b.weight`, `in_proj_a.weight` | `[48, 2560]` | fp16 (qmap None, `pad_to = 1`, `X/modules/gated_delta_net.py:618-623`) |
| `conv1d.weight` | `[10240, 1, 4]` | copy BF16 |
| `A_log`, `dt_bias` | `[48]` | copy BF16 |
| `norm.weight` | `[128]` | copy BF16 (used as `w`, values near 1; checked: layer 1 min 0.945, max 1.109) |
| `out_proj.weight` | `[2560, 6144]` | Q |

### 1.4 Full-attention layers (12 of them)

| Tensor (`...layers.L.self_attn.`) | Shape | exllamav3 |
|---|---|---|
| `q_proj.weight` | `[12288, 2560]` = 24 heads x (256 q + 256 gate), **per-head interleaved** | Q as one 12288-wide tensor |
| `k_proj.weight`, `v_proj.weight` | `[512, 2560]` | Q |
| `o_proj.weight` | `[2560, 6144]` | Q |
| `q_norm.weight`, `k_norm.weight` | `[256]` | copy BF16 (used as `1 + w`) |
| `indexer.index_qk_proj.weight` | `[640, 2560]` = 4 q heads x 128, then 1 k head x 128 | Q |
| `indexer.q_layernorm.weight`, `indexer.k_layernorm.weight` | `[128]` | copy BF16 (`1 + w`) |

### 1.5 PLE layer (in front of layer 1; key prefix `model.language_model.layers.1.ple.`)

| Tensor | Shape | exllamav3 |
|---|---|---|
| `ple_embedding.ngram_embedding.shard_S.weight`, S = 0..127 | `[2500012, 160]` each | quantized **before** the layer loop into `ngram_embedding.safetensors` (section 3.7), never into the shards |
| `ple_embedding.layer_multipliers` | I64 `[3]` | copied into `ngram_embedding.safetensors` as `...ngram_embedding.layer_multipliers` |
| `ple_embedding.ngram_heads_offsets`, `ngram_heads_vocab_sizes` | I64 `[16]` | copied as `...ngram_embedding.head_offsets`, `...head_vocab_sizes` |
| `key_proj.weight` | `[10240, 2560]` | **fp16, not quantized** (PLELayer passes `qmap = None`, `X/modules/ple.py:197-212`) |
| `value_proj.weight` | `[2560, 2560]` | fp16, not quantized |
| `norm_key.weight`, `norm_query.weight`, `norm_conv.weight` | `[10240]` | copy BF16 (grouped norms, `1 + w`) |
| `conv1d.weight` | `[10240, 1, 4]` | F16 (`float2half = True`, `X/modules/ple.py:238-252`) |

Checked: `layer_multipliers = [23703573157769, 20109073645365, 8052911324071]`. The 16 head vocab sizes are the 16
primes after 19,999,999: 20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077, 20000081,
20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171. Offsets are their running sums: 0, 20000003,
40000026, 60000059, 80000106, 100000165, 120000228, 140000297, 160000374, 180000455, 200000548, 220000655, 240000802,
260000955, 280001114, 300001275. Sum 320,001,446, padded to 320,001,536 = 128 x 2,500,012. Global row `r` is
`shard_{r // 2500012}[r % 2500012]` (`X/conversion/ngram.py:286-295`).

### 1.6 MTP head (`mtp.*`, ignored by HF: `_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`, `HF:1313`)

`mtp.fc_embedding.weight [2560, 2560]`, `mtp.fc_hidden.weight [2560, 2560]`, `mtp.pre_fc_norm_embedding.weight [2560]`,
`mtp.pre_fc_norm_hidden.weight [10240]`, `mtp.hyper_connection_mixer.{hc_norm, input_mix_weight_down, _up}`, and one
full-attention block `mtp.layers.0.*` with exactly the tensors of 1.2 and 1.4 (experts `[512, 1280, 2560]` and
`[512, 2560, 640]`, indexer included). No PLE. exllamav3: all Linears Q at `mtp_bits` (default 4), **uncalibrated**
(section 3.8); norms copy; HC weights fp16.

### 1.7 Vision tower (`model.visual.*`, 27 blocks, not used by the text forward)

`blocks.N.attn.qkv.{weight [3456, 1152], bias}`, `attn.proj.{weight [1152, 1152], bias}`,
`mlp.linear_fc1.{weight [4304, 1152], bias}`, `mlp.linear_fc2.{weight [1152, 4304], bias}`, `norm1/norm2.{weight, bias}`
(LayerNorm), `merger.linear_fc1 [4608, 4608]`, `merger.linear_fc2 [2560, 4608]`, `merger.norm`, `patch_embed.proj
[1152, 3, 2, 16, 16]` (+bias), `pos_embed.weight [2304, 1152]`. exllamav3 builds it as the Qwen3-VL tower
(`X/architecture/qwen4_exp.py:46-55`, `X/architecture/qwen3_vl.py`), whose `default_vision_bits` is **6**
(`qwen3_vl.py:172`): with `-vb 0` (default) the qkv/proj/fc1/fc2/merger Linears are Q at 6 bpw uncalibrated
(4304 pads to 4352), the rest copied. With `-vb 16` every `model.visual.*` tensor is copied BF16
(`get_additional_compiled_tensors`, `qwen3_vl.py:158-160`). Recommendation as for Gemma: copy BF16.

---

## 2. Text forward

Notation: `T` tokens per row (2048 in calibration), `D = 2560`, `S = 4` streams. The residual is a stream stack
`X [T, S, D]`; HF stores it flattened `[T, S*D]`, stream-major (stream `s` = columns `s*D .. s*D+D-1`). All matmuls
`y = x @ W^T` with `W [out, in]`.

### 2.1 Norms

```
rms(x)               = x / sqrt(mean(x^2) + 1e-6)                    # fp32, eps inside the root
Norm1p(x, w)         = rms(x) * (1 + w)                              # Qwen4ExpTextRMSNorm, HF:143-166
GroupNorm1p(X, w)    = for each stream s: rms(X[s]) * (1 + w[s*D : s*D+D])   # same class with group_size = D
GatedNorm(o, z, w)   = w * rms(o) * sigmoid(z)                        # HF:171-187, w used as-is, norm before gate
l2norm(x)            = x / sqrt(sum(x^2) + 1e-6)                      # HF:247-250
```

Which norm is which:

| Norm | Kind | Dim |
|---|---|---|
| `hc_norm` (both sites, final mixer) | GroupNorm1p | 4 x 2560 |
| `self_attn.q_norm`, `k_norm` | Norm1p per head | 256 |
| `indexer.q_layernorm`, `k_layernorm` | Norm1p per head | 128 |
| `ple.norm_key`, `norm_query`, `norm_conv` | GroupNorm1p | 4 x 2560 |
| `linear_attn.norm` | GatedNorm per v-head, **w not 1+w**, sigmoid gate | 128 |
| `mtp.pre_fc_norm_embedding` | Norm1p | 2560 |
| `mtp.pre_fc_norm_hidden` | Norm1p; grouping disputed, see 3.8 | 10240 |

All `1 + w` weights are zero-initialized in training (`HF:1333-1335`) and stored as the offset; checked: `hc_norm` values
range -2.30..3.63, so `1 + w` can be negative. Do not clamp.

### 2.2 Embedding and stream expansion

```
e = E[ids]                                      # [T, D], no scaling (unlike Gemma)
X[t, s] = e[t]  for s = 0..3                    # HF:1471 hidden_states.repeat(1, 1, 4); X: ExpandStreams, fp32
```

### 2.3 Gated-residual site (`GatedResidual`, `HF:994-1022`)

Every sublayer (attention or GDN, then MoE) sits between a mix and an inject:

```
def mix(X):                                     # X [T, 4, D]
    N    = GroupNorm1p(X, hc_norm.w)            # [T, 4, D]
    flat = N.reshape(T, 4*D)                    # stream-major
    a    = silu((flat @ Wdown^T) / 4)           # Wdown [320, 10240]
    G    = sigmoid(a @ Wup^T).reshape(T, 4, D)  # Wup [10240, 320]
    y    = mean over s of (G[:, s] * N[:, s])   # MEAN (divide by 4), [T, D]: the sublayer input
    inj  = 2 * sigmoid((flat @ Winj^T) / 4)     # Winj [4, 10240] -> [T, 4]; site form only
    return y, inj

def apply(X, out, inj):                         # out = sublayer output [T, D]
    X[:, s] = X[:, s] + inj[:, s] * out          # on the RAW (un-normed) streams, for every s
```

The final mixer is the same `mix` without `Winj`; it returns `y` only, and `y` goes straight to `lm_head`.
exllamav3: `X/modules/hyperconnections.py:498-511` (`_mix_ref`, the fp32 reference), `:569-636` (the kernel paths),
`:649-687` (`apply_`). SGLang agrees (`S/layers/hyperconnection.py:58-73, 191-199, 210-217`).

### 2.4 PLE layer (layer 1 only, `HF:1071-1246`; `X/modules/ple.py`, `X/modules/ngram_embedding.py`)

Runs on the stream stack entering layer 1, before layer 1's attention site:
`X = X + PLE(X, ids)` (`HF:1274-1277`).

**N-gram ids.** Per calibration row, `hist = [EOS, EOS] + ids` (EOS = 248044; a new sequence starts with two EOS of
context, `HF:1129`, `X/modules/ple.py:447-450`). For a position `p` of `hist` (p >= 2) and shift `s`:

```
tok(p, 0) = hist[p]
tok(p, s) = EOS          if p - s < 0 or any(hist[q] == EOS for q in p-s .. p-1)
          = hist[p - s]  otherwise                      # s = 1, 2
```

This is `_shift_right_ignore_eos` (`HF:1106-1120`, `X/modules/ngram_embedding.py:241-257`) restated: an n-gram never
reaches back across an EOS. The EOS token itself still sees its own preceding context; the token after it starts fresh.
Then, with int64 products (no overflow: every multiplier is below 2^63 / 248320, checked) and XOR:

```
mix2 = tok(p,0)*m0 ^ tok(p,1)*m1
mix3 = mix2 ^ tok(p,2)*m2
row[h] = (mix2 if h < 8 else mix3) % P[h] + O[h]       # h = 0..15, P = vocab sizes, O = offsets (1.5)
emb[p] = concat over h of table[row[h]]                # 16 x 160 = 2560
```

`HF:1150-1171`. All values are non-negative, so C `%` equals `torch.remainder`. In exllamav3 the table is the
**quantized** one (decoded rows, fp16) during the whole calibration (section 3.7).

**Layer.** With `E = emb [T, 2560]`, streams `X [T, 4, D]`:

```
Kn  = GroupNorm1p((E @ Wkey^T).reshape(T, 4, D), norm_key.w)      # Wkey [10240, 2560]
V   = E @ Wval^T                                                   # [T, D], shared by all streams
Qn  = GroupNorm1p(X, norm_query.w)                                 # the RAW streams entering layer 1
g   = sum_d(Kn[:, s] * Qn[:, s]) / sqrt(2560)                      # [T, 4]
g   = sign(g) * sqrt(max(|g|, 1e-6))                               # signed sqrt; sign(0) = 0
Gv[:, s] = sigmoid(g[:, s]) * V                                    # [T, 4, D]
Cn  = GroupNorm1p(Gv, norm_conv.w)                                 # [T, 10240] channels
conv[t, c] = sum_{j=0..3} wc[c, j] * Cn[t - 3*(3 - j), c]          # depthwise, kernel 4, dilation 3, causal,
                                                                   # zero for t - 3*(3-j) < 0 (9-step zero state)
delta = Gv + silu(conv)                                            # [T, 4, D]
X = X + delta
```

`HF:1207-1246`; exllamav3 `X/modules/ple.py:418-434` (reference) and `X/exllamav3_ext/ple.cu:110-175` (fused). The
gate kernel uses exactly `sqrtf(fmaxf(fabsf(g), 1e-6f))` and `sign(0) = 0` (`ple.cu:16-46`).

### 2.5 Gated DeltaNet layer (`HF:456-616`)

```
y, inj = mix_attn(X)                                   # [T, D]
qkv = y @ Wqkv^T                                       # [T, 10240]
z   = (y @ Wz^T).reshape(T, 48, 128)
b   = y @ Wb^T;  a = y @ Wa^T                          # [T, 48]
c   = silu(conv1d(qkv))                                # depthwise, kernel 4, causal (below)
q, k, v = c[:, 0:2048], c[:, 2048:4096], c[:, 4096:10240]
q = q.reshape(T, 16, 128); k = k.reshape(T, 16, 128); v = v.reshape(T, 48, 128)
beta = sigmoid(b)                                      # [T, 48]
g    = -exp(A_log) * softplus(a + dt_bias)             # [T, 48], <= 0; softplus(x) = x if x > 20 else log1p(exp(x))
q, k = repeat_interleave(q, 3, head dim), repeat_interleave(k, 3, head dim)   # v-head i uses k-head i // 3
q = l2norm(q) / sqrt(128);  k = l2norm(k)              # per head, fp32
o = delta_rule(q, k, v, g, beta)                       # [T, 48, 128], zero initial state
o = GatedNorm(o, z, norm.w)                            # per v-head
out = o.reshape(T, 6144) @ Wout^T                      # [T, D]
apply(X, out, inj)
```

Causal conv (`HF:222-242`): `c[t, ch] = silu(sum_{j=0..3} w[ch, j] * qkv[t - 3 + j, ch])`, zero for negative
positions. Channel order is the qkv projection's output order.

`repeat_interleave`, not tile: HF `HF:573-575`; exllamav3 passes 16 q/k heads and 48 v heads to fla, which maps v-head
`i` to k-head `i // (HV / H)` (`X/vendor/fla/chunk_o.py:79-80`, `gdn_wy_fast.py:88`).

**Delta rule, sequential form** (per v-head, state `St [128 (k), 128 (v)]` fp32, `St = 0` at t = 0;
`HF:391-450`):

```
for t in 0..T-1:
    St  = exp(g[t]) * St
    kv  = St^T k[t]                       # [128]: kv[j] = sum_i St[i, j] * k[t, i]
    dlt = beta[t] * (v[t] - kv)
    St  = St + outer(k[t], dlt)
    o[t] = St^T q[t]
```

**Chunked form** (the one to implement on the card; chunk C = 64, `T = 2048` is 32 chunks, no padding;
`HF:254-387`, fla `X/vendor/fla/__init__.py:28-67`). Within a chunk, positions `i, j = 0..63`,
`G[i] = sum_{u <= i} g[u]` (cumulative inside the chunk):

```
A[i, j]  = beta[i] * (k[i] . k[j]) * exp(G[i] - G[j])   for j < i, else 0     # strictly lower
Tinv     = (I + A)^-1                                    # unit lower-triangular solve
U        = Tinv @ (beta * V)                             # [64, 128]
W        = Tinv @ (beta * exp(G) * K)                    # [64, 128]
Vn       = U - W @ S                                     # S = state entering the chunk
P[i, j]  = (q[i] . k[j]) * exp(G[i] - G[j])   for j <= i, else 0           # includes the diagonal
O        = (exp(G) * Q) @ S + P @ Vn
S        = exp(G[63]) * S + (exp(G[63] - G) * K)^T @ Vn
```

HF builds the system matrix with the diagonal included but solves it `unitriangular=True`, so the diagonal is ignored:
the matrix really is `I + strictly_lower(A)`. Mask before `exp` (`-inf` above the diagonal) so `exp(G[i]-G[j])` never
overflows for `j > i`.

### 2.6 Full-attention layer (`HF:810-892`)

```
y, inj = mix_attn(X)
qg = (y @ Wq^T).reshape(T, 24, 512)
q, gate = qg[..., :256], qg[..., 256:]           # per head: first 256 = query, next 256 = gate
gate = gate.reshape(T, 6144)
q = Norm1p(q, q_norm.w)                          # per head
k = Norm1p((y @ Wk^T).reshape(T, 2, 256), k_norm.w)
v = (y @ Wv^T).reshape(T, 2, 256)                # no v norm
q, k = rope(q), rope(k)
o = softmax_causal(q k^T / 16) v                 # scale 256^-0.5; q head h uses kv head h // 12
o = o.reshape(T, 6144) * sigmoid(gate)
out = o @ Wo^T
apply(X, out, inj)
```

**RoPE.** `inv_freq[j] = 1e7 ^ (-2j / 64)` for `j = 0..31` (fp32; the exponent's denominator is the rotary dim 64,
`HF:107-115`). For a head vector `x` (256 dims) at position `pos`, NeoX rotate-half **within the first 64 dims**:

```
for j in 0..31:
    a, b = x[j], x[j + 32];  th = pos * inv_freq[j]
    x[j]      = a*cos(th) - b*sin(th)
    x[j + 32] = b*cos(th) + a*sin(th)
# dims 64..255 untouched
```

(`HF:626-661`, exllamav3 `X/util/rope.py:208-213` plus `rope.cu` `partial_head_dim = 2 * len(inv_freq)`). M-RoPE:
`position_ids` has three rows (T, H, W) and `recomposition_frequencies` interleaves them by `mrope_section [11, 11, 10]`
(`HF:119-140`); for text all three rows equal the token index, so it reduces to the plain RoPE above. Positions are
`0..T-1` per row. `attention_scaling` = 1.

**QSA indexer** (`HF:664-770`). For every query position `p`, with `visible = 0..p` (count `p + 1`):

```
nb      = (p + 1) // 4                     # complete 4-token blocks among the visible tokens
qi      = rope(Norm1p(index_q[p], q_layernorm.w))          # 4 heads x 128, from index_qk_proj columns 0..511
kblk[b] = rope_at(4b)(Norm1p(mean(raw_k[4b .. 4b+3]), k_layernorm.w))   # raw_k = columns 512..639, unnormed
score[b] = sum over the 4 heads of relu(qi[h] . kblk[b]) / sqrt(128)
keep     = top min(512, nb) blocks by score, plus the tail tokens 4*nb .. p
attention mask for query p = causal AND keep
```

`block_topk = 2048 / 4 = 512`. When `nb <= 512` every complete block is kept, and the tail too, so the mask is plain
causal. That holds for `p + 1 <= 2051`. **Calibration rows of 2048 tokens never take the sparse path**: the result is
exactly dense causal attention. exllamav3 states this as `sparse_threshold() = 4 * block_topk + 3 = 2051`
(`X/modules/qsa_indexer.py:469-471`) and runs the indexer only `if seqlen > 2051` (`X/modules/attn.py:890-896`); so in
calibration `index_qk_proj` is **never executed**, but it shares the qmap `block.attn.input` with q/k/v and gets their
Hessian (section 3.3). SGLang runs the selection always and agrees that a 2048-token prefill keeps every block. A
native implementation can skip the indexer entirely for `T <= 2051`.

### 2.7 MoE block (`HF:951-991`)

```
y, inj = mix_mlp(X)                                       # [T, D]
logits = y @ Wr^T                                         # [T, 512]
p      = softmax(logits) (fp32)
top    = the 10 largest p; wts = p[top] / sum(p[top])     # norm_topk_prob = true
routed = sum over e in top of wts[e] * (silu(y @ Wg_e^T) * (y @ Wu_e^T)) @ Wd_e^T
shared = sigmoid(y @ Wsg^T) * ((silu(y @ Wsg_gate^T) * (y @ Wsg_up^T)) @ Wsg_down^T)   # Wsg [1, 2560]
out    = routed + shared
apply(X, out, inj)
```

Renormalized top-k of the full softmax equals the softmax over the 10 selected logits (exllamav3 computes it that way,
`X/exllamav3_ext/routing.cu:607-700`, with `+1e-20` in the denominator).

### 2.8 Decoder layer and whole text forward

```
def layer(X, L):
    if L == 1: X = X + PLE(X, ids)                       # 2.4
    y, inj = mix(X, L.attn_hyper_connection)
    out = GDN(y) if L % 4 != 3 else ATTN(y)              # 2.5 / 2.6
    X = apply(X, out, inj)
    y, inj = mix(X, L.mlp_hyper_connection)
    X = apply(X, MoE(y), inj)                            # 2.7
    return X

X = expand(E[ids])                                       # [T, 4, D]
for L in 0..47: X = layer(X, L)
h = mix_final(X)                                         # hyper_connection_mixer, [T, D]
logits = h @ Wlm^T                                       # no norm, no softcap
```

`HF:1249-1301, 1391-1489, 2394-2550`. The multimodal wrapper only passes `inputs_embeds` plus `ple_input_ids = input_ids`
(`HF:2316-2322`), identical for text.

---

## 3. How exllamav3 converts this architecture

### 3.1 Module list (`X/architecture/qwen4_exp.py:222-319`)

| idx | Module | Key | Quant targets |
|---|---|---|---|
| 0 | `Embedding` | `model.language_model.embed_tokens` | none (copy) |
| 1 | `ExpandStreams` | `hc_expand` | none, no tensors |
| 2 | `TransformerBlock` (GDN) | `model.language_model.layers.0` | 3.3 |
| 3 | `PLELayer` | `model.language_model.layers.1.ple` | **none** (key/value_proj fp16) |
| 4 | `TransformerBlock` (GDN) | `...layers.1` | 3.3 |
| 5 .. 50 | `TransformerBlock` | `...layers.2` .. `...layers.47` (attention at L % 4 == 3) | 3.3 |
| 51 | `GatedResidual` (`use_combine = False`, out fp16) | `model.language_model.hyper_connection_mixer` | none (fp16 copies) |
| 52 | `Linear` | `lm_head` (`qbits_key = head_bits`, qmap `block`) | `lm_head` |

`calibration_all_experts = True` (`:306`). The loop is the one in gemma4-forward.md 3.2: load (Linear weights BF16 to
fp16), capture over all rows (only if the module has qmaps), quantize, reload from the quantized tensors, advance over all
rows. Modules without qmaps (0, 1, 3, 51) are only advanced. `params["input_ids"]` (the row's tokens) is passed to every
module (`X/conversion/convert_model.py:1734, 1915`); the PLE module hashes from it.

### 3.2 State and the n-gram table during calibration

- State per row: `[1, 2048, 2560]` fp32 after module 0, then `[1, 2048, 4, 2560]` fp32 after `ExpandStreams` until the
  mixer, then `[1, 2048, 2560]` fp16. exllamav3 keeps it in host RAM (21 GB for 250 rows).
- The n-gram table is quantized and added to the tensor collection **before** module 0
  (`prepare_ngram_table_for_conversion`, `X/conversion/convert_model.py:1556-1560`, `X/conversion/ngram.py:600-648`).
  Every calibration forward through the PLE layer reads the **quantized** rows. Our converter must therefore finish the
  table first, then decode the rows the calibration tokens touch.

### 3.3 Hessian groups (qmaps) per decoder layer

qmaps are strings, per module capture (two layers never share one). Only the first Linear to see a qmap accumulates
(`X/modules/linear.py:670-691`).

**GDN layer** (`X/modules/gated_delta_net.py:589-659`):

| qmap | Linears | Input captured | H |
|---|---|---|---|
| `block.attn.input` | in_proj_qkv (first), in_proj_z | `y` of the attn site (fp16) `[T, 2560]` | 2560^2 |
| `block.attn.output` | out_proj | gated-norm output (fp16) `[T, 6144]` | 6144^2 |

in_proj_a/b: no qmap (fp16 copies).

**Attention layer** (`X/modules/attn.py:280-356`, `X/modules/qsa_indexer.py:76-83`):

| qmap | Linears | Input captured | H |
|---|---|---|---|
| `block.attn.input` | q_proj (first), k_proj, v_proj, indexer.index_qk_proj | `y` of the attn site (fp16) | 2560^2 |
| `block.attn.o` | o_proj | `o * sigmoid(gate)` (fp16) `[T, 6144]` | 6144^2 |

The indexer never runs at `T = 2048`, so its H is q/k/v's H. Correct either way: its input is the same `y`.

**MoE, both layer types** (`X/modules/block_sparse_mlp.py:397-466`, shared expert `X/modules/mlp.py:655-705`):

| qmap | Linears | Input captured | H |
|---|---|---|---|
| `block.mlp.input` | all 512 experts' gate_proj and up_proj, shared_expert.gate_proj and up_proj (first caller: `experts.0.up_proj`) | `y` of the mlp site (fp16) `[T, 2560]` | one 2560^2 |
| `block.mlp.E.down`, E = 0..511 | experts.E.down_proj | `silu(g_E) * u_E` of expert E over **every** token (fp16) `[T, 640]` | 512 x 640^2 |
| `block.mlp.down` | shared_expert.down_proj | `silu(g_s) * u_s` (fp16) | 640^2 |

Router and `shared_expert_gate`: no qmap (fp16 copies).

**All-experts capture** (`X/modules/block_sparse_mlp_routing.py:110-121`): during the capture pass the router returns
`selected = [0..511]` for every token and `weights = softmax(fp16 logits)` over all 512. Every expert sees every token,
so every expert's down Hessian is the mean over all calibration tokens of its own intermediate, **unweighted** by
routing (the weight multiplies after down_proj, `block_sparse_mlp.py:1858`). The capture output (a sum over all 512
experts) is discarded. The 5 reference rows are rerun with real top-10 routing (`convert_model.py:1736-1748`), and the
advance pass uses real top-10 routing through the quantized layer.

**Head** (module 52): qmap `block`, input = mixer output fp16 `[T, 2560]`, H 2560^2, only if `head_bits <= 8`.

### 3.4 Order inside a layer

As for Gemma: nothing inside a layer is sequential. o_proj/out_proj Hessians come from the unquantized q/k/v (or
qkv/z) of the same layer, and the MoE Hessians from the mlp site after the **unquantized** attention sublayer and its
inject. One capture pass and one advance pass per layer. The PLE module advances once (no capture).

### 3.5 Grouping (batching only)

`group_quant_linears` (`X/conversion/convert_model.py:738-781`; stock `cm.py:495-538` with caps 32768 columns /
3.5e8 and 16 stacked) concatenates same-(qmap, K, in) linears along out and stacks same-shape linears with their own H.
Here: q+k+v+indexer (13,952 cols) one group; qkv+z (16,384) one group; the 1,026 gate/up tensors of `block.mlp.input`
in chunks of 32,768 columns (stock) or 500,000 (**[fork]**); the 513 down projections stacked 16 (stock) or up to 2048
(**[fork]**). exl3-algorithm.md 3.10: per-tensor quantization gives the same result.

### 3.6 Bit allocation (`X/conversion/allocation.py:51-206`, same in stock)

- `select_hq_bits = 2` on the GDN module and its Linears, the Attention module and its q/k/v/o, and the shared expert
  GatedMLP (`X/architecture/qwen4_exp.py:136, 169, 216`). The indexer's `index_qk_proj` and the routed experts have 0.
- `q_priority`: Attention 2 + 2 = 4 (inherited by the indexer), GDN 1 + 2 = 3, everything else 0 (GatedMLP and
  BlockSparseMLP keep the Module default 0).
- qgroups: `...self_attn.qkv` (q, k, v), `...self_attn.o`, `...self_attn.indexer.index_qk_proj` (its own key),
  `...linear_attn.qkvz` (qkv, z), `...linear_attn.o`, `...mlp.block_gud` (all 1,536 routed tensors of the layer, one
  group), `...mlp.shared_expert.gu`, `...mlp.shared_expert.d`.
- Integer `-b K`: every budgeted Linear gets K, the budget is met exactly, no bumps. `--hq`: the floor becomes
  `rate_floor(K + 2)`, so at `-b 4 --hq` attention q/k/v/o, GDN qkv/z/out and the shared expert go to 6 (the indexer
  and routed experts stay at 4); this is applied after the budget fill and raises the final bpw.
- Fractional `-b`: base `rate_floor(b)` (1.5/2.5/3.5 steps below 4 with mul1), then whole qgroups are bumped one step
  in order (priority desc, distance to the nearer end of the stack, layer, index) while the budget allows. A layer's
  `block_gud` group costs 2.5G weights per step, so fractional bitrates mostly land on attention/GDN first.
- Aux targets outside the budget: `lm_head` at `head_bits` (default 6), MTP Linears at `mtp_bits` (default 4; `--hq`
  applies their `select_hq_bits` too: MTP attention/shared expert +2, `fc_hidden`/`fc_embedding` +1), vision at
  `vision_bits`.

### 3.7 The n-gram table (`X/conversion/ngram.py`, `X/modules/quant/exl3_lib/ngram_codec.py`; format unchanged since stock 1.5.3)

**K**: `--ngram_bits`, default `max(1, min(8, round(bits)))` (`ngram.py:634`; Python rounding, so 2.5 -> 2,
3.5 -> 4). `--ngram_file` copies a pre-quantized file instead.

**Per-head bias** (`compute_head_bias`, `:317-335`): for each head h with offset `O[h]` and size `P[h]`, 16 stripes of
8192 consecutive rows starting at `O[h] + (P[h] - 8192) * j // 15`, j = 0..15; the bias is the fp64 mean of those
131,072 source rows, rounded to fp16, shape `[16, 160]`. A row's head is `searchsorted(O, r, right=True) - 1`,
clamped to 0..15 (the 90 padding rows count as head 15).

**Per row** (`quantize_rows`, `:54-112`, default `cs_search = 1`), for K with `(gamma, cs_hi)` from
`CS_HEURISTIC = {1: (4.0, 1.16), 2: (3.6, 0.98), 3: (3.2, 0.98), 4: (3.0, 0.98), 5: (3.0, 0.95), 6: (3.0, 0.92),
7: (3.0, 0.90), 8: (3.0, 0.86)}`, `CS_MIN = 0.55`:

```
w      = nan_to_num(float(row)) - float(bias[head])          # 160 values, non-finite -> 0
rms    = sqrt(mean(w^2))
cs     = clamp(gamma / max(max|w| / max(rms, 1e-12), 1e-6), CS_MIN, cs_hi)
scale0 = float(f16(max(rms / cs, 1e-8)));  if scale0 == 0: scale0 = 1.0
q, st  = trellis_ring(w / scale0, K, mul1, L = 160)           # one tail-biting ring of 160 positions
scale  = f16( sum(w * q) / max(sum(q * q), 1e-12) )          # least-squares; q = decoded codebook values
```

`trellis_ring` is the EXL3 tile encoder (exl3-algorithm.md 4.3) templated on the ring length `L = 160` instead of 256:
inputs rounded to fp16, positions in natural order (no tensor-core permutation), pass 1 starts at `roll = L / 2 = 80`
(`X/exllamav3_ext/quant/quantize_tiles_kernel.cuh:64-338`, instances `_l160` mul1 only, `quantize.cu:31-55`).

**Packed row**: `1 + 10K` int16 words (82 bytes at K = 4, 102 at K = 5). Word 0 = the fp16 bits of `scale`. Words
1..10K = a `160 * K`-bit ring stream, **LSB first within each 16-bit word**: stream bit `b` is bit `b & 15` of word
`1 + (b >> 4)`, and stream bits `[i*K, i*K + K)` hold the low K bits of state `i` (bit j of the symbol at stream bit
`i*K + j`) (`pack_rows`, `ngram_codec.py:49-63`). Decode (`unpack_rows`, `:66-81`; CUDA `ngram.cu:320-380`):

```
sym[i]   = the K bits at stream bits [i*K, i*K+K)
state[i] = (sym[i] | sym[i-1] << K | sym[i-2] << 2K | ...) & 0xFFFF      # indices mod 160
row[i]   = f16( float(f16(mul1(state[i]))) * float(scale) + float(bias[head][i]) )
```

This bit layout differs from the EXL3 linear tiles (those are MSB-first in 32-bit words, exl3-algorithm.md 8.7).

**File** `ngram_embedding.safetensors` in the output directory (`StreamingSafetensorsWriter`, `:155-236`): a regular
safetensors file, header JSON compact (`separators = (",", ":")`) and space-padded to a multiple of 8, in this order:

| Entry | dtype, shape |
|---|---|
| `__metadata__` | `{"format": "exl3_ngram_trellis", "version": "1", "K": "<K>", "codebook": "mul1", "codebook_scale": "heuristic(gamma=3.00, hi=0.98)" (informational), "row_dim": "160", "rows": "320001536", "shard_rows": "2500012", "source": "<abs in_dir>"}` |
| `P.head_bias` | F16 `[16, 160]` |
| `P.head_offsets` | I64 `[16]` |
| `P.head_vocab_sizes` | I64 `[16]` |
| `P.layer_multipliers` | I64 `[3]` |
| `P.shard_0.trellis` .. `P.shard_127.trellis` | I16 `[2500012, 1 + 10K]` each, data contiguous in shard order |

with `P = model.language_model.layers.1.ple.ple_embedding.ngram_embedding`. At K = 4 the file is 26.2 GB. It is not
listed in `model.safetensors.index.json`; the loader globs every `*.safetensors` in the model directory
(`X/loader/safetensors.py:488-492`) and `NGramEmbedding` detects the `.trellis` tensors (`ngram_embedding.py:121-155`).
`compile.py` does not copy source `.safetensors` files, so the source table never reaches the output.

### 3.8 MTP and vision (side models, uncalibrated)

After the trunk, `quantize_side_model` (`convert_model.py:1974-2050`) quantizes the MTP head's modules with
`capture_H = None`: fallback path (exl3-algorithm.md 3.8), no LDLQ, at `mtp_bits` (default 4). MTP module list
(`X/architecture/qwen4_exp_mtp.py:64-115`): `Qwen4ExpMTPInputLayer` (key `mtp`: `fc_hidden`, `fc_embedding` Q,
`pre_fc_norm_*` copy), one full-attention block `mtp.layers.0` (built by the same `build_qwen4_block`, 1,536 expert
linears included), `Qwen4ExpMTPStackOut` owning `mtp.hyper_connection_mixer`. Output names mirror the trunk
(`mtp.layers.0.mlp.experts.E.gate_proj.trellis`, ...).

The MTP input combine has no HF reference. exllamav3 normalizes `pre_fc_norm_hidden` **per 2560 stream** (grouped,
`X/modules/arch_specific/qwen4_exp_mtp.py:137-160`, comment: "a semantic guess"); SGLang uses one Norm1p over all
10240 (`S/models/qwen4_exp_mtp.py:105-115`). Only matters for drafting, not for calibration.

Vision: 1.7.

### 3.9 Output tensors

Everything the trunk modules return from `get_tensors`, sharded (8 GB default) by module order, plus the side models:

- EXL3 (`trellis`, `suh`, `svh`, `mul1`): `layers.L.linear_attn.{in_proj_qkv, in_proj_z, out_proj}`,
  `layers.L.self_attn.{q_proj, k_proj, v_proj, o_proj, indexer.index_qk_proj}`,
  `layers.L.mlp.experts.E.{gate_proj, up_proj, down_proj}`, `layers.L.mlp.shared_expert.{gate_proj, up_proj,
  down_proj}`, `lm_head` (if `head_bits <= 8`), MTP equivalents, vision Linears (if `vision_bits <= 8`).
- F16 `.weight`: `linear_attn.in_proj_{a,b}`, `mlp.gate`, `mlp.shared_expert_gate`, `ple.key_proj`, `ple.value_proj`,
  `ple.conv1d`, every `input_mix_weight_down/up` and `block_inject_weight` (GatedResidual exports its fp16 tables,
  `X/modules/hyperconnections.py:473-483`), `lm_head.weight` if `-hb 16`.
- BF16 copies: `embed_tokens`, `hc_norm`, `q_norm`, `k_norm`, indexer layernorms, `linear_attn.{A_log, dt_bias,
  conv1d.weight, norm.weight}`, `ple.norm_*`, `mtp.pre_fc_norm_*`.
- `ngram_embedding.safetensors` (3.7). Not written: the source hashing buffers under `ple.ple_embedding.*` (their copies
  live in the table file).
- `config.json` with `quantization_config` (`bits` = final bpw, `head_bits`, `mtp_bits`, `vision_bits` if quantized),
  `quantization_config.json`, and the source's non-tensor files (`modeling_qwen4_exp.py`, tokenizer, templates)
  (`X/conversion/compile.py:19-246`).

### 3.10 Fork vs stock 1.5.3 (what changes results)

| Item | Stock 1.5.3 | Fork HEAD |
|---|---|---|
| `--out_scales` default | `always` | `auto` (gate/up judged by captured output sensitivity, which for MoE experts is routing-weighted from a second routing call, `block_sparse_mlp.py:1336-1341`) |
| Grouping caps | 32768 cols / 16 stacked | 500000 / 2048 (batching only) |
| `qwen4_exp.py` | | `GatedResidual.link_sites`, M-RoPE caps for vision (`git diff` 9+/4-): no numeric change |
| `hyperconnections.py` | | int8 decode tables and kernel paths (numerics of the mix only, same math) |
| n-gram codec and table file | | identical (diff only adds `quantize_embedding` for `--embed_bits`) |
| QSA dense threshold, GDN projection dtypes, routing | | identical |

---

## 4. Numerical precision

### 4.1 What each implementation does

| Step | HF (BF16 model) | exllamav3 conversion forward |
|---|---|---|
| Residual streams | **BF16** (repeat, adds in BF16) | **fp32** `[T, 4, D]` |
| Norms | fp32 math, output cast to input dtype | fp32 math; GR site: `normed` fp16 (`hyperconnections.py:626-628`) |
| GR low-rank gate | BF16 GEMMs | fp16 GEMMs (cuBLAS path) or the deterministic int8 hi/lo tile kernel on sm80+ (14-bit fixed-point copies of the fp16 tables); `inj` fp32; mixed `y` fp16 |
| GR apply | BF16 | `x + inj * float(y)` fp32 (`hyperconnections.py:660-661`, out-of-place during capture/advance) |
| Linear weights / inputs | BF16 | BF16 -> **fp16** weights; fp16 inputs; fp32 accumulate |
| q/k/v, indexer, gate/up outputs | BF16 | fp16 |
| q/k norm + RoPE | BF16, cos/sin cast to BF16 | fused: norm fp32 -> fp16, rotate fp32 (fp32 sin/cos) -> fp16 |
| Attention | sdpa, fp32 softmax | flash-style fp16 in, fp32 softmax/accumulate; `o * sigmoid(g)` fp16 |
| o_proj / out_proj / down_proj outputs | BF16 | **fp32** (`out_dtype = torch.float`) |
| GDN qkv projection | BF16 | fp16 (`_proj_dtype`, `gated_delta_net.py:23-25`); z, a, b fp32 |
| GDN conv | BF16 in, fp32 accumulate via F.conv1d | Triton kernel reads fp16 token-major, fp32 accumulate, **bf16** out |
| beta, g | fp32 g, BF16 beta | beta **bf16**, g fp32 (`gdn.cu:249-281`, softplus threshold 20) |
| l2norm, delta rule | fp32 (torch chunk kernel) | l2norm -> bf16 (fla `l2norm_fwd`), fla chunk kernels fp32 internal, state fp32, output **bf16** |
| GDN gated norm | fp32 norm, cast to BF16 **before** `* w`, then `* sigmoid(z)` | bf16 in, all fp32, fp16 out (`norm.cu:505-600`) |
| Router | softmax fp32 over BF16 logits | fp16 logits, top-k on fp16, softmax over the 10 in fp32, weights fp16 |
| Expert/shared SiLU*up | BF16 | **fp16 arithmetic** (`h2exp`, `h2rcp`, `__hmul2`, `activation_kernels.cuh:10-30, 170-185`) |
| Routed sum | BF16 index_add | fp32 (down fp32 out * fp16 weight, `index_add_` fp32); shared added as `sigmoid(z) * y` fp32 |
| PLE | BF16 | rows fp16 (decoded), key/value proj fp16, Kn/Qn/gate/Gv fp32, norm_conv fp16, conv fp16, delta fp32 |
| Mixer output -> head | BF16 | fp16 |
| Hessians | | fp32 `addmm_` of the fp16 inputs; rows with inf/NaN dropped |

### 4.2 What our forward should do

- Streams fp32, everything else as in gemma4-forward.md 5.2: fp16 rounding of every Linear input (the Hessians must
  see fp16 activations), fp32 accumulation, fp32 norms/softmax/RoPE angles/gates.
- GDN: keep the state and the chunk algebra in fp32. Rounding q/k/v/output to bf16 like exllamav3 is optional (it is
  less precise than fp32; matching it only matters for bit-closeness). Round the gated-norm output and the out_proj
  input to fp16 (that is what out_proj's Hessian sees).
- Use `exp` of masked cumulative decays as in 2.5; `g` can be very negative (`dt_bias` down to -11, `A_log` up to 3.8,
  checked), so `exp(G)` underflows to 0 harmlessly but `exp(G[i] - G[j])` for `j > i` must be masked before `exp`.
- MoE all-experts capture: the per-expert intermediate (fp16) is what feeds `block.mlp.E.down`. Computing SiLU in fp32
  and rounding once to fp16 is fine (exllamav3's fp16 SiLU differs by an ulp or two).
- The hyper-connection gate: fp32 math is closer to the trained model than either exllamav3 path; use fp32 GEMMs for
  the rank-320 projections (they are small: `T x 10240 x 320`) and round `y` to fp16 for the sublayer.
- fp16 overflow risk: the site output `y` is a weighted mean of normed streams, magnitudes are small; PLE and stream
  values stay fp32. Count non-finite Hessian rows as exllamav3 does.

---

## 5. Sizes and work, for a 32 GB card

### 5.1 Memory

| Item | Size |
|---|---|
| Calibration state per row, `2048 x 4 x 2560` fp32 | 83.9 MB (80 MiB) |
| State for 250 rows | **20.97 GB** (19.5 GiB) |
| One layer's weights fp16: routed experts `512 x (1280 x 2560 + 2560 x 640)` = 2.517G params | **5.03 GB** |
| GDN Linears (qkv, z, a, b, out) / attention Linears (q, k, v, o, indexer) | 116 MB / 103 MB |
| Two GR sites + router + shared expert | 26 MB + 2.6 MB + 9.8 MB |
| Hessians per layer: `attn.input` 2560^2 | 26.2 MB |
| `attn.o` / `attn.output` 6144^2 | 151 MB |
| `mlp.input` 2560^2 | 26.2 MB |
| 512 expert down 640^2 + shared down 640^2 | 839 MB + 1.6 MB |
| Hessians per layer, total | **1.04 GB** (same for both layer types) |
| GDN state per row (48 x 128 x 128 fp32) | 3.1 MB; fla's chunk temporaries about 112 KB/token = 229 MB per row |
| Attention scores per row if materialized (24 x 2048^2 fp32) | 403 MB (avoid with a flash-style kernel) |
| N-gram ids of the calibration set (250 x 2048 x 16 int64) | 65.5 MB; unique packed rows at most 8.2M x 82 B = 672 MB |
| Final mixer output, 250 rows fp16 | 2.6 GB |

Peak for "state on the card" (the Gemma design): 21 + 5 (weights) + 1 (H) + about 1-2 (row temporaries, quantizer)
= 28-29 GB. That is too tight on a 32 GB card that other jobs also use. Options:

1. Keep the state on the card and stream the routed-expert weights from pinned host memory in batches (e.g. 64
   experts = 630 MB) for both the capture and the advance pass: about 10 GB of PCIe traffic per layer, a few seconds.
   Peak about 24 GB. Recommended.
2. Keep the state in pinned host RAM (21 GB of the machine's 62 GB; about 27 GB were free when this was written) and
   stream rows: 84 MB per row per pass. Riskier for host memory.
3. Fewer rows (e.g. 128) halves the state, at some cost in Hessian quality for the 512 down projections.

### 5.2 Compute

| Work | FLOPs |
|---|---|
| All-experts capture, routed experts, per row: `2 x 2048 x 512 x 4,915,200` | 1.03e13 |
| per layer (250 rows) / whole model (48 layers) | 2.58e15 / 1.24e17 |
| Down-Hessian accumulation, per layer: `250 x 512 x 2 x 2048 x 640^2` | 2.15e14 (whole model 1.03e16) |
| `block.mlp.input` H per layer | 6.7e12 |
| Advance (top-10 + shared), per layer | 5.5e13 |
| GDN/attention projections, per layer per pass | about 6e13 |
| Delta-rule scan, per row per layer | about 2e10 (negligible) |

At an effective 50-100 TFLOPS of fp16 GEMM, the all-experts capture is 26-52 s per layer, 20-40 minutes for the model.
The trellis encoder dominates instead: 123.7G quantized decoder weights = 483M tiles, at the measured ~50k tiles/s
(K = 4, README) about 2.7 hours, plus the n-gram table (320M rings of 160, about 200M tile-equivalents, about 1.1
hours) and the MTP head (2.6G weights uncalibrated, about 10M tiles). The table job also reads 102 GB of BF16 source
rows from disk.

### 5.3 Disk

Output at 4 bpw: decoder about 62 GB + n-gram table 26.2 GB + embedding 1.27 GB (BF16) + GR tables 1.28 GB (fp16) +
head 0.48 GB (6 bpw) + MTP 1.3 GB + vision 0.33 GB (6 bpw) or 0.9 GB (BF16): **about 93 GB**. When this was written the
source folder held 100 GB of 360 GB and the disk had 271 GB free, so the finished download leaves about 11 GB: the
output (and any work folder) will not fit without freeing space.

---

## 6. Tokenizer and calibration rows

### 6.1 tokenizer.json

| Part | Value (verbatim where it matters) |
|---|---|
| model | `BPE`, `byte_fallback false`, `ignore_merges false`, `unk_token null`, `fuse_unk false`, no prefix/suffix; vocab 248044 entries (ids 0..248043); 247,587 merges stored as **pairs** `[["Ġ","Ġ"], ...]` |
| normalizer | `{"type": "NFC"}` |
| pre_tokenizer | `Sequence[ Split(Regex, behavior "Isolated", invert false), ByteLevel(add_prefix_space false, trim_offsets true, use_regex false) ]` |
| post_processor | `ByteLevel` (offsets only; adds no BOS/EOS) |
| decoder | `ByteLevel` |
| added tokens | 33, ids 248044..248076, all `single_word/lstrip/rstrip/normalized = false` |

The Split regex in `tokenizer.json` (the Qwen2 regex):

```
(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

`tokenizer_config.json` carries a different `pretokenize_regex` with `\p{M}` (`[\p{L}\p{M}]+` and
`[^\s\p{L}\p{M}\p{N}]`), the Qwen3.5 regex the vocab was probably trained with (5,434 vocab tokens contain a letter
followed by a mark and are unreachable without it). **Neither HF transformers nor exllamav3 reads it**: exllamav3 loads
`tokenizer.json` with HF `tokenizers` (`X/tokenizer/tokenizer.py:50`), so the calibration rows use the regex above.
Match that.

Alternatives, tried in order at each position (Oniguruma semantics): contraction (case-insensitive); one optional
non-CR/LF non-letter non-number character followed by letters; **exactly one** number character; optional single U+0020
then a run of non-space non-letter non-number characters then any CR/LF; whitespace up to and including the last CR/LF;
`\s+(?!\S)` (a whitespace run not followed by a non-space: the whole run at end of text, else the run minus its last
character); `\s+`. Classes: `\p{L}` = Lu/Ll/Lt/Lm/Lo, `\p{N}` = Nd/Nl/No, `\s` = Unicode White_Space (09-0D, 20, 85,
A0, 1680, 2000-200A, 2028, 2029, 202F, 205F, 3000; **not** 1C-1F).

ByteLevel: GPT-2 `bytes_to_unicode` (bytes 0x21-0x7E, 0xA1-0xAC, 0xAE-0xFF map to the same code point; the other 68
bytes map in increasing order to U+0100, U+0101, ...; space = `Ġ`, LF = `Ċ`). BPE: merge the lowest-rank pair,
leftmost on ties, until none applies.

Special tokens (special = true): `<|endoftext|>` 248044, `<|im_start|>` 248045, `<|im_end|>` 248046, object
ref/box/quad 248047-248052, `<|vision_start|>` 248053, `<|vision_end|>` 248054, `<|vision_pad|>` 248055,
`<|image_pad|>` 248056, `<|video_pad|>` 248057, audio/tts 248070-248076. Non-special added tokens: `<tool_call>`
248058, `</tool_call>` 248059, fim 248060-248063, `<|repo_name|>` 248064, `<|file_sep|>` 248065, `<tool_response>`
248066, `</tool_response>` 248067, `<think>` 248068, `</think>` 248069.

How exllamav3 encodes calibration text: `encode(text)` with `encode_special_tokens = False`, which sets HF's
split-special-tokens mode (`tokenizer.py:262-270`): **special** token strings in the text are encoded as plain text,
but **non-special** added tokens (`<think>`, `<tool_call>`, fim, ...) are still cut out first, longest match first, on
the raw text before NFC, and emitted as single ids. A native encoder must do the same.

Note: `src/io/tokenizer.hpp` already has a byte-level path (NFC, the split, the byte map, ranked BPE) as of
2026-10-24; a stdlib reference encoder agreed with it on 41 strings. Two known gaps: the non-special added tokens above,
and HF's NFC tables come from the `unicode-normalization-alignments` crate (reported as Unicode 9.0, while Python and
`unicode_tables.hpp` are 16.0: 21 newer composites and 120 newer combining classes), which only matters for rare
scripts. Verify against HF on the calibration texts with `tests/tok_check.py`.

### 6.2 Calibration rows (`X/conversion/calibration_data.py`, as in gemma4-forward.md 4)

- BOS/EOS: config `bos_token_id = eos_token_id = 248044`, so even-numbered rows of the article sources wrap every
  article as `248044 ... 248044`. (The exllamav3 tokenizer later rewrites `config.eos_token_id` to 248046 from
  `tokenizer_config.json` `eos_token`, `tokenizer.py:150-156`, but its own `eos_token_id` stays 248044, and the PLE
  module read 248044 at config time, `qwen4_exp.py:80`.)
- These 248044 tokens are **n-gram segment boundaries** in the PLE hashing: they change the n-gram ids of the
  following tokens. Random rows draw from `[0, 248077)` (`actual_vocab_size` = 1 + highest added id, `tokenizer.py:203`),
  so they contain the occasional 248044 too, and special/added ids, all looked up as plain embedding rows.
- No chat template in calibration. Each row is one sequence: positions 0..2047, fresh GDN state, fresh PLE context
  (two EOS), dense causal attention.

---

## 7. Subtle points that make a reimplementation silently wrong

1. **1 + w vs w.** Every RMSNorm in this model is `1 + w` except the GDN output norm (`w`). `hc_norm` and the PLE norms
   are grouped per 2560 (rms per stream), not one rms over 10240.
2. **GDN output gate is sigmoid** (`output_gate_type`), not SiLU. exllamav3 defaults to SiLU when the key is absent.
3. **No final norm**; the mixer output goes straight into lm_head. The mixer's mix is a **mean** over streams.
4. **Inject on raw streams**: `X += inj * out`, with `inj` from the *normed* stack but added to the un-normed streams.
5. **Interleaved q/gate per head** in q_proj: head h rows `[512h, 512h+256)` = query, `[512h+256, 512h+512)` = gate. A
   split into two halves of 6144 is wrong but still runs.
6. **Partial RoPE**: 64 of 256 dims, pairs `(j, j + 32)`, inv_freq exponent over 64. Not Gemma's proportional layout.
7. **GQA mapping**: q head h -> kv head h // 12; GDN v-head i -> k-head i // 3 (repeat_interleave, not tile).
8. **qkv split order** `[q 2048 | k 2048 | v 6144]` contiguous, not Qwen3-Next's per-k-head grouping. The conv runs on
   all 10240 channels in that order, before the split.
9. **Delta-rule details**: q scaled by `1/sqrt(128)` after l2norm (k is not scaled); `beta = sigmoid(b)` scales the
   correction `v - kv`; decay applied before the read (`St = exp(g) St` then `kv = St^T k`); unit-diagonal triangular
   solve in the chunked form.
10. **PLE position**: before layer **1** (1-based `ple_layer_ids = [2]`), on the raw streams, before that layer's
    attention site. The query norm reads the streams, not the mixed input.
11. **PLE hashing**: two EOS of context at row start; EOS (248044) resets context mid-row; heads 0-7 bigram, 8-15
    trigram; signed-sqrt gate with `max(|g|, 1e-6)` and `sign(0) = 0`; dilation 3 (taps at t, t-3, t-6, t-9, weight
    index 3 at t).
12. **The table used in calibration is the quantized one**, so the table must be quantized (with its bias) before layer
    0's calibration, and PLE must decode it exactly as `ngram_dequant` does (fp16 codebook value, times fp16 scale,
    plus fp16 bias, rounded to fp16).
13. **N-gram ring packing is LSB-first in uint16 words**, natural element order, ring length 160, pass-1 roll 80. The
    linear-tile format is different (MSB-first 32-bit words, tensor-core permutation, 256).
14. **QSA is dense at T <= 2051**; implementing the sparse selection for calibration changes nothing but cost.
    `index_qk_proj` takes q/k/v's Hessian.
15. **All-experts capture**: every expert's down Hessian over all tokens, unweighted; gate/up of all experts and the
    shared expert share one H. The **advance** uses real top-10 routing with renormalized weights.
16. **Expert split**: `gate_up_proj[e]` rows 0..639 gate, 640..1279 up; `down_proj[e]` is `[2560, 640]` (out, in).
    The 3-D tensors are `[expert, out, in]` (not transposed).
17. **Unquantized but fp16**: PLE key/value projections, router, shared_expert_gate, in_proj_a/b and all GR tables are
    stored as F16 `.weight`; norms and GDN scalars stay BF16. Shapes stay `[out, in]` (no 128 padding for `pad_to = 1`
    Linears).
18. **Random calibration ids reach 248076** (padding rows of the embedding are never used); the n-gram hash takes them
    as they are.
19. **Tokenizer**: use `tokenizer.json`'s regex (no `\p{M}`), not `tokenizer_config.json`'s; cut out non-special added
    tokens; encode special tokens as text.
20. **Calibration rows longer than 2051 tokens** (`--cols` > 2051) would enable QSA sparsity; keep 2048 or implement the
    indexer (top-k ties: exllamav3/SGLang/HF may order them differently).

---

## 8. Checks to run once the forward exists

1. Hash: recompute `layer_multipliers` from seed 1234 and the prime/offset tables (1.5), and compare the n-gram ids of
   a short text with a Python port of `HF:1106-1171` (CPU, stdlib).
2. PLE with the BF16 table on a few positions against a CPU fp32 implementation of 2.4.
3. GDN chunked scan against the sequential recurrence on one row (fp64 CPU reference), all 48 heads.
4. Attention layer: dims 64..255 of q/k unchanged by RoPE; GQA mapping; result equal to plain causal softmax.
5. GR site: `mix` against an fp64 CPU reference; `apply` adds into the raw streams.
6. MoE: renormalized top-10 equals softmax over the 10 selected logits; all-experts down H of expert e equals
   `mean_t a_e(t) a_e(t)^T` of its intermediate.
7. Layer outputs against exllamav3's reference rows if a CUDA box is available, else against an HF CPU forward of a
   few hundred tokens with `torch_dtype=float32` (needs the full 360 GB checkpoint and much RAM; layer-by-layer with the
   PLE table streamed is the practical form).
8. Table codec: pack/unpack round trip; `decode(encode(row))` equals the `deq` the encoder reports; every ring
   tail-biting.
