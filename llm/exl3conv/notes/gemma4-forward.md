# Gemma 4 31B text forward, and how exllamav3 converts it

Spec for the native C++/SYCL EXL3 quantizer's calibration forward. Target model: `Gemma4ForConditionalGeneration`,
`model_type: gemma4`, dense 31B (a Gemma 4 31B-based model; its siblings share the architecture).

Sources, and the short names used for file:line references below:

| Name | Path |
|---|---|
| `HF` | `~/b70/llm/trellis/image/rootfs/opt/venv/lib/python3.12/site-packages/transformers/models/gemma4/modeling_gemma4.py` (transformers 5.12.1, the version the model's config was written with) |
| `HF-rope` | `.../site-packages/transformers/modeling_rope_utils.py` (same install) |
| `HF-mask` | `.../site-packages/transformers/masking_utils.py` |
| `SGL` | `.../site-packages/sglang/srt/models/gemma4_causal.py` (SGLang 0.5.20, cross-check only) |
| `X/...` | `~/b70/llm/dsv41/exllamav3/exllamav3/...` (the exllamav3 fork, commit d968d96d) |

Config facts come from `https://huggingface.co/ReadyArt/Wicked-Stuff-31B-Final/raw/main/config.json`. Tensor
shapes come from the safetensors headers of both BF16 shards (fetched with HTTP range requests), which agree with
`model.safetensors.index.json` (1188 tensors, 31,273,086,512 parameters, 62.5 GB).

---

## 0. Config, as the forward uses it

| Key (`text_config`) | Value | Note |
|---|---|---|
| `hidden_size` | 5376 | = 42 x 128 |
| `num_hidden_layers` | 60 | |
| `layer_types` | 5 x `sliding_attention`, 1 x `full_attention`, repeated 10 times | full layers: 5, 11, 17, 23, 29, 35, 41, 47, 53, 59 (`L % 6 == 5`) |
| `num_attention_heads` | 32 | both layer types |
| `num_key_value_heads` | 16 | sliding layers |
| `num_global_key_value_heads` | 4 | full layers (only when `attention_k_eq_v`) |
| `head_dim` | 256 | sliding layers |
| `global_head_dim` | 512 | full layers |
| `attention_k_eq_v` | true | full layers have no `v_proj` |
| `attention_bias` | false | no biases anywhere |
| `sliding_window` | 1024 | window includes the query token |
| `rope_parameters.sliding_attention` | `rope_type: default`, `rope_theta: 10000` | |
| `rope_parameters.full_attention` | `rope_type: proportional`, `rope_theta: 1e6`, `partial_rotary_factor: 0.25` | no `factor` key, so factor = 1 |
| `intermediate_size` | 21504 | = 168 x 128 |
| `hidden_activation` | `gelu_pytorch_tanh` | |
| `rms_norm_eps` | 1e-6 | every norm |
| `final_logit_softcapping` | 30.0 | |
| `attn_logit_softcapping` | absent, so none | `X/architecture/gemma4.py:94` reads it with default 0.0 |
| `hidden_size_per_layer_input` | 0 | no per-layer embeddings (PLE) |
| `num_kv_shared_layers` | 0 | no KV sharing across layers |
| `use_double_wide_mlp` | false | |
| `enable_moe_block` | false | dense |
| `use_bidirectional_attention` | `"vision"` | only image tokens attend bidirectionally; text is causal |
| `vocab_size` | 262144 | |
| `tie_word_embeddings` | true | no `lm_head.weight` in the checkpoint |
| `bos_token_id` / `eos_token_id` / `pad_token_id` | 2 / 1 (top level: `[1, 106]`) / 0 | |

There is no `query_pre_attn_scalar`. Gemma 4 uses attention scale 1.0 (section 2.4).

---

## 1. Tensors

### 1.1 Text model (`model.language_model.*`), all BF16 on disk

Shapes are `[out, in]` (PyTorch `nn.Linear` layout). `L` = layer index.

| Tensor | Sliding layer (50 of them) | Full layer (10 of them) | What exllamav3 does |
|---|---|---|---|
| `embed_tokens.weight` | `[262144, 5376]` | | Copied as BF16 (`embed_bits` defaults to 16, `X/conversion/convert_model.py:288`). Also the source of the tied `lm_head` |
| `layers.L.input_layernorm.weight` | `[5376]` | `[5376]` | Copied BF16 |
| `layers.L.self_attn.q_proj.weight` | `[8192, 5376]` (32 x 256) | `[16384, 5376]` (32 x 512) | **Quantized** |
| `layers.L.self_attn.k_proj.weight` | `[4096, 5376]` (16 x 256) | `[2048, 5376]` (4 x 512) | **Quantized** |
| `layers.L.self_attn.v_proj.weight` | `[4096, 5376]` | **absent** (k_proj output is also V) | **Quantized** (sliding only) |
| `layers.L.self_attn.o_proj.weight` | `[5376, 8192]` | `[5376, 16384]` | **Quantized** |
| `layers.L.self_attn.q_norm.weight` | `[256]` | `[512]` | Copied BF16 |
| `layers.L.self_attn.k_norm.weight` | `[256]` | `[512]` | Copied BF16 |
| (`v_norm`) | no tensor, the norm has no weight | no tensor | nothing stored |
| `layers.L.post_attention_layernorm.weight` | `[5376]` | `[5376]` | Copied BF16 |
| `layers.L.pre_feedforward_layernorm.weight` | `[5376]` | `[5376]` | Copied BF16 |
| `layers.L.mlp.gate_proj.weight` | `[21504, 5376]` | same | **Quantized** |
| `layers.L.mlp.up_proj.weight` | `[21504, 5376]` | same | **Quantized** |
| `layers.L.mlp.down_proj.weight` | `[5376, 21504]` | same | **Quantized** |
| `layers.L.post_feedforward_layernorm.weight` | `[5376]` | `[5376]` | Copied BF16 |
| `layers.L.layer_scalar` | `[1]` | `[1]` | Copied BF16; read once as a Python float (`X/modules/transformer.py:85-93`) |
| `norm.weight` | `[5376]` | | Copied BF16 |
| `lm_head` (virtual) | `[262144, 5376]` | | Cloned from `embed_tokens` (`alt_key`, `X/architecture/gemma4.py:406-416`) and quantized at `head_bits` (fork default 6, `convert_model.py:282`; Twilight was made with `head_bits: 16`, so its `lm_head.weight` is stored as F16) |

Seven Linear tensors per sliding layer and six per full layer (366 in total, plus `lm_head`). There are no biases,
no per-expert tensors and no PLE tensors.

**Norm weights are used as `w`, not `1 + w`.** Gemma 2/3 store `w` and multiply by `1 + w`; Gemma 4 does not
(`HF:207-211`, and exllamav3 builds these norms with `constant_bias = 0.0`). The stored values confirm it: Twilight's
layer 0 `input_layernorm` has mean 4.88 and max 444, `norm.weight` max 510, layer 0 `q_norm` is a constant 1.0234,
and layer 5 `k_norm` a constant 0.0615. Getting this wrong breaks the model completely.

**`layer_scalar` is far from 1.** It is a persistent buffer (`HF:1381`) loaded from the checkpoint. Twilight's
values (base Gemma 4 31B, probably the same in Wicked-Stuff): L0 0.0894, L1 0.0654, L2 0.992, ..., L28 0.436, ...,
L58 0.754, L59 0.0364. It multiplies the **whole residual stream** after the layer, not only the layer's branch
(section 2.7).

### 1.2 Vision tower and multimodal projector (not used by the text forward)

| Tensor | Shape |
|---|---|
| `model.vision_tower.patch_embedder.input_proj.weight` | `[1152, 768]` |
| `model.vision_tower.patch_embedder.position_embedding_table` | `[2, 10240, 1152]` |
| `model.vision_tower.encoder.layers.N.{q,k,v,o}_proj.linear.weight` (N = 0..26) | `[1152, 1152]` |
| `model.vision_tower.encoder.layers.N.mlp.{gate,up}_proj.linear.weight` | `[4304, 1152]` |
| `model.vision_tower.encoder.layers.N.mlp.down_proj.linear.weight` | `[1152, 4304]` |
| `model.vision_tower.encoder.layers.N.{input,post_attention,pre_feedforward,post_feedforward}_layernorm.weight` | `[1152]` |
| `model.vision_tower.encoder.layers.N.self_attn.{q,k}_norm.weight` | `[72]` |
| `model.vision_tower.std_bias`, `std_scale` | `[1152]` |
| `model.embed_vision.embedding_projection.weight` | `[5376, 1152]` |

`audio_config` is null: the checkpoint has no audio tower and no `embed_audio`.

What exllamav3 does with them (`X/conversion/convert_model.py:390-434`, `X/conversion/compile.py:79-120`):

- The text conversion loop never touches the vision tower. Calibration is text-only.
- `--vision_bits` 0 means auto. The fork's `Gemma4VisionModel` declares `default_vision_bits: 6`
  (`X/architecture/gemma4.py:497-501`), so **by default the fork quantizes the tower and `embed_vision` to 6 bpw**
  after the text model, uncalibrated (`quantize_side_model`, `convert_model.py:1974-2050`, with `capture_H = None`,
  `state = None`). Its quant targets are the 27 x 7 encoder Linears, `patch_embedder.input_proj` and
  `embed_vision.embedding_projection` (all `qmap` set). Norms, `position_embedding_table`, `std_bias`/`std_scale`
  are copied.
- `--vision_bits 16` (and upstream exllamav3 1.5.3, which made Twilight) copies every `model.vision_tower.*` and
  `model.embed_vision.*` tensor unchanged as BF16 via `get_additional_compiled_tensors` (`gemma4.py:482-488`). The
  Twilight checkpoint has exactly that: the tower and projector are BF16, and its `quantization_config` has no
  `vision_bits`.
- **Recommendation:** copy the tower as BF16. `~/b70/llm/gemma/README.md` says the SGLang plugin keeps
  `vision_tower` weights BF16, and we have no vision calibration anyway.

---

## 2. Forward of one decoder layer

Notation: `T` = tokens in the row (2048 for calibration), `x` = residual stream `[T, 5376]`, `D` = head dim of the
layer (256 or 512), `Hkv` = KV heads (16 or 4). All matmuls are `y = x @ W^T` with `W` of shape `[out, in]`.

### 2.1 RMSNorm (every norm in the model)

```
rmsnorm(x, w):          # over the last dim, n = len(last dim)
    xf  = float32(x)
    r   = (mean(xf^2) + eps) ^ -0.5        # eps = 1e-6; eps is inside the root
    y   = xf * r * float32(w)              # w used as-is, no (1 + w)
    return cast(y, out_dtype)
rmsnorm_noweight(x) = same without the * w   # v_norm
```

HF: `HF:193-211` (`_norm` uses `pow(mean_sq, -0.5)`; the cast is `type_as(input)`, so a BF16 input gives a BF16 output).
exllamav3: `X/modules/rmsnorm.py:71-86` (torch reference) and `X/exllamav3_ext/norm.cu:150-300` (the kernel actually
used): fp32 sum of squares, `rsqrtf(sum/dim + eps)`, weight read as bf16 and widened to fp32, product in fp32, one
rounding to the output dtype. fp16 inputs are clamped to +-65504 when read (`read_half4<true>`).

q_norm and k_norm are per head (over `D`). The exllamav3 kernel fuses them into RoPE: it normalizes in fp32,
multiplies by the bf16 weight in fp32, **rounds to fp16**, then applies RoPE in fp32 and rounds to fp16 again
(`X/exllamav3_ext/rope.cu:200-283`).

### 2.2 Projections and per-head norms

```
h = rmsnorm(x, input_layernorm.w)                 # exllamav3: fp16 out
q = (h @ Wq^T).view(T, 32, D)
k = (h @ Wk^T).view(T, Hkv, D)
if sliding: v = (h @ Wv^T).view(T, Hkv, D)
else:       v = k                                  # attention_k_eq_v: raw k_proj output, before k_norm
q = rmsnorm(q, q_norm.w)         # per head, weight [D]
k = rmsnorm(k, k_norm.w)         # per head, weight [D]
v = rmsnorm_noweight(v)          # per head, no weight, on both layer types
q = rope(q); k = rope(k)         # section 2.3; V gets no RoPE
```

- HF: `HF:1229-1264`. The `v_proj is None` branch is at `HF:1258`: `value_states = ... if self.v_proj is not None else key_states`,
  taken **before** `k_norm` is applied (k_norm returns a new tensor, so V is the un-normalized k_proj output).
  v_norm is `Gemma4RMSNorm(head_dim, with_scale=False)` (`HF:1214`).
- exllamav3: `X/modules/attn.py:698` (`v = ... if not self.use_k_as_v else k`), v_norm in `finish_qkv`
  (`attn.py:733-741`), which runs before the fused q/k norm + RoPE (`attn.py:900-920`), so V is the raw k_proj
  output too. `use_k_as_v = layer_is_full and attention_k_eq_v` (`X/architecture/gemma4.py:236`).
- So on full layers: `K = rope(k_norm(P))`, `V = rmsnorm_noweight(P)`, with `P = h @ Wk^T` (`[T, 4, 512]`).

### 2.3 RoPE (NeoX / rotate-half layout)

Positions are `0 .. T-1` per calibration row (no packing offsets). For a head vector `v` of length `D`, with
`inv_freq` of length `D/2` and `theta_j = pos * inv_freq[j]`:

```
for j in 0 .. D/2-1:
    a = v[j]; b = v[j + D/2]
    v[j]       = a*cos(theta_j) - b*sin(theta_j)
    v[j + D/2] = b*cos(theta_j) + a*sin(theta_j)
```

(`HF:780-806`: `x*cos + rotate_half(x)*sin` with `cos = cat(freqs, freqs)`; `X/util/rope.py:124-141`, kernel
`X/exllamav3_ext/rope.cu:170-195`.)

`inv_freq` per layer type, both computed in fp32:

- **Sliding** (`default`, theta 1e4, `D = 256`): `inv_freq[j] = 10000^(-2j/256)` for `j = 0..127`. All 256 dims
  rotate. (`HF:1123-1157`, which uses `config.head_dim`; exllamav3 `rope.py:219-224`.)
- **Full** (`proportional`, theta 1e6, `D = 512`, `partial_rotary_factor 0.25`):
  `rope_angles = int(0.25 * 512 // 2) = 64`;
  `inv_freq[j] = 1e6^(-2j/512)` for `j = 0..63` (**the exponent's denominator is 512, the full head dim, not 128**),
  and `inv_freq[j] = 0` for `j = 64..255`. Divide by `factor` (1.0). (`HF-rope:187-254`, called with
  `head_dim_key="global_head_dim"` from `HF:1110-1111`; exllamav3 `rope.py:252-277`, `override_head_dim =
  global_head_dim` at `X/architecture/gemma4.py:120-130`.)
  Result: dims 0..63 pair with dims 256..319 and rotate; the other 384 dims see `cos = 1, sin = 0` (identity). This
  is not the usual "rotate the first 128 dims" partial RoPE: the rotated pairs are split across both halves of
  the head.
- `attention_scaling` (attn_factor) is 1.0 for both types.

HF computes `cos`/`sin` in fp32 and then casts them to the activation dtype (BF16 in a BF16 run, `HF:1160-1174`),
then rotates in BF16. exllamav3 keeps sin/cos in fp32 and rotates in fp32 from fp16 inputs.

### 2.4 Attention

```
group = 32 / Hkv                        # 2 on sliding layers, 8 on full layers
for query head hq: kv head = hq // group       (HF repeat_kv, HF:809-818: head i of K/V serves q heads i*group .. i*group+group-1)
s[t, u] = dot(q[t, hq], k[u, hkv]) * 1.0       # scale = 1.0 on both layer types (HF:1194; X/architecture/gemma4.py:242, 278)
mask:  allowed(t, u) = (u <= t)                                    full layers
       allowed(t, u) = (u <= t) and (u > t - 1024)                  sliding layers: the query plus 1023 earlier keys
p = softmax over allowed u (fp32)
o[t, hq] = sum_u p[t, u] * v[u, hkv]
```

- No logit softcapping in attention (Gemma 2 had it; Gemma 4 does not pass `softcap`, `HF:1272-1281`).
- Scale 1.0 is right because q_norm/k_norm already set the magnitudes; do not use `1/sqrt(D)`.
  (`Attention.__init__` computes `sm_scale or head_dim**-0.5`, `attn.py:214`; `1.0 or ...` stays 1.0.)
- Sliding mask: HF `sliding_window_overlay` is `kv_idx > q_idx - sliding_window` on top of causal
  (`HF-mask:88-97`). exllamav3 stores `sliding_window - 1 = 1023` (`gemma4.py:82-84`, comment "HF mask keeps
  sliding_window including the query") and passes `window_size = (1023, 1023)` with `causal = True` to its
  prefill kernel (`X/modules/sliding_attn.py:1008-1017`), which allows keys `t-1023 .. t`. Same set.
- **Calibration rows (2048 tokens) exceed the window**: for `t >= 1024` sliding layers drop the oldest keys. The
  window must be implemented in the calibration forward.
- `use_bidirectional_attention: "vision"` only matters for image soft tokens; text-only calibration is plain causal.
- Output: `o` reshaped to `[T, 32*D]` (head-major, `D` contiguous), then `a = o @ Wo^T` -> `[T, 5376]`.

### 2.5 MLP

```
h2 = rmsnorm(x, pre_feedforward_layernorm.w)
g  = h2 @ Wg^T                                   # [T, 21504]
u  = h2 @ Wu^T
m  = gelu_tanh(g) * u
d  = m @ Wd^T                                    # [T, 5376]
gelu_tanh(z) = 0.5 * z * (1 + tanh(sqrt(2/pi) * (z + 0.044715 * z^3)))
```

HF `HF:1068-1084`, `ACT2FN["gelu_pytorch_tanh"]` = `F.gelu(approximate="tanh")`. exllamav3: `GatedMLP(...,
activation_fn = "gelu", interm_dtype = torch.half, out_dtype = torch.float)` (`gemma4.py:298-311`), kernel
`X/exllamav3_ext/activation_kernels.cuh:42-63, 150-178`: g and u are fp16, `gelu` is computed in fp32 from fp16 `g`
and rounded to fp16, then `__hmul2` multiplies by `u` **in fp16**. The MLP is not sliced (21504 <
`MAX_MLP_INTERMEDIATE` = 55296, `X/constants.py:5`).

### 2.6 Residual structure (both layer types)

```
def decoder_layer(x, L):                 # x: [T, 5376]
    h  = rmsnorm(x, input_layernorm[L])
    a  = attention(h, L)                 # 2.2 - 2.4, includes o_proj
    x  = x + rmsnorm(a, post_attention_layernorm[L])
    h2 = rmsnorm(x, pre_feedforward_layernorm[L])
    d  = mlp(h2, L)
    x  = x + rmsnorm(d, post_feedforward_layernorm[L])
    x  = x * layer_scalar[L]             # whole stream, after both residual adds
    return x
```

HF `HF:1398-1455`: the MoE block (`HF:1428-1439`) and the PLE block (`HF:1445-1452`) are skipped for this config;
`hidden_states *= self.layer_scalar` is at `HF:1454`. exllamav3 `X/modules/transformer.py:143-221`: the post-norms
are run with `residual = x`, i.e. the kernel's `RES_POST` mode `x += norm(y) * w` into the fp32 stream
(`norm.cu:150-153, 250-262`), and `x *= layer_scalar_f` at `transformer.py:218-219`.

### 2.7 Embedding, final norm, head

```
x0 = float32( bf16( E[ids] * bf16(sqrt(5376)) ) )   # bf16(sqrt(5376)) = 73.5 exactly (sqrt = 73.3212)
for L in 0..59: x = decoder_layer(x, L)
y = rmsnorm(x, norm.w)
logits = y @ E^T                                     # tied: lm_head = embed_tokens, no scaling
logits = 30 * tanh(logits / 30)                      # final_logit_softcapping
```

- HF: `Gemma4TextScaledWordEmbedding` multiplies by `embed_scale.to(weight.dtype)` (`HF:1458-1470`, scale set at
  `HF:1602`; the comment there notes the BF16 rounding). The product is a BF16 tensor.
- exllamav3: the multiplier is pre-rounded, `torch.tensor(hidden_size ** 0.5, dtype = torch.bfloat16).float().item()`
  = 73.5 (`gemma4.py:209-217`). The embedding table loads as BF16 (`allow_bf16 = True`, `X/modules/embedding.py:134`);
  `x *= self.multiplier` on a BF16 tensor (`embedding.py:280-282`) rounds the product to BF16, then `to2` widens
  to fp32 (Embedding's default `out_dtype = torch.float`, `embedding.py:80`). Same values as HF.
- `hidden_size_per_layer_input = 0`: no PLE, nothing else is added to the embeddings.
- Softcap: `HF:2532-2535`; exllamav3 `Linear(softcap = final_logit_softcapping)` (`gemma4.py:414`, applied at
  `X/modules/linear.py:739-740`, kernel `X/exllamav3_ext/softcap.cu`: `y = scale * tanh(x / scale)`).
- Token ids: HF's multimodal wrapper replaces image/audio/video placeholder ids (258880, 258881, 258884) with
  `pad_token_id` before the lookup (`HF:2257-2265`). exllamav3 does not (only ids >= 1e9 are multimodal,
  `X/tokenizer/mm_embedding.py:4`); in calibration they are ordinary rows of `E`. It only affects the few random-token
  rows that happen to contain those ids. Match exllamav3 (plain lookup).

### 2.8 Whole text forward, compact

```
x = embed(ids)                                  # fp32 [T, 5376]
for L in 0..59:
    full = (L % 6 == 5)
    D, Hkv = (512, 4) if full else (256, 16)
    h = rmsnorm(x, w_in[L])
    q = rmsnorm_heads(h @ Wq[L]^T, w_qn[L])     # [T, 32, D]
    p = h @ Wk[L]^T                             # [T, Hkv, D]
    k = rmsnorm_heads(p, w_kn[L])
    v = rmsnorm_heads_noweight(p if full else h @ Wv[L]^T)
    q, k = rope_full(q, k) if full else rope_sliding(q, k)
    o = sdpa(q, k, v, scale = 1, causal, window = None if full else 1024, gqa = 32 / Hkv)
    x = x + rmsnorm(o.reshape(T, 32*D) @ Wo[L]^T, w_pa[L])
    h2 = rmsnorm(x, w_pf[L])
    x = x + rmsnorm((gelu_tanh(h2 @ Wg[L]^T) * (h2 @ Wu[L]^T)) @ Wd[L]^T, w_ff[L])
    x = x * layer_scalar[L]
logits = 30 * tanh((rmsnorm(x, w_norm) @ E^T) / 30)
```

---

## 3. How exllamav3 runs the conversion for Gemma 4

### 3.1 Module list (`Gemma4TextModel.modules`, `X/architecture/gemma4.py:194-417`)

The conversion loop walks `model.modules` in order (`X/conversion/convert_model.py:1638`). For this model:

| idx | Module | Quant targets (qmap) |
|---|---|---|
| 0 | `Embedding` `model.language_model.embed_tokens` | none (copied) |
| 1..60 | `TransformerBlock` `model.language_model.layers.{0..59}` | see 3.2 |
| 61 | `RMSNorm` `model.language_model.norm` (out fp16) | none, just forwarded |
| 62 | `Linear` `lm_head` (alt_key `embed_tokens`, qmap `"block"`, `head_bits`, softcap 30) | `lm_head` |

Sliding layers are built as `SlidingAttention` (because `swa_full` defaults to False and the pattern value is 1023 >
0), full layers as `Attention` (`gemma4.py:224-296`). In the `flash_attn_nc` mode the conversion uses, both compute
the same thing as section 2.4.

### 3.2 Per-module loop: capture, quantize, reload, advance

For each module (`convert_model.py:1638-1972`):

1. **Load** the module unquantized (Linear weights BF16 -> **fp16** via `float2half`/no `allow_bf16`,
   `X/modules/linear.py:285-320`; norms and `layer_scalar` stay BF16).
2. **Capture** (`convert_model.py:1716-1760`): run the unquantized module forward once over **every** calibration row,
   with `params = {"attn_mode": "flash_attn_nc", "capture": capture_H, ...}`. Each Linear with a qmap adds its input
   to the Hessian of its qmap (`linear.py:670-691`):
   `H[qmap] += x^T x` in fp32 (x cast from its fp16 activation, rows with any non-finite value dropped),
   `count[qmap] += rows`. Only the **first** Linear that sees a qmap accumulates (`first_key`), and the others
   reuse that H. For the first 5 rows (`num_ref_states = 5`, `convert_model.py:117`) the unquantized output is saved
   as a reference.
3. **Quantize** all Linears of the module (`convert_model.py:1784-1840`). Linears that share a qmap and bitrate are
   grouped and concatenated along out_features (`group_quant_linears`, `convert_model.py:738-783`). The order of
   quantization inside the layer does not matter for the Hessians: all of them were captured in step 2.
4. **Reload** the module from its quantized tensors (`convert_model.py:1863-1875`).
5. **Advance** (`convert_model.py:1877-1935`): run the **quantized** module over every row; its output replaces the
   stored state and becomes the input of the next module. Rows 0..4 are compared against the references from step 2
   (relative Frobenius error, cosine, SQNR). For the last module (lm_head) only those 5 rows are run.

Consequences for our implementation:

- Layer `L`'s inputs come from the quantized layers `0..L-1` (sequential, GPTQ-style across layers).
- **Inside a layer, nothing is sequential**: `o_proj`'s Hessian is captured from the attention output of the
  *unquantized* q/k/v, and gate/up/down Hessians from the output of the *unquantized* attention of the same layer.
  We do not need a re-forward between attention and MLP.
- One capture pass and one advance pass per layer, i.e. two forwards of each layer per row.

### 3.3 Hessian groups of one decoder layer

qmaps come from `qmap = "block.attn"` / `"block.mlp"` plus suffixes (`attn.py:287, 309, 323, 352`, `mlp.py:661, 680, 699`):

| qmap | Linears sharing it | Input captured (the tensor fed to the matmul) | H size | fp32 bytes |
|---|---|---|---|---|
| `block.attn.input` | q_proj, k_proj, v_proj (sliding) / q_proj, k_proj (full) | `h = input_layernorm(x)`, fp16 `[T, 5376]` | 5376^2 | 116 MB |
| `block.attn.o` | o_proj | attention output `o.reshape(T, 32*D)`, fp16 | 8192^2 (sliding) / 16384^2 (full) | 268 MB / 1.07 GB |
| `block.mlp.input` | gate_proj, up_proj | `h2 = pre_feedforward_layernorm(x)`, fp16 `[T, 5376]` | 5376^2 | 116 MB |
| `block.mlp.down` | down_proj | `m = fp16(gelu(g)) * u`, fp16 `[T, 21504]` | 21504^2 | 1.85 GB |
| `block` (module 62) | lm_head | `y = norm(x)`, fp16 `[T, 5376]` | 5376^2 | 116 MB |

The q/k/v Hessian is accumulated at q_proj (the first caller), and the gate/up one at gate_proj (`GatedMLP.forward`
calls the gate first, `mlp.py:874`). No input dimension needs padding: all are multiples of 128.

`optimizer_targets` (for the fork's later bit-allocation tools) groups them as `[[q, k+v, o]]` and `[[g+u, d]]`
(`attn.py:426-431`, `mlp.py:732-737`). The fork also records output sensitivities of gate/up during capture for
`--out_scales auto` (`capture_out_sensitivity`, `mlp.py:66, 877`); see exl3-algorithm.md.

### 3.4 State storage

The calibration state is a list of `cal_rows` tensors of shape `[1, 2048, 5376]`, **fp32**, kept on the host between
modules (`state[i] = rs.cpu()`): 250 x 2048 x 5376 x 4 B = **11.0 GB of host RAM**, plus up to 5 reference rows.
Rows that turn non-finite are dropped from all later steps (job aborts above 10%). The fork can batch rows
(`calib_batch_rows`, default up to 32) for the forwards; batching does not change the math.

---

## 4. Calibration tokens and BOS

Default calibration (`X/conversion/calibration_data.py:61-99`): `cal_rows = 250`, `cal_cols = 2048`, rows taken from
bundled files in `X/conversion/standard_cal_data/` by weight (sum 135):

| Source | Weight | Rows (for 250) | Splitter | Special tokens |
|---|---|---|---|---|
| `c4.utf8` | 20 | 37 | `shuffle_lines` (lines, `random.seed(0)` shuffle) | per article, see below |
| `code.utf8` | 20 | 37 | `split_raw` | none |
| `multilingual.utf8` | 10 | 18 | `shuffle_lines` | per article |
| `technical.utf8` | 10 | 18 | `split_raw` | none |
| `wiki.utf8` | 50 | 92 | `split_wiki` (split on `</doc>`, drop the title line, keep > 50 chars) | per article |
| `tiny.utf8` | 5 | 9 | `split_tiny` (split on `<|endoftext|>`) | per article |
| random | 20 | 39 (the remainder) | `torch.manual_seed(0); randint(0, 262144, (1, 2048))` per row | none |

`split_art` (`calibration_data.py:5-19`): articles are concatenated into the current row and the row is cut at 2048
tokens; the rest of the article that overflowed is dropped and the next article starts the next row. Each article is
encoded with `add_bos = add_eos = (len(t_rows) % 2 == 0)`, so **every article in even-numbered rows (0, 2, 4, ...
of that source) is wrapped as `<bos>=2 ... <eos>=1`, and odd-numbered rows have no special tokens at all**. A BOS can
therefore sit mid-row. `split_raw` encodes the whole file once with no BOS (`encode(text)`: `add_bos` defaults to
False, `X/tokenizer/tokenizer.py:340-389`) and takes consecutive 2048-token slices. Special-token strings inside the
text are encoded as plain text (`encode_special_tokens = False`).

Token ids: `bos_token_id = 2` and `eos_token_id = 1` (`X/model/config.py:146-156`: the first entry of the top-level
`eos_token_id: [1, 106]`, consistent with `tokenizer_config.json` `eos_token: "<eos>"`). The tokenizer is a 262144-entry
BPE (`tokenizer.json`), whose post-processor adds nothing.

No chat template is applied in calibration. (`default_chat_prompt`, `gemma4.py:447-454`, is only for exllamav3's own
chat examples: `<bos><|turn>user\n...<turn|>\n<|turn>model\n`.)

Every row is an independent sequence: positions 0..2047, causal attention across article boundaries inside the row,
no document masking. Many rows (odd rows, code/technical, random) have no BOS at position 0. Gemma models put
large activations on the first token whatever it is, so the first-position statistics differ from a BOS-led row.
That is fine, but our token rows must match.

**Reproducing exllamav3's rows exactly needs torch's CPU RNG** for the random rows (`torch.manual_seed(0)` +
`randint`), which a stdlib/C++ tool cannot reproduce cheaply, and the HF `tokenizers` BPE for the text rows. Practical
option: accept a pre-tokenized calibration file in exllamav3's `--cal_data` format (safetensors `input_ids [rows,
cols]` int, optional `lengths`, `calibration_data.py:102-130`), and generate the default set once with exllamav3's
own code outside the B70 (CPU only). For our own random rows, any seeded uniform `[0, 262144)` generator is
equivalent in distribution.

---

## 5. Numerical precision

### 5.1 What each implementation does

| Step | HF (model loaded in BF16, the default) | exllamav3 conversion forward |
|---|---|---|
| Embedding | BF16 row x BF16(73.5) -> BF16 | same BF16 product, widened to fp32 |
| Residual stream `x` | **BF16** (adds and the `layer_scalar` multiply in BF16) | **fp32** |
| RMSNorm math | fp32, weight fp32, output cast to input dtype (BF16) | fp32, bf16 weight widened, output fp16 (pre-norms, final norm) or added into fp32 `x` (post-norms) |
| Linear weights | BF16 | BF16 converted to **fp16** (exact except below ~6e-8 or above 65504) |
| Linear activations / outputs | BF16 in, fp32 accumulate, BF16 out | fp16 in, fp32 accumulate; q/k/v/o/gate/up out **fp16**, down_proj out **fp32**, lm_head out fp16 |
| q/k norm + RoPE | norm in fp32 -> BF16; cos/sin computed fp32, **cast to BF16**, rotation in BF16 | norm fp32 -> fp16, rotation in fp32 with fp32 sin/cos -> fp16 |
| v_norm | fp32 -> BF16 | fp32 -> fp16 |
| Attention | BF16 q/k/v, softmax fp32 (eager `HF:848`; sdpa/flash similar) | fp16 q/k/v, fp32 softmax and accumulation (flash-style kernels) |
| GELU x up | BF16 | gelu fp32 from fp16 -> fp16, product in fp16 |
| Hessian accumulation | n/a | fp32 `addmm_` of the fp16 activations |
| Logits + softcap | BF16 | fp16 logits, softcap on fp16 |

### 5.2 What our forward needs

The goal is Hessians and propagated states that match exllamav3's to well within quantization noise. HF in BF16 is
the less precise reference; exllamav3 is closer to an fp32 forward everywhere except the fp16 activations. So:

- **Residual stream: fp32.** Required: `layer_scalar` (0.036 at L59, 0.065 at L1) and the large norm weights make the
  stream's scale change a lot between layers, and BF16 adds lose the small branch outputs.
- **Norms, softmax, RoPE angles, GELU, softcap: fp32 math.** Compute `pos * inv_freq` in fp32 the way HF does
  (`inv_freq` as fp32, product as fp32) and evaluate sin/cos in fp32 (or better). Positions only reach 2047, so fp32
  is exact enough.
- **Linear inputs: round to fp16 before each GEMM, accumulate in fp32.** This is exactly what exllamav3 feeds its
  Hessians with (H is built from fp16 activations). BF16 inputs would also work but shift H by about 2^-8 relative
  per element; fp16 keeps us bit-close to exllamav3.
- **GEMM weights:** BF16 weights (exact in fp32, and in fp16 for normal values). Using BF16 weights directly with
  fp32 accumulation equals exllamav3's fp16-weight GEMM up to accumulation order.
- **Intermediate roundings worth copying** if we want near bit-identical states: q/k/v rounded to fp16 after the
  GEMM; q/k rounded to fp16 after the norm and after RoPE; attention output fp16; o_proj output fp16; gate/up
  outputs fp16, `gelu(g)` rounded to fp16, product rounded to fp16; down_proj output kept fp32.
- **fp16 overflow:** pre-norm outputs are `x_hat * w` with `w` up to ~444 (layer 0 input norm) and 510 (final norm),
  so a token with `|x_hat|` in the hundreds would overflow fp16. exllamav3 writes inf there and its capture drops
  any row with a non-finite element (`linear.py:685-689`), counting it. Do the same and report the count; if it
  happens often, compute that layer's activations in BF16 or fp32 instead.
- **Attention accumulation:** fp32 scores and accumulators. Scores are `q . k` with scale 1 after the norms; the
  magnitudes are moderate (k_norm weight 0.0615 on full layers), so overflow is not a concern in fp32.

### 5.3 Checks to run once the forward exists

1. Embedding of a known id equals `bf16(E[id] * 73.5)`.
2. Full-layer RoPE: dims 64..255 and 320..511 of q/k are unchanged by RoPE at any position.
3. Full layer: `V == rmsnorm_noweight(k_proj(h))` per head, without k_norm and without RoPE.
4. Sliding layer at `t = 2047`: the attention weight on keys `u <= 1023` is exactly 0.
5. Per-layer output against exllamav3's reference rows (`num_ref_states` = 5, same tokens): relative Frobenius error
   of the unquantized layer output well below 1e-3.
6. Final logits of a short BOS-led prompt against HF (CPU, fp32) top-k.

---

## 6. Next architectures (where the design needs room)

Both are in the same exllamav3 conversion loop: module list -> capture -> quantize -> reload -> advance.

**Qwen3.8-Flash-Next** (`qwen4_exp`, `X/architecture/qwen4_exp.py`):
- **Hyper-connections:** 4 fp32 residual streams per token (`ExpandStreams` after the embedding). Each layer has
  low-rank gated-residual mixing sites (`GatedResidual`, `attn_hyper_connection` / `mlp_hyper_connection`) in place
  of the input/post layernorms. A final `hyper_connection_mixer` collapses the streams instead of a final norm
  (`qwen4_exp.py:225-290`). The state becomes `[T, 4, hidden]` fp32: 4x the host memory of section 3.4.
- **Mixed attention:** Gated DeltaNet linear-attention layers (`linear_attention`: conv1d, `A_log`, `dt_bias`, fused
  `in_proj_qkv`, `in_proj_z/b/a`, sigmoid-gated output norm) and full GQA layers with an **interleaved output gate**
  (q_proj outputs q and gate interleaved per head, so it is twice as wide), q/k norms with `constant_bias = 1.0`
  (i.e. `1 + w`), and a **QSA indexer** (`QSAIndexer`): above a sequence threshold it picks 4-token key blocks
  under a token budget. The indexer has its own projections (qmap `block.attn`). Calibration rows longer than the
  threshold take the sparse path.
- **MoE:** `BlockSparseMLP` with a softmax router (`gate`), top-k experts, a shared expert with a sigmoid
  `shared_expert_gate`. `calibration_all_experts = True`: during capture every expert sees every token, so all
  `down_proj`s get Hessians, while the advance pass uses real routing. Per-expert Hessians, so many more H matrices
  per layer.
- **PLE / hashed n-gram embedding:** a `PLELayer` before the layers listed in `ple_layer_ids` adds hashed n-gram
  embeddings (a 51.2B-row table of 160-dim rows, streamed from disk) into the stream stack. Conversion quantizes the
  table first (`X/conversion/ngram.py:600+`, one tail-biting trellis ring per row, mul1 codebook) and calibrates
  with the quantized table.
- MTP head (`qwen4_exp_mtp.py`) quantized at `mtp_bits`; Qwen3.5-VL vision tower when present.

**GLM-5.3-Flash** (`glm5_next`, `X/architecture/glm5_next.py`):
- **KDA linear attention** (Kimi-style per-channel decay delta rule) on 3 of every 4 layers, via `GatedDeltaNet` with
  separate q/k/v convs, `f_a/f_b` and `g_a/g_b` low-rank gate projections.
- **NoPE MLA** with a **DSA indexer** compressed by a k-pool (`MLAttention`, `indexer_types` full/shared,
  `index_topk`), with q/kv LoRA ranks.
- **mHC hyper-connections** (DeepSeek-V4 style, Sinkhorn-normalized mixing, `hc_mult` 4) around both sublayers of
  every layer, *with* input/post-attention norms inside them; `HyperHead` collapses with an unweighted mean before
  the final norm.
- **MoE:** sigmoid scoring, `noaux_tc` with `e_score_correction_bias`, normalized top-k, `routed_scaling_factor`,
  shared experts, clamped SiLU (`swiglu_limit` -> `act_limit`); the first k layers are dense.
- MTP head and a GLM-4V vision tower.

Design implications: the per-layer state must allow `[T, S, hidden]` stream stacks; a layer must be able to declare
an arbitrary set of qmap groups (including per-expert ones with "activate all experts" during capture only); Linear
inputs may be produced by non-standard ops (gates, indexers, convs); some Linears (interleaved q+gate, fused qkv)
are split or interleaved views of one checkpoint tensor; and a layer may need sequence-length-dependent attention
paths (QSA/DSA sparse above a threshold, KDA/GDN recurrent scans).
