# Qwen3.8-Flash-Next (`qwen4_exp`) text forward: implementation spec

This spec covers the text model only: trunk, MTP head (brief), and the n-gram table. It is written for a C++/SYCL port
that has no Python reference open. It was derived from code on 2026-11-02; nothing was run on a GPU.

## 0. Sources and conventions

| Prefix | Path | Role |
|---|---|---|
| `X/` | `~/b70/llm/dsv41/exllamav3/exllamav3/` | exllamav3 fork, **authoritative** for the EXL3 pack |
| `HF` | `~/models/llm/Dongfangshuo-Qwen3.8-Flash-Next-exl3/modeling_qwen4_exp.py` | HF remote code (transformers 5.8.0.dev0). Ground truth for semantics; the exllamav3 parity tests compare against it |
| `SG/` | `~/b70/llm/trellis/image/rootfs/opt/venv/lib/python3.12/site-packages/sglang/srt/` | stock SGLang 0.5.20. This is what trellis-serve runs on the B70 |
| `TS/` | `~/b70/llm/state/trellis-serve/xpu/exl3xpu/` | trellis-serve's XPU patches on top of `SG/` |
| `CFG` | `/tmp/claude-1000/-home-horvay-b70/ebf29cfa-12d3-4279-8fce-7c3e358ccf7b/scratchpad/swift/config.json` | Swift 3.05 pack config (`text_config`) |
| `PK/` | `~/models/llm/Swift-1.5-Qwen3.8-Flash-Next-EXL3-3.05bpw-heretic/` | the pack: `model.safetensors.index.json` and `chat_template.jinja` are present; the shards are not downloaded yet |
| `QC` | `CFG` dir `/quantization_config.json` | the pack's per-tensor storage map (shapes, dtypes, bits) |
| `N` | `~/b70/llm/exl3conv/notes/qwen4exp-forward.md` | earlier converter-side spec (same model). It agrees with this one; the conversion details live there |

Notation:
- `T`: tokens in the pass. `D = 2560`. `S = 4` streams.
- `X[t, s, :]`: the residual stream stack, stream-major. HF stores it flattened as `[T, S*D]`, with stream `s` at
  columns `s*D .. s*D+D-1`.
- Every linear is `y = x @ W^T` with `W [out, in]`. `x[a:b]` is half-open.
- `fp16` = IEEE half, `bf16` = bfloat16.
- "X does" = what the exllamav3 kernels do. "HF does" = the reference.
- HF runs the whole model in bf16. X keeps the streams in fp32 and runs most activations in fp16 (§2.2).

---

## 1. Config and derived constants (`CFG.text_config`)

| Name | Value | Derived / note |
|---|---|---|
| hidden `D` | 2560 | |
| layers | 48 | layer `L` is full attention iff `L % 4 == 3` (3, 7, ..., 47): 12 attention layers, 36 GDN layers (`layer_types`) |
| `rms_norm_eps` | 1e-6 | every RMSNorm, the GDN l2norm, the PLE norms |
| `hc_count` `S` / `hc_lowrank` `r` | 4 / 320 | `S*D = 10240` |
| GDN | 16 k-heads, 48 v-heads, head dim 128 (k and v), conv 4 | key_dim 2048, value_dim 6144, conv_dim 10240, v-heads per k-head 3 |
| `output_gate_type` | `sigmoid` | GDN gated-norm activation (`HF:490-492`) |
| `mamba_ssm_dtype` | float32 | GDN recurrent state |
| attention | 24 q heads, 2 kv heads, head_dim 256 | GQA group 12; `sm_scale = 256^-0.5 = 1/16` (`HF:819`) |
| rope | theta 1e7, `partial_rotary_factor` 0.25, `mrope_section [11,11,10]`, `mrope_interleaved` | rotary dim 64 (32 frequencies) out of 256; for text all 3 mrope axes carry the same position, so it reduces to plain rope (§7.2) |
| QSA indexer | 4 q heads x 128, 1 k head x 128, budget 2048, compress ratio 4 | `block_topk = 2048/4 = 512`; dense-exact while `p <= 2050` (§7.4) |
| MoE | 512 experts, top-10, `moe_intermediate_size` 640, shared expert 640 | `norm_topk_prob` absent, defaults true (`N` §0) |
| PLE | `ple_layer_ids [2]` (1-based), `ple_embed_dim` 2560, `ple_conv_kernel_size` 4 | injected in front of 0-based layer **1** (`HF:1259`, `X/architecture/qwen4_exp.py:249-268`); conv dilation = `ngram_size` = 3 (`HF:1191`) |
| n-gram | `ngram_size` 3, `heads_per_ngram` 8, `ngram_vocab_size_base` 2e7, `make_ngram_vocab_size_divisible_by` 128, `split_ngram_parts` 128 | 16 hash heads (8 bigram, 8 trigram), 160 dims each; table 320,001,536 rows in 128 shards of 2,500,012 |
| vocab | 248320 | |
| `bos_token_id` / `eos_token_id` (text_config) | 248044 / 248044 (`<|endoftext|>`) | 248044 is the n-gram segment token |
| `tie_word_embeddings` | false | |
| MTP | 1 layer, full attention | §10 |
| quant | EXL3 1.5.3, 3.05 bpw, `head_bits` 5, `mtp_bits` 3, codebook mul1, `out_scales always` | per-tensor K in §12 |

---

## 2. Whole forward

### 2.1 Order (`HF:1391-1484`, `HF:1264-1301`; X module list `X/architecture/qwen4_exp.py:233-304`)

```
e  = E[ids]                                   # [T, D]; bf16 table, no scaling
X  = repeat e into 4 streams                  # [T, 4, D]  (ExpandStreams, fp32 in X)
for L in 0..47:
    if L == 1: X = X + PLE(X, ids)            # §5, before layer 1's attn site
    y, inj = MIX(X, L.attn_hyper_connection)  # §4
    o = GDN(y) if L % 4 != 3 else ATTN(y)     # §6 / §7
    X = APPLY(X, o, inj)
    y, inj = MIX(X, L.mlp_hyper_connection)
    o = MOE(y)                                # §8
    X = APPLY(X, o, inj)
h = MIX_FINAL(X, hyper_connection_mixer)      # [T, D]; NO final RMSNorm
logits = h @ W_lm^T                           # [T, 248320]; no softcap
```

There are no `input_layernorm` or `post_attention_layernorm` tensors. The GatedResidual mix is the only thing in front
of each sublayer (`X/architecture/qwen4_exp.py:101-107`, `X/modules/transformer.py:156-202`: with
`attn_norm = mlp_norm = None`, `y` is the mix output cast to fp16).

Prefill shortcut (X): tokens `0..n-2` stop after layer 47's attention. Its MoE, the mixer and lm_head are skipped,
because nothing they produce is cached (`X/model/model_ls.py:341-348`, `X/modules/transformer.py:167-168`). The last
token then runs the full forward.

### 2.2 Dtypes, X vs HF vs SGLang

| Tensor | X | HF | SGLang/trellis |
|---|---|---|---|
| stream stack `X` | **fp32** (`X/modules/hyperconnections.py:59-60`) | bf16 | bf16 (`SG/models/qwen4_exp.py:1299, 1326`) |
| mix output `y` (sublayer input) | fp16 (`X/modules/transformer.py:159, 183`) | bf16 | bf16 |
| sublayer outputs `o` (GDN out_proj, o_proj, MoE sum) | fp32 (`X/architecture/qwen4_exp.py:135, 167, 203`) | bf16 | bf16 |
| GatedResidual tables | fp16 (pack) | bf16 | bf16 |
| GDN conv in/out, conv state, beta, core output | bf16 (`X/modules/gated_delta_net.py:276-280, 1276, 1356`) | bf16 | bf16 |
| GDN gate `g`, recurrent state | fp32 | fp32 | fp32 |
| K/V cache | fp16 (or X's quantized cache) | bf16 | **fp8_e4m3** in the B70 serve script (fork report, `TS/sglang_plugin.py:358-362`) |
| PLE key/query/gate/delta | fp32; conv stream fp16 (`X/exllamav3_ext/ple.cu:178-205`) | bf16 | bf16 |
| n-gram decoded row | fp16 | bf16 (table) | fp16 value stored as bf16 (`TS/../csrc/exl3_moe.sycl:1334-1336`) |

HF is the bf16 baseline. X's fp32 stack has more precision than HF, not less. SGLang matches HF.

---

## 3. Norm primitives

```
rms(x)            = x * rsqrt(mean(x^2) + 1e-6)              # fp32, eps inside the root
Norm1p(x, w)      = rms(x) * (1 + w)                         # Qwen4ExpTextRMSNorm, HF:143-163
GNorm1p(X, w)     = for each stream s: rms(X[s]) * (1 + w[s*D : (s+1)*D])   # same class, group_size = D (HF:152-156)
GatedNorm(o,z,w)  = (w * rms(o)) * sigmoid(z)                # HF:178-187: weight used as-is, NOT 1 + w
l2norm(x)         = x * rsqrt(sum(x^2) + 1e-6)               # HF:247-250
```

- "constant_bias = 1.0" in X means `weight + 1` (`X/modules/rmsnorm.py:84`; `X/exllamav3_ext/rope.cu:247-257`).
- Every `Qwen4ExpTextRMSNorm` weight is zero-initialised and stored as the offset (`HF:1333-1335`). `1 + w` can be
  negative (`N` §2.1: hc_norm ranges -2.30..3.63). Do not clamp.
- Grouped norms (`groups = 4` in X) pick the weight row by `row % groups` over rows of D (`X/modules/rmsnorm.py:38-41`).
- HF casts the output back to the input dtype after `* (1 + w)` (`HF:162-163`).

| Norm | Kind | Dim | Tensor |
|---|---|---|---|
| `*_hyper_connection.hc_norm`, `hyper_connection_mixer.hc_norm` | GNorm1p | 4 x 2560 | `[10240]` fp16 in the pack |
| `self_attn.q_norm`, `k_norm` | Norm1p per head | 256 | `[256]` bf16 |
| `indexer.q_layernorm`, `k_layernorm` | Norm1p per head | 128 | `[128]` bf16 |
| `ple.norm_key`, `norm_query`, `norm_conv` | GNorm1p | 4 x 2560 | `[10240]` bf16 |
| `linear_attn.norm` | GatedNorm per v-head, sigmoid | 128 | `[128]` bf16, values near 1 |
| `mtp.pre_fc_norm_embedding` | Norm1p | 2560 | |
| `mtp.pre_fc_norm_hidden` | Norm1p, grouping disputed (§10) | 10240 | |

---

## 4. Hyper-connections (GatedResidual)

### 4.1 ExpandStreams

```
X[t, s, :] = float32(e[t, :])   for s = 0..3
```

`X/modules/hyperconnections.py:47-60`; `HF:1471` (`hidden_states.repeat(1, 1, 4)`). There are no tensors (`hc_expand`
has an empty storage entry, `QC`).

### 4.2 Site mix ("pre"): 4 streams to one sublayer input

Per token, `X [4, D]` fp32. Tensors per site (`QC`):

| Tensor | Shape | dtype |
|---|---|---|
| `hc_norm.weight` | `[10240]` | fp16 |
| `input_mix_weight_down.weight` (`Wd`) | `[320, 10240]` | fp16 |
| `input_mix_weight_up.weight` (`Wu`) | `[10240, 320]` | fp16 |
| `block_inject_weight.weight` (`Wi`) | `[4, 10240]` | fp16 (site form only) |

```
N    = GNorm1p(X, hc_norm.w)                     # [4, D] fp32; rms per stream (2560 values)
f    = N.flatten()                               # [10240], stream-major
a    = silu((f @ Wd^T) / 4)                      # [320]   (/ hc_count BEFORE silu)
G    = sigmoid(a @ Wu^T).reshape(4, D)           # [4, D]
y    = mean_s (G[s] * N[s])                      # [D]: divide by 4, the sublayer input
inj  = 2 * sigmoid((f @ Wi^T) / 4)               # [4], one scalar gate per stream
```

- `HF:1012-1022`; X fp32 reference `X/modules/hyperconnections.py:498-510`, kernels at `:569-636`.
- The norm is applied to the streams **before** both projections. The inject gate reads the **normed** stack too
  (`HF:1021`).
- X's decode kernel (`R <= 8` rows, `FUSED_MAX_R`, `:351, :583-603`) dots the unnormalised streams against the fp16
  tables and applies `1/rms` and `(1 + w)` inside the kernel. The math is the same.
- X fork-only: on CUDA with `EXL3_GR_INT8=1` (default), decode uses int8 copies of the tables with a per-row scale,
  with `(1 + w)` folded into `Wd`/`Wi` (`:14-16, 436-463, 598-600`). That is a lossy approximation of HF. A port
  should start from the fp16 tables.
- Prefill in X uses a deterministic int8 hi/lo tensor-core split, `ext.gr_mix_tiled` (`:604-621`). That is a CUDA
  determinism device; the cuBLAS form at `:622-635` is the plain math: `rms_norm` fp16 out, then `matmul`, `silu`,
  `matmul`, `sigmoid`, mean.

### 4.3 Site apply ("post"): write the sublayer output back

```
X[s, :] += inj[s] * o[:]        for s = 0..3     # on the RAW (un-normed) streams; o = sublayer output
```

`HF:1293-1294, 1299-1300` (`hyper_input + injection`, where `hyper_input` is the unnormed input).
`X/modules/hyperconnections.py:649-686` (`ext.hc_apply` with `comb = None`, in place, fp32). There is no combine
matrix, no Sinkhorn and no stream mixing: each stream is updated independently. This is the "low-rank, elementwise
cousin of mHC" (`:312-319`).

### 4.4 Final mixer (combine-less)

`model.language_model.hyper_connection_mixer` has the same mix with no `Wi` (`use_combine = False`), and returns `y`
only.

```
h = mean_s( sigmoid(silu((f @ Wd^T)/4) @ Wu^T).reshape(4,D)[s] * N[s] )     # [D]
```

- `HF:1019-1020, 1484`; `X/architecture/qwen4_exp.py:281-291` (out fp16); `X/modules/hyperconnections.py:688-704`.
- `h` feeds lm_head directly.
- For MTP, X exports the **pre-mixer** stack (`X.flatten(-2)` as fp16, `:692-698`).

### 4.5 `link_sites`

`X/modules/hyperconnections.py:512-536` (called at `X/architecture/qwen4_exp.py:304`) is a fusion detail only, with no
semantic effect:
- Each site records its successor site in forward order: attn to mlp within a block, mlp to the next block's attn,
  the last one to the final mixer.
- At decode, `apply_` writes the successor's weighted stream copy `X * (1 + w_next)` while the updated streams are in
  registers, and the next mix consumes it (`:538-550, 668-685`).
- Any other module between two sites breaks the chain. Here that is the PLE layer between layer 0's mlp site and
  layer 1's attn site; that mix then applies the weight itself.

A port can do the same fusion, or ignore it.

---

## 5. PLE / hashed n-gram embedding (in front of layer 1)

### 5.1 n-gram row ids (exact)

Constants. Read them from the table file (`P.layer_multipliers`, `P.head_vocab_sizes`, `P.head_offsets`, §5.3). They
equal what HF derives (`HF:1032-1104`, seed 1234 = `config.seed` default, `ple_layer_index = 0`); recomputed here
with the HF formulas:

```
m = [23703573157769, 20109073645365, 8052911324071]          # int64, each < (2^63-1)/248320, so no int64 overflow
P = [20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
     20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171]   # head h: (h+1)-th prime after 19999999
O = running sums of P, starting 0: [0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
     160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275]
total = 320001446, padded to 320001536 (multiple of 128) = 128 x 2500012
```

Derivation of `m`: `splitmix64` with gamma `0x9E3779B97F4A7C15`, M1 `0xBF58476D1CE4E5B9`, M2 `0x94D049BB133111EB`.
Then `m_i = 2 * (splitmix64((1234 + 10007*0 + gamma*(i+1)) mod 2^64) mod (((2^63-1)//248320)//2)) + 1`
(`HF:1025-1048`).

History:

```
hist = [ctx0, ctx1] + ids          # ctx = the 2 previous token ids; at sequence start ctx = [EOS, EOS], EOS = 248044
```

`HF:1126-1147`; `X/modules/ple.py:447-450, 452-472`.

For each position `p` of `hist` with `p >= 2`, and shift `k`:

```
tok(p, 0) = hist[p]
tok(p, k) = EOS            if p-k < 0, or any hist[q] == EOS for q in [p-k, p-1]
          = hist[p-k]      otherwise                              # k = 1, 2
mix2 = tok(p,0)*m0  XOR  tok(p,1)*m1                              # int64 multiply, no overflow
mix3 = mix2  XOR  tok(p,2)*m2
row[h] = (h < 8 ? mix2 : mix3) mod P[h] + O[h]                     # h = 0..15; values are >= 0, so C % == torch.remainder
```

- `_shift_right_ignore_eos`: `HF:1106-1120`, `X/modules/ngram_embedding.py:241-257`, C++ `X/exllamav3_ext/ngram.cu:95-123`.
  The condition "no EOS in `[p-k+1, p-1]`" gives the same values, because `hist[p-k] == EOS` already yields EOS.
- An n-gram never reaches back across an EOS. The EOS token itself still hashes its own preceding tokens.
- Only 248044 (`<|endoftext|>`) resets, not `<|im_end|>` 248046 (`X/architecture/qwen4_exp.py:80` reads
  `text_config.eos_token_id`; `HF:1085` takes element 0 if it is a list). In a normal chat the n-grams therefore run
  straight across turn boundaries.
- Overflow: X's C++ multiplies as uint64 and fixes the sign after `%` (`ngram.cu:105-115`); torch uses int64
  remainder; `ngram_hash_ref.py` uses bigints. All three agree because the products are below 2^63
  (`~/b70/llm/exl3conv/tests/ngram_hash_ref.py:39-41`). **int32 is not enough**: each multiplier is about
  2^45.
- Padding positions (batched HF only) are replaced by EOS before hashing (`HF:1462-1467`).
- Multimodal: X replaces embedding-alias ids (`>= FIRST_MM_EMBEDDING_INDEX`) with `image_token_id` 248056 before
  hashing (`X/modules/ple.py:436-445`). HF hashes the literal placeholder ids. This does not apply to text-only input.
- Decode: `hist = id_state[2] + [new_id]`, giving 16 rows. Then `id_state <- hist[-2:]`.

### 5.2 Rows read per token

- Each token reads **16 rows** (`(ngram_size-1) * heads_per_ngram`), one per head.
- `split_ngram_parts` only shards the table (128 x 2500012 rows): global row `r` is `shard_{r // 2500012}`, row
  `r % 2500012` (`N` §1.5).
- The 16 rows of 160 values are concatenated in head order, 0..15 (`HF:1165-1171`; `X/modules/ngram_embedding.py:279-287`).
  `emb[t] = [row_0 | row_1 | ... | row_15]`, `[2560]`.
- X dedups the unique rows across the whole pass before the gather (`ngram.cu:125-141`). SGLang does not
  (fork report). Dedup changes speed only.

### 5.3 Table format (`ngram_embedding.safetensors`, "K = 5")

The table is a separate regular safetensors file. It is not in `model.safetensors.index.json`; X globs every
`*.safetensors` in the model directory (`N` §3.7). `P = model.language_model.layers.1.ple.ple_embedding.ngram_embedding`
(`QC`):

| Tensor | dtype, shape |
|---|---|
| `P.shard_0.trellis` .. `P.shard_127.trellis` | int16 `[2500012, 51]` each (255,001,224 B), data contiguous in shard order |
| `P.head_bias` | fp16 `[16, 160]` |
| `P.head_offsets`, `P.head_vocab_sizes` | int64 `[16]` |
| `P.layer_multipliers` | int64 `[3]` |
| `__metadata__` | `format exl3_ngram_trellis`, `K`, `codebook mul1`, `row_dim 160`, `rows 320001536`, `shard_rows 2500012` (`N` §3.7) |

Size: 128 x 255,001,224 = **32.64 GB**.

`K = (row_words - 1) * 16 / 160` (`X/modules/ngram_embedding.py:160`): 51 words gives K = 5, 102 B/row. (The
Dongfangshuo pack's 19.84 GB table is 62 B/row, K = 3.)

Row decode (`X/modules/quant/exl3_lib/ngram_codec.py:1-20, 66-116`; CUDA `X/exllamav3_ext/ngram.cu:321-391`). One
row = one tail-biting ring of 160 positions:

```
w[0..50]  = the row as uint16 (little-endian)
scale     = fp16 bits w[0]
bit(b)    = (w[1 + (b >> 4)] >> (b & 15)) & 1            # ring stream, LSB first within each word, b in [0, 160*K)
for i in 0..159:
    state = 0
    for mm in 0..15:                                     # bit mm of the 16-bit state = bit (mm % K) of the symbol at ring pos i - mm/K
        pos = (i - mm / K) mod 160                       # integer division
        state |= bit(pos*K + mm % K) << mm
    prod  = uint32(state * 0x83DCD12D)                   # mul1 codebook
    hsum  = 1024 + (prod & 255) + ((prod>>8) & 255) + ((prod>>16) & 255) + (prod>>24)
    c     = fp16_round(hsum * k_inv + k_bias)            # k_inv = fp16 0x1eee = 0.00676727294921875,
                                                         # k_bias = fp16 0xc931 = -10.3828125, math in fp32
    row[i] = fp16_round(float(c) * float(scale) + float(head_bias[head][i]))
```

- `head` = the hash head the row belongs to. For hashed lookups that is just `h`. X recomputes it as
  `upper_bound(offsets, r) - 1`, so padding rows count as head 15 (`ngram.cu:135-137`). SGLang uses `r % 16` on its
  head-major id layout (fork report); the result is the same.
- This bit layout is **not** the EXL3 linear tile layout: the linear tiles are MSB-first in 32-bit words with a
  tensor-core permutation (`N` §3.7; `~/b70/llm/exl3conv/notes/exl3-algorithm.md` §5).
- Storage modes in X (`X/modules/row_table.py:8-23`, `X/exllamav3_ext/ngram.cu:144-317`): stream from disk with
  threaded `pread` (default), or hold the table in RAM. SGLang/trellis puts the whole 32.6 GB in USM host memory and
  gathers zero-copy over PCIe (`TS/ngram_host.py:78-131`), or uses NVMe plus an 8 GB row cache
  (`TS/ngram_nvme.py`). Only the 16 rows per token are ever needed.

### 5.4 PLE layer math

Tensors (`model.language_model.layers.1.ple.`, `QC`):

| Tensor | Shape | dtype |
|---|---|---|
| `key_proj.weight` (`Wk`) | `[10240, 2560]` | fp16, **not quantized** |
| `value_proj.weight` (`Wv`) | `[2560, 2560]` | fp16, not quantized |
| `norm_key.weight`, `norm_query.weight`, `norm_conv.weight` | `[10240]` | bf16 (GNorm1p) |
| `conv1d.weight` (`wc`) | `[10240, 1, 4]` | bf16 in the source; X loads it as fp16 (`X/modules/ple.py:238-239`) |

Per token `t`, with `E = emb[t]` `[2560]` and `X` the stream stack **entering layer 1** (after layer 0's two applies):

```
Kn  = GNorm1p((E @ Wk^T).reshape(4, D), norm_key.w)          # [4, D]
V   = E @ Wv^T                                                # [D], shared by all streams
Qn  = GNorm1p(X, norm_query.w)                                # [4, D], the RAW streams
g[s]  = dot(Kn[s], Qn[s]) / sqrt(2560)                        # [4]
g[s]  = sign(g) * sqrt(max(|g|, 1e-6))                        # signed sqrt, sign(0) = 0
Gv[s] = sigmoid(g[s]) * V                                     # [4, D]
Cn    = GNorm1p(Gv, norm_conv.w).flatten()                    # [10240] conv channels, channel c = s*D + d
conv[t, c] = sum_{j=0..3} wc[c, 0, j] * Cn[t - 3*(3-j), c]    # depthwise, dilation 3, causal; taps t-9, t-6, t-3, t
delta = Gv.flatten() + silu(conv[t])                          # [10240]
X     = X + delta.reshape(4, D)
```

- `HF:1226-1246` and `_short_conv` `HF:1207-1224`. The conv has no bias.
- `X/modules/ple.py:418-434` (reference) and `X/exllamav3_ext/ple.cu:16-55, 125-206` (fused):
  `sqrtf(fmaxf(fabsf(g), 1e-6f))`, `sign(0) = 0`, `gate_scale = 1/sqrt(hidden_size)` (`ple.py:162`).
- The conv input is the **normed** gated value. The residual term `Gv` is **un-normed** (`HF:1240-1245`).
- Conv state: the trailing 9 columns, `(kernel - 1) * dilation`, of `Cn`. That is `[10240, 9]` per sequence, zero
  at sequence start (`HF:1192, 1211-1218`; `X/modules/ple.py:160-161, 348-362`). Stored as fp16 in X
  (`ple.py:54-58`) and bf16 in SGLang.
- Prefill is the same math over `T` columns `[state | Cn^T]`. X runs it in row slabs of 1024 with the conv state
  carried between them (`ple.py:395-415`).
- Decode: 1 column, then the state shifts by one.
- X state per slot (`ple.py:34-111`): `conv_state [bsz, 10240, 9 + max_history]` fp16 and
  `id_state [bsz, 2 + max_history]` int64 on the CPU, EOS-filled. Hashing runs on the host.

---

## 6. Gated DeltaNet layer (36 layers)

### 6.1 Tensors (`model.language_model.layers.L.linear_attn.`; `QC`, `PK/model.safetensors.index.json`)

| Tensor | Shape | Storage |
|---|---|---|
| `in_proj_qkv` | in 2560, out 10240 | EXL3 K=5 (`trellis [160, 640, 80]`) |
| `in_proj_z` | in 2560, out 6144 | EXL3 K=5 |
| `in_proj_b.weight`, `in_proj_a.weight` | `[48, 2560]` | fp16 |
| `conv1d.weight` | `[10240, 1, 4]` | bf16 (`N` §1.3) |
| `A_log`, `dt_bias` | `[48]` | bf16 |
| `norm.weight` | `[128]` | bf16 |
| `out_proj` | in 6144, out 2560 | EXL3 K=5 |

### 6.2 Math (per token; `HF:503-616`)

```
qkv = y @ Wqkv^T                    # [10240] = [q 0:2048 | k 2048:4096 | v 4096:10240], plain contiguous heads of 128
z   = (y @ Wz^T).reshape(48, 128)
b   = y @ Wb^T;  a = y @ Wa^T       # [48] each
c   = silu(causal_conv1d(qkv))      # depthwise, kernel 4: c[t,ch] = silu(sum_{j=0..3} w[ch,0,j] * qkv[t-3+j, ch]); zero before t=0
q = c[0:2048].reshape(16,128); k = c[2048:4096].reshape(16,128); v = c[4096:].reshape(48,128)
beta = sigmoid(b)                                   # [48]
g    = -exp(A_log) * softplus(a + dt_bias)          # [48], fp32; softplus(x) = x if x > 20 else log1p(exp(x))
v-head j uses k-head j // 3                         # repeat_interleave, NOT tile (HF:573-575)
qh = l2norm(q[j//3]) / sqrt(128);  kh = l2norm(k[j//3])           # fp32
S_j [128 (k), 128 (v)] fp32, 0 at sequence start:
    S_j   = exp(g[j]) * S_j
    kv    = S_j^T kh                                # kv[v] = sum_i S_j[i, v] * kh[i]
    S_j  += outer(kh, beta[j] * (v[j] - kv))
    o[j]  = S_j^T qh                                # [128]
o[j] = GatedNorm(o[j], z[j], norm.w)                # rms over 128, * w (no +1), * sigmoid(z)
out  = o.flatten() @ Wout^T                         # [2560]
```

- Recurrence: `HF:433-444`. Gate: `HF:570-572`; X `X/exllamav3_ext/gdn.cu:47-51, 276-280`.
- X computes beta and `g` in fp32 and stores beta as bf16 with round-to-nearest-even (`trunc_bf16` is
  `__float2bfloat16_rn`, `gdn.cu:27-30`). The conv input is cast fp16 to bf16 (`X/modules/gated_delta_net.py:1356`;
  qkv projection output is fp16 by default, `:26`).
- GatedNorm: HF rounds `rms(o)` to the input dtype before `* w` (`HF:184`). X's kernel stays in fp32 and writes fp16
  (`X/exllamav3_ext/norm.cu:556-599`, sigmoid path `gate_act = 1`, `X/architecture/qwen4_exp.py:129-133`). In X the
  gate `z` is fp32 (`gated_delta_net.py:1348`).

### 6.3 Prefill vs decode

- **Decode (1 token).**
  - The conv reads the previous 3 qkv columns from the conv state, then shifts in the new column.
  - The recurrent update runs once per v-head: 48 x 128 x 128 fp32 read and written.
  - X uses its fused recurrent CUDA kernel when `seqlen < num_v_heads` (48), and the chunked kernel otherwise
    (`X/modules/gated_delta_net_fn/gated_delta_rule.py:208, 232`).
- **Prefill.** Chunked form with chunk size C = 64 (`HF:254-387`; fla `chunk_gated_delta_rule`). X runs it in
  sub-ranges of 2048 tokens with the state carried between them, which gives bit-identical results
  (`gated_delta_rule.py:89-118`). Per chunk, with `G = cumsum(g)` inside the chunk and q already scaled by 128^-0.5:

```
A[i,j] = beta[i] (k_i . k_j) exp(G_i - G_j)   for j < i  (mask -inf above the diagonal BEFORE exp)
U  = (I + A)^-1 (beta * V);   W = (I + A)^-1 (beta * exp(G) * K)       # unit lower-triangular solve (HF:337-349)
Vn = U - W S;   O = (exp(G) * Q) S + P Vn,  P[i,j] = (q_i . k_j) exp(G_i - G_j) for j <= i (diagonal included)
S  = exp(G_last) S + (exp(G_last - G) * K)^T Vn
```

HF builds the system matrix with the diagonal included but solves it `unitriangular=True`, so the diagonal is ignored
(`HF:348-349`).

State per sequence (X `GDNLayerState`, `X/modules/gated_delta_net.py:259-303`):
- conv `[10240, 4 + max_history]` bf16 (only the last 3 columns matter).
- recurrent `[48, 128, 128]` fp32, i.e. 3 MiB per layer and **113 MB for 36 layers**.
- X keeps 2 recurrent rows per slot for speculative rollback.

---

## 7. Full-attention layer with QSA (12 layers)

### 7.1 Projections, gate layout, norms

Tensors (`...layers.L.self_attn.`):

| Tensor | Shape (in, out) | Storage |
|---|---|---|
| `q_proj` | 2560, 12288 | EXL3 K=5 (`trellis [160, 768, 80]`) |
| `k_proj`, `v_proj` | 2560, 512 | EXL3 K=5 |
| `o_proj` | 6144, 2560 | EXL3 K=5 |
| `q_norm.weight`, `k_norm.weight` | `[256]` | bf16 (Norm1p) |
| `indexer.index_qk_proj` | 2560, 640 | **EXL3 K=3** (`trellis [160, 40, 48]`) |
| `indexer.q_layernorm.weight`, `indexer.k_layernorm.weight` | `[128]` | bf16 (Norm1p) |

```
qg = (y @ Wq^T).reshape(24, 512)              # per head h: qg[h, 0:256] = query, qg[h, 256:512] = gate  (interleaved per head)
q  = qg[:, :256];  gate = qg[:, 256:].flatten()     # gate [6144] in head order
q  = Norm1p(q[h], q_norm.w)   per head (256)
k  = Norm1p((y @ Wk^T).reshape(2, 256)[h], k_norm.w)
v  = (y @ Wv^T).reshape(2, 256)               # no v norm
q, k = rope(q, pos), rope(k, pos)             # §7.2, after the norm
o  = attention(q, k, v)                        # q head h uses kv head h // 12; scale 1/16; causal + QSA selection (§7.4)
o  = o.flatten() * sigmoid(gate)               # [6144], elementwise
out = o @ Wo^T
```

- Interleave: `HF:858-861` (`view(..., -1, 512)` then `chunk(2)`); X `ext.deinterleave_qg`
  (`X/modules/attn.py:650-658, 788-791`); sigmoid in `ext.mul_sigmoid_` (`:953-954, 1219-1220`). SGLang does the same
  (`SG/models/qwen3_5.py:1332-1333`).
- Norms: `HF:863-864`; X fuses norm and rope in one kernel with `w + constant_bias` (`X/modules/attn.py:908-920`,
  `rope.cu:233-257`).
- KV cache stores `k` after norm and rope, and raw `v`.

### 7.2 RoPE (text-only)

```
inv_freq[j] = 1e7 ^ (-(2j)/64),  j = 0..31            # fp32; denominator = rotary dim 64 (HF:107-115)
for j in 0..31:  th = pos * inv_freq[j];  a = x[j]; b = x[j+32]
    x[j]    = a*cos(th) - b*sin(th)
    x[j+32] = b*cos(th) + a*sin(th)                   # NeoX rotate-half within the first 64 dims
dims 64.. untouched (256-dim attention heads: dims 64..255; 128-dim indexer heads: dims 64..127)
```

- `HF:619-661`. The cos/sin width is 64 for both the attention heads and the indexer heads (`HF:705, 737-741`), so
  the indexer also rotates only its first 64 of 128 dims.
- M-RoPE recomposition (`HF:131-140`): for text the T, H and W positions are equal, so it is the identity.
- `pos` = absolute token index: past length + offset (`HF:1422-1425`). `attention_scaling = 1`.

### 7.3 QSA indexer projections

```
qk     = y @ Widx^T                          # [640] = [4 q heads x 128 | 1 raw key x 128]
qi[h]  = rope(Norm1p(qk[h*128:(h+1)*128], q_layernorm.w), pos)     # h = 0..3
rk     = qk[512:640]                         # RAW indexer key: unnormed, unroped. This is what gets cached
```

`HF:697-708`; X `X/modules/qsa_indexer.py:132-144`. The indexer key is separate from the attention keys: one 128-dim
head per token.

Block compression (ratio 4). Blocks are aligned at absolute positions `4b .. 4b+3`, and only complete blocks exist:

```
pooled[b] = rope( Norm1p( bf16/fp16_round( mean_fp32(rk[4b], rk[4b+1], rk[4b+2], rk[4b+3]) ), k_layernorm.w ), pos = 4b )
```

- `HF:728-741` (fp32 mean, cast to the raw dtype, norm, rope at the block's **start** position).
- X: `X/modules/qsa_indexer.py:146-162` (non-cached) and `update_planes`, `:483-540` (cached). The paged pool kernel
  also writes the trailing incomplete pool; it is never selected and is rebuilt once the block completes.

### 7.4 Selection and masked attention

For a query at absolute position `p`, the visible tokens are `0..p`:

```
nb       = (p + 1) // 4                                   # complete visible blocks
score[b] = sum_{h=0..3} relu(qi[h] . pooled[b]) / sqrt(128)      # b = 0..nb-1; HF fp32
sel      = top min(512, nb) blocks by score
tokens   = { 4b .. 4b+3 : b in sel }  U  { 4nb .. p }       # tail = the incomplete current block (0..3 tokens)
attention for query p runs over exactly these tokens (all 24 heads share one selection)
```

- `HF:720-770` (the selection becomes a boolean mask ANDed with the causal mask, `HF:846-851`).
- X reference: `token_mask`, `X/modules/qsa_indexer.py:204-238`; index lists `:388-425`.
- No sink or local-window blocks are forced. Only the tail is always kept.
- When `(p + 1) % 4 == 0` the tail is empty. The query's own block is then just one candidate, so **the query can
  fail to attend to itself** in the sparse regime. HF, X and SGLang all do this (fork report; `HF:748-754`).
- Ties: X's `dsa_topk` orders by score descending, then index ascending (`qsa_indexer.py:283-284`). `torch.topk` in
  HF is unspecified on ties.

Dense equivalence:
- When `nb <= 512`, every complete block is kept. Together with the tail, the mask is then plain causal.
- That holds for every query with `p <= 2050`. X skips the indexer selection while
  `max_pos + 1 <= sparse_threshold() = 4*512 + 3 = 2051` (`X/modules/qsa_indexer.py:469-471`;
  `X/modules/attn.py:894, 1127`).
- In the cached path X **always** maintains the raw/pooled planes (`attn.py:1114-1118, 1140-1141`). A port must keep
  writing them during dense-regime prefill and decode, or the pools will be missing once the context passes 2051.

Prefill (multi-row):
- Each row `p` has its own `nb(p)`. Blocks with `b >= nb(p)` get score `-inf` (`qsa_indexer.py:225, 410-411`).
- All pools completed within the chunk must exist before selection.
- Selected tokens are limited to `<= p`.

Decode:
- Score one query against every complete pool: `L/4` pools of 128 values.
- Take the top 512 and gather at most 512*4 + 3 = 2051 KV rows per kv head.
- X's width is `K_pad = ceil32(2051) = 2080` (`:265-267`).

Precision note:
- X scores in **fp16**: `dsa_indexer_scores` returns an fp16 `(R, T)` slab (`X/modules/attention_fn/dsa_triton.py:1181-1198`)
  with fp16 pooled and raw planes (`X/cache/qsa.py:11-37`). HF scores in fp32.
- X's own test accepts a one-pool selection difference at fp16 ties (`X/../tests/parity/test_qwen4_qsa_prefill.py`
  header). Use fp32 scores where it is cheap.

State per attention layer:
- K/V cache.
- raw-key plane `[tokens, 128]` fp16.
- pooled plane `[tokens/4, 128]` fp16 (`X/cache/qsa.py:21-37`, paged; `PAGE_SIZE % 4 == 0`, so blocks never straddle
  pages).

---

## 8. MoE block (every layer)

Tensors (`...layers.L.mlp.`):

| Tensor | Shape | Storage |
|---|---|---|
| `gate.weight` (router `Wr`) | `[512, 2560]` | fp16 |
| `experts.E.{gate_proj, up_proj}` | in 2560, out 640 | EXL3 K=3 (`trellis [160, 40, 48]`), E = 0..511 |
| `experts.E.down_proj` | in 640, out 2560 | EXL3 K=3 (`trellis [40, 160, 48]`) |
| `shared_expert.{gate_proj, up_proj}` | in 2560, out 640 | EXL3 K=5 |
| `shared_expert.down_proj` | in 640, out 2560 | EXL3 K=5 |
| `shared_expert_gate.weight` | `[1, 2560]` | fp16 |

```
logits = y @ Wr^T                          # [512]
p      = softmax(logits)  (fp32)
top    = indices of the 10 largest p
w_e    = p[e] / sum_{top} p                # norm_topk_prob = true; no routed scaling factor
routed = sum_{e in top} w_e * ( (silu(y @ Wg_e^T) * (y @ Wu_e^T)) @ Wd_e^T )
shared = sigmoid(y @ Wsg^T) * ( (silu(y @ Wsg_g^T) * (y @ Wsg_u^T)) @ Wsg_d^T )     # sigmoid of a scalar
out    = routed + shared
```

- `HF:951-991`. The renormalised top-k of the full softmax equals the softmax over the 10 selected logits.
- X computes the latter: fp16 logits, top-k on the logits, `exp(l - max)` in fp32, `/ (sum + 1e-20)`, fp16 weights
  (`X/exllamav3_ext/routing.cu:607-700`; `X/modules/block_sparse_mlp_routing.py:94-157`).
- X fork-only: on CUDA the router may use int8 tables for multi-row calls (`gate_i8`, `routing.cu:321-342`).
- Shared gate: `ext.add_sigmoid_gate` (`X/modules/block_sparse_mlp.py:1929-1934`).
- Expert intermediate: fp16 (`interm_dtype`, `X/architecture/qwen4_exp.py:202, 214`).
- HF's fused tensor `experts.gate_up_proj [512, 1280, 2560]` is split as rows 0..639 = gate, 640..1279 = up
  (`HF:942`). The pack already stores split per-expert EXL3 linears.

---

## 9. Embedding, lm_head, tokens

- `model.language_model.embed_tokens.weight`: `[248320, 2560]` bf16, plain gather. X outputs fp32
  (`X/modules/embedding.py:80`). There is no embedding multiplier.
- `lm_head`: EXL3 K=5, in 2560, out 248320 (`trellis [160, 15520, 80]`, `svh [248320]`, 397 MB). Its input is the
  fp16 mixer output.
- EXL3 linear (all quantized linears): tensors `P.trellis` int16 `[in/16, out/16, 16K]`, `P.suh` fp16 `[in]`,
  `P.svh` fp16 `[out]`, `P.mul1` int32 `[]` (mul1 codebook marker, multiplier 2212286765 = 0x83DCD12D). No `.mcg`,
  no `.bias` in this pack.
  - Math: `y = had128(had128(x * suh) @ W_hat) * svh`.
  - Decode and tile layout: `~/b70/llm/exl3conv/notes/exl3-algorithm.md` §5. `src/exl3.cpp` in this
    engine already implements it.
- Special tokens (`~/models/llm/Dongfangshuo-Qwen3.8-Flash-Next-exl3/tokenizer.json`):

  | Id | Token |
  |---|---|
  | 248044 | `<|endoftext|>` |
  | 248045 | `<|im_start|>` |
  | 248046 | `<|im_end|>` |
  | 248053 / 248054 | `<|vision_start|>` / `<|vision_end|>` |
  | 248056 | `<|image_pad|>` |
  | 248057 | `<|video_pad|>` |
  | 248058 / 248059 | `<tool_call>` / `</tool_call>` |
  | 248068 / 248069 | `<think>` / `</think>` |

- Generation stop ids: `[248046, 248044]`; bos and pad 248044 (`PK/generation_config.json`). The tokenizer's
  `eos_token` is `<|im_end|>`.
- Chat template: `PK/chat_template.jinja`. It has a `reasoning_effort` of xhigh (default), medium or low, written into
  the system prompt (`:47-56`), plus `enable_thinking` / `preserve_thinking`. X's fallback prompt is plain ChatML
  (`X/architecture/qwen4_exp.py:327-334`).

---

## 10. MTP head (brief)

Tensors (`PK/model.safetensors.index.json`, 6203 `mtp.*` entries; `QC`'s storage map omits them):
- `mtp.pre_fc_norm_hidden.weight` `[10240]`, `mtp.pre_fc_norm_embedding.weight` `[2560]`.
- `mtp.fc_hidden`, `mtp.fc_embedding`: EXL3, 2560 to 2560.
- One full-attention block `mtp.layers.0.*` with exactly the tensors of §4.2, §7.1 and §8 (indexer included, no PLE).
- `mtp.hyper_connection_mixer.{hc_norm, input_mix_weight_down, input_mix_weight_up}`.
- It shares the trunk's `embed_tokens` and `lm_head`. HF ignores `mtp.*` (`HF:1313`), so there is **no reference**.

X (`X/architecture/qwen4_exp_mtp.py:64-115`, `X/modules/arch_specific/qwen4_exp_mtp.py:137-168`). For draft
position t, with `stack` = the trunk's pre-mixer stream stack at t and `tok` = token t+1:

```
h_s     = fc_hidden( rms(stack[s]) * (1 + w_hidden[s*D:(s+1)*D]) )        # per stream (X: grouped)
e       = fc_embedding( Norm1p(E[tok], pre_fc_norm_embedding.w) )
X0[s]   = h_s + e                                                         # stream_tap = True
X1      = qwen4 block (attn site, QSA attention, mlp site, MoE) on X0
draft   = argmax lm_head( MIX_FINAL(X1, mtp.hyper_connection_mixer) );  next-step stack = X1
```

Disagreement:
- X's per-stream grouping of `pre_fc_norm_hidden` is marked "a semantic guess" (`qwen4_exp_mtp.py:10-24`).
- SGLang applies **one** Norm1p over all 10240 values (`SG/models/qwen4_exp_mtp.py:74-83, 105-115`).
- SGLang also reuses the draft-extend's QSA selection on MTP decode steps instead of re-running the indexer
  (`SG/models/qwen4_exp.py:1508-1514`). That is an approximation.
- Only acceptance rate can decide between the two.

---

## 11. State summary (per sequence)

| State | Where | Shape / dtype | Size |
|---|---|---|---|
| GDN recurrent | 36 GDN layers | `[48, 128 (k), 128 (v)]` fp32 | 3 MiB/layer, 113 MB total |
| GDN conv | 36 GDN layers | last 3 qkv columns `[10240, 3]` bf16 | 61 KB/layer |
| PLE conv | layer 1 PLE | last 9 normed columns `[10240, 9]` | 184 KB fp16 |
| PLE id context | layer 1 PLE | 2 previous token ids, EOS-filled at start | int64 x 2 |
| KV cache | 12 attention layers | k (normed, roped) and v, `[tokens, 2, 256]` each | 24 KiB/token fp16, all layers |
| QSA raw keys | 12 attention layers | `[tokens, 128]` fp16, unnormed, unroped | 3 KiB/token, all layers |
| QSA pooled keys | 12 attention layers | `[tokens/4, 128]` fp16, normed, roped at 4b | 768 B/token, all layers |

---

## 12. Tensor names as exllamav3 loads them (the Swift 3.05 pack)

EXL3 linears appear as `<name>.trellis`, `.suh`, `.svh`, `.mul1`, with `K = trellis.shape[-1] / 16`. The listing
below comes from `QC` and `PK/model.safetensors.index.json`. `M.` = `model.language_model.`

**Top level**
```
M.embed_tokens.weight                                   bf16 [248320, 2560]
M.hyper_connection_mixer.hc_norm.weight                 fp16 [10240]
M.hyper_connection_mixer.input_mix_weight_down.weight   fp16 [320, 10240]
M.hyper_connection_mixer.input_mix_weight_up.weight     fp16 [10240, 320]
lm_head.{trellis [160,15520,80] i16, suh [2560], svh [248320], mul1}       K=5
```

**GDN layer (L = 0)**
```
M.layers.0.attn_hyper_connection.{hc_norm.weight [10240], input_mix_weight_down.weight [320,10240],
    input_mix_weight_up.weight [10240,320], block_inject_weight.weight [4,10240]}             fp16
M.layers.0.linear_attn.in_proj_qkv.{trellis [160,640,80], suh [2560], svh [10240], mul1}      K=5
M.layers.0.linear_attn.in_proj_z.{trellis [160,384,80], suh [2560], svh [6144], mul1}         K=5
M.layers.0.linear_attn.in_proj_b.weight [48,2560], in_proj_a.weight [48,2560]                 fp16
M.layers.0.linear_attn.conv1d.weight [10240,1,4], A_log [48], dt_bias [48]                    (bf16 per N §1.3)
M.layers.0.linear_attn.norm.weight [128]                                                      bf16
M.layers.0.linear_attn.out_proj.{trellis [384,160,80], suh [6144], svh [2560], mul1}          K=5
M.layers.0.mlp_hyper_connection.{same four as attn_hyper_connection}                          fp16
M.layers.0.mlp.gate.weight [512,2560], mlp.shared_expert_gate.weight [1,2560]                 fp16
M.layers.0.mlp.experts.E.{gate_proj,up_proj}.{trellis [160,40,48], suh [2560], svh [640], mul1}   K=3, E=0..511
M.layers.0.mlp.experts.E.down_proj.{trellis [40,160,48], suh [640], svh [2560], mul1}             K=3
M.layers.0.mlp.shared_expert.{gate_proj,up_proj}.{trellis [160,40,80], ...}                   K=5
M.layers.0.mlp.shared_expert.down_proj.{trellis [40,160,80], ...}                             K=5
```

**Attention layer (L = 3)**: hyper-connection sites and MoE are the same as above, plus
```
M.layers.3.self_attn.q_proj.{trellis [160,768,80], suh [2560], svh [12288], mul1}             K=5
M.layers.3.self_attn.k_proj.{trellis [160,32,80], svh [512], ...}, v_proj same                K=5
M.layers.3.self_attn.o_proj.{trellis [384,160,80], suh [6144], svh [2560], mul1}              K=5
M.layers.3.self_attn.q_norm.weight [256], k_norm.weight [256]                                 bf16
M.layers.3.self_attn.indexer.index_qk_proj.{trellis [160,40,48], suh [2560], svh [640], mul1} K=3
M.layers.3.self_attn.indexer.q_layernorm.weight [128], k_layernorm.weight [128]               bf16
```

**Layer 1 PLE** (layer 1 itself is a GDN layer with the tensors above)
```
M.layers.1.ple.key_proj.weight [10240,2560], value_proj.weight [2560,2560]                    fp16 (unquantized)
M.layers.1.ple.norm_key.weight, norm_query.weight, norm_conv.weight [10240]                   bf16
M.layers.1.ple.conv1d.weight [10240,1,4]
M.layers.1.ple.ple_embedding.ngram_embedding.shard_{0..127}.trellis [2500012,51] i16          (ngram_embedding.safetensors)
M.layers.1.ple.ple_embedding.ngram_embedding.{head_bias [16,160] fp16, head_offsets [16] i64,
    head_vocab_sizes [16] i64, layer_multipliers [3] i64}
```

There is no PLE at any other layer (`ple_layer_ids [2]`; the only `ple` prefix in the index is `layers.1`). HF source
names differ for the hash buffers (`ple_embedding.ngram_heads_offsets` etc., `X/modules/ngram_embedding.py:137-146`);
the converter renames them.

---

## 13. Disagreements and ambiguities

1. **Stream dtype.**
   - X: fp32 streams.
   - HF and SGLang: bf16 throughout (`SG/layers/hyperconnection.py:257, 277, 326`).
   - Both reproduce HF within its own bf16 noise (X parity tests: `X/../tests/parity/test_qwen4_exp_parity.py` header).
2. **GatedResidual decode tables.**
   - X fork: int8 tables on CUDA by default (`EXL3_GR_INT8`).
   - HF and SGLang: no int8. This is a lossy optimisation, not semantics.
3. **QSA score precision.**
   - X: fp16 kernel scores.
   - HF: fp32 scores.
   - SGLang XPU: bf16 q·k inside the sparse attention, bf16 probabilities (`TS/qsa_xpu.py:54-61, 170`), on an fp8 KV
     cache.
4. **PLE/n-gram dtype.**
   - X: fp32 gate math, fp16 rows and conv state.
   - SGLang: everything in bf16. Rows are rounded to fp16 first, then stored as bf16.
5. **MTP `pre_fc_norm_hidden`.**
   - X: grouped per stream.
   - SGLang: one norm over 10240.
   - No reference; decide by acceptance rate. In SGLang, MTP decode also reuses the QSA selection.
6. **GDN gated norm.** HF rounds `rms(o)` to bf16 before `* w`; X does not. The effect is tiny.
7. **No-self-attend case.** In the sparse regime with `(p+1) % 4 == 0`, the query's own block may be dropped. All
   three implementations do this, so it is intended model behaviour, not a bug to fix.
8. **Swift pack storage map.** `QC` lists no `mtp.*` tensors, but `PK/model.safetensors.index.json` has 6203 of them.
   Trust the index.
9. **`norm_topk_prob`.** The config does not set it. HF reads it from its config class default; X asserts true and
   SGLang defaults to true (`N` §0).

---

## 14. Costs that matter for speed (decode, 1 token, bytes from `QC`)

| Item | Per token | Note |
|---|---|---|
| GatedResidual tables, 96 sites + mixer | **1.28 GB** | fp16 `Wd + Wu + Wi`, 13.2 MB per site; 28% of all weight bytes. int8 tables would halve it |
| GDN projections, 36 layers | 1.32 GB | qkv 16.4 + z 9.8 + out 9.8 + a/b 0.5 MB per layer |
| Routed experts, 48 layers x 10 | 0.88 GB | 3 x 614 KB per expert (K=3) |
| lm_head | 0.40 GB | K=5 |
| Attention, 12 layers | 0.38 GB | q 19.7 MB is the largest |
| Shared experts | 0.15 GB | |
| Routers, fp16 | 0.13 GB | |
| PLE key/value projections, fp16 | 66 MB | unquantized |
| **Total weights** | **~4.6 GB/token** | |
| GDN recurrent state | 113 MB read + 113 MB written | fp32, per sequence |
| n-gram rows | **1632 B** | 16 x 102 B, host RAM or disk. From disk it is 16 random reads, i.e. 16 x 4 KiB pages cold |
| QSA, context `L > 2051` | `12 x (64 B x L)` pool reads + up to 12 x 2051 x 2 x 256 x 2 x 2 B = 50 MB K/V | at L = 128K the pool scan alone is about 100 MB/token |
| fp32 stream stack | 40 KB per site read, 40 KB written | negligible in decode; in prefill about 97 x 120 KB = 11.6 MB/token of activation traffic |

In prefill the sizes that grow with `T` are:
- the GatedResidual `[T, 10240] x [10240, 320]` GEMMs (two per site),
- the chunked GDN scan,
- the PLE fp16 GEMMs `[T, 2560] x [2560, 12800]`,
- the per-row QSA selection. It is skipped entirely while `T <= 2051` from position 0.
