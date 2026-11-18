# Edge0-145B on glm.cpp: format and forward spec

Edge0-145B (huggingface.co/Edge0/Edge0-145B-NVFP4, Nov 2026) is Qwen3.8-Flash-Next with every layer's routed MoE
replaced by a **latent MoE** and post-trained. The trunk (hyper-connections, Gated DeltaNet, QSA attention, the n-gram
PLE, the mixer, lm_head) is the `qwen4exp` architecture `docs/qwen4-spec.md` describes; this page covers what differs.
Sources: the release folder itself, and the Edge0 source tree (`~/src/edge0`, github.com/Edge0-AI/Edge0), mainly
`edge0-strata/src/core/latent_dense.cpp`, `edge0-strata/src/kernels/latent_cpu_verify.cpp`,
`edge0-strata/src/kernels/cpu/native_expert.cpp`, `edge0-strata/docs/NVFP4.md`, and llama.cpp at Strata's pinned commit
`3cf03257` (`~/src/llama.cpp-3cf0325/conversion/qwen*.py`) for the GGUF conversion.

Code: `src/e0.*` (model), `src/e0_ops.*` (kernels), `src/gguf.*`, `tools/e0_gen.cpp`, `tests/e0_check.cpp`.

## 1. Release folder

| File | Content |
|---|---|
| `model-skeleton.gguf` (6.0 GB) | the trunk, llama.cpp `qwen4exp` names: Q8_0 linears (attention, GDN qkv/z/out, shared experts, lm_head, token embedding), BF16 small projections (hyper-connections, routers, GDN a/b, indexer, PLE key/value), F32 norms and GDN constants, F16 PLE conv |
| `latent_dense_q8.bin` (3.74 GB) | per layer: D then U, Q8_0 (34-byte blocks). D: 8 groups x 1792 rows x 2560 columns; U: 8 groups x 2560 rows x 1792 columns. `latent_dense_q4.bin` is the same in Q4_0 |
| `labels.bin` | int32 [48, 512]: each expert's D/U group (0..7; group sizes are uneven, 25..165 of 512) |
| `experts.bin` (47.6 GB) | [48][512] records of 1,935,376 B, layer-major |
| `ple-fp8.gguf` (51.2 GB) | `per_layer_token_embd.weight`, I8 [320,001,536 rows, 160], FP8 E4M3 bytes; `strata.ple.scale` = 0.000199 |
| `profile.bin` | `STRP`, u32 version 1, layers 48, experts 512, slots, n; then n (layer u16, expert u16) pairs, hottest first |
| `mtp/` | the MTP draft layer (Strata formats; Q2_0 experts); not used yet |
| `dense.bin`, `index.txt` | Strata's copy of the small tensors, plus the source model's (full-width) expert scales: unused |
| `tokenizer/` | llama.cpp's vocab dump (its `tokenizer.json` is a stub). The vocab and merges equal Qwen3.8-Flash-Next's, so the folder also holds Qwen's `tokenizer.json`, `config.json`, `chat_template.jinja` (Edge0's template differs only by `{% generation %}` markers) |

## 2. The GGUF conversion (undone or used as is at load)

From `conversion/qwen.py` (`Qwen3NextModel`, `_LinearAttentionVReorderBase`) and `conversion/qwen4exp.py`:

- Every `*norm.weight` except the GDN output norm (`linear_attn.norm`) carries its `+1` (hyper-connection norms, q/k
  norms, indexer and PLE norms). Checked on the file: these average 0.89..1.28 (the mixer's 3.7), `ssm_norm` 0.97.
- `ssm_a` = `-exp(A_log)`.
- GDN value heads are stored **tiled** (value slot j of key head k at `j * 16 + k`), HF and glm.cpp's kernels use
  **grouped** order (`k * 3 + j`). The loader permutes back: V rows of `attn_qkv`, rows of `attn_gate` (z), rows of
  `ssm_alpha` / `ssm_beta`, V channels of `ssm_conv1d`, `ssm_a`, `ssm_dt.bias`, and the 128-column head chunks of
  `ssm_out`. Q8_0 rows and 4-block column chunks move whole: nothing is requantized.
- `indexer.index_qk_proj` is split into `indexer.q_proj` (512 rows) and `indexer.k_proj` (128): stacked back.
- `ssm_conv1d` is F32 but bf16-exact (all 40,960 values of layer 0 have zero low bits); loaded as bf16 as for Swift.

## 3. Latent MoE (`latent_cpu_verify.cpp`)

For the MoE site's input `x` (fp16 [2560]), router logits `l = W_r x` (BF16 [512, 2560]), top-10 by logit, weights a
softmax over the ten (as Qwen3.8-Flash-Next):

```
xl_g  = D_g x                               g = 0..7, [1792] each (only the picked experts' groups are needed)
h_k   = silu(s_g * (G_k xl_{g_k})) * (s_u * (U_k xl_{g_k}))      g_k = label[e_k]; G_k, U_k NVFP4 [640, 1792]
yl_k  = s_d * (Down_k h_k)                  Down_k NVFP4 [1792, 640]
out   = sum_k w_k U_{g_k} yl_k  +  sigmoid(w_sg . x) * shared(x)
```

`U` is linear, so the engine sums the weighted `yl` per group first (`z_g = sum_{k: g_k = g} w_k yl_k`) and applies
`U_cat [2560, 8 * 1792]` once to the concatenated `z` (the loader lays U's eight [2560, 1792] blocks side by side).

### Expert record (1,935,376 B)

`gate` 640 rows x 1008 B, `up` 640 x 1008 B, `down` 1792 x 360 B, then 16 B `{s_gate, s_up, s_down, 0}` fp32. A row
is ggml `block_nvfp4`s: per 64 values, 4 UE4M3 scale bytes (one per 16) and 32 bytes of E2M1 codes; in a 16-value
sub-block, byte i's low nibble is value i and its high nibble value i + 8. Value = E2M1(code) x UE4M3(scale), E2M1 =
{0, 0.5, 1, 1.5, 2, 3, 4, 6}, sign bit 3 (ggml writes the same product as `kvalues = 2 x E2M1` times `0.5 x UE4M3`).
UE4M3: exponent bits 6..3 (bias 7), mantissa 2..0, 0 and 0x7F are zero. Every weight is exact in fp16.
The global scales are about 1e-4: **each multiplies its own projection's fp32 output** (folding `s_down` into the up
weights puts the hidden in fp16 subnormals; Strata measured 2..12% error from that, `NVFP4.md`).
`dense.bin`'s `ffn_{gate,up,down}_exps.scale` are **not** these tails: Strata packed `dense.bin` from the source
model's GGUF, and those scales (~7e-5) belong to its full-width experts; the latent experts' tails are ~3e-3 (gate,
up) and ~1e-4 (down). Strata's kernels read the tail (`iq_kernels.cu:2002`, `native_expert.cpp:482`).

## 4. Numerics in glm.cpp

| Part | Edge0's engine (Strata, CUDA) | glm.cpp |
|---|---|---|
| Routed experts, decode | W4A8 (activations q8_1) | fp32 activations, exact weights |
| Routed experts, prompt | W4A8 MMQ (W4A4 on sm_120a optional) | fp16 activations, fp16 exact weights, fp32 accumulation (oneMKL) |
| Q8_0 linears | q8_1 activations | decode: fp16/fp32 activations; prompt: dequantized to fp16, oneMKL |
| D / U | Q8_0 | Q8_0, D output fp32 (decode) / fp16 (prompt) |
| KV cache | int8 | int8, fp16 scale per 64 values (`E0_KV16=1`: fp16) |
| n-gram rows | FP8 x scale | FP8 x scale, fp16 rows |

## 5. Memory

Trunk on the card ~5 GB (Q8_0 kept as int8 + fp16 scales), D/U 3.74 GB, int8 KV 0.8 MB a token across the 12 QSA
layers; the experts fill the rest of the 32 GB less the reserve (hottest first by `profile.bin`), the others sit in
pinned host RAM and are read over PCIe in place. Decode-time adaptive swapping as for Swift (`E0_REBALANCE`).

## 6. Not done yet

The first-token-only shortcut for the last layer's MoE in prompts, a draft-vocabulary head for MTP, a cheaper NVFP4 tile
decode in the DPAS kernels.

## 8. The engine's record layout (tiled)

At load the GPU repacks each file record (`nv_repack_dev`): every matrix (gate [640, 1792], up, down [1792, 640])
becomes tiles of 16 rows x one 16-column sub-block, 144 B each: the 16 rows' UE4M3 scale bytes, then each row's 8
code bytes as in the file. Tile (G, s) of a matrix with C columns sits at `((G * C / 16) + s) * 144` from the
matrix's start; the matrices and the 16-byte tail keep the file record's offsets, so the size is unchanged. A tile is
exactly one DPAS B operand (16 outputs x 16 k), decoded to VNNI fp16 in registers; the decode kernels give each lane
one output row and read a tile's 144 B contiguously.

## 7. MTP draft layer (`mtp/`, Strata's runtime files)

- `dense.txt` lines `name kind rows cols offset bytes` into `dense.bin`: Q8_0 projections (fc_embedding, fc_hidden,
  q/k/v/o, shared expert, indexer), BF16 hyper-connection and router weights, F32 norms **with the +1 applied**
  (`tools/mtp_rt.py`).
- `experts.bin`: 512 Q2_0 records of 1,382,400 B at full width (2560 -> 640): codes of gate/up rows interleaved
  (2r = gate r, 2r+1 = up r) [1280][40][16 B], codes of down [2560][10][16 B], then the fp16 scales of the same rows
  [1280][40], [2560][10]. Q2_0: 64 values a block, value = (code - 1) * d, code = 2 bits, 4 a byte, LSB first.
- `draft_vocab.bin`: int32 token ids (106,299) for a smaller draft head; not used (the full lm_head drafts).
- Forward (Strata `src/core/mtp.cpp:953-1300`), for cell p = (the trunk's last stream stack at p, token p+1):
  `e = fc_embedding(rms(E[tok]) * w_pe)`, `h_s = fc_hidden((rms over all 4 x 2560 of the stack) * w_ph)[s]`,
  `R_s = h_s + e`; then attention site, **dense** attention over the layer's own K/V cache (no indexer), MLP site, a
  softmax top-10 MoE of the Q2_0 experts with a sigmoid-gated shared expert, the mixer, the trunk's lm_head.
- Prompts compute only the cells' K/V. After a verify window the cells of its accepted rows are recomputed with the
  trunk's real stacks (catch-up); the last of them runs the whole block and drafts; further drafts chain on the
  layer's own output stack.
