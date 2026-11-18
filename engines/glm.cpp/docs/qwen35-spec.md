# Qwen3.8-27B (`qwen3_5`): implementation spec

Text and vision forward, and MTP drafting (section 7), as `src/q35*.cpp` run them, derived on 2026-11-04 from the sources below. The model is the
SC 5.00 bpw pack `turboderp/Qwen3.8-27B-exl3` (branch `SC_5.00bpw_H6`) in `~/models/llm/Qwen3.8-27B-exl3-SC_5.00bpw_H6`.

| Prefix | Path | Role |
|---|---|---|
| `HF` | `llm/trellis/image/rootfs/opt/venv/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py` | semantics |
| `HFV` | `.../transformers/vision_utils.py`, `.../qwen2_vl/image_processing_qwen2_vl*` | vision positions, preprocessing |
| `X` | `llm/dsv41/exllamav3/exllamav3/architecture/qwen3_5.py` | which activations are fp16 / fp32 |

## 1. Text shape

`D = 5120`, 64 layers, `I = 17408`, vocab 248320. Layer `L` is full attention when `L % 4 == 3` (16 layers), otherwise
Gated DeltaNet (48). No hyper-connection streams, no routed experts, no n-gram table: it differs from Flash-Next
(`docs/qwen4-spec.md`) in being simpler everywhere except attention.

    x  fp32 embedding row (bf16 table)
    for each layer:
        y = fp16(rms(x) * (1 + input_layernorm))
        x += mixer(y)                               fp32 out
        y = fp16(rms(x) * (1 + post_attention_layernorm))
        x += down(fp16(silu(gate(y)) * up(y)))      fp32 out, MLP intermediates fp16
    logits = lm_head(fp16(rms(x) * (1 + norm)))

`rms(x) = x * rsqrt(mean(x^2) + 1e-6)`, computed in fp32.

## 2. Gated DeltaNet

16 k-heads, 48 v-heads, head dim 128, conv kernel 4: the same sizes as Flash-Next, so `q4ops::gdn_gates`, `gdn_conv` and
`gdn_delta` serve unchanged. The one change is `output_gate_type: swish`: the output norm gates with `silu(z)` where
Flash-Next uses `sigmoid(z)` (`HF` `Qwen3_5RMSNormGated`), `q35ops::gdn_norm`.

## 3. Attention

24 query heads, 4 KV heads (a KV head serves query heads `h / 6`), head dim 256, scale `1 / 16`. `q_proj` gives 24 x
(256 query | 256 gate), interleaved. Query and key are normed per head with `(1 + q_norm)` / `(1 + k_norm)` in fp32,
then rotated. Every key at or before the row's cache position is read: there is no indexer. The output is
`softmax(q k / 16) v * sigmoid(gate)`, then `o_proj`.

KV cache: K (normed, rotated) and V, fp16 `[T, 4, 256]`, 4 KB per token and attention layer, 64 KB per token overall.

## 4. mRoPE

Rotated dims 64 (`partial_rotary_factor` 0.25), theta 1e7, pairs `(j, j + 32)`, NeoX. Each pair follows one axis
(`apply_interleaved_mrope`, sections 11, 11, 10): `j % 3 == 1` H, `j % 3 == 2 and j < 30` W, otherwise T.

Positions (`HF` `get_rope_index`): text takes `T = H = W = c` and `c += 1`. An image's `rows x cols` tokens take
`T = c, H = c + row, W = c + col`, then `c += max(rows, cols)`. So the cache row and the rope position differ after the
first image; the host sends `[T, H, W]` per row (`pid_dev`).

## 5. Vision tower

`D = 1152`, 27 blocks, 16 heads x 72, MLP 4304, patch 16, merge 2. All weights bf16 and unquantized in this pack; the
engine holds them fp16 and multiplies with oneMKL's fp16 GEMM.

**Preprocessing** (`Qwen2VLImageProcessorFast`): RGB, `smart_resize` to multiples of 32 (ties to even, area capped by
`Q35_MAX_PIXELS`, default 4194304 as in `llm/state/trellis-serve/xpu/models/qwen3.8-27b-exl3-4.00bpw/model.yaml`; the
processor's own 16.7 MP cap means ~65K patches), bicubic with antialiasing, `(x / 255 - 0.5) / 0.5`. Patches in merge-unit
order: unit `(i, j)`, then `(a, b)` inside it, each `[c][y][x]`.

**Patch embedding**: a Conv3d over 2 equal frames, so its two temporal kernel slices add into one `[1152, 768]` matrix.
Then `+ pos_embed`, the 48 x 48 table sampled bilinearly at `linspace(0, 47, grid)`.

**Block**: `x += proj(attn(LayerNorm1(x)))`, `x += fc2(gelu_tanh(fc1(LayerNorm2(x))))`, with bias on every linear.
`gelu_tanh(x) = 0.5 x (1 + tanh(sqrt(2 / pi) (x + 0.044715 x^3)))`.

**Attention**: `qkv` splits into `[3][16][72]`. Rotary over all 72 dims of q and k: dim `i` turns by `a[i % 36]`, with
`a = row * f | col * f`, `f[j] = 10000^(-2j / 36)`, `j < 18`, partner `(i + 36) % 72` with the NeoX sign. Global, scale
`72^-0.5`, not causal.

**Merger**: LayerNorm per patch, four patches concatenated, `fc2(exact gelu(fc1(.)))`: one 5120-wide row per unit, which
replaces the embedding of one `<|image_pad|>` token.

## 6. Checks

- `tests/q35v_ref.py` + `tests/q35v_check.cpp`: pixels, row 0's stream per block and the merged output against HF on the CPU.
- `q35_gen <dir> nll @file`: teacher-forced NLL of real text, compared with `llama-perplexity` on the Q4_K_M GGUF.
- End to end: the OCR comparison of bank statement pages.

## 7. MTP drafting

The packs carry the model's one-layer MTP head (`mtp.*`, 4-bit EXL3, uncalibrated; `X` `architecture/qwen3_5_mtp.py`,
`modules/arch_specific/qwen3_5_mtp.py`; llama.cpp `src/models/qwen35.cpp` `graph_mtp` agrees):

    MTP row at position p:  x = fc(cat(rms(embed(token p)) * (1 + pre_fc_norm_embedding),
                                       rms(h[p - 1])      * (1 + pre_fc_norm_hidden)))     10240 -> 5120, fp32 out
                            x = one gated full-attention block (as section 3, its own q/k norms, its own KV cache,
                                rope at token p's position) + MLP, (1 + w) norms
                            draft for p + 1 = argmax lm_head(rms(x) * (1 + mtp.norm))

`h[p]` is the trunk's **post-final-norm** state at p (the fp16 row that goes into lm_head), not the residual stream;
`h[-1]` is zeros. A chain feeds a draft's own `rms(x) * (1 + mtp.norm)` back as the next row's `h`.

**Cache.** The MTP K/V cache (`kv[L]`, 4 KB a token a sequence) holds position p once its row has run with the true
`h[p - 1]`. A prompt fills it for all its rows (one extra layer and fc per row, inside the prefill). In decode, a draft
pass first runs the positions the last verify step accepted (their tokens and `h` from a 16-position ring per sequence),
then the sampled token's row, then the chain; the chain's rows past the accepted point are overwritten later, and no
row reads a key past its own position.

**Verify.** `G4Engine` sends the sampled token plus the drafts as rows of one step and samples every row in order with
the request's sampler; a draft is accepted when the sampled token equals it. That is speculative sampling with a
point-mass draft, so the output distribution is the model's own; and since each row costs one sampler call as it would
without drafts, a seeded request gives the same tokens with MTP on or off.

**Gated DeltaNet rollback.** Each sequence has two state slots per GDN layer (conv window + recurrent state, 3 MB a
layer). A decode step reads slot A and writes slot B; its rows of one sequence run in order inside one kernel
(`gdn_conv_seq`, `gdn_delta_seq`: the arithmetic of the one-row kernels, row after row, state in registers), and keep
their inputs (qkv fp32, beta, g: 41 KB a row a layer). When row i is rejected, `restore()` replays the accepted rows
from slot A into slot B with the same kernels (one recorded graph, 2.0 ms, no weights read). Attention K/V needs only
the position moved back. Prompt checkpoints copy the live slot as before.

**Exactness.** Token-identical output needs each row of a k-row step to be summed exactly as a one-row step:
- EXL3 linears: with MTP all decode rows take the DPAS kernel in one 4-row block (1-4 rows, the same K split), so a
  row's sums do not depend on the row count. (Without MTP the vector GEMV stays, now with a fixed K split and 4 tiles a
  thread for one row: 47.4 ms a step against 50.1.)
- Attention: rows of one sequence share a work-group (`attn_dec_grouped`, up to 4), each row's dot products, softmax and
  value sums in the one-row kernel's order.
- GDN as above. Checked: every greedy and seeded-sampled run in the README's MTP section gave the no-draft tokens.

**Draft head.** The MTP's argmax is taken over the first 65,536 vocabulary ids only (`Q35_MTP_HEAD`): the lm_head
kernels compute that column prefix in place (`Exl3Linear::forward(..., n_cols)`), 1.8 ms a draft instead of 3.1 at 30K,
with the same acceptance (the common tokens have low ids). A chain stops when the product of its drafts' MTP
probabilities falls below `Q35_MTP_MIN_P`.
