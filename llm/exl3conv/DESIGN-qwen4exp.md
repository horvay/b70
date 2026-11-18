# exl3conv: Qwen3.8-Flash-Next (`qwen4_exp`)

The second architecture. Specs: `notes/qwen4exp-forward.md` (forward pass, how exllamav3 converts it),
`notes/qwen4exp-format.md` (the on-disk layout of the EXL3 models the B70 server runs). First target:
`dongfangshuo/Dongfangshuo-Qwen3.8-Flash-Next` (BF16, 360 GB).

## Target recipe

The B70 server (SGLang + trellis-serve + `../trellis-serve.patch`) tiers routed experts between VRAM, pinned RAM and
NVMe only at K = 3; a layer whose experts have another K stays whole in VRAM (1.27 GB at K = 4). It reads integer K
only, and the n-gram table with K <= 6. So the recipe copies the reference model it serves today
(Lygodactylus/Qwen3.8-Flash-Next-Uncensored-exl3-3bpw, 3.05 bpw, 72 GB):

| Tensors | K |
|---|---|
| Routed experts, layer 0 / layers 1-47 | 4 / 3 |
| GDN linears (in_proj_qkv, in_proj_z, out_proj), layers 0-1 / the rest | 5 / 4 |
| Full attention q/k/v/o and the indexer projection | 5 |
| Shared expert | 4 |
| lm_head | 5 (calibrated on the mixer output) |
| MTP head | 3 (uncalibrated) |
| N-gram table | 3 (length-160 rings) |
| Vision tower | 6 (uncalibrated; the server's handling of a BF16 tower is unverified) |

`--recipe flashnext-3` selects this; plain `-b K` stays available.

## What is new compared with Gemma 4

| Piece | Where | Notes |
|---|---|---|
| Hyper-connections: 4 fp32 streams, mix (group norm, rank-320 gate, mean) and inject | `nn/hc.hpp` | small GEMMs in fp32 |
| Gated DeltaNet: causal conv (k = 4), gates, chunked delta rule (C = 64), sigmoid-gated norm | `nn/gdn.hpp` | state and chunk algebra fp32; checked against the sequential recurrence in fp64 on the CPU |
| Full attention: per-head interleaved q/gate, 1+w norms, RoPE on 64 of 256 dims, GQA 12, sigmoid output gate | `nn/ops.hpp` | the QSA indexer is skipped: calibration rows (2048) stay below its 2051 threshold |
| MoE: softmax router, top-10 renormalized, 512 experts + shared expert with a sigmoid gate | `nn/moe.hpp` | capture: every expert sees every token (exllamav3's all-experts capture), one shared gate/up Hessian, 512 down Hessians; advance: real top-10 |
| PLE: hashed 3-gram ids (EOS resets context), 16 heads x 160 from the **quantized** table, signed-sqrt gate, dilated conv | `nn/ple.hpp` | before layer 1 |
| N-gram table: per-head bias, per-row scale, length-160 tail-biting rings, LSB-first packing, `ngram_embedding.safetensors` | `exl3/ngram.hpp` + an L = 160 encoder | quantized first, before layer 0 (51.2B weights, ~1 h) |
| Uncalibrated quantization (no Hessian) for MTP and vision | `exl3/quantize.hpp` | exllamav3's fallback path |
| Expert weights streamed from pinned host RAM | `arch/qwen4exp.hpp` | the 21 GB state stays on the card; a layer's experts (5 GB fp16) go up in batches |
| Byte-level BPE, NFC, Qwen's split regex, non-special added tokens | `io/tokenizer.hpp` | done except the added tokens |
| Output: per-layer shards moved into place, the table file, fp16 tables, MTP, vision | `convert.cpp` | the move avoids needing the output twice on disk |

## Card memory (32 GB)

State 21 GB (250 rows x 2048 x 4 x 2560 fp32) + one layer's non-expert weights + Hessians 1.04 GB + a batch of 64 experts
(0.6 GB) + quantizer working set (largest: 6144^2 Hessians, small next to Gemma's 21504^2) + attention / GDN scratch
for one row: about 25 GB. Fallback if that is too tight beside other jobs: `--rows 128`.

## Build order and checks

| # | Step | Check (no CUDA) |
|---|---|---|
| 1 | Tokenizer: non-special added tokens | HF `tokenizers` on text containing `<think>`, `<tool_call>` |
| 2 | N-gram codec: L = 160 encoder (CPU reference + SYCL), LSB-first pack/unpack, bias, row scale, file writer | bit-exact CPU vs GPU; pack round trip; tail-biting; decode(encode) equals the reported rows |
| 3 | Hashing and PLE | n-gram ids of a text against a Python port of HF's hashing; PLE against a CPU fp64 version |
| 4 | Hyper-connection sites, GDN, attention, MoE forward | each against a CPU fp64 version of the layer on real weights (as test_forward does for Gemma) |
| 5 | Converter loop for qwen4_exp: all-experts capture, per-expert Hessians, expert streaming, recipe, head, MTP and vision uncalibrated | a few layers end to end; tensor names/dtypes/shapes against the reference EXL3 headers (`notes/qwen4exp-format.md`) |
| 6 | Full conversion; serve it with `b70 serve`; KL against the BF16 model on held-out text (exl3eval for qwen4_exp) | |
