# exl3conv: EXL3 quantization on the Arc Pro B70

exl3conv turns a Hugging Face BF16 checkpoint into an EXL3 model, entirely on one Intel Arc Pro B70. It is native C++
with SYCL (DPC++ 2025.3) and oneMKL: no PyTorch, no CUDA and no Python. The output is
the format turboderp's [exllamav3](https://github.com/turboderp-org/exllamav3) writes, so it loads in exllamav3,
TabbyAPI and the B70 servers in `../` (`b70 serve`).

It reimplements exllamav3 1.5.3's converter. The trellis encoder, the heart of EXL3, was CUDA-only until now; here
it runs on the B70 and produces bit-identical states.

**Drafters:** `build/quant_dflash2 -i <DFlash2 dir> -o <out> [-b K] [--attn-bits K]` (`./build.sh quant_dflash2`) quantizes the big linears of Ripple-65B's DFlash2, one
layer per queue piece (`b70q submit --chunked`). **`--scaled-hessian on`** is exllamav3's per-tensor Hessian (its default since 2026-11-03, `scaled_ldl`): error feedback weighted by what a row's error costs at the output, one more
factorization per tensor. Ripple-65B, first 5 layers at K=3, 115K tokens against BF16, same time per layer: KL 0.0522 -> 0.0250, top-1 73.20% -> 80.89%. Gemma 4 (31B-based at K=4, first 5 layers, same time per layer): layer output error 0.6-7% lower against unquantized. Off by default; pass it.

**Supported now:** Gemma 4 dense (`gemma4`, e.g. Gemma 4 31B and its fine-tunes), integer bitrates 1-8 (4.0 bpw is
the usual), head kept at 16 bits, vision tower copied as BF16. Ripple-65B (`ricursive`, see below), integer bitrates
1-8 per Hessian group, head at 16 bits or EXL3. Qwen3.5 / Qwen3.8 dense (`qwen3_5`, e.g. Qwen3.8-27B and the 40B
upscales, see below), from BF16 safetensors or a Q8_0 GGUF, head and MTP block in EXL3, text only. **Next:**
Qwen3.8-Flash-Next and GLM-5.3-Flash (MoE), fractional bitrates.

## Using it

```sh
./exl3 build                                                     # CPU only, ~1 min
./exl3 convert ~/models/llm/Some-Gemma4-31B ~/models/llm/Some-Gemma4-31B-exl3-4.0bpw -b 4
b70q log <id> -f                                                 # the job id it printed
./exl3 eval --ref ~/models/llm/Some-Gemma4-31B --test ~/models/llm/Some-Gemma4-31B-exl3-4.0bpw
b70 serve ~/models/llm/Some-Gemma4-31B-exl3-4.0bpw --port 8000 --drafter ~/models/llm/gemma-4-31B-it-assistant
```

`convert` queues the job on b70q in short pieces (`--chunked`, about 4 minutes: the first layer that ends past
`EXL3_MINUTES`, default 3): after each piece the calibration state is saved to `<out>.work/state.bin` (~10 s each way
for a 31B model), the job exits 99 and goes back in line at normal priority, and the next piece resumes at the next
layer. Other jobs get the card at the queue's next owner turn. Killing a conversion is safe: it resumes from the last
saved state and redoes the layers after it.
Finished layers are written to `<out>.work/` as they finish. The work folder (~32 GB for a 31B model) can be deleted
once the output is complete. `exl3conv` itself takes `-i <in> -o <out> [-b BITS] [-w <work>] [--max-minutes M]
[--rows 250] [--cols 2048] [--layers N] [--out-scales auto|always] [--compile-only]`. Output scales are `auto` by
default, as in exllamav3 1.6.1: the MLP's gate and up projections drop their output channel scales when the
calibration pass predicts a lower error without them; `always` is 1.5.3's behaviour. On Gemma 4 31B-based at
4 bpw the two measured the same (KL 0.865 vs 0.875 on the chat set, paired difference -0.009 [-0.026, +0.008]).

`eval` scores each `--test` model against the BF16 one on 200 human-written conversations (`data/oasst2-eval.jsonl`,
from OpenAssistant oasst2, rebuilt with `tools/chat_eval_set.py`), rendered with the model's chat template; only the
assistant replies are scored. Per token: KL(BF16 || quant) over the full vocabulary, top-1 agreement and
perplexity, with 95% bootstrap intervals over conversations, plus the paired difference between the first `--test`
and each later one (the precise number when comparing two quants). `--wiki` scores raw Wikipedia text instead.
fine-tuned merges are badly overconfident on text they did not write (Wicked-Stuff-31B: perplexity ~900 on these
replies, unquantized), so for them only the comparisons between quants mean much, not the absolute numbers.

Needs: the oneAPI 2025.3 compiler and oneMKL 2025.3 under `~/opt` (see `notes/toolchain-and-reuse.md`; override with
`ONEAPI`, `MKLROOT`), the `engines/sd.cpp` submodule (for nlohmann/json) and the `llm/dsv41/exllamav3` submodule (for
exllamav3's calibration texts, or `--cal-dir`). The card needs ~24 GB free; host RAM use is small.

## Optimized recipes (per-group bits, self-sampled calibration)

exllamav3's optimized-recipe pipeline (`doc/optimize.md` upstream; specified in `notes/optimize-recipes.md`), ported
to the B70 and kept to what `b70 serve` runs: integer K 3-6, one K per fused group (q/k/v, o, gate/up, down).

```sh
D=~/models/llm/Some-Gemma4-31B.exl3opt; mkdir -p $D
# 1. trace: the model's own answers to exllamav3's 157 seed conversations, through b70 serve on a 4 bpw quant
b70q submit exl3conv --chunked -- tools/trace_job.sh ~/models/llm/Some-Gemma4-31B-exl3-4.0bpw $D/trace.json 10
python3 -I tools/make_trace.py pack --trace $D/trace.json --out $D/cal.safetensors --heldout $D/heldout.json
# 2. anchors: each tensor's error in a real quant
b70-lease run exl3conv -- build/exl3eval --ref <bf16> --test <a 4 bpw quant> --rfn $D/rfn.json
# 3. sensitivity: shaped noise per group at the K = 3 and 4 error levels, KL at the output (~1 h, 8 passes)
b70q submit exl3conv --chunked -- build/exl3measure -i <bf16> --trace $D/cal.safetensors --anchors $D/rfn.json -o $D/noise.json --max-minutes 8
# 4. allocation, then the conversion with both
build/exl3opt -m $D/noise.json -a $D/rfn.json -b 4.0 -o $D/recipe.json
./exl3 convert <bf16> <out> -b 4 --recipe $D/recipe.json --cal-data $D/cal.safetensors
```

The trace comes from a 4 bpw quant rather than BF16 or 6 bpw as upstream suggests: BF16 does not fit the card, and
a 6 bpw Gemma 4 31B with our 16-bit head leaves the server no room for its KV cache. `exl3eval --trace
$D/heldout.json` scores the model's own held-out replies.

## Ripple-65B

`./exl3 convert <Ripple BF16> <out> -b 3` makes a quant the engine in `engines/glm.cpp` loads. Five Hessian groups, one K per
group: q/k/g (the input norm), v (its own input, a blend of that norm and the row's embedding), o, gate/up, down. Set them with `-b`,
`--group-bits qkg=4,v=4,o=4,gu=3,d=4` (any of them, over `-b`), or per layer with `--recipe`. `lm_head` is its own weight:
fp16 by default, `--head-bits 3..8` for EXL3. Half-integer K is not reachable: the encoder is integer only, though the engine
could decode it. `exl3measure` and `exl3opt` are still Gemma's, so a sensitivity-driven recipe has to be written by hand.
`--layers N` stops at N and writes a model of N layers (for parts of a checkpoint; the engine reads it with `RP_TEST_LAYERS=N`,
`exl3eval` with `--layers N`).

**Memory.** Calibration state 9.8 GiB + down's Hessian 6.25 + its LDL factor 6.25 + the working set + the layer does not fit the
card's 31.9 GiB together, so while a layer quantizes the state is parked in pinned host memory (0.4 s each way). The peak is
21.8 GiB, in quantization. Before the card is opened the job checks host memory and disk, and before anything is allocated it checks
its VRAM plan against what is free; every allocation then goes through a ledger with that budget (`rt/dbuf.hpp`), so going over is an
error message and never a driver fault.

**Time**, flat K = 3, 250 x 2048 rows, measured over 5 real layers: 134 s a layer, **about 3 hours for 80**, card time.

| Stage | s |
|---|---|
| capture | 21 |
| q/k/g | 12.8 |
| v | 0.8 |
| o | 6.2 |
| gate/up | 46 |
| down | 36 |
| advance | 8.8 |
| writing the layer, parking | about 2 |

Each queue piece adds saving and reloading the state (about 10 s and 3-10 s; per-layer pieces make it about 3.4 hours).

**Quality**, the first 5 real layers against BF16 (`Ripple-65B-Preview-parts`). Those 5 layers alone are not a working model (ppl about
1.8 million), so only the differences mean anything:

| | K = 3 | K = 4 |
|---|---|---|
| `exl3eval`, 199 conversations, 115k tokens: KL, top-1 | 0.0522, 73.20% | 0.0089, 86.86% |
| Engine, `rp_gen score`, 1100 positions against the numpy reference: KL over its top 64, top-1 | 0.004906, 77.82% | 0.001078, 88.82% |

**Disk**, flat K = 3, chunked: output 26.0 GB (80 layers of 0.299, embedding 1.03, head 1.03). The peak is 44.6 GB beside the source,
after layer 78: 79 finished layers (23.6) and two copies of `state.bin` (10.5 each), because the new one is written next to the old.
With the 129.6 GB BF16 that is 174.2 GB. The job refuses to start without 47 GB free beside the source. Every queue piece saves the
state, so `EXL3_MINUTES` above the default writes less of it.

## Qwen3.5 / Qwen3.8 dense (`qwen3_5`)

`./exl3 convert <in> <out> -b 4 [-w <work dir>]` converts models of Qwen3.8-27B's shape, any layer count (the first target:
DavidAU's 96-layer Qwen3.8-40B-Ultimate-Heretic-Uncensored), into exllamav3's qwen3_5 layout, which `engines/glm.cpp`
(`Q35Model`) runs. `<in>` is the Hugging Face BF16 checkpoint, or a folder with one Q8_0 GGUF of it plus its `config.json`
and tokenizer files (`src/io/gguf.hpp` reads it as the HF checkpoint: names, shapes and values, llama.cpp's transforms undone).
Code: `src/arch/qwen35.hpp` (forward), `src/convert_qwen35.hpp` (loop), `src/exl3/quantize_wide.hpp` (the head).

What it writes, as `turboderp/Qwen3.8-27B-exl3` (names, dtypes and shapes of every tensor checked against its headers):

| Tensors | Stored |
|---|---|
| Decoder linears: GDN `in_proj_qkv`, `in_proj_z`, `out_proj`; attention `q/k/v/o_proj`; MLP | EXL3 at `-b` |
| GDN `in_proj_a`, `in_proj_b` | F16 (exllamav3 leaves them unquantized) |
| Norms, `conv1d`, `A_log`, `dt_bias`, the embedding | BF16 |
| `lm_head` | EXL3 at `--head-bits` (default 6), calibrated on the final norm's output |
| `mtp.*` (fc, its norms, one attention block) | EXL3 at `--mtp-bits` (default 4), uncalibrated as exllamav3 does (no activations: identity Hessian); norms BF16 |
| Vision tower | not written (text only) |

Hessian groups are exllamav3's qmaps: `block.attn.input` (in_proj_qkv + in_proj_z, or q/k/v), `block.attn.output`, `block.mlp.input`
(gate + up), `block.mlp.down`. The calibration state (250 x 2048 x 5120 fp32, 10.5 GB) stays on the card; the quantize peak is
18.6 GiB. The 248,320-wide head does not fit `quantize_group` (it holds six or seven copies of the weight, ~30 GB), so
`quantize_wide` does it in 16,384-column chunks as one tensor: output scales over all columns, input scales summed over them,
LDLQ per chunk with the shared factor, the refit's input-side system summed over chunks. Its global scale is the median of a
search per chunk. On a 5120 x 32768 slice of the real head it gives the same error as `quantize_group` (0.000010).

**Time**, Qwen3.8-40B-Ultimate-Heretic (96 layers: 72 Gated DeltaNet, 24 attention) at K = 4, 250 x 2048 rows, from the Q8_0 GGUF
(2026-11-18): 85.8 s per GDN layer and 54.9 s per attention layer (capture 18.8, advance 15.6, the rest quantization), **2.08 h
for the 96 layers**; then the head's Hessian 1 s, the 6-bit head 4.0 min, the MTP block 0.8 min, the output 0.8 min. With the
queue pieces' state saves and loads (2.4 min), about **2.2 h of card time**. `EXL3_MINUTES=15` gives pieces of about ten layers;
the default 3 gives two, and each piece then waits behind whatever else is queued.

**Quality**, the same model, teacher-forced NLL on wikitext-2 test (the first 110 kB, 26 chunks of 1024 tokens, the second half of
each scored: 13,286 tokens; the two tokenizers give identical ids):

| | NLL | PPL |
|---|---|---|
| Q8_0 GGUF, `llama-perplexity -c 1024 -b 1024` (engines/llama.cpp, SYCL, on the B70) | 1.8114 | 6.119 +/- 0.124 |
| This EXL3 4.0 bpw, head 6, `q35_gen <dir> nll @file 1024` (engines/glm.cpp) | 1.8133 | 6.131 |

The 4-bit model is 0.0019 nats (0.19% PPL) above Q8_0. Layer outputs against the unquantized layer: rfn 0.013 on average, 0.023 at
most. Run `llama-perplexity` with `-b` equal to `-c` on this model: with two chunks per batch (`-b 2048`) every second chunk scores
worse (PPL 6.72), apparently the hybrid model's recurrent state across sequences in that build. The output is 22.05 GB (98 shards);
`Q35Model` holds 19.85 GiB on the card at a 1K context.

## How it works

The same steps as exllamav3 (`notes/exl3-algorithm.md` has the full specification, with line references):

1. **Calibration rows**: 250 x 2048 tokens from exllamav3's bundled texts (C4, code, multilingual, technical, Wikipedia,
   TinyStories, random tokens), tokenized by a native reader of `tokenizer.json` (`src/io/tokenizer.hpp`; it matches
   Hugging Face `tokenizers` token for token on 235k tokens of those texts).
2. **Per layer**: the BF16 layer goes to the card as fp16; every row runs through it while one Hessian (mean of
   x xᵀ) accumulates per input group (q/k/v, o, gate/up, down); the rows' hidden states (11 GB) never leave the card.
3. **Per tensor** (`src/exl3/quantize.hpp`): damp and rotate the Hessian (random signs, blockwise 128-point Hadamard),
   block-LDL factor it (blocked fp32 Cholesky on the card; only the 256 x 256 diagonal blocks go to the CPU in fp64),
   normalize the weight's input and output channels and rotate it the same way, pick a global scale by test-encoding
   sampled tiles, then **LDLQ**: quantize 16 rows at a time from the bottom up, each block corrected by the error of the
   rows below through the LDL factor, every 16 x 16 tile through the trellis encoder. Finally refit the stored scales
   to the Hessian with the trellis fixed, and pack.
4. **Advance**: dequantize the stored layer (fp16 scales, as inference reads it) and run every row through it; the
   result is the next layer's input.

The **trellis encoder** (`src/exl3/viterbi.hpp`) is a tail-biting Viterbi search over 65,536 trellis states for each
256-weight tile, two passes around the ring, all in fp16 with fused multiply-add. One work-group per tile; path costs
in shared local memory; codebook values cached in registers; for K >= 7 several work-items share a node and combine
their branch minima with a sub-group butterfly. Every kernel shape compiles without register spills.

## Measured (2026-10-24, Arc Pro B70)

| | |
|---|---|
| Trellis encoder, tiles/s (K = 2 / 3 / 4 / 5 / 6 / 7 / 8) | 21k / 45k / **50k** / 37k / 28k / 29k / 22k |
| Gemma 4 31B layer, 16 calibration rows | ~56 s, almost all of it quantization |
| Gemma 4 31B, full conversion (250 rows) | to be measured |
| Quality: KL against BF16, next to exllamav3 1.5.3's own conversion | to be measured |

## How it was checked

No CUDA card was used; every check runs on the CPU or the B70.

- **Codebooks, packing, tile order**: the three codebooks reproduce exllamav3's test values exactly; pack / unpack
  round-trips for every K; decoding tensors of an EXL3 model that exllamav3 made (Gemma 4 31B-based 4.0 bpw)
  correlates 0.995 with the BF16 weights of a sibling fine-tune (`tests/test_cpu.cpp`).
- **Encoder**: a scalar CPU port of exllamav3's CUDA kernel, exact fp16 arithmetic included, gives valid tail-biting
  codes and the expected error per K; the SYCL encoder reproduces its states **bit for bit** on 60 tiles covering
  K = 1..8, all three codebooks, subnormal and overflowing costs and ties (`tests/test_gpu.cpp`).
- **Quantizer**: on a real 5376 x 8192 weight, decoding the packed output as inference does gives exactly the
  Hessian-weighted error the quantizer reports; LDLQ cuts that error 18x against plain rounding through the trellis
  at 4 bits; the error falls about 4x per bit; q and k quantized side by side match each quantized alone
  (`tests/test_quant.cpp`).
- **Forward**: the Gemma 4 layer forward on the card matches a separate double-precision CPU implementation to
  ~1e-4 (sliding and full attention, with real inputs from the preceding layers), Hessians too (`tests/test_forward.cpp`).
- **Ripple**: the tokenizer matches Hugging Face on 1.3M tokens of the calibration texts, the chat rendering matches `apply_chat_template`
  on all 200 conversations; the engine loads a converted quant and its scores match `engines/glm.cpp/tests/ripple_forward_ref.py` as above; a
  chunked run with resume writes the same bytes as a straight one.
- **Format**: a converted layer has exactly the tensors, dtypes, shapes and byte count of the same layer in the
  exllamav3-made model.
- **qwen3_5**: the calibration forward against llama.cpp's own forward of the same model, both on the B70
  (`tests/llama_dump.cpp` dumps llama.cpp's per-layer hidden states of a GGUF; `tests/test_qwen35_forward.cpp` runs each layer on
  llama.cpp's input and compares), 512 tokens: from the BF16 checkpoint against the Q8_0 GGUF, each layer's contribution within
  0.2-0.6% (cosine >= 0.99998) for three Gated DeltaNet layers and the first attention layer, which is the Q8_0 rounding; from
  the GGUF itself, within 0.02-0.16% over five layers. The tokenizer matches Hugging Face on 298k tokens of the six calibration
  sources. `tests/test_wide.cpp`: the chunked head quantizer against `quantize_group` on a 5120 x 32768 slice of the real head,
  same error, and decoding the packed tensor as inference does gives exactly the reported error. Layout: every tensor name,
  dtype and shape of a 4-layer conversion (embedding, layers, norm, head, `mtp.*`) as in `turboderp/Qwen3.8-27B-exl3`
  (bit widths aside), the same `tensor_storage` modules.

```sh
tests/run_cpu.sh                                     # CPU tests; EXL3_REF / EXL3_BF16 enable the reference decode
./build.sh test_gpu test_quant test_forward          # GPU tests, run inside a queue turn (tests/gpu_run.sh adds a fault guard)
```

## Differences from exllamav3

On purpose, and without effect on the format: the random sign vectors and the random calibration rows come from this
tool's seeded generator rather than torch's, and the line shuffle of two calibration sources is ours, so the output is
not bit-identical to exllamav3's (any such choice gives an equally good model). Not implemented yet: fractional
bitrates and the bit allocation between them, quantized heads on Gemma 4 (`head_bits` < 16), quantized vision towers, padding of
dimensions that are not multiples of 128, MoE models.

## Layout

- `src/exl3/`: f16 arithmetic, codebooks, packing, the CPU reference and SYCL trellis encoders, quantizer kernels and
  linear algebra (oneMKL), the per-tensor quantizer (`quantize.hpp`; `quantize_wide.hpp` for a head too wide for it), dequantization.
- `src/nn/ops.hpp`, `src/nn/ripple_ops.hpp`, `src/nn/qwen35_ops.hpp`, `src/nn/qwen4exp_ops.hpp`: forward-pass kernels.
  `src/arch/gemma4.hpp`, `src/arch/ripple.hpp`, `src/arch/qwen35.hpp`: the decoders.
- `src/io/`: safetensors reading and writing, GGUF reading (`gguf.hpp`), the tokenizer, calibration rows, chat rendering for the eval.
- `data/oasst2-eval.jsonl`, `tools/chat_eval_set.py`: the eval conversations and the script that picks them.
- `src/measure.cpp` (`exl3measure`), `src/optimize.cpp` (`exl3opt`), `tools/make_trace.py`, `tools/trace_job.sh`,
  `data/trace-conversations.json` (exllamav3's seed conversations): the optimized-recipe pipeline.
- `src/convert.cpp` (`exl3conv`), `src/convert_ripple.hpp` (Ripple's loop), `src/convert_qwen35.hpp` (qwen3_5's), `src/eval.cpp` (`exl3eval`), `exl3` (the queue wrapper), `build.sh`.
- `tests/`: the checks above, `vit_variants.cpp` (encoder kernel shapes side by side), `gpu_run.sh`, `llama_dump.cpp`
  (`build_llama_dump.sh`: llama.cpp's per-layer hidden states of a GGUF, the reference for the qwen3_5 forward).
- `notes/`: the specifications this was built from. `DESIGN.md`: the plan.

Credits: the algorithm and file format are exllamav3's (MIT, Copyright (c) 2025 turboderp), itself built on QTIP
(Cornell RelaxML); the calibration texts are read from exllamav3's repository, not copied here.
