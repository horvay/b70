# exl3conv: design

A native EXL3 converter for the Intel Arc Pro B70: it reads a Hugging Face BF16 checkpoint and writes an EXL3 model
that loads in exllamav3, TabbyAPI and the B70 servers in `../` (trellis-serve, the exllamav3 ports). C++17, SYCL
(DPC++ 2025.3), oneMKL and oneDNN; no PyTorch, no Python in the conversion itself. First target: Gemma 4 31B dense
(`gemma4`). Next: Qwen3.8-Flash-Next (`qwen4_exp`) and GLM-5.3-Flash (`glm5_next`).

It reimplements exllamav3 1.5.3's converter (turboderp, MIT). The algorithm and file format are specified in
`notes/exl3-algorithm.md`, the Gemma 4 forward in `notes/gemma4-forward.md`, the toolchain and reusable kernels in
`notes/toolchain-and-reuse.md`. This file says how the pieces fit, in what order they get built, and how each one is
checked without a CUDA card.

## What it must reproduce

Stock exllamav3 1.5.3 with the settings of the reference model (`gemma4-31b-based-exl3-4.0bpw`):
`-b 4.0 -hb 16 -vb 16`, out_scales `always`, codebook `mul1`, 250 x 2048 calibration rows. At 4.0 bpw every decoder
linear gets K = 4; the head stays fp16; the vision tower is copied as BF16.

Bit-identity with exllamav3's output is not possible (its sign vectors come from torch's CUDA Philox stream, its
random calibration rows from torch's CPU generator) and not needed: any ±1 sign vectors and any calibration rows of
the same kind give an equally good model. The format has to be exact; the quality has to match.

## Pipeline

```
calibration rows (tokens) ──► embed ──► for each decoder layer L:
                                           1. load layer L (BF16 → fp16 weights, on the card)
                                           2. capture: forward every row through the original layer,
                                              accumulate one Hessian per input group (qmap)
                                           3. quantize each linear (finalize H → regularize → scale search →
                                              LDLQ with the trellis encoder → back-transform → refit → pack)
                                           4. advance: forward every row through the quantized layer
                                           5. write layer L's tensors; checkpoint
                              ──► final norm, head (copied fp16), vision tower (copied BF16)
                              ──► shards + index + config.json + quantization_config.json + copied files
```

## Where things live on the B70

| Data | Size (Gemma 4 31B) | Where |
|---|---|---|
| Calibration state, 250 x 2048 x 5376 fp32 | 11.0 GB | VRAM, for the whole run (no PCIe traffic between layers) |
| One layer's weights, fp16 | 1.3 GB | VRAM, loaded per layer through 64 MiB pinned staging |
| Hessians: qkv 116 MB, o 268 MB / 1.07 GB, gate/up 116 MB, down 1.85 GB | up to 3.2 GB | VRAM, fp32 |
| LDLQ working set for the largest tensor (H rotated, L, error products) | about 4.5 GB | VRAM, freed per tensor |
| Trellis encoder history, one wave of tiles | 1 MB per tile at K = 4 | VRAM, reused per wave (CUDA does the same) |
| Attention scores of one row (32 heads x 2048 x 2048 fp32) | 512 MB | VRAM, per row |

Peak is about 22 GB, under the card's 32 GB. Host RAM stays small: the mmapped checkpoint, a state checkpoint written
only when a queue turn ends (11 GB to disk), and the output tensors of finished layers (written as they finish).

## Components

```
llm/exl3conv/
  build.sh                 icpx, AOT for bmg-g31 (template: notes/toolchain-and-reuse.md §3)
  src/
    io/                    safetensors reader (mmap, shard index), streaming writer, json (nlohmann, vendored path)
    rt/                    SYCL queue, USM pool, pinned staging, xe-fault watcher, timing
    exl3/
      codebook.hpp         3inst / mcg / mul1 decode; 65,536-entry tables          (host + device)
      pack.hpp             pack / unpack trellis, tile permutation                   (host + device)
      viterbi_ref.cpp      scalar CPU encoder, bit-faithful to the CUDA kernel       (tests only)
      viterbi.sycl         the encoder: one work-group per tile, costs in SLM
      hadamard.sycl        blockwise H128 both sides (sub-group FWHT, from dsv_moe_glue.sycl)
      quantize.cpp         finalize H, regularize, scale search, LDLQ, refit (oneMKL fp32 GEMM;
                           Cholesky in fp64 on the CPU, see "Decisions")
      linear.sycl          EXL3 linear forward for the advance pass: had_in → reconstruct (from
                           exl3_esimd.h) → oneDNN fp16 GEMM → had_out
    nn/                    rmsnorm, rope (default + proportional), attention (oneDNN GEMMs + masked softmax,
                           sliding window), gelu_tanh·up, Hessian accumulate (oneMKL gemm fp16 in, fp32 out)
    arch/
      arch.hpp             the interface below
      gemma4.cpp           Gemma 4 dense
    convert.cpp            the loop, bit allocation, checkpoint / resume, output files
  tools/
    make_caldata.py        calibration rows → caldata.safetensors (CPU, HF `tokenizers`; see "Decisions")
  tests/                   CPU tests (no card) and GPU tests (run under a queue pause)
```

### The architecture interface (room for MoE)

An architecture is a list of modules. Each module declares its tensors, its linears (with qmap, K source and output
names), and one forward that takes a mode:

```cpp
struct Linear { std::string key, src, qmap; int in, out; KSource k; std::vector<Split> splits; };
struct Module {
  virtual std::vector<Linear> linears() = 0;          // what gets quantized, and which Hessian each one reads
  virtual std::vector<std::string> copied() = 0;      // norms, scalars, routers, ... stored as they are
  virtual void forward(State&, Mode, Capture*) = 0;   // Mode: original | quantized; Capture null or a sink
};
```

- `State` is `[rows, T, S, hidden]` fp32, with S = 1 for Gemma, 4 for hyper-connection models.
- `Capture` maps a qmap name to a Hessian. A module can register many (one per expert's down projection) and can be
  told `all_experts` for the capture pass only (exllamav3 routes every token through every expert while capturing,
  and uses real top-k routing when advancing).
- `Split` turns one checkpoint tensor into several linears (fused expert tensors, interleaved q+gate) or several
  checkpoint tensors into one input.

Gemma 4 uses one stream, four qmaps per layer and no splits. The interface costs nothing now and keeps the MoE work
to new arch files plus kernels.

## The trellis encoder on the B70

The encoder is most of the compute: per 256-weight tile, 2 passes x 256 positions x 65,536 candidate states
(independent of K), about 3.4e7 fp16 evaluations; Gemma 4 31B has about 1.2e8 tiles, so about 4e15 evaluations.

Layout, following the CUDA kernel that works (`quantize_tiles_optimized.cuh`):

- One work-group per tile, costs (2 x 2^(16-K) fp16, 16 KB at K = 4) in SLM, sub-groups of 16.
- mul1 decoded procedurally (multiply, byte sum, one fp16 FMA), so no 128 KB lookup table.
- History: one byte per node and position (the K-bit branch) in a global buffer sized for one resident wave and reused,
  so it mostly stays in L2.
- Launches of a few hundred tiles each, well under the xe driver's 5 s job timeout.
- The exact fp16 rules of the spec (§4.3): fp16 squared error, fused fp16 FMA, lowest branch wins ties, the
  bit-reversed rank for the end state. Then the GPU output can be compared bit for bit with the CPU reference.

At peak vector rate the B70 needs about an hour for 4e15 evaluations; at a realistic fraction of peak, **a full
31B conversion should take roughly 4 to 8 hours**. That is a guess until step 2 measures tiles per second.
`b70q submit --chunked` runs it in 30-minute turns with a checkpoint per turn, so other jobs still get the card.

## Build order and how each step is checked

No step needs a CUDA card. CPU steps don't touch the B70; GPU steps run in short windows with `b70q pause`, then
`b70q resume`.

| # | Step | Check | Card |
|---|---|---|---|
| 0 | `build.sh`, device probe (SLM size, sub-group sizes, max allocation) | it builds; the probe prints | 1 short run |
| 1 | Codebooks, pack/unpack, tile permutation, scalar CPU encoder, safetensors IO | codebook test vectors (spec §4.1); pack round trip; every tile tail-biting; an ideal codeword encodes back to itself with zero error; decode one tensor of the Twilight EXL3 and correlate with Wicked-Stuff's BF16 (the spec saw 0.995) | no |
| 2 | SYCL encoder | bit-identical to the CPU encoder on random tiles for K = 1..8; tiles per second | yes |
| 3 | Per-tensor quantizer (finalize, regularize, scale search, LDLQ, refit, pack) | on a real tensor with a Hessian from random activations: decode(packed) equals the quantizer's own Q; proxy error falls as K rises; refit lowers it | yes |
| 4 | Gemma 4 forward (original and quantized linears), Hessian capture | spec §5.3 checks (RoPE identity dims, k_eq_v, sliding mask, embedding scale); a CPU fp32 reference forward of a few tokens matches per layer; logits of a short prompt match the Twilight EXL3 served by trellis-serve, top-k | yes |
| 5 | Full loop, checkpoint/resume, writer | convert `Gemma 4 31B-based` BF16 with our tool and compare with the stock 1.5.3 EXL3 of the same model: same file layout, loads in `b70 serve`; KL against BF16 logits and perplexity within noise of stock's | yes, the long run |
| 6 | Convert Wicked-Stuff-31B; README; `b70` integration; NOTICE credits | serves through `b70 serve` with the Gemma drafter | yes |
| 7 | MoE: Flash-Next, then GLM | per-architecture plan when we get there | |

## Decisions (as built)

1. **Calibration tokens**: native. `src/io/tokenizer.hpp` reads `tokenizer.json` (SentencePiece-style BPE with byte
   fallback; byte-level BPE for Qwen / GLM is still to do) and matches Hugging Face `tokenizers` token for token;
   `src/io/caldata.hpp` builds exllamav3's row mix. No Python anywhere.
2. **Cholesky**: on the card in fp32, blocked (oneMKL `trsm` / `syrk`), with only the 256 x 256 diagonal blocks factored
   on the CPU in fp64. No device LAPACK library needed.
3. **Reference for the quality check**: the BF16 original against the exllamav3-made
   gemma4-31b-based-exl3-4.0bpw, with `exl3eval` (KL against BF16, top-1, perplexity).
4. **Credits**: ExLlamaV3 in `../NOTICE`. No trellis-serve code ended up in here (the dequantizer is our own).
5. **Kernel compilation**: one device module per kernel (`-fsycl-device-code-split=per_kernel`); in a shared module a
   kernel's register allocation depended on its neighbours, and shapes that were clean alone spilled.
