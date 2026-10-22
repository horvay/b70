# Drama 3 on the Intel Arc Pro B70: speed work log

Goal: generate voice-cloned lines with Fish Audio Drama 3 (audio.cpp port, Vulkan) as fast as possible on the B70,
then batch-generate about 25,000 short lines. Metric: wall-clock seconds per second of generated audio (s/s, lower is
better). Targets: S2 Pro does 0.21 s/s end to end at batch 32 in its benchmark and 0.123 s/s in production
(46,115 lines averaging 5.08 s).

Code: `~/b70/engines/audio.cpp` (then `~/ai/drama3/audio.cpp`, branch `b70-fast`; the SYCL port followed). Eval set: `eval.tsv` (20 lines from a production job, 4 voices,
2 female and 2 male, 8 of the lines with inline tags; not in the repo). Scorer: `score.py` (WhisperX large-v3 WER,
WeSpeaker ResNet34 similarity to the reference wav, duration and pauses).

## Results

All rows except the first use the 20-line eval set (`eval.tsv`, 150 s of audio) or the 256-line benchmark
(`bench256.tsv`, 4 voices, about 1,700 s of audio). End-to-end includes reference encoding, generation, codec
decode and writing wavs, but not the one-time model load (10-18 s).

| Setup | Set | Step time | End-to-end s/s | Peak VRAM | Quality: mean WER / mean sim (n) |
| --- | --- | --- | --- | --- | --- |
| Stock CLI, one line at a time (spike agent) | short lines | ~45 ms/frame | 1.18 | 22.9-23.4 GiB | 0% WER |
| Stock CLI, one line at a time | eval 20 | ~45 ms/frame | 0.875 | ~23 GiB | 1.2% / 0.721 (20) |
| Batch engine v1, batch 4 | eval 4 | 32 ms | 0.354 | n/a | 0% / 0.699 (4) |
| Batch engine v1, batch 8 | eval 20 | 44 ms | 0.286 | 24.7 GiB | 7.9% / 0.712 (20); 3.0% / 0.739 without one runaway clip |
| Batch engine v1, batch 32 | bench 256 | 269 ms | 0.311 (decode alone 0.078) | 30.1 GiB | not scored |
| + MoE adds fused (multi_add), fewer copies, codec graphs per 32-frame bucket, batch 8 | eval 20 | 39.7 ms | 0.204 (decode 0.026) | not measured | not scored |
| + mat-vec paths up to 16 columns/tokens, batch 16 | bench 64 | 70.4 ms | 0.154 (decode 0.025) | not measured | not scored |
| same, batch 16, MoE mat-vec in one dispatch for all tokens | bench 96 | 75.6 ms | 0.170 (decode 0.025) | 29.8 GiB | bit-identical to the row above |
| same, batch 32 | bench 96 | 146.6 ms | 0.164 | 31.8 GiB | not scored |
| + common suffix tokens in the cached prefix, batch 16 | bench 96 | 76.2 ms | 0.161 (decode 0.025) | ~30 GiB | see final row |
| **Final (`drama3_clone`), batch 16** | **bench 256 (1,711 s)** | **75.1 ms** | **0.159** (decode 0.026, 93% slot use) | **29.1 GiB** | **0.5% / 0.729 (20)**, no clip under 0.5 sim or over 20% WER (stock: 1.2% / 0.721, two weak clips) |
| Final + fused projections (`D3_FUSE` unset, i.e. on; off by default) | eval 20 | 67.8 ms (-8%) | ~0.148 expected | ~29 GiB | 1.3% / 0.748 (20) |
| Final, batch 32, activations padded to 128 columns (`D3_PAD_N=128 D3_PAD_TYPES=f16,q8_0`) | bench 96 | 131.6 ms | ~0.130 at full slot use | 31.7 GiB (too close to 32) | not scored |
| **ggml SYCL backend instead of Vulkan (`D3_BACKEND=sycl`, horvay/audio.cpp 3bf3607: XMX matmuls for 2-32 rows, fused MUL_MAT_ID, oneMKL transposed conv), batch 16** | **bench 256** | **51.0 ms** | **0.132** | not measured | **0.7% / 0.752 (20)** |

## Where the time goes (stock port)

Read from the code before profiling. Every audio frame runs 16 separate graph computes, each followed by a blocking
readback to the host:

1. token embedding graph, then the semantic-code mix on the CPU;
2. the 40-layer backbone decode graph;
3. the LM head over the full 252,288-token vocabulary (about 550 MB of Q8_0 weights read per frame to use 4,097
   logits), read back as 1 MB of floats;
4. thirteen tower (codebook MTP) layers, each its own graph, each followed by a readback and a CPU sort of 4,096 logits
   for sampling.

The gated DeltaNet layers also repeat the 16 key heads to 32 with 32 slices and 31 concats per tensor (about 120
extra kernels per layer), and every Gemma RMS norm recomputes `1 + weight` with its own kernel.

Profile of the stock CLI (`GGML_VK_PERF_LOGGER=1`, warmup plus one 5 s line, 3,700 graph computes; the logger
serializes kernels, so absolute times are inflated): 450,231 CONCAT calls (the DeltaNet head repeat) take 17% of GPU
time, MUL_MAT_VEC 25% (the full-vocabulary LM head is the biggest single item), FLASH_ATTN_EXT 11% (attention over
the whole 2048-entry cache every step), and the rest is tens of thousands of 3-10 us elementwise kernels.

## What helped and what didn't

| Change | Effect |
| --- | --- |
| Batch engine (`batch_engine.cpp`): one graph per step for B rows (backbone, 4,097-row LM head slice, GPU Gumbel-max sampling with RAS, all 13 tower layers), one 10-int readback per row, continuous refill of finished rows, voice prefix computed once per voice and copied into each row, text suffix teacher-forced | 0.875 -> 0.286 s/s at batch 8 |
| Batch 32 instead of 8 | worse: 269 ms per step. ggml-vulkan's tiled matmul shaders (coopmat2, coopmat1 and scalar alike) have a large fixed cost per k step on this card, so any matmul with more than 8 columns costs 80-400 us, and MoE matmuls above 8 tokens drop from ~480 GB/s to ~130 GB/s |
| `GGML_VK_DISABLE_COOPMAT2=1` / patched Mesa (`~/ai/mesa-b70`, DPAS coalescing) at batch 32 | 269 -> ~220 / ~202 ms per step; not the root cause |
| CPU codec decode on worker threads | too slow: 0.75-1.0 s/s with all 32 threads, so it cannot hide behind generation |
| Codec decode graphs cached per 32-frame bucket (`FISH_DAC_DECODE_BUCKET`) instead of padding every line to the longest line seen | decode 0.078 -> 0.025 s/s |
| MoE expert views expanded before the adds, so ggml-vulkan fuses the 7 adds into one multi_add; CONT removed before norms that accept strided input | batch 8 step 44 -> 39.7 ms |
| GQA-packed decode attention (8 query heads of a KV group as 8 query rows) | bit-identical output, no measurable speedup (39.3 -> 39.2 ms) |
| Fused same-input projections (DeltaNet qkv/z/b/a, attention q/k/v, router + shared gate, shared gate/up, expert gate/up, tower w1/w3) | 39.3 -> 36.6 ms, but output codes diverge from frame 0: under investigation |
| Mat-vec path allowed up to 16 columns and 16 MoE tokens (`GGML_VK_MMV_MAX_COLS=16 GGML_VK_MMVID_MAX_TOKENS=16`), batch 16 | 70.4 ms per 16 rows (4.4 ms per row-step against 4.7 at batch 8): 0.154 s/s end to end |
| Split-K for small n (`GGML_VK_SPLIT_K_SMALL_N`, B70 core count added) | only the tiny 32x2048 weights improve (82 -> 34 us); large ones unchanged |
| MoE mat-vec over all tokens in one dispatch (y = token x expert slot) instead of one dispatch per token | bit-identical; microbench 32 tokens 448 -> ~470 us is unchanged, but 1,200 fewer dispatches per step |
| Mat-vec pipelines above 16 columns compute one row per workgroup | no more register spill (8192x2048 at 32 columns 1,569 -> 404 us), but cost stays linear in columns |
| Batch 16 vs 32 with the mat-vec paths | same cost per row (about 4.6 ms per row-step): the mat-vec kernel re-reads the activations for every weight row, so dense weights are not amortized beyond ~16 rows |
| Fused projections, bisected | expert gate/up fusion is bit-identical (no speedup); every dense fusion changes output from frame 0 (tower fusion after ~6 frames), while legitimate float changes tried as controls (no ggml fusion, no coopmat2, GQA packing) stay bit-identical. The fused weights are byte-identical to the separately loaded ones, and quality is equivalent (WER 1.3%, sim 0.748 against 0.5% / 0.729 unfused on the eval set), so the fused graph gives a different but equally good take. Left off by default because the cause of the early divergence is not explained; turning it on saves ~8% |
| Codec graph cache of 6 buckets | out of memory at batch 16-24 (a long line's decode graph needs ~0.9 GB); 3 graphs is the limit |
| Small-n GEMM shader (`mul_mm_smalln.comp`, weights read once, activations in shared memory, `GGML_VK_SMALLN=1`) | correct (max error 3e-7) but not faster: with weight loads and shared reads disabled it still takes 196 us for 8192x2048 at 16 columns. FMA and DP4A kernels top out near 1.35 T multiply-adds/s on this card, so dense layers are arithmetic-bound, not bandwidth-bound |
| Codec decode in a separate thread on its own Vulkan backend | no gain: the GPU serializes the two backends (step 76 -> 89 ms while decodes run, same total) |
| Common suffix tokens (`<\|im_start\|>user\n<\|speaker:0\|>`) moved into the cached voice prefix | forced steps -11%, 0.170 -> 0.161 s/s |
| Coopmat2 GEMM peak (microbench) | 8192x2048 Q8_0 reaches 7 T MAC/s at 128 columns and 16 at 512; 6144x2048 F16 15 T MAC/s at 128 columns (106 us), against 1.45 at 32 columns (277 us). The matrix engines work, but not for 16-64 columns |
| Forcing the coopmat2 tile size (`GGML_VK_CM2_TILE=s/m/l`) for 16-64 columns | no gain: 250-460 us whatever the tile |
| DeltaNet state written in place (output tensor placed over the persistent state) | segfault in `ggml_vk_build_graph` (pre-placed tensor's buffer context); left off (`D3_GDN_INPLACE=1` opt-in), worth only ~2-3% |

Robustness: each line is capped at 100 + 4.5 frames per spoken character + 65 frames per pause tag (normal speech
is ~1.8 frames per character, p90 2.9), and a line that reaches its cap is regenerated once with another seed; both
events are logged to `<out>/limited.tsv`. This bounds the cost of runaway takes (one in ~100 lines ran to the 500-frame
row limit before).

Where the time goes now (batch 16, 4.6 ms per row-step): each row-step needs about 3.5 B multiply-adds (2.4 B dense
backbone and tower weights, 1 B for the 8 routed experts). At ~1.35 T multiply-adds/s that is ~2.6 ms; attention,
DeltaNet, state copies, sampling and elementwise kernels are the rest. Step time is linear in batch from 8 to 32, so
per-kernel launch overhead is not the limit. Faster from here needs matrix engines (DPAS/XMX), and ggml-vulkan's
coopmat paths on Mesa are slow per k step for these shapes.

Batch-16 profile (`eval/perf_b16.log`): dense mat-vec 43%, MoE mat-vec 17.5%, flash attention 8%, state copies 4.6%,
gated DeltaNet 4.4%, small elementwise ops the rest. (The small-n GEMM kernel tried next did not beat the mat-vec path; see the table.)

Microbenchmark (`drama3_mm_bench`, Q8_0 8192x2048): 1 column 53 us, 8 columns 98 us (mat-vec path), 16 columns
359 us and 32 columns 345 us (tiled path). Allowing the mat-vec path up to 16 columns: 213 us; at 32 columns it
spills registers (1,569 us). MoE Q4_K 512x2048 with 256 experts: mat-vec path per token at 32 tokens 448 us against
1,649 us on the tiled path.

## Final state and production use

- Generator: `~/ai/drama3/drama3_clone` (wrapper: takes the B70 lease, pins Vulkan to the Arc, sets
  `GGML_VK_MMV_MAX_COLS=16 GGML_VK_MMVID_MAX_TOKENS=64`) around `audio.cpp/build/linux-vulkan-release/bin/drama3_batch`.
- A 26,569-line job (~83.7 characters per line, about 53 h of audio): about 8.5 h of B70 time at 0.159 s/s, plus ~20 s
  of model load and ~1 s per voice for its prefix. The voices must be in the voices folder first.
  `drama3_clone jobs.tsv --out wav --voices-dir voices`
- Against the targets: 0.159 s/s end to end beats the S2 Pro benchmark number (0.21) but not S2 Pro's production rate
  (0.123). The remaining gap is arithmetic throughput: FMA/DP4A kernels reach ~1.35 T multiply-adds/s on this card, and
  the matrix-engine (coopmat) paths are only fast at 128+ columns, which needs more batch rows than 32 GB allows.
  The promising next steps: fused projections on by default (-8%), batch 32 with padded activations if VRAM is freed
  (codec decode graphs, capacity 768), and a coopmat kernel tuned for 16-32 columns.
