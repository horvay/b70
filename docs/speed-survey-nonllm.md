# Non-LLM speed survey on the B70 (2026-11-19)

Method: the ceiling-first loop from sybil-solutions/inference-optimisation-skill. For each workload: what it costs
in card time (queue history), what it measures now, what the hardware allows, then levers ranked by
(gap x likelihood x card time). LLM engines (glm.cpp, llama.cpp) are out of scope here.

Card numbers used for ceilings (spec, then measured where we have it): int8 XMX 367 TOPS spec, ~280 TOPS measured in
oneDNN s8 GEMMs; bf16 XMX ~180 TFLOPS spec, 100-150 measured in GEMMs, 87-99 in oneDNN's fused bf16 SDPA; 608 GB/s
memory. The B70 has no fp8 matrix units (oneDNN runs fp8 by converting to f16), so int8 is the 2x path. Practical
ceiling = 50% of spec for compute-bound work.

## Where the card time goes

Queue history, 2026-11-12 to 2026-11-19 (164 h of card time in total; `b70q` history API):

| Workload | Jobs (owner:program) | Card hours / week | Notes |
|---|---|---:|---|
| LLM servers and LLM work | nq-gm, exl3conv, glm-mtp, gguf, e0-* | ~75 | out of scope |
| **MiniMax H3 video** (Ref2VA, 1920x1088, 8 steps) | make_real:run-units.sh, afterglow:h3-ref2va | **~43** | 38 h busy; 115 s per sampling step at 107 frames |
| **Props worker**: H3-Image product shots + sheets, then Hunyuan3D-2mv meshes | props-hy2:*, props-hy:hy_batch.sh | **~28** | H3-Image is most of it (sheets 53 s, product shots ~33 s a picture plus ~75 s of loading per process); meshes 32 s a prop |
| MiniMax Music 3 | music3-lofi, music3-speed | ~9 | another agent is tuning it today (music3-speed); left alone |
| H3-Image stills via b70-image.sh | station-art, afterglow, kidtex, talk-ux, ... | ~2.4 | Qwen Image 0.4 h, MOSS 0.1 h |
| Fish Audio Drama 3 | voice, voice-mira, conan | ~2.6 | already 6.6x over stock |
| SA3 sound effects, Pixal3D, CG-MLLM | sfx-livingroom, props, livingroom-cgmllm | ~0.8 | |

So the H3 stack (one library, `libxpu_h3.so`, serves both the video and the image model) is about 70% of the
non-LLM card time. Everything below starts there.

## H3 video, 1920x1088x107, 8 steps (make_real's depth route)

Measured (production logs, 2026-11-19, `.multi/20261119-164630/run.log`): one job 1088 s = phase 1 (text encoder and
reference encodes) 26 s + sampling 930 s (113-118 s a step; the first step 130 s with the 19.5 GB block-weight load) +
video VAE decode 128 s (6 windows x 60 tiles of 256 px, ~19 s a window) + audio 1.5 s.

Ceiling per step (72,449 tokens; 50 blocks, hidden 5376, 56 heads x 128, SwiGLU 14336; 19.3 B linear weights):

| Part | Work per step | Measured now | Practical ceiling |
|---|---|---|---|
| Attention | 4 S^2 x 7168 x 50 = 7.5 PFLOP | ~87 s (oneDNN bf16 SDPA, 87 TFLOPS: at the bf16 ceiling) | 83 s bf16; **41 s int8** (183 TOPS) |
| Linears (int8 W8A8) | 2 x 19.3 G x S = 2.8 POP | ~10-12 s (~250-280 TOPS) | ~10 s (measured oneDNN peak) |
| Everything else (norms, RoPE, rotate+quantize, LoRA terms, gated adds) | ~1 TB of memory traffic | ~15 s | ~2 s |

(Before this work; the "Done" section has the after.)

Attention is ~75% of a step, and at bf16 it already runs at the practical bf16 ceiling: the only way down is fewer
bits. VAE decode: 360 tiles x ~2.4 B params x 1797 tokens ~ 3.5 PFLOP a job, ~25 s at 150 TFLOPS against 128 s
measured.

## Ranked levers

Score = estimated card hours saved per week (gap x share of the workload x hours), with the likelihood that it works.
Status after three rounds (2026-11-20 00:30): 1-4 done; 7 (as the loader), 8, 9 and 10 done or measured; 5 belongs
to the LivingRoom session (proposal below); 6 measured not worth it. Details per round further down.

| # | Lever | Workload | Gap / estimate | Likelihood | Cost | Status |
|---|---|---|---|---|---|---|
| 1 | int8 attention in libxpu_h3 (SageAttention-style, the MOSS ESIMD kernel at head_dim 128) | H3 video + H3-Image | attention 1.84x at 72K tokens; **~12 h/week** | high (kernel existed, 160-169 TOPS) | 1 day | done |
| 2 | Video VAE: the library's scratch cached from tile to tile (make_real runs `SD_XPU_POOL_CACHE_MB=0`; on a ~26 GB-full card each of ~360 scratch allocations a tile was slow) + sd.cpp's tile merge in cache order | H3 video | VAE 128 -> ~60 s a 107-frame job; **~2.5 h/week** | measured | hours | done |
| 3 | LoRA term accumulated in place (no copy of every qkv / fc1 output) | H3 video with LoRAs | step -7%; **~2.5 h/week** | measured | hours | done |
| 4 | libxpu_h3's scratch cached during a call (same cache-0 churn in the blocks) | H3 video | step -7%; **~2.5 h/week** | measured | hours | done |
| 5 | H3-Image per-process loading: ~75 s per sd-cli process (17 GB text encoder at 0.45-2.5 GB/s depending on the page cache, 19.5 GB of block weights), and the props worker starts a process per hold, often for 1-2 pictures (1 picture 107-119 s vs ~33 s a picture in a 12-picture run) | props worker | **~2-3 h/week** | high for fewer, longer processes; medium for faster loads | 1 day (LivingRoom worker + sd.cpp-h3img) | block weights load 16.4 -> 7.0 s cold (round 2, deployed); the worker change is proposed to LivingRoom (below) |
| 6 | Video VAE: batch a window's 60 tiles into one call (one 108K-row GEMM instead of 60 of 1.8K, block-diagonal attention); the 25% tile overlap is a third of the work | H3 video | ~60 -> ~30 s a job | medium | 1-2 days (sd.cpp tiling + library) | measured: ~5 s a job at most; dropped (round 2) |
| 7 | H3 text encoder (Qwen3-VL 32B Q4_K_M on ggml SYCL): 11-14 s a prompt with references against a ~3-4 s ceiling | H3 video + H3-Image | ~10 s a picture | medium | days (ggml) | it was the 16 GB load, not the compute (~1.3 s): O_DIRECT loader ~6.2 -> 4.5 s (round 2, deployed) |
| 8 | "Other" in the H3 step (norms, RoPE, rotate+quantize, gated adds): ~2.7 s at 32K tokens against ~1 s of memory traffic; `qk_norm_rope` uses a work-group and a barrier per (token, head) | H3 video | ~3% | medium | hours | gated add + q/k RoPE rewritten, -4% a step (round 2, deployed); the LoRA epilogue measured, needs a custom GEMM (round 3) |
| 9 | Hunyuan3D paint stage (21 s of 32 s a prop; int8 attention already on) | props meshes | 10-20% of ~2.6 h | low-medium | days | GroupNorm, upscaler fusion, folded adds: paint 26.4-27.5 -> 21.8 s a prop (round 3, deployed) |
| 10 | Qwen Image int8 attention + run-time LoRA (known open items) | Qwen stills | 0.4 h/week of card time | high | 1 day | run-time LoRA deployed; int8 attention rejected, 14.6% error (round 2) |

## Done

### 1. int8 attention in libxpu_h3 (H3 video and H3-Image)

`xpu_sdpa/xpu_attn8.hpp` turns H3's bf16 q, k, v ([rows, 56, 128] token-major) into the int8 inputs of the MOSS ESIMD
kernel (`xpu_moss_fa.cpp`): per-token int8 Q and K with K's channel mean removed, f32 softmax, u8 P times per-channel
int8 V. libxpu_h3 links libxpu_moss_fa.so and runs it for every call of 4096+ rows (`SD_XPU_H3_SDPA=int8`, the new
default; `graph` and `sycl` as before). The int8 copies of K and V go into the bf16 q buffer and the output into the
k buffer once their bf16 data is dead, so peak memory does not grow. The first call per sequence length is checked
against the exact SYCL kernel on 256 query rows (relative RMS printed; above 2e-2 it falls back to bf16).

Per layer, `bench/attn8_bench` (synthetic H3-like q / k / v, min of 3, interleaved):

| Tokens | oneDNN bf16 SDPA | int8 | Speedup | Rel. error vs f32: bf16 / int8 |
|---:|---:|---:|---:|---|
| 72,449 (1920x1088x107) | 1731 ms (87 TFLOPS) | 941 ms (160 TOPS) | 1.84x | 5.6e-4 / 2.7e-3 |
| 37,440 | 450 ms | 248 ms | 1.82x | 6.2e-4 / 3.1e-3 |
| 16,640 | 82 ms | 62 ms | 1.32x | 7.4e-4 / 4.2e-3 |
| 6,170 | 11.0 ms | 9.1 ms | 1.21x | 8.7e-4 / 5.4e-3 |

At 72K tokens int8 runs at 160 TOPS, 87% of the 183 TOPS practical ceiling.

**H3 video**, make_real's depth-route engine (Realism Turbo Remix v10, two rank-32 LoRAs, 8 steps), one real unit
(scene-03, 1920x1088x39, 32,530 tokens), same build and settings, `SD_XPU_H3_TIMING=1`:

| | bf16 SDPA | int8 SDPA |
|---|---:|---:|
| attention per step | 16.7 s (91 TFLOPS) | 9.0 s (169 TOPS) |
| step | 28.7 s | 20.4 s |
| sampling (8 steps, first job of the process) | 240.6 s | 179.0 s |
| accuracy check on real q / k / v (rel. RMS vs f32) | 9.3e-4 | 4.9e-3 |

Quality (`cmpav.py`: per-frame PSNR / SSIM on the decoded frames, waveform SNR on the generated audio). H3 video is
not bit-reproducible, so the floor is two bf16 runs of the same job:

| Pair | Mean diff | PSNR | SSIM | Audio SNR |
|---|---:|---:|---:|---:|
| bf16 run 1 vs bf16 run 2 (floor) | 1.92 levels | 36.4 dB | 0.978 | 27.8 dB |
| today's production render vs bf16 run 1 | 1.85 | 36.1 | 0.977 | 27.1 |
| bf16 runs vs int8 | 1.94-2.26 | 35.8-36.3 | 0.974-0.977 | 21.8-22.5 |

The video stays inside the run-to-run floor; the generated audio track moves a little more (~5 dB lower waveform SNR).
At 1920x1088x107 attention is ~75% of a step, so the expected step is ~115 -> ~75 s and a 107-frame job ~1090 ->
~770 s (to be confirmed on a production job after the swap).

**H3-Image** (product shots, 2048x2048, 7,962 tokens): step 2.81 -> 2.44 s, a picture 22.9 -> 20.9 s; real-data check
3.7e-3. Unlike video, this H3-Image path was bit-identical run to run with the old library (production = two reruns),
so the int8 change shows: mean 3.6 and 6.1 levels (PSNR 24-26 dB) on the two test pictures. Both look equally clean; the
distilled 6-step model moves details (where a light strip runs, the cabinet's base), like a nudged seed. Sheets
(11,334 tokens) gain a little more. `SD_XPU_H3_SDPA=graph` keeps the old pictures exactly.

### 2. The in-place LoRA term (libxpu_h3)

With LoRAs, each int8 GEMM also yields x A^T as extra output columns; the library then copied the base columns into a
dense buffer and accumulated (x A^T) B^T into the copy. Now the LoRA GEMM accumulates straight into the base columns of
the GEMM output, and the four consumers (q/k norm + RoPE, the V copy, the gated residual add, SwiGLU + quantize) read
it with a row stride. Same arithmetic, one read and one write less per output element, one large allocation less per
chunk. Two rank-32 LoRAs, 39-frame unit: qkv 2.58 -> 2.04 s, fc1 3.49 -> 2.78 s a step; step 20.45 -> 18.98 s. Output
vs the int8 run before it: 2.17 levels, 35.9 dB (run-to-run floor 1.92 levels, 36.4 dB).

### 3. Scratch caches under `SD_XPU_POOL_CACHE_MB=0`

make_real sets `SD_XPU_POOL_CACHE_MB=0` (a 1 GB cache stalled 4-reference long units at 31.5+ GB in October). With no
cache every scratch free drains the queue and returns the block to the driver, and every allocation is a
`malloc_device`, which is slow on a card that is ~26 GB full. Measured in the real pipeline:

| | cache 0 | 1 GB cache | per-library floor (deployed) |
|---|---:|---:|---:|
| H3 step (39 frames, LoRA build) | 18.98 s | 17.60 s | 18.14 s (`SD_XPU_H3_POOL_CACHE_MB=512`) |
| VAE, one 256 px tile in the library | 0.255 s | 0.107 s | 0.111 s (`SD_XPU_H3VAE_POOL_CACHE_MB=512`) |

An idle-card bench of the same tile showed 0.110 s with either setting, which is why the first VAE hypothesis looked
wrong: the cost only appears with the card nearly full. Each library has its own pool, so the floors don't touch the
other libraries; libxpu_h3 still trims at the end of every call and libxpu_h3vae at `xpu_h3vae_release()` (after each
video's decode). Peak pool memory at 32K tokens: 2956 MiB live before; 2578 MiB live + at most 512 cached now.

### 4. sd.cpp's tile merge (`src/runtime/tiling.cpp`)

`sd_tensor_merge_2d` blended each output tile with the plane loop innermost: for an H3 video tile that walks 84 planes
(3 colours x 28 frames) 8 MB apart per pixel, through bounds-checked accessors. Now planes are the outer loop, rows are
contiguous and the blend weights are computed once per row and column. A unit test against the original function (same
compiler and flags, 13 cases: H3 geometry, edges, circular wrap, no overlap) is bit-identical in every case; ~40 ms ->
~4 ms a tile. With 3 the VAE goes from 3.5 to 8.5 tiles a second; the 39-frame unit's decode 39.0 -> 18.2 s.

### All together

The 39-frame unit with everything deployed (new sd-cli, libxpu_h3, libxpu_h3vae; make_real's environment), against the
bf16 control with the same instrumentation: sampling 240.6 -> 162.2 s (1.48x), VAE 40.4 -> 19.6 s, job 325 s (today's
production render) -> 219 s (1.49x). Scaled to make_real's usual 1920x1088x107 job (attention grows with the square of
the tokens): step ~113 -> ~66 s, VAE ~128 -> ~60 s, job ~1090 -> ~630 s, about 1.7x.

**Production check** (make_real's first job on the deployed build, 19:16, scene-15t at 1920x1088x73, 50,811 tokens;
real-data attention check 5.5e-3), against a 73-frame unit rendered earlier the same day on the old build:

| | Before (15:07) | After (19:16) | |
|---|---:|---:|---:|
| step (steady) | 60.3 s | 36.7 s | 1.64x |
| sampling, 8 steps | 502.3 s | 318.3 s | 1.58x |
| VAE decode | 90.8 s | 38.0 s | 2.39x |

Validation also covered: Qwen Image through the new sd-cli (`b70-image.sh -m qwen --xpu`), the same picture to the
eye, mean 1.1 levels apart, from the Qwen prefix K/V cache that the newer commit wires up (documented in
`b70_xpu.md`; `SD_XPU_QWEN21_PREFIX_CACHE=0` turns it off), sampling 10.8 -> 8.8 s.

Deployed 2026-11-19 19:14 (renames at a moment with no sd-cli running): `engines/sd.cpp/build-sycl/bin/sd-cli`,
`xpu_sdpa/libxpu_h3.so`, `xpu_sdpa/libxpu_h3vae.so`; the previous files are kept as `*.prev`. To go back on a single
change without a rollback: `SD_XPU_H3_SDPA=graph`, `SD_XPU_H3_POOL_CACHE_MB=0`, `SD_XPU_H3VAE_POOL_CACHE_MB=0`.

### Tried and dropped

- Keeping the VAE's scratch cached looked useless in an idle-card bench (0.112 vs 0.110 s a tile) and was reverted, then
  measured in the real pipeline (above) and brought back.
- A 1 GB scratch cache for the blocks: 3% faster than 512 MB, but it could push long shots past the old peak.

## Tools added

All in `engines/sd.cpp/xpu_sdpa/bench/` (binaries git-ignored), built with `bench/build_bench.sh NAME` except the two
plain host programs (`g++ -O2 -std=c++17 NAME.cpp -o NAME -ldl`):

- `attn8_bench`: int8 vs oneDNN bf16 attention at head_dim 128, time and error against the f32 SYCL kernel, any
  `S H` list; `ATTN8_DUMP=path` writes the int8 output for bit comparisons between kernel builds.
- `h3vae_tile_bench.cpp` (host): one VAE tile through a given libxpu_h3vae build, timed, output written.
- `h3vae_shapes_bench`: the VAE's GEMMs at one tile's rows vs batched, its attention; `rows M...` for row padding.
- `h3_load_bench.cpp` (host): libxpu_h3's weight load for a given build.
- `read_bench`: buffered vs O_DIRECT reads into pageable vs pinned memory, page cache dropped per slice.
- `lora_bench`: the int8 GEMM with and without LoRA columns and the LoRA accumulate at H3's qkv / fc1 shapes.
- Timers in the libraries: `SD_XPU_H3_TIMING=1` (H3 step: attention, the four GEMM groups, the memory-bound kernels),
  `SD_XPU_H3VAE_VERBOSE=1` (per VAE tile), `HY_OP_TIME=1` / `HY_CONV_TIME=1` / `XN_ATTN_TIME=1` (Hunyuan3D ops,
  convolutions, attention), `GGML_SYCL_PROFILE=1` (ggml-sycl ops in sd-cli).

## Notes for whoever builds next

- `/usr/bin/ninja` is gone from the system. Since round 2 `~/.local/bin/ninja` (on PATH) is a pinned 1.13.2 copy from
  the micromamba package cache, and `build_sycl.sh` passes whatever ninja is on PATH to CMake, so build dirs configured
  with `/usr/bin/ninja` (build-sycl) build again. No root needed.
- `build-sycl/bin/sd-cli` was last built on Nov 10 from a dirty dc210b1; the deployed one is HEAD (65e8d08) plus the
  tile merge, built with `build_sycl.sh <separate dir>` (removed after the copy; `build-sycl/bin/sd-cli.prev` is the old
  binary). Its HEAD build also turns on the Qwen prefix K/V cache that `b70_xpu.md` already describes.

## Round 2 (2026-11-19 evening)

### Proposal for the LivingRoom props worker (H3-Image per-process load)

For the LivingRoom session, which owns `Tools/props/hy2_worker.py` and `hy2_batch.sh`; nothing here is changed in their
code.

What a process costs before its first picture, from the production logs (`Saved/Props/hy2/log/*.batch.log`):
the 17 GB text encoder read (8.5 s warm, 48 s cold), the 19.5 GB of block weights (11.3 s; 2.5 s from a warm page cache
with the parallel reader below), the first condition (~5-11 s). ~60-75 s in all, once per sd-cli process. A process
that ends after one picture costs 107-119 s for it; a 12-picture process ~33 s (product) to ~53 s (sheet) a picture.

Today the worker starts one sd-cli process per stage (product shots, then sheets) per chunk of 12 props, and it holds
the card as a standby server: as soon as any job queues (make_real queues its next batch the moment the last one ends)
it finishes the picture in progress and steps aside, so a hold often ends after 1-6 pictures and the next hold pays the
load again.

Proposed, in order of effect:

1. **When the worker's turn comes through the line (its backlog made the standby ticket urgent), run it as an ordinary
   job chunk**: a fixed amount of work (one full stage of a 12-prop chunk, ~5-10 min) before honouring the stop file,
   instead of stepping aside after the current picture. That is what any other job in the queue gets, and it is inside
   the queue's 15-20 minute guidance. Expected: no 1-2-picture processes; ~75 s of loading spread over 12 pictures.
2. **One process for both H3 stages of a chunk.** The sheets need the product shots as references, and
   `sd-cli --multi-jobs` encodes every job's conditions before sampling, so the two stages can't share one job list as
   is. Either (a) give sd-cli a "stage barrier" line in the jobs file (finish everything above before encoding what
   follows; a small change in the H3-Image sd-cli, `examples/cli/main.cpp` of sd.cpp-h3img), or (b) run product shots
   of chunk N+1 together with the sheets of chunk N in one process. Saves one load per chunk (~60-75 s per 12 props).
3. Keep the text encoder file hot: nothing to do once (1) and (2) cut the number of loads; with ~49 GB of RAM free the
   page cache cannot hold the 17 GB encoder next to make_real's files between jobs (fincore: 0.6 of 17 GB resident).
4. **Port the O_DIRECT model loader to the H3-Image sd-cli.** The main sd.cpp's `src/model_loader.cpp` change (round 2
   below: the text encoder ~6.2 -> 4.5 s when the drive is free) is three hunks; `sd.cpp-h3img`'s loader has the same
   read loop (`read_buffer` / `file.seekg(tensor_storage.offset)`), only its include block differs, so the patch needs a
   small manual port (`git -C ~/b70/engines/sd.cpp diff src/model_loader.cpp`). The block weights already load faster
   there: they come through libxpu_h3.so's parallel reader (deployed 21:14).

### Item 1, H3 video VAE batching: measured, not worth it

With the scratch cache in place the library spends 107-111 ms on a 256 px tile. `bench/h3vae_shapes_bench`
(per tile, per decoder block): the four bf16 GEMMs 1.81 ms alone, 1.37 ms with 60 tiles in one GEMM (176 TFLOPS);
attention 0.38 ms either way. Most of the single-tile loss looked like one oneDNN kernel choice (the qkv GEMM at 1797
rows 0.62 ms, at 1800 or 1824 rows 0.32 ms), but padding the rows to a multiple of 32 inside the library changed
nothing (106.8 vs 107.3 ms a tile, output bit-identical). Batching the 60 tiles of a window would save at most ~16 ms
a tile, ~5 s of a ~630 s 107-frame job, for a restructure of sd.cpp's generic tile loop. Dropped; the padding was
reverted.

### Item 2, H3 text encoder: it is the load, not the compute

`GGML_SYCL_PROFILE=1` on the production prompt (1,256 tokens with the reference video's frames): the Qwen3-VL LLM's
own work (Q4_K / Q6_K matmuls through oneDNN f16, attention, norms) is ~1.3 s. The rest of the "condition graph"
time is reading the 16 GB GGUF: 14.2 s in that run (1.15 GB/s), 6.4 s in a production log (2.5 GB/s). The page cache
does not hold the file between jobs (0.6 of 17 GB resident right after a make_real job). Read benchmark on cold 1-2 GB
slices of the file (`bench/read_bench`, page cache dropped per slice with posix_fadvise):

| Read path | GB/s |
|---|---:|
| buffered, 16 threads, into pageable or pinned memory (sd.cpp's loader today) | 3.1-3.3 |
| O_DIRECT, 16 threads, into pageable memory | 6.0-6.8 |
| O_DIRECT straight into pinned (Level Zero host) memory | 3.2-3.3 |

So the loaders read with O_DIRECT into a pageable bounce buffer and copy from there: `src/model_loader.cpp`
(tensors of 4 MiB+, `SD_LOAD_DIRECT`, default on) and `xpu_native.hpp`'s safetensors reader (8 threads,
`SD_XPU_READ_THREADS`, `SD_XPU_READ_DIRECT`).

Measured on a small production render (the 39-frame unit's prompt and references, 512x288x9, 1 step), and as model
loads in their own processes (`bench/h3_load_bench`, the file dropped from the page cache first for "cold"):

| Load | Before | After |
|---|---:|---:|
| H3 text encoder, 16 GB GGUF, sd.cpp loader (O_DIRECT) | 19.8 s | 4.2 s (twice) |
| that job's phase 1 (references + text encoder) | 30.9 s | 13.6 s |
| that job, total | 51.1 s | 29.0-34.1 s |
| libxpu_h3 block weights, 18.8 GB, cold (8 buffered readers) | 16.4 s | 7.0 s |
| the same, file in the page cache | 3.7 s | 3.1 s |

O_DIRECT readers in `xpu_native.hpp` (8 MiB pieces per thread, one 64 MiB chunk at a time) were slower than the plain
parallel reader there (9.1 s cold, 5.3 s cached) and were taken out; sd.cpp's loader streams whole tensors per thread
without a barrier, which is where O_DIRECT pays. Both O_DIRECT readers were checked byte for byte against plain
reads on 60 random ranges of the real files (edges of the file included); the renders stay inside the run-to-run
spread (3.3 levels against 3.0 between two runs of the new build).

Deployed 2026-11-19 21:14 (no sd-cli running): `build-sycl/bin/sd-cli`, `xpu_sdpa/libxpu_h3.so`, `libxpu_h3vae.so`;
this round's previous files are `*.prev2` (round 1's are still `*.prev`).

**Correction to the table above:** the "before" of 19.8 s was taken while something else read the drive. Old and new
loaders alternating back to back on the same render: 6.61 / 5.81 s (buffered) against 4.40 / 4.60 s (O_DIRECT), so
**~6.2 -> 4.5 s** a load when the drive is free. Production keeps a long tail either way: since the swap make_real's
loads were 4.6, 18.2 and 37.8 s, the slow ones while UnrealEditor or another session read the same NVMe (before the
swap: 4.2-6.6 s free, 14-60 s contended).

**Production, the full stack after rounds 1 and 2** (make_real, 1920x1088x107, 2026-11-19 21:50): 70.1 s a step
(113-118 s before round 1), VAE decode 62.8 s (127-128 s).

### Item 5, Qwen Image: run-time LoRA yes, int8 attention no

`libxpu_qwen21` now keeps the checkpoint's int8 weights as stored when a LoRA is loaded and adds the LoRA term at run
time, as libxpu_moss does (two small GEMMs per linear). Test: `b70-image.sh -m qwen --xpu`, 1024x1024, 10 steps, seed 7,
the rank-16 `cinematic_painterly_qwen21` style LoRA on 224 linears; the reference is the library's bf16 mode, which
merges the LoRA exactly:

| Picture | vs bf16 reference | vs no LoRA |
|---|---:|---:|
| int8, LoRA merged + requantized (deployed before) | 3.72 levels, 30.4 dB | 16.5 levels |
| int8, LoRA at run time (now) | 1.71 levels, 36.7 dB | 17.6 levels |
| bf16 reference | 0 | 17.5 levels |

Same picture twice: 0.00 levels. Without a LoRA the new library gives the deployed library's picture (0.002 levels).
Cost: 0.43 -> 0.47 s a model call with this LoRA. A DMD2-type distillation LoRA, whose weight deltas are below int8
rounding, is where the merge lost most of the effect (MOSS: ~4% kept). `SD_XPU_QWEN21_LORA_MERGE=1` merges as before.
Deployed 22:01 (`xpu_sdpa/libxpu_qwen21.so`, previous file `libxpu_qwen21.so.prev2`).

int8 attention (H3's, generalized to Lq != Lk for Qwen's prefix cache) measured **14.6% relative error** on Qwen's
real q / k / v at 4096 x 4122 in its first-call check (H3: 0.5%), and the guard switched back to bf16. Qwen Image's
attention inputs carry outliers per-token int8 cannot hold. Removed again, with the kernel generalization (no user left),
so the int8 kernel library (`libxpu_moss_fa.so`, also MOSS's) is unchanged.

### Item 4, Hunyuan3D paint stage: profiled (round 3 below acted on it)

Paint-only on two of today's props (their production meshes, `--mesh`), production options:

| | Time |
|---|---:|
| a prop's paint stage (second prop of a batch; a deployed run, twice) | 22.7-22.8 s |
| the same with `--int8 paint` (W8A8 linears, per-row, no rotation) | 22.4 s (-1.5%) |

Profile of one prop (`HY_CONV_TIME=1`, a new opt-in timer in `hunyuan3d/hy_conv.hpp`, plus `XN_ATTN_TIME=1`; both
synchronise every call, so the totals run a little high): condition 2.7 s, reference UNet 1.0 s, 10 steps 13.0 s,
decode 2.2 s, upscale 4.2 s.

- The 10 UNet steps: attention 4.7 s (multi-view 2.3, reference 1.2, self 0.8, DINO 0.3; already int8),
  convolutions ~1.0 s, and ~7 s in the linears, norms and elementwise kernels. The int8 linears gain 1.5%, so most of
  those 7 s is GroupNorm / LayerNorm / GEGLU / RoPE / residual passes; the next step is timing those kernels the way
  libxpu_h3's are timed now.
- Real-ESRGAN (4.2 s for 12 views): convolutions 1.8 s, the 3x3 ones with 32 output channels at ~28 TFLOPS (192->64:
  ~111), one image at a time; LeakyReLU, the dense-block concatenations and the scaled residual adds run as separate
  passes over 512 x 512 x 64-192 channels. Fusing LeakyReLU and the residual adds into the convolutions (oneDNN
  eltwise / sum post-ops) and running 3-4 views per call is the likely ~2 s.

Expected in all: maybe 15-20% of a prop's 32 s, ~0.3-0.5 card hours a week. Left for a next round.

### Item 3, the memory-bound kernels in the H3 step

`SD_XPU_H3_TIMING=1` now also times the kernels between the GEMMs. The 39-frame unit (32,530 tokens) before / after:

| Kernel | Before | After | Floor by memory traffic |
|---|---:|---:|---:|
| gated residual add (twice a block) | 0.92 s | 0.35 s | ~0.35 s |
| q / k RMSNorm + RoPE, V copy | 0.79 s | 0.53 s | ~0.28 s |
| RMSNorm + modulation + rotate + quantize | 0.24 s | 0.24 s | |
| attention-output rotate + quantize | 0.26 s | 0.26 s | |
| SwiGLU + rotate + quantize | 0.36 s | 0.37 s | |
| step (sum) | ~18.5 s | ~17.7 s | |

The gated add now does 8 columns of a row per work-item (vector loads, the segment test once per 8 elements; same
arithmetic per element). The q / k norm + RoPE runs one 16-lane sub-group per (token, head) with the RoPE partner taken
by a shuffle, instead of a 128-thread work-group with a barrier (same per-element math; the sum of squares is added in
another order). Sampling of the unit 162.2 -> 152.0 s.

Quality, and a lesson about this unit: its renders fall into two variants whose generated audio (and with it the lip
timing in some frames) differs completely (waveform SNR about -3 dB between variants, ~24 dB within one), while the
pictures stay at 2.0-2.5 levels within a variant and ~3.5 between. Today's bf16 production render and the rewrite
land on one variant; round 1's library and the deployed one before this change (three renders, serial and parallel
weight reader alike) on the other. Neither is worse: their audio follows the song's loudness at 0.67 and 0.73-0.75
correlation. Tiny numeric changes pick the variant, as with H3-Image's details. Deployed 22:58
(`xpu_sdpa/libxpu_h3.so`, previous file `libxpu_h3.so.prev3`).

## Round 3 (2026-11-19 night)

### The LoRA cost in H3's qkv / fc1 GEMMs: measured, the rest needs a custom GEMM

`bench/lora_bench` at the production shapes (8192-row chunks, two rank-32 LoRAs = 64 extra rows):

| per 8192 rows | qkv (N = 21504) | fc1 (N = 28672) |
|---|---:|---:|
| int8 GEMM, plain N | 5.59 ms (339 TOPS) | 7.47 ms |
| the same with the LoRA's 64 columns appended (as deployed) | 6.22 ms | 8.69 ms |
| LoRA accumulate y += u B^T (oneDNN bf16, K = 64, strided) | 1.53 ms (462 GB/s) | 2.15 ms (437 GB/s) |

The accumulate already runs at the memory limit (one read and one write of the output). The appended columns looked
like a 11-16% GEMM penalty, so the A R rows went into their own 64-column GEMM (libxpu_moss's way). In the real
pipeline that changed nothing: qkv 2.14 -> 2.11 s, fc1 2.92 -> 2.87 s a step, sampling 153.9 -> 155.4 s (the GEMMs run
~20% under the benchmark's clocks at 275 W anyway). Reverted. What is left of the LoRA cost (~0.9 s of a 17.7 s step at
39 frames) is the two extra passes over each qkv / fc1 output; every arrangement oneDNN offers keeps them, and only a
custom int8 GEMM with the low-rank term in its epilogue would remove them (~3-5% of a step, a large ESIMD job). Not
done here; round 4 removed them without a custom GEMM (below). A SYCL kernel for the rank-64 update was 10x slower than
oneDNN and was dropped.

One more data point on the two-variant H3 renders: the deployed library landed on variant B this time where the same
build gave variant A before, so the variant is picked per run, not by the library.

### Hunyuan3D paint: the upscaler's epilogues fused

Real-ESRGAN's dense blocks ran LeakyReLU after each of their convolutions as a separate strided pass, then a scaled
residual add and a copy; the RRDB residual and the upsampling LeakyReLUs were separate passes too. They now run inside
the convolutions as oneDNN post-ops (`hy_conv.hpp` `ConvPost`: scale, sum with the destination, scale, add a tensor,
LeakyReLU, all in f32 with one rounding; `HY_ESRGAN_FUSED=0` restores the passes). Two props, paint-only on their
production meshes, the same test binary:

| | upscale, a prop | textures vs an unfused run |
|---|---:|---|
| separate passes (twice) | 3.8-3.9 s | 0.37 / 0.53 levels, 51.2 / 48.8 dB (run to run) |
| fused | 2.5-2.6 s | 0.38 / 0.52 levels, 50.8 / 49.0 dB |

### Hunyuan3D paint: GroupNorm and the UNet's elementwise passes

An opt-in op timer (`HY_OP_TIME=1`, next to `HY_CONV_TIME=1`; both synchronise every call) on one prop's paint stage:
GroupNorm 2.71 s over 910 calls, head transposes 0.53 s, residual adds 0.42 s, GEGLU 0.31 s, LayerNorm 0.23 s, against
~1.9 s of linears, 3.3 s of convolutions and 4.8 s of (int8) attention. GroupNorm's normalising pass did two 64-bit
divisions per element and its statistics passes one global atomic per lane, with idle lanes at 320 channels.

- **GroupNorm v2** (`hy_conv.hpp`, `HY_GN_V2=0` for the old kernel): lanes own channels, sums meet per group in local
  memory (one atomic per group and work-group), the normalising pass runs on a 3-D range with 16-byte loads; still
  mean first, then the variance around it. GroupNorm 2.71 -> 0.49 s.
- **Residual adds folded** into the GEMM or convolution before them (`lin(..., accumulate)`, `conv2d(..., true)`):
  all eight separate add passes of a transformer block and a resnet gone (one rounding instead of two).
- **GEGLU** on a 2-D range with 16-byte loads (0.31 -> 0.20 s); **head transposes** (`xn::heads_to_tokens`, shared
  by every library that rebuilds) as 16-byte copies on a 3-D range when D is a multiple of 8 and both pointers are
  16-byte aligned (0.53 -> 0.19 s, the same bytes).

Two props, paint-only on their production meshes (seconds for the second prop of each batch; textures against a
run of the binary before each step, run to run 0.36-0.53 levels / 48.8-51.2 dB):

| Build | 10 steps | VAE decode | condition | upscale | a prop | textures |
|---|---:|---:|---:|---:|---:|---|
| deployed (separate passes) | 11.6 s | 2.0-2.1 s | 1.4 s | 3.7-3.9 s | 26.4-27.5 s | |
| + fused upscaler | 11.6 s | 2.0 s | 1.4 s | 2.5 s | 25.7 s | 50.8 / 49.0 dB |
| + GroupNorm v2 | 9.8 s | 0.8 s | 0.6 s | 2.6 s | 22.0 s | 50.9 / 50.4 dB |
| + adds folded, GEGLU, transposes | 9.1 s | 0.8 s | 0.6 s | 2.6 s | 21.8 s | 51.8 / 49.7 dB |

Full pipeline (shape + paint) on one prop, the deployed binary against the round-3 build: 40,000 faces both, mean
nearest-vertex distance 1.84e-3 of the bounding-box diagonal, the same as two runs of the deployed binary (1.84e-3).
Deployed 2026-11-20 00:22 (`xpu_sdpa/xpu_hunyuan3d`, previous binary `xpu_hunyuan3d.prev`), agreed with the LivingRoom
session. `HY_GN_V2=0` and `HY_ESRGAN_FUSED=0` switch the two big changes off without a rebuild.

## Round 4 (2026-11-20 night): the LoRA term inside its consumers

There is no fast hand-written int8 GEMM to extend here (glm.cpp's W8A8 prompt path calls oneDNN too; its ESIMD DPAS
kernels decode trellis weights for a few rows), so oneDNN's int8 GEMM stays as it was: it still leaves u = x A^T in
the 64 columns after each output. The term u B^T now goes into the kernel that reads that output anyway, instead of
a separate oneDNN pass that read and wrote every qkv / fc1 output once more. Both kernels are ESIMD in a new
library, `xpu_sdpa/libxpu_h3_esimd.so` (`xpu_h3_esimd.cpp`, built by `build_native.sh` like `libxpu_moss_fa.so`).

- **qkv** (`qkv_lora_norm_rope`): a thread takes 16 rows of one head, u as DPAS A tiles in registers, B^T in VNNI
  2-D block loads shared through L1 by the 8 threads of a work-group (bf16 DPAS, f32 sums), then per row: add, round
  to bf16, q / k RMSNorm and RoPE, v copy. Same rounding points as the old path.
- **fc1** (`fc1_lora_swiglu_quant`): the row's int8 scale needs the max over all 56 rotation groups of a row, more
  than a thread can hold, so it runs twice: pass 1 writes each (row, group) max, pass 2 recomputes the LoRA term,
  SwiGLU and the 256-point Hadamard (in `hadamard256`'s exact operation order) and quantizes. 9 bytes of traffic per
  output element instead of 13, but both passes run at ~320 GB/s against oneDNN's ~495, so it gains little.
- `SD_XPU_H3_LORA_FUSE=0` restores the separate pass for both, `SD_XPU_H3_LORA_FUSE_FC1=0` for fc1 only;
  `SD_XPU_H3_LORA_FUSE_CHECK=1` compares the first chunk of each with the separate-pass path and prints the result.
  Only 64 LoRA rows (two rank-32 LoRAs, make_real's setup), head_dim 128 and RoPE halves of 32 / 48 / 64 take the fused
  path; anything else (H3-Image without LoRAs, other ranks) runs the old code.

**Pitfall:** the first version used SYCL `joint_matrix` and died on its first call: "no matrix hardware on the target
device". oneAPI 2025.3's runtime has no matrix table for Battlemage G31 (the B70) and refuses every joint_matrix
kernel, AOT or not. DPAS on this card goes through ESIMD `xmx::dpas`, which needs the vector backend and so its own
library (it drops ordinary SYCL kernels' names), as `libxpu_moss_fa.so` and glm.cpp's `libglm_esimd.so` already do.

The 39-frame 1920x1088 unit, make_real's depth engine with two rank-32 LoRAs, one library with the switch off and on,
back to back (per step; the categories bracket each kernel with queue waits):

| | separate LoRA pass | fused |
|---|---:|---:|
| qkv GEMM (+ the separate pass) | 2.09-2.14 s | 1.83-1.85 s |
| q / k norm + RoPE + v | 0.52-0.53 s | 0.35 s |
| fc1 GEMM (+ the separate pass) | 2.84-2.91 s | 2.46-2.49 s |
| SwiGLU + rotate + int8 | 0.35-0.37 s | 0.66-0.67 s |
| 50 blocks, a steady step | 17.26-17.69 s | 16.88-17.08 s |
| sampler, steps 2-8 (8-step render) | 18.04-18.09 s/it | 17.46-17.55 s/it (-3.1%) |

qkv gains 0.43 s a step, fc1 0.07 s. The kernels' own checks against the separate pass on a real 8192-row chunk:
q / k relative RMS 1.6e-5, v identical; fc1 int8 activations 0.0001% off by one, row scales within 2.2e-7. The 8-step
render against the deployed library: 2.10 levels mean, 35.3 dB PSNR, SSIM 0.973, audio 21.1 dB, where two runs of the
deployed library differ by 2.28 levels, 35.8 dB, SSIM 0.970, audio 24.2 dB: inside run-to-run noise. The LoRA passes
grow with the token count and attention faster, so at 107 frames the saving should be ~2% of a ~70 s step (scaled,
not measured; see the production check below).

Tried and dropped: 2-D prefetches of each thread's tiles at kernel start (into L2, or L1 + L2) made both kernels
slower (fc1 0.66 -> 0.88-0.90 s, qkv 0.35 -> 0.38-0.39 s a step).

### H3-Image was no longer reproducible: fixed

Checking the new library on H3-Image product shots (no LoRA, so the fused paths stay off) found the pictures differing
from run to run with *either* library: two runs of the deployed one 6.3 / 2.5 levels apart (PSNR 24-26 dB), two runs
of the new one 8.6 / 3.1, every old-vs-new pair in the same range. Round 1's int8 attention caused it: its K / V
channel sums (`xpu_attn8.hpp` `stats`) were float atomic adds across 256-token blocks, so the means moved in their
last bits with the finishing order, some int8 values flipped, and the distilled 6-step model turned that into a
different picture (round 1 had found these shots bit-identical with the bf16 attention before it). The block sums
now go to a small buffer and are added in block order. Two runs of the fixed library: byte-identical pictures; the
int8 attention checks are unchanged (3.7-3.9e-3 image, 4.9e-3 video) and so is its speed (9.03 s a step, 168 TOPS).
H3 video has other run-to-run variation (bf16 runs differed before int8 attention), so it stays non-reproducible.
`hunyuan3d/hy_attn8.hpp`, `hy_conv.hpp`'s GroupNorm and `xpu_moss_kernels.hpp`'s V stats use the same float
atomics (Hunyuan3D's paint already varied 0.36-0.53 levels run to run before round 3); not changed here.

Deployed 2026-11-20 03:22 (inside a queued job, after the checks above): `xpu_sdpa/libxpu_h3.so` (fused LoRA term +
deterministic stats) and the new `xpu_sdpa/libxpu_h3_esimd.so`; the previous library is `libxpu_h3.so.prev4`. The
first production unit on it (make_real scene-43, 90 frames, 03:34) ran clean; its steps swung 49-60 s/it and no
90-frame unit ran before the swap, so there is no like-for-like production timing yet (the old library's 73-frame
units that night also drifted from 40 to 51 s/it within a unit: something else was loading the card). LivingRoom's
H3-Image stages on it ran clean too (04:45-05:07: 12 sheets, 14 product shots, all saved).

### Hunyuan3D paint-stage linears: int8 gains nothing

`--int8 paint` (W8A8 linears) on the two production props, paint only: the 10 denoising steps take the same time (10.1
vs 10.2 s, 9.0 vs 9.1 s), only the one-off weight upload is shorter (3.3 -> 1.0 s, 1.8 GiB less resident), and the
textures move 1.0-1.4 levels (38.2-44.5 dB). Not proposed to LivingRoom.

## Rollback copies (2026-11-20)

All next to the live files; nothing deleted. "Needed" = the one step back from what runs now.

| File | What it is | Needed for rollback? |
|---|---|---|
| `build-sycl/bin/sd-cli.prev` | the original sd-cli (Nov 10, dirty dc210b1) | only to undo everything at once |
| `build-sycl/bin/sd-cli.prev2` | round 1's sd-cli (HEAD + tile merge, buffered loader) | **yes**: undoes the O_DIRECT loader |
| `xpu_sdpa/libxpu_h3.so.prev` | the original H3 library (Oct 22, bf16 attention) | only to undo everything at once |
| `xpu_sdpa/libxpu_h3.so.prev2` | round 1's library (int8 attention, LoRA in place, scratch cache) | no: `.prev3` is the later step back |
| `xpu_sdpa/libxpu_h3.so.prev3` | round 2's library (+ parallel reader), before the RoPE / gated-add kernels | **yes** |
| `xpu_sdpa/libxpu_h3vae.so.prev` | the original VAE library (Oct 22) | only to undo the scratch cache too |
| `xpu_sdpa/libxpu_h3vae.so.prev2` | round 1's VAE library (scratch cache), before the parallel reader | **yes** |
| `xpu_sdpa/libxpu_qwen21.so.prev` | an older Qwen library (Oct 22) that was there before this work | not mine; ask whoever made it |
| `xpu_sdpa/libxpu_qwen21.so.prev2` | the Qwen library before run-time LoRA (Oct 25) | **yes** |
| `xpu_sdpa/xpu_hunyuan3d.prev` | the Hunyuan3D binary before round 3 (Nov 18) | **yes** |
| `xpu_sdpa/libxpu_h3.so.prev4` | the H3 library deployed 2026-11-19 22:58 (RoPE / gated-add kernels), before the fused LoRA term | **yes** (or `SD_XPU_H3_LORA_FUSE=0`) |

Stale once the current builds have run in production for a while: `sd-cli.prev`, `libxpu_h3.so.prev`,
`libxpu_h3.so.prev2`, `libxpu_h3vae.so.prev` (~310 MB together, almost all of it `sd-cli.prev`), and since round 4
`libxpu_h3.so.prev3`. `libxpu_h3_esimd.so` is new in round 4 (the old library does not load it), so going back to
`.prev4` needs nothing else.
