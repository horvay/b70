# b70

Everything that runs on the workstation's **Intel Arc Pro B70** (Battlemage, 32 GB): the engines, the scripts that
drive them, and the queue that shares the card. Intel's own stack only: SYCL, oneDNN, oneMKL and Level Zero. No
Vulkan; torch-XPU only in the two older TTS pipelines under `tts/`.

Models live outside the repo, in `~/models/<category>/<model>` (`~/models/INDEX.md`, managed by `tools/mstore.ts`).

## Layout

```text
queue/              b70q: the FIFO queue for the card, its dashboard (http://127.0.0.1:7870/) and b70-lease
tools/              the commands below, plus mstore.ts (the model store)
engines/sd.cpp      stable-diffusion.cpp fork (submodule, with the ggml fork): the one sd-cli for every image, video
                    and music job, and the native XPU libraries in xpu_sdpa/
engines/audio.cpp   audio.cpp fork (submodule): Fish Audio Drama 3 (drama3_batch)
engines/onednn      the local oneDNN v3.13.4 patch and build.sh for libdnnl_h3 (~/opt/onednn-sycl-graph), which every
                    SYCL binary here links
llm/                LLM servers: Gemma 4, Qwen3.8-Flash-Next, GLM-5.3-Flash, DeepSeek-V4.1-Flash (the `b70` command)
tts/                Fish S2 Pro and Qwen3-TTS on torch-XPU (older projects)
```

## What runs here

| Job | Command | Engine |
|---|---|---|
| MiniMax H3 video (Ref2VA, guides, ControlNet, multi-shot batches) | `tools/h3-ref2va` | `engines/sd.cpp/build-sycl/bin/sd-cli` + `xpu_sdpa/libxpu_h3.so`, `libxpu_h3vae.so` |
| Qwen Image 2.1 stills (Noct Q Anime) | `tools/b70-image.sh -m qwen --xpu`, `--batch` | sd-cli + `xpu_sdpa/libxpu_qwen21.so` |
| MiniMax Music 3 songs | `engines/sd.cpp/xpu_sdpa/xpu_music3` | standalone SYCL + oneDNN |
| Fish Audio Drama 3 voice | `tools/drama3_clone <jobs.tsv> --out DIR [--submit]` | `engines/audio.cpp/.../drama3_batch` |
| LLM servers | `llm/b70 serve <model>` | SGLang / exllamav3 ports in a bubblewrap image |
| Fish S2 Pro, Qwen3-TTS | `tts/` | torch 2.11 XPU venvs |

Every job takes the card through the queue: `b70-lease run <owner> -- <cmd>` for a job you wait on,
`b70q submit <owner> -- <cmd>` for one that should outlive your shell, `b70q serve <owner> --idle 15m -- <cmd>` for a
server. The tools above do this themselves. See `queue/README.md`.

## How much faster it got

Each engine started as a stock build or a first working port on the B70 (Music 3 had no B70 version at all) and was
then tuned on the card. The table compares that starting point with today; the linked notes have the steps in between and the quality checks.
`s/s` is seconds of B70 time per second of audio generated (lower is better).

| Engine | Starting point on the B70 | Now | Gain |
|---|---|---|---|
| **MiniMax H3** video, Ref2VA ([notes](https://github.com/horvay/stable-diffusion.cpp/blob/master/docs/b70_xpu.md)) | stock sd.cpp on Vulkan: "device lost" on the first step. First working Vulkan port: ~570 s per denoising step at 1344x768 and 107 frames, 42 min per clip | ~71 s per step and ~9 min per clip with the transformer blocks on the XPU; the XPU video decoder and int8 blocks then took a shot from 238.7 s to 133.0 s | **8x** per step, **4.7x** per clip, then **1.8x** more |
| **Qwen Image 2.1** stills ([notes](https://github.com/horvay/stable-diffusion.cpp/blob/master/docs/b70_xpu.md)) | sd.cpp on Vulkan: 8.4 s per step, 226 s per 1024x1536 image (25 steps, CFG 3) | 1.4 s per step, ~43 s per image | **5.3x** |
| **MiniMax Music 3** songs | ran only on an RTX 3080 Ti, through ComfyUI and torch | 70 s of stereo 44.1 kHz audio in 97 s, native SYCL + oneDNN | new on the B70; no like-for-like baseline |
| **Fish Audio Drama 3** voice cloning ([notes](bench/drama3/OPTIMIZE.md)) | stock audio.cpp CLI on Vulkan, one line at a time: 0.875 s/s | 0.132 s/s, 16 lines at a time on SYCL | **6.6x** |
| **Fish S2 Pro** voice cloning ([notes](tts/README.md)) | stock fish-speech, bf16: 4.32 s/s (an RTX 3080 Ti: ~3.0) | 0.21 s/s, 32 lines at a time | **21x** (14x the 3080 Ti) |
| **Qwen3-TTS 1.7B** ([notes](tts/README.md)) | stock qwen-tts: 2.33 s/s one line at a time, 0.39 at batch 8 | 0.050 s/s, 32 lines at a time | **47x** (8x stock's batch 8) |
| **DeepSeek-V4.1-Flash** 262B, ~305 GB on disk ([notes](llm/dsv41/README.md)) | first working version: 3.7 tok/s; a 40K-token prompt in 632 s | 11-15 tok/s; 40K prompt in 354 s | **3-4x** decode, 1.8x prompt |
| **GLM-5.3-Flash** ~105 GB ([notes](llm/glm/README.md)) | first working version: 5.2 tok/s; 2K prompt ~40 s; 8K out of memory | 15-18 tok/s; 2K prompt 9.0 s; 8K 14.4 s | **~3x** decode, 4.4x prompt |
| **Qwen3.8-Flash-Next** 68 GB ([notes](llm/flashnext/README.md)) | stock SGLang on XPU (eager): ~6.5 tok/s | 41.7 tok/s; 89-100 tok/s in total for 4 users; 18K prompt in 15.9 s | **6.4x** |
| **Gemma 4 31B** ([notes](llm/gemma/README.md)) | off the shelf, llama.cpp SYCL Q4_K_M: ~25 tok/s; 12K prompt at 930 tok/s | 42-45 tok/s with its drafter, 135 in total for 4 users; 12K prompt in 6.5 s | **1.7x** decode, 2x prompt |

**Video and stills.** Stock sd.cpp couldn't run H3 on the B70 at all: its Vulkan flash attention fell back to a scalar
shader whose dispatches outlived the xe driver's 5 s job timeout. Porting Intel's prefill flash attention (3.5 -> ~18
TFLOPS) and chunking the MLP and QKV under Vulkan's 4 GiB buffer limit got it running; moving the transformer blocks to
the XPU (oneDNN bf16 matmuls at ~100-140 TFLOPS, attention ~95) made each step 8x faster
([horvay/stable-diffusion.cpp@e498bba](https://github.com/horvay/stable-diffusion.cpp/commit/e498bba)). Since then the
blocks run int8 W8A8 GEMMs with the rotate-and-quantize step fused into the kernels before them, the video decoder runs
on the XPU (VAE decode 91.5 -> 22.3 s per shot), the blocks dropped libtorch for native SYCL + oneDNN, and the rest of
sd-cli moved from Vulkan to ggml's SYCL backend (a small check job 38.7 -> 25.6 s). Qwen Image uses the same int8 block
library: a model call went from 1.18 s in bf16 to 0.67 s, and VRAM from 22.7 to 15.8 GiB.

**Voice.** Drama 3's stock CLI ran 16 separate graphs per audio frame, each with a blocking readback. One graph per step
for a whole batch of lines took it from 0.875 to 0.286 s/s; codec decode graphs cached per length bucket and wider
mat-vec kernels brought Vulkan to 0.159. The SYCL backend then put the 2-32-row matmuls on the matrix engines (ESIMD
DPAS), fused the MoE expert mat-vec into one launch and moved the codec's transposed convolution to oneMKL: 75.1 ->
51.0 ms per step and 0.132 s/s ([horvay/audio.cpp@3bf3607](https://github.com/horvay/audio.cpp/commit/3bf36079)). Word
error rate went from 1.2% to 0.7% and speaker similarity from 0.721 to 0.752. (The stock figure is from a 20-line eval
set, the final one from a 256-line benchmark.) S2 Pro and Qwen3-TTS gained mostly from XPU graphs per frame plus
batching, with WER still 0% for S2 Pro.

**LLMs.** The three MoE models are two to ten times the card's 32 GB, so their experts are split across VRAM, RAM and
NVMe, with the most-used experts pinned in VRAM. Qwen3.8-Flash-Next gained from decode graphs (~6.5 -> ~29 tok/s), a
profiled hot-expert tier (-> ~34) and fused router and norm kernels (-> 41.7), while its host RAM fell from 45.8 GB
pinned (the upstream B70 setup peaks at 67 GB, more than this machine has) to ~15 GB. GLM-5.3-Flash and DeepSeek-V4.1
are ports of exllamav3 to XPU: GLM gained from routing without host syncs, attention in XPU graphs and one-pass prompts
(8K now fits); DeepSeek from fused MoE, router and hyper-connection kernels, an asynchronous expert pager and DSpark
speculative decoding. Each one's notes report its agreement with the reference model (top-1 and KL divergence).

## Building

All builds use DPC++/oneMKL 2025.3 and a SYCL oneDNN build from `~/opt` (paths are compiled into the binaries) and
compile kernels ahead of time for BMG G31. They run on the CPU only; keep parallelism low on a shared machine.

```sh
git submodule update --init --recursive engines/sd.cpp engines/audio.cpp
engines/sd.cpp/build_sycl.sh                 # sd-cli -> engines/sd.cpp/build-sycl/bin/sd-cli
engines/sd.cpp/xpu_sdpa/build_native.sh      # libxpu_h3.so, libxpu_h3vae.so, libxpu_qwen21.so, xpu_selftest
engines/sd.cpp/xpu_sdpa/build_music3.sh      # xpu_music3 (after sd-cli)
engines/audio.cpp/build_sycl.sh              # drama3_batch -> engines/audio.cpp/build/linux-sycl-release/bin/
(cd queue && bun install)                    # then install queue/systemd/b70q.service and queue/bin/* (queue/README.md)
```

## License

MIT (`LICENSE`), except where a folder says otherwise: the engines keep their upstream licenses, and `tts/fish/` is under the Fish Audio Research License (`tts/LICENSE-FISH-AUDIO.md`).
