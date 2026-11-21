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
engines/llama.cpp   horvay/atomic-llama-cpp-turboquant (the Atomic fork), built for SYCL with
                    build_sycl.sh: `b70 serve` runs GGUF folders on its llama-server
engines/onednn      the local oneDNN v3.13.4 patch and build.sh for libdnnl_h3 (~/opt/onednn-sycl-graph), which every
                    SYCL binary here links
llm/                LLM servers: Gemma 4, Qwen3.8-Flash-Next, GLM-5.3-Flash, DeepSeek-V4.1-Flash (the `b70` command)
tts/                Fish S2 Pro and Qwen3-TTS on torch-XPU (older projects)
```

## What runs here

| Job | Command | Engine |
|---|---|---|
| MiniMax H3 video (Ref2VA, guides, ControlNet, multi-shot batches) | `tools/h3-ref2va` | `engines/sd.cpp/build-sycl/bin/sd-cli` + `xpu_sdpa/libxpu_h3.so` (int8 W8A8 blocks, int8 attention by default since 2026-11-19: `SD_XPU_H3_SDPA=graph` for bf16; since 2026-11-20 the LoRA term of qkv / fc1 inside their consumer kernels, `libxpu_h3_esimd.so`, -3% a step: `SD_XPU_H3_LORA_FUSE=0` for the old pass), `libxpu_h3vae.so` |
| Qwen Image 2.1 stills (Noct Q Anime) | `tools/b70-image.sh -m qwen --xpu`, `--batch` | sd-cli + `xpu_sdpa/libxpu_qwen21.so` |
| MOSS-Image stills (Base, Dirty-Moss; Turbo LoRA, 1.5x hires; `(word:1.3)` prompt weights) | `tools/b70-image.sh -m moss\|dirty-moss [--turbo] [--hires] [--xpu]`, `--batch` | `engines/sd.cpp-moss` sd-cli (upstream MOSS support, branch `moss-merge`); `--xpu`: + `xpu_sdpa/libxpu_moss.so` |
| MiniMax-H3-Image stills and edits (33B, 6 steps, 4-16 MP, up to 16 `<Picture N>` references) | `tools/b70-image.sh -m h3`, `--batch` (one process: `sd-cli --multi-jobs`) | `engines/sd.cpp-h3img` sd-cli (upstream master with H3-Image, branch `h3-image`; its ggml on branch `b70-sycl` carries the B70 SYCL patches that still apply, the pool-cache release and the oneDNN f16 matmul) + `xpu_sdpa/libxpu_h3.so` int8 blocks (int8 attention since 2026-11-19: 2048x2048 product shots 2.81 -> 2.44 s a step; it moves details, `SD_XPU_H3_SDPA=graph` gives the old pictures bit for bit). 4 MP ~1.2 s/step, 16 MP ~7.8 s/step; ~1.5 / 2.3 min per image with loading, text encoder and VAE; `--batch` encodes every prompt and reference first, then samples and decodes with the blocks resident (product shots 62 -> 46 s each), an edit's reference encode runs as 2D convolutions (11.5 -> 3.8 s a picture: the causal 3D convolutions of one frame only ever met zero padding with two of their three taps), and the text encoder's 16 GB stay staged on the card for the whole conditioning pass instead of being copied for each job (`SD_MULTI_TE_KEEP=0` turns that off) |
| MiniMax Music 3 songs | `engines/sd.cpp/xpu_sdpa/xpu_music3` (one song), `tools/music3/batch JOBS.jsonl --out DIR --one-chunk` (8 songs a queue chunk); prompting and instrumentals: `engines/sd.cpp/xpu_sdpa/MUSIC3_PROMPTING.md`; vocal/clipping/duplicate checks on the 3080: `tools/music3/qc` | standalone SYCL + oneDNN |
| Fish Audio Drama 3 voice | `tools/drama3_clone <jobs.tsv> --out DIR [--submit]` | `engines/audio.cpp/.../drama3_batch` |
| Pixal3D image -> 3D mesh (GLB, per-vertex colour) | `tools/pixal3d IMAGE -o OUT.glb [--res 512\|1024\|1536] [--seed N] [--fov RAD] [--decimate FACES] [--no-tex]` | `engines/sd.cpp/xpu_sdpa/xpu_pixal3d` (SYCL + oneDNN) |
| CG-MLLM 4B image -> 3D mesh (GLB, shape only) | `tools/cgmllm IMAGE -o OUT.glb [--seed N] [--steps N] [--cfg X] [--octree N] [--decimate FACES]` | `engines/sd.cpp/xpu_sdpa/xpu_cgmllm` (SYCL + oneDNN) |
| Hunyuan3D 2.1 image -> textured 3D mesh (GLB: 40k triangles, UVs, base colour + metallic-roughness textures) | `tools/hunyuan3d IMAGE -o OUT.glb [--seed N] [--faces N] [--octree 256\|384\|512] [--paint-steps N] [--no-paint] [--png] [--keep DIR]` | `engines/sd.cpp/xpu_sdpa/xpu_hunyuan3d` (SYCL + oneDNN) |
| Hunyuan3D-2mv views (front, left, back, right) -> textured 3D mesh (2mv shape + 2.1 paint) | `tools/hunyuan3d REF.png --mv F.png,L.png,B.png,R.png [--mv-model turbo\|mv] [--steps N] -o OUT.glb`; many props: `xpu_hunyuan3d --batch FILE` | same binary (`hunyuan3d/hy_dit_mv.hpp`) |
| LLM servers | `llm/b70 serve <model>` | SGLang / exllamav3 ports in a bubblewrap image; engines/glm.cpp; GGUF folders on engines/llama.cpp |
| Fish S2 Pro, Qwen3-TTS | `tts/` | torch 2.11 XPU venvs |

Pixal3D: about 40 s per 1024 mesh (50 s with texture, 35 s for a full body at 1536). IMAGE is RGBA with a real mask
or a studio shot on a plain neutral-grey backdrop (keyed out without a model; BiRefNet and the MoGe-2 FOV estimate are
not ported yet, so `--fov` sets the camera, default 0.4 rad). Output is in glTF's frame (Y up, facing +Z), with
consistent winding, small holes closed and detached pieces under 10% of the main body dropped; `--decimate` also
writes `OUT_full.glb`. Validation tools and notes: `engines/sd.cpp/xpu_sdpa/pixal3d/` (`ref/`).

CG-MLLM (JreamH/CGMLLM-4B-i2o, 8.5 B parameters: Qwen3-VL-4B doubled into two experts, Hunyuan3D-2.1's shape decoder):
about 60 s per mesh (9 s weights, 41 s for 49 sampling steps = 90 model calls, 10 s for the surface on a 512 grid);
upstream's code on torch XPU takes 68 s for the sampling alone. 14.4 GiB of weights on the card. IMAGE as for
Pixal3D (RGBA with a real mask, or a plain-backdrop studio shot). Against upstream each stage agrees to about 1%
(bf16 rounding) and the final surfaces to 0.03 cells of a 128 grid, the same as two runs of the engine differ.
Reference and check tools: `engines/sd.cpp/xpu_sdpa/cgmllm/ref/`.

Hunyuan3D 2.1 (tencent/Hunyuan3D-2.1, shape and PBR paint models): about 96 s per prop with textures. Shape 54 s
(2 s weights and image encoder, 20 s for 49 sampling steps, 31 s for the field on a 385^3 grid, 1 s for marching cubes
and the reduction to 40,000 faces); paint 42 s (4 s weights, 1 s condition, 1 s reference UNet, 19 s for 15 steps over
6 views x 2 materials x 3 guidance batches, 2 s VAE decode, 4 s Real-ESRGAN 4x, 2.6 s UV unwrap, 2 s view rendering,
baking and inpainting on the CPU). Upstream's code on torch XPU takes 74 s for the shape sampling, 28 s for the paint
sampling and 22 s for the upscale. Against upstream, single model calls agree to fp16 rounding (shape: 5e-4 relative;
paint: 1-3%), the shape's field to 1.5% after the whole chain, the painted views to 1-4 of 255 levels. Output in
glTF's frame (Y up, front +Z), textures 2048 JPEG. Licence: Tencent Hunyuan 3D 2.1 Community License (not valid in the
EU, UK and South Korea; outputs may not be used to train other AI models; over 1 M monthly users needs Tencent's
permission). Check tools: `engines/sd.cpp/xpu_sdpa/hunyuan3d/` (`ref/`, `tex_test.cpp`).

Hunyuan3D-2mv (tencent/Hunyuan3D-2mv, multi-view shape model on Hunyuan3D 2.0's flow DiT: 16 double- and 32
single-stream blocks of 1024, DINOv2-giant tokens of up to four views (front, left, back, right) with a sin/cos view
embedding): `xpu_hunyuan3d --mv` reuses the engine's DINOv2, shape decoder (the 2.0 VAE, 3072 latents), field and mesh
code and textures the result with the 2.1 paint stage. Base model, 30 steps with CFG 5 (upstream's README): 16-20 s of
sampling (0.55-0.7 s a step; upstream on torch XPU 0.67); turbo (the default), 5 steps with the guidance embedding:
1.5-1.8 s, shapes indistinguishable from the base model's on the 8 test props. `--batch FILE` (one prop per line) runs
every prop of a list in one process, stage by stage, so the shape DiT, the decoder and the paint models load once and
the UV unwraps run on a CPU thread meanwhile: 2.1 props 58 s each instead of 72 s, 2mv base + paint 15 steps 64 s,
turbo + paint 10 steps about 39 s, 37.7 s with `--attn8 field,paint` (int8 attention, the MOSS kernel at head
dim 64: `xpu_hy_fa.cpp`, `hunyuan3d/hy_attn8.hpp`; paint steps 13.8 -> 11.5 s, texture PSNR vs f16 44-48 dB where two
f16 runs differ by 48-51; field SDF 0.2-0.6%; `xpu_hunyuan3d --attn8-test` checks the kernel). Since 2026-11-20 the paint stage runs ~19% faster (GroupNorm without
per-element 64-bit divisions, Real-ESRGAN's LeakyReLU and residual adds inside its convolutions, the UNet's residual adds
folded into its GEMMs): 26.4-27.5 -> 21.8 s a prop, textures within run-to-run noise; `HY_GN_V2=0` / `HY_ESRGAN_FUSED=0`
switch the first two off ([survey](docs/speed-survey-nonllm.md)). The 2mv DiT stays f16:
int8 attention there changed the SDF 1-5% for 0.3 s a prop. `--field-pre 4 --field-margin 2` runs the field's coarse pass only near the surface a 129^3 pre-pass finds: 8.5-10.9 -> 2.3-5.9 s, marching-cubes face counts within 0.005% on 12 props with thin poles and rings (a part thinner than ~2% of the object lying wholly between pre-pass nodes could be missed). Keep `--octree 512` (coarse to fine, 11 s); 384 runs dense (28 s). Against upstream's code on torch XPU (same views and noise): single model calls
1e-3 relative at the ends of the schedule, 1-3% in the middle (a few very large activation channels), the 128^3 field
3.1% (base, 50 steps) and 1.8% (turbo) after the whole chain. Weights used as published (fp16 safetensors, 4.9 GB a
variant): `hunyuan3d/fetch_models.sh --mv [turbo|mv]` (turbo by default; only turbo is kept in the model store); check:
`hunyuan3d/ref/ref_shape_mv.py` (needs the base model and a clone of Tencent-Hunyuan/Hunyuan3D-2). Licence: Tencent Hunyuan
3D 2.0 Community License (same territory and user limits as 2.1). Upstream's timestep embedding passes its time
factor as `max_period` (frequencies down to 1/1000, not FLUX's 1/10000); a port that "fixes" that matches at t = 0
and drifts from the first step on.

Every job takes the card through the queue: `b70-lease run <owner> -- <cmd>` for a job you wait on,
`b70q submit <owner> -- <cmd>` for one that should outlive your shell, `b70q serve <owner> --idle 15m -- <cmd>` for a
server, `--standby` for the card's idle model (loaded whenever nothing else wants the card). The tools above do this
themselves. See `queue/README.md`.

H3 logs show `[xpu_h3] LoRA ... on 200 linears`, `control: ...` and `loaded 50 blocks` only with `SD_XPU_H3_VERBOSE=1`.
The LoRA and control apply either way; set it when a run must prove they loaded.

H3-Image is not bit-reproducible on the B70: the same jobs with the same seeds, run twice by the same sd-cli, differ
by 1.8-6.3 levels (of 255) mean per image (2026-11-18, 8 turnaround sheets). The pictures look the same; edges and
fine texture move. The noise and the reference latents are deterministic (the reference VAE encode samples with a
fixed `MT19937RNG(42)`), so the spread most likely comes from GPU reductions whose order varies (atomics, split-K
GEMMs). Not chased. To check that a change keeps H3's output, compare it with a second run of the old path, not with
a zero difference. (2026-11-20: product shots, 2048x2048, are byte-identical run to run again after the int8
attention's float-atomic K / V sums were made ordered; sheets not re-measured. See `docs/speed-survey-nonllm.md`.)

## How much faster it got

Each engine started as a stock build or a first working port on the B70 (Music 3 had no B70 version at all) and was
then tuned on the card. The table compares that starting point with today; the linked notes have the steps in between and the quality checks.
`s/s` is seconds of B70 time per second of audio generated (lower is better).

| Engine | Starting point on the B70 | Now | Gain |
|---|---|---|---|
| **MiniMax H3** video, Ref2VA ([notes](https://github.com/horvay/stable-diffusion.cpp/blob/master/docs/b70_xpu.md)) | stock sd.cpp on Vulkan: "device lost" on the first step. First working Vulkan port: ~570 s per denoising step at 1344x768 and 107 frames, 42 min per clip | ~71 s per step and ~9 min per clip with the transformer blocks on the XPU; the XPU video decoder and int8 blocks then took a shot from 238.7 s to 133.0 s; int8 attention, the in-place LoRA term, scratch caches and a cache-friendly tile merge then took a 1920x1088x73 shot's step from 60.3 to 36.7 s and its VAE decode from 90.8 to 38.0 s ([survey](docs/speed-survey-nonllm.md)) | **8x** per step, **4.7x** per clip, then **1.8x** and **1.6x** more |
| **Qwen Image 2.1** stills ([notes](https://github.com/horvay/stable-diffusion.cpp/blob/master/docs/b70_xpu.md)) | sd.cpp on Vulkan: 8.4 s per step, 226 s per 1024x1536 image (25 steps, CFG 3) | 1.4 s per step, ~43 s per image | **5.3x** |
| **MOSS-Image** stills ([notes](https://github.com/horvay/stable-diffusion.cpp/blob/master/docs/b70_xpu.md)) | upstream sd.cpp on ggml SYCL, Q8_0: 2.26 s per step, 64 s per 1024x1536 image (25 steps, CFG 3) | 1.03 s per step, 31 s per image, int8 blocks and int8 attention on the XPU (`--xpu`) | **2.2x** (1.8x with Turbo + LoRA + hires) |
| **MiniMax Music 3** songs ([notes](engines/sd.cpp/xpu_sdpa/MUSIC3_PROMPTING.md)) | ran only on an RTX 3080 Ti, through ComfyUI and torch | one song: 1.4 s/s; 8 songs batched (int8 KV, rewritten decode attention, batched + fused DiT, 25 % window overlap, CFG on 2/3 of the steps): 0.47 s/s, ~58 two-minute songs an hour | new on the B70; **3x** over one at a time, **1.9x** over the first batched version (0.88 s/s) |
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
engines/sd.cpp/xpu_sdpa/build_native.sh      # libxpu_h3.so (+ libxpu_moss_fa.so, libxpu_h3_esimd.so), libxpu_h3vae.so, libxpu_qwen21.so, xpu_selftest
engines/sd.cpp/xpu_sdpa/build_music3.sh      # xpu_music3 (after sd-cli)
engines/sd.cpp/xpu_sdpa/build_pixal3d.sh     # xpu_pixal3d (weights: xpu_sdpa/pixal3d/fetch_models.sh [--tex])
engines/sd.cpp/xpu_sdpa/build_cgmllm.sh      # xpu_cgmllm (weights: xpu_sdpa/cgmllm/fetch_models.sh)
engines/sd.cpp/xpu_sdpa/build_hunyuan3d.sh   # xpu_hunyuan3d (weights: xpu_sdpa/hunyuan3d/fetch_models.sh)
engines/audio.cpp/build_sycl.sh              # drama3_batch -> engines/audio.cpp/build/linux-sycl-release/bin/
(cd queue && bun install)                    # then install queue/systemd/b70q.service and queue/bin/* (queue/README.md)
```

## License

MIT (`LICENSE`), except where a folder says otherwise: the engines keep their upstream licenses, and `tts/fish/` is under the Fish Audio Research License (`tts/LICENSE-FISH-AUDIO.md`).
