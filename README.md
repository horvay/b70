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
