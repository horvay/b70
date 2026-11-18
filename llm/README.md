# EXL3 models on an Intel Arc Pro B70

Serving EXL3 models on one **Intel Arc Pro B70** (Battlemage, 32 GB) in a desktop with 62 GB of RAM and a PCIe 4 NVMe
SSD, including mixture-of-experts models far larger than the card: the experts a workload uses most stay on the card,
more sit in pinned RAM, and the rest come from an NVMe store; the router is nudged towards experts that are close by,
at a measured cost in quality.

The biggest is **DeepSeek-V4.1-Flash-1017** (262B parameters, vcruz305's
[SAGE EXL3 1.62 bpw pack](https://huggingface.co/vcruz305/DSV4.1-Flash-1017-SAGE-EXL3-1.62bpw)): ~305 GB on disk with its
Engram tables, nearly ten times the card's memory, at 11-15 tok/s.

| Model | Size | Decode, 1 user | Prompt | Host RAM | Server |
|---|---|---|---|---|---|
| **[DeepSeek-V4.1-Flash-1017](dsv41/README.md) 1.62 bpw SAGE EXL3 (262B, the largest)** | ~305 GB (116 GB + 189 GB Engram tables) | ~11-15 tok/s with DSpark | 512 tokens in ~20 s; 40K in ~6 min | ~30 GB | exllamav3 port (`dsv41/`) |
| [GLM-5.3-Flash](glm/README.md) 2.51 bpw EXL3 | ~105 GB | ~15-18 tok/s | 8K tokens in 14.4 s | ~32 GB | exllamav3 port (`glm/`) |
| [Qwen3.8-Flash-Next](flashnext/README.md) 3.05 bpw EXL3 | 68 GB | 41.7 tok/s (68 total for 2 users, ~95 for 4) | 18K tokens in 15.9 s | ~15 GB | SGLang + trellis-serve + patch |
| [Ripple-65B](../engines/glm.cpp/README.md) 3.0 bpw EXL3 (dense, 65B), our own quant | 25 GB + 2.6 GB drafter | 14 tok/s plain; with its DFlash2 58 code, 43 reasoning, 20-24 prose; 59 total for 8 | 661 tok/s at 2K, 566 at 32K | ~65K context with the drafter, ~106K without | `engines/glm.cpp` |
| [Gemma 4 31B](../engines/glm.cpp/README.md) 5.0 bpw EXL3 (dense, Gemma 4 31B-based) | 24 GB | 35-49 tok/s with its drafter (87 total for 4, 101 for 8) | 8K tokens in ~6 s, 52K in 60 s; next turn 0.3-0.7 s; context ~150K, nothing packed | small | engines/glm.cpp (C++) |

No Docker and no root: every server runs inside an unpacked copy of the `lmsysorg/sglang` XPU image (PyTorch 2.13 XPU,
Triton XPU, oneAPI 2026.0) through bubblewrap.

## GGUF folders

A GGUF runs as is on `engines/llama.cpp`: horvay's Atomic fork of llama.cpp (the build the client runs on the 3080), compiled
for SYCL with `engines/llama.cpp/build_sycl.sh`. Atomic speaks the client's dialect itself (thinking prefill, `/control` for
"Answer now"), so the client passes requests through and only fills in sampling and the reasoning budget. A folder holds the
GGUF plus:

- `model.json`: `{"family": "gguf", "gguf": "<file>.gguf", "mtp": true, "bits": 4.95}` (`mtp`: the GGUF carries a NextN
  layer, served with `--spec-type draft-mtp`; `GGUF_MTP=0` turns it off)
- `config.json`: `model_type` (the client's family: `qwen3_5` for Qwen3.8), `hidden_size`, `vocab_size`
- `chat_template.jinja` (the GGUF's `tokenizer.chat_template`) and `generation_config.json` (its `general.sampling.*`)

Every layer goes on the B70 with flash attention, q8_0 K and V (`GGUF_CACHE_K`, `GGUF_CACHE_V`) and 2048-token batches
(`GGUF_BATCH`, `GGUF_UBATCH`; 512 read prompts 14% slower). Two settings the fork needs on the B70, set by default:
`ZES_ENABLE_SYSMAN=1`, and `GGML_SYCL_ENABLE_VMM=0` (the VMM pool asserts that frees come in reverse order, and Qwen3.8
with MTP breaks that at the first request). The build must find Level Zero (`build_sycl.sh` checks), or the device
memory query aborts at load.

First one: `~/models/llm/Qwen3.8-40B-Ultimate-Heretic-GGUF` (DavidAU's 96-layer Qwen3.8 grown from the 27B,
MTP-Q4_K_M, 24.2 GB). At the client's 84K context it takes 30.3 of 31.9 GiB and loads in ~30 s. Decode 24-27 tok/s with its
MTP layer (45-69% of drafts accepted); prompt 227 tok/s at 7.6K tokens, so a cold 30K prompt takes ~2.5 minutes
(EXL3 Gemma 31B on glm.cpp reads ~870 tok/s).

## Running it

You need Linux with the `xe` kernel driver, `bwrap` (bubblewrap), `git` and `python3` on the host, ~25 GB for the
image, the model, and NVMe space for the expert store (GLM ~95 GB, DSV4.1 ~105 GB, Flash-Next ~23 GB) on a filesystem where O_DIRECT
works (on btrfs, a `chattr +C` directory).

```sh
git clone https://github.com/horvay/b70 && cd b70/llm   # setup fetches the two exllamav3 submodules it needs
./b70 setup                              # once, CPU only: the image, kernels and plugins for every model
./b70 prepare /path/to/glm-model         # once per GLM or DSV4.1 model: pack its NVMe expert store (Flash-Next fills its own)
./b70 serve /path/to/model --port 8000   # OpenAI-compatible server (its turn on the B70 via b70q); Ctrl-C or `./b70 stop` stops it
```

`b70 serve` picks the server from the model's `config.json` (`gemma4`, `ricursive`, `qwen4_exp`, `glm5_next` or `deepseek_v41`) and
options: `--ctx` (context tokens), `--parallel` (requests at once; GLM and DSV4.1 serve one), `--name` (model name in the
API),
`--drafter` (Gemma: its assistant model; Ripple: its DFlash2), `--store` (expert store directory), `--chat-template` (a template to use
instead of the model's own), `--prepare` (run `prepare` first if needed). `./b70 info <model>` says whether setup and
prepare are done. The requested port opens only after one real warm-up request, so a client never meets a server that
is up but still compiling.

Who gets the B70 is up to the b70q queue (`../queue`) when its daemon runs: `serve` waits in line for the
card (and for the RAM it pins), printing its place every minute, then holds it as a server (`--queue OWNER`, default
`$B70Q_OWNER` or `b70`; `--idle DUR`, default 15m). A server idle on the GPU for that long is stopped by the queue
(SIGTERM) only while another job waits; start it again when it is needed. Without the daemon, `serve` just starts.

`--standby` makes the server the card's idle model instead: loaded whenever nothing else wants the B70. It is the
default for the queue owners listed in `STANDBY_OWNERS=` of `~/.config/b70/serve.env` (a machine's own setting, e.g.
`STANDBY_OWNERS=my-server`). The first load waits in line like any server (at the back, whatever the owner) and opens the
port once warm; after that the port stays open while the model comes and goes. Several standby servers can be up at
once, one per queue owner (a newer one replaces its owner's old one). The one loaded last is the current one: it is the
one that loads again when the card is free. The others stay unloaded until a request comes for one; that one then
loads and is the current one, so the card doesn't switch between them on its own.



- When another job waits and no request has run for a minute, the model unloads (its VRAM and pinned RAM freed) and the
  server waits at the back of the queue. A request in flight is never cut off: the server tells the queue whether it is
  in use.
- If it is the current one, it loads again once the card is free, then reads the last request's prompt again (one token,
  answered to nobody), so the next request finds it in the prompt cache (a cold 17K-token prompt takes DeepSeek-V4.1
  about 5 minutes).
- A request that comes while it is unloaded waits for it and puts the server at the back of the line like any job (no
  priority, `nq-…` owners included). A streamed one gets keep-alive chunks meanwhile (llama.cpp's `prompt_progress` with `processed: 0` when it
  asked for progress, SSE comments otherwise), so a client's first-token watchdog doesn't end it. `/v1/models` answers
  from the last load.

The minute is the queue's (`B70Q_STANDBY_QUIET_S`); it loads again as soon as nothing is queued (`B70Q_STANDBY_SETTLE_S`, default 0). SIGTERM stops a
standby server for good.

`serve` also guards the machine: it refuses to start with too little free RAM or beside a Steam game that has the B70
open, and stops the server at the first `xe` fault, hang or reset for the B70 in the kernel log (faulting kernels that
keep running have wedged the driver) or when RAM runs low or memory pressure gets high. The server runs with
`oom_score_adj` 1000, so the OOM killer takes it first.

State (image, overlays, caches, stores, logs) lives in `state/` (`B70_STATE` elsewhere; `B70_ROOTFS` reuses an image
already unpacked).

## Layout

- `b70`: the command above.
- `common/`: the image puller (`oci_pull.py`, by digest, resumable, OCI whiteouts) and the image's environment.
- `glm/`: GLM-5.3-Flash on exllamav3 (submodule [horvay/exllamav3](https://github.com/horvay/exllamav3) `b70-xpu`) with
  the three-tier expert cache, SYCL/ESIMD and Triton kernels, and its server; history from
  [horvay/b70-glm53-flash](https://github.com/horvay/b70-glm53-flash).
- `dsv41/`: DeepSeek-V4.1-Flash on vcruz305's exllamav3 fork (submodule
  [vcruz305/exllamav3](https://github.com/vcruz305/exllamav3) plus `dsv41/exllamav3.patch`, applied by `b70 setup`), on
  the GLM port's kernels and its own expert tiers, DSpark speculative decoding, and a server whose prompts go through
  DeepSeek's reference encoder (tools, thinking, reasoning effort 1-100).
- `trellis-serve.patch`: everything changed in trellis-serve (SGLang's EXL3 plugin and kernels) for Gemma and Flash-Next.
- `flashnext/`: the VRAM and RAM expert lists, and tools to rebuild them from your own workload.
- `gemma/`: notes and measurements.
