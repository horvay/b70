# GLM-5.3-Flash on an Intel Arc Pro B70

GLM-5.3-Flash (a ~105 GB EXL3 mixture-of-experts model: 42 MoE layers × 288 experts, KDA linear attention,
MLA/DSA sparse attention, mHC hyper-connections) running with one **Intel Arc Pro B70** (Battlemage, 32 GB) as the
only GPU, on a desktop with 62 GB of RAM and one PCIe 4.0 NVMe SSD. The model does not fit in VRAM or RAM, so the
routed experts are spread over three tiers: VRAM, pinned RAM, and an NVMe expert store.

It started from [glm-flash-lite](https://github.com/sybil-solutions/glm-flash-lite) (which serves the model from a
24 GB RTX 3090 + RAM + NVMe on CUDA exllamav3) and [exllamav3](https://github.com/turboderp-org/exllamav3), and
runs exllamav3's Python model code on PyTorch XPU with our own SYCL/ESIMD and Triton kernels underneath.

Model used: [MikeRoz/GLM-5.3-Flash-Uncensored-2.51bpw-h6-exl3](https://huggingface.co/MikeRoz/GLM-5.3-Flash-Uncensored-2.51bpw-h6-exl3)
(routed experts 2-bit in 22 layers, 3-bit in 20).

Expert bitrates: the MoE kernel takes 2, 3 and 4 bits, and the half-integer 2.5 and 3.5 bits of exllamav3 1.6's "sc"
quants (turboderp's `sc_2.60bpw`: experts 2.5-bit in 41 layers, 3.5 in one, MTP 4; dense layers 4/6-bit). A half-integer
tile alternates KA- and (KA+1)-bit trellis steps (upstream `frac.cu`), so 64 values fill 2K words: the kernels decode it
with the same period scheme as integer K (`exl3_esimd.h` `Geo`, template codes 25 and 35), each lane covering two
output columns. `tests/units.py` checks every K against an fp64 CPU decode. On the 2026-10-30 GM-LoRA setup, sc_2.60
against sc_3.10: store 100 vs 119 GB, 2496 vs ~2080 VRAM experts, decode 9-10 vs 7-8 tok/s after the first 100
tokens (bonus 0.1, 14 GB RAM tier), prose perplexity 2.99 vs 3.12-3.14.

**Result** (single user, greedy):

| | First working version | Now |
|---|---|---|
| Decode | 5.2 tok/s | **~15-18 tok/s** |
| Prompt, 2K tokens | ~40 s | **9.0 s** (225 tok/s) |
| Prompt, 4K tokens | 72 s for 3.6K | **10.5 s** (382 tok/s) |
| Prompt, 8K tokens | out of memory | **14.4 s** (556 tok/s), one pass |
| Chat follow-up (2K cached + a short message) | re-read the whole history | **1.7-3 s** to the first token |

## Where the code is

- `xpu/glm_xpu/`: the runtime.
  - `tier.py`: the three-tier expert cache (VRAM slots with frequency-gated admission, pinned RAM read zero-copy
    by the kernels over PCIe, NVMe store with parallel O_DIRECT reads; prompt staging).
  - `block_graphs.py`: per-block XPU graphs for decode, the no-sync router and its RAM margin, MTP verify steps.
  - `mla_graph.py`: MLA/DSA attention with device-side lengths so a whole layer fits in a graph.
  - `k_hc.py`, `k_kda.py`, `sparse_mla.py`: Triton / oneDNN kernels (mHC mix, KDA step, sparse MLA).
  - `live.py`: live speed meter (`tail -f run/live.log`).
- `xpu/csrc/`: SYCL/ESIMD kernels. `exl3_moe.n128.sycl` is the grouped EXL3 MoE kernel (GEMV for decode, DPAS for
  prompts) reading expert weights through a pointer table, so VRAM- and RAM-resident experts look the same to it.
  The EXL3 linear kernels started from [trellis-serve](https://github.com/0xSero/trellis-serve) (`xpu/TRELLIS_COMMIT`).
  `tier_native.cpp` (pybind11, plain C++ threads) is the tier's host path: the NVMe reader, its LRUs and the common
  part of a decode layer's cache step. Where decode time goes (2026-10-30, sc_2.60bpw, ~100 ms per token): ~70 ms of
  GPU work, ~17 ms waiting on NVMe demand reads, ~5 ms of host bookkeeping after each layer's routing sync. An XPU
  graph replay blocks the host until the GPU has (nearly) caught up (`tests/graphbench.py`; no Level Zero setting
  changes it), so host and GPU take turns at every layer's sync rather than overlapping. The MoE kernel alone
  (`tests/moebench.py`, M=1, 8 experts): all in VRAM 0.22 ms per layer, 2 of them read from the RAM tier 0.70 ms
  (PCIe, ~25-30 GB/s), so the ~70-90 RAM-tier picks per token cost ~15-20 ms, the largest single item; dense
  linears read 6.3 GiB per token at ~450 GB/s (~15 ms).
  Prompts (2026-10-30, sc_2.60bpw, 4.9k tokens): half-integer K had been left out of the DPAS prefill mask
  (`glm_xpu/ops.py`), so prompts ran the grouped GEMV: 29-31 s warm, now 13-14 s, NVMe-bound at ~6 GB/s. The first
  prompt after a start also faulted in the ~5 GB read-ahead ring (now touched at load) and JIT-compiled its kernels;
  `build.sh` compiles the MoE kernels ahead of time for the B70 (`_moe_glm.so`, default GRF; its DPAS prompt path in
  `_moe_prefill.so`, large GRF; the routing alignment kernel JIT in `_moe_align.so`). Real server: warm-up request
  31-33 -> 25 s, first 4.4k-token prompt 65-69 -> 38.5 s. A follow-up of a few hundred tokens still routes to nearly
  all 288 experts of every layer, so it reads ~50 GB of experts from NVMe (~10 s) whatever the kernels do.
- `xpu/exllamav3_ext/`: stands in for exllamav3's compiled CUDA extension on XPU.
- `xpu/pack_store.py`: packs the routed experts into the NVMe store.
- `xpu/devserver.py`, `xpu/dev.py`, `xpu/snip/`: a persistent dev server (model loaded once) and the measurement
  snippets every number below comes from.
- **exllamav3 fork:** [horvay/exllamav3 `b70-xpu`](https://github.com/horvay/exllamav3/tree/b70-xpu), also as
  `patches/exllamav3.patch` against turboderp-org/exllamav3 `958ec93` (v1.5.1).
- `docs/dev-notes.md`: the full working notes, including hardware facts and Level Zero pitfalls.

## The bottlenecks, in the order they were hit

### Decode

| # | Symptom | Cause | Fix | Effect |
| --- | --- | --- | --- | --- |
| 1 | ~5 tok/s | One host sync per MoE layer: the CPU had to read the router's picks and make those experts resident before the GPU could continue (42 times per token) | **No-sync routing**: the router only picks experts already in VRAM or RAM (an NVMe-only pick becomes the next-best resident one); cache bookkeeping runs once per token, one token late, overlapped with the dense layers | ~12-13 tok/s |
| 2 | MLA attention 15.7 ms/token | ~70 separately launched ops per layer, host-bound; their grids depended on host-side lengths | Device-side lengths and fixed-size top-k selection, whole MLA layer captured in an XPU graph (`mla_graph.py`) | 8.9 ms/token |
| 3 | ~15 ms/token in tiny ops | The mHC hyper-connection mix ran its 1.5 MB reduction in a single work-group at batch 1, twice per block | Split the reduction over 64 work-groups + a finishing kernel | 166 → 17 µs per call |
| 4 | ~20 ms/token reading RAM experts | Every RAM-resident pick is streamed over PCIe (~27 GB/s vs ~450 GB/s VRAM) | **RAM margin** in the router: a RAM expert's score is lowered by 0.2 for ranking only, so near-ties go to VRAM experts | 12.9 → 17.8 tok/s at the same quality (below) |

RAM margin sweep (top-1 agreement and KL against exact routing, teacher-forced over 192 positions; speed on 128-token
generations):

| Margin | Top-1 vs exact | KL | Decode |
| --- | --- | --- | --- |
| 0 (plain no-sync) | 92.7% | 0.072 | 12.9 tok/s |
| 0.05 | 92.7% | 0.061 | 15.2 tok/s |
| 0.1 | 90.6% | 0.083 | 17.0 tok/s |
| **0.2 (default)** | **91.7%** | **0.085** | **17.8 tok/s** |
| 0.3 | 88.5% | 0.135 | 18.7 tok/s |
| 0.5 | 88.5% | 0.171 | 18.9 tok/s |
| VRAM only | 77.1% | 0.524 | 21.2 tok/s |

### Prompts

| # | Symptom | Cause | Fix | Effect |
| --- | --- | --- | --- | --- |
| 5 | Slow prompts in half the layers | The DPAS (matrix engine) prompt kernel was 3-bit only; the 2-bit layers fell back to GEMV | Generalized the DPAS B-operand build to 2- and 4-bit | prompt kernel for every layer |
| 6 | Every prompt fetched all experts twice | exllamav3 runs a separate forward for the last page to save a recurrent (KDA) checkpoint | Stash the checkpoint inside the single forward (the KDA layer splits itself at that position) | 2K prompt 16-20 s → 9 s |
| 7 | Out of memory above ~3.5K tokens; 0.7 GiB per 1K tokens | KDA prefill kept dead fp32 tensors alive; MLA's sparse path held ~200 KB/token of per-row tensors; XPU SDPA materialized the scores (quadratic) | Free early, in-place cumsum, per-layer splits (KDA every 2048, MLA 1024-token sub-chunks, MLP row slices) | ~85 MiB per 1K tokens; 8K in one pass |
| 8 | Crash on the first token after a long prompt | PyTorch kept the prompt's freed blocks reserved; the next graph replay got `OUT_OF_DEVICE_MEMORY` | Release the cache at the end of a large forward | fixed |
| 9 | 18.5 s for a short chat follow-up | The prompt path streamed every non-resident expert of the layers ahead | Short prompts read only the experts they picked, in parallel | 1.7 s |

### MTP (speculative decoding): tried, not a win here

The model ships an MTP head, and it works (68-88% of drafts accepted), with verify steps on the graph/no-sync path.
But consecutive tokens share only ~1 of 8 experts per layer, and expert reads dominate the step, so checking 2 tokens
costs about twice one token. At margin 0.2: MTP off ~15-18 tok/s, MTP 1 ~15-20, MTP 2 ~12-16, MTP 4 ~10. MTP pays off
when the experts are in VRAM; with a 105 GB model on a 32 GB card it is off by default (`GLM_XPU_MTP`).

Retried 2026-10-30 on the synced path (verify steps of Q rows: per-row bonus routing, one host sync per layer for the
union of the rows' picks; `tests/mtpcheck.py`, `tests/startbench.py --mtp N`), sc_2.60bpw, bonus 0.1, sampling at
temperature 1: no MTP 9-10.5 tok/s after the first 100 tokens; MTP 1 7.9-10 (62% of drafts accepted), MTP 2 6.7-8.8
(45%). The GM LoRA has no layer-45 (MTP) weights, but it is not what holds acceptance down: with the LoRA off, so the
head drafts for its own model, MTP 1 accepted 64% and ran 5.9-9.1 tok/s. Greedy accepts 92-95%. Greedy output with
and without drafts first differs at a near-tie (logits 17.98 vs 17.88).

Hybrid verify, 2026-10-31 (`GLM_XPU_SPEC_RO=1`): the verify step routes row 0 (the token) exactly and the draft rows
among VRAM/RAM experts only, so a draft is accepted only if the full-quality model agrees with the first one and the
drafts add no NVMe reads (4-5 per token, as without MTP). Forced 200 tokens, warm pass: KL 0.017 (MTP 1 and 2) against
0.0075 with every row exact and 0.13-0.42 in fast mode. Speed after the first 100 tokens: MTP 1 7.2-9.5 tok/s (54% of
drafts accepted), MTP 2 7.6-10.9 (46%), against 9-11.5 without MTP; MTP 3 segfaulted once in 1000 tokens (a 200-token
rerun was clean). Host profile (PROF=2) per emitted token: 100-120 ms against 92-96; the draft step and generator glue
doubled "everything else" (9 -> 17-19 ms) and the draft rows add 20-30 zero-copy RAM picks. A round has to cost under
~50 ms more than one token to pay at 54% acceptance; it costs ~75.

Drafting on a second stream during the token's idle gaps does not help either: a background stream of small fp16
matmuls beside decode (`tests/startbench.py --bg`) got 47% of the GPU's alone rate and slowed decode to 58% (12.7 ->
7.4 tok/s). The GPU time-slices the two queues; the side stream takes its share from the token, not from the gaps.

## Running it

From the repo root (see the [top-level README](../README.md) for what the host needs):

```bash
./b70 setup glm                                   # once: the shared image, glm/pylib, the SYCL kernels (CPU only)
huggingface-cli download MikeRoz/GLM-5.3-Flash-Uncensored-2.51bpw-h6-exl3 --local-dir /path/to/glm53-model
./b70 prepare /path/to/glm53-model                # once: pack the routed experts into the NVMe store (~100 GB written for sc_2.60bpw)
./b70 serve /path/to/glm53-model --port 8053      # OpenAI-compatible server (glm/xpu/serve.py), one request at a time
```

The server takes chat completions (streamed or not; the model's own chat template, tools, thinking as
`reasoning_content`, `chat_template_kwargs.enable_thinking` and `reasoning_effort`) and raw completions. A chat
follow-up that repeats the previous prompt reuses its cached pages and recurrent checkpoint. The load fills VRAM and
the RAM tier (~2 s for 13 GB once its pinned memory is pre-touched; written into fresh pages by the readers it took
20-45 s), and Triton kernels are cached as native code (`TRITON_XPU_GEN_NATIVE_CODE`; as SPIR-V the driver compiled
them again in every process, ~18 s). Expect the first decode after a start to be slower: the VRAM tier adapts to what
the model actually uses. A host short of memory (much of it in zram swap) makes every new page cost a reclaim, and the
fill and first reads slow down several times over.

For development, `scripts/dev_restart.sh` starts a persistent dev server and `xpu/dev.py` runs a Python snippet inside
it (`model`, `tok`, `gen` (exllamav3 Generator), `tier` are in scope; see `xpu/snip/`). `scripts/in_rootfs.sh` runs a
command in the image with this folder at `/w` (`GLM53_MODELS`, `GLM53_NVX` set the model and store binds).

Settings (environment; `b70 serve` also reads `GLM_MODE` and any `GLM_XPU_*` key from `~/.config/b70/serve.env`, the
environment winning):

| Variable | Default | |
| --- | --- | --- |
| `GLM_MODE` | quality (launcher) | `quality`: the synced bonus router below, ~10-11 tok/s, KL vs the exact model ~0.010-0.012. `fast`: `GLM_XPU_NOSYNC=1`, ~18-20 tok/s, but 20-29% of picks are forced substitutes, KL 0.13-0.25, top-1 agreement 77-83% (sc_2.60bpw, 2026-10-30) |
| `GLM_XPU_RAM_MARGIN` | 0.2 | router margin against RAM experts (0 = plain no-sync) |
| `GLM_XPU_NOSYNC` | 0 | 1 = no-sync routing: non-resident picks replaced by resident experts; fast, but with the 3.10 bpw quant it replaced 37% of picks and the output fell apart |
| `GLM_XPU_RES_BONUS` | 0.1 (launcher; 0 in the code) | exact decode with a thumb on the scale: an expert in VRAM or RAM gets this added to its routing score for selection (weights still from the true scores), so a pick only comes from NVMe when it beats a cached one by this much. 300-token decode (14 GB RAM tier): 0 = 2.9-3.0 tok/s, 0.03 = 2.9-5.0, 0.1 = 5.3-7.7; KL vs the exact model over 200 forced tokens 0.003 / 0.008 / 0.012, no step above 0.5 (no-sync's 0.52 was word salad). 0.2 was slower than 0.1. `tests/sanity.py --force` reports the share of picks swapped |
| `GLM_XPU_MISS_PF` | 0 (launcher; -1 in the code) | with the bonus, how many passed-over exact picks per layer are read from NVMe for later tokens, best first (-1 = all). Reading them cost 30-50% of the speed and did not change KL; they still count toward VRAM admission either way |
| `GLM_XPU_RES_BONUS_RAM` | = `GLM_XPU_RES_BONUS` | the bonus for an expert in RAM (read over PCIe on every use) instead of VRAM. 0.1 / 0.05 / 0.02 / 0 halved the RAM picks but did not change decode speed, which the host, not the GPU, bounded |
| `GLM_XPU_PF_WIDTH` | 12 | next-layer experts prefetched per layer (predicted from the current layer's input; with the bonus, predicted with it). Forced 200-token pass with priority reads: 8 28.4 s, 12 26.2 s, 16 27.2 s, 24 30.1 s |
| `GLM_XPU_FUSED_ROUTER` | 1 | the bonus / multi-row router's tail (sigmoid, bias, bonus, top-ks, weights, picks) as one Triton kernel (`glm_xpu/k_router.py`): 103 -> 44 us per layer; `tests/routertest.py` checks it against the torch ops |
| `GLM_XPU_NATIVE_READER` | 1 | NVMe expert reads in C++ threads (`xpu/csrc/tier_native.cpp` `Reader`; `build.sh native`) instead of a Python pool: no GIL contention with the decode loop, and reads the step waits on go ahead of queued prefetches (behind 64 queued reads: 4 ms instead of 73) |
| `GLM_XPU_NATIVE_CORE` | 1 | the tier's LRUs and the common part of `ensure()` / the post-sync bookkeeping in C++ (`Lru`, `Core.ensure_fast`, `Core.after_sync`); `test_native_core.py` checks them against the Python path |
| `GLM_XPU_SPLIT_SHARED` | 1 | the shared expert as its own graph phase between the router and the routed experts: the host waits only for the picks' copy (an event) and does the layer's cache step while the GPU runs the shared expert |
| `GLM_XPU_MIX_INTO` | 1 | mHC mix and the following RMSNorm write straight into the block state (no workspace copies: 3 fewer kernels per mix, 2 mixes per block) |
| `GLM_XPU_LORA_ADDMM` | 1 | adapter delta as `x.addmm_(in @ A, B)`: 2 kernels per adapted linear instead of 3 (one rounding of the add instead of two) |
| `GLM_XPU_STAGE_AHEAD` | 0 | layer-ahead copy of predicted RAM-tier experts into reserved VRAM rows. 89% of RAM picks were predicted, but only ~45% of staged rows were used and the extra PCIe traffic made decode slower (forced pass 25.7 s off, 31.2 s on, 27.4 s top-8, 28.1 s top-5) |
| `GLM_XPU_ADMIT_PER_CALL`, `GLM_XPU_ADMIT_MIN` | 2, 4 | VRAM admissions per layer call and the decayed pick count needed; 1-4 / 2-8 all measured within noise |
| `GLM_XPU_CURATE` | 0 | after each request (queue empty), move the most-picked RAM experts into VRAM and read the most-picked NVMe experts into RAM. Four 250-token turns averaged 9.64 tok/s with it on and 9.64 off: the next turn picks other experts |
| `GLM_XPU_PROF` | 0 | 1: per-phase GPU time (timing events, slow decode ~50%) and host timers; 2: host timers only. `tests/startbench.py --window N` prints them |
| `GLM_XPU_MTP` | 0 | MTP draft tokens |
| `GLM_XPU_CHUNK` | 8192 | prompt tokens per forward |
| `GLM_XPU_MLA_GRAPH` | 1 | MLA attention inside the decode graphs |
| `GLM_XPU_LORA_SCALE` | 1 | strength of a LoRA adapter (below); 0 leaves it off |

Host RAM: the server takes its RAM tier plus ~10 GB (the prompt read-ahead ring of 384 pinned records, the NVMe bounce
ring, ~6 GB of process memory). `b70 serve` uses a 22 GB tier unless `--ram-gb` says otherwise: 27 GB left 3.7 GB
available on a desktop with 42 GB free, under the guard's 4 GB floor.

### LoRA adapters

A PEFT LoRA trained on zai-org/GLM-5.3-Flash runs on top of the EXL3 model at load time (`xpu/glm_xpu/lora.py`, `serve.py
--lora DIR`); `b70 serve` passes it when the model folder has a `lora/` subfolder. A folder of hard links to the model's
files plus a `lora` link to the adapter makes a second model without a second copy of the weights (hard links, not
symlinks: the client's model scan skips a folder whose `config.json` is a symlink):

```bash
mkdir GLM-5.3-Flash-GM-LoRA-exl3 && ln /path/to/glm53-model/* GLM-5.3-Flash-GM-LoRA-exl3/      # same disk
ln -s /path/to/GLM-5.3-Flash-GM-LoRA GLM-5.3-Flash-GM-LoRA-exl3/lora     # cgg507/GLM-5.3-Flash-GM-LoRA
./b70 serve GLM-5.3-Flash-GM-LoRA-exl3 --port 8053
```

Quantized linears get exllamav3's runtime LoRA (`out += x A B`); the KDA q/k/v adapters are stacked onto the fused
`qkv_proj`; `kv_b_proj` (kept as the MLA kernels' w_uk / w_uv flats, not a Linear) gets the delta merged in; GatedMLP's
fused decode paths, which skip `Linear.forward`, are turned off where an adapter applies. The adapter must be in place
before the first decode graph is captured; `lora.set_scale` changes its strength in place afterwards (A/B tests).
Checks (`xpu/snip/lora_eq.py`): every kind of target matches x (sBA)^T from the adapter file to 3e-4. Decode with
cgg507's rank-64 adapter (496 targets): 15-17 tok/s on the dev server, 17 tok/s through `b70 serve`, the same as
without it.

The loader accepts both `model.layers.*` and `model.language_model.layers.*` adapter names. Targets in unloaded
layers, such as the MTP head with drafting off, are skipped. An adapter with no applied targets fails at startup.

The VRAM cache reserves enough slots in its largest main-model record class to stage a whole layer. This matters
for quants with a single 4-bit main-model layer among 3-bit layers: a split purely by layer count can give that
class fewer than the required 288 slots. The reservation stays inside the cache's byte budget.

CPU-only regression checks, without importing torch or loading the card:

```bash
python3 llm/glm/xpu/test_tier_layout.py
python3 llm/glm/xpu/test_lora_paths.py
python3 llm/glm/xpu/test_ram_slots.py
```

For a queued startup and repeated long-prompt check, `tools/smoke_server.py` starts its own `b70 serve`, saves both
responses with timings, and stops that server when finished:

```bash
b70q submit glm-smoke --ram-gib 36 -- python3 llm/glm/tools/smoke_server.py /path/to/model \
  --request /path/to/chat-request.json --output /tmp/glm-smoke.json
```

Verified 2026-10-27, queue job #701, `GLM-5.3-Flash-GM-LoRA-exl3`, 22 GiB RAM tier, 49,152-token cache:
the GM adapter applied to 485 runtime linears and 11 merged projections; the rank-2 output-projection adapter applied
to 29 runtime linears and skipped the unloaded MTP layer. A 13,359-token fantasy-world prompt completed twice.
The first request took 128.520 s to its first token, then generated 168 tokens at 5.75 tok/s. The repeat reused
13,312 prompt tokens, took 10.475 s to its first token, and generated 146 tokens at 10.93 tok/s. These are startup,
long-context and cache-reuse checks, not an evaluation of refusal behavior or factual accuracy. The smaller quant's
short-context speed at the top of this document is a separate measurement.

## Known limits

- The first prompt after a server start takes ~60 s (one-time JIT compile of the SYCL kernels).
- Prompts over 8K tokens run in 8K pieces, each fetching the experts again; 12K in one pass runs out of VRAM.
- Batch size 1: one request at a time.
- Faults seen while developing (all fixed, see `docs/dev-notes.md`): a failed XPU graph capture leaves PyTorch's
  graph pool recording and later replays fault the GPU, so a capture failure now disables graphs for the session;
  multi-row `topk`/`sort` touched the pinned host allocator during capture.

## License

MIT (see `LICENSE`). Third-party credits in `NOTICE`.
