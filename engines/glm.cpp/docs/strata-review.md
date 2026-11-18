# Strata review: what to take into glm.cpp for Flash-Next and Edge0 on the B70

Written 2026-11-15 from code and docs only. Nothing was built or run on a GPU.

Sources:

- **Strata upstream** `Niko1221/Strata`, `main` at `baf16363b4` (2026-11-15). The `sycl/` files cited below were
  fetched through the GitHub API at that commit, because the shallow clone into `~/src/strata` was still downloading
  (114 MB repo on a busy link). Paths are repo-relative.
- **Edge0's fork** `~/src/edge0/edge0-strata` at `257ff84`. It has no `sycl/` but shares the CUDA core (`src/core`,
  `src/prefill`, `src/kernels`) that the SYCL port is migrated from, plus the same Intel docs.
- **Ours**: `engines/glm.cpp` as of today (`src/q4*.cpp`, `src/moe*.cpp`, `src/exl3.cpp`, `src/engine.cpp`).

Their B70 numbers below are for Flash-Next **IQ3_S** (1.97 MB experts, 11.5K of 24.6K in VRAM). Ours are for Swift
EXL3 3.05 bpw (1.86 MB records, 14.4K in VRAM). Same model shape, same card, different quant.

## Ranked summary

| # | What | Their measured gain (card) | Applies to us? | Expected gain for us | Port cost |
|---|---|---|---|---|---|
| 1 | **Stage RAM experts into VRAM for prompts** on the copy engine, once per layer, overlapped with compute (instead of zero-copy PCIe reads per 2048-row chunk) | Not isolated; it is the core of their 1,000-1,360 tok/s prompt (B70) | Yes, directly | Likely the main part of the 2x prompt gap (estimate: 617 → 1,000+ tok/s at 5.4K). Needs one profile run first | Medium (1-2 weeks) |
| 2 | **MTP speculative decoding** with a verify window that leaves GDN state alone and replays accepted tokens on commit | Coder IQ1_M: 20 → 43-45 tok/s; IQ3_S: 30.5 (prose) / 40.8 (code) tok/s with MTP (B70) | Yes. Our draft/verify loop exists; Q4 lacks the MTP head, multi-row graph and GDN rollback | 1.3-1.8x decode (estimate), more on code than prose | Large (2-3 weeks) |
| 3 | **Copy-engine queue** built by hand through Level Zero (the v2 adapter puts every SYCL queue on the compute engine) | 0.8 s of a 3.8 s 4K prompt was compute waiting on its own copies (B70, unitrace) | Yes. All our copies sit on the compute queue | Prerequisite for #1; a few % on decode swaps | Small (1-2 days) |
| 4 | **Int8 KV** (per token, per KV head, per 64 dims, fp16 scale) and the **int8 ESIMD DPAS prompt attention** it enables | Prompt attention was 22-24% of a prompt in fp32; int8 KV halves KV memory | Yes, we planned int8 KV anyway | Memory: 24.6 → 12.7 KB/token. Prompt: maybe 10-20% at 5K+, more at long context (estimate) | Medium (KV: days; DPAS attention: 1-2 weeks) |
| 5 | **Bigger prompt chunks** (4K-8K, MoE over the whole super-chunk) and a short first chunk | 4096 → 8192: +10-16% at 16K-32K (B70) | Partly; our super-chunk is already layer-major over 16K rows | 5-15% once #1 is in (estimate) | Small |
| 6 | **N-gram reads issued at prompt start** (all chunks) and collected at layer 1; `POSIX_FADV_RANDOM`; higher queue depth | First chunk's PLE rows read beside layer 0: 784 → 825 tok/s at 2.2K (B70); 64 threads: 466 → 330 ms of reads | Yes | 0-5% of prompt time (0 on a warm page cache) | Small |
| 7 | **xe driver guards**: alias check on big VRAM allocations, VRAM headroom, bounded spins | Fixed NaN prompts from aliased 2 MiB pages in a 22 GiB arena (B70) | Yes: our expert pool is one ~26 GiB `malloc_device` | Correctness insurance, no speed | Small (hours) |
| 8 | Lend `lm_head` VRAM to prompt buffers (`--prefill-head-alias`) | 8 GB card: chunk 1,792 → 2,048 | Yes, as a place for #1's staging ring | Frees 397 MB during prompts | Small |
| 9 | Conversation cache checkpoints (system-prompt root, message boundary, tail) | 33% faster first edit of a long agent history (RTX 3080) | Partly; we keep 2 GDN checkpoints | Only for edited/branched chats | Small |
| - | Int8 DPAS experts with q8_1 activations | +12-21% prompt below ~130 rows per expert (B70) | **No for EXL3** (decoded weights are not integers). Maybe for Edge0 NVFP4 (see §1.4) | - | - |
| - | Chunked GDN prefill | Not used on the B70 (they run a recurrent kernel; DeltaNet is 5.8% of a 40K prompt) | Not the cause of the gap | Small | - |
| - | Their adaptive expert residency | Same idea as ours (decayed pick counts, bounded swaps) | Ours is equivalent | None | - |
| - | KVA (approximate late-layer state for long prompts) | +11-28% prompt, costs answers | Quality-affecting, off by default in Strata too | Skip | - |

What is not better than ours: decode residency policy (§3), the GDN prompt kernel (§1.5), the dense EXL3 prompt path
(dequantize + oneMKL, same as theirs), and the n-gram row decode itself.

---

## 1. Prompt path: why they are about 2x faster

### 1.1 Their numbers on the B70

`docs/INTEL.md` ("Arc Pro B70 with a model that does not fit: Flash-Next IQ3_S"), engine 0.1.40-0.1.43, B70, xe,
PCIe 13.3 GB/s, 60 GB RAM, INT8 KV, `--stream-experts --vram-reserve-mib 1024 --ple-io ram`:

| Workload | Result |
|---|---|
| 4,095-token prompt, `--prefill 4096` | 980-1,002 tok/s |
| 8,169-token prompt, `--prefill 8192`, lending cache slots | 1,117 tok/s |
| 16K / 32K prompt, `--prefill 8192` vs 4096 | 1,347-1,383 vs 1,187-1,222 tok/s |
| Fused int8 experts vs FP16 chain, 4,095 tokens | 1,407.7 vs 1,257.0 tok/s (+12%) |

Their GPU timeline at 40K (`docs/INTEL.md`, "Long prompts"): expert dequant 15.7%, expert GEMMs 23.4%, prompt attention
19.2%, host grouping 6.5%, QSA select 6.8%, DeltaNet 5.8%. The rest (~23%) is dense projections and hyper-connections.

Ours: 5,400 tokens in 8.75 s (617 tok/s), `prefill_chunk` 2048 (`src/q4.h:87`), RAM experts read zero-copy by the DPAS
kernels, GDN recurrent, QSA attention a plain SYCL kernel (`README.md`, Swift section).

### 1.2 The main suspect: how RAM experts reach the prompt kernels

**What we do.** `Q4Model::prefill` (`src/q4.cpp:902-931`) is layer-major over super-chunks of `8 * C` = 16,384 rows,
but calls `block_forward` per 2048-row chunk. Each chunk's MoE goes to `moe_prefill` (`src/moe_prefill.cpp:17-58`),
which reads every routed expert through its pointer. For the 10.2K RAM records the pointer is pinned host memory, so
every chunk reads them over PCIe again.

Arithmetic (estimate, not measured):
- A 2048-row chunk routes 20,480 picks over 512 experts per layer: every expert is touched.
- So each chunk reads all 10.2K RAM records: 10.2K x 1.86 MB = **19 GB per chunk**.
- 5,400 tokens = 3 chunks = **57 GB over PCIe = 4.4 s at 13 GB/s**, half of our 8.75 s. It is worse if the 32-row
  blocks of one expert (`MB = 32`, about 40 rows per expert here, so 2 blocks) miss L2 and read the record twice.

**What they do.** `src/prefill/prefill.cpp:182-232` (Edge0 fork, same in upstream) and `docs/INTEL.md`:
- From a chunk of 1,024 tokens up ("stream all"), every non-resident expert of every layer is copied host → VRAM in a
  fixed order through a ring of 96-1,024 VRAM slots, **without waiting for that layer's routing**, so the copies keep
  running through the attention half of the layer (`prefill.cpp:194-197`, `:228-232`, `:356-375`).
- The copies go on the GPU's copy engine (§2 below, item 3 of the table). Without it, 0.8 s of a 3.8 s 4K prompt was the
  compute engine stalled behind its own copies (`sycl/include/strata/copy_queue.hpp:3-9`).
- Each expert crosses PCIe **once per chunk**, then the kernels read it from VRAM at full bandwidth. Their 4K chunk moves
  ~26 GB (13K non-resident IQ3_S experts) and still reaches 1,000 tok/s, because the copy overlaps compute.

**What to port.**
1. A copy-engine queue (§1.3).
2. A VRAM staging area for one layer's RAM records, double-buffered (layer L computing, layer L+1 copying). Per layer
   we have ~213 RAM records on average (10.2K / 48; the hot list interleaves layers rank by rank, `src/q4.cpp:476-487`,
   so the spread is even). Two layers at ~215 x 1.86 MB is ~0.8 GB.
3. In `prefill`, before layer L's chunks, swap the layer's `ptrs_dev` entries for its RAM experts to the staging copies
   (a second pointer table per buffer, or a patched copy of the layer's row). All 8 chunks of the super-chunk then read
   VRAM. Our loop order (`for b in blocks: for c0 in chunks`) already makes one copy per layer per super-chunk enough.
4. Record an event after layer L+1's copies; make layer L+1's first MoE kernel wait on it.

PCIe floor after the change: 19 GB per super-chunk = 1.5 s for a 5.4K prompt, hidden behind ~4 s of compute. Expected
result (estimate): 1,000-1,300 tok/s at 5.4K. **Before building it, run one prompt with `prof_point` timing on** and
confirm that "MoE: routed experts" dominates; if it does not, this estimate is wrong.

Where the ~0.8 GB comes from (pick one, measure decode after):
- Lower the VRAM expert count by ~430 records (3%). Simplest; decode loses a little.
- Lend `lm_head`'s 397 MB to the staging ring during prompts and copy it back before the first token
  (Strata `--prefill-head-alias`, `docs/PREFILL_HEAD_ALIAS.md`). Our last prompt row still needs `lm_head`, so restore it
  before `forward()` of the last token.
- Lend the coldest VRAM slots and refill them after the prompt (Strata's "prefill borrow", `docs/INTEL.md`: ~1 s of
  refill per prompt). Needs a RAM copy of those records, which we do not keep. Not worth it here.

A smaller ring is possible (copy in routing order inside the layer, one event per group), but the per-layer double
buffer is simpler and the VRAM cost is acceptable.

### 1.3 Copy-engine queue (Level Zero v2 adapter)

`sycl/include/strata/copy_queue.hpp:1-17`, `sycl/src/core/copy_queue.cpp:73-138`.

- With the Level Zero v2 adapter (the default on Battlemage), **every** SYCL queue runs on the compute engine, including
  a queue that only does `memcpy`. A 2 MB copy blocks kernels for its duration.
- Fix: `zeDeviceGetCommandQueueGroupProperties`, pick the group with `COPY` and without `COMPUTE`
  (`copy_queue.cpp:111`), `zeCommandListCreateImmediate` on it (in-order, async), then
  `sycl::make_queue<ext_oneapi_level_zero>` (`:127`). `memcpy`, events and cross-queue barriers keep working. The queue
  runs copies and barriers only, no kernels (not even `fill`).
- Only copy from pinned (USM host) memory on it. Our `src/rt.cpp:108-109` already notes that the copy engine faults on a
  non-resident page of a mapped file.
- Ours: `rebalance()` (`src/q4.cpp:572-645`) does its swap copies on the compute queue with `q.wait()` between steps,
  between decode tokens. Moving them to the copy-engine queue and ordering with an event lets them overlap the next
  token (Strata's `STRATA_ADAPT_OVERLAP`, +4-8% on some CUDA cards; not measured on the B70).

Cost: about 100 lines. This is the first thing to do; #1 depends on it.

### 1.4 Int8 DPAS experts (`STRATA_PREFILL_MMQ`, default `auto` on Battlemage)

`docs/INTEL_INT8_EXPERTS.md`, `sycl/include/strata/prefill/esimd_moe.hpp:1-80`, `sycl/src/prefill/esimd_moe.dp.cpp`
(1,052 lines; host entry points `:886-1052`).

- ESIMD, `grf_size<256>`. One hardware thread owns 16 output rows of one expert, all of K, and MT x 8 activation rows
  (MT = 2/4/8 by the group's mean rows per expert). A work-group is 8 threads on 128 consecutive outputs; activation
  tiles are double-buffered in SLM, 128 of K per chunk.
- Weights (IQ3_S etc.) are decoded in registers straight into the s8 DPAS B operand (8x16x32). Activations are q8_1
  (per row, per 32 of K, `d = amax/127`). Each 32-block's int32 dot is exact, then scaled in float.
- Groups of up to 16 experts, 4 launches per group (quantize, gate/up, SwiGLU + quantize, down).
- `auto` times int8 against the FP16 chain at the first layer and uses int8 below the crossover (130-250 rows per
  expert); it stops paying at ~194 rows (1.03x) against 1.4-1.9x at 60 rows.
- Gain on B70: +21% at 1,024 tokens, +12% at 4,095. Quality vs FP16: KL 2e-4 mean, top-1 99.7%.

**For Swift (EXL3): does not apply.** The int8 trick works because IQ codebook entries are small integers. EXL3's
decoded weights are fp16 values from the trellis codebook, not integers; rounding them to int8 would be a new lossy
quantization. Our fused EXL3-decode + fp16 DPAS (`kernels/exl3_moe_kernels.h:370-656`) is the right shape already. What
we can take from their kernel:
- Decode once per 64 rows, not 32. Our `MB = 32` (`src/moe_prefill.cpp:20`) re-decodes the trellis for every 32-row
  block. Their MT = 8 tiles (64 rows) per thread, with the register budget measured (MT 12 spilled and ran 2-5x slower).
- A crossover timer instead of a fixed `M >= 256` switch (`src/moe.cpp:89`).

**For Edge0 (NVFP4 experts): maybe.** E2M1 values times 2 are integers (0, ±1, ±2, ±3, ±4, ±6, ±8, ±12), so they fit the
s8 B operand exactly. The catch: NVFP4 has one FP8 scale per 16 values and an s8 DPAS step is 32 deep, so either two
DPAS per 32 with half of each zeroed (Strata does this for IQ2_S, which has per-16 scales: `INTEL_INT8_EXPERTS.md`,
"Scales") or per-16 partial sums. At half efficiency int8 matches fp16 DPAS throughput, so the win is only in skipping
the fp16 decode work. The Edge0 fork's CUDA prompt path uses this W4A8 form on non-Blackwell cards
(`src/prefill/mmq_nvfp4_w4a8.cu:1-9`). Decide with a micro-benchmark, not by porting first.

### 1.5 GDN prompt kernel: not the cause

On Battlemage, Strata's prompt GDN is a **recurrent** kernel too. The sub-group variant (`STRATA_GDN_XE`,
`sycl/include/strata/prefill/gdn_xe.hpp:33-...`) is off on Battlemage because it was 3.5% slower there; the chunked CUDA
kernel (`src/prefill/gdn_chunk.cu`) is not used on SYCL. DeltaNet is 5.8% of their 40K prompt. Our `gdn_delta`
(`src/q4_ops.cpp:117`) has the same layout (one work-group per v-head and 32 columns, state in registers). Leave it.

### 1.6 QSA prompt attention

- Default on Battlemage: `qsa_prompt_attn_esimd` (`sycl/src/kernels/cuda/qsa_prompt_attn_esimd.dp.cpp:1-21, 406-443`;
  dispatched at `sycl/src/kernels/cuda/qsa_prompt_attn.dp.cpp:2014`). **Int8 KV only.**
  - One work-group of 4 threads per (query, KV head); thread t owns output dims 64t..64t+63 (= V's scale group t).
  - Q.K on s8 DPAS: A = 8 cells x 32 dims of K codes, B = 32 dims x 16 heads of Q (the 12 GQA heads plus 4 zero lanes).
    Q is split into hi + lo int8 per head (~15 bits); K's per-64 scale is applied to each group's int32 sum.
  - Softmax in fp32 over 128-cell super-steps; P.V again on s8 DPAS with P as hi + lo int8.
  - The fp32 kernel it replaced was compute-bound at ~60% of fp32 peak and 22-24% of a prompt.
- Their `joint_matrix` (XMX) version was **24-27% slower** than the fp32 fallback (`docs/INTEL.md`, "Other prompt-path
  trials"). Write this in ESIMD, not `joint_matrix`.
- Ours: `attn_select` + `attn_core` (`src/q4.cpp:740-780`), plain SYCL, in pieces of up to 256 rows. At 5.4K it is
  probably 5-10% of our time (estimate); it grows with context. After #1, port their kernel together with int8 KV (§4).
  With fp16 KV the same tiling works on fp16 DPAS (12 heads on the M side, as our MoE DPAS does).

### 1.7 Chunk size and the first chunk

- Their chunk: 4,096, or 8,192 on a 24 GB+ Battlemage above 32K context. 8,192 vs 4,096: +10-16% at 16K-32K. A different
  chunk is not bit-identical (first-token KL 1.1e-2 at 16K, top-1 the same).
- `STRATA_PREFILL_FIRST`: a short first chunk (256 tokens) to start the GPU while the PLE rows are read; on by default
  only when VRAM holds every expert (765 vs 711 tok/s at 2.2K on Coder IQ1_M).
- Ours: once #1 is in, try `GLM_PREFILL_CHUNK` 4096 for attention/GDN and run the MoE of a layer over the whole
  super-chunk (more rows per expert: better DPAS use, fewer weight decodes per row). Measure; do not assume.

---

## 2. MTP speculative decoding

### 2.1 What it bought them on the B70

`docs/INTEL.md`, history table (Coder IQ1_M, 19-token prompt, 256 greedy tokens):

| Change | Decode |
|---|---|
| Suffix drafter only, `--spec 2` | 20.2-20.8 tok/s |
| MTP draft layer, `--spec 4 --mtp`, 81-87% accepted | 42.7-45.3 tok/s |
| Commit graph left running while the drafter runs on its own queue | 46.5 → 49.8 tok/s |

IQ3_S (our case): 30.5 tok/s on a story (66% of drafts accepted), 40.8 on code (90%). No IQ3_S number without MTP is
published (a native pack refuses `--spec` below 2). Draft depth: `--spec 6` vs 4 is +6.5% on code and -1.5% on prose,
so they keep 4 with `--spec-min-p 0.5`. Decode on prose: 30-39 tok/s at 55-72% acceptance; code 45-51 at 77-85%.

Our decode without MTP is already 29-37 tok/s, so our base is faster than theirs; MTP multiplies it.

### 2.2 How they run the draft layer

`include/strata/core/mtp.hpp:1-20` (Edge0 fork; upstream `sycl/include/strata/core/mtp.hpp` is the SYCL copy):

- **Cell convention** (vLLM's `qwen4_exp` MTP): cell i pairs the trunk's **final multi-stream residual** at position i
  (the pre-mixer stack, 10240 values) with the token at i+1, at **rope position i**. Its output predicts token i+2; its
  own residual feeds the next draft step.
- **`pre_fc_norm_hidden`: one RMS over all 10240 values by default.** Per-stream is opt-in (`--mtp-hnorm stream`,
  `mtp.hpp:60-63`, "as llama.cpp's qwen4exp MTP graph does"). Their probe (`tools/mtp_probe.py`) measured greedy
  acceptance 0.89 / 0.86 / 0.85 at draft steps 1-3 with the single norm. This settles `docs/qwen4-spec.md` §13 item 5
  as far as evidence goes: start with SGLang's single norm, keep per-stream as a switch.
- **Attention is dense** over every cell the layer has seen (window default 32,768 cells, `--mtp-window`), no indexer.
  Identical to the sparse rule below 2,051 cells; speculative cells are simply overwritten later.
- **All 512 MTP experts in VRAM** (708 MB for IQ; ~950 MB for our K=3 records), grouped "hit" kernels.
- **Draft head over a token subset** (`rt/draft_vocab.bin`; `--draft-vocab en` is ~40K tokens, ~110 MiB less VRAM).
  Large projections in Q8_0, optionally Q4_0 (`--mtp-q4`). Only draft quality depends on these choices; the verify
  window decides every token.
- **Chain rule** (`src/program/generate.cpp:13269-13277`): the first draft always; another while the previous draft's
  probability under the draft head is at least `spec_min_p`, up to `spec - 1` drafts.
- **Prompt cells**: the prompt path computes the MTP layer's K/V in batches (`Prefill::draft_kv`, `mtp.hpp:169-175`),
  only for cells the window can still reach.
- The drafter runs on its own queue while the commit graph finishes (+7% above).

### 2.3 How they verify

`include/strata/core/verify.hpp:1-21`, `src/core/verify.cpp:3933-3990`:

- T tokens (the last accepted one plus T-1 drafts) go through all 48 layers in **one captured graph**, and every row's
  argmax is exactly what one-token decode would give (multi-row kernels whose per-row arithmetic is the one-row
  kernel's). So with greedy decoding the output does not depend on drafting.
- **GDN state is not touched by the window.** The window stores its per-token inputs; `commit(n_keep)` replays the
  accepted tokens through the conv history and recurrent state. The QSA indexer key tail is restored from a snapshot
  and the accepted keys re-appended; the PLE history is set to its snapshot after token n_keep-1. K/V cells past the
  accepted prefix are overwritten later. The commit is itself a captured graph, with n_keep read from pinned memory
  (`verify.cpp:3941-3943`).
- Cost of a window: dense weights read once for T rows; routed experts = the union of the T rows' experts, measured at
  1.75x one token's misses for T=2, 2.4x for T=3, 3.05x for T=4 (`verify.hpp:10-11`).
- Sampling: by default drafts are argmax and each row's token is the target's own sample (as in our `Engine::run`).
  `STRATA_SPEC_COUPLED=1` / `STRATA_SPEC_PROB=1` (opt-in) draft by sampling and accept probabilistically, which raises
  acceptance at temperature > 0.

### 2.4 Fit with glm.cpp

We already have: the draft/verify loop with per-row sampling (`src/engine.cpp:60-100`), multi-row `forward()` up to
`qmax = 8` (`src/q4.cpp:877-900`), and MTP support for GLM. Missing for Q4:

1. The MTP head: `mtp.fc_hidden`, `mtp.fc_embedding` (EXL3), one QSA block (dense attention is enough), its MoE,
   `mtp.hyper_connection_mixer`. 6,203 `mtp.*` tensors are in the Swift index (`docs/qwen4-spec.md` §10).
2. A multi-row decode graph for Q = 2..5 (today only Q = 1 is a graph, `src/q4.cpp:882`).
3. **GDN rollback.** Today a rejected draft calls `m_.restore(p0 + i + 1)` (`src/engine.cpp:94`), which for Q4 means a
   113 MB GDN checkpoint copy, and only at checkpoint positions. Port Strata's scheme: the verify pass computes outputs
   for all T rows without writing the recurrent and conv state, and saves its per-row inputs (conv input columns, q/k/v,
   beta, g); commit replays n_keep rows. The replay reads and writes the 113 MB state once (~0.4 ms at 600 GB/s).
   exllamav3's alternative (`docs/qwen4-spec.md` §6.3: "keeps 2 recurrent rows per slot") costs 113 MB per kept row.
4. PLE conv state and `ids_ctx` snapshots per row, and QSA pooled-key tail restore, as above.

Expected gain (estimate): our token is ~27-30 ms against ~8 ms of weight bytes, so it is launch- and latency-bound, and
a 4-row window should cost well under 2x one token. With 66-90% acceptance at spec 4 that is 1.3-1.8x. Generic text
with 19% RAM picks gains less (3x the RAM expert reads per window). GM text (2% RAM picks) gains most.

VRAM: ~1 GB for the MTP experts, attention and buffers, i.e. ~540 fewer trunk expert slots.

---

## 3. Expert residency and streaming

**Not better than ours for decode.** Details:

- **Static profile** (`data/expert-profile-sycl.bin`): a ranking from the routing of 45 varied prompts (chat, code,
  non-English), `tools/make_profile_mix.py`, `tools/collect_profile_traces.py`. Same idea as our
  `llm/flashnext/hot/gm-7500.json` + `gm-ram-4500.json`. Ours is tuned to our own chat-style text.
- **Adaptive tier** (`--adapt-every`, `--adapt-swaps`, `--adapt-decay`; `docs/DETAILS.md:147, 238-245`): decayed pick
  counts, a bounded number of swaps per window. The same as our `rebalance()` (every 32 tokens, decay 0.5, 2x
  hysteresis, at most 64 pairs). Two refinements worth taking only if cheap:
  - copies on the copy engine, overlapped with the next window (§1.3);
  - `STRATA_ADAPT_LAG=2`: use swaps once they are two windows old, so decode never waits for them.
- **"Predicts loads a step ahead"**: this is the **SSD file tier** (`STRATA_IO_PREFETCH`, `docs/DETAILS.md:272`): I/O
  threads `pread` the router-predicted experts of the next layers into the page cache. We have no SSD tier (everything
  is VRAM or pinned RAM), so it does not apply. `STRATA_FS_SLOTS` (foresight VRAM slots per layer) lost in every box they
  measured (`DETAILS.md:1030`).
- **Disjoint caches** (`docs/DISJOINT_EXPERT_CACHE.md`) are for a helper GPU plus a primary GPU. Not our setup.
- **Batched DMA** (`docs/BATCHED_DMA.md`): `cudaMemcpyBatchAsync`, CUDA only, not shown to raise tok/s.
- **Residency-biased routing** (`STRATA_ROUTE_RESIDENT`, opt-in): biases routing toward VRAM experts. Changes the
  model's output; skip.

---

## 4. Int8 KV cache (concrete detail for our port)

### 4.1 Format

`include/strata/kernels/kv_q8.hpp:1-12, 23-27` (Edge0 fork; SYCL copy `sycl/src/kernels/cuda/kv_q8.dp.cpp`):

- Group: **64 consecutive values of one KV head of one token** (`KV_Q8_GROUP = 64`). Head dim 256 → 4 groups.
- `scale = fp16(max|x| / 127)`; `code = clamp(rint(x / scale), -127, 127)` computed **against the stored fp16 scale**
  (`kv_q8.cu:55-61`); zero group → scale 0, codes 0. Dequant `x' = code * scale` (fp32, then fp16 if staged).
- Separate pools for K and V. Codes `[page][kv_head][page_size][head_dim]` int8, scales
  `[page][kv_head][page_size][head_dim/64]` fp16.
- K is stored after q/k norm and rope, V raw (as now). The QSA indexer planes are **not** quantized (Strata keeps the
  pooled indexer keys in fp32).
- Size for our model: per token and layer 2 heads x 256 x (K, V) = 1,024 B codes + 32 B scales. Over 12 layers:
  **12,672 B/token** vs 24,576 fp16 (spec §11). At 64K context: 0.81 GB instead of 1.57 GB, i.e. ~400 more VRAM expert
  slots.

### 4.2 Kernels

- **Append** (`kv_q8.cu:33-74`): one work-group of 64 per (KV head, 64-group, K or V); amax by sub-group reduce, then a
  fixed-order combine; position from device memory (graph-capturable). A multi-token variant
  (`kv_append_q8_steps`) does T tokens in one launch for verify windows and prompt chunks. For us: fuse it into
  `attn_prep` (`src/q4_ops.cpp`), which already writes `c.k`/`c.v`. A 16-lane sub-group per 64 values (4 values a
  lane) fits our kernels better than their 64-thread blocks.
- **Decode attention** reads codes directly: `load8_q8` (`src/kernels/cuda/qsa_decode_attn.cu:54-62`): 8 int8 in one
  8-byte load, one scale per 8 values (8 divides 64), `out = code * scale` in fp32. For us: change the K/V loads in
  `attn_core` (`src/q4_ops.cpp:335-...`).
- **Gather** to fp16 scratch (`kv_q8.cu:78-106`) exists only for kernels that still want fp16 K/V. We do not need it.
- **Prompt attention**: the ESIMD s8 DPAS kernel of §1.6 reads the codes and scales directly.

### 4.3 Quality

- Strata uses int8 KV as **the default** on every card (`docs/DETAILS.md:214`) and as the baseline its other KV formats
  are compared against: `q4_0` KV costs +8-12% perplexity on long documents (needles still pass), `k8v4` (K int8, V
  Hadamard-rotated Q4_0) is "more precise than q4_0".
- **I found no published int8-vs-fp16 KV measurement** in either repo (docs, `bench/`). Their parity test bounds only
  the per-value error: within 0.624 quantization steps of the fp16 path (`src/kernels/kv_q8_parity.cpp:1-8`).
- For us: measure with our own harness. Teacher-forced KL against SGLang on the 347-token and 5,400-token GM chats
  (current: KL 0.006 and 0.016, `README.md`), int8 vs fp16 KV, same build. My expectation (not a measurement): a KL
  increase in the 1e-3 range. Note SGLang's B70 serve script runs **fp8_e4m3** KV (`docs/qwen4-spec.md` §2.2), so int8
  with per-64 scales is at least as precise as the SGLang setup the client used before.
- Build flags they found load-bearing for byte-exact quantizer tests (`docs/INTEL.md`, "How it is made"):
  `-fp-model=precise`, and `-cl-fp32-correctly-rounded-divide-sqrt` for the device compiler (the Arc's fp32 divide is
  not correctly rounded by default, and `amax / 127` must match a host reference bit for bit).

### 4.4 Port plan and cost

1. Pools + scale arrays in `Q4Cache::init` (`src/q4.cpp:251-256`), a `--kv int8|fp16` switch.
2. Quantize in `attn_prep` (prompt and decode); dequant in `attn_core` loads.
3. A host-reference test like `kv_q8_parity` (codes and scales bitwise, then gather bitwise).
4. KL check as in 4.3.
5. Later: the int8 ESIMD prompt attention (§1.6).

Steps 1-4: a few days. Step 5: 1-2 weeks.

---

## 5. N-gram reader, conversation cache, and B70/xe specifics

### 5.1 N-gram (PLE) reader

`include/strata/ngram/ple_reader.hpp:1-15`, `src/ngram/ple_reader.cpp`, `src/platform/direct_file.cpp:264-290`,
`src/program/generate.cpp:719-722`.

- `--ple-io direct` (default): `O_DIRECT` 4 KiB `pread`s on a thread pool where the thread count is the queue depth;
  pages deduplicated and sorted by offset per ticket; up to `--ple-inflight` 256 reads in flight ("64 left the SSD half
  idle (32K: 303 -> 189 ms)").
- A bounded set-associative **row cache** (8 ways, 1M rows ≈ 95 MB): 1M rows "would serve up to ~82% of reads"; within
  one long prompt 20-34% of rows recur.
- **Split API**: `issue()` as soon as token ids are known, `collect()` right before layer 1. For a prompt, a chunk's reads
  run while the previous chunk computes; "first chunk's PLE rows read beside layer 0" was +5% at 2.2K (784 → 825 tok/s).
- SSD keep-alive reads (some SSDs stall 50-150 ms after ~250 ms idle; seen on Windows).
- On the B70 they run `--ple-io ram` (whole 26.8 GiB IQ4_NL table in RAM, 60 GB box, experts streamed from the GGUF).
  We cannot: our table is 32.6 GB and our pinned expert pool takes 19 GB of 62 GB.

Ours (`src/q4.cpp:192-237`, `:802-812`): buffered `pread` of exactly one row (102 B) per (token, head) on a 15-thread
pool, decoded on the host; called synchronously at layer 1 of every 2048-row chunk, and before every decode graph.

Worth taking (small):
- `posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM)` at open, so cold misses do not trigger readahead. Keep buffered reads: the
  page cache is our free row cache.
- In `prefill`, issue the reads for **all** prompt rows at the start (on the pool, into a pinned buffer of n x 2560 fp16,
  ~27 MB at 5.4K) and wait only at layer 1. Today only part of each chunk's read time overlaps the GPU.
- Raise the in-flight count for prompts (more threads or io_uring) if a cold-cache profile shows the reads exposed.
- Decode: 16 reads per token, mostly page-cache hits; not worth changing.

### 5.2 Conversation / prompt cache

`docs/MESSAGE_BOUNDARY_CACHE.md`, `docs/PROMPT_CACHE_TAIL.md`, `src/core/conversation_*.cpp`.

- A bounded cache of state checkpoints: a root at the end of the system prompt (`--prompt-cache-root`), periodic marks
  (`--prompt-cache-every`), one near-tail mark (`--prompt-cache-tail`), and an opt-in mark at the last message boundary
  (`STRATA_CACHE_MESSAGE_BOUNDARY=1`, prefixes ≥ 8,192 tokens with a tail ≤ 1,024). Snapshots can go to disk.
- Measured (2x RTX 3080): first edit of a long agent history 17.2 → 11.5 s, combined with a chunk-size change.
- Ours (`src/engine.cpp:22-46`, `src/q4.cpp:946-1001`): two pinned GDN+PLE checkpoints at multiples of `ckpt_every`;
  K/V is append-only so a restore needs no K/V copy. That covers the client's normal flow (a growing conversation).
- Worth adding only if the client edits or regenerates earlier turns: a third checkpoint pinned at the end of the system
  prompt, and one at the start of the last user message. 113 MB pinned each.

### 5.3 B70 / xe things they found that we should know

From `docs/INTEL.md` ("Things that matter on this GPU", "What the port had to get right") and
`docs/INTEL_INT8_EXPERTS.md`:

- **Aliased pages in big allocations.** On the B70 (xe) a 22 GiB arena came back with two 2 MiB pages mapped onto the
  same memory; the prompt went NaN at layer 4. Their guard (`sycl/include/strata/sycl_queue.hpp:176-214, 267-299`):
  for every allocation ≥ 32 MiB, write a tag per 64 KiB in one kernel, read them back in a second, and if any block
  fails, free and allocate again behind a growing spacer (up to 8 tries; < 0.5 s at start). **Our expert pool is one
  ~26 GiB `sycl::malloc_device`** (`src/q4.cpp:499`). Add this check; it costs hours.
- **VRAM over-commit on xe does not fail**: the driver evicts buffers to host RAM and the machine can livelock until the
  watchdog resets it. Compute sizes first; leave ~1.5 GB free.
- **Host-mapped flags are not reliably visible to a running kernel** on xe (`volatile` loads do not bypass caches;
  system-scope atomics do, but host-to-device visibility mid-kernel stayed unreliable; logits went NaN). Do not build
  in-kernel host handshakes; wait on whole graphs instead (`STRATA_VERIFY_NO_HOST=1`).
- **Bounded device spins**: a spin that never sees its flag makes xe time out the queue and reset the GT; the card can
  stay wedged until reboot. Cap every spin.
- **ESIMD compiler bugs** (oneAPI 2026.1, bmg-g31): a dynamic index into a kernel-argument array read garbage (GPU page
  faults); `t % n` and `t / n` with a run-time `n` returned wrong values (they use 2-D grids instead); `gather` with
  VS > 1 is not available; a lambda capturing a `simd` by reference puts it in private memory. Accumulator tiles past
  the 256-GRF budget spilled and ran 2-5x slower.
- **ocloc** from Compute Runtime 26.22.38646 / IGC 2.36.3 crashes on Battlemage kernels (IGC floating point
  exception, exit 245).
- **oneMKL GEMM past 4 GiB** into an allocation read the last partial tile from the wrong place (A770, oneMKL 2026.1).
  Our oneMKL GEMMs take workspace buffers, but keep it in mind if a GEMM operand ever points deep into a big arena.
- `SYCL_CACHE_PERSISTENT=1` segfaulted on Xe2 during the first JIT compile (AOT builds are not affected).
- Under the Level Zero v2 adapter no host thread waits on a queue's event reliably for their stager; they had the copy
  queue write a sequence number into pinned memory and polled it. If our staging (§1.2) waits on host, prefer
  in-queue event dependencies.

---

## 6. Order of work

1. Copy-engine queue (§1.3) and the allocation alias check (§5.3). Days.
2. Profile one 5.4K prompt with `prof_point` to confirm the PCIe estimate in §1.2.
3. Prompt expert staging (§1.2), then bigger chunks / whole-super-chunk MoE (§1.7) and the prompt-start n-gram reads
   (§5.1).
4. Int8 KV (§4) with the KL check; then the int8 ESIMD prompt attention (§1.6).
5. MTP with GDN replay commit (§2).

MTP is the largest decode win but also the largest job; it can start in parallel with 3-4 if someone else takes it.
