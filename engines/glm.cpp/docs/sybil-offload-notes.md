# Notes on sybil-solutions/qwen38-flash-next-b70-offload (read 2026-11-02)

Source: https://github.com/sybil-solutions/qwen38-flash-next-b70-offload, `main` at db27d3c (merge of PR #2
"n130-16g", last push 2026-10-12). It was cloned into a scratch directory and read only. Nothing in it was built, run
or imported.

**License:** MIT, "Copyright (c) 2026 0xSero" (`LICENSE`). 0xSero also wrote trellis-serve, which our SGLang path
(`llm/flashnext`) is built on, so their `kernels/moe-a8/exl3_esimd.h` is an older copy of our ESIMD header. Ours is a
superset (we added the half-integer K codes). Several kernel files say they are ports of Strata
(github.com/Niko1221/Strata, MIT, "Copyright (c) 2026 Niko1221 and the Strata contributors"): `qsa_row_sycl.sycl`,
`qsa_select_sycl.sycl`, `n112-gdn/csrc/gdn_sycl.sycl` and `n114-decode/csrc/*`. If we copy code from those files, we
keep both MIT notices. The weights have their own license.

**What their system is:** SGLang 0.5.20-xpu + exl3xpu plugin on one B70, base Qwen3.8-Flash-Next EXL3 3.05 bpw. Only
8,000 VRAM expert slots (14.9 GB) as a device-managed LRU, a 2 or 7 GB RAM tier, and all experts on a 4x NVMe RAID0
(26 GB/s fio). Decode is **not exact**: if a pick is in neither VRAM nor RAM, it gets weight 0 for that step.
Headline (`README.md:9-12`, `docs/results.md:9-40`): 29.2 tok/s at 1 user with a 16 GB cap (decode KL 0.055 vs
prefill) and 25.6 tok/s at 32 GB (KL 0.021). Prefill is 1,139-1,193 tok/s at 8k/32k. Our engine is exact, keeps all
experts resident, gets 29-32 tok/s on generic text, and prefills at 617 tok/s (5400 tokens in 8.75 s). So their
decode is no faster than ours, but their **prefill is about 1.85x faster**. Most of what we can use is on the prompt
side.

How much of it was measured varies a lot. Every item below is tagged:
- **[measured B70]**: a GPU number on a B70, in their results or result JSONs.
- **[CPU only]**: built and checked on a CPU or OpenCL CPU device, never run on a GPU.
- **[projection]**: arithmetic or an estimate, with no run behind it.

## Ranked summary

Payoffs are my estimates for our engine. Most of them need one of our own profiles first (see the last section).

| # | idea | where it lands in ours | expected payoff | output |
|---|---|---|---|---|
| 1 | Stage each layer's RAM experts into VRAM with the copy engine before the prompt MoE, double-buffered, instead of zero-copy reads per 32-row block | `Q4Model::moe_forward_` / `prefill` in `src/q4.cpp`; reuse the `Tier::prestage` pattern (`src/tier.cpp:972`) | prefill: ~1-2 s of 8.75 s on a 5.4K prompt if the re-read estimate holds (617 -> ~750-850 tok/s) | exact |
| 2 | QSA prompt attention: one work-group per (row, kv head) with online softmax (no split-K + merge), lane-per-cell Q.K with broadcast q; 8-bit radix top-k for selection | `q4ops::attn_core`, `attn_select` (`src/q4_ops.cpp:261-417`) | 0.2-0.6 s per 5.4K prompt (needs our QSA time first) | same math, fp32 reordering |
| 3 | GDN prompt recurrence without barriers: a thread holds a slice of one value column's state, k reductions as shuffles over 4 lanes, norms in a prepass, next token prefetched | `q4ops::gdn_delta` (`src/q4_ops.cpp:117-174`) | 0.2-0.8 s per 5.4K prompt (needs our GDN time first; their kernel never ran on a GPU) | same math, fp32 reordering |
| 4 | Prefetch the RAM experts a layer will probably pick on the copy engine while attention/GDN runs, so PCIe is not idle during the non-MoE part of a token | decode graph in `Q4Model::forward` / `block_forward` | up to ~7.7 ms/token at 30% RAM picks (29 -> ~37 tok/s at best); realistically half of that | exact (a wrong guess only wastes bandwidth) |
| 5 | Better initial expert ranking for generic text: blend their 74M-pick prior (`serve/qwen_expert_prior.json`) with our GM lists | `--hot` lists, `Q4Model::load_experts` | about 0.26-0.35 ms/token for each 1% of picks moved from RAM to VRAM | exact |
| 6 | Use DPAS (not GEMV) for prompt pieces under 256 rows when they pick RAM experts | `moe_forward` threshold (`src/moe.cpp:88`) | faster short turns (TTFT); size unknown | exact (fp rounding only) |
| 7 | Rebalance swaps off the token path: write-through promotion (the `fill` path is already in our kernel header), victims rebuilt on the CPU instead of copied D2H at 8.8 GB/s | `Q4Model::rebalance` (`src/q4.cpp:572-617`) | ~0.5-1 ms/token on generic text, only while swaps happen | exact |
| 8 | Do all n-gram lookups for a prompt before (or beside) the GPU work, not one 2048-row chunk at a time at layer 1 | `Q4Model::ple_host` / `prefill` | unknown; measure the host stall at layer 1 | exact |
| 9 | Larger MoE calls in prefill (M = 8192 or the whole super-chunk), later MB = 64/128 row blocks | `prefill` (`prefill_chunk` <= 2048), `moe_prefill` (`MB = 32`) | ~0.15 s (bigger M); up to ~0.3 s (MB 64, a projection) per 5.4K prompt | exact |
| 10 | MTP self-speculation | new | low for us (see below) | exact with proper verify |

Items 1-3 are the prompt-speed gap. Items 4, 5 and 7 are the generic-text decode gap. All of them keep the output
exact or change it only by fp32 summation order.

---

## 1. Staged prefill: copy RAM experts to VRAM once per layer

**What they do.** `plugin/exl3xpu/nvtier.py:529-641` (`_stage_setup`, `_stage_kick`, `_stage_forward`):
- For every prefill MoE layer they copy **all** of the layer's experts that are not in VRAM into one of two VRAM
  staging buffers of `E * rec` bytes (2 x 953 MB). The copy runs on a side stream (the copy engine) while the
  previous layer computes.
- The MoE kernel then gets a per-layer pointer table: cache slots for resident experts, the staging buffer for the
  rest (`nvtier.py:617-619`). One device sync per forward, at the first MoE layer (`:579`).
- The same thing in `moe_offload.py:345-...` (`forward_prefill_staged`, X004).

**Their numbers [measured B70].**
- Copy engine H2D: 26.7 GB/s from anonymous THP memory and 28.4 GB/s from pinned USM
  (`research/18-b70-ingest-ceiling.md:78`). N109 `dma`: 26.8-27.0 GB/s USM and 17-19 GB/s SVM
  (`results/N109-svm-vs-usm/n109_A.json`).
- With staging, prefill is GPU-bound: per 8k forward, 4.2 of 5.4 s is attention/GDN/dense work, 0 ms is spent
  waiting on H2D (25.7 GB/s), and a deeper ring changes nothing (`docs/results.md:57-60`).
- MoE kernel on VRAM experts: 25.8 ms per layer at M = 8192, i.e. 33-37 GB/s of distinct expert data. That is more
  than the link can deliver (`research/18...:78`).
- The DPAS kernel reading host pages directly ran at about 1.6 GB/s when it first-touched fresh SVM pages
  (`nvtier.py:530`). That figure is for SVM, not the pinned USM we use.

**Why it should matter for us (my estimate, needs a profile).** Our prompt MoE reads RAM experts zero-copy inside
`GateUpDpasKernel` / `DownDpasKernel`. Each 32-row block of an expert reads the whole 1.86 MB record again, and our
pieces are 2048 rows (`prefill_chunk`, capped at 2048 in `q4.cpp:272`). The arithmetic:
- About 213 RAM experts per layer (10.2K / 48) x ~1.2 blocks each x 1.86 MB is ~475 MB over PCIe per layer per piece.
  At ~26 GB/s that is ~18 ms per layer, ~0.9 s per 2048 rows, ~2.3 s for the 5.4K prompt.
- Staged once per layer per super-chunk (`SC = 8 * C` rows, `q4.cpp:887`), that becomes one 396 MB copy (~15 ms) per
  layer, overlapped with the previous layer when the copy queue really runs beside compute.
- The upper bound is about 2 s saved of 8.75 s.
- The block count per RAM expert is a guess: RAM experts are the cold ones. The GPU L2 might also catch part of the
  re-reads.
- A per-layer `prof_point` of "MoE: routed experts" during a prompt, against how many RAM experts the layer has,
  settles this in one run.

**Mapping.**
- The GLM tier already does this: `Tier::prestage` (`src/tier.cpp:972-1010`) copies on `Dev::get().copy` into two
  slot sets, with events per set.
- For Q4, in `prefill()`'s layer loop: before layer `li`'s first piece, queue the copies of layer `li+1`'s RAM records
  into staging set `(li+1) % 2`, then point a per-layer copy of `ptrs_dev` at it.
- VRAM cost: two sets x ~213 records ≈ 0.8 GB, which is ~425 fewer VRAM slots (~+1% RAM picks in decode, ~0.3
  ms/token). Ways around it:
  - stage only the experts that get two or more blocks;
  - use one set and accept no overlap (~0.7 s of copies per 16K rows, still less than the re-reads if the estimate is
    right);
  - borrow slots during the prompt and rebuild the dropped VRAM experts from the safetensors afterwards
    (`q4_build_record`, ~2 ms of CPU each, on the thread pool).
- **Check overlap first:** see the queue pitfall in the pitfalls section.

## 2. QSA prompt attention and selection (SYCL, Strata port) [measured B70]

**What they do.** `kernels/n107-qsa/csrc/qsa_row_sycl.sycl`, kernel 2 "bcast" (`BcastKernel`, `:302-418`):
- One 256-thread work-group per (row, kv head); all chunks of that row in one work-group, online softmax (`cpw = 0`,
  no partials and no merge kernel).
- Q.K: lane = cell. Each sub-group loads its q slice once per 16 dims from SLM and broadcasts it lane by lane
  (`group_broadcast`), so there are no cross-lane reductions.
- Chunk softmax by sub-groups over heads; P.V with thread = dimension, 4 cells at a time with a `float4` read of P.
- fp32 scores, `native::exp2` with log2(e) folded into the scale, fp8/bf16 KV read in place through the slot list.

Selection, `kernels/n107-qsa/csrc/qsa_select_sycl.sycl`:
- `ScoreKernel` (`:61`): lane per block, the block's key in registers as packed bf16, RPW rows of q in SLM, broadcast.
- `TopkKernel` (`:134`): one work-group per row, **8-bit radix passes** (4 passes with a 256-bin SLM histogram), exact
  selection with ties to the lowest index.

**Their numbers** (`kernels/n107-qsa/README.md:254-285`), 8192 query rows, fp8 KV:

| variant | 8K ctx ms/layer | 32K ctx ms/layer |
|---|---|---|
| exl3xpu union (dense over the union) | 255.1 | 459.0 |
| bcast, one WG per (row, kv head) | **55.9** | **65.5** |
| bcast with split-K cpw 1/2/4 | 63.9 / 60.3 / 58.6 | - |
| Strata's own structure (64-cell split-K + merge, sg32) | 111.4 (76.9 with all chunks in one WG) | 128.0 |

- 6.8 µs per query-layer with 2051 keys. Sub-group 16 beat sub-group 32. Large GRF with sg16 was much worse (99.9).
  Exact fp8 decode cost +16%.
- Selection: 1.54 ms vs 7.54 ms for the torch path at 8K ctx, 7.8 vs 49.8 at 32K. Selected sets are 100% identical.

**Compared with ours.**
- `attn_core` (`src/q4_ops.cpp:335-417`) is split-K: 17 chunks of 128 cells per (row, kv head), each a work-group,
  partials of `2*12 + 12*256` floats per chunk, then a merge kernel. That is ~108 MB of partials per 256 rows,
  written and read back.
- Their sweep says one work-group per (row, kv head) without the merge was 1.05-1.45x faster than split-K for both
  kernel shapes. Our Q.K is "thread = cell, 6 heads per thread, q from SLM", close to their bcast, so the big
  difference is the split.
- `attn_select` (`:261-326`) runs 32 one-bit passes, each with a 1024-wide `reduce_over_group`. Theirs runs 4 passes
  of 8 bits.
- Their scoring kernel holds the key in registers and broadcasts q. Ours reads all 4x128 fp32 q values per (row,
  block) work-item from global memory.

**Not done by them, but suggested by their numbers.** 6.8 µs per row-layer is only ~7 TFLOPS. The 12 query heads of
one kv head share one selection, so Q.K is a [12 x 256] x [256 x 2051] product. That fits DPAS (M = 8 x 2 tiles, 75%
used) with gathered K rows. Their README ends at "the next step is an ESIMD gather + DPAS kernel" (`:196-197`).
`kernels/fa_esimd.h` would be our starting point. Unmeasured, but it is the largest QSA lever if QSA turns out big in
our profile.

**Output:** same math. Selection is exact (same tie rule as ours, "ties to the lower block"); attention changes only
in fp32 summation order.

## 3. GDN prompt recurrence (Strata port) [CPU only in this repo]

**What they do.** `experimental/n112-gdn/csrc/gdn_sycl.sycl`:
- `RecKernel` (`:100-200`): work-group = CB value columns x RG row groups (RG 4 by default), sub-group 16.
- Each thread keeps `128/RG = 32` fp32 state entries of **one value column** in registers for the whole prompt.
- `S.k` and `S.q` reduce over only RG adjacent lanes with `permute_group_by_xor` (2 shuffles). No SLM, no work-group
  barrier in the token loop.
- The q/k L2 norms come from a prepass (`NormKernel`, `:202`); the next token's q/k/v/g/beta are fetched while the
  current token computes (`:160`).

**Numbers.**
- CPU-validated only: output rel 1.66e-3 vs fp64, final state 4e-9, chunk continuation bit-identical
  (`docs/results.md:160`). "GPU test pending."
- Strata's own B70 figure, second-hand in `research/19-strata-lessons.md:22-23`: GDN prompt recurrence "0.9-2 s ->
  ~0.4-0.5 s per 8K".

**Compared with ours.** `gdn_delta` (`src/q4_ops.cpp:117-174`):
- One 128-thread work-group per (v-head, 32 v columns); thread `i` owns state row `k = i` across 32 v columns.
- Per token it runs 2 norm reductions, 64 sub-group reductions (32 for `S.k`, 32 for `S.q`), SLM partials and **4
  work-group barriers**. That is latency-bound.
- Their layout reduces over k, the axis a thread owns locally. Ours reduces over k, the axis spread across the work
  group. Swapping the ownership (thread = one v column, k split over 4-8 lanes) removes the barriers and almost all
  of the shuffles.

**Output:** same fp32 math with different summation order. `gdn_conv` and `gdn_norm` stay as they are.

## 4. Overlap RAM-expert reads with the non-MoE part of decode

**The measured basis [measured B70]** (`results/N109-svm-vs-usm/n109_B.json`, `read`): decode MoE GEMV at M = 1,
10 picks, X% of them in host USM:

| host share | ms/layer | ms per token (x48) |
|---|---|---|
| 0% | 0.064-0.070 | 3.1-3.4 |
| 10% | 0.095 | 4.5 |
| 30% | 0.225 | 10.8 |
| 50% | 0.37-0.38 | 18.3 |
| 100% | 0.719 | 34.5 (28.7 GB/s, the link) |

- USM and SVM read the same here. One host expert per layer hides behind the VRAM work; at three or more the kernel is
  bound by the link.
- For us at 30-35% RAM picks this is 7-9 ms per token over the all-VRAM cost. Nothing else in a token keeps the link
  busy: while attention, GDN and the hyper-connection sites run, PCIe sits idle.

**What they have.** Infrastructure only, no measured gain:
- `moe_prefetch` / `moe_prefetch_plan` plus a host worker thread that turns device-written plans into copy-engine
  copies (`kernels/moe-a8/exl3_moe.a8.sycl:1047-1225`, `moe_offload.py:252-300`).
- Their first try showed no gain because the queue did not overlap (`exl3_moe.a8.sycl:1096-1099`; pitfall below).
- `research/18...:98` ranks "take RAM-tier reads off the decode critical path" #1 for decode (derived, not run).

**For us.**
- At the start of each layer's attention/GDN, guess the layer's MoE picks: run the layer's router GEMV on that layer's
  site-0 output `y16` (cheap: 2560 x 513). Copy the guessed RAM experts (≤10 x 1.86 MB) on the copy queue into a
  small double-buffered staging area (~40 MB). The MoE uses the staged copy for every pick that was guessed right.
- How good the guess is can be measured offline with the existing `grab()` dump before any kernel work.
- Graph question: the copy has to be a node in the decode graph on another queue (SYCL graphs can record several
  queues with event edges). Whether L0 runs that memcpy node on a copy engine at the same time as compute must be
  checked.
- Wrong guesses cost only bandwidth.

## 5. Expert ranking prior for generic text [data, MIT]

- `serve/qwen_expert_prior.json`: 24,576 keys (`layer*512 + e`, hottest first) with counts. Built by
  `serve/make_prior.py` from "K02" routing traces (decode and prefill ids). The trace names in `n109_A.json` meta look
  like code/tech prompts (`C0..C7`, `L1_modelcard_summary`, `L2_offload_cache_code_review`, ...).
- 74.3M picks. On their own traces, the top 14,400 cover 86.6% of picks (computed here). That is in-sample, so it
  overstates the hit rate on new text.
- It comes from the base model. Ours is the Swift 1.5 heretic finetune; routing should be close but is not proven.
- Cheap test: turn it into our `--hot` format and run `expert_stats()` on a generic-text bench. Our GM lists give
  65-70% VRAM picks on generic text.
- N109 slope: each 1% of picks moved to VRAM saves ~0.26 ms/token (~0.35 by `research/18...:98`).

## 6. DPAS for small prompt pieces that touch RAM experts

- `plugin/sglang_plugin.tier.py:142-144`: in tier mode they set `prefill_min_m = 3`. Their reason: "the GEMV path run
  non-cached over many rows is ~100x slower on SVM memory (test_pf.py)".
- Measured in VRAM only (`exl3_moe.a8.sycl:759-760`): GEMV vs DPAS µs/layer at M = 96: 2323 vs 2977; 128: 2669 vs
  3253; 256: 3536 vs 3561; 1024: 11338 vs 4474. So 256 is the right switch point for VRAM experts, which is what we
  use (`src/moe.cpp:88`).
- For us: in GEMV align mode, an expert with `cnt` rows is read once per MR = 4 rows. An expert with 40 rows in a
  200-row follow-up turn is read about 10 times over PCIe if it lives in RAM; DPAS reads it once per 32 rows.
- Fix: send M in [~16, 256) to DPAS when the layer has RAM picks, or split the call into VRAM (GEMV) and RAM (DPAS)
  experts. This matters for game-master turn latency. Output: same math.

## 7. Cheaper adaptive residency (rebalance)

**Measured facts [measured B70]:**
- D2H is slow: 1.86 MB VRAM -> pinned USM takes 0.21 ms (8.8 GB/s); to memfd, 0.32-0.44 ms
  (`results/N109-svm-vs-usm/n109_C.json`). H2D runs at 27 GB/s.
- A victim written back from inside the decode kernel costs ~0.9-1.0 ms per expert (`STATUS_N111.md` section 1,
  `docs/results.md:140-141`).
- Write-through fills, where the decode GEMV stores the host bytes it already loaded into the VRAM slot, cost about as
  much as the read alone: N109 `cach`, "plain", 49-57 µs per miss. A write-back costs ~1 ms per miss.

**Ours.** `rebalance()` (`src/q4.cpp:572-617`) runs every 32 tokens with up to 64 pairs: `q.wait()`, D2H victims to a
bounce buffer, H2D promoted records, a host memcpy, then 64 rounds of four small memcpys, each followed by `q.wait()`.
With 64 swaps that is roughly 13 ms D2H + 5 ms H2D + host copying, on the token path (my estimate, ~0.5-1 ms/token
averaged).

**Their tricks that apply:**
- **Write-through promotion.** Our `kernels/exl3_moe_kernels.h:47-49,138,274,346-355` already has the `fill` /
  commit path from the same source, but `src/moe.cpp` never sets it. When a RAM expert is picked in decode and chosen
  for promotion, the GEMV writes it into a free VRAM slot as it reads it, and the combine kernel repoints `ptrs`. The
  bytes cross PCIe once.
- **"a7: weights are immutable"** (`exl3_moe.a8.sycl:1006`): a victim that still has a valid RAM copy needs no
  write-back. For us, a victim's RAM copy can be **rebuilt on the CPU from the safetensors** (`q4_build_record`)
  into a spare pinned slot, so nothing goes D2H at all.
- **Only the host publishes residency** (pitfall below). Flip `ptrs`/`in_vram` only after the data landed.
- **Copy queue.** Use the copy queue and events for whatever copies remain, not `q.wait()` between tokens.

Payoff depends on how concentrated the RAM picks are; `expert_stats()` already prints the share of the hottest
256/1024/4096 RAM experts. Their own device LRU (8,000 slots = 33% of experts) reported a 64% VRAM hit rate
(`research/18...:29,80`), but on bench prompts with a "10-word vocabulary" (`:137`). Don't read that as evidence that
LRU beats our static+adaptive scheme on real text.

## 8. n-gram (PLE) rows

- **Their measurements [CPU, in `research/18...:104`]:** cold prompt chunks cost 0.14-0.34 s per 8k chunk (worst
  case 0.45 s) on natural text; decode costs 0.45-0.9 ms per step.
- **Their GPU-side table [code only]:** `experimental/n111-victim-ring/csrc/ngram_nvme_xpu.sycl`:
  - a CLOCK row cache of 102 B rows in pinned, device-mapped host memory;
  - misses sorted by file offset and merged into 4 KiB O_DIRECT runs through native AIO;
  - a graph-safe publish/wait handshake with a host service thread, so the lookup sits inside the decode graph.
- **Ours.** `ple_host()` runs at layer 1 of each 2048-row chunk inside `block_forward` (`q4.cpp`), after
  `e_ev.wait()`. In layer-major prefill, chunk c+1's lookup waits until the GPU has reached chunk c's upload. While
  the host does chunk c+1's 32K preads it cannot queue more work, so the GPU can idle.
  - Cheap fix: run all lookups of the prompt (or super-chunk) up front, or on a thread as soon as the request
    arrives, into a pinned buffer of `SC` rows (16K x 2560 x 2 B = 84 MB).
  - Strata's version, second-hand (`research/19...:34`): a 256-token first chunk so the PLE rows load while the GPU
    starts, 857 -> 986 tok/s for them.
- **Decode.** The lookup is serial with sampling. It could overlap layer 0 by splitting the decode graph after layer
  0 (layer 0 does not use the PLE). Small (~0.5 ms/token).

## 9. Bigger MoE calls and row blocks in prefill

- Their fit [projection from measured points] (`research/18...:103`): prompt MoE time ≈ 0.7 ms + blocks x 1.86 MB /
  ~215 GB/s, where blocks are the 32-row blocks.
- Fewer partly filled blocks per expert means less trellis decoding.
- A 16K chunk measured MoE 6.8k vs 6.6k tok/s at 8K [measured B70, `:105`].
- MB = 64/128 is a projection only (1.24 -> ~0.7 s per 8k chunk).
- For us: run the MoE once over all rows of a super-chunk (activations: `y` is npairs x 2560 fp16, 420 MB at 8192
  rows), or at least 8192-row pieces. My estimate is ~0.15 s on a 5.4K prompt. MB = 64 needs new DPAS tiling in
  `GateUpDpasKernel` / `DownDpasKernel` (large GRF already on).

## 10. MTP / speculative decoding [projection only]

- `experimental/n113-mtp/`: a plan and a plugin override (MTP experts pinned in VRAM, a pruned-vocabulary draft head
  at `plugin/sglang_plugin.tier.py:581-583` that stays exact because the full head verifies). "Projected 32-43 tok/s
  exact", never run (`docs/results.md:162`).
- Strata (second-hand, `research/19...:7-13`): accept 2.6-3.4 tokens/round on code, 55-72% on prose, with a
  confidence gate and a reduced-vocab draft head.
- **Why it's low for us:**
  - Their measured resident MoE cost grows almost linearly with rows: 3.0 / 5.8 / 10.0 ms at B = 1/2/4
    (`research/18...:21`). A verify step therefore costs nearly k times the expert bytes, and RAM picks multiply too.
  - Our own SGLang run measured accept ~1.8 of 3, +5% on short turns and a loss on long context
    (`llm/flashnext/README.md`, MTP bullet).
- Revisit only if a GM-text profile shows non-MoE work dominating.

## 11. Fused decode kernels [CPU only]

- `experimental/n114-decode/README.md`:
  - GDN decode in 2 kernels per layer (conv; recurrence + gated norm, optionally with `in_proj_ba` folded in);
  - hyper-connection mix in 3-4 kernels with weights read once for up to 8 rows;
  - decode QSA in 2 kernels per layer.
- Their gain is mostly fewer graph nodes, "~5 µs per graph node" (Strata figure), relative to SGLang's ~40-node
  layers.
- Our C++ graph is already fused much more than SGLang's. Count our nodes per token before porting anything: at
  ~5 µs each, 1,000 nodes would be 5 ms.
- The cross-site fold (one half's combine/apply fused into the next mix's norm, `kernels/n107-hc/README.md` "Further
  fusions") saves one pass over X per site. That is ~168 MB per call at 8192 rows, a prefill gain only.
- The hyper-connection weights are a 1.26 GB/token floor in fp16 (~2.1-2.6 ms). int8 would halve that but changes
  output. Our SGLang path found int8/fp8 HC GEMVs slower inside a graph anyway (`llm/flashnext/README.md`).

---

## Pitfalls they found on this hardware

Each item is measured on their B70 unless it says otherwise.

1. **D2H is a third of H2D.** 8.8 GB/s to pinned USM, 4.2-5.8 GB/s to memfd, vs 26.7-28.4 GB/s H2D (`n109_C.json`,
   `STATUS_N111.md` "D2H blocking"). D2H into pageable memory blocks the submitting host thread for the whole copy
   (~0.41 ms per expert).
2. **Never write host memory from the decode critical path.** An in-kernel victim store to host costs ~0.9-1.0 ms per
   1.86 MB expert. srv26 with no write-back ran 34.5 vs 24.0 tok/s (`docs/results.md:140-150`).
3. **The device must not publish residency.** The device set "resident" after the host's lagged snapshot while a hole
   punch was still queued. The page was zeroed while flagged resident: 50-52 bad output rows in a stress sim
   (`STATUS_N111.md` section 3, `docs/results.md:166-170`). Rule: only the host flips residency flags, and only after
   the data landed. Our current `rebalance()` follows this (host-side, after `q.wait()`); keep it that way if
   write-through goes in.
4. **Host-USM coherence [code comments, not a published measurement]** (`ngram_nvme_xpu.sycl:81-82, 675-677,
   715-717`):
   - "GPU reads of host USM on B70 do not snoop the CPU caches". They flush host stores with clflushopt + sfence before
     publishing.
   - GPU loads and atomics to host USM "can sit in the GPU L3 between kernels of one in-order queue".
   - A system-scope atomic spin inside a kernel never saw the host's store; only ESIMD LSC loads with L1+L3 uncached
     hints did.
   - **Possible hazard for us:** `rebalance()` overwrites RAM records with a host `memcpy`, at addresses kernels read
     earlier, and the next kernels read them. If either claim holds for plain reads, a swapped expert could be read
     stale (CPU cache not written back, or old lines in GPU L3).
   - Our checks build records once at load and never test after a swap. Add a check that compares a forward through
     swapped experts against the same experts at their VRAM address, or write the swap with non-temporal stores plus
     sfence.
5. **Atomics on host USM fault the GPU** (our own finding, `llm/flashnext/README.md`, the n-gram writes bullet; their
   upstream `nvme_publish` used system-scope atomics). Keep host-USM writes from the GPU as plain or uncached stores.
6. **Copy/compute overlap depends on the queue** [measured, conflicting]:
   - "a queue built here with `sycl::queue(ctx, dev, in_order)` did not overlap its memcpys with compute on B70
     (X006b v1)", while memcpy on a torch side stream did (`exl3_moe.a8.sycl:1096-1099`).
   - `research/18...:18-19,83-85`: in one process, side-stream H2D and compute "never overlap", across two processes
     they do.
   - The later N130 server shows 0 ms H2D wait, so it is possible in one process.
   - Our `Dev::copy` is exactly `sycl::queue{ctx, dev, in_order}` (`src/rt.cpp:22`). Before building items 1, 4 or 7
     on it, time one copy beside one long kernel.
7. **SVM first touch is slow.** A freshly filled 2 MiB page costs 1.9 ms (0.97 GB/s) when the kernel takes the fault,
   and 24 GB/s after (`n109_A.json` `ft`; `exl3_moe.a8.sycl:775-776` says 5-8 ms). `q.prefetch` on the range first
   helps. We use pinned USM, so this only matters if SVM ever comes back.
8. **`lsc_prefetch` in the decode GEMV was slower**: M = 1 went from 82 to 128-130 µs (`exl3_moe.a8.sycl:76-79`).
9. **SGLang's fused GDN backend** (`--linear-attn-backend intel_xpu`) produced runaway outputs (3 of 4 streams to ~57k
   tokens) (`docs/results.md:135-136`). Be careful porting that chunked GDN kernel for numerics, not only speed.
10. **Box health:**
    - marginal PCIe links; both B70s dropped off the bus once and needed a BMC power cycle (`docs/results.md:171-173`);
    - a stuck khugepaged/kcompactd THP collapse froze page faults on the box (`research/18...:141-144`);
    - stopping the container OOM-killed the scheduler as it faulted in a sparse 48 GB mapping
      (`docs/results.md:73-75`).
11. **Strata's list** (second-hand, `research/19...:44-48`):
    - VRAM over-allocation on xe evicts to host and can livelock the box; keep at least 1.5 GB free (we keep 2 GB,
      `q4.h:78`);
    - the L0 v2 event wait can deadlock (poll a sequence number, or `SYCL_UR_USE_LEVEL_ZERO_V2=0`);
    - unbounded device spins wedge the GT;
    - `SYCL_CACHE_PERSISTENT=1` segfaults on Xe2;
    - misaligned 16-byte loads split (2.3-4.7x per kernel);
    - sub-group 16 races in code written for warp 32;
    - joint_matrix XMX kernels ran 1.4-5x slower than their alternatives on the B70.

## Not useful to us

- NVMe expert store, admission control, decode-fill queues, masked picks, VRAM warm start (`nvtier.py:643-900`). They
  exist because they run with 16-32 GB of RAM and 8,000 slots. We keep everything resident and exact.
- The victim ring (`experimental/n111-victim-ring`). It fixes write-back from inside their kernel; we never write back
  from a kernel. Its design rules (host owns flags, copy-engine drain) are in pitfall 3 and item 7.
- Concurrency results (C2/C4) and the SGLang sync removals (`.item()`, `.tolist()`, `unique`). Our C++ path has none
  of these.

## What to measure in ours before building

All profiling (no GPU work beyond normal runs through the queue):
1. A per-layer prompt profile (`prof_point`) of a ~5K prompt: MoE routed-experts time per layer vs that layer's RAM
   expert count, "attn: select + softmax", "GDN: delta rule", the PLE host stall at layer 1. This orders items 1, 2,
   3, 8 and 9.
2. One copy on `Dev::copy` beside a long compute kernel: does it overlap (pitfall 6)?
3. Offline: how often the router on site-0 `y16` predicts the MoE's real top-10 (item 4), using `grab()` dumps.
4. `expert_stats()` on generic text with their prior blended in (item 5).
5. Decode graph node count per token (item 11).
