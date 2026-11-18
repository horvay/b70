# Speed review

Reviewed 2026-11-01. Target: beat the roughly 95 ms/token PyTorch quality-mode decode with the GM LoRA, while keeping the existing parity. No production source, kernel, test, Makefile, reference, or README was edited by this review.

For implementation, I would start with phase graphs, LoRA layout and fusion, and tiled MLA while investigating cold expert memory. The ranking below puts the larger but less certain memory opportunity ahead of LoRA. Work on exact routing immediately if long-context exact mode matters. The measured expert phase really is mostly GPU execution. Replacing its C++ containers will not recover 20 ms.

The gain ranges below are engineering estimates, not measured improvements unless explicitly labeled. They are reductions from the reviewed implementation, not promises, and they overlap. A reasonable target for the first combined round is **20-30 ms/token at the short quality workload**, enough to approach 90-100 ms/token. Reaching 95 ms is plausible; the evidence does not yet establish it. At long context, exact routing and storage traffic dominate the available savings.

## Scope and evidence

Read `README.md` and `docs/forward-spec.md` first, then the model, runtime, linears, LoRA, attention, routing, MoE, tier, prefill, serving and sampling paths. Compared with the read-only reference in `llm/glm/xpu`, especially `glm_xpu/block_graphs.py`, `mla_graph.py`, `sparse_mla.py`, `tier.py`, `ops.py`, `csrc/tier_native.cpp` and the SYCL kernels. Reference paths below are relative to `~/b70/llm/glm/xpu/`; engine paths are relative to this directory.

The checkout changed during the review, as expected. File:line references describe the version first read unless marked current. `review/source-sha256.json` records that version's hashes. The private build's `model.o` and `mla.o` predate the concurrent position-buffer and tiled-attention changes. `review/model-snapshot.h` matches the original `model.h` hash, avoiding a header/library ABI mismatch while compiling the diagnostic executable. Do not rebuild the private objects from the changing checkout and assume they still describe these measurements.

Existing measurements were reused, not rerun as baselines. Copies of the relevant logs are in `review/prior-*.log`:

- `prior-prof3.log`: the supplied short quality profile, 118.34 ms in phase intervals, 120.9 ms/token wall, LoRA on.
- `prior-moeb2.log`: hot and cold expert-memory measurements. The rotating-set tests appear in this existing log even though the current `tests/moe_bench.cpp` no longer contains those additional cases.
- `prior-moesweep.log`: split tuning and input-Hadamard probes.
- `prior-stage.log` and `prior-stage2.log`: speculative RAM-to-VRAM staging lost time in the reference.
- `prior-long1eng.log`: 266.1 ms/token, 6,868-token prompt, exact routing, LoRA; KL 0.0022, NLL 1.6889 versus 1.6844.
- `prior-imp2.log`: importing anonymous host memory silently used VRAM; it is not a solution for a true RAM tier.
- `prior-aot1.log` and `prior-prio.log`: AOT and queue-priority results.

Only three GPU jobs were added, all through `b70q submit sol-review` followed by `b70q wait`. Each had a timeout below ten minutes:

| Job | Purpose | Artifact |
|---|---|---|
| 1040 | Short quality decode, LoRA, 100 forced steps; actual MoE event durations and complete tier counters | `review/nq-trace.log` |
| 1041 | SYCL graph capability and replay smoke test | `review/graph-smoke.log` |
| 1042 | Long exact decode, LoRA, 100 forced steps; phase and MoE event durations | `review/mid-trace.log` |

Build: `B=build-review ./build.sh B=build-review`, then `bash review/build-trace.sh`. The latter links a host-instrumented copy of `src/moe.cpp` from `review/moe_trace.cpp`; kernel arithmetic is unchanged. `review/relink-trace.sh` relinks just the diagnostic runner. `bash review/build-graph.sh` builds the graph smoke test. `review/build-sha256.txt` identifies the resulting libraries and objects. `ldd build-dev/decode_check` showed SYCL, UMF and oneMKL, with no torch, c10 or Vulkan dependency.

### What the new measurements say

| Phase, ms/token | Short quality, LoRA | Long exact, LoRA |
|---|---:|---:|
| mHC + norms | 3.96 | 4.01 |
| KDA | 15.05 | 15.05 |
| MLA | 9.04 | 12.86 |
| Dense/shared MLP | 6.16 | 6.21 |
| Router + picks copy | 3.17 | 17.02 |
| Tier interval | 34.75 | 88.90 |
| Expert interval | 45.51 | 88.57 |
| Embed/head | 1.13 | 1.13 |
| Phase total | 118.76 | 233.77 |
| Wall decode | 121.3 | 237.8 |

These are diagnostics, not replacement README numbers. `decode_check` resets the counters after producing the first forced distribution, then performs 99 new forwards, but divides the phase report by 100. Multiply phase values by 100/99 when comparing with per-forward kernel totals. Wall decode already divides by 99. The profiler also inserts 309 barrier events per forward. Their cost and the normal run's timing variation must not be credited as production speedups.

Short diagnostic parity: KL mean 0.0067, max 0.0435, top-1 0.960. Long diagnostic parity: KL mean 0.0022, max 0.0143, top-1 0.990, NLL 1.6889. Quality-mode routing changes with cache readiness, so this run's short KL is not a bitwise-repeatability claim.

The old 266.1 ms long result and this 237.8 ms diagnostic have the same reported long parity and NLL. That 28 ms variation alone rules out explaining the old wall result with MLA code inspection. Compare the long exact workload with short **exact**, roughly 220 ms in the brief, rather than with short **quality**, roughly 121 ms. Both context and routing mode otherwise change together.

### Why experts cost more than the simple estimate

The short run executed 4,158 MoE calls, or 42 layers times 99 forwards:

- VRAM hits: 21,061, or 212.74/token.
- Admissions to VRAM: 1,601, or 16.17/token.
- Actual zero-copy picks: **10,602, or 107.09/token**.
- Reported fast RAM hits: 7,326, or 74.00/token.
- Demand reads: 589; used prefetches: 2,728. These are slow-path categories, not extra fast-RAM hits.

The counters reconcile exactly: `21,061 + 1,601 + 10,602 = 33,264 = 4,158 * 8`. In `tier.cpp:533-549`, `ram_hit` counts only ready RAM picks that stay on the fast path. Slow-path zero-copy use increments `zero_copy` at `:615-628`, not `ram_hit`. Thus the supplied 73 RAM hits substantially undercount PCIe expert execution. `zero_copy` should be printed in the normal summary.

Actual event execution, divided by 99 forwards:

| MoE component | Short quality | Long exact |
|---|---:|---:|
| Gate/up kernels | 31.152 ms | 60.910 ms |
| Down kernels | 14.291 ms | 27.335 ms |
| Combine kernels | 0.088 ms | 0.089 ms |
| Gaps between those kernels | 0.027 ms | 0.022 ms |
| Host time submitting MoE calls | 1.685 ms | 4.069 ms |

Host submission overlaps execution; do not add it to kernel time. The short phase interval, corrected for its denominator, is 45.97 ms, versus 45.56 ms for the three kernels and their internal gaps. Only roughly 0.4 ms remains for the shared-output add, boundary events and the pre-kernel gap. There is no hidden 20 ms host gap inside this expert interval.

Using the supplied approximation with the corrected zero-copy count gives `42 * 0.19 + 107.09 * 0.24 = 33.68 ms`, already 8.2 ms above the original estimate based on 73 picks. Using the measured hot mixed-memory benchmark gives `42 * 0.247 + 107.09 * ((0.718 - 0.247) / 2) = 35.59 ms`.

The hot benchmark is not the full model. The existing cold-memory log reports 2.532 ms for a repeated all-RAM set, 3.656 ms for a different spread set each call, and 3.053 ms on a second pass through those sets. Its cold incremental coefficient is approximately `(3.656 - 0.248) / 8 = 0.426 ms/RAM pick`. Applying that coefficient and the fixed VRAM cost gives about **56.0 ms**, not 45.6 ms. Actual execution at 45.53 ms lies between the hot estimate and this cold extrapolation. Mixed-memory latency is not strictly linear, so these are diagnostic bounds, not a fitted hardware model.

The kernel alone matches the reference because it uses the same implementation and split choices: `moe.cpp:20-48`, `kernels/exl3_moe_kernels.h:102-334`, reference `ops.py:461-481`. The full forward has a much larger allocation/translation working set, different expert addresses each layer, active storage-reader copies, and background write-backs. The isolated benchmark does not. The long run has 141.52 zero-copy picks/token, about 1.12 GB of expert records, 127 speculative reads/token, 54.3 used prefetches and 20.9 demand reads. Its 88.33 ms of actual kernels exceeds even the simple cold extrapolation of about 70.7 ms. Concurrent memory traffic and address-working-set effects still need hardware-counter separation. The evidence establishes the accounting error and real GPU execution; it does **not** identify TLB misses versus DDR contention versus PCIe transaction efficiency conclusively.

### Launch and dependency comparison

Static count for Q=1 quality decode with this adapter, excluding copies and profiling barriers:

| Work | Kernel launches/token |
|---|---:|
| 34 KDA layers, 33 each | 1,122 |
| 11 MLA layers, 35 each | 385 |
| 45 dense/shared MLPs, 16 each | 720 |
| mHC mix/norm/apply | 360 |
| Routed experts and shared-output add | 168 |
| Current/predicted router GEMVs and routing tails | 125 |
| Expand, mean, final norm, head | 6 |
| Total | **2,886** |

The 428 loaded modules are **417 runtime LoRA applications plus 11 merged MLA kv_b modules**, from `model.cpp:110-131`. Runtime LoRA therefore adds 834 kernels; the g_a/f_a output casts add another 68. It does not add 856 runtime launches just because the log says 428 modules.

Exact mode adds 126 launches for duplicate current-router GEMVs, serial exact tails and picks overwrites. Both modes also submit 42 picks copies and up to 42 pointer-row and 42 bonus-table uploads, plus admissions/write-backs. `GLM_PROF=2` adds 309 phase barriers.

Reference decode captures A/Am, B, S and C graphs. There are nominally 177 phase replays/token, not 177 device kernels. C++ runs fewer incidental tensor operations than Python, but submits almost 2,900 kernels individually. Reference `block_graphs.py:431-451,454-501,583-681` captures LoRA, attention, experts and residual updates inside these phases.

Both implementations already do this sequence:

1. Submit routing and an asynchronous picks copy.
2. Submit the independent shared expert.
3. Wait only for the picks-copy event.
4. Start demand reads, then speculative reads, resolve picks and admissions.
5. Submit routed experts after required data and pointer uploads.

Engine evidence: `model.cpp:217-255`; reference: `block_graphs.py:596-608,634-676`. The comment about overlap is substantially implemented. More overlap requires faster submission, earlier routing/prediction, a separate dependency chain, or less tier work. Moving the existing shared expert above `pe.wait()` does nothing because it is already there. `GLM_PROF=1` drains the queue at every mark and destroys this overlap; use it only as a diagnostic.

## Ranked decode opportunities

Numerics labels describe the implementation itself. **Bit-identical** means preserve the original operation order and rounding. **Rounding-only candidate** still needs validation because tiny changes can flip MoE picks. **Quality-affecting** changes selection, precision, represented weights or skipped work. In quality mode, even a mathematically exact cache optimization can alter the residency bonus. Preserve the logical bonus/residency decisions separately from temporary physical copies, or treat that end-to-end change as quality-affecting until measured.

The numbered list ranks the short quality workload by expected gain, with long exact estimates beside each item. For long exact, prioritize item 2 at 15-35 ms, item 3 at 8-20 ms and item 6 at 11-14 ms, then graphs at 5-10 ms and MLA at 5-9 ms. Item 6 has stronger evidence than the more uncertain memory gain. These ranges share work and are not additive.

### 1. Replay phase graphs

**Expected gain: 5-10 ms/token in either mode. Confidence: medium; graph capability measured, whole-model gain unmeasured.**

Evidence: the launch inventory above; no graph path in `model.cpp:460-504`; one submission for every small operation in `exl3.cpp:135-182`, `lora.cpp:150-185` and `kda.cpp`. Job 1041 reported full and limited graph aspects supported, correctly executed the recorded work, and replayed a 32-kernel ordinary-SYCL graph in 40.372 us, 1.262 us/kernel. That proves this compiler/device combination can replay graphs. It does not yet prove mixed ESIMD capture or the whole model's benefit.

Implementation: record the reference's A/Am, B, S and C boundaries first. Keep the picks copy and host tier decision between B and C, with S queued before waiting for the copy. Capture fixed workspaces and pointer-table addresses; update table contents, not allocations. Move position and lengths into stable device scalars. Bucket MLA score grids if scoring the entire maximum context wastes work. Keep runtime allocations, `wait`, event queries and CPU tier actions out of recording. The concurrent checkout already has a position-buffer change in progress.

The 5-10 ms estimate corresponds to roughly 2-4 us of removable dispatch/idle cost across about 2,700 eliminated host submissions. GPU execution time remains. This overlaps the launch-related parts of items 4, 5, 7 and 9; do not sum them independently.

Risk: kernel arguments/workspace addresses captured from an old layer, unsupported recording of a library operation, and graph variants growing with Q or context. Test ordinary SYCL plus the ESIMD MoE/EXL3 libraries together before committing to the design. Numerics: bit-identical for the same routes and dependency order.

### 2. Start urgent tier work sooner and reduce exposed storage waits

**Expected gain: 4-10 ms/token short quality; 15-35 ms/token long exact. Confidence: medium-low; these are targets within measured waits.**

Evidence: supplied NVMe wait 19.8 ms/token; diagnostic 24.3 short and 71.5 long. `model.cpp:237-252` starts host demand work only after submitting the shared expert's 16 kernels and returning from `pe.wait()`. `tier.cpp:469-498` starts urgent reads before speculative reads correctly, but a missed prediction gets only the remaining shared-expert execution to hide it. The long run reads far more speculative data per useful hit than short quality.

Implementation: graph S so the host reaches `after_sync` quickly; keep routing-copy synchronization scoped to its event. Split a fast demand-dispatch routine from optional maintenance. Submit all urgent pieces first, then do speculative planning, reaping and admission bookkeeping while those reads run. Dispatch predictions before unrelated container maintenance. Record demand queued/start/read/copy/published timestamps, not only `t_nvme`. Put a byte/outstanding-read limit on speculative work and evaluate prediction confidence, rather than blindly increasing width. Keep the established width 12 as the initial policy; previous width sweeps already favored it.

A one-layer shared expert supplies at most about 6 ms/token of GPU work across the model to overlap. Moving demand dispatch alone cannot hide all 20-24 ms of waits. Larger gains require earlier useful predictions or faster actual reads/copies. A two-layer predictor is an experiment, with additional router cost and false-prefetch traffic, not a free extension.

Risk: speculative reads evict useful RAM entries, starve urgent I/O or change bonus availability. Numerics: scheduling and copied bytes can be bit-identical with unchanged logical residency; altered prefetch/admission policy is quality-affecting in bonus mode unless isolated from bonus decisions. Exact mode has no such residency-selection dependency.

### 3. Improve the cold zero-copy expert path, using selected demand copies as a controlled alternative

**Expected gain: 3-8 ms/token short; 8-20 ms/token long. Confidence: low-medium, bounded by the measured hot/cold gap.**

Evidence: actual kernels take 45.53/88.33 ms; the corrected hot estimates are about 35.6/43.7 ms. Cold sets slow the existing microbenchmark. Approximately 0.85/1.12 GB of host expert records are consumed per token, before admissions or write-backs. `kernels/exl3_moe_kernels.h:128-188,260-298` reads trellis strips from arbitrary host addresses while reader threads copy other experts into the same large RAM allocation.

Implementation: first capture PCIe bytes, DRAM bandwidth and translation/cache counters for these existing workloads. Use rotating expert sets and concurrent reader/write-back traffic in any new microbenchmark. The repeated-eight-expert benchmark is already available and should not be rerun as the only evidence.

An exact alternative is to copy **only currently selected RAM experts** into temporary VRAM rows, then run the same kernels against those rows. Leave logical tier residency and bonus unchanged. Source RAM slots must remain held until copy completion; temporary rows must remain held until the expert completion event. Measure `copy + compute`, not just the now-fast VRAM kernel. At 25 GB/s, the short run's 0.85 GB of copies alone costs about 34 ms, plus roughly 10 ms for all-VRAM expert compute. With no overlap that barely beats the measured expert phase. A substantial gain therefore needs useful overlap, reduced cold-transaction overhead, or measured higher effective transfer efficiency. Do not promise 20 ms from staging on the strength of the all-VRAM number.

Other exact candidates: larger contiguous read transactions, translation-friendly host allocation and page mapping, pre-touch of all relevant mappings during warm-up, and more memory-level parallelism without spilling. Huge-page backing or host-memory registration must be shown to retain real host residency. The imported-memory experiment already demonstrated a false apparent speedup caused by VRAM migration.

Risk: copy/compute contention and VRAM scratch reducing the persistent cache. Numerics: byte copies and unchanged expert reductions are bit-identical, subject to the logical-bonus rule. Structural expert math changes are unnecessary here.

### 4. Fix LoRA memory layout and exploit the qkv block diagonal

**Expected gain: 3-6 ms/token, including some dispatch savings. Confidence: medium.**

Evidence: `lora.cpp:97-122` builds dense rank-192 qkv B matrices even though two thirds are zero. `:169-184` has one scalar output per work-item looping over 64 or 192 rank entries. Consecutive output lanes read B rows separated by R half elements. Across 417 runtime modules that is 834 launches. The supplied paired phase profiles show LoRA adds about 5.1 ms in KDA, 1.1 ms in MLA and 2.5 ms in dense/shared MLP. These phase deltas are useful scale estimates; tier differences prevent interpreting the whole run as an isolated LoRA experiment.

Implementation:

- Upload an extra decode layout `B[rank][out]` so lanes producing neighboring outputs load neighboring values for each rank. Keep the same j loop and fp32 accumulation to retain the existing sum order. Keep the row-major layout for prefill GEMMs if needed.
- Run qkv B as three rank-64 blocks in one launch, reading only each block's nonzero entries. Preserve each output's nonzero rank order. Handle signed zero and nonfinite inputs explicitly before claiming bit identity.
- Group independent A projections sharing the same raw input. KDA qkv, b, f_a and g_a and MLP gate/up are candidates. Use one launch with independent reduction groups, not a sum across unrelated modules.
- Fuse B accumulation into HadOut for EXL3 targets, preserving the base output's original rounding first. For fp32 outputs, keep `half(delta)` before widening/addition. For fp16 outputs, accumulate into the base value and round only at the original final point.

Risk: algebraic LoRA merge/requantization changes the represented model; do not do it. Simply replacing the fp16-output path by `half(base + half(delta))` also changes numerics. Numerics: transpose/grouping can be bit-identical; a new vector reduction or XMX implementation is a rounding-only candidate requiring parity checks. Sparse qkv removes structural zero work, not model contributions.

### 5. Replace the naive MLA attention with tiled gather/GEMM/softmax/GEMM

**Expected gain: 3-6 ms/token short; 5-9 ms/token long. Confidence: medium. Ceiling: the measured 9.0/12.9 ms MLA phase, not the whole long-context slowdown.**

Evidence: reviewed `mla.cpp:223-269` launches one work-group per head/row. QK gives a token to one lane and runs a serial 512-element dot; neighboring lanes access different 1 KiB rows. PV runs a serial loop over up to 2,051 rows for every latent dimension and head. At full selection, the two passes logically load about 269 MB of latent data per MLA layer before caching, versus a 2.1 MB gathered latent matrix shared by all heads. Across 11 layers this is substantial repeated traffic. Reference `sparse_mla.py:15-40` gathers once per query row and uses two batched GEMMs.

Implementation: gather selected rows once to contiguous scratch, batch all 64 heads as GEMM rows for QK, apply the original score rounding/scale/mask and fp32 softmax, cast normalized probabilities to fp16, then GEMM PV. For prefill, process query blocks so gathered KV and score scratch stay bounded. Tile absorb/unfold too, especially prefill; `mla.cpp:79-85` and `:272-282` are simple scalar/subgroup products over large unquantized weights.

For a lower-risk first step, coalesce QK and share gathered rows across heads while retaining the current rounded intermediates. The concurrent source now contains a tiled step-9 implementation around current `mla.cpp:228`. `prior-att2.log` reports 213.7 ms/token and 26.1 s prefill at long exact, versus the old 266.1 ms/37.4 s. It also reports 108.4 ms/token for 200 short quality steps. These are useful implementation results, but not an isolated measurement of MLA's gain. The long NLL changed from 1.6889 to 1.6980, and KL from 0.0022 to 0.0026. Check the intended numerical tolerance rather than calling the entire 52.4 ms wall reduction an attention-only or parity-neutral gain.

Risk: changed reduction order, local-memory/register pressure and scratch. A conventional online flash softmax changes the fp16-normalized-P boundary. Numerics: a new GEMM/tiling scheme is a rounding-only candidate if all documented casts remain; online softmax that removes those casts is quality-affecting until separately accepted. Selection order must remain the implementation's current order unless that change is validated too.

### 6. Remove duplicate exact routing and parallelize its serial tail

**Expected gain: 11-14 ms/token in long exact mode; 0-1 ms/token in quality mode. Confidence: high that the work is avoidable.**

Evidence: exact router phase is 17.02 ms/token in job 1042 versus quality's 3.17. `model.cpp:221-235` first runs current/predicted GEMVs and `route_tier`, then calls `route_exact`, which repeats the current GEMV and executes the whole 288-expert top-8 on one work-item per row in `ops.cpp:246-274`. It then overwrites picks in another kernel. This repeats the exact kind of serial-router bottleneck already fixed for the bonus tail.

Implementation: reuse the first current-router logits; route current exact selections directly from the fp16 centered bias. Use the bonus tail's group-max method with the exact bias, tie rule and final half-rounded weights. Write both selected and exact pick sections once. Continue generating predictions with their original fp32 bias, since predictions and exact bsz-1 selection have different documented numerics. Preserve the original serial order of the eight selected-weight additions, rather than using an arbitrary tree sum.

Sketch: replace the `if (!bon)` recomputation with `route_exact_from_logits(lc, bias_c, ...)`, and specialize the current branch of `route_tier` so it does not compute an unused fp32-bias selection. The upper bound is the 17 ms phase; allow 3-6 ms for remaining projections, predictions, picks transfer and dispatch.

Risk: fp16 centered bias is not interchangeable with the quality router's uncentered fp32 bias. Numerics: bit-identical is attainable, including tie order and sequential weight normalization. Directly reusing quality weights is quality-affecting.

### 7. Fuse KDA preparation and improve recurrent-state access

**Expected gain: 1-3 ms/token after graphing; another 1-2 ms of dispatch benefit may appear before graphs. Confidence: medium-low until subkernel timings exist.**

Evidence: 33 launches/layer with LoRA, 34 layers. `kda.cpp:83-124` separates forget activation, beta, qkv cast and conv. `:135-189` stores state as `[h][k][v]` while adjacent lanes own k rows, so a load for fixed v has a 512-byte lane stride. Four value-column tiles repeat q/k normalization. The implementation has five group barriers per token in the recurrence, despite the comment mentioning three.

Implementation: fuse qkv fp32-to-bf16 into conv; fuse beta/gate preparation where dependencies allow; precompute normalized q/k once per head and decay values once per channel. Preserve bf16 product rounding inside the conv, bf16 beta, bf16 core and fp16 gated-norm output. Store recurrent state physically as `[h][v][k]` so lanes across k read contiguous values, adapting checkpoint/restore layout consistently. Compare VB=8/16/32 with register and occupancy data. Remove the final next-iteration barrier when S=1 and no cross-lane consumer remains. Hoist `exp(A_log[h])` at model load without changing its value.

Risk: changing subgroup/reduction geometry changes the recurrence indefinitely, and state layout mistakes survive many tokens. The ideal state traffic is only 34 times 8 MiB per token, so state bandwidth alone cannot account for all 15 ms. Most savings must come from dispatch, access efficiency and redundant reductions/projections. Numerics: layout/cast-preserving fusion can be bit-identical; new reduction trees are rounding-only candidates with long-sequence checks.

### 8. Copy victims on the compute engine, and make background-copy dependencies explicit

**Expected gain: 0.5-2 ms/token; possibly more host submission relief in exact mode. Confidence: medium-low.**

Evidence: `tier.cpp:432-433` does a D2D `q.memcpy` before a D2H copy on `Dev::copy`. Reference `tier.py:1110-1118` explicitly uses an elementwise D2D copy because its D2D copy-engine path blocked the host for about 100 us. At 15.4 write-backs/token that is a plausible 1.5 ms host effect, though this runtime may behave differently. The roughly 120 MB/token of D2H traffic is real either way.

Implementation: use a simple vectorized byte/uint64 copy kernel for VRAM slot to victim ring, then retain an event dependency for ring-to-host copy. Measure host API time and actual D2D/D2H event duration. Confirm which engine the runtime uses; a queue named `copy` is not proof of a dedicated hardware copy engine. Keep the decode and copy queues at compatible priorities. Existing low-priority results show background starvation, so giving decode high priority is not an automatic improvement.

Risk: global barriers serializing unrelated work, or a ring slot returned before D2H completes. Numerics: byte-identical. Dropping write-backs instead is a cache-policy change and is quality-affecting in bonus mode.

### 9. Fuse dense/shared MLP epilogues and mHC boundaries

**Expected gain: 1-3 ms/token for MLP work, 0.3-1 ms for mHC, overlapping graph/LoRA savings. Confidence: medium-low.**

Evidence: `model.cpp:241-245` runs three independent EXL3 linears plus SwiGLU; `exl3.cpp:159-181` uses HadIn, GEMV and HadOut for each. `model.cpp:257-260` adds the rounded routed result to shared output in another kernel, followed by `hc_apply` at `:479`. mHC costs about 4 ms total and 360 kernels, so it is worth fusion but cannot deliver 10 ms.

Implementation: dispatch gate/up base projections and their LoRA work together while preserving distinct suh vectors. Combine the two HadOut/LoRA epilogues with the original fp16 gate/up roundings and clamped SwiGLU. Fuse expert combine, the **fp16 rounding of the routed sum**, the fp32 shared add and the subsequent mHC apply. Fuse attention apply with the next mHC partial calculation where possible. Fold collapse plus weighted RMSNorm together only if the collapsed fp16 values are still formed and the reduction order is preserved.

Existing fused EXL3 support in `kernels/exl3_esimd.h:242-294,456-617` is not wired into `src/exl3.cpp`. Its atomic last-arrival HadOut path is an experiment, not a drop-in speed win. Keep partition ordering, memory visibility and counter reset correct, and handle fp32 outputs/LoRA explicitly.

Risk: replacing the routed fp16 sum with an fp32 add changes the model. Removing activation clamping is also wrong. Numerics: bit-identical fusion is possible; altered split/reduction order is rounding-only and must be checked.

### 10. Shrink host tier overhead without overengineering eight-element sets

**Expected gain: 0.1-1.5 ms/token from containers; 0.2-2 ms from eliminating excessive event queries. Confidence: low until host timers separate them.**

Evidence: `tier.cpp:472-497,520-531,576` allocates small vectors repeatedly; `:366-390` scans in-flight reads/write-backs; `:381` queries each write-back's device event every ensure. `Reader::done` locks a shared map, `:91-95`. `has()` is a linear find, but decode sets contain only 8 or 12 elements. A hash set for eight integers may be slower. Do not assign all 31 ms of the tier phase to these operations: it includes storage waits and admission copies.

Implementation: fixed stack arrays or reusable small-vector scratch for Q=1; generation-tagged per-key marks for larger Q; flat per-key arrays for job IDs/flags where key bounds are known. Retire write-backs by completion order and query the oldest pending event rather than every entry, respecting actual completion ordering. Limit reaping to the entries relevant to reuse, then batch maintenance. Retain error propagation. Profile `after_sync`, event queries, slot search, submission and waits separately.

Risk: stale key/slot generations and reordered admission victims. Numerics: bit-identical only if logical victim/admission ordering remains unchanged. Avoid replacing deterministic scans with an unordered iteration order.

### 11. Reduce reader copy pollution and tune thread placement

**Expected gain: 0.5-3 ms/token short; 2-8 ms long, largely overlapping item 3. Confidence: low.**

Evidence: `tier.cpp:128-159` performs O_DIRECT pread into a thread-local bounce buffer, then ordinary `memcpy` into pinned USM. Sixteen readers, eight pieces per expert, shared deque/map/condition variables and completion atomics interact with the host decode thread. The source records no copy-only time. Actual topology from `lscpu`: physical cores 0-7 plus SMT 16-23 share L3 0; cores 8-15 plus SMT 24-31 share L3 1. There is one NUMA node, not two NUMA nodes.

Implementation: measure `t_copy` separately from `t_pread`, bytes and ready-to-publish latency. For sufficiently large aligned pieces, try non-temporal destination stores with the necessary fence before the final completion publication. Tune urgent piece count, reader count and per-job allocation pooling; more readers can hurt a memory-bandwidth-limited workload. Give the decode submission thread a stable physical core and keep reader workers off its SMT sibling. Compare reader placement within/across CCDs only if there is no existing placement result. Leave enough cores for Greg's other work.

O_DIRECT directly into USM host memory already fails in the reference. Do not spend another round rediscovering that. An anonymous/imported-memory design requires proving host residency and coherency under CPU rewrites; the earlier import experiment consumed VRAM.

Risk: non-temporal visibility bugs, remote cache-line bouncing or worse CPU sharing. Numerics: byte-identical when publication ordering is correct. Changed completion timing can affect quality-mode bonus decisions.

### 12. Remove unnecessary transfers and exact no-selection work

**Expected gain: 0.2-1 ms/token; context-dependent. Confidence: medium-low.**

Evidence: picks transfer `model.cpp:237` copies two int64 top-8 lists plus 12 predictions, only 224 bytes; bandwidth is not the problem. Bonus upload copies all 43*288 floats at `tier.cpp:345-348`, pointer update copies 2,304 bytes at `:632`. Latency and ordering matter more than byte count. The bonus host buffer can be updated by readers while copies are in flight, so a redesign must also define a stable publication snapshot.

Implementation: upload only dirty bonus rows or device-side update records, using immutable per-submission staging buffers. Exact mode can omit redundant selected/exact buffers after item 6. A device-gathered array of eight current expert pointers can avoid broad pointer-row updates, but it still needs the host's residency resolution. Below 2,048 visible tokens, all complete pools are selected, so indexer score/radix work is mathematically unnecessary; replace it with the same ascending selection list only if it matches the current implementation's order, while continuing to update key/pool caches for later tokens.

Risk: broad host-atomic stores do not make an asynchronous DMA a consistent snapshot. Bypassing below-threshold scoring is bit-identical only after proving finite-score behavior, masks and ordering. Do not skip generating cache state needed above the threshold.

### 13. Tune expert arithmetic only after memory and dispatch

**Expected gain: 0-2 ms/token for unproven decode tuning; do not budget a large gain. Confidence: low.**

Evidence: the isolated expert kernel already matches PyTorch. `prior-moesweep.log` found precomputing input Hadamard had almost no benefit in the mixed case. GU split 4 changed 6V+2R from 0.706 to 0.662 ms/call, but all-VRAM slowed from 0.219 to 0.255. GU16/down16 and explicit prefetch were worse. `kernels/exl3_moe_kernels.h:59-60` records rejected prefetch settings. Compile spills in `review/build-trace.log` are for large-row DPAS/K4 variants, not the decode K25/MR1 kernel. They do not explain the short expert phase.

Implementation: keep power-of-two splits and preserve the existing accumulation partitions initially. Tune prefetch depth, transaction grouping and register use against cold mixed sets, with full end-to-end routing validation. Specialized integer codebook operations that provably return the same half bits are safer than moving the affine out of each decoded value. The optional EXL3 LUT needs cache-pressure evidence, not just fewer instructions.

Risk: split changes alter accumulation order; aggressive prefetch raises register pressure. Folding codebook affine out of the dot product removes intermediate fp16 rounding and is quality-affecting. Numerics: exact bit transformations are bit-identical; split/reduction changes are rounding-only candidates.

### 14. Keep sampling and streaming off the critical path in the server

**Expected gain: 0.2-3 ms/token in serving; zero claimed against `decode_check` kernel timings. Confidence: low until sampled.**

Evidence: `engine.cpp:49-54` downloads every logit and samples on the CPU. `sampler.cpp:24-61` converts the whole vocabulary, exponentiates it and fully sorts surviving candidates. The forced decode checker does no sampling. Do not claim a sampler optimization closes its measured 121-to-95 ms gap.

Implementation: greedy mode can reduce argmax on the GPU and download one ID, preserving lowest-index ties. For bounded top-k, a CPU partial selection followed by sorting the kept candidates can preserve the current comparator and distribution. Keep stochastic probability normalization/RNG semantics explicit. Avoid blocking network writes in the inference token callback; buffer output to a separate writer with bounded backpressure. Prefix restore is already implemented and should be retained.

Risk: different sorting ties, RNG or top-p normalization changes generated sequences. Numerics: exact greedy/identical candidate ordering can be bit-identical; stochastic GPU reductions are rounding-only candidates at best.

## Prefill opportunities, ranked separately

### P1. Use tiled MLA matrix operations and bounded query scratch

**Expected gain: 5-15 seconds on the old 6.9k prefill, roughly 1-4 seconds at 613 tokens. Confidence: medium; not additive to concurrent MLA work.**

`Model::prefill`, `model.cpp:394-400`, sends up to 2,048 rows through the same gathered implementation, repeating serial absorb, scoring and attention work for every row. Replace this with query-blocked gathered GEMMs or a parity-checked dense path below the selection threshold. Reuse gathered data across heads and avoid full-query intermediate replication. The old long prefill was 37.4 s; the concurrent tiled implementation reports 26.1 s, an 11.3 s observed reduction across that implementation, not an isolated kernel proof.

Risk/numerics: rounding-only candidate. The reference's MHA-form dense path materializes fp16 K/V and does not have identical rounding to absorbed latent attention. Do not silently switch formulations under the bit-identical label. `kernels/fa_esimd.h` is experimental, not wired in, has D=256 rather than latent D=512, and uses online softmax; it needs adaptation and a numerical decision.

### P2. Fix prompt read-ahead bookkeeping complexity

**Expected gain: 0.5-4 seconds for large prompts, workload-dependent. Confidence: low-medium.**

`tier.cpp:776-791` scans every ring slot to count free/completed slots for each planned expert. `free_ring_slot`, `:738-770`, scans again and queries completion events. With 437 ring slots and thousands of planned records, this is millions of slot checks and potentially many driver queries. This path is prefill-only and cannot explain decode time.

Maintain explicit free, reading and copying lists, retire copy completions in order and keep an O(1) free-count. Feed a bounded layer-ahead read window. Keep ascending staging order and exact same copied expert records initially. Numerics: bit-identical for exact prefill. Risk: returning a pinned ring slot before DMA has consumed it, or evicting a read still referenced by the stage table.

### P3. Stage prompt experts in parallel and overlap shared work

**Expected gain: 0.5-3 seconds at 6.9k; smaller at 613 tokens. Confidence: low.**

`tier.cpp:806-849` can issue a demand read and wait for it before advancing to the next unplanned expert; read-ahead often masks this, but the short diagnostic reported only 3,553 read-ahead hits out of 7,239 NVMe staged records. `model.cpp:428-442` resolves and stages the layer before submitting any shared MLP work for that layer.

Submit all unplanned reads for a staging batch, then wait as needed to copy them; overlap reader work with shared MLP chunks and independent attention/projection work using separate scratch. Keep a held source-slot list and release each only after its copy event. Avoid moving experts to VRAM twice. Numerics: bit-identical with the existing prefill calculations. Risk: staging scratch overwritten before routed kernels finish, or excessive speculative I/O.

### P4. Parallel KDA prefill

**Expected gain: perhaps 1-5 seconds at 6.9k; measure a KDA subprofile first. Confidence: low.**

`kda.cpp:145-189` loops serially through all S tokens in each state-tile group, unlike the reference's chunked FLA/WY algorithm for S>=64. A chunked XMX algorithm can trade more operations for much more token parallelism. Keep subchunk/checkpoint boundaries and short tails explicit.

Numerics: **requires separate approval as a numerical change** under this task's tolerance. Reference chunked KDA rounds normalized q/k and intermediate matrices to bf16; current recurrence keeps normalized q/k fp32. Even a mathematically equivalent parallel algorithm changes state-reduction ordering. First try exact cast/fusion/access fixes from item 7; do not present the full FLA port as guaranteed rounding noise.

### P5. Avoid repeated reconstruction and poor small-tail dispatch

**Expected gain: 0.2-2 seconds for multi-piece prompts; 0.1-0.8 seconds for small tails. Confidence: low.**

`exl3.cpp:117-123` reconstructs an fp16 weight slice for every call. `model.cpp:394-410,435-442` processes chunks, so the same layer's weights may be reconstructed repeatedly. Reuse one reconstructed linear slice across row blocks where memory allows, or run all rows of that linear before releasing the reconstruction. Prioritize layers whose temporary weights fit inside existing scratch. Keep the same GEMM shape/order when claiming bit parity.

`moe.cpp:57,82` switches abruptly at M=256; smaller tails use grouped GEMV. `ops.cpp:203-240` serializes up to 16 rows inside output groups. Consider batching tails with padding and dispatch tuning without changing valid-row arithmetic. The ordinary EXL3 DPAS MB40/48/64 variants spilled in the private build, so awkward 17-128-row tails deserve attention. Measure changes against existing prompt outputs, including per-layer routing, not just GEMM throughput.

Risk: higher resident scratch evicts experts and hurts decode, padding uses a different reduction path, or GEMM shape changes outputs. Numerics: reconstruction reuse can be bit-identical; algorithm/shape switches are rounding-only candidates.

## Memory and startup opportunities

### M1. Stop allocating prefill-sized logits and decode-only buffers

**Memory recovered: about 0.76 GiB with a small patch. Direct compute gain: zero; possible 1-4 ms/token through a larger physical expert cache, requiring quality-mode validation.**

`model.cpp:142-166` allocates every persistent buffer with Q=max(qmax,prefill_chunk)=2,048. But prefill uses its own Rall and never computes logits. `logits` alone is 605 MiB; qmax=8 needs only 2.36 MiB. `R` saves another 127.5 MiB when sized to qmax, and `fin`/`fin16` about 47.8 MiB. Decode-only picks, nlogits and route-output buffers can also shrink after checking all callers. `x_in`, y, collapse and attention/MLP chunk scratch still serve prefill and cannot all be reduced to eight rows.

Sketch: introduce separate `Qdecode=qmax` and `Qpiece=prefill_chunk` allocation groups. Allocate `R`, `fin`, `fin16`, `logits` and the actual decode-only route buffers using Qdecode. Numerics: bit-identical for the existing forward path. Increasing the persistent expert pool with reclaimed bytes changes bonus-mode routing unless logical bonus capacity stays fixed.

### M2. Size workspaces by live phase and budget actual peak memory

**Potential memory: another few hundred MiB to around 1 GiB, depending on lifecycle. Startup/phase transition cost must be included.**

Act is allocated at 684 MiB and MoE workspace at 320 MiB, `model.cpp:50-51`, plus the maximum linear reconstruction workspace and about 1,104 MiB of common persistent tensors before caches. Decode uses a small fraction. Separate decode and prompt scratch, reuse disjoint phase storage, and release prompt-only allocations only at an event proving completion. A prompt arena can be recreated for the next request; avoid per-layer allocation.

The 3 GiB reserve is not wholly available for experts: the victim ring is allocated after the expert budget, and large-prompt Rall/yall, checkpoints, future graphs and cache growth need space. `Tier::split_slots` protects the 288-slot staging minimum correctly. Tune reserve from measured peak allocation at the supported maximum context and account for every extra ring, rather than taking all reported free VRAM. Numerics: memory management itself is bit-identical; bonus cache-capacity changes need separate validation.

### M3. Refill or preserve staging-class experts after prefill

**Expected gain: roughly 1-4 ms/token over the first decode tokens, possibly more in a miss-heavy first layer; unmeasured.**

`tier.cpp:715-728` drops up to 288 largest-main-class experts while borrowing prompt staging, and `:859-869` returns those slots empty. The reference's `_end_large_forward`/`_refill`, `tier.py:1009-1074`, repopulates available RAM-resident candidates from warm order. NVMe refill is disabled by default because it previously held the first decode step for about 0.5 seconds. The engine refills only as decode admissions occur. This is a concrete reference difference, especially for the first K35 MoE layer, although the benefit is limited when that class has few RAM candidates. Preserve displaced hot records in RAM or refill the most useful ready rows after prompt completion, overlapping other independent work if available.

Risk: restoring 288 records synchronously can cost hundreds of milliseconds of first-token latency and recreate the same evictions. Measure payback over the request. Numerics: exact mode is bit-identical; quality bonus routing changes and must be evaluated separately.

### M4. Batch startup transfers and release adapter source data

**Expected startup gain: roughly 0.5-3 seconds; host memory reclaimed: at least about 1.07 GiB of raw adapter arrays, plus fused copies. Confidence: medium for memory, low for time.**

`st.cpp:86-93` calls synchronous upload for every loaded tensor; `rt.cpp:40-42` waits on every zero initialization, and `model.cpp:16-22` waits during every mHC conversion. Batch uploads in a loading arena, retain source mappings until the final copy event, and wait once per useful group. Do not defer errors until after the memory budget is computed.

The adapter contains 992 bf16 tensors totaling 547.89 MiB. `LoraSet::raw_` stores fp32 copies, at least 1,095.78 MiB, and keeps them after runtime matrices and merged kv_b are ready. It also creates fused qkv buffers and retains uploaded module objects. Release raw adapter arrays after construction if runtime scaling/reload does not require them. Sequentially touching the prompt ring's 4.5 GiB also adds first-prefill delay, `tier.cpp:701-713`; allocate/touch it in chunks before it is needed or in parallel with startup I/O, within host-memory limits.

Numerics: bit-identical. Risk: asynchronous source lifetime and losing an intended adapter-rescale capability.

### M5. Fix ownership before repeated model construction or checkpoint reuse

**Steady decode gain: zero; avoids increasing memory use and teardown stalls across requests/reloads.**

`Workspace` owns a raw `base` with no destructor in `rt.h`; `Model::picks_h` is a raw pinned allocation; checkpoint `conv`/`rec` vectors contain raw pinned allocations without owning cleanup. `ops.cpp:182-193` retains thread-local GEMM scratch and waits when it grows. Add explicit ownership and appropriate completion events. This is not the explanation for a single-run expert slowdown. `Tensor::release`, `rt.cpp:44-50`, drains the main queue for each owning tensor; batch safe teardown instead of repeated global waits where needed. Also wait for copy-queue users before freeing their source/destination storage.

Numerics: bit-identical. Risk: destructor order and in-flight DMA lifetime.

### M6. Larger RAM tiers and better cache policies have upside, but affect quality routing

**Potential gain: a few to tens of ms/token, especially exact mode; not a numerically neutral default change.**

Fourteen GiB holds only about 1,883 records across classes. More physical RAM cache can remove NVMe reads, but the machine is shared and its 64 GiB must also hold the prompt ring, adapter data, model mappings and user applications. Use measured available memory and a cap. Better admission, retained RAM shadows of VRAM victims, larger RAM, prompt-derived warm statistics and idle-time curation all change what receives the 0.1 bonus unless physical storage is separated from logical bonus residency.

For exact routing, cache policy is mathematically transparent, and these are valid performance experiments. For quality mode, list them with the numerical changes below until they pass a frozen-route comparison and the original parity workload. Prior reference curation results were modest and inconsistent, so do not budget a guaranteed win from replacing LRU alone.

## Quality-affecting or separately gated ideas

These must not be folded into a parity-preserving speed patch:

1. **No-sync resident-only routing, dropped picks, fewer experts or a larger bonus.** They can remove much of the 42-sync/storage cost, but change experts and weights. Reference `block_graphs.py:55-67` records severe quality failures for no-sync variants. Keep them separate.
2. **Different quantization, lower state/cache precision, merging LoRA into re-quantized EXL3, approximate activations, fewer Sinkhorn iterations, or omitted indexer/cache work.** Gains range from small to large but represent changed inference. The measured mHC/head phases put strict limits on what some shortcuts can save.
3. **Removing MLA's fp16 QK or normalized-P rounding, changing attention formulation, or folding the EXL3 codebook affine across half-rounding boundaries.** These are not guaranteed rounding noise. Preserve existing casts in the first optimized path.
4. **Cache-policy/capacity/readiness changes in bonus mode.** Extra RAM, altered victim selection, dropping write-backs and new prefetch rules can change selected experts despite exact byte arithmetic. Temporary physical staging can avoid this by leaving the logical bonus cache unchanged.
5. **MTP/speculative decoding.** Full target verification can preserve a sampling distribution mathematically, but this engine's Q>1 EXL3 path uses different accumulation, its exact router has different Q=1/Q>1 rounding in the reference, and `Model::block_forward` currently calls KDA with history=false. Rewind support alone is insufficient. Add target history, rejection/acceptance, cache rewinds, target sampling correction and LoRA-aware draft behavior before claiming equivalence. Existing MTP logs show variable acceptance and often no gain. Budget no speedup until verification costs and accepted tokens per pass are measured. A draft is not an automatic 2x improvement.

## Things not worth repeating

- AOT already gave no measurable decode gain, and the code is already built AOT. Its startup value is separate.
- The expert GEMV is already the reference kernel, including the power-of-two splits and grouped M>=2 behavior. A wholesale rewrite has no demonstrated advantage.
- Input-Hadamard removal barely changed mixed expert timing in the prior probe. It is not the missing 20 ms.
- Broad speculative RAM staging already lost time. With 12 predicted rows, only about 45% were used. Try demand-only/selective copies with whole-path timing if revisiting it.
- Low-priority GPU work was starved in the existing test. Queue priority does not manufacture overlap. Keep the external B70 queue as the sole card scheduler.
- Imported anonymous memory appeared VRAM-fast because it consumed VRAM. It did not demonstrate fast zero-copy host access.
- Do not start with a new queue, busy-poll loops, a global `q.wait()` removal, or broad runtime environment changes. Source-slot lifetime and copy completion are real dependencies. The `q.wait()` at `tier.cpp:603` is a RAM-exhaustion path for large pick sets, not an unconditional decode wait.

## Acceptance and next measurements

Keep README quality/parity numbers unchanged. Preserve the supplied before measurements and use the new implementation as the after measurement. Save per-step expert selections and weights when a rounding or cache change is proposed, plus KL/NLL/top-1 using the same forced references. Test both short quality with LoRA and long exact with LoRA; a faster short bonus run says little about the 17 ms exact-router path. For rearranged recurrence, include a long continuation and checkpoint/restore check. For memory optimizations, include maximum-context prefill followed by decode so a larger cache is not purchased with an OOM.

Next useful instrumentation is actual event timing for KDA/LoRA/MLA subkernels and copy API/event durations, plus PCIe/DRAM/translation counters during the existing full workload. The MoE diagnostic has already answered the host-gap question; another unchanged eight-expert baseline will not answer the memory-working-set question.

For graph/runtime implementation, consult the [SYCL graph extension specification](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_graph.asciidoc). It defines in-order graph dependencies and recording restrictions. Runtime copy-engine and adapter knobs vary by version; the [Intel runtime environment documentation](https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md) and [Unified Runtime Level Zero reference](https://oneapi-src.github.io/unified-runtime/core/LEVEL_ZERO.html) distinguish adapter behavior. Verify against the installed 2025.3 libraries and measure a specific hypothesis before changing them. No environment-variable speedup is assumed in the estimates above.
