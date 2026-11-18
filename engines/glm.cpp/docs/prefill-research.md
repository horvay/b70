# Prefill research: MLA/DSA, KDA and MoE on the B70

Web research, 2026-11-02. Nothing here was run on the card. Each idea is labelled:

- **EXACT**: the same arithmetic in the same order. Bit-identical output is possible.
- **MATH-EXACT**: the same function with a different rounding or summation order. Expect small logit drift. Near the
  cutoff, a top-k selection or an expert pick can flip. Parity-check it against the reference.
- **QUALITY**: changes what the model computes. Keep these out of a parity patch.

Model facts used below come from `forward-spec.md`. The MLA layers are NoPE (`qk_rope_head_dim` 0), so the MHA form is
qk = v = 256 and the absorbed (MQA) form is qk = v = 512, with K and V being the same latent row. The indexer selects
512 pools of 4 tokens, plus the tail tokens.

Hardware ceiling: Intel rates the B70 at 367 INT8 TOPS (32 Xe2 cores, 256 XMX engines, 2.8 GHz) with 608 GB/s of
memory bandwidth ([Intel spec](https://www.intel.com/content/www/us/en/products/sku/245797/intel-arc-pro-b70-graphics.html),
[datasheet](https://www.intel.com/content/dam/www/central-libraries/us/en/documents/2026-03/datasheet-b70-gpu.pdf)).
XMX runs fp16/bf16 at half the INT8 rate, which puts the fp16 peak near 180 TFLOPS. That figure is my derivation, not
an Intel number.

## Ranked ideas

| # | Idea | Expected payoff | Numerics |
|---|------|-----------------|----------|
| 1 | Sparse MLA prefill in MQA form on XMX: one query token's 64 heads form the M dimension; gathered latent tiles are shared by QK and PV | most of the ~7 s MLA time at 4K; the largest single win at 12.8K+ | MATH-EXACT |
| 2 | Dense MHA (head 256) flash attention for rows whose visible context is ≤ 2048 tokens, skipping indexer and top-k | removes all indexer/top-k work for short prompts and the first 2K rows of every fresh prompt | EXACT selection; MATH-EXACT attention (the reference already does this) |
| 3 | Indexer scores as an XMX GEMM (indexer heads on the DPAS M dimension), and threshold top-k that emits ascending indices without a sort | indexer plus top-k drop to a few ms per layer | scoring MATH-EXACT; top-k EXACT |
| 4 | Fused EXL3 dequant + DPAS MoE kernel for 16–256 rows per expert, count-sorted grouped launches | likely a large share of the ~6 s KDA+MoE time at 4K (about 114 rows per expert) | MATH-EXACT unless it reproduces the reference kernel |
| 5 | Chunked KDA (FLA `chunk_kda` port, WY/UT, 64-token chunks, 16-token sub-chunks) instead of the serial token loop | seconds at 4K–13K | the *reference itself* is chunked FLA, so this moves glm.cpp toward the reference |
| 6 | Expert IO: read every NVMe expert for long prompts without waiting for routing, prefetch layers ahead, DMA once into VRAM | hides most of the ~3 s IO; matters most at 12.8K | EXACT |
| 7 | Prefix and state reuse across chat turns: snapshot the KDA state at the end of each prompt, reuse the reply's KV | skips re-prefilling whole replies | EXACT |

## 1. DSA prefill: what the reference systems do

- **DeepSeek's reference code** (`DeepSeek-V3.2-Exp/inference/model.py`) runs prefill as dense MHA with the top-k
  turned into a −inf mask (`index_mask ... scatter_(-1, topk_indices, 0)`) and decode as absorbed MQA. The indexer
  applies a Hadamard rotation, quantizes q/k to FP8 and calls `fp8_index`, then `topk`.
  [repo](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp)
- **Production kernels use MQA form for sparse prefill.** FlashMLA's `flash_mla_sparse_fwd` takes
  `q[s_q,h_q,d_qk]`, `kv[s_kv,1,d_qk]`, `indices[s_q,1,topk]` and `d_v` = 512. V is the prefix of the gathered latent
  row. It reports up to 640 TFLOPS on H800 and 1350 TFLOPS on B200. [FlashMLA](https://github.com/deepseek-ai/FlashMLA).
  TensorRT-LLM also runs DSA in MQA mode for both phases "to maximize kernel efficiency".
  [TRT-LLM blog](https://nvidia.github.io/TensorRT-LLM/latest/blogs/tech_blog/blog15_Optimizing_DeepSeek_V32_on_NVIDIA_Blackwell_GPUs.html)
- **Tiling with no TMA gather.** The best template for Xe2 is FlashMLA's Ascend write-up
  ([deep dive](https://github.com/deepseek-ai/FlashMLA/blob/main/docs/20260930-ascend-prefill-deep-dive.md)). Ascend
  has no TMA gather either. In that kernel:
  - one core handles the 64 heads of a single query token (`MMA_M = 64`), and different query tokens are always
    processed separately;
  - the top-k list is consumed in blocks of `B_TOPK` = 64–128 gathered rows;
  - the PV matmul of block i−1 is deferred and overlapped with QK of block i;
  - a "skip-scale" online softmax only rescales O when the running max moves by more than a threshold (6);
  - with 64 heads the kernel is compute-bound, at 410 TFLOPS (95% of peak).

  On Xe2: 8 subgroups × DPAS M = 8 rows each = 64 heads. Gather a block of 32–64 latent rows (512 fp16 = 1 KB each)
  into SLM once and let all 8 subgroups use it for both QK and PV, since K = V here.
- **Short prompts go dense.** SGLang switches DSA prefill to plain MHA below 2048 tokens of KV
  (`SGLANG_DSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD`). Raising the threshold is allowed but flagged as an accuracy
  trade-off. [SGLang docs](https://docs.sglang.io/basic_usage/deepseek_v32.html). TRT-LLM skips MQA and top-k
  entirely when N ≤ 2048 (PR-9524). exllamav3, the reference this engine follows, does the same: `sparse = visible >
  index_topk`, and the dense path uses MHA form, "~2.8× fewer FLOPs per query-past pair"
  ([mla_attn.py](https://github.com/turboderp-org/exllamav3/blob/master/exllamav3/modules/mla_attn.py),
  [env_vars.md `EXL3_MLA_PREFILL`](https://github.com/turboderp-org/exllamav3/blob/master/doc/env_vars.md)).
- **Cost model for this model.**
  - MHA form costs 256 + 256 MACs per head-pair; MQA form costs 512 + 512. MHA also pays a K/V up-projection of
    about 33 MFLOP per token, but that is amortized over the whole chunk.
  - Dense MHA is cheaper than sparse MQA while a row's visible length is below about 2 × 2048 ≈ 4K tokens.
  - Estimated XMX work for sparse MQA: about 3.4 TFLOP per layer for 12.8K query rows, or roughly 38 TFLOP over the
    11 layers. At 30–50% of peak that is 0.4–0.7 s, against the current tens of seconds.
  - For a fresh 4K prompt (dense first half, sparse second half): about 0.7 TFLOP per layer, which is about 0.1 s in
    total.
- **Per-row regime (my suggestion).**
  - The reference decides dense vs sparse per 1024-token sub-chunk.
  - Deciding per row is EXACT for selection: a row with ≤ 512 complete pools selects everything.
  - It is MATH-EXACT for attention, because MHA and absorbed forms round differently (fp16 K/V vs fp16 QK and P).
    For bit-parity, keep the reference's per-sub-chunk decision.
- **Masked dense MHA above 2048** (DeepSeek's "masked MHA mode to simulate DSA" for short prefill) is MATH-EXACT:
  dense MHA with the top-k mask, skipping 64-key tiles that no row in the query tile selected. It is only worth it up
  to about 4K visible tokens, and it rounds differently from the reference's sparse path.

## 2. Flash attention and SDPA on Xe2

- **oneDNN Graph SDPA**
  ([dev guide](https://uxlfoundation.github.io/oneDNN/dev_guide_graph_sdpa.html)):
  - The optimized GPU path needs f16/bf16 with f32 intermediates, D ≤ 512, D_qk == D_v, and XMX.
  - Both forms here qualify: MHA has D = 256; MQA has D = 512 with one KV head.
  - It supports an implicit causal mask (GenIndex/GreaterEqual/Select, top-left or bottom-right) and explicit
    Add/Select masks.
  - Anything off-pattern falls back to a reference kernel that is O(S²) in memory.
  - Older docs said the implicit causal mask was CPU-only. Check against the installed oneDNN.
  - It is a quick way to get the dense ≤ 2048 path or a numerics oracle. It cannot do gathered sparse attention.
- **vllm-xpu-kernels Xe2 chunk prefill** (sycl-tla based) has head 256 and 512 policies with paged, causal and LSE
  variants. Its config lists GLM-4.7-Flash at 256.
  - The tile is `ShapeQK = <256 q, 32 kv, 32 d>` with 32 subgroups (8 q rows each).
  - Head 512 reuses a 256-wide output tile "due to register pressure".
  - Sources: [chunk_prefill_default.conf](https://github.com/vllm-project/vllm-xpu-kernels/blob/main/csrc/xpu/attn/kernel_configs/chunk_prefill_default.conf),
    [fmha_utils.hpp](https://github.com/vllm-project/vllm-xpu-kernels/blob/main/csrc/xpu/attn/fmha_utils.hpp).
  - The repo is a torch extension. Lift the SYCL kernel headers, not the bindings.
- **sycl-tla (CUTLASS for Intel)** targets `intel_gpu_bmg_g31` directly and needs DPC++ 2025.1+.
  [repo](https://github.com/intel/sycl-tla)
  - Example `06_bmg_flash_attention` covers prefill, decode, cached KV and LSE, but only for heads 64/96/128/192.
  - Its prefill tiles: head 128 uses `<256,32,32>` with 16 subgroups; head 192 uses `<256,64,32>`.
  - `12_xe20_moe_gemm_cute_interface` and `04_bmg_grouped_gemm` are grouped/MoE GEMM references.
- **joint_matrix on BMG/PVC**
  ([extension spec](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_matrix/sycl_ext_oneapi_matrix.asciidoc)):
  - shapes: fp16/bf16 M ≤ 8, N = 16, K = 16; int8 M ≤ 8, N = 16, K = 32;
  - the load stride in bytes must be a multiple of 8, and the base pointer 4-byte aligned.
- **Intel's tuning guide**
  ([joint_matrix guide](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/current/programming-intel-xmx-using-sycl-joint-matrix.html)):
  - make each subgroup compute about 32×64×16 out of several DPAS calls that reuse A, B and C registers;
  - use the large GRF mode;
  - `joint_matrix_prefetch` gave about 30% on PVC.
- **Gather trick (my analysis, not from a source).** In QKᵀ, the B operand is Kᵀ. Its VNNI layout puts consecutive
  k-dimension values of one key in one lane. So each lane can load 32 bytes from its own gathered key row and get a
  DPAS-ready B operand, with no SLM and no 2D block load. PV needs keys packed in pairs along K, so stage the gathered
  V/latent tile through SLM. Here it is the same tile as K.

## 3. Top-k selection

- **Radix select** is the TRT-LLM / vLLM `top_k_per_row` design. It already has a SYCL port in
  [vllm-xpu-kernels `topk_per_row.cpp`](https://github.com/vllm-project/vllm-xpu-kernels/blob/main/csrc/topk_per_row.cpp):
  - the first pass is an 11-bit histogram (2048 bins) over the fp16 form;
  - fp32 passes of 11/11/10 bits follow only if needed;
  - it stops as soon as the threshold bin fits a 2048-slot final buffer.

  TRT-LLM measured 7.4× faster than `torch.topk` on [64, 9295] rows.
- **Guess-verify-refine**
  ([arXiv 2604.22312](https://arxiv.org/html/2604.22312v1)):
  - guess a threshold from correlated data (previous step there; the previous query row in prefill);
  - count once, collect the candidates in [K, 3K] into SLM, then resolve exactly with a histogram;
  - 1.88× faster than radix, bit-exact sets, with a radix fallback.

  Also see [AIR top-k (SC23)](https://sc23.supercomputing.org/proceedings/tech_paper/tech_paper_pages/pap294.html)
  and [DeepSelect](https://github.com/deepseek-ai/DeepSelect).
- **EXACT tip for this model.**
  - Rows are short: at most about 4.9K pools at 19.5K tokens, and the scores are fp16. Each row fits in one
    workgroup's SLM.
  - The reference does a stable descending sort, takes 512, then re-sorts ascending.
  - Find the 512th-largest fp16 value T with a 16-bit radix/histogram (two 8-bit passes, or 11+5).
  - Then, in index order, emit every score > T, and as many == T as are still needed, lowest index first.
  - That reproduces the stable tie-break and the ascending output order with no sort at all.
  - Ties are common with fp16 scores, so test them explicitly.

## 4. KDA chunked prefill

- **Algorithm** ([Kimi Linear, arXiv 2510.26692](https://arxiv.org/html/2510.26692v2), building on
  [Gated DeltaNet](https://arxiv.org/abs/2412.06464)):
  - WY representation with a UT transform, `M = (I + StrictTril(diag(β)(Γ⊙K)(K/Γ)ᵀ))⁻¹ diag(β)`, then
    `W = M(Γ⊙K)` and `U = MV`;
  - per chunk: `S ← diag(γ)S + (Γ⊙K)ᵀ(U − WS)` and `O = (Γ⊙Q)S + Tril((Γ⊙Q)(K/Γ)ᵀ)(U − WS)`;
  - everything except the triangular inverse is a matmul.
- **Stability.** Dividing by Γ overflows over long spans.
  - FLA uses 64-token chunks and 16-token sub-chunks, works in the log2 domain, subtracts a midpoint before
    exponentiating, and computes the diagonal 16×16 blocks token-parallel in fp32.
  - It needs the gate lower bound (−5, as in this model).
  - FlashKDA uses 16-token chunks to stay inside bf16 range.
  - Sources: [FlashKDA](https://github.com/MoonshotAI/FlashKDA),
    [FLA `fla/ops/kda`](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/kda).
  - Use bf16 for off-diagonal DPAS (bf16 has the exponent range); fp16 would overflow.
- **A SYCL template already exists.**
  [vllm-xpu-kernels `chunk_gated_delta_rule_kernels_xe2.hpp`](https://github.com/vllm-project/vllm-xpu-kernels/blob/main/csrc/xpu/gdn_attn/xe_2/chunk_gated_delta_rule_kernels_xe2.hpp)
  implements chunked gated delta rule on Xe2:
  - 64-token chunks;
  - 64×64×32 GEMM tiles for A, WU and O;
  - the inverse as 16×16 diagonal blocks solved in registers, then merged with 16×16×16 DPAS.

  It is GDN (scalar gate per head). KDA needs the per-channel Γ scaling of q and k from the step above.
- **Numerics.** The reference's prefill is FLA's chunked algorithm (bf16 q/k after l2norm, bf16 Aqk/Akk, fp32 state;
  `forward-spec.md` §4.6). A faithful port of FLA's rounding points is closer to the reference than the current fp32
  serial loop. exllamav3 also notes that splitting the scan into 2048-token sub-ranges with the fp32 state carried
  across is bit-identical, and 2048 measured fastest (`EXL3_GDN_SUB_CHUNK`).

## 5. MoE prefill with offloaded experts

- **What exllamav3 does** (the closest system: EXL3 experts, streaming, GLM-5.3 support). Source:
  [env_vars.md](https://github.com/turboderp-org/exllamav3/blob/master/doc/env_vars.md).
  - **Three tiers by rows per expert.**
    - The fused dequant + MMA MoE kernel handles up to 256 rows.
    - Separate 32- and 64-row tile instances (`EXL3_MOE_MTILE`) cut fused time by 30–47% at 24–128 rows. The 16-row
      tile re-dequantizes the weights once per 16 rows.
    - Above 256 rows, a batched reconstruct-to-fp16 path is followed by a strided-batched GEMM over count-sorted
      expert groups, which cut launches and gave +16% at 8K tokens with 512 experts.
    - Dense linears switch to reconstruct + GEMM above 144 rows.
  - **CPU vs GPU for streamed experts.**
    - An expert is DMA'd to the GPU only if it has at least `EXL3_MOE_STREAM_T` rows. That defaults to 16 and
      scales inversely with the measured pinned→device bandwidth.
    - Experts with fewer rows run on the CPU.
    - Streaming never engages below 32 prompt rows.
    - Experts are staged in batches of 24 through a 2-slot pinned ring.
- **Layer-wise streaming** (OSDI '26, [arXiv 2606.10493](https://arxiv.org/html/2606.10493)):
  - loader, compute and unloader threads with event pairs, and a ring buffer of one layer of experts;
  - it shrinks to ping-pong above about 50K tokens;
  - CPU MoE saturates around 512 tokens, and the authors switch to GPU streaming above about 4K;
  - result: 32K-token prompts in under 30 s.

  ktransformers exposes the same switch as `--kt-gpu-prefill-token-threshold`
  ([issue](https://github.com/kvcache-ai/ktransformers/issues/1608)). llama.cpp copies CPU-resident experts to the
  GPU when a batch has ≥ 32 tokens (`GGML_OP_OFFLOAD_MIN_BATCH`), so a larger `-ub` amortizes the copy
  ([guide](https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide)).
- **What follows for glm.cpp** (my inferences):
  - **Kernel gap.** At 4K tokens the average expert gets about 114 rows, under the current M ≥ 256 DPAS threshold.
    So most experts take the grouped GEMV path, which serializes 16-row groups. A 32/64-row-tile fused EXL3 + DPAS
    kernel is the biggest MoE kernel gap. It is MATH-EXACT unless it matches the reference's own kernel and threshold
    choice.
  - **Launch count.** One launch per expert per projection is about 36K launches per prefill. Batch count-sorted
    expert groups per launch. EXACT if the per-expert arithmetic is unchanged.
  - **Read everything early.** For prompts over a few thousand tokens nearly every expert is hit, so read all
    NVMe-tier experts of upcoming layers without waiting for routing, as far ahead as the staging RAM allows. Start
    at request arrival. EXACT.
  - **The NVMe floor is unavoidable.** At 6–7 GB/s, every GB on NVMe costs about 0.15 s per prefill, so place the
    NVMe tier by prefill hit frequency.
  - **DMA once, not zero-copy.** For prefill, copy each expert into VRAM with a copy engine (a separate Level Zero
    copy-queue ordinal) rather than letting a row-tiled kernel read host memory in place. Zero-copy re-reads the
    weights over PCIe once per row tile.
  - **Fixed combine order.** Keep the expert-combine summation order fixed (a separate combine pass) so out-of-order
    expert completion stays EXACT.
  - **CPU compute buys little here.** It only saves PCIe traffic, not NVMe reads, and the GPU is far faster at these
    row counts. Low priority.

## 6. Serving-level tricks

- **Reuse prefixes across turns (EXACT).**
  - vLLM can only resume hybrid linear-attention models at checkpointed state boundaries (`--mamba-cache-mode
    align`). [vLLM APC](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html)
  - For one chat, snapshot the KDA recurrent and conv state, plus the MLA latent/indexer-plane lengths, at the end of
    each prompt and at the end of each reply.
  - Check whether the GLM chat template drops earlier turns' reasoning. If it does, the cached prefix diverges at the
    previous reply, and the snapshot at the end of the previous prompt lets you prefill only the stripped reply plus
    the new message.
- **Chunk size.** With streamed experts, bigger chunks amortize weight IO. Chunk only to bound activation and score
  scratch. The exllamav3 indexer slabs 256 rows × 32K-key tiles with an exact running top-k merge. In the MLA layers,
  sub-chunking changes nothing numerically as long as the dense/sparse decision rule is kept.
- **Not exact, kept out of the plan (QUALITY):**
  - FP8/INT8 indexer (DeepSeek's FP8 + Hadamard). INT8 DPAS on Xe2 is 2× fp16, but it changes scores and selections.
  - FP8 sparse attention (TRT-LLM saw GPQA move by about 1–2 points).
  - Raising the dense threshold above top-k without the mask.
  - KTransformers expert deferral (under 0.5% accuracy change by its authors' measure).
  - Speculative prefill and prompt compression.

## Suggested order

1. Sparse MQA prefill kernel (idea 1), together with the indexer GEMM and the sort-free top-k (idea 3). Validate
   selections bit-exactly against the reference top-k before timing.
2. Dense MHA head-256 path for ≤ 2048 visible tokens (idea 2), using oneDNN SDPA first as an oracle and a quick win.
3. Profile KDA vs MoE inside the 6 s. Then do the 32/64-row EXL3 DPAS MoE tiles and grouped launches (idea 4), and
   the chunked KDA port (idea 5).
4. IO pipeline (idea 6), then turn snapshots (idea 7).
