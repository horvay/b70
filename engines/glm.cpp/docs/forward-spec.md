# GLM-5.3-Flash forward pass: what the PyTorch-XPU stack computes today

This is the reference the C++/SYCL engine (glm.cpp) is checked against. It describes the forward pass of
GLM-5.3-Flash (exllamav3 arch `glm5_next`, `Glm5NextForConditionalGeneration`, text component only) as served on the
Intel Arc Pro B70 by `llm/glm/xpu/serve.py`. It is written from the code as of 2026-10-31; nothing was run.

## 0. Conventions and scope

### 0.1 Path prefixes

| Prefix | Path |
|---|---|
| `EX/` | `~/b70/llm/glm/exllamav3/exllamav3/` (fork, commit `f994e7a` plus uncommitted edits in `modules/mla_attn.py` and `modules/attention_fn/mla_triton.py`; the working tree is what runs) |
| `GX/` | `~/b70/llm/glm/xpu/glm_xpu/` |
| `XP/` | `~/b70/llm/glm/xpu/` (`serve.py`, `pack_store.py`, `csrc/`, `exl3xpu/`, `exllamav3_ext/`) |
| `CS/` | `~/b70/llm/glm/xpu/csrc/` |
| `M/` | `~/models/llm/GLM-5.3-Flash-GM-LoRA-exl3/` |

`llm/glm/patches/exllamav3.patch` is the same change set as commit `f994e7a` (already applied). It covers KDA prefill
memory, MLA XPU prefill, MLP row slices, the in-pass recurrent checkpoint and the fla cumsum.

### 0.2 What "runs today" means

The served process is `XP/serve.py` → `GX/runner.py:load` → `block_graphs.install` (`XP/serve.py:288-321`):

- `GX/runner.py:8` sets `EXL3_XPU=1`, `EXL3_BC_GDN=0`, `EXL3_BC_ATTN=0` and `TRITON_F32_DEFAULT=ieee`.
- `GX/compat.py:17-39` aliases `torch.cuda.*` to `torch.xpu.*`.
- `EX/ext.py:29-31` imports the Python shim `XP/exllamav3_ext/__init__.py`. That shim re-exports `GX/ops.py`; any op it
  doesn't implement raises when called.
- `GX/runner.py:35` calls `moe_patch.install`, which replaces `BlockSparseMLP.load` and `BlockSparseMLP.forward`.
- `GX/runner.py:39` sets `no_reconstruct = True`, so EXL3 linears always run the kernel and never dequantize plus GEMM.
- `GX/runner.py:44-45`: the cache is `Cache(..., layer_type=CacheLayer_fp16, max_history=mtp)`.
- `GX/runner.py:48-52` loads the optional MTP draft (`GLM_XPU_MTP`).
- `GX/runner.py:53-58`: `lora.apply`.
- `GX/runner.py:69-71` builds the expert tier.
- `GX/block_graphs.py:689-695` replaces `TransformerBlock.forward` with the decode graph path.
- `GX/graphs.py`, the older per-layer KDA graph, is **not** installed in serving. Only `XP/snip/graph_test.py` uses it.

Launcher defaults: `~/b70/llm/b70:1309-1335`. The default `GLM_MODE=quality` sets `GLM_XPU_RES_BONUS=0.1`,
`GLM_XPU_RAM_MARGIN=0.2` and `GLM_XPU_MISS_PF=0`. The LoRA folder `M/lora` is passed with `--lora`, and
`GLM_XPU_LORA_SCALE` defaults to 1.

**The production decode router is residency-biased (§6.4b). A parity reference must run with `GLM_XPU_RES_BONUS=0`.**

### 0.3 Dtype names

- fp16 = IEEE binary16.
- bf16 = bfloat16.
- fp32 = binary32.
- "fp16 GEMM" means `torch.matmul` with fp16 inputs and fp16 output. oneDNN accumulates in fp32 and rounds once to fp16
  (comment at `GX/ops.py:108`).

---

## 1. Hyperparameters

Sources: `M/config.json` (`text_config`), `M/quantization_config.json`, and the parser at
`EX/architecture/glm5_next.py:29-135`.

| Item | Value | Notes |
|---|---|---|
| hidden_size | 4096 | |
| num_hidden_layers | 45 | layers 0..44; MTP head stored as layer 45 |
| vocab_size | 154880 | multiple of 128, so lm_head needs no padding |
| tie_word_embeddings | false | `lm_head` is its own EXL3 tensor |
| rms_norm_eps | 1e-5 | every RMSNorm, the mHC RMS and the KDA o_norm |
| hidden_act | silu, with `swiglu_limit` 10.0 | clamped SwiGLU, §6.2 |
| Layer schedule `layer_types` | `linear_attention` (KDA) on 0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30,32,33,34,36,37,38,40,41,42,44 (34 layers); `deepseek_sparse_attention` (MLA + DSA) on 3,7,11,15,19,23,27,31,35,39,43 (11 layers) | `glm5_next.py:168-224` |
| `mlp_layer_types` | dense on 0,1,2; sparse (MoE) on 3..44 | `first_k_dense_replace` 3; `glm5_next.py:97-101,254-307` |
| `indexer_types` | "full" on all 45 layers (only MLA layers use it) | `glm5_next.py:218` |
| **MLA** | 64 q heads; q_lora_rank 1536; kv_lora_rank 512; qk_nope_head_dim 256; **qk_rope_head_dim 0 (NoPE, asserted, `glm5_next.py:51-52`)**; v_head_dim 256; num_key_value_heads 64 (but the latent cache is 1-head MQA) | `glm5_next.py:44-57` |
| MLA softmax scale | `qk_head_dim**-0.5` = 256^-0.5 = 0.0625 | `glm5_next.py:56`. No mscale / YaRN; `rope_settings=None` (`:57,211`) |
| RoPE | **none anywhere** (main attention or indexer) | `max_position_embeddings` 1048576 unused |
| **DSA indexer** | index_n_heads 32; index_head_dim 128; index_topk 2048; index_kpool 4 (k-pool compression); always_select_tail true; kpool_compress true | `glm5_next.py:72-80`. `indexer_rope_interleave: true` and `index_share_for_mtp_iteration: true` are **never read** |
| **mHC** | hc_mult H=4 streams; hc_sinkhorn_iters 20; hc_eps 1e-6; `mhc` true | `glm5_next.py:82-86` |
| **KDA** | num_heads 64 (k = v); head_dim 128 (Dk = Dv); short_conv_kernel_size 4; gate_lower_bound -5.0 | `glm5_next.py:65-70,169-199` |
| **Dense MLP** | intermediate 12288 | layers 0-2 |
| **MoE** | n_routed_experts 288; num_experts_per_tok 8; n_group 1 / topk_group 1 (ungrouped, asserted); scoring sigmoid; topk_method noaux_tc with `e_score_correction_bias`; norm_topk_prob true; routed_scaling_factor 2.5; n_shared_experts 1; moe_intermediate_size 2048 (shared expert 2048 too) | `glm5_next.py:88-109`. **`moe_router_dtype: "float32"` is never read** (router logits are fp16, §6.4) |
| MTP | num_nextn_predict_layers 1 (layer 45): enorm, hnorm, eh_proj 8192→4096, one MLA + MoE block **without mHC**, shared_head.norm; the trunk lm_head is shared | `glm5_next_mtp.py` |
| Quantization | EXL3 v1.6.1, mul1 codebook, `bits` 2.6 avg, `head_bits` 6, `mtp_bits` 4, `out_scales` auto | §7 lists the per-tensor K |
| eos_token_id | 154820, 154827, 154829; pad 154820 | generation_config: temperature 1.0, top_p 0.95 |

The vision tower (`vision_config`, 1007 `model.visual.*` tensors) is out of scope.

---

## 2. Whole-model forward

### 2.1 Module list (`EX/architecture/glm5_next.py:149-344`)

```
Embedding(model.language_model.embed_tokens)       # CPU-resident bf16 table
ExpandStreams(H=4)                                 # x -> 4 identical fp32 streams
45 x TransformerBlock(layer L):
    attn_hc  = HyperConnection(layers.L.hc_attn_*)
    attn_norm= RMSNorm(layers.L.input_layernorm)
    attn     = GatedDeltaNet (KDA)  |  MLAttention (MLA+DSA)
    mlp_hc   = HyperConnection(layers.L.hc_ffn_*)
    mlp_norm = RMSNorm(layers.L.post_attention_layernorm)
    mlp      = GatedMLP (L<3)  |  BlockSparseMLP + shared GatedMLP (L>=3)
HyperHead(mean=True)                               # unweighted mean over the 4 streams
RMSNorm(model.language_model.norm, out fp16)
Linear(lm_head, EXL3 K=6)                          # fp16 logits
```

### 2.2 One decode step (bsz 1, one token at position p, L = p tokens already cached)

| # | Op | Shape and dtype out | Implementation |
|---|---|---|---|
| 1 | `embed_tokens[id]` | (1,1,4096) fp32 (bf16 values, exact) | `EX/modules/embedding.py:46-55,145-151`. Table loaded bf16 (`allow_bf16`), on the CPU (`caps prefer_cpu`, `:37-39`), uploaded |
| 2 | ExpandStreams: `x.float().unsqueeze(2).expand(...,4,...)` | R = (1,1,4,4096) fp32, 4 equal copies | `EX/modules/hyperconnections.py:37-38` |
| 3 | 45 blocks, §2.3 | R updated | `GX/block_graphs.py:504-686` |
| 4 | HyperHead: `R.mean(dim=2)` | (1,1,4096) fp32 | `hyperconnections.py:639-641` (torch mean = sum/4). No `hc_head` tensors exist |
| 5 | RMSNorm `model.language_model.norm` (weight bf16 [4096]), eps 1e-5 | (1,1,4096) **fp16** | `glm5_next.py:328-333` → `GX/ops.py:253-258` → Triton `GX/k_hc.py:184-211` |
| 6 | lm_head (EXL3 K=6, 4096→154880, no out_dtype) | (1,1,154880) **fp16** logits | `glm5_next.py:334-343` → `GX/ops.py:60-80` → `exl3_gemm_small` |
| 7 | sampler | token id | `GX/ops.py:380-415` (`fused_sampler`: greedy argmax, or Gumbel with top-k/top-p/min-p filters) |

### 2.3 One trunk block on the decode graph path

`GX/block_graphs.py:_forward` applies when `bsz == 1`, `1 ≤ Q ≤ 8`, there is a recurrent slot, the step is not prefill,
and it is not (history and Q == 1) (`:511-520`). R is a static (1,Q,4,4096) fp32 buffer updated in place.

The step is split into XPU-graph phases. The order below is the exact execution order.

| Phase | Op | Out | File:line |
|---|---|---|---|
| A / Am | `attn_hc` mix: post_a (1,Q,4) fp32, comb_a (1,Q,4,4) fp32, collapsed (Q,4096) **fp16** | §3 | `block_graphs.py:161-189,192-193` |
| A / Am | `input_layernorm`, eps 1e-5 | y_attn (1,Q,4096) fp16 | `block_graphs.py:188-189` → `k_hc._rms_norm_kernel` |
| A | KDA attention (in the graph, keyed per recurrent slot) | attn_out (1,Q,4096) **fp32** | §4; `block_graphs.py:564-566,195` |
| Am | MLA attention (in the graph, `mla_graph.decode`) | attn_out fp32 | §5; `block_graphs.py:559-562` |
| B (B1) | `attn_hc` apply: `R[h] = post_a[h]·attn_out + Σ_g comb_a[g,h]·R[g]` | R in place | `block_graphs.py:231-233` → `k_hc._hc_apply_kernel` |
| B (B1) | `mlp_hc` mix, then `post_attention_layernorm` | post_f, comb_f, y_mlp (1,Q,4096) fp16 | `block_graphs.py:234` |
| B (B2) | MoE only: router (and the next layer's prefetch prediction) | sel (Q,8) int32, w (Q,8) fp32 | §6.4; `block_graphs.py:237-285` |
| host | MoE only: copy the picks to the host, sync on an event, `TIER.ensure` makes the experts resident | | `block_graphs.py:591-678` |
| S | MoE only: shared expert `GatedMLP(y_mlp)` | sh (1,Q,4096) fp32 | `block_graphs.py:436-437,602` |
| C | dense MLP `m.forward(y_mlp)`, **or** routed experts `moe_forward(...)` (fp16) `.float()` `+= sh` | y (1,Q,4096) fp32 | `block_graphs.py:440-451` |
| C | `mlp_hc` apply: `R[h] = post_f[h]·y + Σ_g comb_f[g,h]·R[g]` | R in place | `block_graphs.py:451` |

The eager `TransformerBlock.forward` (`EX/modules/transformer.py:137-214`) does the same math: mix → `.half()` → norm
(fp16) → sublayer → apply. It is used for prefill, for the MTP block and for any shape the graph path declines.

### 2.4 Prefill (many tokens): what differs

- **Chunks.** The generator runs the prompt in chunks of `max_chunk_size` = `GLM_XPU_CHUNK` (default 8192,
  `XP/serve.py:319`). The **last prompt token is not prefilled**: it is the first decode input.
- **Prefill-only forward** (`EX/model/model_ls.py:297-313`) stops at the last cache-writing module, block 44. Block 44
  runs only its attention, to advance the KDA state, and returns (`transformer.py:160-161`). There is no final norm or
  lm_head.
- **With MTP on**, the prompt instead runs a full `model.forward` with `last_tokens_only=1`
  (`EX/generator/job.py:1405-1409`). The final norm then exports fp16 states for every prompt token. lm_head runs on the
  last token only (`model_ls.py:326-327`).
- **Every block is eager** (`transformer.py:137-214`):
  - mHC: Triton when R ≤ 64 rows (fp16 `fn` copy when R ≤ 32, fp32 `fn` otherwise), torch for R > 64 (fp32 `fn`)
    (`GX/ops.py:208-234`, `hyperconnections.py:144-151`).
  - KDA: 2048-token sub-chunks, plus a split at the in-pass checkpoint; chunked fla algorithm when a piece has ≥ 64
    tokens (§4.6).
  - MLA: 1024-token sub-chunks. Dense MHA SDPA at ≤ 2048 total context, sparse indexer above (§5.5).
  - MLP: 2048-row slices (`EX/modules/mlp.py:734-742`).
  - MoE: DPAS prefill kernel when M ≥ 256 (§6.5).
- **EXL3 linears with M > 128** use the reconstruct + fp16 GEMM path, and their output is fp16 even when fp32 was asked
  for (§7.4).

### 2.5 MTP speculative decoding (`GLM_XPU_MTP=n`; off unless set)

**Draft model** (`EX/architecture/glm5_next_mtp.py`):

1. Input layer (`EX/modules/arch_specific/qwen3_5_mtp.py:88-114`, `constant_bias = 0.0`):
   - `e = enorm(embed(tok).half())`: RMSNorm, eps 1e-5, fp16. **The embedding is cast bf16 → fp16 here.**
   - `h = hnorm(target_hidden)`: fp16.
   - `x = eh_proj(cat(e, h))`. **The order is [embedding ‖ hidden]** (`:112`). EXL3 K=4, 8192→4096, out fp32.
2. One plain-residual block (no mHC; `glm5_next_mtp.py:60-142`):
   - `y = attn(input_layernorm(x))`. MLA + DSA with its own indexer, its own cache and K=4 weights.
   - `r = x + y`, fused into the next norm through `rms_norm_res_in` (`transformer.py:168-169,181-182`,
     `GX/ops.py:261-264`).
   - `x = r + moe(post_attention_layernorm(r))`. Exact router, MoE store layer li 42.
3. `shared_head.norm` (RMSNorm, fp16) → the trunk lm_head → argmax over `[:vocab_size]`
   (`glm5_next_mtp.py:147-152,222-239`).

**Inputs.** `target_hidden` is the trunk's final-norm output (`model.language_model.norm`), exported as fp16
(`EX/modules/rmsnorm.py:169-173`, `glm5_next_mtp.py:206-211`).

- Pairing: draft position p takes token p with target hidden p−1. During prompt prefill the hidden states are shifted
  by one, with zeros for the first position (`job.py:1423-1443`).
- Draft iteration i > 0 feeds the draft's own `shared_head.norm` output back in as `target_hidden`
  (`EX/generator/generator.py:755-786`).
- Draft length is dynamic, from confidence calibration (`generator.py:174,760-786`).

**Verify.** The trunk runs Q = 1 + n_draft ≤ 8 tokens in one forward with `recurrent_history=True`
(`generator.py:1036`). This takes the graph path with Q > 1 (`block_graphs.py:528-534`).

- The router for Q > 1 is always the Triton `k_router` (§6.4c).
- After acceptance the KDA conv/recurrent state is rewound (§8.3).
- The draft cache is back-filled with the accepted tokens' target hidden states (`generator.py:1301-1333`).

---

## 3. mHC hyper-connections (every trunk layer, two sites per layer)

### 3.1 Tensors per site

`model.language_model.layers.L.hc_attn_*` and `hc_ffn_*`, loaded `.float()` (`hyperconnections.py:88-93`):

| Tensor | Dtype / shape | Meaning |
|---|---|---|
| `hc_{attn,ffn}_fn` | F32 [24, 16384] | Rows m: 0-3 → pre, 4-7 → post, 8-23 → comb (row-major 4×4: m = 8 + 4i + j). Columns: the flattened stream stack, col = h·4096 + d |
| `hc_{attn,ffn}_base` | F32 [24] | Same row order |
| `hc_{attn,ffn}_scale` | F32 [3] | scale[0] pre, [1] post, [2] comb |

There is no weight on the internal RMS (`unweighted`, `hyperconnections.py:79-80`).

### 3.2 mix (per row r; s = R[r] flattened to 16384 fp32)

Reference: the docstring at `GX/k_hc.py:1-9`. Implementation: `k_hc._hc_mix_partial_kernel` + `_hc_mix_final_kernel`
(`GX/k_hc.py:66-129`, launched at `:139-155`). For R > 64 the torch twin `GX/ops.py:219-234` runs instead.

```
mix[m] = Σ_c fn[m, c] · s[c]                    # fp32 accumulate
rmr    = 1 / sqrt( Σ_c s[c]² / 16384 + 1e-5 )   # rms_norm_eps, NOT hc_eps; unweighted RMS over all 4 streams
z[m]   = mix[m] · rmr · scale[sidx(m)] + base[m]  # sidx = 0 (m<4), 1 (m<8), 2 (else)
sig    = 1 / (1 + exp(-z))
pre[h]  = sig[h] + 1e-6                         # hc_eps
post[h] = 2 · sig[4 + h]
C[i][j] = z[8 + 4i + j]
C = softmax over j (row-wise) + 1e-6
C = C / (column sums + 1e-6)                    # first column pass
repeat 19 times (sinkhorn_iters - 1):
    C = C / (row sums + 1e-6)
    C = C / (column sums + 1e-6)
comb = C                                        # (4,4) fp32, stored row-major [i][j]
collapsed[d] = Σ_h pre[h] · s[h·4096 + d]       # fp32 sum, stored FP16
```

**fn precision.**
- Decode, and any call with R ≤ 32 rows, reads an **fp16 copy** of `fn` (`fn_h = fn.half()`, `hyperconnections.py:146-149`,
  `block_graphs.py:184-185`). The products are fp16 × fp32, accumulated in fp32.
- R > 32 reads fp32 `fn`.
- So **decode rounds the 24 × 16384 mixing weights to fp16; prefill does not.**

**Reduction order** (decode R = 1):
- `hc_mix_num_chunks` gives 64 chunks of 256 columns (`GX/ops.py:202-205`).
- Each chunk is reduced in 256-wide blocks. The 64 chunk partials and the Σs² sum are added up by `tl.sum` in the final
  kernel.

**Collapsed output.** It is always fp16: the decode static workspace is fp16 (`hyperconnections.py:143`,
`block_graphs.py:183`), and the prefill buffer is also allocated fp16. The block casts it with `.half()` before the norm
regardless (`transformer.py:152,176`).

### 3.3 Norm after the mix

`input_layernorm` or `post_attention_layernorm`. The weight is bf16 [4096], kept bf16 (`rmsnorm.py:54`,
`allow_bf16=True`). Triton `_rms_norm_kernel` (`GX/k_hc.py:184-211`):

```
x = fp32(collapsed fp16) clamped to ±65504          # CLAMP because input is fp16
rmf = 1/sqrt(Σx²/4096 + 1e-5)
y = fp16( (x · (fp32(w) + 0.0)) · rmf )             # NOTE: x·w first, then ·rmf
```

The torch fallback (`GX/ops.py:267-282`) uses the same order. The exllamav3 `forward_torch`
(`EX/modules/rmsnorm.py:71-86`) differs: it normalizes, casts, then weights. That function is not used.

### 3.4 apply (residual update)

Triton `_hc_apply_kernel` (`GX/k_hc.py:159-180`). It is used when D % 512 == 0, which is always the case here
(`GX/ops.py:237-242`). In place, fp32:

```
for h in 0..3:  R'[h, d] = post[h] · fp32(y[d]) + Σ_g comb[g][h] · R[g, d]
```

i.e. `R' = post ⊗ y + combᵀ R` (`hyperconnections.py:180-192`). Output stream h uses **column** h of C. The column
normalization ran last, so each column sums to ≈1.

`y` is the sublayer output: attention fp32, MLP fp32.

### 3.5 Final collapse

`HyperHead(mean=True)`: `R.mean(dim=2)`, a plain fp32 mean, then `model.norm`. There are no hc_head parameters
(`hyperconnections.py:544-546,639-641`; `glm5_next.py:320-327`).

The MTP export uses the post-norm state, not the mean (§2.5).

---

## 4. KDA linear attention (GatedDeltaNet), 34 layers

Prefixes: `GDN` = `EX/modules/gated_delta_net.py`, `FN/` = `EX/modules/gated_delta_net_fn/`, `FLA/` =
`EX/vendor/fla/`, `KKDA` = `GX/k_kda.py`, `OPS` = `GX/ops.py`.

### 4.1 Parameters

| | Value | Source |
|---|---|---|
| Heads (k = v) | 64 | `glm5_next.py:173-176` |
| Dk = Dv | 128; k_dim = v_dim = 8192 | `GDN:385-386` |
| Conv channels | 24576 = [q 8192 \| k 8192 \| v 8192], head-major, 128 contiguous per head | `GDN:393` |
| Short conv | kernel 4, causal, depthwise, **no bias**, SiLU | one fused `conv1d.weight` [24576,1,4] bf16 |
| gate_lower_bound | −5.0, which selects the "safe gate" form | `glm5_next.py:70,191` |
| beta_scale | 1.0 | `GDN:333` |
| o_norm eps | 1e-5 | `GDN:545-547` |
| q/k L2 eps | 1e-6 | `KKDA:28-29`, `OPS:332-333`, `FLA/l2norm.py:84` |
| Readout scale | Dk^-0.5 = 128^-0.5 | `KKDA:51`, `OPS:339` |
| Module out dtype | fp32 | `glm5_next.py:197` |

KDA mode is on because `key_f_a` is set (`GDN:399`). There are no z/a projections: a low-rank sigmoid output gate
replaces z, and a low-rank per-k-channel forget gate replaces a.

**Path switches.**
- `EXL3_BC_GDN=0` disables the fused `BC_GatedDeltaNetSplit` (`GDN:750-800`). Every pass therefore runs the torch KDA
  branch `GDN:1126-1152`, then conv `:1177-1185`, delta rule `:1188-1204`, norm `:1208`, o_proj `:1212`.
- `x.is_cuda` is False on XPU, so the conv always uses Triton (`FN/conv1d.py:385-417`).
- The checkpoint stores `qkv_proj` fused. The alt keys `q_proj/k_proj/v_proj` (`glm5_next.py:180-181`) are unused. The
  `fused_qkvbfg_a_proj` named in the fp8 config does not exist in this checkpoint.

### 4.2 Decode step (Q = 1, not an MTP verify). Input y (1,1,4096) fp16 from `input_layernorm`

1. **qkv.** `qkv = qkv_proj(y)`: EXL3 K=6, 4096→24576, **fp32** (`GDN:1128`), plus LoRA (§9).
2. **Transpose to bf16.** `mixed_qkv = empty(1,24576,1, bf16).copy_(qkv.transpose(1,2))` (`GDN:1129-1131`). This is the
   fp32 → bf16 rounding point (RNE). Layout is channel-major (B, F, S).
3. **Output gate.** `z = g_b_proj(g_a_proj(y).to(fp16))` viewed as (1,1,64,128) (`GDN:1134-1135`).
   - Both are fp16 Linears with out_dtype fp32, implemented as `OPS:105-112 hgemm`: an fp16 GEMM, **rounded to fp16**,
     then widened to fp32.
   - g_a: 4096→128. g_b: 128→8192.
4. **Beta.** `beta = sigmoid(fp32(b_proj(y)) · 1.0).to(bf16)`, shape (1,1,64) (`GDN:1137-1138`). b_proj is fp16
   4096→64, fp16-rounded like step 3.
5. **Forget gate.** `GDN:1143-1150`, all fp32:
   ```
   f = f_b_proj(f_a_proj(y).to(fp16))            # 4096->128->8192, fp16-rounded values held in fp32
   g = (f + dt_bias[c])                          # dt_bias F32 [8192], PER CHANNEL (h*128+d)
   g = g.view(1,1,64,128)
   g = -5.0 · sigmoid( g · exp(A_log[h]) )       # A_log F32 [64]; g in (-5, 0), natural-log decay
   ```
   The softplus form `−exp(A_log)·softplus(f+dt_bias)` (`GDN:1151-1152`) is **not used**.
6. **Short conv.** Triton `_causal_conv1d_update_slotted_kernel` (`FN/conv1d.py:16-113`, launched at `:280-302`) gives
   (1,1,24576) **bf16**. Per channel c and token s:
   ```
   window[k] (k=0..3), src_t = s+k+1:  src_t < 4 ? conv_state[slot,c,src_t] : x[c, src_t-4]
   acc = Σ_k window[k]·w[c,k]        # fp32 accumulator; the bf16×bf16 product may be bf16-rounded first (Triton rule; verify)
   out = acc · sigmoid(acc)          # SiLU fp32, stored bf16
   ```
   - `conv_state[slot,c,1..3]` holds the previous 3 inputs, oldest first. Entry `[...,0]` is kept but never read.
   - Update without history: `conv_state[slot,c,j] = concat(state[0:4], x[0:S])[S+j]` for j = 0..3
     (`FN/conv1d.py:89-113`).
7. **Delta rule.** Seqlen < 64, so the recurrent branch runs (`FN/gated_delta_rule.py:150-174`):
   `ext.cuda_recurrent_gated_delta_rule` → `OPS:307-318` → Triton `KKDA:_kda_step_kernel`. The grid is (B, 64 heads,
   Dv/32); each program owns a (128 × 32) fp32 state tile (`KKDA:12-40`).
   ```
   q = fp32(conv[h*128 : ]),  k = fp32(conv[8192 + h*128 :]),  v = fp32(conv[16384 + h*128 :])
   q = q / sqrt(Σq² + 1e-6);  k = k / sqrt(Σk² + 1e-6)      # fp32, NOT rounded back to bf16
   S = S · exp(g[h,:])[:,None]                               # 1. decay rows (k index)
   u = v − Σ_k S[k,:]·k[k]                                   # 2. with the DECAYED S
   S = S + (fp32(beta[h])·k)[:,None] · u[None,:]             # 3. rank-1 update
   o = (Σ_k S[k,:]·q[k]) · 128^-0.5                          # 4. readout from the UPDATED S; scale on o, not q
   out[h,:] = bf16(o);   state[slot,0] = S
   ```
   **Torch fallback** (`OPS:321-349`): same math, but `q·rsqrt(...)` and einsum reductions. It runs when the Triton
   conditions fail, in particular **whenever MTP is on (state.shape[1] = mtp+1 > 1) and the step is not a history
   step**: MTP rounds without drafts, and prefill pieces shorter than 64 tokens.
8. **Gated RMSNorm (o_norm).** `EX/modules/gated_rmsnorm.py:104-121` → `OPS:285-292`, torch fp32:
   ```
   h = fp32(core) (1,1,64,128);  h = h · rsqrt(mean(h², -1) + 1e-5)     # per head over 128
   h = h · (fp32(o_norm.weight) + 0.0)                                  # weight bf16 [128], shared by all heads
   h = h · sigmoid(fp32(z))                                             # gate AFTER weight; sigmoid (gate_act=1), not SiLU
   -> fp16 (GatedRMSNorm out_dtype half, GDN:546), view (1,1,8192)
   ```
9. **o_proj.** EXL3 K=6, 8192→4096, **fp32** out (`GDN:1212,1218`), plus LoRA. It is consumed by `attn_hc.apply_`.

The whole layer runs inside block-graph phase A, one graph per (phase, slot) and (Q, hist) state. The slot index comes
from a static device int32 tensor (`block_graphs.py:536-546`).

### 4.3 MTP verify (Q = 1 + drafts ≤ 8, `recurrent_history=True`)

Same ops with S = Q.

- **Conv, history mode** (`FN/conv1d.py:89-113`).
  - The state width is `4 + max_history` (`GDN:187-191`).
  - Reads still use `[...,0:4]`.
  - Writes put the last `min(width, 4+S)` entries of `concat(state[0:4], x)` at the **tail** of the buffer.
  - After a history pass `[0:4]` is stale. **Rewind is mandatory even when every draft is accepted** (`rewind(0)`).
- **Recurrence, history mode** (`KKDA:37-40`, requires `nhist ≥ S`). The state after token t (t < S−1) goes to
  `state[slot, t+1]`, and the final state to `state[slot, 0]`. `advance_recurrent_states` sets `last_history = S−1`
  (`EX/cache/recurrent_util.py:74`).

### 4.4 Rewind (`GDN:118-126,241-253`)

On XPU this always runs the Python `GDNLayerState.rewind` (`GDN:46`). Let r = number of rejected tokens:

```
if r > 0:            state[slot,0] = state[slot, last_history + 1 − r]      # the state after the last accepted token
if last_history > 0: p = width − r; conv_state[slot,:,0:4] = conv_state[slot,:,p−4:p].clone()
```

It is called at `generator.py:1078-1089` (after rejection) and at `generator.py:1245-1246` (`rewind(0)`, all accepted).

### 4.5 Prefill pieces (`GDN:1035-1056`)

With states, bsz 1 and no history, x is split recursively:
- every `EXL3_KDA_SUBCHUNK` = 2048 tokens (`GDN:33`);
- at the in-pass checkpoint offset c (`recurrent_ckpt_offset`, page-aligned, `job.py:1311-1318,1400-1402`).

Each piece runs the whole layer and chains through the slot state. At c, `rsl.stash(slot)` copies `state[slot,:1]` and
`conv_state[slot,:,:4]` to the CPU (`GDN:1050-1055,293-298`). That copy becomes the checkpoint at the last full page.
The in-pass checkpoint is **disabled with an MTP draft attached**; a separate forward over the last page runs instead
(`job.py:1314-1322`).

### 4.6 Prefill delta rule (chunked)

**The chunked path runs only if the piece has `seqlen ≥ num_v_heads` (= 64; the threshold is the head count, not the
chunk size)** and it is not a history pass (`FN/gated_delta_rule.py:114`). Shorter pieces use the recurrent kernel.

Chunked path (`FN/gated_delta_rule.py:115-148`, `FLA/__init__.py:70-120`):
- q and k are L2-normalized by `l2norm_fwd` (eps 1e-6). **The output is rounded to bf16** (`FLA/l2norm.py:74-78`).
  Decode keeps them in fp32.
- g is overwritten in place by its 64-token chunk-local cumsum × 1/ln2 (log2 domain; `FLA/__init__.py:107`,
  `FLA/cumsum.py:126-133,196-197`).
- `chunk_kda_fwd_intra`, token-parallel, sub-chunks of 16 (`FLA/kda_chunk_intra.py:502-602`):
  - `Aqk[i,j] = scale·Σ q_i k_j exp2(g_i−g_j)` for j ≤ i.
  - `Akk[i,j] = β_i Σ k_i k_j exp2(g_i−g_j)` for j < i.
  - `A = (I+Akk)^-1`. Aqk and Akk are stored bf16; the diagonal blocks stay fp32.
- WY: `w = A(β k exp2 g)`, `u = A(β v)`, `kg = k exp2(g_last − g)` (`FLA/kda_wy_fast.py:85-128`).
- `chunk_gated_delta_rule_fwd_h` (`FLA/chunk_delta_h.py:156-300`), state in fp32, per chunk:
  1. store h as bf16;
  2. `v_new = u − w·h`;
  3. `h *= exp2(g_last)`;
  4. `h += kgᵀ v_new`.
- `chunk_gla_fwd_o_gk`: `o = scale·(q exp2 g)·h + tril(Aqk)·v_new`, bf16 (`FLA/gla_chunk.py:94-123`).
- The final fp32 state is copied to `state[slot,0]`.
- `TRITON_F32_DEFAULT=ieee`; no TF32.

The conv uses one kernel for S ≤ 256 and output + state kernels otherwise (`FN/conv1d.py:303-337`); the math is the
same.

### 4.7 KDA gotchas

1. The gate is `g = −5·sigmoid(exp(A_log[h])·(f + dt_bias[c]))`: natural log, per k-channel. `dt_bias` has 8192
   entries.
2. The output gate is a **sigmoid** applied after RMSNorm·weight. The norm runs per head over 128 with one [128] weight;
   the output is fp16.
3. All five small fp16 projections are fp16-rounded before being held in fp32.
4. q/k normalization: fp32 at decode, bf16-rounded at prefill.
5. Order within a token: decay, then u from the decayed S, then update, then readout from the updated S.
6. Prefill pieces are cut at 2048 and at the checkpoint. Tails shorter than 64 tokens switch to the recurrent algorithm.
7. With MTP on, non-history recurrent steps take the torch loop.
8. Conv: only entries 1..3 of the 4-wide state are read. History passes leave `[0:4]` stale until rewind.

---

## 5. MLA attention with the DSA indexer (layers 3,7,…,43 and MTP 45)

Prefixes: `MLA` = `EX/modules/mla_attn.py`, `MT` = `EX/modules/attention_fn/mla_triton.py`, `DT` =
`EX/modules/attention_fn/dsa_triton.py`.

### 5.1 Which path runs

| Situation | Path |
|---|---|
| Trunk decode (bsz 1, Q = 1) and MTP verify (Q ≤ 8) | `block_graphs.py:556-562` → `GX/mla_graph.py:71-155`, captured in graph phase "Am". **Always the gathered (sparse) form, at every context length.** Gated by `GLM_XPU_MLA_GRAPH` (default 1, `block_graphs.py:70`) and `mla_graph.supported` (`GX/mla_graph.py:57-59`) |
| Prefill, MTP draft layer 45 (no mHC, so the graph path declines), anything else | eager `MLAttention.forward` (`MLA:411-462`) → `decode_flash_attn` (`MLA:1023-1076`) → `_attend` (`MLA:810-978`) |
| `bc_mla` | **disabled** by `EXL3_BC_ATTN=0` (`EX/modules/attention_fn/bc_mla.py:574-575`) |
| Triton `dsa_attn`, `mla_attn_triton_prefill*` | not reached; the XPU branches at `MLA:935,1002` take over |

### 5.2 Module constants

- H = 64 q heads with an MQA latent (`MLA:126`). D_c = 512, nope = 256, rope = 0, v = 256.
- `sm_scale` = 0.0625 (`MLA:143-145`). It follows the unabsorbed head dim 256, not 512. No l4/mscale (`l4_beta` = 0,
  `MLA:329`).
- Indexer: H_i = 32, D_i = 128, topk 2048, P = 4, tail on.
- **`index_norm_eps` = 1e-6**: the constructor default (`MLA:110`), not rms_norm_eps.
- The q_a / kv_a RMSNorms use eps 1e-5 (`MLA:230-232,248-250`). Their weights are bf16, kept bf16.
- Out dtype fp32.

### 5.3 Weights and load-time transforms (`MLA:225-367`)

| Tensor (`layers.L.self_attn.`) | Format | Shape | Use |
|---|---|---|---|
| `q_a_proj` | EXL3 K=6 (MTP K=4) | 4096→1536 | fp16 out |
| `q_a_layernorm.weight` | BF16 | [1536] | RMSNorm 1e-5 |
| `q_b_proj` | EXL3 K=6 (K=4) | 1536→16384 (64×256) | fp16 out |
| `kv_a_proj_with_mqa` | EXL3 K=6 (K=4) | 4096→512 (+0 rope) | fp16 out |
| `kv_a_layernorm.weight` | BF16 | [512] | RMSNorm 1e-5 |
| `kv_b_proj.weight` | **F16, unquantized** | [32768, 512] | rewritten into `w_uk_flat` and `w_uv_flat` |
| `o_proj` | EXL3 K=6 (K=4) | 16384→4096 | **fp32 out** |
| `indexer.wq_b` | EXL3 K=6 (K=4) | 1536→4096 (32×128) | fp16 out; input is q_resid (post q_a_layernorm) |
| `indexer.wk.weight` | F16 | [128, 4096] | fp16 GEMM |
| `indexer.k_norm.{weight,bias}` | F16 | [128] | **biased LayerNorm**, eps 1e-6 |
| `indexer.weights_proj.weight` | F16 | [32, 4096] | fp16 GEMM, fp16 out |
| `indexer.index_kpool_compress_gate` | F16 | [128, 4096] | raw tensor, `F.linear(x.half(), G)` (`MLA:325-327,513`) |
| `indexer.index_kpool_compress_ape` | **F32** | [4, 128] | raw, kept fp32 (`MLA:322-324`) |

**kv_b layout.** Row `h·512 + j`: j < 256 is W_UK[h,j,:], j ≥ 256 is W_UV[h,j−256,:]. Load rewrites it into:
- `w_uk_flat[c, h·256+j] = W[h·512+j, c]`, shape (512, 16384) fp16;
- `w_uv_flat[c, h·256+v] = W[h·512+256+v, c]`, shape (512, 16384) fp16 (`MLA:362-367`).

The LoRA delta for kv_b is merged into these flats (§9).

### 5.4 Decode, graph path (`GX/mla_graph.py:decode`)

Input x (1,Q,4096) fp16 (y_attn). Query row t sits at position `pos_t = cs + t`, where cs is the number of tokens cached
before this step.

Static inputs (`GX/mla_graph.py:27-54`):
- `S.cs` int32 (1);
- `S.bt` int32 (1, n_pages_total): the job's block table, zero-padded, so the padding points at page 0 and is masked;
- `S.indices` int32 (8, 2080) = (QMAX, ceil((512·4 + 3)/32)·32), initialized to −1.

Steps:

1. **Queries** (`project_q`, `MLA:465-472`).
   - `q_resid = RMSNorm(q_a_proj(x))`: (1,Q,1536) fp16.
   - `q = q_b_proj(q_resid)` viewed as (Q,64,256) fp16.
2. **Latent.**
   - `ckv = kv_a_layernorm(kv_a_proj_with_mqa(x)[..., :512])`: (1,Q,512) fp16. This is the cached value.
   - `k_pe` is empty.
3. **Absorb** (`MT:1055-1070`, kernel `_mla_absorb_kernel` `MT:555-591`).
   - `q_lat[h,r,c] = Σ_{j<256} q[r,h,j]·w_uk_flat[c, h·256+j]`.
   - One `tl.dot` over K = 256, fp32 accumulation, **stored fp16**. Shape (64,Q,512), head-major.
4. **Append the latent** (`MT:791-810`, kernel `MT:113-146`). Row t goes to physical row `bt[(cs+t)//256]·256 + (cs+t)%256`
   of `layer.k`.
5. **Indexer key + gate** (`MLA:498-514`, uncommitted working-tree version).
   - `k = LayerNorm(wk(x), w, b, eps=1e-6)`: (1,Q,128) fp16 (`EX/modules/layernorm.py:97-106`). **No rope**:
     `_indexer_rope_` returns at once (`MLA:481`).
   - `gate = F.linear(x.half(), compress_gate)`: (1,Q,128) fp16. **It is a projection of the hidden x, not of k.**
   - Two `mla_plane_append` launches (`MT:813-837`, kernel `MT:149-177`, DST_D = 256) write k at offset 0 and gate at
     offset 128 of the 256-wide `k_idx` row.
6. **Pooled key, recomputed every step** (`GX/mla_graph.py:99-108`).
   - `pool = pos // 4`; its members are tokens `4·pool + 0..3`, read from k_idx through bt.
   - `probs = softmax_over_members(fp32(gate_m) + ape[m])`, **per dimension** (dim = member, 4-way).
   - `pool_key = fp16(Σ_m probs·k_m)`, written to `k_pool` row `bt[pool//64]·64 + pool%64`.
   - An incomplete pool holds stale members but is invisible to the scorer until the step that completes it rewrites it.
7. **Scores over every addressable pool slot** (n_pools = n_pages_total·64) (`GX/mla_graph.py:111-121`):
   ```
   kp    = k_pool[prow]                      (n_pools,128) fp16
   q_idx = wq_b(q_resid).view(Q·32, 128)     fp16
   w     = weights_proj(x).view(Q, 32)       fp16
   sc[t,j] = Σ_h (fp32(w[t,h]) · 32^-0.5) · relu( fp32(q_idx[t,h])·fp32(kp[j]) · 128^-0.5 )     # all fp32
   sc[t,j] = -inf where j >= (pos_t + 1) // 4      # only COMPLETE pools are visible
   sc = fp16(sc)
   ```
8. **Selection** (`GX/mla_graph.py:125-140`).
   - Per row: `torch.sort(sc[t], descending, stable)`, so ties go to the lower index.
   - Take the first k_sel = 2048/4 = **512 pools**; −inf entries become −1.
   - Each pool p expands to tokens 4p..4p+3 in columns [0, 2048), **in score order**.
   - Tail: with `vis = pos+1` and `tc = vis % 4`, columns 2048..2050 get `vis − tc + i` for i < tc, else −1.
   - At most 2051 tokens are selected.
9. **Gathered latent attention** (`GX/sparse_mla.py:15-40`, row chunks of 128).
   ```
   rows = bt[idx//256]·256 + idx%256;  kv = latent[rows]   (c, K, 512) fp16
   s = fp32( bmm(q_lat_fp16, kvᵀ) )   ← bmm OUTPUT IS FP16-ROUNDED, then ·0.0625; −1 entries → −inf
   p = fp16( nan_to_num(softmax_fp32(s), 0) )
   o_lat = bmm(p, kv)                 (64, Q, 512) fp16
   ```
10. **Unfold** (`MT:1073-1086`, kernel `MT:594-623`).
    - `o[r,h,v] = Σ_c o_lat[h,r,c]·w_uv_flat[c, h·256+v]`: four fp32-accumulated K = 128 dots, **stored fp16**.
    - Result (Q,64,256), reshaped to (1,Q,16384).
11. **o_proj**: EXL3, **fp32** out.

At ≤ 2048 context the selected set is "all tokens 0..pos", so the result equals dense attention mathematically. It is not
bitwise identical to the eager dense kernels.

### 5.5 Eager path (prefill, MTP layer, graph declined), `MLA:_attend`

- **Sub-chunks.** A chunk longer than `EXL3_MLA_SUBCHUNK` = 1024 runs as sequential 1024-token recursive forwards with
  `cache_seqlens + a` (`MLA:418-442`, `:41`).
- **Sparse decision.** `sparse = max(host_seqlens) + seqlen > 2048`, evaluated per (sub)chunk (`MLA:829-834`).
- **Order.** Q proj → kv_a + norm → absorb (if not MHA) → latent append (`EX/cache/mla.py:113-128`) → indexer keys + two
  plane appends → pool update → attention (`MLA:838-895`).
- **Pool update** (`MLA:611-644`). Writes **only pools completed by this chunk**: pools `pos0//4` up to
  `(pos0+seqlen)//4`. Same per-dim fp32 softmax(gate + ape), cast to fp16.
- **Dense, seqlen ≤ 16** (this is the MTP draft layer below 2048 context): Triton `mla_attn_triton_decode`
  (`MT:909-1052`; `_mla_decode_split_kernel` `MT:205-343`, combine `MT:346-396`).
  - fp16 dot, fp32 accumulation, **fp32 scores**, ×scale, causal `n ≤ total − q_len + row`.
  - Online softmax in fp32; p cast to fp16 before PV; fp32 acc.
  - Flash-decoding splits with `block_n = 32`, `num_splits = min(2·SM/programs, 128, ceil(max_k_len/128))`.
  - Then unfold and o_proj.
- **Dense, seqlen > 16** (prefill ≤ 2048 total): `mla_prefill_mha_sdpa` (`GX/sparse_mla.py:66-110`).
  - Gather every cached latent row of the sequence (past + chunk).
  - Per group of 16 heads: `K = ckv @ w_uk_flat[:, grp]` and `V = ckv @ w_uv_flat[:, grp]` as fp16 GEMMs; **K and V are
    materialized in fp16**.
  - Query blocks of 512.
  - `F.scaled_dot_product_attention(scale=0.0625)`: `is_causal` for the first block when past = 0, else the boolean
    mask `j ≤ past + i`.
  - Output (R,64,256) fp16 → o_proj.
- **Sparse** (`MLA:900-926`, `_indexer_topk_kpool` paged branch `MLA:687-760`).
  - Rows in slabs of 256; `k_sel = min(512, (L + r1)//4)`.
  - Scorer `dsa_indexer_scores` (`DT:1111-1182`): `_dsa_indexer_fewq_kernel` for R ≤ 4 (`DT:664-730`), else
    `_dsa_indexer_kernel` (`DT:603-659`).
    - `score = (1/64)·Σ_h fp32(w_h)·relu(fp32 dot(q_h, kpool))`. The combined scale 128^-0.5·32^-0.5 = 1/64 is applied
      **after** the head sum.
    - Causal bound `(pos+1)//4`. Stored fp16; pool tiles of 32768 with an exact running merge above that.
  - `ext.dsa_topk` = `GX/ops.py:354-367`:
    1. fp32 stable sort descending; take k;
    2. −inf/NaN → sentinel;
    3. **sort ascending**;
    4. sentinel → −1.
  - **Order is ascending index**, unlike the graph path's score order.
  - Expand pools to tokens and append the tail (`MLA:673-685`, same rule).
  - `sparse_mla_latent` (`MLA:1002-1009`) → unfold → o_proj.

| | Decode (graph) | Prefill / eager |
|---|---|---|
| Sub-chunking | none | 1024 tokens |
| Context ≤ 2048 | gathered over all tokens | MHA SDPA (seqlen > 16) or Triton absorbed decode (seqlen ≤ 16) |
| Context > 2048 | gathered | Triton scorer + torch sort + gathered |
| Pool plane | current pool rewritten every step | completed pools only |
| Scorer | torch fp32 matmul over all cache pool slots; scale before the head sum | Triton over visible pools; scale after the head sum |
| Index order | score order | ascending |

### 5.6 MLA gotchas

1. No RoPE anywhere, despite `indexer_rope_interleave` in the config. Position enters only through causal bounds and
   pool/tail arithmetic.
2. k_norm is a biased LayerNorm with eps 1e-6. q_a / kv_a are RMSNorms with eps 1e-5.
3. The pool gate is a linear of x. The pool softmax runs per dimension over 4 members, and APE is fp32, added before
   the softmax.
4. The indexer score applies ReLU per head **before** the weighted head sum. weights_proj is raw (no softmax/sigmoid),
   fp16. Scores are rounded to fp16 before top-k, and the stable sort breaks ties toward the lower index.
5. **The current token is not guaranteed to be attended.** When (pos+1) % 4 == 0 its pool is complete and must rank in
   the top 512. Otherwise the tail includes it.
6. The graph decode never runs a dense kernel. Eager decides `cache + chunk > 2048` per 1024-token sub-chunk.
7. Rounding differs per path:
   - `sparse_mla_latent` rounds QK to fp16 **before** the 0.0625 scale (overflow risk above 65504) and rounds P to fp16;
   - the Triton dense decode keeps fp32 scores;
   - MHA prefill rounds the up-projected K/V to fp16.
8. q_lat, o_lat and the unfold output are fp16. Only o_proj emits fp32.

---

## 6. MLP and MoE

Prefixes: `MLP` = `EX/modules/mlp.py`, `BSM` = `EX/modules/block_sparse_mlp.py`, `RT` =
`EX/modules/block_sparse_mlp_routing.py`, `MOE` = `CS/exl3_moe.n128.sycl`.

The library built and loaded is `CS/exl3_moe.n128.sycl`, compiled to `_moe_glm.so` / `_moe_prefill.so` with
`-DEXL3_ACT_LIMIT=10.0f` (`XP/build.sh:31-43`). `CS/exl3_moe.sycl` is **not built**.

### 6.1 Construction

- **Dense layers 0-2** (`glm5_next.py:254-270`): `GatedMLP(4096 → 12288 → 4096, silu, act_limit 10, interm fp16, out
  fp32)`, one slice.
- **MoE layers 3-44** (`glm5_next.py:271-307`): `BlockSparseMLP(288 experts, top-8, router_type "dots", rsf 2.5,
  act_limit 10, interm fp16, out fp32)`.
  - Shared expert: `GatedMLP(4096 → 2048 → 4096)`, same act / dtypes.
- **MTP layer 45**: the same, with `mtp_bits` (`glm5_next_mtp.py:102-140`).
- **Router gate**: fp16 `Linear(4096 → 288)`, stored transposed (4096, 288); routing uses `gate_tensor[:, :288]`
  (`BSM:222-230`, `GX/ops.py:184`). `e_score_correction_bias` is F32 [288], loaded `.float()`
  (`GX/moe_patch.py:29-44`).

### 6.2 Clamped SwiGLU (dense, shared, and routed experts)

```
act = min(silu(fp32 g), 10) · clamp(fp32 u, −10, 10)      # clamp AFTER silu, from above only; up clamped both sides
```

- Dense and shared experts use torch `silu_mul` (`GX/ops.py:147-154`), storing fp16 in place into u's buffer.
- Routed experts compute it inside the SYCL kernel (macro `MOE:37-44`).
- The comment there notes that this differs from sglang, which clamps before silu.

### 6.3 Dense MLP and shared expert (per-linear path; LoRA disables the fused `BC_GatedMLP`, `GX/lora.py:156-161`)

1. `g = gate_proj(y)`: EXL3 K=6, **fp16** out, plus LoRA.
2. `u = up_proj(y)`: fp16, plus LoRA.
3. `a = fp16(min(silu(g),10)·clamp(u,±10))`.
4. `d = down_proj(a)`: EXL3 K=6, **fp32** out, plus LoRA (`MLP:724-808`, `:541,577`).

The shared expert runs the same steps with I = 2048 (K=4 in MTP).

### 6.4 Router math

**(a) Exact bsz-1 router.** Used by decode with Q = 1 and `RES_BONUS = 0`, by prefill, and by the MTP draft layer. Path:
`routing_dots` (`RT:210-259`) → `ext.routing_ds3_nogroup` = `GX/ops.py:177-193`.

```
logits = fp16( y_fp16[1,4096] @ gate_fp16[4096,288] )           # fp16 GEMM; moe_router_dtype=float32 is IGNORED
s      = sigmoid(fp32(logits))
key    = s + fp32(bias_h),  bias_h = fp16(esb − mean(esb))      # mean-centred, fp16-rounded (RT:43-51)
idx    = topk(key, 8, sorted=True)                              # descending, torch tie rule
w      = s[idx];  w = w · 2.5 / (Σ w + 1e-20)                   # fp32, unbiased scores, normalized over the 8
outputs: idx int64; w stored FP16 (bsz-1 / bucketed buffers are fp16, BSM:751-753, RT:19-40)
```

`block_graphs` copies them into `st.sel` int32 and `st.w` fp32 (`block_graphs.py:267-277`), so the **weights are
fp16-rounded**.

**(b) Production "quality" decode** (`RES_BONUS = 0.1`). Fused branch (`block_graphs.py:244-261`) with the Triton
`k_router.route` (`GX/k_router.py:21-84`):
- `lc = fp16 GEMM(y, gate)`;
- `s = 1/(1+exp(−fp32 lc))`;
- `key = s + fp32 esb` (**uncentered fp32 bias**);
- `exact = top8(key)`, kept for bookkeeping;
- `sel = top8(key + bonus)`, where bonus = 0.1 for experts resident in VRAM or RAM (`GX/tier.py:589-594`);
- `w = s[sel]·2.5/(Σ+1e-20)` in **fp32**, with no fp16 rounding;
- top-k runs as K rounds of argmax-and-mask, so ties go to the lowest index.

**This makes routing depend on cache residency. Not model math.**

**(c) MTP verify (Q > 1).** Always the `k_router` kernel (`FUSED_ROUTER` and `Q > 1`), bonus 0 when `RES_BONUS = 0`.
This gives an exact selection with the fp32 bias and fp32 weights. So the two exact routers differ numerically:
(a) uses an fp16 centered bias and fp16 weights, (c) uses an fp32 bias and fp32 weights.

**Other modes, off by default** (`block_graphs.py:24-67`): NOSYNC ("fast" mode), SUBST_KEEP < 8, SPEC_RO. All of them
are approximations.

The prefetch prediction (`block_graphs.py:215-223`, `GX/moe_patch.py:75-86`) only gives residency hints and never
changes the math.

### 6.5 Routed experts kernel (`MOE`)

Entry `moe_forward(x fp16 [M,4096], sel int32 [M,8], w fp32 [M,8], ptrs int64 [288], I=2048, Kcode, E=288)` →
`moe_forward_impl` (`MOE:1363-1470`). It returns **fp16** [M,4096] (`MOE:1414`); the caller does `.float()` then
`+= shared` (fp32).

Configuration (`GX/ops.py:461-482`):
- k-splits gate/up P = 8, down P = 4;
- pair mode for M = 1;
- DPAS prefill path for M ≥ 256, for K codes 2, 3, 4, 25, 35.

K codes: half-integer K → `10·⌊K⌋ + 5` (2.5 → 25, 3.5 → 35) (`GX/tier.py:38-44`).

**GateUpKernel** (`MOE:154-288`), per (pick, 128-column block):

1. **Input Hadamard, separately for gate and up**, per 128-element k block:
   ```
   v = fp16( fp32(x)·fp32(suh) );  v = fp16( fwht128(v)·0.08838834764831845 )
   ```
   `fwht128` is the unnormalized Sylvester transform in fp32 (`CS/exl3_esimd.h:222-235`).
2. **Trellis GEMV** (mul1 decode, `exl3_esimd.h:50-70,507-532`).
   - Products accumulate in **fp16 (`hacc`) within one 16×16 tile-row**, then are added into fp32.
   - fp32 continues across tile-rows and across the P = 8 splits, which are combined through SLM in order 0..P−1.
3. **Epilogue.**
   ```
   g = fp16(fwht128(g)·rsqrt128·svh_g);   u = fp16(fwht128(u)·rsqrt128·svh_u)
   act = min(g/(1+exp(−g)), 10) · clamp(u, ±10)                 # -ffast-math exp
   act = fp16(act); act = fp16(act·suh_d); fwht128; abuf = fp16(act·rsqrt128)   # the down input Hadamard, fused
   ```

**DownKernel** (`MOE:291-388`): same GEMV, P = 4. `y[pair] = fp32( fwht128(v)·(rsqrt128·w)·svh_d )`, where w is the
routing weight. `y` is fp32 [npairs, 4096].

**CombineKernel** (`MOE:391-418`): `out[m] = fp16( Σ_{k=0..7} y[pair(m,k)] )`, summed in fp32 in **pick (sel column)
order**.

**M in 2..255** (MTP verify, short prefill pieces): the align kernel (`CS/exl3_moe_align.sycl:8-49`) groups picks per
expert. MR = 2 for M ≤ 64, else 4. The math per row is the same.

**M ≥ 256 prefill**: `moe_prefill` (`MOE:891-944`; `GateUpDpasKernel` `:484-610`, `DownDpasKernel` `:613-697`, MB = 32).
The rounding points are the same, but the GEMMs accumulate in **fp32 DPAS**, with no fp16 tile partials, and the
per-pair y is stored **fp16** (`MOE:901`) before the combine.

### 6.6 MoE layer flow, eager (prefill and the MTP layer; `GX/moe_patch.py:68-94`)

1. `y = x.view(−1,4096).half()`.
2. `sel, w = routing_fn(...)`, the exact router (a).
3. `out = TIER.moe(li, y, sel, w).float()` (`GX/tier.py:1314-1343`).
4. `out += shared_experts(x)`.
5. Returned as fp32.

### 6.7 MoE gotchas

1. The production decode router is not exact (bonus). Prefill and MTP are exact.
2. There are two exact routers with different rounding (§6.4a vs c).
3. Router logits are fp16; `moe_router_dtype` is ignored.
4. The clamp comes after silu.
5. fp16 roundings inside the expert kernel:
   - x·suh;
   - after the input FWHT;
   - g and u after svh;
   - act;
   - act·suh_d;
   - the down input;
   - the combined output, which is rounded to fp16 before the fp32 shared-expert add.

   Decode GEMV also uses fp16 tile-row partials.
6. The kernel's exp is fast-math. Dense and shared experts use torch fp32 SiLU.
7. Expert K varies per layer: 3.5 (layer 3), 2.5 (layers 4-44), 4 (MTP). Dense and shared experts are K=6 (MTP K=4).

---

## 7. EXL3 linear (trellis) and fp16 linears

### 7.1 Dispatch

Every non-expert EXL3 linear goes through the same chain:

1. `no_reconstruct=True` (`GX/runner.py:39`) makes `LinearEXL3.forward` call `bc.run_alloc(x, out_features,
   dtype == float)` (`EX/modules/quant/exl3.py:137-142`). The input must be contiguous (`exl3.py:135`).
2. The output dtype is the `out_dtype` argument, else `Linear.out_dtype`, else fp16 (`exl3.py:75,141`).
3. That lands in `BC_LinearEXL3.run` (`GX/ops.py:60-80`):
   - x is cast to fp16 first, so fp32 inputs get rounded.
   - M ≤ 128 rows: `exl3_gemm_small`, writing straight into y (fp16 or fp32).
   - M > 128 rows: `E.linear(...)`, then `y.copy_(out)`.
   - The bias is added last. No linear in this model has one.

Only the mul1 codebook and integer K run on this path (`GX/ops.py:46-47`). `suh` is viewed as [1, K] and `_sonb` is
int32 zeros of length N/128 (`:55-58`).

`Linear.forward` (`EX/modules/linear.py:594-637`) wraps the call: input pad, inner forward, output trim, LoRA add, then
pre_scale, softcap and post_scale. GLM sets none of the scales and no padding triggers.

### 7.2 Math

```
xh[m,k] = fp16( (1/√128) · FWHT128( fp32( fp16(x[m,k] · suh[k]) ) ) )      # per 128-block of k
acc[m,n] = Σ_k xh[m,k] · W[k,n]                                           # W = trellis decode, Hadamard domain
y[m,n]  = out_dtype( svh[n] · (1/√128) · FWHT128_n( acc ) )                # per 128-block of n
```

**Input side.** `x·suh` is rounded to fp16 before the transform (`CS/exl3_esimd.h:309-317`). The FWHT is unnormalized
fp32 (`:222-233`) and uses `kRsqrt128 = 0.08838834764831845f` (`:235`).

**Output side** (`HadOutKernel`, `exl3_esimd.h:332-354`):
1. Sum the fp32 split partials over p = 0..P−1, in that order.
2. FWHT.
3. Multiply `v · kRsqrt128 · svh`, in that order, in fp32 (`:347`).
4. Convert to the output dtype.

The CPU reference with both Hadamards folded is `get_weight_tensor` (`EX/modules/quant/exl3.py:239-249`).

### 7.3 Trellis layout and mul1 decode

**Storage.**
- Tensors are int16 `[in/16, out/16, 16·K]`; K = last dim / 16 (`exl3.py:66-68`).
- The kernel reads them as uint32 `[in/16][out/16][8K words]`, little-endian pairs of the int16s (`CS/exl3_ops.sycl:87-88`).
- Tile (r, c) starts at word `(r·tiles_n + c)·8K` (`exl3_esimd.h:574-579`).

**One 16×16 tile** is 256 values in an MSB-first bit stream that wraps around inside the tile:
- Value t is the 16-bit window ending at bit (t+1)·K, so the state's LSB is stream bit (t+1)K−1.
- Stream bit b sits in word b/32 at bit 31 − (b % 32).
- Windows that start before bit 0 wrap to the tile's last word (`exl3_esimd.h:132-185`).

**Placement of value t** (`exl3_esimd.h:3-8`):
- `row (k) = 8·((t>>1)&1) + 2·((t>>3)&3) + (t&1)`
- `col (n) = (t>>5) + 8·((t>>2)&1)`

**Half-integer K** (experts only): positions alternate KA and KA+1 bits, and the odd positions get the extra bit (mask
0xAAAA). Value u ends at bit `(u+1)·KA + (u+1)/2` (`exl3_esimd.h:26-46`).

**mul1 decode, bit-exact** (`exl3_esimd.h:50-80`):

```
x  = st · 0x83DCD12D  (uint32, mod 2^32; st = 16-bit state)
h  = fp16 bits (0x6400 + byte0(x)+byte1(x)+byte2(x)+byte3(x))   # = 1024 + bytesum, exact
w  = fp16( h · fp16(0.00676727294921875) + fp16(−10.3828125) )   # one fp16 FMA rounding (__hfma 0x1eee, 0xc931)
```

The `.mul1` I32 scalar in each linear holds 0x83DCD12D. It is only a codebook flag (`exl3.py:77-80`). The mcg and 3INST
codebooks are unused.

### 7.4 Accumulation depends on M (`CS/exl3_ops.sycl:258-321,651-757`)

| M rows | Kernel | Accumulation |
|---|---|---|
| ≤ 2 (decode) | vector GEMV | fp16 per-lane partials within a tile-row (`hacc`, about 8 fp16 MADs per lane-half at K=6), folded into fp32 (`exl3_esimd.h:589-592`). Lanes are summed in fp32. Split-K partials go to `part[P,M,N]` fp32 |
| 3..128 (MTP verify, short pieces) | DPAS/XMX `dpas<float,float,fp16,fp16>` | fp32. Rows padded to MB ∈ {8,16,24,32,40,48,64} (`:276-289`) |
| > 128 (prefill) | `exl3_linear` | had_in to fp16 → reconstruct fp16 W in 16384-column slices (`GX/ops.py:22`) → `at::matmul_out` fp16 GEMM, **fp16 output** (`:752`) → `exl3_had_out_h` with P=1 → output in **x's dtype, fp16** (`:658`), widened by `GX/ops.py:78` |

- The split count is `P = clamp(ceil(target_threads/(n_strips·m_blocks)), 1, tk)` with targets 1024, 1408 and 2048
  (`:177-179,299-304`). The summation order therefore depends on M and N.
- The fused single kernel is off (`g_fused=false`, `:255`), and so is W8A8.
- Supported (K, cb) pairs: K ∈ {3,4,5,6} with mul1 (`:767-774`).

### 7.5 fp16 Linears (`EX/modules/quant/fp16.py:69-100`)

- The weight is stored transposed as (in, out) fp16 (`EX/modules/linear.py:212`).
- Same dtype in and out → `torch.matmul(x, W, out=y)`.
- fp16 in, fp32 out → `GX/ops.py:105-112 hgemm`: an fp16 GEMM whose output is **rounded to fp16, then widened**. This
  covers KDA b_proj, f_a, f_b, g_a and g_b.
- The indexer `wk`, `weights_proj` and the router are fp16 → fp16.
- `index_kpool_compress_gate` is a raw `F.linear` in fp16.

### 7.6 Embedding and lm_head

- **Embedding**: bf16 [154880, 4096] on the CPU. Plain lookup with no scaling. Trunk output is fp32
  (`out_dtype=torch.float`, `EX/modules/embedding.py:19`). The MTP input layer asks for fp16 instead.
- **lm_head**: EXL3 K=6, no out_dtype, so **fp16 logits**. It is outside the block graphs and is not tied.

### 7.7 Weight inventory (safetensors headers; all 15 shards have empty `__metadata__`)

Every EXL3 linear has `trellis` I16 `[in/16, out/16, 16K]`, `suh` F16 `[in]`, `svh` F16 `[out]` and `mul1` I32 scalar.
There is no su, sv, mcg or bias.

**Top level**

| Tensor | Format | Shape |
|---|---|---|
| `model.language_model.embed_tokens.weight` | BF16 | [154880, 4096] |
| `model.language_model.norm.weight` | BF16 | [4096] |
| `lm_head` | EXL3 K=6 | 4096→154880 (trellis [256, 9680, 96]) |

There are no `hc_head_*` or `hc_expand` tensors.

**Every trunk layer** (prefix `model.language_model.layers.N.`)

| Tensor | Format | Shape |
|---|---|---|
| `hc_attn_fn`, `hc_ffn_fn` | F32 | [24, 16384] |
| `hc_attn_base`, `hc_ffn_base` | F32 | [24] |
| `hc_attn_scale`, `hc_ffn_scale` | F32 | [3] |
| `input_layernorm.weight`, `post_attention_layernorm.weight` | BF16 (kept bf16) | [4096] |

**KDA** (`self_attn.`)

| Tensor | Format | Shape |
|---|---|---|
| `qkv_proj` | EXL3 K=6 | 4096→24576, trellis [256,1536,96] |
| `o_proj` | EXL3 K=6 | 8192→4096, trellis [512,256,96] |
| `conv1d.weight` | BF16 | [24576,1,4] (one fused conv; no q_conv1d / k_conv1d / v_conv1d) |
| `b_proj.weight` | F16 | [64,4096] |
| `f_a_proj.weight`, `g_a_proj.weight` | F16 | [128,4096] |
| `f_b_proj.weight`, `g_b_proj.weight` | F16 | [8192,128] |
| `A_log` | F32 | [64] |
| `dt_bias` | F32 | [8192] |
| `o_norm.weight` | BF16 | [128] |

**MLA**: see the table in §5.3. All five EXL3 projections there are K=6 in layers 3-43 and K=4 in layer 45.

**Dense MLP** (layers 0-2)

| Tensor | Format | Shape |
|---|---|---|
| `mlp.gate_proj`, `mlp.up_proj` | EXL3 K=6 | 4096→12288, trellis [256,768,96] |
| `mlp.down_proj` | EXL3 K=6 | 12288→4096, trellis [768,256,96] |

**MoE** (layers 3-44 and 45)

| Tensor | Format | Shape |
|---|---|---|
| `mlp.gate.weight` | F16 | [288,4096] |
| `mlp.gate.e_score_correction_bias` | F32 | [288] |
| `mlp.shared_experts.{gate,up}_proj` | EXL3 K=6 (L45: K=4) | 4096→2048, trellis [256,128,96] (L45: [...,64]) |
| `mlp.shared_experts.down_proj` | EXL3 K=6 (L45: K=4) | 2048→4096 |
| `mlp.experts.{0..287}.{gate,up}_proj` | EXL3 | 4096→2048, trellis [256,128,16K] |
| `mlp.experts.{0..287}.down_proj` | EXL3 | 2048→4096, trellis [128,256,16K] |

- **Routed-expert K**: layer 3 is 3.5 (width 56), layers 4-44 are 2.5 (width 40), layer 45 is 4 (width 64).
- These expert tensors are present in the safetensors, but the trunk never loads them (`GX/moe_patch.py:29-44`). They
  come from the store blob (§10).

**MTP (layer 45)**

| Tensor | Format | Shape |
|---|---|---|
| `enorm.weight`, `hnorm.weight` | BF16 | [4096] |
| `eh_proj` | EXL3 K=4 | 8192→4096, trellis [512,256,64] |
| `shared_head.norm.weight` | BF16 | [4096] |

Layer 45 also has the MLA + MoE block with all EXL3 at K=4 and no `hc_*` tensors.

---

## 8. Caches and recurrent state

### 8.1 Paging

- `PAGE_SIZE = 256` (`EX/constants.py:2`).
- `Cache(model, max_num_tokens=--cache, layer_type=CacheLayer_fp16, max_history=mtp)` (`GX/runner.py:44-45`). The
  launcher passes `ctx ≤ 65536`, default 49152 (`llm/b70:1312`).
- `block_table` is int32 (bsz, pages_per_seq) of physical page ids, shared by all layers.
- `cache_seqlens` is int32 (bsz), counting the tokens cached *before* this forward. Appends write at `cache_seqlens + i`
  (`MT:134-138,169-173`).

### 8.2 MLA layer cache

MLA overrides the layer type with `CacheLayer_MLA_fp16` (`MLA:285-298`, `EX/cache/cache.py:149-158`). All four planes
are zero-initialized (`EX/cache/mla.py:77-82`):

| Plane | Shape | Dtype | Content |
|---|---|---|---|
| `k` (latent) | (pages, 256, 1, 512) | fp16 | post-kv_a_layernorm latent; serves as both K and V |
| `v` (rope key) | (pages, 256, 1, 0) | fp16 | empty (NoPE) |
| `k_idx` | (pages, 256, 256) | fp16 | per token `[LayerNorm(wk(x)) 128 ‖ gate = x·Gᵀ 128]` (`MLA:170-173`) |
| `k_pool` | (pages, 64, 128) | fp16 | one pooled key per 4 tokens; pools never straddle pages (`EX/cache/mla.py:56-59`) |

- Cost: 1600 B per token per MLA layer.
- Addressing: token t → `bt[t//256]·256 + t%256`; pool p → `bt[p//64]·64 + p%64`.
- `copy_page` copies all four planes (`EX/cache/mla.py:158-167`).
- There is no separate DSA cache: `EX/cache/dsa.py` is unused.
- After a rewind, stale rows past the new length are simply overwritten. Pools are recomputed when real tokens complete
  them.

### 8.3 KDA state (per KDA layer; `GDN:187-219`)

| Tensor | Shape | Dtype |
|---|---|---|
| `conv_state` | [max_batch, 24576, 4 + max_history] | bf16, zero-init |
| `recurrent_state` | [max_batch, max_history+1, 64, 128 (k), 128 (v)] | fp32, contiguous; index [slot, j, h, k, v]; j = 0 current, j ≥ 1 MTP history |

- `max_history = mtp`. A new sequence zeroes its slot (`GDN:98-101,228-231`).
- **History**: §4.3 covers how a verify pass writes it, §4.4 how rewind reads it.

### 8.4 Checkpoints (prefix reuse)

- Checkpoints and stashes copy `state[slot,:1]` (4,194,304 B) and `conv_state[slot,:,:4]` (196,608 B) per layer
  (`GDN:203-207,293-305`).
- Interval: 2048 tokens (`glm5_next.py:352`, `generator.py:238-246`). The coarser prefill interval is 32768, rounded up
  to max_chunk_size (`generator.py:247`, `job.py:1580-1593`).
- The in-pass checkpoint is described in §4.5.
- With MTP, page reuse keeps at least one real prompt token to re-run, so the carry hidden state can be recovered
  (`job.py:1145-1150,1285-1286`).

### 8.5 MTP draft cache

The draft has its own `Cache` (`GX/runner.py:50`) with one MLA layer (layer_idx 0) of the same shape, its own block
table and staging buffers (`generator.py:712-733`). It has no recurrent state.

---

## 9. LoRA (`GX/glm_xpu/lora.py`, applied at load before the tier and the graphs; `GX/runner.py:53-58`)

**Adapter** (`M/lora/adapter_config.json` and its header):
- PEFT LoRA, r = 64, `lora_alpha` 32, not rsLoRA, not DoRA.
- 992 BF16 tensors: `lora_A` [64, in], `lora_B` [out, 64].
- Several comma-separated adapters add up.

**Targets** (regex in the adapter config, matching the header):

| Layers | Modules |
|---|---|
| 0-2 | `mlp.{gate,up,down}_proj` |
| 3-44 | `mlp.shared_experts.{gate,up,down}_proj` |
| KDA | `self_attn.{q,k,v,o,b}_proj`, `forget_gate.f_a_proj`, `forget_gate.f_b_proj`, `g_a_proj`, `g_b_proj` |
| MLA | `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `kv_b_proj`, `o_proj` |

**Not targeted**: routed experts, the router, the indexer, mHC, and layer 45.

**Scale.** `s = alpha/r · lora_scale = 0.5 · lora_scale` (`lora.py:35,44,91`). `lora_scale` comes from `--lora-scale` or
`GLM_XPU_LORA_SCALE` (default 1, `XP/serve.py:700-701`).

**Name mapping.**
- Strip `base_model.model.`.
- `.forget_gate.` becomes `.`.
- `model.layers.` falls back to `model.language_model.layers.` (`lora.py:39-42,108-114`).

**Tensor prep** (`lora.py:47-56,115`).
1. A and B are converted to fp32, and B is multiplied by s.
2. `a = Aᵀ` is padded to (in, r) and `b = (sB)ᵀ` to (r, out).
3. Both are cast to **fp16**. The cast happens after scaling.

**Special cases.**
- **KDA q/k/v → fused `qkv_proj`.** A is stacked to (192, 4096), and B is block-diagonal (24576, 192) in q, k, v order.
  The effective rank is 192 (`lora.py:122-124,141-154`).
- **`kv_b_proj` is merged, not applied at runtime** (`lora.py:125-135`).
  - `d = (sB) @ A` in fp32, viewed as (64, 512, 512).
  - `w_uk_flat.view(512,64,256) += fp16(d[:, :256, :].permute(2,0,1))`, and the same for `w_uv` with
    `d[:, 256:, :]`.
  - The result is `fp16(W + fp16(Δ))`.
- **GatedMLP with any LoRA'd linear.** `bc = None` and `multi_gu = [None]` force the per-linear path
  (`lora.py:156-161`).

**Applying the delta at runtime** (`EX/modules/linear.py:615-629` with `_apply_lora_addmm`, `lora.py:59-80`; default
`GLM_XPU_LORA_ADDMM=1`):
- The input is the Linear's **raw input x**, before suh and the Hadamard.
- The delta is added after the base output and the trim, before any scales.
- Formula:
  ```
  t = fp16(x) @ a                                  # fp16 GEMM, fp16 out
  if y is fp16 and contiguous:  y.addmm_(t, b)     # one fp16 rounding of y + t·b
  else (fp32 y):                y += fp32( fp16(t @ b) )
  ```
- fp32-output targets: KDA qkv / o / b / f_a / f_b / g_a / g_b, MLA o_proj, MLP down.
- fp16-output targets: q_a, q_b, kv_a, MLP gate/up.
- The LoRA'd `f_a` / `g_a` outputs are rounded to fp16 (`.to(half)`) before `f_b` / `g_b`.
- The LoRA on `q_a_proj` also shifts the indexer query, because q_resid feeds `wq_b`.

**Graphs.** The LoRA GEMMs are captured inside the block graphs. `set_scale` (`lora.py:180-191`) rescales B in place
and adjusts the merged kv_b incrementally, which drifts in fp16.

---

## 10. Routed-expert store and tier

**Files.**
- `~/b70/llm/state/stores/glm/glm53_flash_exl3_experts_blob.{bin,json}`.
- The server reads `/nvx/glm53_flash_exl3_experts_blob.bin` (`llm/b70:1330`).
- Total 100,113,186,816 B.

**JSON header.**
- format `glm53-expert-store-blob-v2`; E = 288; n_layers 43; align 4096; `mtp_layer_li` 42.
- `mul1_markers_hex` `['2dd1dc83']` (0x83DCD12D little-endian).
- 12,384 sha256 hashes, one per stored record.

**Record (li, e)** sits at `layers[li].offset + e·record_bytes`, layer-major (`XP/pack_store.py:202-206`).

| li | Model layer | K (code) | record_bytes | offset |
|---|---|---|---|---|
| 0 | 3 | 3.5 (35) | 11,046,912 | 0 |
| 1..41 | 4..44 | 2.5 (25) | 7,901,184 | 3,181,510,656 + (li−1)·2,275,540,992 |
| 42 | 45 (MTP) | 4 | 12,619,776 | 96,478,691,328 |

The trunk mapping is li = model_layer − 3 (`GX/runner.py:33-35`, `GX/moe_patch.py:24-44`).

**On-disk order is NOT the JSON `fields` order.** `fields` describes the raw record: gate.trellis, up.trellis,
down.trellis, then g.suh, g.svh, u.suh, u.svh, d.suh, d.svh. The stored bytes are that record permuted by `blob_perm`
(`XP/pack_store.py:35-79,227-229`). Verified layout, matching the kernel `Geom` (`MOE:74-86`):

```
uint32 words. K2 = 2K (2.5→5, 3.5→7, 4→8); W = 4·K2 words per 16x16 tile; D = K2 / gcd(K2, 64); G = W / D
[0]      gate|up trellis  [H/16=256 k-rows][(2I)/16 = 256 n-tiles: 128 gate tiles then 128 up tiles][W]   planar4
[+GU]    down trellis     [I/16=128 k-rows][H/16=256 n-tiles][W]                                         planar4
then fp16: g.suh[4096] u.suh[4096] g.svh[2048] u.svh[2048] d.suh[2048] d.svh[4096]     (36,864 B; no padding)
GU = 256·256·W·4 B;  down = 128·256·W·4 B
```

- **planar4.** Within each k-row, the n-tiles are taken in groups of 4. Raw word (tile t ∈ 0..3, word g·D+i) moves to
  `[i][t][g]`, i.e. the order is `[n/4][D][4][G]`.
- K = 4 (D = 1) is the identity. K = 2.5 has D = 5, G = 4; K = 3.5 has D = 7, G = 4.
- The raw trellis words are the safetensors int16 data read as little-endian uint32.

**Tier.** Residency only, not math. The kernel reads experts through `TIER.ptrs[li]` (int64 [288] device addresses).

- VRAM slot pools are per K class (`GX/tier.py:126-145`).
- A pinned-USM RAM tier, read zero-copy over PCIe (`:226-257,324`).
- O_DIRECT NVMe reads (`:381-411`).
- Decode calls `ensure()` (`:620-686`). Prefill with more than 192 picks uses `stage_layer()` (`:1236-1312`).

Residency changes results only through the quality-mode bonus (§6.4b).

---

## 11. Implementation map (decode unless noted)

| Op | XPU implementation | Where |
|---|---|---|
| Embedding lookup | torch `nn.Embedding` on the CPU | `EX/modules/embedding.py:145-151` |
| ExpandStreams | torch | `EX/modules/hyperconnections.py:37-38` |
| mHC mix (R ≤ 64) | Triton `_hc_mix_partial_kernel` + `_hc_mix_final_kernel` | `GX/k_hc.py:66-155` via `GX/ops.py:211-216` |
| mHC mix (R > 64, prefill) | torch `hc_mix_torch` | `GX/ops.py:219-234` |
| mHC apply | Triton `_hc_apply_kernel` | `GX/k_hc.py:159-180` via `GX/ops.py:237-242` |
| RMSNorm (all weighted norms, fp16/fp32 out) | Triton `_rms_norm_kernel` | `GX/k_hc.py:184-211` via `GX/ops.py:253-258` |
| Fused residual + RMSNorm (MTP block) | torch add + Triton norm | `GX/ops.py:261-264` |
| HyperHead mean | torch `mean` | `hyperconnections.py:640-641` |
| EXL3 linear M ≤ 128 | ESIMD `exl3_gemm_small` (`exl3xpu/_C.so`) | `GX/ops.py:73-74`, `CS/exl3_ops.sycl:258-321`, `CS/exl3_esimd.h` |
| EXL3 linear M > 128 | `exl3_linear`: had_in + recon + oneDNN fp16 GEMM + had_out | `GX/ops.py:75-78`, `CS/exl3_ops.sycl:651-757` |
| fp16 Linear | `torch.matmul` / `hgemm` (oneDNN) | `EX/modules/quant/fp16.py:69-100`, `GX/ops.py:96-112` |
| LoRA delta | torch matmul / addmm (oneDNN) | `GX/lora.py:59-80` |
| KDA fp32→bf16 transpose, gates, beta | torch | `EX/modules/gated_delta_net.py:1126-1150` |
| KDA short conv | Triton `_causal_conv1d_update_slotted_kernel` (prefill: `_output_kernel` + `_state_kernel`) | `EX/modules/gated_delta_net_fn/conv1d.py:16-113,280-337` |
| KDA recurrent step | Triton `_kda_step_kernel` (torch fallback when MTP is on and no history) | `GX/k_kda.py:12-52`, `GX/ops.py:307-349` |
| KDA chunked prefill | vendored fla Triton (`chunk_kda`) | `EX/vendor/fla/__init__.py:70-120` and the kda_* / chunk_* files |
| KDA gated RMSNorm | torch | `GX/ops.py:285-302` |
| MLA absorb / unfold | Triton `_mla_absorb_kernel` / `_mla_unfold_kernel` | `EX/modules/attention_fn/mla_triton.py:555-623,1055-1086` |
| MLA latent / plane append | Triton `_mla_kv_update_kernel`, `_mla_plane_update_kernel` | `mla_triton.py:113-177,791-837` |
| Indexer k LayerNorm | torch `F.layer_norm` | `EX/modules/layernorm.py:97-106` |
| Indexer pool, scores, selection (decode) | torch | `GX/mla_graph.py:99-140` |
| Indexer scores (prefill sparse) | Triton `_dsa_indexer_kernel` / `_fewq_kernel` | `EX/modules/attention_fn/dsa_triton.py:603-730,1111-1182` |
| Indexer top-k (prefill) | torch sort (`dsa_topk`) | `GX/ops.py:354-367` |
| Sparse latent attention | torch gather + oneDNN bmm + softmax | `GX/sparse_mla.py:15-40` |
| Dense MLA prefill | oneDNN fp16 GEMM + `F.scaled_dot_product_attention` | `GX/sparse_mla.py:66-110` |
| Dense MLA small-seq (MTP draft) | Triton `mla_attn_triton_decode` | `mla_triton.py:205-396,909-1052` |
| Router (Q = 1 exact, prefill) | torch (`routing_ds3_nogroup`) | `GX/ops.py:177-193` |
| Router (Q > 1, or bonus) | torch GEMM + Triton `_route_kernel` | `GX/k_router.py:21-84`, `GX/block_graphs.py:244-261` |
| Routed experts | SYCL ESIMD `moe_forward` (`_moe_glm.so`); prefill `moe_prefill` DPAS | `CS/exl3_moe.n128.sycl:154-418,484-697,891-944,1363-1470` |
| Expert grouping (2 ≤ M < 256) | SYCL `exl3_moe_align` | `CS/exl3_moe_align.sycl:8-49` |
| Clamped SwiGLU (dense, shared) | torch fp32 | `GX/ops.py:147-154` |
| Sampler | torch | `GX/ops.py:380-415` |
| Decode orchestration, graphs | `torch.xpu.XPUGraph` per phase | `GX/block_graphs.py:454-686` |

---

## 12. Things that are surprising or easy to get wrong (summary)

1. **No RoPE at all.** MLA is NoPE (`qk_rope_head_dim 0`) and the indexer rope is a no-op. The config keys
   `indexer_rope_interleave` and `index_share_for_mtp_iteration` are ignored.
2. **mHC.**
   - The internal RMS uses `rms_norm_eps` (1e-5) over all 16384 stream values, unweighted. `hc_eps` (1e-6) is used
     only in `pre + eps` and in Sinkhorn.
   - Sinkhorn is softmax (rows) + eps, then a column pass, then 19 × (row, col).
   - The apply uses **columns** of comb: `R'[h] = post[h]·y + Σ_g C[g][h] R[g]`.
   - Decode reads an fp16 copy of `hc_*_fn`; prefill reads fp32.
   - The collapsed stream is fp16 before the norm.
3. **The final collapse is an unweighted mean** of the 4 streams. There are no hc_head tensors.
4. **RMSNorm kernel order:** `(x·w)·rsqrt(...)` in fp32, with the fp16 input clamped to ±65504.
5. **KDA.**
   - The gate is the lower-bound sigmoid form, with per-channel `dt_bias` [8192].
   - The output gate is a sigmoid after the weighted norm.
   - The small projections are fp16-rounded.
   - q/k are normalized in fp32 at decode and rounded to bf16 at prefill.
   - The 64-token threshold selects chunked vs recurrent.
   - Rewind is mandatory after every verify.
6. **MLA indexer.**
   - k_norm is a biased LayerNorm with eps 1e-6.
   - The pool gate comes from x; the per-dim softmax over 4 members adds an fp32 APE.
   - ReLU applies per head, then a weighted sum with scale 1/64. Scores are rounded to fp16 before the sort.
   - The current token may be left out.
   - The graph path always gathers. Its index order is score order, while eager is ascending.
7. **Router.**
   - fp16 logits.
   - Selection by sigmoid + bias; weights from the unbiased sigmoid, normalized and ×2.5.
   - The Q = 1 path uses an fp16 mean-centered bias and fp16 weights. The Q > 1 path uses an fp32 bias and fp32 weights.
   - Production adds a residency bonus.
8. **Clamped SwiGLU:** `min(silu(g),10)·clamp(u,±10)`.
9. **EXL3 accumulation depends on M:** fp16 tile partials when M ≤ 2, fp32 DPAS for 3..128, and an fp16-rounded GEMM
   output when M > 128. fp32-out linears are fp16-rounded in prefill. `x·suh` is rounded to fp16 before the FWHT.
10. **The routed-expert sum is rounded to fp16** before the fp32 shared-expert add.
11. **Store records are permuted** (planar4, suh/svh reordered). The JSON `fields` offsets are only valid for the raw
    order. Expert K is per layer: 3.5, then 2.5, then 4 (MTP).
12. **LoRA.**
    - kv_b is merged into the fp16 `w_uk` / `w_uv`.
    - q/k/v become a rank-192 block-diagonal LoRA on the fused qkv.
    - The fp32-output path rounds the LoRA product to fp16 first.
    - Routed experts carry no LoRA.
13. **Prefill stops after block 44's attention** unless MTP is on. The last prompt token always goes through the decode
    forward.
14. **MTP.**
    - The input is cat(enorm(emb fp16), hnorm(hidden)), in that order.
    - The draft block has no mHC.
    - The draft uses the target's post-final-norm fp16 state, paired with the previous position.
    - The draft feeds back its own shared_head.norm output.

---

## 13. Open questions

1. **Which numerics does the C++ engine target?** The same model math runs through different rounding on different
   paths:
   - graph decode vs eager prefill (fp16 vs fp32 `hc_fn`; MLA gather vs SDPA vs Triton decode);
   - Q = 1 vs Q > 1 router.

   Bit parity is only possible per path. Decide between bit-matching "decode graph, Q = 1, RES_BONUS = 0" and a
   tolerance-based check.
2. **The HF reference.** It was not available here. Three points to check against it:
   - the SwiGLU clamp convention (exllamav3: after silu);
   - whether `moe_router_dtype=float32` should make the router fp32;
   - whether the indexer applies any rope despite `qk_rope_head_dim 0` (`indexer_rope_interleave: true` hints that the
     reference may rotate part of the indexer key/query).
3. **`sparse_mla_latent` rounds QK to fp16 before scaling.** Can real activations overflow 65504? If C++ keeps fp32
   scores, it diverges by design.
4. **KDA conv.** Does Triton round the bf16 × bf16 tap product to bf16 before the fp32 accumulate? This needs a check
   (or the Triton IR) on the target.
5. **The SYCL MoE kernel uses `-ffast-math` exp.** It is not bit-reproducible against IEEE expf; the C++ port should
   reuse that kernel or accept ulp-level differences.
6. **`index_share_for_mtp_iteration` is ignored.** The MTP layer scores with its own indexer. The code says this is
   identical below 2048 tokens of context, but not above.
