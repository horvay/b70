# EXL3 conversion: algorithm and on-disk format (dense models)

A reimplementation spec for a native C++/SYCL/oneMKL EXL3 quantizer whose output must load in
exllamav3 (and TabbyAPI, which runs on it) exactly like a model made by turboderp's `convert.py`.

## 0. Sources, versions and conventions

- Source tree: `~/b70/llm/dsv41/exllamav3` (vcruz305 fork, `version.py` = 1.6.1.post1).
- The reference model `~/models/llm/gemma4-31b-based-exl3-4.0bpw` was made by **stock
  1.5.3**. The fork's history contains turboderp's commit `d3739fd3 "Bump to v1.5.3"` (2026-09-27), so
  stock 1.5.3 can be read exactly: `git -C <repo> show d3739fd3:<path>` (I extracted it to
  `/tmp/exl3_153` while writing this).
- **Line references are to the v1.5.3 tree unless marked `HEAD`.** Abbreviations:
  - `Q` = `exllamav3/modules/quant/exl3_lib/quantize.py`
  - `CM` = `exllamav3/conversion/convert_model.py`
  - `CAL` = `exllamav3/conversion/calibration_data.py`
  - `ALLOC` = `exllamav3/conversion/allocation.py`
  - `COMP` = `exllamav3/conversion/compile.py`, `QC` = `exllamav3/conversion/quant_config.py`
  - `LIN` = `exllamav3/modules/linear.py`, `EXL3` = `exllamav3/modules/quant/exl3.py`
  - `QTK` = `exllamav3_ext/quant/quantize_tiles_kernel.cuh`, `QTO` = `.../quantize_tiles_optimized.cuh`,
    `QCU` = `.../quantize.cu`, `CB` = `.../codebook.cuh`, `PACK` = `.../pack.cu`, `DQ` = `.../exl3_dq.cuh`,
    `FRAC` = `.../frac.cu`, `HAD` = `.../hadamard.cu`
  - `TL` = `tests/testlib/trellis.py` (HEAD only, an independent NumPy reference of the format; it agrees
    with everything below and is the best executable spec to port tests from).
- Matrix orientation: **everything in EXL3 is (in_features, out_features)**, i.e. the transpose of the
  HF `nn.Linear.weight` (out, in). "Rows" = input channels (k), "columns" = output channels (n).
- `f16(x)` = IEEE binary16 round-to-nearest-even. `Hk`/`Hn` = block-diagonal 128x128 Hadamard (section 3.2).

### 0.1 What was verified against the reference model (no GPU used)

With stdlib-only Python (`python3 -I`, header parse + seek reads, scripts in `/tmp/exl3_scripts/`):

- Every tensor name/dtype/shape (section 6.1).
- `mul1` marker = int32 scalar -2082680531 = 0x83DCD12D.
- For 64 tiles each of layer-0 `q_proj` and `down_proj`: every unpacked tile is a valid tail-biting ring
  under the bit layout of section 5; decoded mul1 values span [-3.453, 3.348], the exact min/max of the
  mul1 codebook.
- Full decode of a 128x128 block of layer 59 `q_proj` and `down_proj` (unpack -> codebook -> tile
  permutation -> `diag(suh) H W_hat H diag(svh)`) correlates **0.995** with the same block of the sibling
  bf16 Gemma 4 31B fine-tune on disk (`Wicked-Stuff-31B-Final`, HF weight transposed), and **0.00** with
  the untransposed weight. So the decode path in section 5 is right end to end: bit order, tile
  permutation, codebook, Hadamard order, suh on input rows, svh on output columns.

---

## 1. Overall conversion flow (`CM` `main`, 1080-1588)

### 1.1 Setup

1. Arguments and defaults (`CM` 69-101, 256-279): `bits` (required), `head_bits` = **6** by default
   (the reference used 16), `mtp_bits` = 4, `vision_bits` = 0 (auto, see 1.7), `cal_rows` = 250,
   `cal_cols` = 2048, `codebook` = `mul1`, `shard_size` = 8192 MB, `out_scales` = **"always"** in 1.5.3
   (`CM` 99; mapped at `CM` 299 to `apply_out_scales = True/False/None`), `hq` = False.
2. Model built from `config.json` with exllamav3's architecture class (`get_base_model`, `CM` 343).
   `model.modules` is a flat list: `[Embedding, Block 0 .. Block N-1, final RMSNorm, lm_head Linear]`
   for Gemma 4 (`architecture/gemma4.py` 209-421).
3. Bit strategy (`ALLOC`, section 2).
4. Calibration token rows (section 1.2).

### 1.2 Calibration data (`CAL` 61-99)

The default set is a weighted mix of the bundled files in `exllamav3/conversion/standard_cal_data/`
(`CAL` 73-81). With `rows = 250`, `cols = 2048`, `target_rows = max(1, int(weight / 135 * rows))`
(`CAL` 87), and the random source takes whatever remains (`CAL` 94):

| source | weight | processor | rows @250 |
|---|---|---|---|
| `c4.utf8` | 20 | `shuffle_lines` | 37 |
| `code.utf8` | 20 | `split_raw` | 37 |
| `multilingual.utf8` | 10 | `shuffle_lines` | 18 |
| `technical.utf8` | 10 | `split_raw` | 18 |
| `wiki.utf8` | 50 | `split_wiki` | 92 |
| `tiny.utf8` | 5 | `split_tiny` | 9 |
| random tokens | 20 | `random_data` | 39 (= 250 - 211) |

Rows are concatenated in that order; each row is a `(1, 2048)` int64 token tensor.

- `split_art(articles, rows, cols)` (`CAL` 5-19): concatenate tokenized articles into the current row; truncate
  to `cols`; when it reaches exactly `cols`, emit it and start an empty row (the overflow of the last article is
  **discarded**, the next row starts with the next article). **Special tokens**: when the number of rows emitted
  so far is even, every article appended to the current row is encoded with `add_bos = add_eos = True`; for odd
  rows, without. The parity is local to each processor call.
- `split_wiki`: split on `"</doc>\n"`, drop each article's first line (the title line), keep articles longer than
  50 characters, then `split_art`.
- `split_tiny`: split on `"<|endoftext|>"`, strip, `split_art`.
- `shuffle_lines`: split on `"\n"`, drop whitespace-only lines, `random.seed(0); random.shuffle(lines)` (Python's
  Mersenne Twister), `split_art`.
- `split_raw`: tokenize the whole file once (no BOS/EOS); row i = tokens `[i*cols, (i+1)*cols)`.
- `random_data`: `torch.manual_seed(0)`; per row `torch.randint(0, actual_vocab_size, (1, cols))` (torch CPU
  Mersenne Twister).
- Tokenizer: `Tokenizer.encode(text, add_bos, add_eos)` with `encode_special_tokens = False`: special-token text in
  the corpus is **not** parsed as special tokens (split-special mode); BOS/EOS are inserted as ids by `encode`
  itself, never by the HF post-processor (`tokenizer/tokenizer.py` 262-268, 340-389). For Gemma 4: BOS = 2, EOS = 1.
- Alternative `--cal_data FILE` (`CAL` 102-130): safetensors with `input_ids (rows, cols)` and optional `lengths`.

Bit-exact reproduction of the token rows needs the HF `tokenizers` BPE result, Python's `random.shuffle`
and torch's CPU `randint` stream. **This affects only quantization quality, never format compatibility**.
The simplest faithful option for the native tool is a one-time export of the 250 rows to a file (exllamav3
supports `--cal_data`) and to read that file.

### 1.3 The calibration state

- `state` = list of 250 per-row tensors. Initially the token ids. After the Embedding module each row is
  `(1, 2048, hidden)` **fp32** (Embedding `out_dtype = torch.float`, `modules/embedding.py` 19; Gemma 4 multiplies
  by `float(bf16(sqrt(hidden)))`, `gemma4.py` 211-217). The state lives in **CPU RAM**: for Gemma 4 31B that is
  250 x 2048 x 5376 x 4 B = 11.0 GB.
- `original_input_ids` (the token rows) are passed as `params["input_ids"]` to every forward (`CM` 1155-1159).
- Every forward uses `attn_mode = "flash_attn_nc"`: one causal sequence per row, no cache.
- Rows whose output is non-finite are excluded from everything afterwards; the job aborts if more than 10% of rows
  break (`CM` 740-749).

### 1.4 Layer-by-layer loop (`CM` 1209-1503)

For each `idx, module in enumerate(model.modules)`:

1. **Load the unquantized module** (`CM` 1235-1258). Linear weights are read from the checkpoint and **converted to
   fp16** (`LIN` 188-243: `get_tensor(..., float2half = True)`; bf16 is always converted to fp16 unless `allow_bf16`,
   `loader/safetensors.py` 774-853), **transposed** to (in, out) (`transposed_load = True`) and **zero-padded** to
   multiples of 128 in both dims (`Linear.pad_to = 128`, `LIN` 71-74, 119-135). Norms, embeddings and scalars keep
   their checkpoint dtype.
2. **Capture pass** (`CM` 1264-1329), only if the module has quant targets (`get_qmaps()` non-empty): every calibration
   row is forwarded through the **original, unquantized module**. Linear layers with a `qmap` accumulate the Hessian
   of their input (section 1.5). The first 5 rows (`num_ref_states = 5`, `CM` 103) also keep their unquantized output as
   reference states for the error printout. The Hessians are then swapped to CPU (`CM` 1346-1349).
3. **Quantize every Linear of the module** (`CM` 1352-1374), each with `quant_args.seed = idx` (the module index,
   `CM` 456-471). With one device the "parallel" path runs one worker thread that walks the groups of
   `group_quant_linears` (`CM` 495-538, section 3.10). Tensors with strategy 16 are left as fp16.
4. Collect the module's output tensors (`m.get_tensors()` for every submodule), save them to
   `work_dir/qtensors/<module.key>.safetensors` (`CM` 1376-1388).
5. **Reload the module from the quantized tensors** (`CM` 1396-1401) and **advance the state through the quantized
   module** (`CM` 1409-1471): `state[i] = module.forward(state[i])` (output to CPU). The last module (lm_head) is
   not forwarded, except for the 5 reference rows (`CM` 1450). So layer `i+1` is calibrated on hidden states produced
   by the already quantized layers `0..i`, but within a layer every Linear's Hessian comes from the unquantized
   layer (o_proj sees attention computed with the original q/k/v; down_proj sees the original gate/up).
6. Reported per module: `rfn = ||x - ref||_F / ||ref||_F`, cosine error, SQNR over the 5 reference rows. These are
   diagnostics only.
7. Checkpointing to `work_dir/ckpt` (state, job.json); not needed for a reimplementation.

### 1.5 Hessian capture (`LIN` 579-612)

- One Hessian per **qmap** (shared-input group) per module: `H_data = {"H": zeros(k, k, fp32), "count": 0, ...}`
  (`init_H_data`, `LIN` 579-589). Gemma 4 qmaps (`modules/attn.py` 253-321, `modules/mlp.py` 542-580):
  - `block.attn.input`: q_proj, k_proj, v_proj (input = `input_layernorm` output)
  - `block.attn.o`: o_proj (input = attention output)
  - `block.mlp.input`: gate_proj, up_proj (input = `pre_feedforward_layernorm` output)
  - `block.mlp.down`: down_proj (input = `gelu_tanh(gate) * up`)
  - `block`: lm_head (input = final norm output), used only if `head_bits <= 8`
- Only the **first Linear that sees a qmap** accumulates (`first_key`, `LIN` 598); the others share it.
- Per forward call: `x` = the Linear's input as the module computed it, **fp16** for all Gemma 4 linears (norm
  outputs are `out_dtype = half`, `modules/transformer.py` 156, 184; the MLP intermediate is `interm_dtype = half`,
  `gemma4.py` 299-311), zero-padded to `in_features` (`LIN` 633-637), reshaped to `(tokens, k)` and promoted to **fp32**.
  Rows containing inf/NaN are dropped (`LIN` 600-609). Then `H += x^T x` (fp32 `addmm`) and `count += tokens`.
- Normalization happens at finalize: `H /= count` (`Q` 1014), i.e. **H is the mean of x x^T over all calibration
  tokens** (512,000 for 250x2048). Not 2/N, no extra scaling.
- The capture pass of block `i` runs with the state produced by quantized blocks `0..i-1` (1.4).

### 1.6 Non-quantized modules

- **Embedding** (`model.language_model.embed_tokens.weight`): stored unchanged (bf16 in the reference). Stock 1.5.3
  has no embedding quantization (`--embed_bits` appeared on 2026-09-29, after 1.5.3).
- **Norms**, `layer_scalar` and other small tensors: copied unchanged (bf16).
- **lm_head**: `Linear("lm_head", qbits_key = "head_bits", alt_key = "<prefix>.embed_tokens" if tied)` (`gemma4.py`
  407-417). With `head_bits = 16` it is stored as **`lm_head.weight`, fp16, shape (vocab, hidden)** (HF orientation;
  `LinearFP16.get_tensors` stores `weight.T`, `modules/quant/fp16.py` 62-67), **even for tied embeddings**: it is
  cloned from `embed_tokens` and converted bf16 -> fp16. With `head_bits <= 8` it is quantized like any other
  Linear, with the Hessian of the final-norm output.
- **Vision tower**: see 1.7.

### 1.7 Side models (MTP, vision) (`CM` 1506-1582)

After the main loop, MTP and vision modules are quantized **uncalibrated** (`capture_H = None, state = None`), so
`H_data["H"]` is a meta tensor and the tensor goes through the fallback path (3.8): same regularization and scale
search, no LDLQ, no refit.

- `vision_bits = 0` (auto) takes `vision_model.caps["default_vision_bits"]`, which is **6 for Gemma 4** in 1.5.3
  (`gemma4.py` 498-501; introduced 2026-08-26, before 1.5.3). `16` keeps the tower unquantized and copies its tensors
  through `get_additional_compiled_tensors` (all `model.vision_tower*` and `model.embed_vision*` tensors under their
  original names, `gemma4.py` 485-489).
- **The reference model has an unquantized vision tower** (bf16, original names such as
  `model.vision_tower.encoder.layers.N.self_attn.q_proj.linear.weight`, and no `vision_bits` in its config), so it was
  converted with `-vb 16`, not with the 1.5.3 default.
- When quantized, vision linears are named `...self_attn.q_proj.linear.{suh,svh,trellis,mul1}` (keys `q_proj.linear`,
  `gemma4.py` 527-583), plus `model.embed_vision.embedding_projection.*` (`gemma4.py` 611-617). Vision dims that are not
  multiples of 128 are padded (e.g. intermediate 4304 -> 4352).

### 1.8 out_scales (stock 1.5.3)

`--out_scales always` (the 1.5.3 default and the reference's setting): every tensor uses output channel scales
(section 3.4, step 2). `never`: never. `auto` in 1.5.3: on iff the input-state skew `sum of the largest 2% of
sqrt(diag(H))` / `sum of all` is < 0.15 (`Q` 1325-1341). HEAD changed the default and the meaning of `auto` (section 7).

---

## 2. Bit allocation (`ALLOC` 51-206)

There is **no measurement or optimization pass in `convert.py`**. The per-tensor rate comes from a greedy budget fill.
A measured, optimized allocation exists as a separate pipeline (`sc_measure.py` -> `sc_optimize.py` -> a YAML recipe
passed with `-rcp`, `ALLOC` 209-277); ignore it for a first version.

1. Walk all modules (main model + MTP; vision gets the fixed `vision_bits` outside the budget). For every Linear with a
   qmap:
   - `qbits_key == "bits"` (all decoder linears): `base = rate_floor(bits, half_steps)` where
     `rate_floor(b) = floor(2b)/2` if half steps are allowed (mul1 codebook) **and b < 4**, else `floor(b)`, clamped to
     [1, 8] (`ALLOC` 18-21). Accumulate `sum_numel += k*n` (padded dims) and `sum_bits += k*n*base`.
   - `head_bits`: fixed `head_bits`; `mtp_bits`: fixed `mtp_bits`; outside the budget.
2. `max_bits = int(bits * sum_numel)` (`ALLOC` 154).
3. Group the targets by `qgroup` (Gemma 4: `<layer>.self_attn.qkv`, `.o`, `<layer>.mlp.gu`, `.d`). Order the groups by
   `(-max priority, distance to the nearer end of the layer stack, layer index, insertion order)` (`ALLOC` 158-173).
4. Repeat until nothing changes: walk the groups in that order; if raising every tensor of the group to the next rate
   (`+0.5` below 4 bpw with half steps, else `+1`, max 8) fits in the remaining budget, do it (`ALLOC` 175-185).
5. `--hq`: clamp each tensor to at least `rate_floor(base + select_hq_bits)` (Gemma 4 dense: `select_hq_bits = 0`).
6. Reported `bits` = `final_bits / sum_numel` rounded to 2 decimals (`CM` 1194).

**`-b 4.0` gives exactly K = 4 for every decoder linear** (`base = 4`, `sum_bits == max_bits`, no bumps). The head is
16 (unquantized) in the reference. Fractional targets such as 4.25 give a mix of K = 4 and K = 5 per qgroup, with the
first and last layers bumped first.

**Fractional K** (`frac_k`, `Q` 81-90; `FRAC`; `quantize_tiles_frac_kernel.cuh`): only K = 1.5, 2.5, 3.5, only with
the mul1 codebook, only reached by targets below 4 bpw or explicit head/MTP/recipe rates. It is never used at 4.0 bpw.
Section 5.4 has the format, for completeness.

---

## 3. Per-tensor quantization (stock 1.5.3)

Input per Linear: `W` = (k, n) fp32 = `float(f16(W_hf^T))`, zero-padded; its qmap's `H_data`; `K`; codebook (mul1);
`seed = module idx`; `apply_out_scales` (True for "always"). Entry points: `quantize_exl3` (`Q` 1471-1692) and the
batched `quantize_exl3_batch` (`Q` 1762-1966), which the converter actually uses for multi-tensor groups. Both are
mathematically the same apart from RNG consumption (3.1).

### 3.1 Random sign vectors su0 / sv0

- `torch.manual_seed(seed)` with `seed = module index` (`Q` 1525; batch: `Q` 1811, 1836).
- Input signs, drawn once per H_data at finalize (`Q` 1042): `su0 = sign(sign(randn(k, device = cuda)) + 1e-5)`, i.e.
  +1/-1 with 0 mapped to +1, shape (k, 1). All tensors sharing a qmap share su0 (q/k/v; gate/up).
- Output signs (`Q` 1543, batch `Q` 1845): `sv0 = sign(sign(randn(n)) + 1e-5)`, shape (1, n).
- Stream subtleties (observed in the reference): the serial path draws `randn(k)` for su0 (if this tensor finalizes the
  H) and then `randn(n)` from the **same** stream; the batch path re-seeds before the sv0 draw, so every tensor of a
  batch with equal n gets identical sv0 (k_proj and v_proj svh signs agree 100% in the reference), and su0 and sv0 of
  the first tensor start with the same Philox prefix. q/k/v (16,384 output columns <= 32,768) run batched; gate and up
  (21,504 each) run serially, so up_proj, which finalizes the shared H, gets a different sv0 than gate_proj (50% sign
  agreement in the reference).
- Exact replication needs torch's CUDA Philox `randn` and isn't worth it: **any ±1 vectors give a valid, equally good
  model**. Use a documented counter-based RNG seeded by `(module idx, tensor key)`.

### 3.2 Hadamard transforms

- `had_k = had_n = 128` (`Q` 15). `H128 = Sylvester(128) / sqrt(128)` in **natural (Sylvester) order**:
  `H128[i][j] = (-1)^popcount(i & j) / sqrt(128)`. Built by `get_hadamard(128)` = repeated
  `sylvester(h) = [[h, h], [h, -h]]` from `hadamard_1.txt = "+"` (`util/hadamard.py` 97-118). It is symmetric and its
  own inverse.
- Block-diagonal application (`Q` 401-447), all in fp32:
  - `had_l(X)`: for each 128-row block, `X[b] = H128 @ X[b]` (input side, rows).
  - `had_r(X)`: for each 128-column block, `X[:, b] = X[:, b] @ H128` (output side, columns).
- Hence k and n must be multiples of 128 (guaranteed by padding). Inference applies the same `H128` with the
  constant `0.088388347648 = 1/sqrt(128)` (`HAD` 144).

### 3.3 Hessian finalize (`finalize_capture_H`, `Q` 982-1074), once per H_data

```
H = H_sum / count                                   # fp32
dm = mean(diag(H))
fallback = (count == 0) or (dm < 1e-20) or not finite(dm)     # -> 3.8
H += sigma_reg * dm * I        with sigma_reg = 0.025         # Q 1030 (also applied in the fallback case)
diag_H = diag(H)                                    # kept for the 1.5.3 "auto" out-scales rule only
su0 = random signs (k)                              # Q 1042
Hr = Hk · diag(su0) · H · diag(su0) · Hk            # Q 1046-1049: H*=su.T; had_r; H*=su; had_l
L  = block_ldl(Hr, 16); L[i, i] = 0 for all i       # Q 1059-1063
keep Hr (moved to CPU) for metrics and refit
```

`block_ldl(H, b = 16)` (`Q` 472-546):

```
C = cholesky(H)                     # lower, H = C C^T, fp32 (cuSOLVER/torch). On failure: H += 2*0.025*mean(diag(H))*I, retry up to 10x
for each diagonal block i (16x16): D_i = C[i-block, i-block];  Dinv_i = inv(D_i)
for each block column i: C[:, i-block] = C[:, i-block] @ Dinv_i
set every diagonal 16x16 block of C to I
return C                            # unit block-lower-triangular L with H = L · blockdiag(D_i D_i^T) · L^T
```

After `L[i, i] = 0` the diagonal blocks are all zero (they were identities), so **L is strictly block-lower-triangular**.

Note: L is built from the **sign- and Hadamard-rotated** H only. It does not include the per-row input scales that
`regularize` folds into the weight (3.4). Stock 1.5.3 behaves this way, and that is what `--scaled_hessian` changes
later in HEAD (section 7).

### 3.4 Regularization (`regularize`, `Q` 1292-1399)

```
W0: (k, n) fp32 original weight;  su0 (k,1), sv0 (1,n) signs
codebook_scale = 1.24371088                          # Q 16 (the RMS of the 3INST codebook, used for every codebook)

# 1. output channel scales
r[j]  = sqrt(mean_i W0[i, j]^2)                      # block_rms over dim 0 (fp32, summed in 32-row blocks)
mu    = mean_j r[j]
if mu > 1e-30:  r /= mu   (zeros = False)  else: zeros = True
zero  = |r| < 1e-30
if apply_out_scales:                                 # always True for "always"
    r[zero] = 0.1
    sv = sv0 * r + 1e-10                             # (the 1e-10 is literally added to the signed value)
else:
    sv = sv0
W1 = W0 / sv                                         # per column
sv[zero] = 0.0                                       # all-zero output channels decode to exactly 0
W2 = had_r(W1)                                       # output-side Hadamard

# 2. input channel scales
q[i]  = sqrt(mean_j W2[i, j]^2);  q[q < 1e-30] = 0.1
su    = su0 * q / (-codebook_scale) + 1e-10          # NOTE the minus sign: stored suh = -sign * scale
W3    = W2 / su                                      # per row  -> every row has RMS 1.24371088
W4    = had_l(W3)                                    # input-side Hadamard

# 3. global scale (3.5)
g     = scale_search(W4)
W5    = W4 * g
su    = su / g
return W5 (the "regularized" weight that gets quantized), su, sv
```

Identity: `W0 = diag(su) · Hk · W5 · Hn · diag(sv)` (Hadamard self-inverse). The reference confirms the sign
convention: `suh` signs are exactly the negatives of the `svh` signs in their shared prefix (3.1), and `suh` RMS is
about 0.018 while `svh` RMS is about 1.0.

### 3.5 Global scale search (`sample_scale_tiles`, `g_scale_search_batch`, `g_scale_gss`; `Q` 1109-1238)

Tile sampling on W4 (`Q` 1115-1142), `width = 3`, `tk = k/16`, `tn = n/16`, `w4[a, :, b, :]` = 16x16 tile (a, b):

```
D = max(tk, tn)
diag tiles:  for d in 0..D-1, w in 0..2:  (a, b) = (d % tk, (d + w) % tn)          # 3*D tiles, order d-major
extreme:     ms[t] = mean of squares of tile t (flat t = a*tn + b)
             m = min(max(8, (3*D) // 16), (tk*tn + 1) // 2)
             hi = topk(ms, m, largest) indices;  lo = topk(ms, m, smallest) indices    # appended hi then lo
tiles = [row-major 256-vector of each sampled tile] then permuted by tensor_core_perm (5.2)
tiles *= LDLQ_DRIFT[K]   with {1: 1.08, 1.5: 1.035, 2: 1.018, 2.5: 1.009, 3: 1.004}, else 1.0   # K = 4 -> 1.0
```

Search (`Q` 1145-1208):

```
mse(S, s) = mean( (Viterbi(S * s) / s - S)^2 )       # Viterbi = the full trellis quantizer of section 4
coarse  = [0.1 + 0.2*i for i in 0..9]                 # 0.1 .. 1.9
c       = argmin_s mse(tiles[::3], s)                 # every third sampled tile
fine    = [c + 0.075*(i - 2) for i in 0..4]
y       = [mse(tiles, s) for s in fine];  b = argmin y
if 0 < b < 4:
    den = y[b-1] - 2*y[b] + y[b+1]
    off = clamp(0.5*(y[b-1] - y[b+1]) / den, -0.5, 0.5) if den > 0 else 0
else: off = 0
g = max(fine[b] + off*0.075, 0.01)
```

The search scores plain tile MSE (no Hessian). Because W4 rows have RMS 1.2437 and the mul1 codebook has RMS 1.0003,
`g` usually lands below 1 (decoded RMS in the reference tiles is about 0.89).

### 3.6 LDLQ (`ldlq`, `Q` 549-666)

Quantize W5 (k, n) from the **last 16-row block to the first**, feeding back errors of the rows already done through L:

```
For block rows [a, a+16) with a = k-16, k-32, ..., 0:
    X[a+t, :] = W5[a+t, :] + Σ_{r >= a+16} L[r, a+t] · E[r, :]      for t = 0..15
    (E = W5 - Q for rows already quantized; the diagonal block of L is zero)
    split X[a:a+16, :] into tn tiles of 16x16 (tile b = columns 16b..16b+15), each as a row-major 256-vector,
    permute each vector with tensor_core_perm (5.2), run the Viterbi quantizer (section 4) ->
        indices[256] (in permuted order, KEPT that way) and decoded values (fp16 -> fp32)
    un-permute the decoded values (inverse perm), write Q[a:a+16, :]
    store indices into encoded[a/16, b, 0..255]
```

The implementation batches the sum (`Q` 592-662) with `buf_size_k = 128` (requires k % 128 == 0): within a 128-row span
the feedback from rows of the same span is added per 16-row block (`bb_L = L[i+bj : j, i+bi : i+bj]`,
`compensation = prod_cache[rows] + bb_L^T @ (W - Q)[i+bj : j]`); after each span, `prod_cache += L[i:j, :]^T @ (W - Q)[i:j]`
accumulates its errors for every other row. The result equals the formula above. All GEMMs are fp32 (torch default, no
TF32; `check_system` warns if `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE` is set, `CM` 109-117). Columns are independent given L,
which is why tensors sharing an H can be concatenated along n and run through one LDLQ (3.10).

Memory for the largest Gemma 4 31B tensor (down_proj, k = 21504): H and L are 1.85 GB fp32 each, `prod_cache` is
k x n fp32 = 462 MB.

### 3.7 Back-transform, metrics and scale refit

1. Proxy error (printed only): `tr(E^T Hr E) / tr(W5^T Hr W5)` with `E = W5 - Q` (`Q` 1601-1637).
2. Back to the original basis (`Q` 1641-1645): `Q0 = diag(su) · Hk · Q · Hn · diag(sv)` (`had_l`, `*su`, `had_r`, `*sv`).
3. **Refit the stored scales with the trellis fixed** (`refit_scales`, `Q` 1414-1468, called at `Q` 1648-1655 unless
   fallback). `Hd = unrotate_H(Hr, su0) = diag(su0) Hk Hr Hk diag(su0)` = the damped H in the original basis
   (`Q` 1402-1411). `W` = original fp32 weight. Two rounds of:

```
HW = Hd @ W                                         # once
# output side, closed form per column
c[j] = (Q[:, j] · HW[:, j]) / (Q[:, j] · (Hd @ Q)[:, j])     (1 if the denominator <= 1e-30)
Q   *= c (per column);  sv *= c
# input side, one k x k solve
A    = (Q @ Q^T) ∘ Hd                               # elementwise product
A   += 1e-6 * mean(diag(A)) * I
b[i] = Σ_j Q[i, j] * HW[i, j]
r    = solve(A, b);  r[i] = 1 where r is non-finite or <= 0
Q   *= r (per row);  su *= r
```

   This minimizes `tr((W - Q)^T Hd (W - Q))` over per-column and per-row scales. The fit uses the **damped** Hd
   (0.025 included). HEAD only removes the damping from the *reported* error.
4. **Stored**: `suh = f16(su.flatten())` (k), `svh = f16(sv.flatten())` (n), `trellis = pack(encoded)`, plus the codebook
   marker (`Q` 1663-1688).

### 3.8 Fallback (uncalibrated) path (`fallback_quant`, `Q` 757-872)

Used when there is no Hessian (side models, `count == 0`, non-finite or tiny H): same `regularize` (with
`apply_out_scales` as forced), same scale search, then every 16-row block is quantized with no error feedback, and
there is no refit. Output format is identical.

### 3.9 Codebook marker

For mul1, store `<key>.mul1` = int32 scalar (shape `[]`) with value `0x83DCD12D` reinterpreted as int32
(`-2082680531`). For mcg: `<key>.mcg` = `0xCBAC1FED` as int32 (`-877912083`). 3INST: no marker. **Only the presence of
the marker is read** by inference (`LIN` 434-436, `EXL3` 82-85).

### 3.10 Grouping (`group_quant_linears`, `CM` 495-538; `quantize_exl3_batch`, `Q` 1762-1966)

- Tensors with the same `(qmap, K, in_features)` and a captured H are concatenated along n in chunks of at most
  `cap = max(n_first, min(32768, int(3.5e8) // k))` columns. Gemma 4: q+k+v (16,384 or 18,432 cols) form one group;
  gate (21,504) and up (21,504) don't fit together and each runs alone; o_proj and down_proj run alone.
- Leftover single tensors with an individual H and the same shape stack (up to 16, `ldlq_batched`, `Q` 889-978): the
  MoE expert down projections.
- Per tensor inside a group: regularize (scale search deferred), one batched scale search, shared-H LDLQ on the
  concatenation (exact), then per-tensor metrics, back-transform, refit and packing. The only observable difference
  from the serial path is RNG consumption (3.1). **A reimplementation can simply quantize each tensor on its own.**

---

## 4. The trellis quantizer (Viterbi, `QTK`)

### 4.1 Codebooks (`CB` 57-98; `TL` decode())

The trellis state is a 16-bit integer `s`. The decoded weight is fp16:

```
3INST (cb 0): x = (s * 89226354 + 64248484) mod 2^32
MCG   (cb 1): x = (s * 0xCBAC1FED) mod 2^32
   then for both: x = (x & 0x8FFF8FFF) ^ 0x3B603B60        # lop3 immLut 0x6A = (a & b) ^ c
                  value = f16( half_bits(x & 0xFFFF) + half_bits(x >> 16) )   # fp16 add, one rounding
MUL1  (cb 2): x = (s * 0x83DCD12D) mod 2^32
              b = byte0 + byte1 + byte2 + byte3 of x          # 0..1020 (dp4a with 0x01010101)
              h = 1024 + b                                     # fp16 bits 0x6400 + b, exact
              value = f16( h * K_INV + K_BIAS )                # __hfma: one rounding of the exact h*K_INV + K_BIAS
              K_INV  = half 0x1EEE = 0.00676727294921875
              K_BIAS = half 0xC931 = -10.3828125
```

To get bit-exact fp16 results on a CPU, compute in float64 (exact for these operands) and round once to fp16.
Codebook statistics over all 65,536 states, as test vectors:

| cb | mean | RMS | min | max | v[0] | v[1] | v[2] | v[3] | v[0x1234] |
|---|---|---|---|---|---|---|---|---|---|
| 3inst | 0.00021 | 1.24371 | -3.9570 | 3.9727 | 0.76806640625 | -0.91943359375 | 0.931640625 | 0.29443359375 | -2.19921875 |
| mcg | -0.00004 | 1.24411 | -3.9492 | 3.9492 | 1.84375 | 0.134521484375 | -0.75927734375 | 0.83984375 | 1.5146484375 |
| mul1 | -0.00193 | 1.00031 | -3.453125 | 3.3477 | -3.453125 | 0.64111328125 | -0.448486328125 | 0.1944580078125 | -0.5771484375 |

Precompute a 65,536-entry fp16 table per codebook (128 KB). That is what the CUDA K = 6 specialization does (`QCU`
69-92).

### 4.2 Trellis structure

- One "tile" = 256 weights in tensor-core order (5.2), treated as a **ring** of 256 positions (tail-biting).
- Each position consumes K new bits. The 16-bit state at position i is a shift register:
  `s_i = ((s_{i-1} << K) | new_bits_i) & 0xFFFF`. The low K bits of `s_i` are new; its top 16-K bits equal the low
  16-K bits of `s_{i-1}`.
- Tail-biting: `s_0 >> K == s_255 & (2^(16-K) - 1)`, so all 256 states come from a circular bitstream of 256*K bits.
- DP nodes ("edges", `QTK` 64): `E = 2^(16-K)` nodes; node `e` at position i = the low 16-K bits of `s_i`. A node plus a
  branch `k ∈ [0, 2^K)` gives the full state `s = (k << (16-K)) | e`, and its predecessor node is `pred = s >> K`.
- Cost = sum of squared errors `(decode(s_i) - w_i)^2` along the ring, in fp16.

### 4.3 Pseudocode (bit-faithful to `QTK` 59-341)

```
def quantize_tile(w[256] float32, K, cb):
    wh[i] = f16(w[i])                                   # QTK 92 __float2half_rn
    Kr = 16 - K;  E = 1 << Kr;  NB = 1 << K
    V  = codebook table (fp16) for cb

    def forward(roll, pre):                             # pre = -1: unconstrained start
        for t in 0..255:
            p = (t + roll) mod 256                      # ring position
            for e in 0..E-1:
                for k in 0..NB-1:
                    s = (k << Kr) | e;  pred = s >> K
                    d = f16(V[s] - wh[p])               # __hsub
                    if t == 0:
                        c = f16(d * d)                  # __hmul, no incoming cost
                        if pre >= 0 and pred != pre: c = +inf
                    else:
                        c = fma16(d, d, cost_prev[pred])   # __hfma: f16(d*d + cost_prev) with ONE rounding
                    if k == 0 or c < best:              # strict <: lowest k wins ties
                        best = c; arg = pred
                cost[e] = best
                hist[p][e] = arg                        # pass 1 only needs p < 128 (QTK 196-201)
            cost_prev = cost

    def argmin_end():                                   # QTK 277-304, deterministic tie order
        key(e) = (half_bits(cost_prev[e]) << 16) | rank(e)
        rank(e) = (brev5((e & 1023) >> 5) << 10) | (brev5(e & 31) << 5) | (e >> 10)   # brev5 = reverse 5 bits
        best = min over e of key(e)
        if best >= 0x7C000000: return 0                 # every cost inf/NaN
        return the e whose rank equals best & 0xFFFF    # i.e. lowest cost; ties -> lowest rank(e)

    def backward(roll, write, edge):
        for t in 255 down to 0:
            p = (t + roll) mod 256
            pe = hist[p][edge]
            s  = (pe << K) | edge                        # the full 16-bit state at position p
            edge = pe
            if write: idx[p] = s; out[p] = float(V[s])
            elif p == 0: break                           # pass 1: stop after position 0
        return edge                                     # pass 1: node at position 255 (ring predecessor of 0)

    forward(128, -1)                                    # pass 1: start mid-ring, unconstrained
    e_end = backward(128, False, argmin_end())          # node of position 255 on the best pass-1 path
    forward(0, e_end)                                   # pass 2: force s_0 >> K == e_end
    backward(0, True, e_end)                            # trace from node e_end at position 255: closes the ring
    return idx[256] (uint16 states, in ring = permuted order), out[256]
```

Notes:

- All costs are **fp16**, accumulated with fused fp16 FMA. Costs are non-negative, so comparing half bit patterns as
  unsigned integers orders them correctly. `+inf` = 0x7C00. On the B70, use `sycl::half` with `sycl::fma`. Emulating
  the fp16 math in fp32 or with separate mul and add rounding changes the chosen paths slightly. The output stays
  valid, only bit-identity with CUDA is lost.
- Pass 2 may report `+inf` for the closing node only if it is unreachable. It is always reachable, because the last
  ceil((16-K)/K) branches set the low bits freely.
- `hist` holds, per tile, 256 x E predecessors. The CUDA "optimized" kernels (`QTO`) store just the K-bit branch
  `k = pred >> (16 - 2K)` (one byte, or one bit for K = 1) and rebuild `pred = (k << (16-2K)) | (e >> K)`. Their minima
  use `min((cost_bits << 16) | pred)`, which is the same rule (lowest k = lowest pred wins ties), so optimized and
  reference kernels produce the same encodings (`QTO` 80-100, 330-335; `QCU` 94-104 picks per GPU arch).
- `test_encode_ideal` and `test_quantize_dispatch` (`tests/kernels/quant/test_quantize_tiles.py`) are good tests to
  port: any valid tail-biting codeword, once decoded, must quantize back to itself with zero error. Also check
  `is_tail_biting` (`TL`) on every output.
- Work per tile is 2 passes x 256 steps x 65,536 candidate evaluations (E nodes x 2^K branches = 2^16 for every K),
  about 3.4e7 fp16 evaluations per tile, independent of K. A 31B dense model has about 1.2e8 tiles, so about 4e15
  evaluations for the LDLQ pass alone. The scale search (3.5) adds 10 coarse evaluations on a third of the sampled
  tiles and 5 fine ones on all of them. The encoder dominates conversion time.
- Length-160 rings (n-gram tables) and fractional K use the same structure (sections 5.4 and 7).

---

## 5. On-disk format of one EXL3 linear (ground truth = the decoder)

### 5.1 Tensors

For Linear key `P` (e.g. `model.language_model.layers.0.self_attn.q_proj`), with padded `k = in`, `n = out`:

| tensor | dtype | shape | content |
|---|---|---|---|
| `P.trellis` | I16 | `[k/16, n/16, 16*K]` | packed tiles, tile (a, b) = rows 16a.., cols 16b.. of W_hat |
| `P.suh` | F16 | `[k]` | input scales (with signs) |
| `P.svh` | F16 | `[n]` | output scales (with signs) |
| `P.mul1` | I32 | `[]` | codebook marker (mul1); or `P.mcg`; none for 3INST |
| `P.bias` | F16 | `[n]` | only if the layer has a bias |

`K = trellis.shape[-1] / 16` (`EXL3` 70-73). The loader recognizes EXL3 storage by the presence of
`{suh or su, svh or sv, trellis}` (`LIN` 412-416). `su`/`sv` (packed int16 sign bitfields, `pack_signs`, `PACK` 185-240)
are a legacy alternative that 1.5.3 never writes.

### 5.2 Tile element order (tensor_core_perm, `Q` 22-50)

Ring position `i = 8t + j` (t = 0..31, j = 0..7) holds tile element `(r, c)`, row-major index `16r + c`:

```
r = 2*(t % 4) + (j & 1) + 8*((j >> 1) & 1)
c = t // 4 + 8*((j >> 2) & 1)
perm[8t + j] = 16*r + c
```

Quantize: `tile_vec_perm[i] = tile_rowmajor[perm[i]]`. Indices are stored in ring (permuted) order. Decoded values are
un-permuted (`out_rowmajor[perm[i]] = out[i]`) before they go back into Q.

### 5.3 Packing (`pack_trellis_kernel`, `PACK` 9-57; `TL` pack/unpack)

Per tile, integer K:

```
bitstream B[0 .. 256K-1]: for ring position i, B[iK .. iK+K-1] = low K bits of idx[i], MSB first
u32 word j (j = 0 .. 8K-1) = B[32j .. 32j+31] with B[32j] at bit 31 (MSB)
store word j little-endian: uint16[2j] = word_j & 0xFFFF;  uint16[2j+1] = word_j >> 16
tile = 16K uint16 (int16 in the file) = trellis[a][b][0 .. 16K-1]
```

Decoder (`DQ` 15-31, `unpack_trellis_kernel` `PACK` 95-137): the 16-bit state at ring position i is
`B[(i+1)K - 16 .. (i+1)K - 1]` (indices mod 256K, MSB first). Its low K bits are position i's own bits, its top bits
the previous positions' bits. Only the low K bits of each index are packed, so tail-biting is what makes the decode
equal the encoder's states.

### 5.4 Fractional K (mul1 only; `FRAC`, `TL`)

`K = KA + 0.5`, MASK = 0xAAAA: position i consumes `D(i) = KA + ((MASK >> (i & 15)) & 1)` bits. `S(i)` = prefix sum of
D; the window of position i = ring bits `[S(i) - 16, S(i))`. Ring length `R = 256K` bits = 16K uint16, same MSB-first
u32 words stored little-endian.

### 5.5 Inference math (`EXL3` 179-221; `TL` dequant)

```
W_hat[16a + r, 16b + c] = decode(state at ring position perm^-1(16r + c) of tile (a, b))     # fp16
W = diag(suh) · Hk · W_hat · Hn · diag(svh)            # (k, n), Hk/Hn = blockwise H128
y = x @ W + bias = ( had_r(x * suh) @ W_hat ) |> had_r |> * svh  + bias
```

`had_r_128(x, xh, pre_scale = suh)` multiplies by suh **before** the Hadamard; the output pass multiplies by svh
**after** it (`HAD` 125-205). Kernels need k and n to be multiples of 128.

---

## 6. Output files

### 6.1 Reference model inventory (gemma4-31b-based-exl3-4.0bpw, stock 1.5.3)

`config.json` text model: hidden 5376, intermediate 21504, 60 layers (every 6th full attention; `attention_k_eq_v = true`,
so full-attention layers have **no v_proj**), head_dim 256 / global 512, 32 q heads, 16 kv (4 global kv), vocab 262144,
tied embeddings. 2,419 tensors in 3 shards:

```
    1 F16   lm_head.weight                                         [262144, 5376]
    1 BF16  model.language_model.embed_tokens.weight               [262144, 5376]
    1 BF16  model.language_model.norm.weight                       [5376]
   60 BF16  model.language_model.layers.*.{input_layernorm,post_attention_layernorm,
                                           pre_feedforward_layernorm,post_feedforward_layernorm}.weight  [5376]
   60 BF16  model.language_model.layers.*.layer_scalar             [1]
   60 BF16  model.language_model.layers.*.self_attn.{q_norm,k_norm}.weight   [256] (full layers: [512])
   60x7 / 50 (v_proj)  EXL3 linears, each {suh F16, svh F16, mul1 I32 [], trellis I16}
   vision tower + embed_vision: original BF16 tensors under original names (unquantized, see 1.7)
```

Layer 0 (sliding attention):

| tensor | dtype | shape |
|---|---|---|
| `self_attn.q_proj.trellis` | I16 | [336, 512, 64] |
| `self_attn.q_proj.suh` / `.svh` | F16 | [5376] / [8192] |
| `self_attn.k_proj.trellis` | I16 | [336, 256, 64] (svh [4096]) |
| `self_attn.v_proj.trellis` | I16 | [336, 256, 64] (svh [4096]) |
| `self_attn.o_proj.trellis` | I16 | [512, 336, 64] (suh [8192], svh [5376]) |
| `mlp.gate_proj.trellis` / `mlp.up_proj.trellis` | I16 | [336, 1344, 64] (svh [21504]) |
| `mlp.down_proj.trellis` | I16 | [1344, 336, 64] (suh [21504], svh [5376]) |
| `*.mul1` | I32 | [] = 0x83DCD12D |

Layer 5 (full attention): `q_proj.trellis [336, 1024, 64]` (svh 16384), `k_proj.trellis [336, 128, 64]` (svh 2048),
`o_proj.trellis [1024, 336, 64]`, q_norm/k_norm [512], no v_proj and no v_norm tensor (`v_norm` is unweighted).
64 = 16 x K with K = 4. Measured statistics, layer 0: suh RMS 0.013-0.019; svh RMS about 1.01, |svh| from 0.3 to 5.8;
no zeros.

### 6.2 Safetensors files (`COMP` 18-148; writer `loader/safetensors_alt.py` 222-277)

- Standard safetensors: u64 LE header length, minified JSON header (`separators = (",", ":")`, no `__metadata__`, **no
  padding or alignment**: the reference headers are 99,217 / 150,263 / 58,343 bytes and data offsets are unaligned),
  then raw tensor bytes in header order. Loaders don't care about tensor order (the reference uses module insertion
  order, not alphabetical).
- Sharding (`COMP` 39-104): modules in order (`model.modules` + MTP + vision), whole modules per shard, start a new shard
  when `current + module_size > shard_size * 1024^2` (8192 MiB) and the current shard is non-empty; then the "additional"
  tensors (vision extras when unquantized) are appended the same way, tensor by tensor.
- Names: `model.safetensors` if one shard, else `model-%05d-of-%05d.safetensors`. With more than one shard, write
  `model.safetensors.index.json` = `{"metadata": {"total_size": <sum of tensor bytes>}, "weight_map": {name: file}}`
  (`json.dumps(indent = 4)`).
- Reference layout: shard 1 = embed (2.8 GB) + layers 0-22, shard 2 = layers 23-57, shard 3 = layers 58-59 + norm +
  lm_head (2.8 GB) + vision tensors. total_size = 21,449,963,840.

### 6.3 config.json (`COMP` 187-239)

Copy of the source `config.json` (`json.dumps(indent = 4)`) with:

```json
"quantization_config": {
    "quant_method": "exl3",
    "version": "1.5.3",
    "bits": 4.0,                    // round(final_bpw, 2) over the budgeted ("bits") linears
    "head_bits": 16,
    "calibration": {"rows": 250, "cols": 2048},
    "out_scales": "always",         // always / never / auto
    "codebook": "mul1"
}
```

Plus `"vision_bits": N` if the vision model was quantized (`!= 16`), and `"mtp_bits"` if there is an MTP model. If the
source already had a non-EXL3 `quantization_config` it goes into `original_quantization_config`. `update_config` sets
`tied_word_embeddings = True` only if that (misspelled) key exists, so in practice it does nothing (`QC` 5-12). Use our
own version string. exllamav3's loader never reads `quantization_config` (format detection is per tensor); keep it
identical for TabbyAPI and other tools.

### 6.4 quantization_config.json (`QC` 15-63)

`quantization_config` from above plus `"tensor_storage"`: one entry per **leaf module of the main (text) model** in
model order (vision is not listed), including leaves with no tensors (`"stored_tensors": {}` for the unweighted
`v_norm`). Modules absent from the model, such as the full-attention `v_proj`, are missing entirely. Each entry:

```json
"model.language_model.layers.0.self_attn.q_proj": {
    "stored_tensors": {
        "<name>": {"shape": [...], "n_bytes": N, "dtype": "torch.float16"},   // torch dtype names: torch.int16, torch.int32, torch.bfloat16
        ...
    },
    "quant_format": "exl3",          // EXL3 linears only
    "bits_per_weight": 4,            // trellis.shape[-1] / 16, int if integral
    "mul1_multiplier": 2212286765    // or "mcg_multiplier": 3417055213
}
```

Informational only (nothing in the exllamav3 loader reads it), but write it the same way.

### 6.5 Copied files (`COMP` 148-178)

Every regular file in the source directory except `*.safetensors`, `config.json`, `model.safetensors.index.json` and
`*.bin/.ckpt/.pth/.pt` (tokenizer files, `chat_template.jinja`, `generation_config.json`, `processor_config.json`,
README, images, `.gitattributes`...). Subdirectories are not copied.

---

## 7. Fork and post-1.5.3 features vs stock 1.5.3

Origin from `git log -S`: "upstream" = turboderp after 1.5.3, "fork" = vcruz305 / r0b0tlab.

| feature | where | origin | default | effect on format | first Gemma 4 dense version |
|---|---|---|---|---|---|
| refit_scales (3.7) | `Q` 1414 | stock 1.5.3 | on | none | **implement** (cheap, measurable gain) |
| grid + parabola scale search, LDLQ drift | `Q` 1109-1208 | stock 1.5.3 | on | none | **implement** |
| `--out_scales auto` = skew rule | `Q` 1325-1341 | stock 1.5.3 | "always" is the default | none | skip (use always) |
| `--out_scales` default **auto**, new rule: on except gate/up with `out_sensitivity` predicting `E[s*e]/(E[s]E[e]) > 1.05` | HEAD `Q` 1373-1505, `modules/mlp.py` 66-140 | upstream 2026-09-28 | auto | none | skip; optional later |
| `--out_scales yaqa` / `out_hessian_diag`: `t ~ (sigma^2/s)^(1/4)` | HEAD `Q` 1406-1432 | upstream 09-29 | off | none | skip |
| `--scaled_hessian`: per-tensor LDL of `P diag(su) H diag(su) P` | HEAD `Q` 1592-1612 | upstream 09-30, experimental | off | none | skip (a good, cheap later experiment) |
| refit damping only removed from the reported error | HEAD `Q` 1621-1690 | upstream | n/a | none | n/a |
| `--hessians` (YAQA Kronecker factors, two-sided LDLQ `ldlq_2hess`) | `Q` 668-755, `CM` 432-453 | stock 1.5.3 (experimental) | off | none | skip |
| `--embed_bits` (token-embedding trellis, `TableEmbedding` with `.signs`) | HEAD `CM` 84, `modules/embedding.py` | upstream 09-29 | 16 | **new tensor kinds** | skip (stock = bf16 copy) |
| fractional K 1.5/2.5/3.5 | `FRAC`, `quantize_tiles_frac_kernel.cuh` | stock 1.5.3 | below 4 bpw only | 16K words per tile | skip for >= 4 bpw |
| `--recipe` (sc_measure / sc_optimize) | `ALLOC` 209 | stock 1.5.3 | off | none | skip |
| n-gram tables (`--ngram_bits`, length-160 rings, `ngram_embedding.safetensors`) | `conversion/ngram.py` | stock 1.5.3 | per arch | separate file | skip (Flash-Next only) |
| MTP (`--mtp_bits` 4, uncalibrated) | `CM` 1579-1580 | stock 1.5.3 | 4 | normal EXL3 linears | skip for Gemma 4 |
| batched calibration rows (`-cbs`, `-cdn`) | HEAD `CM` 97 | fork (V. Cruz) | auto (32 rows) | none (numerics only) | performance choice |
| `--dump_hessians`, `--retry_outliers` (re-seed outlier tensors) | HEAD `CM` 105-112, `conversion/retry_outliers.py` | fork (r0b0tlab) | off | none | skip; the retry idea is useful later |
| `EXL3_LDLQ_DTYPE=tf32/bf16`, `EXL3_GS_*` env hooks | HEAD `Q` 14-22, 594-608, 1261-1271 | fork | unset | none | skip |

For a first Gemma 4 dense version: implement sections 1-6 with `out_scales = always`, `head_bits` 16 (or 6 with the
lm_head Hessian), `vision_bits` 16 (copy through), mul1 codebook, integer K, refit on. That reproduces the reference's
configuration.

### 7.1 What MoE models need on top (Gemma 4 MoE, GLM, Qwen3.8-Flash-Next)

- **Expert calibration** (`modules/block_sparse_mlp_routing.py` 106-116, 139-150, 174-199; `CM` 1268-1328): with
  `model.calibration_all_experts = True`, the capture pass routes **every token through every expert**
  (`selected_experts = all`, weights = full softmax/sigmoid). Only the capture pass does this; the reference outputs and
  the state advance use normal top-k routing through the quantized layer.
- **Hessians**: all experts' gate/up share one H, the routed-expert input (qmap `<moe>.input`, or `.latent` for latent
  MoE; `block_sparse_mlp.py` 310-351). Each expert's down projection has its own H (qmap `<moe>.<e>.down`) from its own
  intermediate activations over all tokens, **not weighted by routing probabilities** (routing weights are applied
  after down). Shared experts are a normal GatedMLP with their own qmaps. Gemma 4 MoE uses a separate qmap `block.moe`
  because of `routed_pre_norm` (`gemma4.py` 362).
- **Grouping**: expert gate/up concatenate (shared H); expert downs stack (up to 16 per batch, `ldlq_batched`). That
  is pure batching; per-tensor processing gives the same result.
- **Loading**: fused checkpoint tensors (`experts.gate_up_proj`, `experts.down_proj`; `fkey`/`frange`/`fidx` in `LIN`)
  are split per expert. Output names are per expert (`...experts.<e>.gate_proj.trellis`, etc.; check the arch file
  for each model).
- **Allocation**: `--hq` raises attention (+2 bits) and shared experts (+1) on Gemma 4 MoE (`select_hq_bits`,
  `gemma4.py` 260, 310). Router/gate matrices are not quantized.
- **Memory**: the capture pass with all experts on is E times the MLP work. Per-expert H for down is
  `moe_intermediate^2` fp32 each.
- **GLM** (`glm4_moe*.py`): DS3-style sigmoid routing with `e_score_correction_bias` and group top-k, first dense
  layers, shared experts, an MTP layer (uncalibrated, `mtp_bits`).
- **Qwen3.8-Flash-Next** (`architecture/qwen4_exp.py`): GatedDeltaNet (linear attention) layers, QSA sparse attention
  with an indexer, gated-residual hyper-connections (4 fp32 streams instead of one residual), a PLE n-gram embedding
  table (51.2B rows, quantized separately with the length-160 tail-biting codec in `conversion/ngram.py` into
  `ngram_embedding.safetensors`) and an MTP head. Its calibration forward is far more complex than the trellis work.
- **HEAD auto out-scales** depend on gate/up `out_sensitivity` captured from the down projection; MoE experts get it
  per expert. Not needed if we stay on "always".

---

## 8. Subtle points that make a reimplementation silently wrong

1. **Orientation**: quantize W^T (in, out). The trellis is `[in/16, out/16, 16K]`. suh has `in` entries and is applied to
   x **before** the input Hadamard; svh has `out` entries, applied after the output Hadamard.
2. **Weights go through fp16 first**: `W = float(f16(bf16 weight))`, not the raw bf16 values. bf16 values beyond fp16
   range would become inf and small ones lose precision as subnormals. Harmless for Gemma 4, but match it (or clamp
   deliberately).
3. **Padding**: pad k and n with zeros to multiples of 128 *before* everything (H capture included). The stored tensors
   have the padded sizes. Zero output columns get svh = 0 exactly.
4. **Hadamard**: natural-order Sylvester, scaled 1/sqrt(128), blockwise over 128, both sides. A Walsh (sequency) ordered
   or unnormalized transform, or one over the full dimension, decodes to garbage while still "loading".
5. **Sign of su**: `su = su0 * rms / (-1.24371088)`. The minus sign is in stock 1.5.3. Any consistent sign works because
   suh absorbs it, but the transform chain must stay consistent.
6. **Tile permutation**: the Viterbi runs along `tensor_core_perm` order, indices are packed in that order, and decoded
   values are un-permuted before error feedback. Running the trellis in row-major order gives a "valid" file that
   decodes to scrambled tiles.
7. **Bit packing**: low K bits of each state, MSB first in a 256K-bit ring, 32-bit words with bit 0 at the MSB, words
   stored little-endian (so the two uint16 halves of each word are swapped relative to a big-endian 16-bit stream).
   `PACK` 56 does this with `SWAP16`. Test against `TL.pack`/`TL.unpack` and the reference model (0.1).
8. **Tail-biting is mandatory**: the decoder reads each state as the 16 bits ending at (i+1)K, so the encoder must close
   the ring (two-pass scheme, 4.3). A non-tail-biting encoder gets the first ceil(16/K) weights of every tile wrong.
9. **Viterbi precision**: inputs rounded to fp16, fp16 squared error, fp16 FMA accumulation, fp16 costs. Tie rules:
   lowest branch k; end-state ties by the bit-reversed rank. The scale search uses the same quantizer, so the precision
   affects g as well.
10. **Codebook constants**: mul1 multiplier 0x83DCD12D, bias via fp16 0x6400 + byte-sum, K_INV 0x1EEE, K_BIAS 0xC931,
    single rounding (FMA). Check against the table in 4.1 (min -3.453125 / max 3.3477 match the reference tiles).
11. **Hessian**: fp32 sum of x^T x of the **fp16** activations, divided by token count; damping
    `+0.025 * mean(diag)`; rotate with the **signs only** (su0), not the scaled su; block LDL with 16x16 blocks, zero
    diagonal blocks; LDLQ bottom-up with `X = W + L^T E` restricted to rows below. Using `L` instead of `L^T`, top-down
    order, or the unrotated H gives worse quality but still a loadable file. That makes these the hardest bugs to see:
    compare proxy errors with exllamav3's printed numbers on a small model.
12. **Hessian source**: from the **original** weights of the current layer, on inputs produced by the **quantized**
    previous layers. Only the first linear of each qmap accumulates.
13. **Activation dtype in the calibration forward**: the residual stream is fp32, norm outputs and MLP intermediates are
    fp16 (exllamav3 conventions; see the separate `gemma4-forward.md` in this directory for the Gemma 4 forward). The H
    reflects fp16-rounded inputs.
14. **Refit uses the damped H in the original basis** (undo Hadamard and su0 signs) and the original fp32 weight. su and
    sv are multiplied in place, then rounded to fp16 only at the very end.
15. **`+1e-10`** is added to signed su/sv values (negligible, but it means a channel scale is never exactly 0 except
    where sv is reset to 0 for zero columns).
16. **lm_head with tied embeddings**: still written as `lm_head.weight` (fp16) or as EXL3 `lm_head.*`; inference expects
    the `lm_head` key, not the embedding.
17. **Per-module seed**: all tensors of a module share `seed = module index`. Only the quality spread between tensors
    depends on it; reusing identical sign vectors for q/k/v or gate/up (as stock does in places) is fine.
18. **K is per tensor** (`trellis.shape[-1] / 16`). Mixed K within a model is normal; every K from 1 to 8 must be
    encodable. Only 1.5/2.5/3.5 are valid fractional K (mul1).
19. **fp32 GEMMs everywhere** in LDLQ, refit and the Hadamards (TF32 is off in stock torch). oneMKL sgemm in fp32 is the
    matching choice. Cholesky in fp32 can fail on near-singular H. Stock retries with
    `+2 * 0.025 * mean(diag) * I` up to 10 times. fp64 Cholesky on the CPU is a safe alternative.

---

## 9. Minimal implementation checklist (dense, mul1, integer K)

1. Loader: HF safetensors -> per-linear (in, out) fp32 of the fp16-rounded weight, padded to 128.
2. Calibration rows (export once from exllamav3, or the native re-implementation of section 1.2).
3. Model forward (Gemma 4) with exllamav3 dtype conventions; per-qmap fp32 Hessian accumulation.
4. Per tensor: finalize H -> regularize -> scale search -> LDLQ with the Viterbi kernel -> back-transform -> refit -> pack.
5. Reload the quantized layer (decode as in 5.5) and advance the state.
6. Writer: shards + index + config.json + quantization_config.json + copied files.
7. Tests: codebook tables (4.1), pack/unpack round trip vs `TL`, tail-biting of every tile, the zero-error ideal-codeword
   test, decode of the reference model (0.1), and a proxy-error comparison with exllamav3 on a small dense model run on a
   CUDA box (the B70 can't run the CUDA extension).
