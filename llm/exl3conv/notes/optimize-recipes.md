# exllamav3 "optimized quantization recipes": implementation spec for a native port

This file describes exllamav3's per-tensor bit allocation pipeline (self-sampled calibration trace, noise-injection
sensitivity, rfn anchors, the optimizer, the recipe in `convert.py`) and the half-integer trellis rates, precisely
enough to reimplement them in exl3conv (C++/SYCL, no PyTorch). Everything here was read from source; no Python or
GPU code was run. Statements marked **(inference)** are mine, not something the source states.

## 0. Sources and versions

Local copy: `~/b70/llm/dsv41/exllamav3` (`__version__ = "1.6.1.post1"`, fork `vcruz305/exllamav3`, HEAD
`d968d96d`). All line numbers refer to it. Abbreviations:

| Tag | File |
|---|---|
| `DOC` | `doc/optimize.md` |
| `TR` | `sc_trace.py` |
| `ME` | `sc_measure.py` |
| `RP` | `sc_rfn_probe.py` |
| `OP` | `sc_optimize.py` |
| `QBP` | `eval/qbench_prompts.py` |
| `CM` | `exllamav3/conversion/convert_model.py` |
| `ALLOC` | `exllamav3/conversion/allocation.py` |
| `CAL` | `exllamav3/conversion/calibration_data.py` |
| `COMP` | `exllamav3/conversion/compile.py` |
| `QC` | `exllamav3/conversion/quant_config.py` |
| `Q` | `exllamav3/modules/quant/exl3_lib/quantize.py` |
| `EXL3` | `exllamav3/modules/quant/exl3.py` |
| `G4` | `exllamav3/architecture/gemma4.py` |
| `FRAC` | `exllamav3/exllamav3_ext/quant/frac.cu` |
| `QTF` | `exllamav3/exllamav3_ext/quant/quantize_tiles_frac_kernel.cuh` |
| `QCU` | `exllamav3/exllamav3_ext/quant/quantize.cu` |
| `DQ` | `exllamav3/exllamav3_ext/quant/exl3_dq.cuh` |
| `BK` | `exllamav3/exllamav3_ext/quant/bits_k.cuh` |
| `RC` | `exllamav3/exllamav3_ext/quant/reconstruct.cu` |
| `CB` | `exllamav3/exllamav3_ext/quant/codebook.cuh` |
| `TL` | `tests/testlib/trellis.py` (NumPy reference for the format) |
| `TQF` | `tests/kernels/quant/test_quantize_tiles_frac.py` |

History (local `git log`): pipeline added by turboderp in `abf49117` (2026-08-21), layer streaming in `sc_measure`
by UnstableLlama (PR #364, `6f336da2`, merged `6f4e4de6` 2026-09-19), grouped targets `1d641116` (2026-09-19),
"Fractional trellis support" `07b8a2e0` (2026-09-20, between v1.5.0 and v1.5.1), LDLQ drift in the scale search
`d1feab12` (2026-09-19). So stock 1.5.3, which exl3conv ports, already contains half-integer rates and the recipe
flags.

### 0.1 Local copy vs upstream master (turboderp-org/exllamav3, master = v1.6.4, fetched 2026-10-24)

Byte-identical: `doc/optimize.md`, `sc_optimize.py`, `sc_rfn_probe.py`, `calibration_data.py`, `quant_config.py`,
`frac.cu`, `quantize_tiles_frac_kernel.cuh`, `bits_k.cuh`, `exl3_dq.cuh`, `gemma4.py`, `eval/qbench_prompts.py`.
Differences that matter here:

1. `sc_trace.py`: the fork adds `--extra_conversations FILE` / `--only_extra` (TR 595-602, 776-777; merged from
   r0b0tlab in `710c7b21`). Upstream has neither. Defaults unchanged.
2. `sc_measure.py`: upstream passes the row's token ids as `params["input_ids"]` into every module forward (for
   per-layer-embedding "PLE" layers). Gemma 4 31B has `hidden_size_per_layer_input = 0`, so no effect on it.
3. `allocation.py` / `convert_model.py`, upstream only:
   - recipe keys may be fnmatch patterns (`*.linear_attn.*`); entries resolve top to bottom, the last matching entry
     wins; every budgeted tensor must be covered and every entry must match at least one budgeted tensor;
   - `--hq` is an integer multiplier (`-hq`, `-hq 2`, ...): floor = `rate_floor(base + hq * select_hq_bits)`;
   - with `--recipe`, `--hq` still promotes MTP layers (locally `--hq` is ignored with a warning, CM 234-235).
4. The fork also adds calibration row batching (`--calib_batch_rows`, CM 1095-1170), which stacks rows assuming equal
   length. Upstream has no batching. **(inference)** With `-cd` the rows have different lengths, so in the fork the
   batched path would fail to stack them; irrelevant for us, just don't copy that assumption.

`DOC` says `sd_trace.py` in two places (DOC 83, 116, 255); the script is `sc_trace.py`. `DOC` 204-206 and the
`--tie` help text (OP 329) say the default tie is `k_proj+v_proj,gate_proj+up_proj`; the actual default is
`q_proj+k_proj+v_proj,gate_proj+up_proj,in_proj_qkv+in_proj_z` (OP 327).

### 0.2 Gemma 4 31B facts used below

From Dino0040's `3.00bpw` `config.json` and `G4`: 60 decoder layers, hidden 5376, intermediate 21504, vocab 262144,
`attention_k_eq_v = true` (the 10 full-attention layers have no `v_proj`), `tie_word_embeddings = true`,
`hidden_size_per_layer_input = 0`. exllamav3 module list: index 0 embedding, 1..60 decoder blocks, 61 final norm,
62 `lm_head` (the Linear has `qbits_key = "head_bits"`, `alt_key = embed_tokens` when tied, `softcap =
final_logit_softcapping`, G4 406-416). Budgeted ("bits") tensors: 410 (60 q, 60 k, 50 v, 60 o, 60 gate, 60 up,
60 down) with 29,286,727,680 weights. The residual stream between modules is fp32 (G4 `out_dtype = torch.float`).
Key names: `model.language_model.layers.{L}.self_attn.{q,k,v,o}_proj`, `model.language_model.layers.{L}.mlp.{gate,up,down}_proj`, `lm_head`.

---

## 1. `sc_trace.py`: the self-sampled calibration trace

### 1.1 What generates it

Whatever model `-m` points at, loaded with exllamav3's `model_init` and sampled with its dynamic `Generator`
(TR 604-617). The docstring (TR 25-27) and DOC 92-93 say: the unquantized model or a high-bitrate quant ("6 bpw is
minimally distorted and much faster"). The trace is then reused for measurement, recipe optimization and conversion
of every lower-bitrate quant of the same model. In Dino0040's Gemma 4 31B example, `cal_trace.json` records
`"model": "/workspace/q6"`, which **(inference)** is a 6 bpw EXL3 quant, not BF16.

### 1.2 Prompts

`CONVERSATIONS` (TR 36-584): **157 seeded conversations**, each `(seed prompt, [follow-ups])`; 97 have one turn, 60
have two (217 turns total, counted by parsing the list). Sections: reasoning/puzzles, mathematics, coding, science,
medicine, history/geography, economics/finance, philosophy/ethics, creative writing, professional writing,
data/structured output, summarization of provided text, casual chat/roleplay, cooking, sports/games/music, DIY,
technology explainers, translation/linguistics, then prompts written in Chinese, Japanese, Korean, Spanish, French,
German, Russian, Portuguese, Italian, Arabic, Hindi, Turkish, Dutch, Polish, Swedish, Vietnamese, Thai,
Indonesian, Finnish, and a mixed/code-switching section. A port should copy the list verbatim (MIT, credit
turboderp). Fork-only: `--extra_conversations` appends a JSON list of `[seed, [follow-ups]]`; `--only_extra` drops
the built-in set (TR 595-602).

### 1.3 Chat template and template variables

- `TEMPLATE_VARS = {"enable_thinking": true, "reasoning_effort": "high"}` (`QBP` 323) updated with `-tv JSON`
  (TR 619-620).
- Rejected variables are dropped one at a time: probe `[{"role":"user","content":"hi"}]` with
  `add_generation_prompt=True`; while it throws, find a key whose removal makes it work and delete it; if none, use
  `{}` (TR 622-640).
- Context tokens: `tokenizer.hf_chat_template(messages, add_generation_prompt=True, **TEMPLATE_VARS)` (TR 668-669),
  which is HF `AutoTokenizer.apply_chat_template(..., tokenize=True, return_dict=True)` (tokenizer.py 757-790).
- **BOS**: comes only from the template. No BOS is added by sc_trace, by `-cd` loading, or by sc_measure. In the
  Gemma 4 trace every row starts `2, 105, 9731, 107, 98, 106, 107, 105, 2364, 107` (BOS, `<|turn>system\n`, the
  thinking marker, `<turn|>\n<|turn>user\n`) and the context ends `105, 4368, 107` (`<|turn>model\n`).

### 1.4 Sampling

- Each `Job` gets no sampler, so `DefaultSampler()` (job.py 166-168): `SS_MinP(0.08)` (mask tokens with
  `p < 0.08 * p_max`), then `SS_Temperature(0.8)`, then `SS_Sample()` (generator/sampler/presets.py 3-12).
  This is exllamav3's default, **not** the model's `generation_config.json` (Gemma recommends temperature 1.0,
  top-k 64, top-p 0.95).
- `max_new_tokens = 3072` (TR 772), `stop_conditions = config.eos_token_id_list`, `decode_special_tokens=True`.
  The stop token is not in `response_ids` (checked in the Gemma trace: responses end on ordinary tokens).
- Per-job seed `zlib.crc32(f"{seed}|{epoch}|{c_idx}|{t_idx}".encode())`, `--seed` default 0 (TR 676, 774).
- Generator: `max_chunk_size = 2048` (TR 616); model_init defaults set by TR 766: cache 131072 tokens,
  `-ambs` 32; draft-model options available. DOC's example uses `-cs 262144 -ambs 32`.

### 1.5 Loop and lengths

```
target_tokens = --target_tokens or cal_rows * cal_cols            # default 250 * 2048 = 512,000 (TR 642)
rows = []; total_in = total_out = 0
for epoch in 0 .. max_epochs-1 (default 4):                        # TR 650
    if total_in + total_out >= target_tokens: break
    convs = fresh copies of CONVERSATIONS (messages = [], alive)
    for t in 0 .. max_turns-1:                                     # turn-major waves, TR 660
        if total >= target_tokens: break                           # checked only between waves
        for every alive conv with a turn t: append {"user": turn_t}, template, enqueue a Job
        run the wave to completion
        for every finished job: if the response is empty -> drop the user message, conv dead
                                else rows.append({conversation, turn, epoch, eos_reason,
                                                  input_ids = context tokens, response_ids = sampled tokens})
                                     total_in += len(ctx); total_out += len(resp)
                                     append {"assistant": clean_response_for_context(resp)}
```

A wave always completes, so the total overshoots by up to one wave. `clean_response_for_context` (QBP 330-342)
keeps the text after the last harmony `<|channel|>final<|message|>`, or after the last `</think>`, else the whole
response decoded **without** special tokens. **Gemma 4 quirk (observed in the data):** Gemma 4's thinking block is
`<|channel>thought ... <channel|>`, which matches neither rule, so the next turn's context contains the previous
turn's full reasoning text as plain text (turn-1 contexts in the trace contain `45518` "thought" right after
`<|turn>model\n`). A deployed chat loop would strip it. A port may reproduce this or fix it; fixing changes the data.

In Dino0040's trace: 374 rows (epoch 0 turn 0: 157, epoch 0 turn 1: 60, epoch 1 turn 0: 157), 128,492 context +
640,584 response tokens, 368 stopped on a stop token and 6 hit 3072; row length (context + response) min 390,
median 1825, max 5145; 120 of 374 exceed 2048.

### 1.6 Output files

`cal_trace.json` (TR 729-740), qbench-compatible:

```json
{"model": "<-m path>", "vocab_size": <tokenizer.actual_vocab_size>, "template_vars": {...},
 "meta": {"rows": N, "input_tokens": I, "output_tokens": O},
 "rows": [{"conversation": c, "turn": t, "epoch": e, "eos_reason": "stop_token"|"max_new_tokens",
           "input_ids": [...], "response_ids": [...]}, ...]}
```

`cal_trace.safetensors` (TR 742-761; path `-co`, default the JSON path with `.safetensors`):

```
order = list(range(len(rows))); random.Random(seed).shuffle(order)    # Python's Mersenne Twister, seed 0
order = order[:cal_rows] if len(order) > cal_rows
input_ids : int64 [n_rows, cal_cols]   row r = (rows[order[r]].input_ids + response_ids)[:cal_cols], right-padded with 0
lengths   : int64 [n_rows]             true length of row r (<= cal_cols)
```

One trace row per packed row, always from position 0 (BOS, system turn, whole turns), never a window cut out of a
concatenated stream; the comment (TR 742-746) says Gemma4-it is "wildly unstable" on mid-conversation windows.
Dino0040's file: header `{"input_ids":{"dtype":"I64","shape":[250,2048]},"lengths":{"dtype":"I64","shape":[250]}}`;
mean length 1774, min 390, 83 rows full (2048), 443,537 tokens in total (vs 512,000 for the default corpus mix).

### 1.7 How `convert.py -cd` consumes it

`get_file_calibration` (CAL 102-130), called from `prepare_state` (CM 436-453) **instead of** the bundled corpus
mix. It does not mix with it (no corpus rows, no random-token rows):

```
data = load_file(cal_data); packed = data["input_ids"]; lengths = data.get("lengths")    # tensor names exact
rows, cols = cal_rows (250), cal_cols (2048)
if packed.shape[0] < rows: warn; rows = cal_rows = packed.shape[0]     # the reduced count is what gets recorded
if packed.max() >= tokenizer.actual_vocab_size: error
row i (i = 0 .. rows-1, file order) = packed[i, :n] as int64, n = cols if lengths is None else min(cols, lengths[i])
n < 1 -> error
```

So calibration is a list of rows of **different lengths**. The capture accumulates H over exactly those tokens
(count = their total). No BOS is prepended. `-cd` and `-rcp` are independent (DOC 242-243).

---

## 2. `sc_measure.py`: per-tensor sensitivity by noise injection

### 2.1 Inputs, rows and defaults

- `-m`: the **unquantized** HF model; every target's inner layer must be `LinearFP16` (ME 415-418, 766-768), so the
  BF16 weights are loaded as fp16 and perturbed in fp16.
- `-r/--rows 10`, `-l/--length 1024`, `-hr/--h_rows 64`, `--shaped` off by default (DOC: "strongly recommended"),
  `-rfn 0.29,0.145` (global levels), `-rr` per-tensor anchors (overrides `-rfn`), `-rs 1.0,0.5`, `-dr/--draws 1`,
  `--load-mode auto|resident|streaming`, `-ms/--max-sys` GB (ME 886-910).
- With `-tr trace.safetensors` (ME 310-331): eval rows = trace rows `0 .. rows-1`, capture rows = trace rows
  `rows .. rows+h_rows-1` (disjoint from each other, but both inside the first 250 rows that `convert -cd` uses).
  Each row truncated to `min(length, lengths[i])`, never padded. Without `-tr`: wikitext-2 test split, tokenized
  as one stream, cut into consecutive `length`-token rows (ME 251-270).
- Dino0040 ran the defaults: 10 eval rows x 1024 (all trace rows are >= 1024, so 10,240 eval tokens) and 64 capture
  rows x 1024 (65,536 tokens), `--shaped`, `-rr rfn_2.00bpw.json -rs 1.0,0.5`.

### 2.2 Targets (what gets perturbed together)

`collect_targets` (ME 207-248), per top-level module. For Gemma 4 dense:

| Target key | Members | numel |
|---|---|---|
| `...layers.L.self_attn.qkv` | q_proj, k_proj, v_proj (no v_proj where `use_k_as_v`) | sum of members |
| `...layers.L.self_attn.o_proj` | itself | |
| `...layers.L.mlp.gate_up` | gate_proj, up_proj | sum |
| `...layers.L.mlp.down_proj` | itself | |
| `lm_head` | itself (`qbits_key = "head_bits"`) | 1,409,286,144 |

241 targets for Gemma 4 31B (Dino0040's `noise_attrib.json` has exactly these). A grouped target's members are
perturbed **simultaneously**, each at its own level and seed, and get one KLD. The optimizer then gives the group one
bitrate. (MoE layers group all routed experts; GatedDeltaNet groups qkv/z.)

### 2.3 Noise levels

With `-rr` (ME 429-438): per member `level_l = rfn_ref[member] * scale_l`, scales `1.0, 0.5`. With anchors from a
2 bpw probe that is the K=2 error and half of it (~ the K=3 error, since the amplitude ratio per bit is 1.96).
Without `-rr`: every tensor gets the global levels `0.29, 0.145`. A grouped target's nominal level for the record
is the numel-weighted rms of its members' levels (ME 444-450). Measurements per target: `levels x draws`, ordered
`for li: for draw:` (ME 655-658).

### 2.4 The perturbation

Weights are stored `(in, out) = (k, n)` (exllamav3 `LinearFP16.weight`). Seeds use torch generators
(`zlib.crc32(f"{key}|{li}|{draw}") & 0x7fffffff`, ME 593); a port can use any Gaussian generator.

**iid** (ME 494-517): `sigma = rfn * sqrt(sum(W^2) / numel)`; `W += sigma * N(0,1)` in fp32, rounded to fp16;
returns realized `rfn_actual = ||W_new - W|| / ||W||` (after fp16 rounding), and `||W||`.

**shaped** (ME 519-579, used when `--shaped` and the capture for that qmap is usable):

```
# shaping factors per qmap, from the capture rows (2.5): L (k x k, strictly lower: unit block-16 LDL factor of the
# rotated, damped H with its diagonal zeroed) and su (k signs), exactly as the quantizer computes them
col_sq[j]  = sum_i W[i, j]^2                         # per output channel
w_sq       = sum_j col_sq[j]
col_scale  = sqrt(col_sq); col_scale /= sqrt(mean(col_scale^2)); clamp(col_scale, min = 1e-4)
for each chunk of output columns (width max(128, 2^26 // k)):
    eta = N(0,1) of shape (k, cols)                     # fp32
    x   = solve(L^T x = eta), unit diagonal implicit     # x = L^-T eta
    x   = blockdiag(H128^T) x                            # H128 = Sylvester Hadamard / sqrt(128), per 128-row block
    x  *= su[:, None]                                    # input-channel signs
    x  *= col_scale[None, cols]                          # "out_scales-like" per-output-channel weighting
dW = all chunks;  factor = rfn * sqrt(w_sq / sum(dW^2));  W = fp16(W + factor * dW)
rfn_actual = ||W_new - W_old|| / ||W_old||
```

The generator is re-seeded and the noise regenerated for the second pass, so the full fp32 noise is never held
(two passes: norm, then apply). Note what it leaves out **(inference)**: the quantizer's input-channel scales,
the output-side Hadamard and sv signs, and the actual trellis error spectrum; it only reproduces the LDLQ input-side
colouring and an output-channel weighting.

Grouped target (ME 583-609): each member perturbed as above at its own level; realized group rfn =
`sqrt(sum err_sq / sum w_sq)`. Saved weights are restored after each measurement (to host RAM if the group exceeds
1 GiB). If `finalize_capture_H` reports fallback for a qmap, that qmap gets iid noise and the record's
`shaped = false`.

### 2.5 Hessians for the shaped noise

Capture rows run through the **clean** model (never through perturbed or quantized layers), one module at a time,
capturing H only at modules that have targets still to measure (ME 617-632, 747-755, 806-813). Per module and qmap:
`finalize_capture_H(h_data, {"sigma_reg": 0.025}, False)` (ME 638-653, Q 1059-1152): `H /= count`;
`H.diag += 0.025 * mean(diag(H))`; random signs su (torch seeded with `crc32(f"su|{module_idx}")`); `H *= su^T`,
blockwise Hadamard right, `H *= su`, blockwise Hadamard left; `block_ldl(H, 16)`; `L[i, i] = 0`. This is the
quantizer's own finalize, so exl3conv's existing implementation (notes/exl3-algorithm.md §3.3) serves unchanged.
**(inference)** Conversion computes H from states that passed through the already-quantized earlier layers; the
measurement uses clean states. Small difference, by design of the upstream script.

### 2.6 The score

```
ref pass: for every eval row r, run all modules with clean weights; keep
          boundary[i][r] = input state of module i (token ids for i = 0)        # cloned, fp32 hidden states
          ref_logits[r]  = final output .half() on the host                     # Gemma: softcapped logits
for every target t in module i, level l, draw d:
    perturb t;  for r: x = boundary[i][r]; run modules i .. end (clean except t); 
                       kl_r = mean over positions p of KL( softmax(ref_logits[r][p]) || softmax(x[p]) )
    KLD(t, l, d) = mean_r kl_r;  restore t
```

- KL direction: `compute_kl_div(input, target)` = KL(softmax(target) || softmax(input)), with `target` = clean
  reference, `input` = perturbed run, over `min(vocab_size, logit widths)` entries, in 256-position chunks
  (ME 336-345, 365-375; util/measures.py 210-219).
- Every position of the row counts, prompt included (qbench, by contrast, scores response positions only).
- Per row the mean is over its own positions; rows are then averaged equally.
- Control: per module with targets, the same suffix pass without perturbation must give exactly 0.0, else abort
  (ME 697-701). **(inference)** That only holds because the head emits fp16 logits, so `.half()` of the reference
  is lossless; a port that keeps fp32 logits must store the reference in fp32 or skip the exact check.
- Also recorded: `inj_rfn = ||state_after_module_i(perturbed) - clean|| / ||clean||` over all eval rows
  (ME 674-684). Informational.

### 2.7 Output: `noise_attrib.json`

Written after every target (resident) or every pass (streaming) via `.tmp` + rename; a rerun resumes and skips keys
already present (ME 452-490).

```json
{"model": "...", "rows": 10, "length": 1024, "draws": 1, "mode": "shaped"|"iid", "trace": "<path>|null",
 "h_rows": 64|null, "rfn_ref": "<path>|null", "rfn_scale": "1.0,0.5"|null, "rfn": "0.29,0.145"|null,
 "results": [
   {"idx": <top-level module index>, "key": "<target key>", "qbits_key": "bits"|"head_bits",
    "numel": <sum of member numel>, "shaped": true,
    "levels": [{"rfn": nominal, "rfn_actual": realized, "draw": 0, "kld": KLD, "inj_rfn": ...}, ...],
    "members": ["...q_proj", "...k_proj", "...v_proj"],          // grouped targets only
    "w_norm": ||W|| over the target}
 ]}
```

Example (Dino0040, layer 0 qkv): numel 88,080,384; level 0 rfn 0.29277, kld 8.72e-5; level 1 rfn 0.14638, kld
3.33e-5. `lm_head` (idx 62): rfn 0.02016 (its 6 bpw probe), kld 4.10e-5.

### 2.8 Data dependency, and what streaming needs

For target t in module i the score needs: (1) the clean input state of module i for every eval row, (2) module i
with t perturbed, (3) **all later modules clean, through the head**, (4) the clean reference logits. Measurements
are independent of each other. The dependency is "one perturbed state per measurement, carried from module i to the
end", which makes the work quadratic in depth and the memory a question of how many of those states are alive at
once.

Upstream's streaming mode (ME 57-61, 703-798) does exactly this:

1. Reference pass streaming one module at a time: `boundary[i]` (all eval rows) for every module and
   `ref_logits` kept in **host RAM**.
2. Count measurements per module: `1 (control) + targets x levels x draws`. Gemma 4 31B with 2 levels: 9 per
   decoder layer, 3 for the head module, 543 in total.
3. `state_bytes` = size of one module's boundary over all eval rows (Gemma: 10 x 1024 x 5376 x 4 B = 220 MB).
   Budget = `--max-sys` GB, default half of `MemAvailable` (ME 157-168). Split modules into passes so that
   `pending_count x state_bytes <= budget`, keeping a module's measurements in one pass (ME 171-184).
4. Per pass: load modules from the pass's first member (or the capture cursor) to the end, once each. At each loaded
   module: advance the capture rows (accumulating H only if this module spawns), push every pending state through
   the clean module, then for each new target: perturb, run the boundary rows through this module, restore, append
   the result as a new pending state. At the last module, pending states turn into KLDs (ME 737-745).
5. Device holds one module plus its shaping factors (`L`, `su` kept on the host and moved per perturbation in
   streaming) plus one row in flight. "The largest module plus Hessian/noise workspace must still fit on the
   device" (ME 60).

Memory for Gemma 4 31B, defaults (10 x 1024 eval, 64 x 1024 capture, fp32 states):

| Item | Size | Upstream resident | Upstream streaming |
|---|---|---|---|
| fp16 weights | ~62 GB | VRAM | one module at a time (layer 1.3 GB, head 2.8 GB) |
| clean boundary states, 63 modules x 10 rows | 13.9 GB | VRAM | host RAM |
| reference logits, 10 x 1024 x 262144 fp16 | 5.4 GB | host | host |
| capture rows, 64 x 1024 x 5376 fp32 | 1.4 GB | VRAM | VRAM |
| H + L per layer (qkv 116 MB, o 268 MB / 1.07 GB, gate/up 116 MB, down 1.85 GB, each x2) | up to ~6.5 GB | VRAM | VRAM per module |
| one row's logits through the head, fp16 / fp32 | 0.5 / 1.1 GB | VRAM | VRAM |
| pending perturbed states, 543 x 220 MB | 119 GB if one pass | n/a | host RAM, split into passes |

Dino0040 reports "~1 h sensitivity measurement on an unquantized model, ~120 GB of VRAM" (resident, GB10 unified
memory). DOC: a 27B with two levels takes ~35 minutes on a fast GPU, "fits comfortably in 96 GB".

**A streaming design for one 32 GB B70 (inference, my proposal):**

- The clean boundary need not be stored for all modules: carry the clean eval states as one more pending state
  (it *is* the control) and spawn from it as each module is reached. Then per pass only `pending + 1` states and
  the capture rows are alive. Each pass starts from the embedding (or a saved clean state at the pass start).
- Pending states fit on the card: with ~16 GB for them, 72 states = 8 layers of spawn per pass, so ~8 passes and
  ~270 layer loads in total. This machine has 62 GB of RAM (~29 GB available today), so host-RAM pending states as
  upstream does would give ~14 GB budget and ~9-10 passes anyway.
- Work: sum over layers of `9 x 10 rows x (60 - i)` = 164,700 row-layer forwards of 1024 tokens plus 543 x 10
  head passes. Roughly 165 PFLOP for the layers and 16 PFLOP for the head **(inference, back-of-envelope)**: about
  an hour on the B70 at 50 TFLOPS effective, much less than a conversion. The control pass is a sanity check and
  can be skipped (saves 1/9).
- Storing pending states as bf16 halves memory but adds a rounding to every measurement; keep fp32 unless memory
  forces it.
- Nothing in the measurement needs the whole model resident.

---

## 3. `sc_rfn_probe.py`: rfn anchors

An "anchor" is the measured relative Frobenius error of one tensor in a real EXL3 quant, at that quant's K:

```
rfn_t = ||W_q - W_ref||_F / ||W_ref||_F
```

- `W_q = LinearEXL3.get_weight_tensor()`: the full dequantized weight in the original basis
  (`diag(suh) H W_hat H diag(svh)`), `(in, out)` fp16. `W_ref = LinearFP16.get_weight_tensor()` from the HF model.
  Cropped to the common `(in, out)` (padding), fp32 sums over 8192-column chunks (RP 59-72).
- Both models streamed one top-level module at a time (RP 40-86). Every Linear with a qmap whose quantized inner is
  EXL3 is probed, head included.
- Output (RP 96-97): `{"model_q": ..., "model_r": ..., "results": [{"key", "K", "qbits_key", "numel", "rfn",
  "ref_norm"}]}`, one entry per **tensor** (not grouped). `K` is the probed tensor's bitrate (from its trellis width).
- Dino0040 (fresh uniform 2 bpw quant, exllamav3 v1.6): 411 entries, K = 2 for all 410 decoder tensors, `lm_head`
  K = 6 rfn 0.0202. Spread at K = 2 (min / median / max): q 0.293/0.310/0.344, k 0.293/0.307/0.349,
  v 0.293/0.307/0.333, o 0.284/0.309/0.351, gate 0.293/0.298/0.330, up 0.293/0.299/0.317, down 0.280/0.308/0.333.
  The minimum sits at the default global anchor 0.292.

**Extrapolation to other K** (OP 58-70): `rfn_t(K) = rfn_anchor_t * bit_ratio^(K_anchor_t - K)`,
`bit_ratio = 1.96` (`-br`), applied to half-integer K as well (one half bit = factor 1.96^0.5 = 1.40). Without `-rr`
every tensor uses `--anchor 2:0.292`. DOC 133-135: K = 2 anchors "have been validated to extrapolate across the full
K range via the amplitude-halving law". Sanity check from `TQF` 25-27: unit-Gaussian tile MSE ~0.28 / 0.068 / 0.018
at K = 1 / 2 / 3, i.e. amplitude ratios 2.03 and 1.94 per bit.

For a native port **(inference)**: the converter can write `rfn` for every tensor it quantizes (it has `W` and the
dequantized result), so a probe is simply a uniform conversion at some K with that logging on. The anchor depends
on the calibration (through LDLQ), but weakly; K = 2 is what upstream validated.

---

## 4. `sc_optimize.py`: the allocation

### 4.1 Model

```
tensors = results with qbits_key == "bits"      # the budget; grouped targets are single entries
head    = results with qbits_key == "head_bits" # only reported, never budgeted   (OP 39-41)

alpha   = --alpha, else median over ALL results (head included) of
          log(kld_0 / kld_last) / log(rfn_actual_0 / rfn_actual_last)        (OP 43-56)
anchor[key] = (K, rfn) from --rfn_ref, else (2, 0.292) for every key          (OP 58-66)
grouped target without its own anchor: (most common member K, sqrt(mean of member rfn^2))   # unweighted rms (OP 81-92)
rfn_at(key, K) = anchor_rfn * 1.96^(anchor_K - K)

S_t = exp( mean over levels with kld > 0 of log(kld / rfn_actual^alpha) )      (OP 72-79)
kld(t, K) = S_t * rfn_at(t, max(K, aK))^alpha * ( (rfn_at(t, K) / rfn_at(t, aK))^alpha_low  if K < aK )   (OP 106-112)
alpha_low = --alpha_low, default alpha
```

Note (inference): `S_t` is fitted against the numel-weighted realized rfn of a group, while the group's anchor is
the unweighted rms of its members, so predictions at the anchor K differ slightly from the measured level-0 KLD for
groups whose members have different rfn.

DOC's recommended invocation: `-al 2.0` (fitted values run ~1.7-1.9; real-quant transitions measure ~2.0). The fit
on Dino0040's file (computed by me with plain Python from the JSON): min -0.38, median 1.86, max 3.61.

### 4.2 Problem

```
minimize    sum_t kld(t, K_t)
subject to  sum_g numel_g * K_g <= budget,   budget = int(bitrate * sum_numel),  sum_numel = sum over "bits" results
            K_g in {min_k, min_k + step, ..., max_k}
            every tensor of a tie group g has the same K
```

- `-b/--bitrate` (required): mean bits per weight over the **budgeted (non-head) tensors only**; head, embedding,
  norms and vision are outside the sum and outside the average (same accounting as `convert.py --bits`).
- `-hb/--head_bits` (int, default 6): only copied into the recipe.
- `-mink` (float, default 1), `-maxk` (float, default 8), `-ks/--k_step` (1 or 0.5, default 1).
- `-tie` (default `q_proj+k_proj+v_proj,gate_proj+up_proj,in_proj_qkv+in_proj_z`): tensors whose leaf name is in one
  set and that share a parent key form one group (OP 118-130). With grouped measurements (`.qkv`, `.gate_up`) the
  leaf names don't match, so every result is its own group and the measurement grouping does the tying.

### 4.3 Algorithm (OP 135-216)

```
groups in order of first appearance in results (= module order)
K_g = min_k for all;  spent = min_k * sum_numel;  assert spent <= budget
heap = [(-(kld_g(min_k) - kld_g(min_k + step)) / numel_g, g, min_k + step) for g]   if min_k < max_k
while heap:
    (negd, g, k_next) = pop smallest              # largest gain per weight; ties: smaller g first
    if spent + numel_g * step > budget: continue  # drop g for good
    K_g = k_next; spent += numel_g * step
    if k_next < max_k - 1e-9: push (-(kld_g(k_next) - kld_g(k_next + step)) / numel_g, g, k_next + step)

repeat up to 100 times:                           # exchange repair (OP 172-212)
    leftover = budget - spent
    cands = [(loss_j / (numel_j * step), loss_j, numel_j * step, j) for j with K_j > min_k + 1e-9]
            loss_j = kld_j(K_j - step) - kld_j(K_j);   sorted ascending
    best = None
    for g with K_g < max_k - 1e-9:
        gain = kld_g(K_g) - kld_g(K_g + step);  need = numel_g * step - leftover
        take cands in order (skipping j == g) while need > 0: demote += [j]; loss += loss_j; need -= numel_j * step
        if need > 0: skip
        if gain - loss > 0 and better than best: best = (gain - loss, g, demote)
    if best is None: break
    K_g += step; spent += numel_g * step; for j in demote: K_j -= step; spent -= numel_j * step

achieved_bpw = spent / sum_numel
predicted_kld = sum_t kld(t, K_t)
```

The printout also gives an "ends-first baseline" (OP 218-234): K = floor(bitrate) everywhere, then +1 for single
tensors in order of distance to the nearer end of the stack while the budget allows. Diagnostic only.

### 4.4 Recipe YAML (OP 260-282)

Written by hand, line by line (Python `str()` of each value), parsed with `yaml.safe_load` by convert:

```yaml
# Quantization recipe generated by sc_optimize.py
model: /workspace/gemma-hf
measurement: noise_attrib.json
target_bpw: 3.0
achieved_bpw: 2.9999                 # round(., 4)
predicted_kld: 0.050253              # round(., 6)
head_bits: 6
tie: q_proj+k_proj+v_proj,gate_proj+up_proj,in_proj_qkv+in_proj_z
tensors:
  model.language_model.layers.0.self_attn.q_proj: 2
  model.language_model.layers.1.mlp.gate_proj: 2.5
  ...                                # every member key of every budgeted target; int if integral, else float
```

Only `tensors`, `achieved_bpw`/`target_bpw` and `head_bits` are read by convert.

Pitfalls:
- `-ks 0.5` with the default `-maxk 8` can emit 4.5, 5.5, ...; convert accepts only 1.5 / 2.5 / 3.5. Dino0040's
  README: use `-ks 0.5 -maxk 4`.
- `-t/--table` calls `range(args.min_k, args.max_k + 1)` with float arguments (OP 297, `-mink`/`-maxk` are
  `type=float`), which **(inference)** raises `TypeError` in Python; the table option looks broken as written.

### 4.5 Dino0040's 3.00 bpw recipe (`-b 3 -hb 6 -al 2.0 -mink 2 -ks 0.5 -maxk 4`)

Budget-weighted mean 2.99991 over 410 tensors. Histogram (tensors, share of weights): K 2: 63 (12.0%), 2.5: 68
(16.1%), 3: 154 (37.1%), 3.5: 107 (29.7%), 4: 18 (5.2%). Mean K by type: q/k 2.81, v 2.80, o 2.92, gate/up 2.95,
down 3.31. Layer 0 is entirely at K = 2 (the U-shape flattens, as DOC 24-28 says); the middle (layers 30-40) and the
last two layers get the most. Share of level-0 KLD in the measurement: gate_up 41.8%, down 32.1%, qkv 16.5%, o
9.6%; by depth (layers 0-9 / 10-19 / ... / 50-59): 4.7 / 10.1 / 16.9 / 26.6 / 19.6 / 22.1%.

---

## 5. `convert.py` with a recipe

### 5.1 Flags

- `-rcp/--recipe FILE` (CM 78, 206-235): YAML must have a non-empty `tensors` mapping; each value an integer 1-8,
  16 (stored unquantized), or one of `HALF_RATES = (1.5, 2.5, 3.5)` (ALLOC 11); bools rejected.
  `bits` (for reporting) = `achieved_bpw` or `target_bpw`; explicit `--bits` then only warns. `head_bits` default =
  recipe `head_bits`, else 6; an explicit `--head_bits` wins (CM 278). `--hq` is ignored with a warning (local).
  Half rates require `--codebook mul1` (the default) (CM 306-317). The tensor map is stored in the job's `args.json`
  as `recipe_strategy`, so a resumed job ignores later edits of the file (CM 319-321).
- `-hb/--head_bits`, `-mb/--mtp_bits` accept 1-8, 16, or a half rate with mul1 (CM 79-80, 308-313).
- `-cd/--cal_data` (section 1.7).

### 5.2 Strategy

`create_q_strategy_from_recipe` (ALLOC 209-277): walk every Linear with a qmap; `qbits_key == "bits"` takes
`recipe[key]` (missing key -> error); `head_bits` -> head_bits; `mtp_bits` -> mtp_bits; vision -> fixed vision
bits. Keys in the recipe that match no budgeted tensor -> error. Reported `final_bits = sum(numel * K) / sum(numel)`
over recipe tensors with `K <= 8` (16-bit entries are excluded from both sums), rounded to 2 decimals (CM 1613).

Per tensor the only change is `quant_args["K"] = strategy[key]` (CM 532-552, 751, 819-850); the seed is still the
module index, every other step of the quantizer is K-agnostic except the trellis encoder, the scale-search drift
factor and packing (section 6).

### 5.3 Hessian groups and K

Tensors sharing a qmap share one captured H, finalized once (one `su` per qmap). They **need not** share K:
`group_quant_linears` (CM 738-785) batches tensors only when `(qmap, K, in_features)` match, so q at 2.5 and k at 3
read the same H and run separate LDLQ passes. What breaks with mixed K is only fusion at inference: the qkv sliced
bundle and k/v pair need equal K (attn.py 463, 517), gate/up fusion needs equal K (mlp.py 766). Hence the grouped
measurement and the tie groups. exl3conv already quantizes q/k/v side by side from one H, so per-tensor K is
plumbing: pass K per linear instead of one global K.

### 5.4 Metadata written

`config.json` → `quantization_config` (COMP 194-240):

```json
{"quant_method": "exl3", "version": "<__version__>", "bits": <final_bits>, "head_bits": <head_bits>,
 "calibration": {"rows": <cal_rows actually used>, "cols": <cal_cols>},
 "out_scales": "auto"|"always"|"never", "codebook": "mul1"}      // + "mtp_bits", "vision_bits" when present
```

No recipe path, no `cal_data` path, no per-tensor K, no flag for "mixed". Dino0040's 3.00 bpw:
`bits: 3.0, head_bits: 6, calibration {250, 2048}, out_scales auto, codebook mul1, version 1.6.1`.

`quantization_config.json` (QC 19-67) = that dict plus `tensor_storage`: per leaf module its stored tensors
(shape, n_bytes, torch dtype name), and for EXL3 linears `"quant_format": "exl3"`, `"bits_per_weight":
trellis.shape[-1] / 16` written as int if integral else float (2.5), and `"mul1_multiplier": 2212286765`. This is
the only place per-tensor K appears besides the trellis shape itself. Example (Dino0040):
`layers.1.mlp.gate_proj.trellis` shape `[336, 1344, 40]` int16, `bits_per_weight: 2.5`; `layers.2.mlp.down_proj`
`[1344, 336, 56]`, 3.5; `lm_head.trellis` `[336, 16384, 96]`, 6.

### 5.5 Default allocation also uses half rates (no recipe)

Since `07b8a2e0`, `create_q_strategy` is called with `half_steps = (codebook == "mul1")` (CM 1607-1611):
`rate_floor(b) = floor(2b)/2` for `b < 4` else `floor(b)`, and promotions step +0.5 below 4 bpw (ALLOC 18-26).
So `-b 2.5` gives K = 2.5 everywhere and `-b 3.25` gives K = 3 plus ends-first promotions to 3.5. `-b 3.0` and
`-b 4.0` stay all-integer. Modules can opt out with `q_half_bits = False` (only DSV4 attention does, dsv4.py).

---

## 6. Half-integer K (1.5, 2.5, 3.5)

### 6.1 Which rates exist

- Loader/inference: `bits_from_K` accepts integer 1..8 or `bits + 0.5` with `bits <= 3` (BK 13-20). So
  **1.5, 2.5, 3.5 only; 4.5 and up do not exist.** `HALF_RATES = (1.5, 2.5, 3.5)` in convert.
- Encoder instances: `quantize_tiles_frac_kernel<KA, 0xAAAA>` for KA = 1, 2, 3 (comp_units/quantize_tiles_frac_inst.cu,
  QCU 288-294). Pack/unpack accept KA 1..7 and any MASK with an even bit count per 16 (FRAC 88-97), but nothing
  else is instantiated or loadable.
- mul1 codebook only (EXL3 81 asserts it at load; Q 141-146; RC checks).
- The mask is not stored anywhere: `K = trellis.shape[-1] / 16`; a non-integer K means MASK = 0xAAAA.

### 6.2 Bit layout

`K = KA + 0.5`, `D(i) = KA + ((0xAAAA >> (i & 15)) & 1)`: **even ring positions take KA bits, odd take KA + 1.**
`S(i) = D(0) + ... + D(i)`; ring length `R = S(255) = 128 (2 KA + 1) = 256 K` bits = `16 K` uint16 = `8 K` uint32
(24, 40, 56 uint16 for 1.5, 2.5, 3.5).

- State (16-bit codeword) of position i = ring bits `[S(i) - 16, S(i))` mod R, read MSB first (FRAC 9-44). Its low
  `D(i)` bits are position i's own bits; its top `16 - D(i)` bits are the low bits of the previous state:
  `s_i = ((s_{i-1} << D(i)) | new_i) & 0xFFFF`, circularly (tail-biting).
- Packing (FRAC 49-69): for i = 0..255, write the low `D(i)` bits of `s_i`, MSB first, at ring positions
  `S(i) - D(i) .. S(i) - 1`. Ring bit p lives in 32-bit word `p >> 5` at bit `31 - (p & 31)`. Words are stored
  little-endian into the int16 tensor: `uint16[2j]` = low half of word j, `uint16[2j+1]` = high half.
- Same convention as integer K (where `D(i) = K`, `S(i) = (i+1) K`), so integer packing is the special case
  MASK = 0. exl3conv's `src/exl3/pack.hpp` (`Rate`, `pack_tile`, `unpack_tile`, `is_tail_biting`) already
  implements this layout for odd `K2`.
- Tile element order: unchanged. Ring position i holds row-major tile element `perm[i]`,
  `perm[8t + j] = 16 r + c`, `r = 2 (t % 4) + (j & 1) + 8 ((j >> 1) & 1)`, `c = t / 4 + 8 ((j >> 2) & 1)`
  (Q 53-76, FRAC 40-47 `frac_perm`).

### 6.3 Tensors in the file

Identical set to integer K: `<key>.trellis` int16 `(in/16, out/16, 16 K)`, `<key>.suh` fp16 `(in)`, `<key>.svh`
fp16 `(out)`, `<key>.mul1` int32 scalar shape `[]` = 0x83DCD12D (2212286765), optional bias. Tile `(a, b)` covers
`W_hat[16a .. 16a+15, 16b .. 16b+15]`.

### 6.4 CPU reference decoder (bit-exact)

```
def decode_tile(w16[16K]):                                # one tile's uint16 words
    R = 256 K;  word[j] = w16[2j] | (w16[2j+1] << 16)
    bit(p) = (word[(p mod R) >> 5] >> (31 - ((p mod R) & 31))) & 1
    S = 0
    for i in 0..255:
        S += D(i)
        s = 0; for b in 0..15: s = (s << 1) | bit(S - 16 + b)
        v = mul1(s)                                       # fp16
        tile[perm[i]] = v                                 # row-major 16x16
    return tile

mul1(s):  x = (s * 0x83DCD12D) mod 2^32
          sum = 0x6400 + byte0(x) + byte1(x) + byte2(x) + byte3(x)       # dp4a(x, 0x01010101, 0x6400)
          h = fp16 with bit pattern (sum & 0xFFFF)
          return fma16(h, fp16(0x1EEE), fp16(0xC931))                  # one rounding (CB 36-52)

W = diag(suh) · H128_blockwise · W_hat · H128_blockwise · diag(svh)     # H128 = Sylvester / sqrt(128)
```

This is `TL` `unpack` + `decode` + `dequant_rotated` + `dequant` (TL 85-176), with the half rate handled only by
`step_widths`. The CUDA decode (`dq8_half`, DQ 254-285) reads 8 consecutive ring positions per lane
(`t_offset = lane * 8`), finds the end bit of window 7 as `((t_offset >> 1) + 4) (2 KA + 1)` (+ one ring for the
wrap), takes 32 bits ending there with one funnel shift, and derives windows 6, 5, 4 by shifting right by
`KA + 1`, `KA`, `KA + 1` (likewise windows 3..0 from window 3's end, `2 (2KA+1)` bits earlier). Four windows span
`18 + 3 KA <= 27` bits, so they lie within two consecutive 32-bit words and one funnel shift of that pair yields all four. Same values as the reference above.

### 6.5 Encoder (QTF 103-233)

The same tail-biting Viterbi as integer K (notes/exl3-algorithm.md §4.3), with two differences: the node width
changes with position, and the trellis step at position p has `KIN = D(p)`, `KOUT = D(p + 1)` (indices mod 256;
since 256 is a multiple of 16, `D(256) = D(0) = KA`).

```
def quantize_tile_frac(w[256] float32, KA):              # ring order (already permuted by perm)
    wh = fp16(w)
    D(i) = KA + ((0xAAAA >> (i & 15)) & 1)

    def step(p, first, pre):                             # QTF 27-101
        KIN = D(p); KOUT = D(p + 1); Kr = 16 - KOUT
        for e in 0 .. 2^Kr - 1:                          # out-node = low 16-KOUT bits of s_p
            for k in 0 .. 2^KOUT - 1:
                s = (k << Kr) | e;  pred = s >> KIN      # in-node = top 16-KIN bits of s_p
                d = fp16(mul1(s) - wh[p])
                if first: c = fp16(d * d); if pre >= 0 and pred != pre: c = +inf
                else:     c = fma16(d, d, cost_prev[pred])
                if k == 0 or c < best: best = c; arg = pred          # strict <: lowest k (= lowest pred) wins
            cost[e] = best; hist[p][e] = arg
        cost_prev = cost

    def forward(roll, pre):
        for t in 0..255: step((t + roll) mod 256, first = (t == 0), pre)
        # pass 1 needs hist only for p < 128 (QTF 159-161)

    def argmin_end():                                    # over 2^(16 - D(0)) nodes (QTF 123, 177-203)
        key(e) = (half_bits(cost_prev[e]) << 16) | rank(e)
        rank(e) = (brev5((e & 1023) >> 5) << 10) | (brev5(e & 31) << 5) | (e >> 10)
        best = min key; return 0 if best >= 0x7C000000 else the e with rank(e) == best & 0xFFFF

    def backward(roll, write, edge):                     # QTF 205-227
        for t in 255 down to 0:
            p = (t + roll) mod 256
            pe = hist[p][edge]
            s  = ((pe << D(p)) | edge) & 0xFFFF
            edge = pe
            if write: idx[p] = s; out[p] = float(mul1(s))
            elif p == 0: break
        return edge

    forward(128, -1)
    e_end = backward(128, False, argmin_end())           # in-node of position 0 on the best pass-1 path
    forward(0, e_end)                                    # position 0 constrained to in-node e_end
    backward(0, True, e_end)                             # out-node of 255 = in-node of 0: ring closes
    return idx (uint16 states, ring order), out (decoded, float)
```

Details that keep it bit-identical with the CUDA kernel:

- fp16 costs with fused fp16 FMA, the same as integer K. The CUDA step evaluates out-nodes in pairs with `half2`
  arithmetic, which does not change any value.
- Node count per step: `2^(16 - D(p+1))` out-nodes, `2^D(p+1)` branches each; always 65,536 candidates per step.
  History stores the full predecessor (int16). CUDA scratch: `temp_costs (128, 2, 65536 >> KA)` fp16,
  `temp_edges (128, 256, 65536 >> KA)` int16 = 2 GiB / 1 GiB / 512 MiB for KA = 1 / 2 / 3, one 512-thread block per
  tile, 128 tiles per launch (Q 131-149, QCU 255-310). **(inference)** For fixed out-node e the predecessor is
  monotonic in k, so storing only the branch k (<= 4 bits, one byte) and rebuilding
  `pred = ((k << (16 - KOUT)) | e) >> KIN` gives the same paths with 1 byte per node, as exl3conv's integer
  encoder already does.
- The scale search runs this encoder on sampled tiles multiplied by `LDLQ_DRIFT[K]` = 1.035 for 1.5, 1.009 for 2.5,
  1.0 for 3.5 (Q 1181-1190; exl3conv `quantize.hpp` 168 has only integer entries). Everything else in the
  per-tensor quantizer (finalize, regularize, LDLQ, refit) is unchanged.
- Packing: `pack_trellis` uses `trellis_words(K) = 16 K` words and `pack_trellis_frac(KA, 0xAAAA)` (Q 1155-1170).

Acceptance tests to port (TQF): every output is tail-biting at the alternating widths and `out` equals
`mul1(idx)` exactly; MSE on unit-Gaussian tiles below 0.2 / 0.05 / 0.013 for 1.5 / 2.5 / 3.5 and below 0.75 x the
integer rate KA's MSE on the same tiles; a decoded random valid codeword is re-encoded exactly; results independent
of batch composition.

---

## 7. `--hq` on Gemma 4

`--hq` raises a tensor's floor to `rate_floor(base + select_hq_bits)` (ALLOC 109, 187-198). Gemma 4 sets
`select_hq_bits = 2 if use_moe else 0` on attention and `1 if use_moe else 0` on the dense MLP / shared expert
(G4 259, 295, 310). **On Gemma 4 31B (dense) `--hq` promotes nothing.** On the MoE variant it lifts attention by
2 bits and the shared-expert MLP by 1. With `--recipe` it is ignored locally (upstream: MTP layers only). Priorities
in the default allocation are attention `q_priority = 2`, MLP `1` (attn.py 200, sliding_attn.py 293, mlp.py 164).

---

## 8. Reported results and claims

DOC:
- Shaped-noise predictions vs ground-truth swap attribution of a real conversion: Spearman 0.84 (iid 0.61); iid
  overestimates by up to ~6x (v_proj worst), median 1.95x (ME 34-38), residual error "nearly uniform across tensors".
- rfn at fixed K: about ±10% spread with structure by module type.
- alpha fitted ~1.7-1.9, real-quant transitions ~2.0, use `--alpha 2.0`; `alpha_low` ~2.35 for K2→K1 on
  Qwen3.8-27B.
- Measurement on a ~27B dense with two levels: ~35 minutes on a fast GPU; full run (trace, measurement,
  optimization, conversion) ~2 hours, mostly conversion; 27B BF16 measurement "fits comfortably in 96 GB".
- Self-sampled-trace sensitivities transfer better than raw-corpus ones, "including to domains absent from the
  trace". Evaluate on a disjoint trace (`eval/qbench_prompts.py`, `--tool_frac 0.3` default).
- No KLD numbers in DOC itself.

Dino0040 (Gemma 4 31B-it, qbench vs BF16 on its held-out `qbench_prompts_gen.json`, 29 rows, 39,814 context +
47,547 response tokens, generated by the same 6 bpw model; all quants 3.00 bpw):

| 3.00 bpw quant | Mean KL | Median KL | p90 KL | PPL |
|---|---|---|---|---|
| bf16 reference | — | — | — | 1.1830 |
| noise floor | 0.0006 | ~0 | 0.0016 | 1.1831 |
| turboderp's uniform `3.00bpw_mul1` (July 2026, v0.0.43) | 0.0561 | 5.5e-5 | 0.119 | 1.2497 |
| uniform, v1.6 | 0.0528 | 5.4e-5 | 0.115 | 1.2464 |
| recipe | 0.0381 | 3.5e-5 | 0.082 | 1.2276 |
| **recipe + trace calibration** | 0.0301 | 2.7e-5 | 0.059 | 1.2187 |

Claimed: quantizer update -6% mean KL, recipe vs uniform -28%, trace calibration on top -21%, total -46%. Cost:
measurement ~1 h with ~120 GB VRAM on GB10 cloud time; conversion ">20 GB VRAM, an hour or two". The README
recommends `-mink 2` (otherwise one tensor drops to 1 bit). The recipe's own `predicted_kld` is 0.0503 (an
all-position KL on 10 trace rows, so not directly comparable with the qbench numbers).

---

## 9. Porting notes

### (a) Self-sampled calibration, minimum set

1. **A trace generator.** Not part of the converter. It needs a Gemma 4 31B that runs on the B70 and the exact
   recipe of section 1: the 157 conversations, chat template with `enable_thinking=true` (Gemma's template has no
   `reasoning_effort`; the probe in 1.3 drops it if the template rejects it), sampler min_p 0.08 then temperature
   0.8, `max_new_tokens` 3072, turn-major waves, follow-ups re-templated with the cleaned previous answer, stop at
   512,000 tokens, shuffle with seed 0, write `input_ids` int64 [250, 2048] + `lengths` int64 [250]. BF16 31B
   (62 GB) does not fit on one 32 GB card, so in practice the trace comes from an EXL3 quant through the B70 servers
   (`b70 serve`), the same choice as Dino0040's 6 bpw. Python's `random.Random(0).shuffle` order is only needed to
   reproduce the file bit for bit; any fixed shuffle is equivalent.
2. **In exl3conv:** a `--cal-data FILE` reader with the rules of 1.7, and variable-length rows. **(inference)** The
   simplest way keeps the fixed `[250, 2048]` state and masks: text attention in Gemma 4 is causal, so tokens after
   a row's length cannot affect earlier positions; run the padded rows unchanged and accumulate H only over
   positions `< lengths[r]` (H count = total valid tokens). The state shrinks only if stored ragged (443,537 tokens
   x 5376 x 4 B = 9.5 GB vs 11 GB).
3. No BOS handling: rows already start with BOS.

### (b) Per-tensor allocation with integer K only

1. **Measurement (`sc_measure --shaped` equivalent).** Pieces exl3conv already has: clean Gemma 4 forward,
   Hessian capture, `finalize_capture_H` (damping, su, blockwise H128, block-16 LDL). New: final norm + `lm_head`
   forward with logit softcap; full-vocab KL(ref ‖ perturbed) per position; storage of 10 x 1024 x 262144 reference
   logits (5.4 GB, host); a Gaussian RNG; `L^T x = eta` triangular solve (oneMKL `trsm`) and the noise of 2.4; the
   fan-out scheduler of 2.8; resume from partial JSON. Measure the same grouped targets (qkv, o, gate_up, down,
   lm_head). With q/k/v and gate/up tied, Gemma 4 needs only 4 K values per layer, and the qkv bundle stays fused
   at inference.
2. **Anchors.** Either log `rfn` per tensor in exl3conv and run one uniform K = 2 conversion, or skip the probe and
   use the global anchor `2:0.292` (loses the ±10% per-tensor spread). Dino0040's `rfn_2.00bpw.json` and
   `noise_attrib.json` are for google/gemma-4-31B-it itself and can be reused directly for that model;
   fine-tunes (Twilight, Wicked-Stuff) need their own measurement **(inference: sensitivities of a merge may differ)**.
3. **Optimizer.** Port section 4 exactly (pure CPU, milliseconds). With `-ks 1 -mink 2 -maxk 8`.
4. **Converter.** Read the recipe (section 5.1-5.2), set K per linear (encoder shapes K = 1..8 already exist),
   drift factor and packing per K, `quantization_config.bits = round(sum(numel K) / sum(numel), 2)`,
   `tensor_storage[..].bits_per_weight` per tensor. The recipe's `head_bits` needs quantized heads, which exl3conv
   does not have yet (head stays 16 bits; outside the budget either way).

### (c) What half-integer K adds

The encoder variant of 6.5 (alternating node widths; branch byte history), the K = 1.5 / 2.5 drift factors, packing
and unpacking (already in `pack.hpp`), the decoder in the advance pass and in `exl3eval` (6.4), `bits_per_weight`
as a float, and `-ks 0.5 -maxk 4` in the optimizer. mul1 only. Dino0040's recipe puts 46% of the weights on half
rates, so most of its gain at 3.00 bpw presumably depends on them **(inference)**.

### What does not fit layer-streaming on 32 GB

- Generating the trace from the **BF16** model: it has to run the whole model for autoregressive sampling; on one
  B70 that means a quant (6 bpw EXL3 ~23 GB plus KV cache) or impractically slow layer-streamed generation.
- Everything else streams: the measurement (largest resident item: `lm_head` 2.8 GB fp16 or the `down_proj` L
  factor 1.85 GB, plus pending states), the probe (one module of each model at a time), the optimizer (CPU), and
  the conversion as today. The cost of streaming the measurement is repeated layer loads (~270 for 8 passes), not
  feasibility.
