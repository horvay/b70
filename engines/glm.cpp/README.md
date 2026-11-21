# glm.cpp

GLM-5.3-Flash (EXL3 sc_2.60bpw, GM LoRA) on the Arc Pro B70 in C++ and SYCL, with no PyTorch and no Python. It
replaces the PyTorch-XPU stack in `llm/glm/xpu`. That stack stays `b70 serve`'s default until the client has run on this engine;
`GLM_ENGINE=cpp b70 serve <model>` (or `GLM_ENGINE=cpp` in `~/.config/b70/serve.env`) serves with `build/glm_server`.

`docs/forward-spec.md` describes exactly what the PyTorch stack computes: every op, dtype and rounding point, with
file:line references. It is the reference for this code.

## Build and run

```
./build.sh                  # DPC++ 2025.3 (the toolchain sd-cli uses), JOBS=3; kernels JIT for now
b70-lease run me -- build/decode_check <model dir> <store .bin> <ref prefix> [steps] [expert VRAM GiB]
```

Every GPU run goes through the b70q queue. `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` picks the B70.

## Layout

| File | What |
|---|---|
| `src/rt.*` | device, in-order queues (decode high priority, `low` for background work), tensors, workspaces |
| `src/st.*`, `src/json.h` | safetensors (mmap) and JSON (insertion-ordered objects, `dump` = Python `json.dumps`) |
| `src/tokenizer.*` | the HF BPE: byte-level (PCRE2 pre-tokenizer; GLM, Qwen) and SentencePiece-style with byte fallback (Gemma), picked from `tokenizer.json`; exllamav3's streaming detokenizer; host only |
| `src/chat.*` | the chat template ported by hand, and serve.py's prompt rendering around it; host only |
| `src/exl3.*` | EXL3 linear: port of `exl3_gemm_small`, same ESIMD kernels (`kernels/exl3_esimd.h`); prompts in int8 through oneDNN (section "EXL3 int8 prompt linears") |
| `src/ops.*` | RMSNorm, mHC mix / apply / mean, clamped SwiGLU, fp16 GEMV, exact router |
| `src/kda.*` | KDA linear attention (recurrent; conv, gates, delta rule, gated norm) |
| `src/mla.*` | MLA + DSA indexer, decode graph path (gathered attention, top-512 pools by radix select) |
| `src/moe*`, `kernels/exl3_moe_kernels.h` | routed experts (kernels extracted from `exl3_moe.n128.sycl`), routing alignment |
| `src/tier.*` | expert tier: VRAM / RAM (memfd, read in place) / NVMe, reader threads, prefetch, admission, prompt staging |
| `src/lora.*` | PEFT LoRA adapters (the GM LoRA): runtime deltas, merged fp16 weights, shared A projections |
| `src/model.*` | the trunk: embedding, 45 blocks, final norm, lm_head; layer-major prefill, KDA checkpoints, decode graphs |
| `src/sampler.*`, `src/engine.*` | ComboSampler's order of penalties and filters; a request: prefix reuse, prefill, decode |
| `src/sample_gpu.*` | sampling's selection on the card for `G4Engine` (section "Sampling on the card") |
| `src/http.*`, `tools/glm_server.cpp` | the OpenAI-compatible server, a drop-in for `serve.py` (same routes, arguments, SSE) |
| `tests/` | `exl3_check` (EXL3 vs a CPU trellis decoder), `exl3_i8_check` (every EXL3 linear of a model, int8 prompt path vs fp16: error and time), `decode_check` (vs `dumpref.py`), `acts_check` (per layer vs `dumpacts.py`), `tok_check` (tokenizer, stream decode, prompts vs `tok_ref.py`; CPU only), `gemma_check` (Gemma's tokenizer, stream decode, prompts vs `gemma_ref.py`, its parsers vs cases worked out from SGLang's source; CPU only) |
| `src/g4*`, `kernels/g4_*.h`, `src/chat_gemma.cpp` | Gemma 4 31B dense (`gemma4`, e.g. Gemma 4 31B-based): the model, several games at once (`g4_engine`), its drafter, chat template and parsers (below) |
| `src/rp*`, `kernels/rp_dec.h`, `src/chat_ripple.cpp` | Ripple-65B (`ricursive`): the model, its DFlash2 drafter, several games at once, its chat template and parsers (section below) |
| `src/q4*`, `src/chat_qwen.cpp` | Qwen3.8-Flash-Next (`qwen4_exp`, e.g. Swift 1.5): the model, its kernels (`q4_ops`), its chat template (below) |
| `src/q35*`, `kernels/q35_*.h`, `tools/q35_gen.cpp` | Qwen3.5 / Qwen3.8 dense (`qwen3_5`: 27B, the 40B Ultimate-Heretic): Gated DeltaNet + gated attention, its vision tower, MTP drafting (`src/q35_blocks.h`: the draft head's blocks), prompt and decode kernels (`docs/qwen35-spec.md`, sections below) |
| `src/e0*`, `src/gguf.*` | Edge0-145B (a latent-MoE Qwen3.8-Flash-Next, NVFP4 experts) from its release folder: `docs/edge0-spec.md` |
| `src/chat_gemma.cpp` | Gemma 4 (`gemma4`) text: its chat template ported by hand with the client's prefill hook, SGLang's message shaping, its `gemma4` reasoning split (`GemmaSplitter`) and tool-call parser; host only |
| `tools/` | `prio` (queue priority), `gen_bench` (free-running decode), `launch_bench`, `linear_bench`, `read_bench` (NVMe), `udmabuf_test`, `i8_eval` (the int8 prompt path against fp16 on a whole model: KL, NLL, speed) |

## Parity with the PyTorch stack (2026-10-31)

Base model without LoRA, exact routing (`GLM_XPU_RES_BONUS=0`):

- **EXL3 linear** vs an independent CPU trellis decoder: relative error 2-5e-4 for K=4 and K=6, M=1..64 (fp16 output
  rounding).
- **Every sublayer on the reference's own inputs** (`acts_check`, one token):
  - routed and dense MLPs: bit-exact, the same 8 expert picks;
  - KDA attention: 0 on most layers, at most 2e-4;
  - MLA: 2e-5 to 1e-4;
  - the mHC mix: at most 5e-5.
- **Whole decode, 64 forced tokens** (`decode_check`): mean NLL of the next tokens 4.073 against the reference's
  4.055. Top-1 agreement 86%, mean KL 0.12.

**The KL is the model's own sensitivity, not an engine error.** The trunk amplifies 1e-5 differences in summation order
(through bf16/fp16 roundings) until a near-tie in expert routing flips. The PyTorch stack does the same to itself:
switching only its mHC mix from Triton to torch moved that token by KL 0.76 and changed its top-1. Compare
implementations by NLL on real text, not by KL against the other stack.

Runs are deterministic.

## Queue priority (tools/prio.cpp)

The B70 honors SYCL queue priority. A low-priority queue does not slow a high-priority decode loop (−1%). It also got
no work done beside it, even though the GPU idled 20% of each token: the scheduler does not hand short gaps to the low
queue. Background work only runs when decode is idle.

## With a real prompt (2026-10-31)

The client prompt (`run/nqfake/tiny-prompt.txt`, 613 tokens), then 200 forced tokens (`force.txt`); the references come from
`dumpref.py --prompt-file` (exact; `--allow-bonus` for quality mode):

| | KL vs the PyTorch stack | NLL engine / reference | Prompt | Decode |
|---|---|---|---|---|
| Exact routing, prompt in 8-token pieces | 0.0067 | 1.5293 / 1.5296 | 47 s | 173 ms/token |
| Exact routing, prefill | 0.0087 | 1.5337 / 1.5296 | 15.1 s | 205 ms/token |
| Quality mode (bonus 0.1), prefill | 0.0154 | 1.5137 / 1.5353 | 16.9 s | 100 ms/token |
| Quality mode, no prefill, 8-row pieces | 0.033 | 1.5274 / 1.5353 | | 88 ms/token |

The PyTorch stack's own quality mode is KL 0.021 from its exact mode. It swaps 12.9% of picks; this engine swaps 11.4%.
Exact mode is about as close to the PyTorch stack as that stack's graph decode is to its cache-less forward (0.0075).
The prefill's small rise comes from its own roundings (DPAS expert GEMMs, fp32 mHC weights, fp16 GEMM linears), which
the PyTorch prefill also has.

### Expert tier (`src/tier.*`)

A port of `glm_xpu/tier.py` with its native reader / LRU / decision core:
- exclusive VRAM / RAM residency;
- a 14 GiB RAM tier of memfd pages: NVMe reads land in it with O_DIRECT, and the GPU reads it in place through
  udmabuf dma-bufs imported into Level Zero (64 MB windows; `tools/udmabuf_test`). `GLM_MEMFD=0`: pinned USM with a
  bounce copy per read;
- 16 O_DIRECT reader threads, urgent reads (16 pieces) ahead of speculative ones (8 pieces, at most 12 threads);
- next-layer prefetch (top 12 predicted);
- frequency-decayed admission (2 per layer), with write-back of the victim;
- the 0.1 residency bonus.

Prompts stage each MoE layer's experts once for all rows. VRAM experts are used in place; RAM and NVMe experts go into
staging slots borrowed from the 3.5-bit class, whose pool gets at least 288 slots as in the PyTorch stack. A pinned
4.5 GiB ring reads the next layers' non-resident experts ahead; it is allocated at load (~3 s to pin and touch;
`GLM_RING_AT_LOAD=0`: on the first prompt). The prompt's per-row buffers (residual streams and MLP inputs, 72 KB a row:
1.4 GB at 20K tokens) are lent from the same pool (`Tier::lend`) and returned after it.

Prompts of 1152+ tokens (32 picks per expert) use nearly every expert of every layer, so they borrow a second set of
288 staging slots: the copy queue fills layer L+1's set with every non-VRAM expert whose bytes are ready (RAM tier,
landed NVMe reads) while layer L computes, and again while L+1's router runs; `stage()` copies what is left and puts
one barrier on the main queue (`Tier::prestage`, `GLM_PROMPT_DBL=0`: one set, copied on the main queue at each
routing). Bit-identical logits; at 8K tokens it took the staging copies (2.8 s, on the main queue before) off the
critical path. The dropped VRAM experts come back through admissions within a few decode tokens.

MLA for prompts (more than 16 rows, `GLM_MLA_GEMM=0`: the decode kernels): absorb, unfold and the indexer scores are
oneMKL fp16 GEMMs (fp32 accumulation); attention gathers each row's 2051 selected latent rows (8 rows per block, 16 MB,
which stays in L2; `GLM_MLA_RA`) and runs scores and the weighted sum as batched GEMMs, with the same fp16 roundings as
the decode kernels. 8K-token logits vs the per-row kernels: KL 0.00001, same top 5.

VRAM outside the expert pool is a fixed 3 GiB reserve (`GLM_VRAM_RESERVE_GIB` in `glm_server`), and decode's graphs
take ~1.1 GiB of it (fdinfo: 29.71 GiB after load, 31.36 after the first decode). With the prompt buffers allocated
beside the pool, a 6.9K-token prompt after a decode went to 31.79 of 31.9 GiB and the GPU never finished its next
kernel: no fault, no reset, the server stuck in `queue.wait()` (2026-11-02). Lent from the pool, the server stays at
31.39 GiB through 6.9K and 19.5K-token prompts. `GLM_PREFILL_TRACE=1|2|3` prints prefill's progress per layer, step or
`prof_point` (each waits), `GLM_PREFILL_CHUNK` sets the rows per piece; `tools/prefill_bench` runs prompts alone.

### Decode profile (`GLM_PROF=1`, quality mode, ms per token)

| Phase | Before the fixes | Now |
|---|---|---|
| Router | 24.9 | 4.8 (routing tail was one work-item; now a group-max per round) |
| KDA | 13.0 | 12.2 (delta rule on sub-group shuffles) |
| Expert kernels | | 33-36 (AOT made no measurable change) |
| Tier (host, NVMe waits included) | | 23-28 |
| MLA | | 8.8 |

## Speed (2026-11-01, GM LoRA, quality mode)

| | Engine | PyTorch stack |
|---|---|---|
| Free-running 600 tokens, mean of tokens 100-600 (3 runs each) | 94.7 ms/token | 95.9 ms/token |
| Same, first 100 tokens | 119 ms/token | 194 ms/token |
| the client prompt + 200 forced tokens (`decode_check`) | 94.4 ms/token | 120.5 ms/token |
| 6.9k-token context, exact routing, forced | 184.9 ms/token | 267 ms/token |

Since then, on by default: the experts of the picks that are already readable run before a layer waits on NVMe, the
late ones after, combined in pick order (`moe_split_begin`; bit-identical; `GLM_MOE_SPLIT=0` turns it off). 97.2 -> 94.4
ms/token, KL vs exact routing unchanged (0.0105 / 0.0101).

What a token costs, and what bounds it:
- RAM experts: ~88 per token read in place over PCIe (4.0 x16: the B550 / Ryzen 5000 root port, though the card does
  5.0). 0.31 ms each alone, ~0.42 in decode. dm-crypt's in-place decryption of NVMe reads (cached stores into DRAM)
  slows the GPU's host reads: 8 RAM experts take 2.50 ms alone, 5.78 beside 7 GB/s of reads from the LUKS root, 2.71-2.80
  beside the same rate from an unencrypted partition. Since 2026-11-01 the expert store lives on its own unencrypted
  partition (`/mnt/b70-fast`, nvme0n1p3, 110 GiB; `~/models/glm53-store` links there): decode 94.2-94.7 -> 90.9-94.9
  ms/token, exact routing 188 -> 181 ms/token. At decode's ~2.6 GB/s most of the 0.42 is not dm-crypt.
- NVMe waits: ~12 ms per token. The drive reads 7 GB/s through dm-crypt (`tools/read_bench`); decode uses ~2.6 GB/s.
  Wider or earlier prefetch (`GLM_PF_WIDTH` > 12, fewer `GLM_SPEC_MAX` threads, the layer after next) costs more than it
  saves: the extra reads slow the RAM experts (above).
- Trunk EXL3 linears (6-bit): 530-620 GB/s at M = 1, at the B70's bandwidth (`tools/linear_bench`).
- Launches: 2.5 us each on the host, 0.74 us in a graph (`tools/launch_bench`); the decode segments are graphs.

Measured and dropped: prefetching the layer after next too (NVMe wait 12.2 -> 8.0-9.9 ms, decode 97.1 -> 98.6-102.9);
copying the next layer's predicted RAM experts into VRAM staging slots on a side queue (the copy engine overlaps
VRAM-only kernels in isolation, but in decode every staged copy cost about what its zero-copy read saved: 1 per layer
94.2 -> 98 ms, 3 -> 110 ms, fenced or not). Bytes over PCIe are the currency. Admission thresholds: `GLM_ADMIT_MIN` 4 /
`GLM_ADMIT_PER_CALL` 2 (the defaults) are the best of 3-8 / 1-3.

Quality-mode options that are faster but route differently (KL vs exact routing; Greg decides):

| | ms/token | KL vs exact |
|---|---|---|
| Now: bonus 0.1 VRAM and RAM, prediction width 12 | 94.4 | 0.010-0.011 |
| Prediction width 8 (`GLM_PF_WIDTH=8`) | 91.9 | 0.0116 |
| Bonus 0.15 VRAM / 0.05 RAM (`GLM_XPU_RES_BONUS=0.15 GLM_XPU_RES_BONUS_RAM=0.05`) | 81.0 | 0.018 |

Prompts (2026-11-02, GM LoRA; `prefill_bench` with the route stats warms the tiers as `glm_server` does):

| | Morning | Now | PyTorch stack |
|---|---|---|---|
| `prefill_bench` 4K / 8K / 16K tokens | 17.3 / 28.4 / - s | 7.9 / 10.4 / 19.8 s | 10.5 / 14.4 / out of memory |
| `glm_server`: 613 / 6.9K / 19.5K (6.7K cached) tokens, first token | 10.9 / 20.4 / 45.1 s | 8.8 / 8.6 / 16.3 s | |

The morning's runs read every expert from NVMe (empty tiers) and ran MLA on the per-row decode kernels. 8K profile now
(`GLM_PROF=5`, ms per token): routed experts 0.48 (about 2/3 trellis decode, re-done per 32-row block: the
`EXL3_MOE_PROBE_DPAS_NODECODE` build gives 0.16), MLA attention 0.18, KDA delta rule 0.19, KDA qkv 0.08. Up to ~4K
tokens a prompt is bound by NVMe: ~7K experts of the 94 GB store live only there and the 990 PRO gives ~6.4 GB/s.
`docs/prefill-research.md`: what else others do (fused sparse MLA, chunked KDA, MoE kernel shapes).

## Status

1. Decode, the expert tier, prefill, the GM LoRA, tokenizer and chat template: done, at parity with the PyTorch stack
   (above and `docs/forward-spec.md`).
2. Server: `tools/glm_server` (`/health`, `/v1/models`, `/v1/chat/completions` with tools, reasoning and
   `return_progress`, `/v1/completions`, `/quit`, cancel on hang-up); `b70 serve` runs it with `GLM_ENGINE=cpp`.
3. A chat client has served from it since 2026-11-02 (`GLM_ENGINE=cpp` in `~/.config/b70/serve.env`).
4. Next: prompt MoE decode shared across row blocks, a fused sparse-MLA kernel, chunked KDA (above); the decode
   graphs' 1.1 GiB of VRAM (~150 more VRAM experts if it shrank); MTP / speculation.

Libraries: `libglm.so` (SYCL, AOT), `libglm_esimd.so` (EXL3 + decode MoE, `-vc-codegen`, links oneDNN from `~/opt/onednn-sycl`; no plain SYCL kernels in its
files), `libglm_moe_pf.so` (prompt MoE, `-vc-codegen -doubleGRF`). `B=build-dev ./build.sh B=build-dev` builds beside
a `build/` that a queued run uses. Start a build dir empty, never as a copy of another: the `.d` files name their own
dir's objects, so in a copy a header change rebuilds nothing that did not change itself, and objects with two layouts of
one struct (or two device images of one kernel name, across the libraries) give wrong results or faults (2026-11-20).

## Gemma 4 31B-based model, 2026-11-03

`glm_server` picks `G4Model` from the folder's `model_type`; `b70 serve` runs it by default (`GEMMA_ENGINE=python`: the
old SGLang path). `llm/exl3conv/notes/gemma4-forward.md` is the forward pass, `tools/g4_gen.cpp` the test tool.

- **Model** (`src/g4.*`, `src/g4_ops.*`): everything on the card but the embedding table (one row a token, read on the
  host). `--cache` is each game's context, `--parallel` the games at once; `G4Model` fits them to the VRAM left after the
  weights, shrinking the prompt scratch before the context, and says so when it cannot give what was asked.
- **Caches**: sliding layers a ring of K, V per game (RING = window + 128). Full layers keep the raw `k_proj` output P,
  1 / rms(P) and Kr = rope(k_norm * P), the 128 rotated dims: K and V are functions of P, so a token and layer costs 5 KB,
  not 8, and no precision is lost. Going back up to 129 tokens costs nothing (a rejected draft); further needs a
  checkpoint of the rings (pinned RAM, ~0.9 GB), taken at each prompt's end, where the next turn's prompt diverges
  because the reply's thought is dropped.
- **Decode attention** (`kernels/g4_dec.h`): ESIMD with 2D block loads and DPAS, pieces of keys combined at the end. Plain
  SYCL with `vec` loads was 4x slower at 8K context (gathers, 16 KiB of scratch a thread). On full layers a pair of threads
  shares each 8 query heads, each owning half the dims, so no key byte is read twice; scores cross in SLM once a block.
- **Prompt attention** (`kernels/g4_fa.h`): the same pairs on full layers (+17% whole-prompt at 32K), 64-key softmax blocks
  on sliding layers; full layers rebuild K, V a key block at a time and merge by log-sum. Prompts run layer-major over
  super-chunks. A short prompt reads as DPAS pieces of 32 rows; from 33 rows one GEMM pass beats DPAS (64 rows: 287 ms by
  DPAS, spilling registers; 128: 542; GEMM flat at ~280).
- **EXL3 rows**: the vector GEMV beats the DPAS kernel up to 8 rows (3 rows 49 ms a step against 57, 8 rows 57 against 70),
  so `kVecMaxM` is 8. 256-register kernels must live in `libglm_moe_pf.so`: under `-vc-codegen` the per-kernel `grf_size`
  does not take effect (vISA "spans complete GRF file"). `g4_fa.o` has no `-ffast-math`: its masks are -inf.
- **Drafter** (`--drafter`, Google's `gemma-4-31B-it-assistant`, 0.9 GB, plain fp16): query heads over the target's last
  sliding and full layers' caches. Input: E[tok] * 73.5 and the target's normed hidden state (then its own
  `post_projection` output), RoPE at tok's position for the whole chain, keys up to position - 1. Chains run together
  and stop at a token within 3 logits of its runner-up. A wasted step costs ~5 ms. Every reply token is still sampled from
  the target's logits at its position, so the reply has the target's distribution; greedy output is identical with and
  without the drafter (113 of 113 tokens).
- **Games** (`src/g4_engine.*`): a request takes the free game that can bring back the most of its tokens
  (`G4Model::reach`: the ring's last 128 positions, or a checkpoint). The longest shared prefix alone was wrong. The client's
  Turns think and its hidden passes do not, and the template opens a thinking prompt with `<|think|>`, so the two
  share four tokens and no cache: they are two conversations. Four tokens still beat an empty game's none, so after a
  restart each took the other's game and read 20K tokens again (17 s after every reply, 16 s before every Turn) with
  two games empty. A request no game can bring anything back for goes to the game continued least, then idle longest
  (an empty one is both); a game untouched for 30 minutes counts as never continued. The request log ends each line
  with its game and the prefix they shared. A step reads a piece of one
  prompt (2048 tokens while others decode), then one batched decode step of at most 8 rows. Before a waiting request picks
  its game, the server checks the running requests' sockets and frees the games of clients that hung up: the client cancels a
  request and sends it again with the reply's thought added, and the second one used to arrive while the first still held
  its game, go to another, and read the whole prompt again (53 of 92 such resends in the client's log, 18 s each).
- **Checkpoints** (`G4Model::save_checkpoint`): going back more than 128 tokens needs the sliding layers' rings as they
  were, ~0.9 GB of pinned RAM a checkpoint. A game keeps one at the end of its last prompt and two at turn starts (the
  prompt's last two `<|turn>` tokens; `G4_TURN_CKPTS`, 0 for none). The end alone was rarely usable by the client: its prompts
  end in an opened thought (4 tokens with thinking off, a prefilled opener with it on) that the next prompt drops, and a
  hidden pass's instruction is dropped a turn earlier, so the next prompt branched before the checkpoint and was read
  from its first token (57% of the requests in that log). A checkpoint less than 128 tokens past another is skipped.
  `tests/g4_cache_check.py` plays the client's requests against a running server and compares cached with cold replies.
- **Text** (`src/chat_gemma.cpp`, `src/tokenizer.*`): the template by hand, SGLang's gemma4 reasoning and tool-call parsers
  (a bare `null` argument is JSON null here, SGLang's string `"null"` is a bug; a string that lost its opening quote is read whole, where SGLang cuts it at its first comma or `]` and returns the pieces as junk arguments). `tests/gemma_check` against HF:
  65M tokens, 4,896 template renders, no difference. Image, audio and video parts get a 400: no vision tower yet.
- **`upload()`** copies through pinned staging: the copy engine reading a cold page of a mapped model file faulted the card.

**Checks** on Gemma 4 31B-based at 5.0 bpw, teacher-forced over 11,410 tokens of the model family's own held-out replies, against
`exl3eval --dump` (dequantized weights, fp32 GEMMs, whole-matrix attention): same top token 99.90%, KL 0.00003, NLL 0.18446
vs 0.18447. Against BF16: KL 0.00337, the reference forward's 0.00334. This test found a real bug a 2K prompt missed: below 1024
tokens, unwritten ring slots counted as visible keys.

**Speed**, Gemma 4 31B-based 5.0 bpw (18.3 GB of trellis), one game, drafter on: 35-42 tok/s on prose, ~60 on list-like text,
45-49 at 52K context on technical Q&A; plain decode 20.8 tok/s, 48 ms a token; 1 / 2 / 4 / 8 games 35 / 56 / 87 / 101 tok/s in
total. A 52K prompt reads in 60 s, the next turn's first token comes in 0.3-0.7 s. Short prompts: first token ~0.25 s. On the 4
bpw Twilight quant, before the 5 bpw model existed: decode 44 ms a token at 8K context, 49 ms at 64K (the exact cache, nothing
packed); prompts 1,444 tok/s at 8K, 788 tok/s at 64K (81 s).

**The quant** (`~/models/llm/<gemma4-31b-exl3-5.0bpw-trace>`): `exl3conv -b 5 --cal-data`, calibrated on 250 rows of the model's
own replies, generated through this server (8 games, ~112 tok/s, 50 minutes). Against the stock calibration on held-out replies:
KL 0.0027 vs 0.0035 (5 conversations, paired -0.0007 [-0.0012, -0.0003]) and 0.0025 vs 0.0034 (10, paired -0.0009 [-0.0018, -0.0001]),
top-1 99.0% vs 98.7%. On generic human chat they tie (KL 0.514 vs 0.515; fine-tuned merges are far from BF16 on text they did not write).

**Left on the table** (Sol review, estimates): EXL3 vector kernel scheduling at K = 5, 2-4 ms a token (GEMV reaches 491 GB/s of
trellis at M = 1, 399 at M = 8); launch fusion, 1-3 ms a step (1,593 kernels; HadIn of q/k/v and gate/up, HadOut into gelu);
adaptive draft lengths per game, low confidence.

## Ripple-65B, 2026-11-04

`src/rp*`: 80 layers, 80 query heads x 128 over 8 KV heads, 64 sliding (window 1024) and 16 global without RoPE (`L % 5 == 4`), scores divided by
sqrt(128), a sigmoid gate on attention's output, `v_proj` reading `lerp(embedding, n, sigmoid(lambda))`, clamped silu MLP, its own head, no softcap. It
is `modeling_ricursive.py` read by hand; nothing of the checkpoint's code runs. Serve it with `b70 serve <folder> [--drafter <DFlash2 folder>]`.

**Any SAGE or plain EXL3 pack loads:** K comes from each trellis (1 to 8, and 1.5, 2.5, 3.5 on every path now), and a tensor without `mul1` takes the
original codebook, as some packs' heads do. REXL3 packs are the same format as EXL3 (same names, `quant_method`, multiplier).

**Checked against** `tests/ripple_forward_ref.py`, an independent numpy forward, on the first 5 layers of the real BF16 weights (`RP_TEST_LAYERS=5`; plain
weights load through `RpLin`), 1100 tokens of prose and code, so past the window: one token at a time KL 0.00003, top-1 99.55%; 1000 as a prompt then
decode 100%; 64 then decode 99.52%. `exl3_check synth` covers every K and both codebooks on the vector, DPAS and GEMM paths.

**Our quant**, `Ripple-65B-Preview-exl3-3.0bpw` (`llm/exl3conv`, flat K=3, scaled Hessians, 16-bit head, 25 GB, 2.9 h of card time), and its DFlash2 quantized (`-exl3`, K=4, 2.6 GB):

| | |
|---|---|
| Quality against BF16, 199 conversations, 115K tokens | KL 0.0134 [0.0126, 0.0142], top-1 95.65%, perplexity 2.656 vs 2.646 |
| Decode, no drafter | 70 ms a token (14 tok/s), 74 at 32K: attention is a few ms of it |
| Rows of a step | 1: 70 ms, 4: 85, 8: 98, 16: 142 (the vector kernel to 8 rows, except K under 3) |
| With the drafter (short replies, greedy) | code 58 tok/s, reasoning 43, factual prose 24, a story 20; a block 12 ms |
| Games at once, total, drafter up to 4 | 3: 42 tok/s, 4: 49, 8: 59 (short replies; 8 without early finishers should reach ~80) |
| Prompts | 661 tok/s at 2K, 566 at 32K; first token of a short prompt 0.25 s |
| Context on the card, one game | 64.9K with the drafter, 105.8K without; a context token costs 64 KB |

A synthetic full-size pack (`tests/ripple_synth_pack.py`) measured load, VRAM and speed before any quant existed; its logits are noise.

**The drafter** is the checkpoint's DFlash2 (`src/rp.cpp`), five layers the size of the target's. Taps of the target's stream at layers 1, 20, 39, 58, 77 go through
`fc` and `hidden_norm` to its K / V (rings of 4160 and 16640 slots: the 16K window its last layer was trained on). A block is the anchor token and 15 masks,
each sublayer wrapped in the grouped dynamic convolutions; the target's head proposes each row's top 32 and the selector walks them greedily. Every reply token
is still sampled from the target's logits. Its chain matches numpy token for token on the real weights, and replies are identical with it on or off (prompts, past the
window, several games), including every rollback. Its MLP overflows fp16 (silu(g) * u past 1e5, an inf is a NaN out of `down_proj`), so that product goes through
fp16 at a per-row power-of-2 scale. As fp16 it takes 7.5 GiB and does not fit beside a 3.0 bpw target at all; `llm/exl3conv` `quant_dflash2` quantizes its big linears (no calibration: its error costs acceptance,
never quality), and that block takes 12 ms against fp16's 37. Chains end at a token within 1 logit of its runner-up (`RP_DRAFT_MARGIN`): acceptance 83% on code and 44% on a story, where always verifying 15 drafts got
40% and 7% and a story ran slower than with no drafter. Past 4 games together it stays off. Accepted tokens per step depend on the text: it is tuned for code.

## Qwen3.8-Flash-Next (Swift 1.5), 2026-11-02

The same engine serves `qwen4_exp` EXL3 folders (first: jeffpeng3/Swift-1.5-Qwen3.8-Flash-Next-EXL3-3.05bpw-heretic in
`~/models/llm`). `glm_server` picks `Q4Model` from the folder's `model_type`; `FLASHNEXT_ENGINE=cpp` (environment or
`~/.config/b70/serve.env`) makes `b70 serve` run it instead of SGLang. `docs/qwen4-spec.md` is the forward pass, from
the HF reference and the exllamav3 fork, with file:line references.

- **Model** (`src/q4.*`, `src/q4_ops.*`): 4 fp32 hyper-connection streams with the gated-residual sites, Gated
  DeltaNet (36 layers, recurrent kernel, 16 k-heads shared by 48 v-heads), QSA attention (12 layers: GQA 24/2 x 256,
  sigmoid output gate, the indexer's 4-token pools; dense while the context is at most 2051 tokens, then each query's
  top-512 pools plus its tail), MoE 512 x top-10 with a shared expert, the hashed n-gram PLE before layer 1 (16 rows of
  the 32.6 GB table per token, `pread` on a thread pool and decoded on the host), the combine-less mixer, lm_head.
- **Experts are all resident**: the VRAM left after the weights holds ~14.4K of the 24.6K records (K=3, 1.86 MB),
  the rest are pinned RAM records the kernels read in place. No NVMe store. Slots go hottest-first by
  `llm/flashnext/hot/gm-7500.json` then `gm-ram-4500.json` (`--hot`). The records are built from the safetensors at
  load in file order (~50 s).
- **Decode** is one SYCL graph per token (the position read from device memory); the host side is the embedding row
  and the n-gram lookup. It made no measurable difference (24.2 vs 23.9 tok/s on a busy machine): the GPU time is the
  routed experts, in particular the RAM-resident ones read over PCIe.
- **Adaptive residency**: GPU counters of decode picks per expert; every 32 decode tokens the most-picked RAM experts
  trade places with the least-picked VRAM ones (decayed counts, 2x hysteresis, at most 64 pairs, between tokens).
  Generic text: 28.8 -> 32.1 tok/s (RAM picks 36% -> 19%). Game-master text only routes ~2% of picks to RAM with the
  static ranking; short GM replies through the server decode at 35-37 tok/s. `Q4_REBALANCE=0` turns it off.
- **Kernels that were missing for this shape**: dense EXL3 K=3 and K=5, routed experts K=3 with no SwiGLU clamp, and
  the decode expert kernels' k-splits for H = 2560 / I = 640 (GLM's split of 8 put gate/up runs across Hadamard
  blocks: relative error ~1.2 until fixed).
- **Checks**: `exl3_check` on K=3/K=5 tensors (all paths ok); `tests/q4_moe_check` (expert kernels on the built
  records, VRAM and RAM, M = 1..300, vs the three linears: rel err 2e-4..1.2e-3); `tests/q4_chat_check` (the template
  against HF `apply_chat_template` + the client's hook, 10 cases with tools, tool responses, effort levels, content parts and
  NFC text: byte-identical, same token ids); teacher-forced logprobs against SGLang (fp16 KV, fp32 states) on a
  347-token GM chat: KL 0.006 (median 0.0009), same top token 98.8%, NLL 2.300 vs 2.304; after a 5400-token prompt
  (sparse QSA, 600 decode positions): KL 0.016 (median 0.004), same top token 96.0%, NLL 1.166 vs SGLang's 1.171.
  GLM is unchanged by the shared changes (decode_check KL 0.0032, NLL 1.5848, as before).
- **Prompts**: 5400 tokens in 8.75 s (617 tok/s). Not tuned yet: RAM experts are read zero-copy by the DPAS prompt
  kernels (no staging into VRAM), GDN runs its recurrent kernel over the prompt, QSA attention is a plain SYCL kernel.
- **Chat client**: `FLASHNEXT_ENGINE=cpp` is in `~/.config/b70/serve.env` and the model is linked into
  the client's models folder (2026-11-02). `Q4_SERVE_REASONING_EFFORT` (xhigh / medium / low) sets the effort of
  requests that name none; unset = the template's xhigh.

## Edge0-145B (NVFP4), 2026-11-15

`E0Model` (`src/e0.*`, kernels `src/e0_ops.*`) runs Edge0/Edge0-145B-NVFP4 from its release folder as shipped
(`~/models/llm/Edge0-145B-NVFP4`; `experts.bin` and `ple-fp8.gguf` link to `/mnt/b70-fast/edge0-145b-nvfp4`). The trunk
is Q4Model's (same kernels); the weights come from `model-skeleton.gguf` (llama.cpp's conversion undone where it
reorders), the MoE is the latent one. Format, conversion details and numerics: `docs/edge0-spec.md`.
`glm_server` picks it from `model.json` (family Edge0); `b70 serve <folder> --port N` runs it.

- **Weights**: Q8_0 linears stay Q8_0 on the card (int8 + fp16 scales), D/U Q8_0, experts NVFP4 records as shipped:
  ~10K in VRAM (hottest by `profile.bin`, in checked 2 GiB pieces), the rest in pinned RAM (~26 GiB), adaptive
  swapping as Swift's. 29.9 GiB VRAM; load ~32 s.
- **VRAM check**: every expert piece is tagged per 64 KiB and read back. A single 16.9 GiB allocation came back with
  its last GiBs mapped over its first (block i held block i + 16 GiB's tag), every time; 2 GiB pieces pass.
  Q4Model (Swift) still allocates one ~26 GiB pool.
- **KV**: int8, an fp16 scale per 64 values (Strata's kv_q8). `E0_KV16=1`: fp16.
- **MTP** (`mtp/`, docs/edge0-spec.md section 7): drafts up to 3 tokens (`E0_SPEC`, 0 = off and not loaded), a draft
  under probability 0.5 ends the chain (`E0_SPEC_MIN_P`). Verify windows back up the GDN / PLE state and replay the
  accepted rows on rollback (`e0_gen rollback`: KL 2e-4 against one token at a time). Each distinct expert is read
  once per window (`nv_*_g`).
- **Checks** (`tests/e0_check`): Q8_0 matvec / GEMM, NVFP4 decode kernels (VRAM and RAM records, grouped), the prompt
  path and a whole latent layer against host references from ggml's definitions: rel 1e-7 (matvec), 2e-4 (fp16
  GEMMs); dequantized weights bit-equal to `dequantize_row_nvfp4`. `e0_check bench`: kernel timings.
- **Quality** (`e0_gen ppl`, Linux-PAM SAG, 5,000 tokens): nll 1.2781 with int8 KV (per-64 scales), 1.2776 fp16 KV
  (per-256 scales had cost 0.0094). One 4,000-token prefill then 999 scored positions matches the same positions run
  token by token.
- **Expert records** are repacked at load (on the GPU) into 16-row tiles (`e0_ops.h`): one tile is a DPAS B operand,
  and the decode kernels run a lane per output row (`nv_*_t`: 234-282 GB/s from VRAM against 171-191 for the file's
  row layout; decode end to end unchanged, it is bound by the RAM experts over PCIe).
- **Prompts' routed experts on XMX** (`src/e0_dpas.cpp`, ESIMD DPAS, 256 registers): picks sorted by (D/U group,
  expert), chunks of 8,192; blocks of 64 rows of one expert, the NVFP4 tiles decoded exactly to fp16 in registers, A by
  2D block loads; gate/up + SwiGLU and down (s_d) in two kernels; U one oneMKL GEMM per group run; RAM experts with
  more than one block staged to VRAM first; the rows' outputs combined per row in a fixed order (a scatter with
  repeated rows raced: two identical prompts gave different logits until `combine_rows`). `E0_MOE_PATH=expert`: the
  one-expert-at-a-time oneMKL path.
- **Speed**: decode 25.6 tok/s without drafts (from 19-21: the Q8_0 matvec takes 4 weight rows a sub-group, D / U skip
  the groups no pick uses, a parallel top-10), 26-29 tok/s with MTP (server, 65-86% of drafts accepted). Decode is
  bound by the ~30% of picks read from pinned RAM over PCIe 4.0 x16. Prompt 963 tok/s at 4K, 1,114 at 12K (from 615
  this morning; one 16K-row MoE piece reached 1,327 but costs ~0.6 GB of expert slots).
- **Long prompts**: a 12,000-token prompt prefilled in one call then 400 scored positions: nll 0.229, against 0.249 for
  the same positions after the context ran 8 tokens at a time; at 3K the two agree to 0.003. The gap grows with the
  context for every prompt MoE path (DPAS in one or two pieces, one expert at a time), so it is the two paths' numerics
  (fp16 GEMMs vs fp32 matvecs) moving the QSA block selection, not the prompt kernels.
- **Open**: an xe page fault at roughly 1 in 7 process exits (the queue blames the next holder): with the copy engine
  it was every exit; without it, still sometimes. Not found yet.
- **Tried, kept off**: the copy-engine queue (`GLM_COPY_ENGINE=1`: every process exit logged an xe page fault, no
  gain measured), staging RAM experts through VRAM for prompts (`E0_STAGE=1`: 739 vs 778 tok/s), oneMKL grouped GEMMs
  for prompt experts (`E0_MOE_GROUPED=1`: 140-390 GFLOP/s at ~80 rows an expert, slower than one expert at a time).
- **Next**: the DPAS kernels' in-register NVFP4 decode is their limit (~14 TFLOPS of gate/up), the GDN prompt kernel
  (17% of a prompt, recurrent), multi-row hyper-connection and small-N matvec kernels (MTP's verify windows).

## Qwen3.5 / Qwen3.8 dense: MTP drafting, 2026-11-18

`Q35Model` drafts with the pack's own one-layer MTP head and verifies through `G4Engine` (sample every row, accept on a
match): on by default when the pack has `mtp.*`. Design and exactness: `docs/qwen35-spec.md` section 7.

- **Switches**: `Q35_MTP` drafts a step (default 2; 0 = MTP not loaded, one GDN state slot, the vector GEMV),
  `Q35_MTP_MIN_P` (0.25: a chain stops when its drafts' probability product falls below it), the draft head
  (`Q35_MTP_BLOCKS`, `Q35_MTP_HEAD`: section "Qwen3.5 / Qwen3.8 dense: decode").
- **Rollback**: two GDN state slots per sequence, a step reads one and writes the other; a rejection replays the
  accepted rows from the saved per-row inputs (2.0 ms then, 1.4 now: two launches, section "decode"; no weights).
  Attention K/V: the position moves back.
- **Memory**: +0.76 GiB at 84K context (28.41 against 27.65 GiB): MTP weights ~0.2, its K/V 4 KB a token, the
  second GDN slot 216 MB a sequence.
- **Exact**: greedy and seeded sampling give the no-draft tokens, checked over 4 GM prompts x 2 turns (prefix cache
  restores, fresh reads) at drafts 1/2/3, greedy and T 0.6 / 1.0 top-k 20; one 8K and one 30K prompt read in one call;
  two sequences decoding together; and the server (thinking, a cut thought, a cancelled stream then the client's `nq_prefill`
  answer-now, a follow-up turn, seeded sampling) against `Q35_MTP=0 GLM_EXL3_VECMAX=0`. Plain `Q35_MTP=0` rounds
  differently (vector GEMV, 5% faster a step) and so may pick other tokens at near-ties.
- **Speed** (`q35_gen spec`, GM prose, tok/s; acceptance per draft):

| Context | Sampling | MTP off (`Q35_MTP=0`) | no drafts, MTP kernels | 2 drafts (default) | 3 drafts |
|---|---|---|---|---|---|
| ~150 | greedy | 19.2-21.0 | 19.1-19.7 | 30.7-33.4 (43-52%) | 31.3-35.1 |
| ~150 | T 0.6, top-k 20 | | 19.3-19.6 | 28.9-35.7 (38-57%) | 28.1-36.3 |
| ~150 | T 1.0, top-k 20 | | 19.2 | 28.0 (44%) | 28.3 |
| 7.8K | greedy | | 18.6 | 30.2-31.4 (46-51%) | 29.6-29.9 |
| 7.8K | T 1.0, top-k 20 | | 18.5 | 26.3-27.4 (42-50%) | 26.4-27.7 |
| 30.6K | greedy | 16.6 | 15.7 | 21.8-25.4 (38-53%) | 20.7-23.4 |
| 30.6K | T 1.0, top-k 20 | 16.6 | 15.4 | 20.3-23.2 (38-53%) | 19.6-21.2 |

  Greedy and T 0.6 short rows: whole draft head, no chain cut (the defaults came after); the others with the defaults.
  The server with thinking on: 31-40 tok/s at short context (51-69% accepted). llama.cpp's MTP on the Q4_K_M GGUF
  did 24-27.
- **Step costs** (`q35_gen bench`, ms): a verify step of 1 / 2 / 3 / 4 rows 50.4 / 52.1 / 54.0 / 54.5 at 241 tokens,
  55.5 / 58.3 / 61.0 / 62.4 at 7.8K, 67.6 / 72.8 / 78.9 / 81.8 at 30.6K; a draft 1.8 ms (64K-id head; 3.1 with the
  whole head), a replay 2.0. Prompts: the MTP layer fills its cache inside the prefill, 3.5% of a 30K read (196 against
  203 tok/s).
- **What it took**: the vector GEMV's cost grows past 2 rows (3 rows 63.5 ms against 51.3 for 2), DPAS stays flat;
  attention read each row's whole cache (+10 ms a row at 30K) until rows of a sequence shared a work-group.
- **Next**: done on 2026-11-20 (section "Qwen3.5 / Qwen3.8 dense: decode"): the decode attention kernel, the replay in
  two launches. A trained (calibrated) MTP quant may accept more.

## EXL3 int8 prompt linears (W8A8), 2026-11-19

Every `Exl3Linear` call with more than `exl3_large_above` rows (prompt pieces: 129+ rows on Qwen3.5/3.8, 33+ on Gemma 4
and Ripple) runs in int8 by default, for every model that uses `Exl3Linear` (`src/exl3.cpp` `forward_large_i8`). Run here
on Qwen3.8-40B and Gemma 4 31B; GLM-5.3, Swift and Ripple take the same path but their weights are gone, so only
`exl3_check synth` (every K code at M = 300) covers them:

1. The input Hadamard as before, then int8 with one scale per row (`HadInQ8WgKernel`, a work-group of 16 threads a row).
   In the Hadamard domain the activations have no outlier channels, so one scale a row is enough.
2. The trellis decoded to int8 at one static scale (`ReconstructKernel<..., int8_t>`): mul1 codebook values lie within
   ±3.453125, so `rne(w * 127 / 3.453125)` never clips. Slices of 32768 columns.
3. A oneDNN s8 × s8 matmul (`~/opt/onednn-sycl`, 3.13, SYCL interop on the caller's in-order queue) with the row scales
   (the weight scale folded in) applied in its epilogue, fp16 out. Rows are padded to a multiple of 256, so a handful
   of shapes serve every prompt; each is built once (~10 ms) and needs no scratchpad.
4. The output Hadamard as before.

Decode-size calls (M <= `exl3_large_above`) are untouched: `exl3_check` at M = 1..300 printed the same numbers with
`build-q35`'s libraries as with this build, before the 40-row fix below changed M = 33..40. A queue that is recording a SYCL graph gets the fp16 path. The method and
kernels 1-2 come from trellis-serve (sybil-solutions, MIT; `xpu/docs/PROGRESS.md`), which measured NLL +0.17% on its
Qwen3.5 with every linear in int8.

**Switches**: `GLM_EXL3_INT8=0` the fp16 path (oneMKL fp16 GEMM, as before); `GLM_EXL3_INT8_CHECK=1` each call also
runs the fp16 path and prints the relative error (slow); `GLM_EXL3_I8_STAGES=1` waits after each stage and sums its time
(`exl3_i8_stages_report()`); `GLM_EXL3_I8_VERBOSE=1` prints each GEMM shape as oneDNN builds it.
**Tools**: `tests/exl3_i8_check <dir> [M] [iters] [name filter]` (every linear of a model, both paths: error and time);
`tools/i8_eval <dir> text|trace|tracecat|speed ...` (both paths in one process: KL over the whole vocabulary, top-1
agreement, NLL; prompt tok/s). Decode stays fp16, so scored tokens see int8 only through the cache and recurrent state
that the prompt left.

**Error per linear** on real activations (a 2048-token prompt, `GLM_EXL3_INT8_CHECK`): Qwen3.8-40B 1.34% mean, median
1.31% (1216 calls; max 4.0% on `mtp.fc`, whose input joins two differently scaled halves; it only feeds drafts);
Gemma 4 31B 0.84% mean, max 2.1%. About half of it is the weights' int8 rounding (its step alone gives 0.78%), half
the activations'. For scale, the 4.0 bpw quant's own layer error is rfn 0.013 (`llm/exl3conv` README).

**Quality** (`i8_eval`, fp16 prompt vs int8 prompt, then the same tokens scored):

| | scored | KL(fp16 ‖ int8) | top-1 agreement | NLL fp16 -> int8 |
|---|---|---|---|---|
| Qwen3.8-40B 4.0 bpw, wikitext, 8 x (4095-token prompt + 128) | 1024 | 0.00040, median 0.00009 | 99.51% | 1.69481 -> 1.69355 (-0.07% ± 0.11%) |
| Gemma 4 31B 5.0 bpw (Lunar-Trace), its own chats joined into 4.2K-6.9K-token prompts (`tracecat`), 8 x 128 | 1024 | 0.00015, median 0 | 99.51% | 0.19415 -> 0.19410 (-0.03% ± 0.45%) |
| Gemma 4, the held-out replies (`trace`, prompts of 52-260 tokens) | 1606 | 0.00001 | 100% | 0.23215 -> 0.23213 |
| Gemma 4, wikitext, 8 x (4095 + 128) | 1024 | 0.19, median 0.0038 | 90.9% | 8.0788 -> 8.0595 (-0.24% ± 0.65%) |

Gemma's wikitext rows are not its kind of text (NLL 8, top-1 27%): there it is far more sensitive to any change.
Two fp16 implementations (oneMKL GEMM against the DPAS kernel, both at 2048-row pieces, 3 pieces) already differ by
KL 0.039 (top-1 96.6%); int8 at the same pieces gives KL 0.21 (top-1 90.1%), the NLL unchanged in both. On its own chat
text the difference is KL 0.00015.

**Speed**, prompt tok/s (`i8_eval speed`, median of 3, both paths in the same process; built from the committed tree
plus this change, without the Q35 prompt work that was in progress beside it):

| | fp16 | int8 | |
|---|---|---|---|
| Qwen3.8-40B 4.0 bpw, 8K | 535 | 648 | +21% |
| Qwen3.8-40B 4.0 bpw, 30K (fp16 one run; 196 on 2026-11-18 too) | 196 | 209 | +7% |
| Gemma 4 31B 5.0 bpw (Lunar-Trace), 8K | 1,491 | 2,311 | +55% |
| Gemma 4 31B 5.0 bpw, 30K | 1,147 | 1,587 | +38% |

Linear-only time (`exl3_i8_check`), one 2048-row piece through all 608 Qwen3.8-40B linears: fp16 1,249 ms, int8 716 ms
(1.74x); 8192 rows on a GDN layer 1.85x. At 2048 rows the int8 time splits into GEMM ~70% (~280-300 TOPS of the card's
367), the reconstruct 13%, the output Hadamard 10% and the input Hadamard 6%; both Hadamards run at the card's memory
bandwidth. In Qwen's 8K profile (`GLM_PROF=5`, ms per token) MLP went 0.518 -> 0.288, GDN qkv/z 0.129 -> 0.067, GDN
out_proj 0.054 -> 0.032, attention projections 0.057 -> 0.032; attention (0.73) and the delta rule (0.35) were then about
70% of a Qwen prompt. After the prompt-reading rewrite (next section) the EXL3 linears are most of it again: 3.7 of
4.5 s at 8K, 13.5 of 20.8 s at 30K. In a build with that rewrite (in progress then), `i8_eval speed` at 8K gave fp16
1,054 and int8 1,761 tok/s (+67%, 2 runs each). Gemma's 8K profile: gate/up 0.273 -> 0.136, down 0.140 -> 0.067, q/k/v 0.106 -> 0.052,
o_proj 0.075 -> 0.034; attention 0.140 is now its largest part.

**Fixed on the way** (fp16, decode-size rows): the 40-row DPAS block gave wrong sums at K = 4 (`exl3_check` rel err
0.42 at M = 33..40 on Qwen3.8's `in_proj_z` and `o_proj`, the same with `build-q35`'s libraries; K = 5 was right, the
48-row block right at both). Qwen sends prompt pieces of 33..128 rows there (`exl3_large_above` 128), so a turn of 33..40
new tokens got wrong linears. 33..40 rows now take the 48-row block (a GDN layer's linears: 2.4 ms, against 1.4 wrong
and 3.5 with the 64-row block); `GLM_EXL3_MB40=1` brings the old block back for a test. Gemma and Ripple
(`exl3_large_above` 32) never used it.

**Left**: Q35 reads a prompt in 2048-row pieces per layer, so the trellis was decoded again for each piece; the int8
weight cache (section "Qwen3.5 / Qwen3.8 dense: decode", 2026-11-20) now decodes it once per layer and super-chunk. The output Hadamard could move into the epilogue of a
GEMM of our own (~10%). The root cause of the 40-row block's error is not found.

## Qwen3.5 / Qwen3.8 dense: prompt reading, 2026-11-19

Prompt rows no longer run attention and the delta rule through the decode kernels. Qwen3.8-40B 4.0 bpw, one sequence, MTP
on (its layer fills its cache inside the prefill), `q35_gen prefill`, medians of 3 (80K: one run):

| Prompt | Decode kernels | Now |
|---|---|---|
| 8K | 13.4 s, 613 tok/s | 4.5 s, 1,815 tok/s |
| 30K | 146.4 s, 210 tok/s | 20.8 s, 1,479 tok/s |
| 80K | ~890 s, ~92 tok/s (from the 30K profile) | 76.1 s, 1,077 tok/s |

"Decode kernels" is the same code with `Q35_PF_ATTN=0 Q35_PF_GDN=0`: both columns run the int8 prompt linears of the
section above (the MTP section's 460 and 203 tok/s were before them).

Where the time goes (`GLM_PROF=5`, seconds for the whole prompt):

| Stage | 8K before | 8K now | 30K before | 30K now |
|---|---|---|---|---|
| Attention core | 5.95 | 0.46 | 121.5 | 5.65 |
| Delta rule | 2.89 | 0.22 | 10.74 | 0.84 |
| GDN gates + conv | 0.29 | 0.17 | 1.07 | 0.65 |
| EXL3 linears (MLP, GDN qkv / z / out, attention q / k / v / o) | 4.22 | 3.68 | 12.88 | 13.54 |

- **Attention** (`kernels/q35_fa.h`, `src/q35_pf.cpp`): Gemma's prompt kernel (`kernels/g4_fa.h`: a thread is 8 rows x 1
  head, DPAS, 64-key blocks) at head dim 256, 6 query heads a KV head, ran at 32 TFLOPS. `q35_gen fabench` (2048 rows at
  position 28.7K on random data, with parts of the kernel switched off) put 23 of a layer's 36 ms on the transposed 2D
  loads that make Q K^T's K^T tiles: a transposed dword load costs ~2.5x a VNNI load per byte on this card. A prompt now
  keeps two copies of the current layer's cache for the selected sequence: keys with each key's d pairs side by side
  (`ktp`, [4][128][T][2]) and values with key pairs interleaved (`vtp`, [4][T / 2][256][2]), so every DPAS B tile is a
  plain 2D load. A layer's first piece in a `prefill` call fills them from the cache (`kt_fill`, `vt_fill`), later pieces
  append. The kernel also rescales O lazily: only when a row's max passes its reference by 2^8 (P stays below 256, in
  fp16). One layer at 30K: 25.7 ms (58 TFLOPS), against 46.9 for Gemma's kernel and 36.3 without the copies. Tried and
  dropped: two threads a head (D split, scores through SLM), 32-key blocks, prefetching the next block into L1, plain
  loads transposed in registers (strided register moves cost more than the transposed loads). The copies' row pitch must
  be a multiple of 16 bytes: a 1023-key pitch read garbage in the test, so it is rounded to 16 keys and checked.
- **Delta rule** (`kernels/q35_gdn.h`, `q35ops::gdn_chunked`): the chunked WY form of FLA's chunk_gated_delta_rule, 64
  rows a chunk, with the triangular system solved by substitution inside the scan instead of inverting (I + A). One
  kernel makes every chunk's A = beta (k k^T) decay and Mq = (q k^T) decay (DPAS); one walks the chunks in order, a thread
  a v-head and 8 value columns (768 threads), the state transposed in registers in fp32: K S0 and Q S0 by DPAS, then
  Delta = beta x - A Delta row by row in fp32 (the chunk's A staged in SLM by the work-group), then the output and the
  state update by DPAS. DPAS inputs are fp16 (q, k after their norms; S0, Delta, Mq), sums fp32. A 2048-row call takes
  0.55 ms (q / k norm 0.10, prep 0.04, scan 0.42) against 16.4 ms for the per-token kernel. The first version inverted
  (I + A) per chunk and spent 2.0 ms in that 64 x 64 forward substitution alone; staging A in SLM took the scan from 0.49
  to 0.42 ms. `q35_gen gdncheck` compares it with `gdn_delta` on random data (1 .. 2048 rows): output relative error
  1.3e-3 (the bf16 output's rounding), state 3e-4.
- **Conv**: `gdn_conv_prompt` computes every (row, channel) at once, where `q4ops::gdn_conv` walks the rows; the same
  sums in the same order, bit-identical (`gdncheck`).
- **Switches**: `Q35_PF_ATTN` (0 the decode kernel, 1 `g4_fa.h`, 5 `q35_fa.h` without copies, 13 the key copy only, 15
  both: the default), `Q35_PF_GDN` (0: the per-token kernels and the serial conv).
- **Memory**: the copies take 4 KB a token of context (336 MB at 84K), allocated at the first prompt. The delta rule's
  scratch (27 KB a row) comes from the existing prompt workspace.
- **Tools** (`tools/q35_gen.cpp`): `prefill` (speed, by stage with `GLM_PROF=5`, `Q35_GEN_PATHS` to time several kernel
  choices in one load), `kl` (the decode kernels against the new ones on the same text: NLL, KL, top-1 of the tokens
  after the prompt), `gdncheck`, `fabench`.
- **Checks**: `q35_gen kl` on held-out text (wikitext-103 validation), the prompt read by the decode kernels and by the
  new ones in one process, then 96 tokens (64 at 30K) scored one at a time:

  | Prompt | NLL decode kernels / new | KL mean (max) | Same top-1 |
  |---|---|---|---|
  | 600 | 1.0786 / 1.0752 | 0.00038 (0.0070) | 95 / 96 |
  | 1023 | 2.2390 / 2.2371 | 0.00047 (0.032) | 96 / 96 |
  | 4097 | 1.2504 / 1.2499 | 0.00013 (0.0009) | 96 / 96 |
  | 8192 | 2.2507 / 2.2481 | 0.00032 (0.0045) | 95 / 96 |
  | 30000 | 1.8250 / 1.8166 | 0.00036 (0.0026) | 63 / 64 |

  That is with the int8 prompt linears on both sides; they re-round whatever reaches them, which magnifies small
  differences. With `GLM_EXL3_INT8=0` on both sides the kernels alone give KL 0.000012 / 0.000016 / 0.000020 at 600 /
  1023 / 8192 tokens (max 0.00044), NLL within 0.0006, the same top-1 for 96 of 96 tokens each.

  Three random codes hidden at 12%, 47% and 83% of a wikitext document are recalled exactly at 705, 29,834 and 77,069
  tokens (`gen`, greedy). `q35_gen spec` (greedy, and seeded T 1.0 top-k 20; prompts of 113, 8.3K and 30.8K tokens, two
  turns each, the second from the cached prefix): every run with 2 drafts gave the no-draft tokens. Decode is
  untouched: `q35_gen bench` verify steps of 1 / 2 / 3 / 4 rows take 63.1 / 68.4 / 72.4 / 81.8 ms at 29.8K (67.6 / 72.8
  / 78.9 / 81.8 at 30.6K in the MTP section).
- **Left**: the EXL3 linears are now ~65% of a 30K prompt and ~80% of an 8K one. Q35 calls each linear once per
  2048-row piece; since 2026-11-20 the decoded int8 weights stay for a layer's later pieces (the weight cache, next
  section: +7% at 8K and 30K; 4K-row MLP pieces would have saved half as much for ~214 MB). Attention is still bound by
  its loads: with no K or V loads at all the same layer took 4.6 ms (`fabench` dbg 3) against 25.7. The GDN gates + conv
  step (0.65 s at 30K: a oneMKL GEMM with 96 outputs, the gates, the conv) could fold into fewer kernels.

## Qwen3.5 / Qwen3.8 dense: decode, 2026-11-20

Qwen3.8-40B 4.0 bpw, MTP on (2 drafts) unless said. "Before" is the build deployed on 2026-11-19 22:35 (int8 prompt
linears and prompt kernels, `build-q35`) and the numbers recorded in the sections above. Built and tested in
`build-q35dec6`.

**Decode speed** (`q35_gen spec`, the same GM prompts as the MTP section's gate: 113 / 8.3K / 30.8K tokens, two turns of
up to 96 tokens; tok/s, turn 1 / turn 2):

| Sampling | Context | No drafts, before -> now | 2 drafts, before -> now |
|---|---|---|---|
| greedy | 113 | 19.6 / 19.3 -> 21.3 / 21.0 | 39.6 / 33.3 -> 43.4 / 36.4 |
| greedy | 8.3K | 18.5 / 18.2 -> 20.6 / 20.4 | 30.0 / 31.8 -> 34.6 / 36.0 |
| greedy | 30.8K | 15.8 / 15.6 -> 19.3 / 19.0 | 26.7 / 24.0 -> 35.1 / 32.3 |
| T 1.0, top-p 0.95 | 113 | 16.1-16.9 -> 20.4 / 20.3 | 21-25 -> 36.5 / 36.5 |
| T 1.0, top-p 0.95 | 8.3K | | -> 29.8 / 31.6 |
| T 1.0, top-p 0.95 | 30.8K | | -> 27.6 / 31.4 |

Greedy turn 2 at 30.8K and the sampled 30.8K turns are not the same text before and after (the new attention rounds
otherwise, so near-ties go the other way). Acceptance is unchanged where the text is the same (53/79, 43/74, 48/78 on
the first turns). The sampled "before" numbers are the ones measured on 2026-11-19 (short and 7K); the sampled "now"
rows were measured before the linear merges (below), which take another 1-2 ms off a step.

**Where a step went** (`q35_gen linbench`, `bench`): the 4-bit linears run at 450-505 GB/s on the DPAS kernel at 1-4 rows
(`lm_head`, K = 6: 555), ~40.7 ms of a 3-row step at short context; `k_proj` / `v_proj` (2.5 MB) take 22 us each, 121
GB/s. Most of the rest was decode attention at long context (below) and launches: a step was ~3,400 kernel launches, and
removing ~650 of them took ~1.1 ms off (below), ~1.7 us a launch.

**Verify steps** (`q35_gen bench`, ms for 1 / 2 / 3 / 4 rows):

| Context | Recorded before | Now |
|---|---|---|
| 241 / 705 | 50.4 / 52.1 / 54.0 / 54.5 (241) | 46.6 / 47.9 / 49.3 / 51.0 (705) |
| 7.8K / 7.5K | 55.5 / 58.3 / 61.0 / 62.4 (7.8K) | 50.2 / 51.3 / 52.6 / 55.0 (7.5K, before the merges) |
| 29.8K | 63.1 / 68.4 / 72.4 / 81.8 | 52.2 / 53.6 / 56.0 / 61.2 |

A draft costs 1.07 / 1.13 / 1.29 ms at 705 / 7.5K / 29.8K tokens (1.64 at 29.8K before); a replay after a rejected
draft 0.96-1.02 ms (2.0 before). By change, at 705 tokens: the pad memsets 1.1-1.2 ms a 1-3-row step, the GDN delta
kernel 0.9-1.5 ms, the merged linears 1.0-2.0 ms; the decode attention ~0.4 ms here and 8-14 ms at 30K (1-3 rows).

3 drafts against 2 (greedy, the spec prompts, tokens the same): 47.4 / 39.7 against 42.1 / 35.3 tok/s at 113 tokens,
34.9 / 36.1 against 33.3 / 35.5 at 8.3K, 32.2 / 33.4 against 33.7 / 31.1 at 30.8K. Faster at short context now that a
row costs less; not measured sampled, so the default stays 2 (`Q35_MTP=3`).

**Decode attention** (`kernels/q35_dec.h`, `q35ops::attn_dec` + `attn_dec_combine`): Gemma's decode kernel shape at head
dim 256. A thread is one row x the 6 query heads of a KV head x a piece of that row's keys: K^T tiles by transposed 2D
loads into DPAS (a DPAS row per head), the softmax in base 2, P V by VNNI loads, O (6 x 256 fp32) in registers
(256-register ESIMD, `libglm_moe_pf.so`). A plain kernel merges the pieces in order and applies the output gate. The
plain SYCL kernel (`attn_core`) ran at ~215 GB/s, and its verify rows shared keys through L2 at best.

- Exactness: a thread reads nothing of the step's other rows. Its pieces come from its own key count (32 pieces of whole
  32-key blocks), its 2D surface ends at its own last key (loads past it read zeros), and the combine merges its own
  pieces. So each row of a verify step is bit-identical to a one-row step at its position (`q35_gen decbench`: random
  data, rows 1..4 at 241 / 7.8K / 30.6K / 80K keys, every output value) and MTP keeps giving the no-draft tokens.
- One layer, random data (`decbench`, ms, the plain kernel -> this one):

  | Keys | 1 row | 2 rows | 3 rows | 4 rows |
  |---|---|---|---|---|
  | 7.8K | 0.133 -> 0.078 | 0.175 -> 0.083 | 0.202 -> 0.091 | 0.369 -> 0.145 |
  | 30.6K | 0.586 -> 0.245 (511 GB/s) | 0.749 -> 0.250 | 0.866 -> 0.282 | 1.253 -> 0.413 |
  | 80K | 1.490 -> 0.582 (563 GB/s) | 1.905 -> 0.593 | 2.193 -> 0.658 | 3.122 -> 1.043 |

  Pieces: 16 / 32 / 64 / 128 measured; 32 is best for 1-3 rows (64 for 4 rows, 0.299 at 30.6K, but a row's pieces may
  not depend on the step's row count). Error against the plain kernel 3e-4 relative (P in fp16 for the DPAS).
- Switches: `Q35_DEC_ATTN=0` the plain kernel (at load: the partials are sized for it), `Q35_DEC_NS` pieces (32).
- Memory: the softmax partials were 768 MB (a floor for prompts read by the plain decode kernel); now 6 MB, and the 768
  MB only with `Q35_PF_ATTN=0` at load.

**GDN delta rule, decode** (`delta_lane_one` in `src/q35_ops.cpp`, `Q35_DELTA`): the step kernel had a lane per key row
and summed over k with a sub-group reduction per value column (~600 operations a lane a row, strided state loads). Now
a lane per value column: a work-group (v-head, 32 columns) of 128 lanes, each owning 32 key rows of its column, so the
state loads run along the columns and the sums over k are a lane's 32 rows, then the 4 quarters in order through SLM.
The step and the replay both take it, so a replay still leaves the step's state bit for bit (the bench's state hashes).
Against the old kernel on random data (`q35_gen gdncheck`): output 1e-6..3e-5 relative, state 1e-7, for 1..4 rows.
Steps 0.9-1.5 ms faster at 1..4 rows, a replay 1.45 -> 0.97 ms. `Q35_DELTA=0`: the old kernel.

**GDN replay** (`q35ops::gdn_replay`): a rejected draft's replay was two kernels per GDN layer (144 launches, 2.0 ms);
now every layer's conv in one launch and every layer's delta rule in another, the step kernels' own per-row code, so the
state is bit-identical (`q35_gen bench` hashes the state after the fused and the per-layer replay of 1..3 of 2..4 rows:
all the same). `Q35_REPLAY=0`: the per-layer launches.

**Launches**: the DPAS kernel's pad rows (a step's 1-3 rows fill a 4-row block) were zeroed by a memset before each
linear; the input-Hadamard launch now writes them (`HadInKernel::zero_pad`, every model on `Exl3Linear`): 1-3-row steps
1.1-1.2 ms faster, 4 rows (no pad) unchanged, outputs identical. Decode folds each residual add into the next RMSNorm
(`q35ops::add_norm16`, bit-identical; the MLP's add moves into the next block's first norm). Linears that read the same
input are merged at load into one (`Exl3Linear::merge`, `forward_split`): gate + up, GDN `in_proj_qkv` + `in_proj_z`,
attention q + k + v, in every layer and the MTP layer. The parts' column tiles sit side by side in one trellis and the
kernels' input shards (`sonb`, an input Hadamard per part) do the rest; the output Hadamard writes each part to its own
buffer. A merged call is three launches instead of six or nine, and `k_proj` / `v_proj` (2.5 MB, 121 GB/s alone) ride
with q. On the prompt path each part keeps its own int8 GEMM (the outputs are bit-identical to separate calls); on the
decode path the K split follows the merged width (outputs 1e-5 apart, every row count the same split, so MTP stays
exact). `q35_gen mergecheck` (`Q35_GEN_ALL=1`: every layer) compares merged with separate calls at 1 / 3 / 8 / 300 rows.
Steps at 705 tokens: 46.8 / 48.2 / 49.6 / 51.3 ms against 48.8 / 50.0 / 50.9 / 52.3 unmerged. `Q35_MERGE` (1 gate + up,
2 GDN, 4 attention; default 7, 0 none).

**Prompt linears' weight cache** (`exl3_i8_cache_mb`, `src/exl3.cpp`): a prompt runs each layer over 2048-row pieces of a
16K-row super-chunk, and every int8 call decoded the whole trellis again. The int8 path now keeps the decoded weights of
the last linears it ran in a 448 MB arena (filled in call order, wrapping; an entry is gone once something is written
over it), so a layer's trellis is decoded once per super-chunk; the largest set, the MTP layer's 8 linears, is ~424 MB.
The cached bytes are the ones the decode writes: the logits after an 8K and a 30K prompt hash the same with
`GLM_EXL3_I8_CACHE_MB=0`. `q35_gen prefill`, cache off -> on: 8K 4.51 -> 4.20 s (1,816 -> 1,950 tok/s), 30K 20.66 ->
19.37 s (1,487 -> 1,586 tok/s). Allocated at the first prompt piece over 128 rows; Q35 sets the size,
`GLM_EXL3_I8_CACHE_MB` overrides it for any model (0: off).

**Draft head** (`src/q35_blocks.h`, `Q35Model::build_draft_head`): the draft argmax was taken over ids < 65,536, so MTP
could never propose `<|im_end|>` (248046), `</think>` (248069) or any other id above. The head is now 512 blocks of 128
ids, the same size: the blocks the model emits most, counted on 23K tokens of replies to 96 synthetic GM prompts
(`q35_gen many`, 8 games, T 0.8; synthetic prompts from a template generator, no chat logs), with the special tokens'
blocks. Their lm_head column tiles are gathered at load into a head of their own (the output Hadamard works on
128-column groups, so the gathered columns are exact; 240 MB more VRAM). Ids >= 65,536 were 2.7% of the emitted tokens
(the two specials 0.5%); the blocks cover 99.1% of them, the prefix covered 97.3%. On 32 held-out prompts (another seed,
2 games, the same replies both ways): 3,370 of 6,346 drafts accepted (53.1%, 54.9 tok/s in total) against 3,348 of 6,352
(52.7%, 53.8 tok/s). `Q35_MTP_BLOCKS=file` takes block numbers from a file; `Q35_MTP_HEAD=N` the first N ids as before
(65536), 248320 the whole head.

**Quality** (decode changes against the deployed numerics, `Q35_DEC_ATTN=0 Q35_DELTA=0 Q35_MERGE=0` in the same build):
`q35_gen nll` on held-out wikitext (validation), each position scored one token at a time with `Q35_GEN_DUMP` and
compared by `lpcmp.py` over each side's top 64:

| | positions | NLL deployed / now | KL mean (max) | same top-1 |
|---|---|---|---|---|
| 2 x (1,280-token prompt + 767 scored) | 1,534 | 1.63741 / 1.63763 | 0.000011 (0.00043) | 1,530 |
| 30,000-token prompt + 63 scored (before the merges) | 63 | 1.82225 / 1.82206 | 0.000009 (0.00003) | 62 |

MTP exactness (`q35_gen spec`, prompts of 113 / 8.3K / 30.8K tokens, two turns each, drafts 0 against 2): greedy,
seeded T 1.0 top-k 20 and seeded T 1.0 top-p 0.95 all gave the no-draft tokens. Needles (`gen`, greedy): the three codes
recalled exactly at 29,834 and 77,069 tokens. "Deployed numerics" is `Q35_DEC_ATTN=0 Q35_DELTA=0 Q35_MERGE=0` in the
same build: its greedy text is the deployed build's, but it shares the rebuilt EXL3 Hadamard kernels, which moved
Gemma's trace by KL ~5e-5 (section "Loading").

**Memory**: at 84K context 28.56 GiB after a 77K prompt, everything included, against 28.54 GiB at "model ready"
before (before any prompt; the prompt's key / value copies, 336 MB, came on top of that). The softmax partials went from
768 to 6 MB; the weight cache adds 448 MB at the first long prompt, the gathered draft head 240 MB. At 32K context
"model ready" is 23.48 GiB against 23.46.

**Left**: fusing more small kernels (the GDN gates into the conv, the norm and swiglu into the next linear's input
Hadamard): ~1.7 us a launch, ~1,300 launches a step left. The draft chain in a
graph: a draft is ~1.1 ms against ~0.9 ms of weights read (its head is half), so little to gain. int8 KV: not done; the
DPAS decode kernel would have to widen int8 to fp16 in registers (DPAS takes no fp16 x int8), so half the bytes would
not be half the time. Cold loads are bound by the disk: 2.7 GB/s with 8 or 16 readers (O_DIRECT through LUKS and
btrfs).

## Sampling on the card (G4Engine), 2026-11-20

`src/sample_gpu.*`, `Sampler::sample_pool`; every model on `G4Engine` (Qwen3.5/3.8, Gemma 4, Ripple). A step copied
each row's logits (248,320 fp16 for Qwen) to the host and sampled there; with top-k unset the sampler took an exp and a
sort over every token within 1e-9 of the best, ~8 ms a step at T 1.0. Now one work-group a row finds the row's best
(after the row's penalties) and keeps every token within the sampler's floor window (-ln(max(min_p, 1e-9)) * T logits,
plus room for roundings), up to the best 4,096 of them (a 1,024-bin histogram of the window). It sends back their ids
and raw logits, with the count and the exp-sum of the window's tokens it left out. The host then runs the sampler's own
code on that pool: penalties from the raw logits, temperature, min-p, top-k, top-p, the same RNG draw.

- Exact (the same token as the host sampler): greedy; any top-k up to the pool, with any penalties, temperature, min-p
  and top-p; any setting whose window held at most 4,096 tokens.
- Top-k unset with top-p < 1 and more than 4,096 tokens in the window: the top-p total adds the left-out tokens' mass,
  summed on the card (fp32, Kahan), to the pool's (double, sorted). A token can differ from the host sampler's only
  when the top-p cut lands within ~1e-7 of a boundary; 0 of 1,772 such rows differed (below).
- A row the pool cannot settle (its top-p cut past the pool, top-p 1 with tokens left out, fewer than k candidates, the
  top bin alone over 4,096) copies its whole row and samples on the host as before.
- Penalties: the card ranks by the past ids' penalties plus those of the step's earlier rows' drafts (the tokens a row
  is only reached after); the host recomputes them exactly on the pool.
- Checks (`GLM_SAMPLE_CHECK=1`: every row is also sampled the old way, from a copy of the request's sampler and RNG
  state): Qwen3.8 `q35_gen spec` seeded T 1.0, top-k 20: 2,084 rows, 0 differ; top-k unset, top-p 0.95: 1,936 rows (1,772
  with the card's top-p rest), 0 differ, no whole rows needed. Gemma 4 (`g4_gen batch`, 4 games, seeded T 1.0 top-p 0.95,
  200 tokens each): 800 rows, 0 differ; 63.8 tok/s in total against 63.0 with whole rows (`GLM_SAMPLE_GPU=0`).
- Speed, Qwen3.8-40B (`q35_gen spec`, seeded T 1.0 top-p 0.95 top-k unset, tok/s at 113 / 8.3K / 30.8K tokens): no
  drafts 16.1-16.9 before (short, 7K) -> 20.3-20.4 / 19.7 / 17.8-18.0; 2 drafts 21-25 before -> 36.5 / 29.8-31.6 /
  27.6-31.4 (with this section's other changes).
- `GLM_SAMPLE_GPU=0`: whole rows as before.

## Loading in one parallel pass (SafeTensors::flush), 2026-11-20

`SafeTensors::defer` / `flush`, used by Q35Model and G4Model. `load()` used to read each tensor through the mapping
into one 32 MB pinned buffer and copy it to the card, one tensor after another (~1 GB/s from a cold page cache). Now
loads only allocate and note the copy, and one `flush()` after the layers reads everything: 8 reader threads
(`GLM_LOAD_THREADS`), 32 MB pieces in file order, each thread with two pinned buffers and its own in-order queue, so a
piece's copy to the card overlaps the thread's next read. A piece whose pages are in the page cache (`mincore`, 90%) is
copied from the mapping; the rest is read with O_DIRECT. The embedding table's `MADV_POPULATE_READ` (2.5 GB; Q35 now
does it too) runs on 4 threads beside it. Qwen3.8-40B, `q35_gen` from start to "model ready" (warm-up included):

| Page cache | Before | Now |
|---|---|---|
| cold (the folder's pages dropped with POSIX_FADV_DONTNEED) | 21.1-21.7 s | 9.7 s (weights 19.4 GB in 7.2 s, 2.7 GB/s) |
| warm | 4.2-4.6 s | 3.2-3.7 s (weights in 1.8-2.5 s, 7.8-10.6 GB/s) |

On this disk (btrfs on LUKS) the O_DIRECT reads still fill the page cache: the load after a cold one was warm. A cold
load is bound by the disk: 16 readers read at the same 2.7 GB/s as 8. Gemma 4 31B 5.0 bpw (Lunar-Trace), cold: the
weights, 21.1 GB, in 8.3-10.6 s (2.0-2.55 GB/s); the deployed build's whole load took 28.3 s (`g4_gen gen`, cold). Its
held-out trace (`g4_gen scoretrace`, 2,604 positions) against the deployed build: KL ~4.6e-5 over the top 64, NLL
0.19295 -> 0.19269, the same top-1 at 2,597. Not bit-identical, where the shared changes should be (the loader copies
the same bytes, the pad zeros are the memset's): the likely cause is the rebuilt fast-math EXL3 Hadamard kernels
(`HadInKernel`, `HadOutKernel` gained fields and a branch); not isolated.
