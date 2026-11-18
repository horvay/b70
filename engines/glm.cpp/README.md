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
| `src/exl3.*` | EXL3 linear: port of `exl3_gemm_small`, same ESIMD kernels (`kernels/exl3_esimd.h`) |
| `src/ops.*` | RMSNorm, mHC mix / apply / mean, clamped SwiGLU, fp16 GEMV, exact router |
| `src/kda.*` | KDA linear attention (recurrent; conv, gates, delta rule, gated norm) |
| `src/mla.*` | MLA + DSA indexer, decode graph path (gathered attention, top-512 pools by radix select) |
| `src/moe*`, `kernels/exl3_moe_kernels.h` | routed experts (kernels extracted from `exl3_moe.n128.sycl`), routing alignment |
| `src/tier.*` | expert tier: VRAM / RAM (memfd, read in place) / NVMe, reader threads, prefetch, admission, prompt staging |
| `src/lora.*` | PEFT LoRA adapters (the GM LoRA): runtime deltas, merged fp16 weights, shared A projections |
| `src/model.*` | the trunk: embedding, 45 blocks, final norm, lm_head; layer-major prefill, KDA checkpoints, decode graphs |
| `src/sampler.*`, `src/engine.*` | ComboSampler's order of penalties and filters; a request: prefix reuse, prefill, decode |
| `src/http.*`, `tools/glm_server.cpp` | the OpenAI-compatible server, a drop-in for `serve.py` (same routes, arguments, SSE) |
| `tests/` | `exl3_check` (EXL3 vs a CPU trellis decoder), `decode_check` (vs `dumpref.py`), `acts_check` (per layer vs `dumpacts.py`), `tok_check` (tokenizer, stream decode, prompts vs `tok_ref.py`; CPU only), `gemma_check` (Gemma's tokenizer, stream decode, prompts vs `gemma_ref.py`, its parsers vs cases worked out from SGLang's source; CPU only) |
| `src/g4*`, `kernels/g4_*.h`, `src/chat_gemma.cpp` | Gemma 4 31B dense (`gemma4`, e.g. Gemma 4 31B-based): the model, several games at once (`g4_engine`), its drafter, chat template and parsers (below) |
| `src/rp*`, `kernels/rp_dec.h`, `src/chat_ripple.cpp` | Ripple-65B (`ricursive`): the model, its DFlash2 drafter, several games at once, its chat template and parsers (section below) |
| `src/q4*`, `src/chat_qwen.cpp` | Qwen3.8-Flash-Next (`qwen4_exp`, e.g. Swift 1.5): the model, its kernels (`q4_ops`), its chat template (below) |
| `src/q35*`, `tools/q35_gen.cpp` | Qwen3.5 / Qwen3.8 dense (`qwen3_5`: 27B, the 40B Ultimate-Heretic): Gated DeltaNet + gated attention, its vision tower, MTP drafting (`docs/qwen35-spec.md`, section below) |
| `src/e0*`, `src/gguf.*` | Edge0-145B (a latent-MoE Qwen3.8-Flash-Next, NVFP4 experts) from its release folder: `docs/edge0-spec.md` |
| `src/chat_gemma.cpp` | Gemma 4 (`gemma4`) text: its chat template ported by hand with the client's prefill hook, SGLang's message shaping, its `gemma4` reasoning split (`GemmaSplitter`) and tool-call parser; host only |
| `tools/` | `prio` (queue priority), `gen_bench` (free-running decode), `launch_bench`, `linear_bench`, `read_bench` (NVMe), `udmabuf_test` |

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

Libraries: `libglm.so` (SYCL, AOT), `libglm_esimd.so` (EXL3 + decode MoE, `-vc-codegen`; no plain SYCL kernels in its
files), `libglm_moe_pf.so` (prompt MoE, `-vc-codegen -doubleGRF`). `B=build-dev ./build.sh B=build-dev` builds beside
a `build/` that a queued run uses.

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
  `Q35_MTP_MIN_P` (0.25: a chain stops when its drafts' probability product falls below it), `Q35_MTP_HEAD` (65536:
  the draft argmax over the first ids only; 248320 = all).
- **Rollback**: two GDN state slots per sequence, a step reads one and writes the other; a rejection replays the
  accepted rows from the saved per-row inputs (2.0 ms, no weights). Attention K/V: the position moves back.
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
- **Next**: decode attention runs at ~225 GB/s (13 ms of a 30K step for one row); a faster kernel helps with and
  without drafts. The replay could be 2 launches instead of 144. A trained (calibrated) MTP quant may accept more.

