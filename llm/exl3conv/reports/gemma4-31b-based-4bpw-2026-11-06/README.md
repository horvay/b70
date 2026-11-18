# Gemma 4 31B-based model, EXL3 4 bpw

Completed 2026-11-06. Conversion queue series #1491 finished after 19 runs, ending with #1521. Evaluation and the server check #1527 completed with exit code 0.

## Model

- Original BF16: `~/models/llm/<gemma4-31b-bf16>`
- Output: `~/models/llm/<gemma4-31b-exl3-4.0bpw>`
- Gemma 4, mul1 EXL3, 4 bpw linears, 16-bit head, automatic output scales, scaled Hessians enabled.
- All 62 shards referenced by the safetensors index exist. Their combined file size is 21,450,264,401 bytes.
- `gemma-4-31B-it-assistant` works as its drafter.

## Quantization fidelity

Each evaluation compares the finished quant directly with the BF16 original on the same inputs. These are existing completed evaluation results; no baseline or evaluation was repeated during handback.

| Evaluation | Scored tokens | Conversations | KL, 95% interval | Top-token agreement, 95% interval | BF16 perplexity | Quant perplexity |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| Held-out chat trace | 13,835 | 10 | 0.0122 [0.0075, 0.0187] | 97.35% [96.57, 98.08] | 1.207 | 1.217 |
| Broader chat, oasst2 | 101,700 | 199 | 0.9131 [0.8326, 0.9948] | 74.49% [73.14, 75.93] | 753.199 | 815.346 |

The held-out evaluation uses a trace from a sibling Gemma 4 31B-based model rather than a self-trace of this one. The larger chat set shows materially lower agreement. Its BF16 perplexity is already high; that is not caused by this conversion. These measurements describe quantization fidelity on the supplied datasets, not the quality of a long played session.

Raw summaries are in [eval-chat.txt](eval-chat.txt) and [eval-heldout.txt](eval-heldout.txt). The chart is [quality.html](quality.html).

## Serving verification

One new queued check, #1527, loaded the completed quant with the Gemma 4 31B assistant drafter.

- Server initialized a 65,536-token context using 23.33 GiB VRAM including drafter and buffers.
- A short narration request completed with 156 generated tokens, 0.241 seconds to first token, and 37.80 decode tokens/s.
- Tool calling returned `add_numbers` with numeric arguments `a=17`, `b=25`.
- A direct arithmetic request returned `17 * 23 is 391.`
- A second narration request completed with 154 generated tokens, 0.226 seconds to first token, and 39.81 decode tokens/s.
- These generation prompts were short, around 95 tokens. This was not a long-context test.
- The check stopped its own server afterward. The queue resumed the existing standby server.

See [serve-check.log](serve-check.log) and [server.log](server.log).
