#!/usr/bin/env bash
# Serve Lygodactylus/Qwen3.8-Flash-Next-Uncensored-exl3-3bpw on the Arc Pro B70 with trellis-serve (SGLang 0.5.20-xpu +
# exl3xpu), from the unpacked image under bubblewrap. OpenAI API on http://127.0.0.1:${PORT:-30250}/v1
#
# Memory: experts ranked hottest by Strata's routing profile (HOT, default the top 8,000) stay pinned in device slots
# with no host copy (EXL3_MOE_STATIC, this repo's b70-static-experts patch); layer 0's 4-bit experts all sit in their own
# device store (1.27 GB); the rest of the 3-bit experts (~31.8 GB) live in pinned host
# memory behind SLOTS - HOT LRU slots. The n-gram table is read from NVMe behind a ${NGRAM_RAM_GB:-4} GB RAM cache.
#
# Env: PORT (30250), CTX (65536), SLOTS (8500), HOT (7500; only with HOT_JSON=), HOT_JSON (hot/gm-7500.json), NGRAM_RAM_GB (4),
# MEMFRAC (0.75), CHUNK (8192; 16384 is ~5-10% faster on long prompts but ran out of VRAM after an image request), MAMBA_CACHE,
# SPEC (0; 1 = MTP speculative decoding with the MTP experts pinned on the GPU), SPEC_STEPS (1 draft step),
# USERS (1: concurrent requests), GRAPHS (1: XPU decode graphs, which SGLang leaves off on XPU unless asked; 0xSero measured eager 16 -> graphs 32 tok/s);
# RAM_JSON (unset: every non-static expert in RAM) = an EXL3_MOE_RAM list (route_to_hot.py ram): only those experts keep
# a host copy, the rest live in the NVMe cold store under QWEN_NVX (written on the first load with that list, ~1.86 MB
# per expert); decode never routes to them, prompts read the ones they pick. RAM_MARGIN (0) = EXL3_MOE_RAM_MARGIN.
# EXL3_* / SGLANG_* set by the caller are passed through.
set -euo pipefail
# run by hand outside a queue job: take a turn on the B70 through the b70q queue first, as a server it may stop after
# IDLE (15m) idle on the GPU while another job waits
if [ -z "${B70Q_TICKET:-}" ]; then
    exec b70-lease run trellis --idle "${IDLE:-15m}" --ram-gib "${RAM_START_GB:-45}" --note "trellis-serve Flash-Next" -- "$0" "$@"
fi
S=$(dirname "$(readlink -f "$0")")
T=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
MODEL=Qwen3.8-Flash-Next-Uncensored-exl3-3bpw
PORT=${PORT:-30250}; CTX=${CTX:-65536}; SLOTS=${SLOTS:-8500}; HOT=${HOT:-7500}
mkdir -p "$T/run"
# The static tier defaults to hot/gm-7500.json: the 7,500 experts game-master turns route to most (route_to_hot.py from
# a routing profile; it covers ~94% of held-out GM decode routings against 39% for the Strata profile, and lifts decode
# from ~29 to ~34 tok/s). HOT_JSON= (empty) uses the Strata profile with HOT experts instead.
HOT_JSON=${HOT_JSON-$T/hot/gm-7500.json}
if [ -n "$HOT_JSON" ]; then         # a hot list built elsewhere (e.g. route_to_hot.py from a routing profile)
    [ "$(readlink -f "$HOT_JSON")" = "$(readlink -f "$T/run/hot.json")" ] || cp "$HOT_JSON" "$T/run/hot.json"
else
    python3 "$S/profile_to_hot.py" "$HOME/work/strata-b70/Strata/data/expert-profile.bin" "$HOT" "$T/run/hot.json" 0 >/dev/null   # layer 0: its own 4-bit store
fi
COLD_ENVS=()
if [ -n "${RAM_JSON:-}" ]; then
    NVX=${QWEN_NVX:-$HOME/models/qwen38-store}
    if [ ! -d "$NVX" ]; then
        mkdir -p "$NVX"
        chattr +C "$NVX" 2>/dev/null || true    # btrfs: no copy-on-write / checksums, so O_DIRECT works
    fi
    [ "$(readlink -f "$RAM_JSON")" = "$(readlink -f "$T/run/ram.json")" ] || cp "$RAM_JSON" "$T/run/ram.json"
    COLD_ENVS=(--setenv EXL3_MOE_RAM /run/trellis/ram.json --setenv EXL3_MOE_COLD_STORE /nvx/qwen38_cold_experts)
fi
[ -n "${RAM_MARGIN:-}" ] && COLD_ENVS+=(--setenv EXL3_MOE_RAM_MARGIN "$RAM_MARGIN")
X=/opt/trellis-serve/xpu/exl3xpu
ENVS=(
    --setenv HF_HUB_OFFLINE 1
    --setenv ZE_AFFINITY_MASK 0
    --setenv EXL3_MODEL_PATH "/models/$MODEL"
    --setenv EXL3_LIB "$X/_C_sgl.so"
    --setenv EXL3_MOE_LIB "$X/_moe_sgl.so"
    --setenv EXL3_MOE_SLOTS "$SLOTS"
    --setenv EXL3_MOE_CACHE 1
    --setenv EXL3_MOE_STATIC 1
    --setenv EXL3_MOE_HOT /run/trellis/hot.json
    --setenv EXL3_NGRAM_TIER nvme              # read by ngram_host.py
    --setenv SGLANG_EXL3_NGRAM_TIER nvme
    --setenv SGLANG_EXL3_NGRAM_RAM_GB "${NGRAM_RAM_GB:-4}"
    # GPU wait for n-gram rows from the NVMe service thread: give up (stale rows for that step) after 1.5 s instead of
    # the 60 s default. xe resets any compute job running over 5 s (ccs job_timeout_ms), and on 2026-10-06 a decode step
    # spun past it while RAM was nearly exhausted, an engine reset that the guard then stopped the server for.
    --setenv SGLANG_EXL3_NGRAM_TIMEOUT_S "${NGRAM_TIMEOUT_S:-1.5}"
    --setenv TORCHINDUCTOR_COMPILE_THREADS 1
    # growable allocator segments instead of fixed blocks: long prefills left 1.4 GB of torch's cache in pieces too small
    # to reuse, and on 2026-10-06 two image/long prefills died on a 320 MB allocation at the torch cap (OOM)
    --setenv PYTORCH_ALLOC_CONF "${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
    # scripts/pyhooks/sitecustomize.py: every server process hands freed heap memory back to the OS (malloc_trim) after
    # the load and every TRIM_EVERY_S, instead of keeping ~4 GB of freed load-time memory that then sits in swap
    --setenv PYTHONPATH /opt/pyhooks
    --setenv TRIM_EVERY_S "${TRIM_EVERY_S:-300}"
    # SGLang holds back a waiting request whose prompt shares >= 32 tokens with another waiting one, hoping to reuse
    # the first one's cache. Here that backfires: every prefill forward streams the non-resident experts (~31 GB from
    # RAM / NVMe), and GDN prefix states are often gone by then, so 4 concurrent GM turns ran as 4 separate prefills
    # (28-35 s to the first token, 2026-10-12). A huge threshold batches them into one forward.
    --setenv IN_BATCH_PREFIX_CACHING_DEPRIORITIZE_THRESHOLD "${INBATCH_THRESHOLD:-1000000000}"
)
ENVS+=("${COLD_ENVS[@]}")
while IFS='=' read -r k v; do ENVS+=(--setenv "$k" "$v"); done < <(env | grep -E '^(EXL3_|SGLANG_)' || true)
# in_rootfs.sh takes the extra bwrap env through BWRAP_EXTRA
export BWRAP_EXTRA="${ENVS[*]@Q}"
GRAPH_ARGS=()
# USERS=N (default 1): run up to N requests at once, with decode graphs for batch sizes 1, 2, 4, .. N
USERS=${USERS:-1}
GRAPH_BS=$(b=1; while [ "$b" -lt "$USERS" ]; do echo -n "$b "; b=$((b * 2)); done; echo "$USERS")
[ "${GRAPHS:-1}" = 1 ] && GRAPH_ARGS=(--cuda-graph-backend-decode full --cuda-graph-bs-decode $GRAPH_BS)
SPEC_ARGS=()
if [ "${SPEC:-0}" = 1 ]; then
    # the MTP layer's 512 experts get pinned device slots on top of SLOTS (0.95 GB)
    SPEC_STEPS=${SPEC_STEPS:-1}; MEMFRAC=${MEMFRAC:-0.80}     # the draft needs its own GDN states on top
    SPEC_ARGS=(--speculative-algorithm NEXTN --speculative-num-steps "$SPEC_STEPS" --speculative-eagle-topk 1
               --speculative-num-draft-tokens $((SPEC_STEPS + 1)) --mamba-ssm-dtype float16)
    ENVS+=(--setenv EXL3_MOE_STATIC_MTP 1 --setenv EXL3_MOE_SLOTS $((SLOTS + 512)))
    export BWRAP_EXTRA="${ENVS[*]@Q}"
fi
# WARMUP=1 (default): once the server answers, send one ~5K-token prompt (hot/warmup_prompt.json, 1 token out) from a
# background helper. The first long prompt after a start runs ~2.5 s slower than later ones (one-time setup); this
# pays that before anyone is waiting. The helper gives up after 15 minutes if the server never comes up.
if [ "${WARMUP:-1}" = 1 ]; then
    (
        for _ in $(seq 300); do
            curl -s --max-time 2 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
            sleep 3
        done
        sed 's/"max_tokens": *[0-9]*/"max_tokens": 1/' "$T/hot/warmup_prompt.json" |
            curl -s --max-time 300 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
                -d @- >/dev/null 2>&1 && echo "serve_b70: warm-up prompt done" >&2
    ) &
fi
# MAMBA_CACHE=N caps SGLang's GDN state cache (default 10 states, 1.16 GB; the extra states keep cached prompt prefixes)
MAMBA_ARGS=(); [ -n "${MAMBA_CACHE:-}" ] && MAMBA_ARGS=(--max-mamba-cache-size "$MAMBA_CACHE")
# MAMBA_PATH_STATES=N keeps at most N cached GDN states per conversation (SGLang default unlimited: a long prompt leaves
# one per 8K prefill chunk, and with few slots those pushed every other conversation's states out, so interleaved
# conversations re-read their whole prompt). MAMBA_STRATEGY: extra_buffer (default) or extra_buffer_lazy (a running
# request takes its second tracking slot only at a track boundary).
[ -n "${MAMBA_PATH_STATES:-}" ] && MAMBA_ARGS+=(--mamba-max-states-per-path "$MAMBA_PATH_STATES")
[ -n "${MAMBA_STRATEGY:-}" ] && MAMBA_ARGS+=(--mamba-radix-cache-strategy "$MAMBA_STRATEGY")
# MAX_TOTAL_TOKENS pins the KV pool: without it SGLang sizes the pool from whatever VRAM is left, so VRAM freed elsewhere
# (fewer expert slots) became KV instead of headroom
[ -n "${MAX_TOTAL_TOKENS:-}" ] && MAMBA_ARGS+=(--max-total-tokens "$MAX_TOTAL_TOKENS")
exec "$S/in_rootfs.sh" --gpu python3 -m sglang.launch_server --model-path "/models/$MODEL" --quantization exl3 \
    --trust-remote-code --device xpu --host 127.0.0.1 --port "$PORT" --served-model-name flashnext-uncensored \
    --disable-shared-experts-fusion --kv-cache-dtype fp8_e4m3 --context-length "$CTX" \
    --mem-fraction-static "${MEMFRAC:-0.75}" --chunked-prefill-size "${CHUNK:-8192}" --max-running-requests "$USERS" \
    --dtype bfloat16 --reasoning-parser qwen3 --tool-call-parser "${TOOL_PARSER:-qwen3_coder}" \
    --linear-attn-prefill-backend intel_xpu --linear-attn-decode-backend intel_xpu "${GRAPH_ARGS[@]}" "${SPEC_ARGS[@]}" "${MAMBA_ARGS[@]}" "$@"
