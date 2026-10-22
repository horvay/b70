#!/usr/bin/env bash
# Start the B70 server through the b70q queue under guarded.sh (RAM / B70 fault guard) with the everyday settings: 200K
# context, general-text hot list, 12 GDN state slots. Any of them can be overridden from the environment
# (e.g. HOT_JSON=$PWD/hot/gm-7500.json for game-master use). Log: run/serve.log. Stop with scripts/stop_server.sh.
T=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
# MEMFRAC 0.80 -> 0.785 with SLOTS 7800: SGLang sizes the KV pool from what is left, so without it the 0.6 GB freed from
# expert slots went straight into KV (208K -> 256K tokens) instead of VRAM headroom
export CTX=${CTX:-204800} MEMFRAC=${MEMFRAC:-0.785}
# prefix cache across interleaved conversations (OMP main session + subagents): resuming a conversation needs a cached
# GDN state (113 MB each, fp32). With 6 slots a running request held 3 and the rest were pushed out by any other
# conversation, so every turn re-read the whole prompt (100K tokens: 111 s). 12 slots, at most 2 states per
# conversation, keep 3 interleaved conversations cached (the same 100K turn: 8 s). The extra 0.68 GB of VRAM comes out
# of the LRU expert slots (8500 -> 8135, decode 29.4 vs 29.8 tok/s on general text), so context and VRAM slack stay.
# (SGLang's int8 GDN checkpoints would be 4x smaller but refuse this model: "incompatible with Qwen4-Exp PLE side states")
# SLOTS 8135 -> 7800 (-0.6 GB) on 2026-10-06: in real 30K-170K OMP use VRAM still peaked at 31.4-31.5 of 31.8 GB, and three
# times that day the xe driver evicted the ~15 GB expert arena into system RAM (VRAM 31.4 -> 16.7 GB, RAM -14 GB in
# seconds, guard stop). Together with the 0.90 torch cap below this leaves ~1.3 GB more VRAM headroom.
# SLOTS 7800 -> 7600 and the KV pool pinned at 217,088 tokens (MAX_TOTAL_TOKENS) the same evening: an image prefill
# then hit torch OOM at the 0.90 cap (27.1 GB allocated + 1.4 GB fragmented cache, 0.4 GB free on the card)
export MAMBA_CACHE=${MAMBA_CACHE:-12} MAMBA_PATH_STATES=${MAMBA_PATH_STATES:-2} SLOTS=${SLOTS:-7600}
export MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-217088}
# 200K context, not 262K: at 262K VRAM sat at 31.5 of 31.8 GB, and a long prefill made the xe driver evict whole GPU
# buffers (the 15.8 GB expert arena among them) into system RAM, draining ~10 GB in seconds. These settings keep
# ~2.5-3 GB of real VRAM slack.
# cap torch's caching allocator at 92% of the card: after a long prefill its cached blocks filled the card and the
# next decode graph replay (Level Zero allocates outside torch) failed with OUT_OF_DEVICE_MEMORY
# (0.92 -> 0.90 on 2026-10-06: after a long prefill the cache still grew ~1.5 GB over idle and pushed VRAM to the edge)
export EXL3_TORCH_MEM_FRACTION=${EXL3_TORCH_MEM_FRACTION:-0.90}
# log torch vs card VRAM every 30 s when it changes (exl3xpu mem: lines in serve.log), to see what fills the card
export EXL3_MEM_LOG=${EXL3_MEM_LOG:-30}
# give torch's cached prefill blocks back before the reply (~3.5 GB of VRAM headroom while decoding; see
# _patch_release_after_prefill in the plugin)
export EXL3_RELEASE_AFTER_PREFILL=${EXL3_RELEASE_AFTER_PREFILL:-1}
# n-gram table RAM cache 0.5 GB instead of 4: the server already pins ~33 GB of RAM, and beside other jobs
# (an sd-server, CPU background removal) or an Unreal build the machine ran out at peak; misses read from NVMe
export NGRAM_RAM_GB=${NGRAM_RAM_GB:-0.5}
export HOT_JSON=${HOT_JSON:-$T/hot/general-7500.json}
# RAM_JSON (e.g. hot/gm-ram-4500.json): keep only those non-static experts in RAM, the rest in the NVMe cold store
# (QWEN_NVX); RAM_MARGIN: router penalty against experts outside VRAM in decode. Empty = all in RAM, as before.
export RAM_JSON=${RAM_JSON:-} RAM_MARGIN=${RAM_MARGIN:-} QWEN_NVX=${QWEN_NVX:-$HOME/models/qwen38-store}
# the queue starts it once RAM_START_GB is available (45: the all-in-RAM server pins ~33 GB); with a RAM list the
# server pins ~15 GB (gm-ram-4500), so 25 is enough
[ -n "$RAM_JSON" ] && export RAM_START_GB=${RAM_START_GB:-25}
# keep the previous run's log (its prefill lines show prefix-cache hits per request)
mkdir -p "$T/run/logs"
[ -s "$T/run/serve.log" ] && mv "$T/run/serve.log" "$T/run/logs/serve-$(date -r "$T/run/serve.log" +%m%d-%H%M).log"
: > "$T/run/serve.log"
# its turn on the B70 through the b70q queue (~/ai/b70q): the daemon starts it once the card and RAM_START_GB of RAM are
# free, runs it in its own unit (b70q-<id>.service, with this environment), and stops it after IDLE (15m) idle on the
# GPU while another job waits; start it again when it is needed
cd "$T"
export RAM_START_GB=${RAM_START_GB:-45}
id=$(b70q serve trellis --idle "${IDLE:-15m}" --ram-gib "$RAM_START_GB" --note "trellis-serve Flash-Next" \
    --log "$T/run/serve_guard.out" -- bash -c 'exec scripts/guarded.sh run/serve.log scripts/serve_b70.sh < /dev/null' | head -1)
echo "$id" > "$T/run/serve.ticket"
echo "queued as b70q #$id (log $T/run/serve.log; \`b70q\` shows its place); ready ~2.5 min after it gets the B70, at http://127.0.0.1:30250/v1"
