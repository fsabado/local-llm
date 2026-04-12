#!/bin/bash
# Gemma 4 26B A4B (effective 4B active, 25.23B total MoE) via llama.cpp
#
# Architecture: 30 layers — 25 SWA (window=1024) + 5 full-attention (every 6th layer)
#   MoE: 128 experts, 8 active per token
#   SWA layers: 8 KV heads @ head_dim=256, full-attn: 2 KV heads @ head_dim=512
#   Native context: 262,144 tokens
#
# Two backends available:
#
#   f16 KV  (upstream llama.cpp):
#     ceiling p=4 at 131K/slot | ctx cliff: ctx*p > 131K → 7× slowdown
#     tg: ~124 tok/s
#
#   turbo4 KV  (llama-cpp-turboquant-gemma4, RECOMMENDED):
#     ceiling p=50+ at 131K/slot (iSWA plateau) | full 262K native context at p=4
#     tg: ~114 tok/s (-8% vs f16) | KV VRAM: -3.8 GB at 256K vs f16
#
# Usage:
#   bash start.sh [--turbo4] [model] [port] [ctx_total] [parallel]
#
# Defaults (f16):  port=10444  ctx=131072  parallel=4   (32K/slot)
# Defaults (turbo4): port=10444  ctx=1048576  parallel=4  (256K/slot, native max)

TURBO4=0
if [ "$1" = "--turbo4" ]; then
  TURBO4=1; shift
fi

MODEL="${1:-/home/fsabado/models/gemma-4-26b-a4b-it/google_gemma-4-26B-A4B-it-Q4_K_M.gguf}"
PORT="${2:-10444}"

if [ "$TURBO4" = "1" ]; then
  LLAMA_BIN="/home/fsabado/src/llama-cpp-turboquant-gemma4/build-cuda/bin"
  CTX="${3:-1048576}"    # 262K/slot at p=4 — full native context
  PARALLEL="${4:-4}"
  KV_FLAGS="--cache-type-k turbo4 --cache-type-v turbo4"
  echo "Mode: turbo4 KV (3.8× compression, p=50+ ceiling)"
else
  LLAMA_BIN="/home/fsabado/src/llama.cpp-upstream/build-cuda/bin"
  CTX="${3:-131072}"     # 32K/slot at p=4 — safe from context cliff
  PARALLEL="${4:-4}"
  KV_FLAGS=""
  echo "Mode: f16 KV (ceiling p=4 at 131K/slot — avoid ctx*p > 131072)"
fi

export LD_LIBRARY_PATH="$LLAMA_BIN:$LD_LIBRARY_PATH"

echo "Starting llama-server: $(basename $MODEL) on port $PORT"
echo "Context: $CTX total | Parallel: $PARALLEL | Per-slot: $((CTX / PARALLEL))"
echo ""

"$LLAMA_BIN/llama-server" \
  --model "$MODEL" \
  --port "$PORT" \
  --host 0.0.0.0 \
  -ngl 99 \
  --ctx-size "$CTX" \
  --parallel "$PARALLEL" \
  $KV_FLAGS \
  --cont-batching \
  --flash-attn on \
  --no-warmup \
  --log-disable
