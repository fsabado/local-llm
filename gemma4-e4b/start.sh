#!/bin/bash
# Gemma 4 E4B (effective 4B, actual 7.52B) via upstream llama.cpp
# Model: Q4_K_M, 5.02 GiB, 131K native context (iSWA hybrid architecture)
#
# REQUIRES: ~/src/llama.cpp-upstream built with CUDA (gemma4 arch support)
#
# iSWA KV cache breakdown (per slot, f16):
#   Non-SWA layers: ~2048 MiB per slot at 131K per-slot context
#   SWA layers: ~40 MiB per slot (ring-buffered, bounded regardless of ctx)
#
# Parallelism / context tradeoffs on 24GB:
#   parallel=4,  ctx=524288   → 131K/slot  (15615 MiB)  ← max context quality
#   parallel=8,  ctx=1048576  → 131K/slot  (23401 MiB)  ← 8 full-context users (ceiling)
#   parallel=32, ctx=131072   →   4K/slot  (10469 MiB)  ← balanced concurrency
#   parallel=64, ctx=131072   →   2K/slot  (11485 MiB)  ← high concurrency (peak 467 tok/s)

set -e

LLAMA_BIN="/home/fsabado/src/llama.cpp-upstream/build-cuda/bin"
MODEL="${1:-/home/fsabado/models/gemma-4-e4b-it/google_gemma-4-E4B-it-Q4_K_M.gguf}"
PORT="${2:-10333}"
CTX="${3:-1048576}"   # 1M total = 131K per slot at parallel=8
PARALLEL="${4:-8}"

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
  --cont-batching \
  --flash-attn on \
  --no-warmup \
  --log-disable
