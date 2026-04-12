#!/bin/bash
# Gemma 4 E2B (effective 2B, actual 4.65B) via upstream llama.cpp
# Model: Q4_K_M, 2.88 GiB, 131K native context (iSWA hybrid architecture)
#
# REQUIRES: ~/src/llama.cpp-upstream built with CUDA
#   cd ~/src/llama.cpp-upstream && mkdir -p build-cuda && cd build-cuda
#   cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
#     -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc && cmake --build . -j$(nproc)
#
# WHY NOT turboquant/animehacker builds:
#   gemma4 arch support requires llama.cpp >= v8530 (ggml v0.9.11 / commit 66c4f9ded)
#
# iSWA KV cache breakdown (per slot, f16):
#   SWA layers (12 layers, ring-buffered at 1024 tokens): bounded, ~12 MiB/slot
#   Full-attn layers (3 layers): 768 MiB per slot at 131K per-slot context
#
# Parallelism / context tradeoffs on 24GB:
#   parallel=4,  ctx=524288   → 131K/slot  (8570 MiB)   ← max context quality
#   parallel=16, ctx=2097152  → 131K/slot  (17930 MiB)  ← 16 full-context users
#   parallel=24, ctx=3145728  → 131K/slot  (~22.5 GiB)  ← ceiling at native ctx
#   parallel=64, ctx=131072   →   2K/slot  (6986 MiB)   ← max concurrency, short ctx

set -e

LLAMA_BIN="/home/fsabado/src/llama.cpp-upstream/build-cuda/bin"
MODEL="${1:-/home/fsabado/models/gemma-4-e2b-it/gemma-4-E2B-it-Q4_K_M.gguf}"
PORT="${2:-10222}"
CTX="${3:-2097152}"    # 2M total = 131K per slot at parallel=16
PARALLEL="${4:-16}"

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
