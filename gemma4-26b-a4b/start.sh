#!/bin/bash
# Gemma 4 26B A4B (effective 4B active, 25.23B total MoE) via upstream llama.cpp
#
# Architecture: 30 layers — 25 SWA (window=1024) + 5 full-attention (every 6th layer)
#   MoE: 128 experts, 8 active per token
#   SWA layers: 8 KV heads @ head_dim=256
#   Full-attn layers: 2 KV heads @ head_dim=512 (GQA ratio = 8)
#
# VRAM breakdown (Q4_K_M, 24 GB card):
#   Model:            20,539 MiB  (~15.9 GB file + GPU overhead)
#   KV @ ctx=131K p=1:  2,385 MiB
#   KV @ ctx=524K p=4: 23,669 MiB total — ceiling on 24 GB
#
# Performance (observed on RTX 4090):
#   Raw:  pp=4404 tok/s (2048 tokens), tg=117 tok/s
#   API (ctx=131K total, p=4, 32K/slot):
#     single:  ~83 tok/s, ~3s latency
#     c=4 agg: ~188 tok/s
#   ⚠️  CLIFF: ctx_total > 131K causes >7x generation slowdown (10 tok/s)
#      Stay below 131K total context (ctx * parallel).
#
# Port assignments:  10444 (default)

LLAMA_BIN="/home/fsabado/src/llama.cpp-upstream/build-cuda/bin"
MODEL="${1:-/home/fsabado/models/gemma-4-26b-a4b-it/google_gemma-4-26B-A4B-it-Q4_K_M.gguf}"
PORT="${2:-10444}"
CTX="${3:-131072}"   # total = 131K = 32K per slot at parallel=4
PARALLEL="${4:-4}"

export LD_LIBRARY_PATH="$LLAMA_BIN:$LD_LIBRARY_PATH"

echo "Starting llama-server: $(basename $MODEL) on port $PORT"
echo "Context: $CTX total | Parallel: $PARALLEL | Per-slot: $((CTX / PARALLEL))"
echo "WARNING: keep ctx*parallel <= 131072 to avoid 7x speed cliff"
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
