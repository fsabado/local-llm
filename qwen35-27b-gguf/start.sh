#!/bin/bash
# Qwen3.5-27B Opus-Distilled-v2 Q4_K_M via llama.cpp (TurboQuant animehacker build)
# tq3_0 KV cache: 4.57x compression vs fp16, enables 262K native context on 24GB VRAM
#
# Parallelism / context tradeoffs (RS buffer = 149.6 MiB/slot is the limit):
#   parallel=4,  ctx=262144 → 65K/slot  (853 MiB free)  ← max context quality
#   parallel=8,  ctx=262144 → 32K/slot  (355 MiB free)  ← balanced
#   parallel=16, ctx=196608 → 12K/slot  (509 MiB free)  ← API concurrency
#   parallel=32, ctx=65536  →  2K/slot  (965 MiB free)  ← max parallelism (p=32 is ceiling)

set -e

LLAMA_BIN="/home/fsabado/src/llama-turboquant-animehacker/build-cuda/bin"
MODEL="${1:-$(dirname "$0")/Qwen3.5-27B.Q4_K_M.gguf}"
PORT="${2:-10111}"
CTX="${3:-262144}"      # 262K = full native context; tq3_0 KV cache makes this fit on 24GB
GPU_LAYERS="${4:-99}"  # 99 = all layers on GPU

export LD_LIBRARY_PATH="$LLAMA_BIN:$LD_LIBRARY_PATH"

echo "Starting llama-server: $(basename $MODEL) on port $PORT"
echo "Context: $CTX tokens | GPU layers: $GPU_LAYERS"
echo ""

"$LLAMA_BIN/llama-server" \
  --model "$MODEL" \
  --port "$PORT" \
  --host 0.0.0.0 \
  --ctx-size "$CTX" \
  --n-gpu-layers "$GPU_LAYERS" \
  --parallel 4 \
  --cont-batching \
  --flash-attn on \
  --cache-type-k tq3_0 \
  --cache-type-v tq3_0 \
  --jinja \
  --chat-template-file "$(dirname "$0")/chat_template.jinja" \
  --reasoning-format deepseek \
  --log-disable
