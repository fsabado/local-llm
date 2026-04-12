#!/bin/bash
# Gemma4 E4B (effective 4B) via vLLM — RTX 4090 (24GB)
# Model: ~8GB VRAM BF16, 128K context, multimodal (text+image+audio)
# Fast decode, small footprint, great for agentic/coding tasks

set -e

MODEL="${1:-google/gemma-4-E4B-it}"
PORT="${2:-8001}"
MAX_LEN="${3:-65536}"

echo "Starting vLLM server: $MODEL on port $PORT"
echo "Max context: $MAX_LEN tokens"
echo ""

source "$(dirname "$0")/../.venv/bin/activate"

vllm serve "$MODEL" \
  --port "$PORT" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization 0.85 \
  --enable-prefix-caching \
  --reasoning-parser gemma4 \
  --tool-call-parser gemma4 \
  --chat-template examples/tool_chat_template_gemma4.jinja \
  --host 0.0.0.0
