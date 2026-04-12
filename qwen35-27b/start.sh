#!/bin/bash
# Qwen3.5-27B Opus-Distilled-v2 via vLLM — RTX 4090 (24GB) with AWQ INT4 quantization
# Key mitigations:
#   - Qwen2TokenizerFast patch in tokenizer_config.json (TokenizersBackend bug)
#   - gpu-memory-utilization 0.78 to avoid Triton autotuner OOM on first request
#   - max-num-batched-tokens 2112 (GDN block size minimum)
#   - enforce-eager to avoid CUDA graph capture failures with AWQ+GDN layers
#   - kv-cache-dtype fp8 to save VRAM for KV cache

set -e

MODEL="${1:-QuantTrio/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2-AWQ}"
PORT="${2:-10111}"
MAX_LEN="${3:-4096}"

export OMP_NUM_THREADS=4

echo "Starting vLLM server: $MODEL on port $PORT"
echo "Max context: $MAX_LEN tokens"
echo ""

source "$(dirname "$0")/../.venv/bin/activate"

vllm serve "$MODEL" \
  --served-model-name qwen35-27b \
  --port "$PORT" \
  --host 0.0.0.0 \
  --dtype float16 \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 2112 \
  --quantization awq \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --enforce-eager \
  --skip-mm-profiling \
  --trust-remote-code
