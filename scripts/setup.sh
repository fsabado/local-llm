#!/bin/bash
# Setup shared vLLM environment for local-llm
# Run once from /home/fsabado/src/local-llm/
# Requires: CUDA 13.0+, RTX 4090

set -e

echo "=== Setting up shared vLLM environment ==="
echo "CUDA version detected: $(nvcc --version 2>/dev/null | grep release || echo 'check nvidia-smi')"
echo ""

# Create shared venv (both models use this)
uv venv --python 3.12 .venv
source .venv/bin/activate

echo "Installing vLLM nightly (cu130 for CUDA 13.x)..."
uv pip install -U vllm --pre \
  --extra-index-url https://wheels.vllm.ai/nightly/cu130 \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  --index-strategy unsafe-best-match

echo "Installing transformers (pinned for Gemma4 compatibility)..."
uv pip install "transformers>=5.5.0"

echo "Installing extras for Gemma4 audio support..."
uv pip install "vllm[audio]"

echo ""
echo "=== Setup complete ==="
echo "Activate with: source .venv/bin/activate"
echo ""
echo "To start Qwen3.5-27B (port 10111): cd qwen35-27b  && bash start.sh"
echo "To start Gemma4 E4B  (port 8001):  cd gemma4-e4b  && bash start.sh"
echo ""
echo "NOTE: Only one model at a time on a single 24GB GPU."
echo "      Kill the current llama-server first: kill \$(pgrep llama-server)"
