#!/bin/bash
# Run Claude Code CLI against a local llama-server
#
# Usage:
#   claude-local.sh [model] [extra claude args...]
#
# Models:
#   e2b       Gemma 4 E2B  — port 10222, p=4, 131K/slot  (~6.6 GB, 155 tok/s)  [default]
#   e4b       Gemma 4 E4B  — port 10333, p=4, 131K/slot  (~15 GB, 103 tok/s)
#   26b       Gemma 4 26B  — port 10444, p=4, 131K/slot  (~22 GB, 83 tok/s)
#   26b-t4    Gemma 4 26B  — port 10444, p=4, 262K/slot  turbo4 KV
#
# Examples:
#   claude-local.sh              # start with Gemma E2B (default)
#   claude-local.sh e4b          # start with E4B
#   claude-local.sh 26b-t4       # start with 26B turbo4
#   claude-local.sh e2b --resume # resume last session
#
# Key env vars (set by this script):
#   ANTHROPIC_BASE_URL            — points to llama-server's /v1/messages endpoint
#   ANTHROPIC_AUTH_TOKEN          — dummy value accepted by llama-server
#   ANTHROPIC_API_KEY             — must be "" (empty) to prevent real API calls
#   ANTHROPIC_DEFAULT_*_MODEL     — all three tiers mapped to local model (prevents
#                                   Claude Code from falling back to claude-haiku-* etc)
#   DISABLE_TELEMETRY             — stops background calls to api.anthropic.com

set -e

MODEL="${1:-e2b}"
shift 2>/dev/null || true   # remaining args passed to claude

LLAMA_BIN_UPSTREAM="/home/fsabado/src/llama.cpp-upstream/build-cuda/bin"
LLAMA_BIN_TURBO4="/home/fsabado/src/llama-cpp-turboquant-gemma4/build-cuda/bin"
MODELS_DIR="/home/fsabado/models"

case "$MODEL" in
  e2b)
    BIN="$LLAMA_BIN_UPSTREAM"
    GGUF="$MODELS_DIR/gemma-4-e2b-it/gemma-4-E2B-it-Q4_K_M.gguf"
    PORT=10222; CTX=524288; PAR=4; KV_FLAGS=""
    MODEL_ID="gemma-4-E2B-it-Q4_K_M.gguf"
    DESC="Gemma 4 E2B Q4_K_M — 131K/slot × 4 — ~155 tok/s"
    ;;
  e4b)
    BIN="$LLAMA_BIN_UPSTREAM"
    GGUF="$MODELS_DIR/gemma-4-e4b-it/google_gemma-4-E4B-it-Q4_K_M.gguf"
    PORT=10333; CTX=524288; PAR=4; KV_FLAGS=""
    MODEL_ID="google_gemma-4-E4B-it-Q4_K_M.gguf"
    DESC="Gemma 4 E4B Q4_K_M — 131K/slot × 4 — ~103 tok/s"
    ;;
  26b)
    BIN="$LLAMA_BIN_UPSTREAM"
    GGUF="$MODELS_DIR/gemma-4-26b-a4b-it/google_gemma-4-26B-A4B-it-Q4_K_M.gguf"
    PORT=10444; CTX=131072; PAR=4; KV_FLAGS=""
    MODEL_ID="google_gemma-4-26B-A4B-it-Q4_K_M.gguf"
    DESC="Gemma 4 26B A4B Q4_K_M — 32K/slot × 4 — ~83 tok/s"
    ;;
  26b-t4)
    BIN="$LLAMA_BIN_TURBO4"
    GGUF="$MODELS_DIR/gemma-4-26b-a4b-it/google_gemma-4-26B-A4B-it-Q4_K_M.gguf"
    PORT=10444; CTX=1048576; PAR=4; KV_FLAGS="--cache-type-k turbo4 --cache-type-v turbo4"
    MODEL_ID="google_gemma-4-26B-A4B-it-Q4_K_M.gguf"
    DESC="Gemma 4 26B A4B Q4_K_M — turbo4 KV — 262K/slot × 4 — ~114 tok/s"
    ;;
  *)
    echo "Unknown model '$MODEL'. Choose: e2b, e4b, 26b, 26b-t4" >&2
    exit 1
    ;;
esac

LLAMA_URL="http://127.0.0.1:${PORT}"
export LD_LIBRARY_PATH="$BIN:$LD_LIBRARY_PATH"

# ── Start llama-server if not already running ─────────────────────────────────
if curl -sf "${LLAMA_URL}/v1/models" > /dev/null 2>&1; then
  echo "✓ llama-server already running on port $PORT"
else
  echo "Starting: $DESC"
  nohup "$BIN/llama-server" \
    --model "$GGUF" \
    --port "$PORT" \
    --host 0.0.0.0 \
    -ngl 99 \
    --ctx-size "$CTX" \
    --parallel "$PAR" \
    $KV_FLAGS \
    --cont-batching \
    --flash-attn on \
    --jinja \
    --no-warmup \
    --log-disable > "/tmp/llama-${MODEL}.log" 2>&1 &

  echo -n "Waiting for llama-server"
  for i in $(seq 1 60); do
    sleep 2
    if curl -sf "${LLAMA_URL}/v1/models" > /dev/null 2>&1; then
      echo " ready (${i}×2s)"
      break
    fi
    echo -n "."
    if [ $i -eq 60 ]; then
      echo " TIMEOUT — check /tmp/llama-${MODEL}.log" >&2
      exit 1
    fi
  done
fi

VRAM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null || echo "?")
echo "Model : $DESC"
echo "VRAM  : $VRAM"
echo "URL   : ${LLAMA_URL}/v1"
echo ""

# ── Launch Claude Code against local server ───────────────────────────────────
# Claude Code uses 3 model tiers internally (Haiku / Sonnet / Opus).
# All three must point to the local model or it falls back to claude-haiku-* etc.
export ANTHROPIC_AUTH_TOKEN="local"
export ANTHROPIC_API_KEY=""          # must be empty string, not just unset
export ANTHROPIC_BASE_URL="${LLAMA_URL}"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="$MODEL_ID"
export ANTHROPIC_DEFAULT_SONNET_MODEL="$MODEL_ID"
export ANTHROPIC_DEFAULT_OPUS_MODEL="$MODEL_ID"
export DISABLE_TELEMETRY=1           # prevents background calls to api.anthropic.com
export MAX_THINKING_TOKENS="0"

exec claude --model "$MODEL_ID" "$@"
