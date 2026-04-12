#!/bin/bash
# Gemma 4 E2B — 20 parallel slots, 131K context per slot (~19.3 GB VRAM)
# Practical sweet spot for agentic multi-user load on a single RTX 4090.

exec "$(dirname "$0")/start.sh" \
  /home/fsabado/models/gemma-4-e2b-it/gemma-4-E2B-it-Q4_K_M.gguf \
  10222 \
  2621440 \
  20
