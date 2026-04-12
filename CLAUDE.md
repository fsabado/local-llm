# local-llm

Personal workspace for running, benchmarking, and optimizing LLMs locally on an RTX 4090 (24 GB VRAM, WSL2).

## Hardware & Stack

| Item | Value |
|------|-------|
| GPU | NVIDIA RTX 4090, 24 GB VRAM |
| OS | Ubuntu on WSL2 (Linux 6.6) |
| CUDA | 13.0 (`/usr/local/cuda`) |
| Inference (Qwen) | llama-turboquant-animehacker (`~/src/llama-turboquant-animehacker/build-cuda/bin`) |
| Inference (Gemma 4) | upstream llama.cpp (`~/src/llama.cpp-upstream/build-cuda/bin`) |
| Inference (vLLM) | `.venv/` — shared Python env, activate with `source .venv/bin/activate` |

> **WSL2 note:** loopback (`127.0.0.1`) TCP is blocked inside the Claude Code sandbox.
> Always use the WSL2 bridge IP `172.18.0.1` when connecting to local servers.

---

## Directory Layout

```
local-llm/
├── CLAUDE.md                   ← you are here
│
├── docs/
│   └── LEARNINGS.md            ← experiment log: configs, results, key insights
│
├── scripts/
│   ├── bench.py                ← API-level benchmark (TPS, parallel, reasoning quality)
│   ├── run_bench.py            ← automated: start llama-server → bench → kill server
│   ├── test.py                 ← quick smoke test for any running vLLM/llama.cpp server
│   ├── setup.sh                ← one-time vLLM venv setup (CUDA 13, nightly)
│   └── lm_eval_tasks/
│       └── arc_challenge_gen.yaml  ← custom lm-eval task (ARC-Challenge, generate_until)
│
├── results/                    ← benchmark output files (gitignored)
├── logs/                       ← server log files (gitignored)
│
├── gemma4-e2b/
│   └── start.sh                ← Gemma 4 E2B Q4_K_M via upstream llama.cpp, port 10222
│
├── gemma4-e4b/
│   └── start.sh                ← Gemma 4 E4B Q4_K_M via upstream llama.cpp, port 10333
│
├── gemma4-26b-a4b/
│   └── start.sh                ← Gemma 4 26B A4B Q4_K_M via upstream llama.cpp, port 10444
│
├── qwen35-27b/
│   └── start.sh                ← Qwen3.5-27B AWQ via vLLM, port 10111
│
└── qwen35-27b-gguf/
    ├── start.sh                ← Qwen3.5-27B Q4_K_M via llama-turboquant-animehacker, port 10111
    ├── chat_template.jinja     ← thinking enabled (injects <think> in generation prompt)
    └── chat_template_nothink.jinja  ← thinking disabled
```

---

## Models

| Model | Dir | Backend | Port | VRAM | Single tok/s |
|-------|-----|---------|------|------|-------------|
| Qwen3.5-27B AWQ (vLLM) | `qwen35-27b/` | vLLM | 10111 | ~20 GB | 14 |
| Qwen3.5-27B Q4_K_M (llama.cpp) | `qwen35-27b-gguf/` | llama-turboquant-animehacker | 10111 | ~15 GB | 40 |
| Gemma 4 E4B (vLLM) | `gemma4-e4b/` | vLLM | 8001 | ~8 GB | — |
| Gemma 4 E4B Q4_K_M (llama.cpp) | `gemma4-e4b/` | upstream llama.cpp | 10333 | ~10–23 GB | 100–105 |
| Gemma 4 E2B Q4_K_M (llama.cpp) | `gemma4-e2b/` | upstream llama.cpp | 10222 | ~6–18 GB | 155–167 |
| Gemma 4 26B A4B Q4_K_M (llama.cpp) | `gemma4-26b-a4b/` | upstream llama.cpp | 10444 | ~22–24 GB | 83–90 |

> Only one model at a time on a single 24 GB GPU (except Gemma 4 E2B/E4B, which leave headroom).

---

## Quick Start

### Gemma 4 E4B via llama.cpp (better quality, 131K context, port 10333)

```bash
# Default: 8 parallel slots, 131K context per slot (~23.4 GB VRAM — ceiling on 24GB)
cd gemma4-e4b && bash start.sh

# Single-user max context (4 slots, 131K per slot, ~15.6 GB VRAM)
bash start.sh /home/fsabado/models/gemma-4-e4b-it/google_gemma-4-E4B-it-Q4_K_M.gguf 10333 524288 4

# High concurrency (64 slots, 2K per slot, ~11.5 GB VRAM)
bash start.sh /home/fsabado/models/gemma-4-e4b-it/google_gemma-4-E4B-it-Q4_K_M.gguf 10333 131072 64
```

### Gemma 4 E2B (fastest — 160 tok/s, low VRAM, 131K context)

```bash
# Default: 16 parallel slots, 131K context per slot (~18 GB VRAM)
cd gemma4-e2b && bash start.sh

# Max full-context parallelism — 40 slots at 131K/slot (~24 GB, iSWA ceiling)
bash start.sh /home/fsabado/models/gemma-4-e2b-it/gemma-4-E2B-it-Q4_K_M.gguf 10222 5242880 40

# Single-user max context — full 2M native context, 1 session (~21.7 GB VRAM)
bash start.sh /home/fsabado/models/gemma-4-e2b-it/gemma-4-E2B-it-Q4_K_M.gguf 10222 2097152 1

# High concurrency (64 slots, 2K per slot, ~7 GB VRAM)
bash start.sh /home/fsabado/models/gemma-4-e2b-it/gemma-4-E2B-it-Q4_K_M.gguf 10222 131072 64
```

### Gemma 4 26B A4B (highest quality, MoE 25B/4B active, port 10444)

```bash
# Default: 4 parallel at 32K/slot — 188 tok/s peak (~22 GB VRAM)
cd gemma4-26b-a4b && bash start.sh

# Single-user, small ctx — max speed (~90 tok/s, ~21 GB VRAM)
bash start.sh /home/fsabado/models/gemma-4-26b-a4b-it/google_gemma-4-26B-A4B-it-Q4_K_M.gguf 10444 4096 1

# ⚠️  DO NOT use ctx*parallel > 131072 — causes 7× slowdown
```

### Qwen3.5-27B Q4_K_M via llama.cpp (best quality, tq3_0 KV cache)

```bash
cd qwen35-27b-gguf && bash start.sh
```

### Qwen3.5-27B via vLLM (tool-call parser, reasoning parser)

```bash
source .venv/bin/activate
cd qwen35-27b && bash start.sh
```

---

## Benchmarking

```bash
# Smoke test — check a running server (pass port as arg)
python3 scripts/test.py 10222

# Full API benchmark (TPS, parallel throughput, reasoning quality)
python3 scripts/bench.py 10222

# Automated: start server → benchmark → kill (Qwen3.5-27B GGUF variants)
python3 scripts/run_bench.py [model_filename.gguf]
```

Benchmark results go in `results/`. Server logs go in `logs/`.

---

## Key Findings (see docs/LEARNINGS.md for full detail)

### Gemma 4 26B A4B (llama.cpp) — highest quality
- **MoE:** 25.23B total, ~4B active per token — generation speed close to dense 4B model
- **~83 tok/s** single-request, **~188 tok/s** peak aggregate at p=4
- **Context cliff:** ctx×parallel > 131K causes **>7× slowdown** (full-attention layer KV scaling)
- Best config: `--ctx-size 131072 --parallel 4` (32K/slot, ~22 GB VRAM)
- Parallelism ceiling: **p=4** at 131K/slot — model base consumes 20.5 GB, leaving only ~4 GB for KV
- Port: **10444**

### Gemma 4 E4B (llama.cpp) — balanced
- **~100 tok/s** single-request, ~467 tok/s peak
- Max parallel at full 131K per slot: **p=8** (~23.4 GB VRAM)
- Port: **10333**

### Gemma 4 E2B (llama.cpp) — fastest / lowest VRAM
- **~160 tok/s** single-request — 4× faster than Qwen3.5-27B at single-request
- **iSWA architecture**: sliding window bounded at 1024 tokens (most layers), only 3 full-attention layers → KV cache barely scales with context
- 131K native context; single-slot max is **2M tokens** (21.7 GB VRAM)
- Max parallel at 131K/slot: **p=40** on 24 GB — VRAM plateaus ~24 GB from p=28 onward (each extra slot only pays for 1024-token SWA window, not full 131K)
- `reasoning_content` field for thinking (same as DeepSeek format); needs `max_tokens ≥ 1024`
- **Build requirement**: upstream llama.cpp ≥ ggml v0.9.11 (`gemma4` arch added ~April 2026)

### Qwen3.5-27B (llama.cpp, tq3_0 KV cache)
- **~40 tok/s** single-request, **113 tok/s** peak at n=8
- tq3_0 KV compression (4.57×) → up to 65K context per slot on 24 GB
- GDN recurrent state = 149.6 MiB/slot → **p=32 is the ceiling**
- Use llama-turboquant-animehacker build (vLLM is 3× slower due to `--enforce-eager` + AWQ+GDN conflict)

### Qwen3.5-27B (vLLM)
- `--enforce-eager` mandatory for AWQ + GDN hybrid layers — CUDA graphs cause OOM or are 3× slower
- FP8 KV cache (`--kv-cache-dtype fp8`) doubles context to ~8K, 100s warmup on first request
- Best multi-user vLLM config: FP8 + p=8 + 7840 ctx

---

## Build References

### upstream llama.cpp (required for Gemma 4)
```bash
cd ~/src/llama.cpp-upstream
mkdir -p build-cuda && cd build-cuda
cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build . --target llama-server llama-bench -j$(nproc)
```

### llama-turboquant-animehacker (Qwen3.5-27B tq3_0)
```bash
cd ~/src/llama-turboquant-animehacker
mkdir -p build-cuda && cd build-cuda
cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build . -j$(nproc)
```

### vLLM environment
```bash
bash scripts/setup.sh   # run once from repo root
source .venv/bin/activate
```
