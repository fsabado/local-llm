# local-llm

Personal workspace for running, benchmarking, and optimizing LLMs locally on an RTX 4090 (24 GB VRAM, WSL2).

## Hardware & Stack

| Item | Value |
|------|-------|
| GPU | NVIDIA RTX 4090, 24 GB VRAM |
| OS | Ubuntu on WSL2 (Linux 6.6) |
| CUDA | 13.0 (`/usr/local/cuda`) |
| Inference (Qwen) | llama-turboquant-animehacker (`~/src/llama-turboquant-animehacker/build-cuda/bin`) |
| Inference (Gemma 4 f16 KV) | upstream llama.cpp (`~/src/llama.cpp-upstream/build-cuda/bin`) |
| Inference (Gemma 4 turbo4 KV) | llama-cpp-turboquant-gemma4 (`~/src/llama-cpp-turboquant-gemma4/build-cuda/bin`) |
| Inference (vLLM) | `.venv/` — shared Python env, activate with `source .venv/bin/activate` |

> **WSL2 note:** `127.0.0.1` works fine from the host shell. Inside the Claude Code sandbox
> loopback may be restricted — use the WSL2 bridge IP `172.18.0.1` from within sandbox tool calls.

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
│   ├── bench_e2b.py            ← E2B benchmark: TTFT, tok/s, parallel throughput (both quants)
│   ├── run_bench.py            ← automated: start llama-server → bench → kill server
│   ├── test.py                 ← quick smoke test for any running vLLM/llama.cpp server
│   ├── setup.sh                ← one-time vLLM venv setup (CUDA 13, nightly)
│   ├── claude-local.sh         ← launch Claude Code CLI against local Gemma model
│   ├── test_e2b_capability.py  ← 20-parallel capability test via Anthropic SDK
│   └── lm_eval_tasks/
│       └── arc_challenge_gen.yaml  ← custom lm-eval task (ARC-Challenge, generate_until)
│
├── results/                    ← benchmark output files (gitignored)
├── logs/                       ← server log files (gitignored)
│
├── gemma4-e2b/
│   ├── start.sh                ← Gemma 4 E2B Q4_K_M via upstream llama.cpp, port 10222
│   ├── launch.py               ← configurable launcher: named flags + presets + idempotent
│   └── gemma-e2b-20p.sh        ← preset: 20 slots × 131K (agentic sweet spot, ~19.3 GB)
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
| Gemma 4 E2B Q4_K_M (llama.cpp) | `gemma4-e2b/` | upstream llama.cpp | 10222 | ~6–19 GB | 136–211 |
| Gemma 4 E2B F16 (llama.cpp) | `gemma4-e2b/` | upstream llama.cpp | 10222 | ~21.6 GB | 93 (p=14 max) |
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

### Gemma 4 E2B (fastest, lowest VRAM, 131K context)

```bash
# Recommended: 20 slots × 131K — agentic sweet spot (~19.3 GB VRAM, 136 tok/s, 394 tok/s peak)
bash gemma4-e2b/gemma-e2b-20p.sh
# or with named flags / presets:
python3 gemma4-e2b/launch.py                    # default = 20p preset
python3 gemma4-e2b/launch.py --preset 20p
python3 gemma4-e2b/launch.py --list-presets     # see all presets
python3 gemma4-e2b/launch.py --restart          # kill existing + relaunch

# Manual via start.sh positional args:
#   start.sh [model] [port] [total_ctx] [parallel]
cd gemma4-e2b && bash start.sh                  # default: 16p × 131K/slot

# Max full-context parallelism — 40 slots at 131K/slot (~24 GB, iSWA ceiling)
bash start.sh /home/fsabado/models/gemma-4-e2b-it/gemma-4-E2B-it-Q4_K_M.gguf 10222 5242880 40

# Single-user max context — full 2M native context, 1 session (~21.7 GB VRAM)
bash start.sh /home/fsabado/models/gemma-4-e2b-it/gemma-4-E2B-it-Q4_K_M.gguf 10222 2097152 1

# High concurrency (64 slots, 2K per slot, ~7 GB VRAM)
bash start.sh /home/fsabado/models/gemma-4-e2b-it/gemma-4-E2B-it-Q4_K_M.gguf 10222 131072 64
```

### Gemma 4 26B A4B (highest quality, MoE 25B/4B active, port 10444)

```bash
# turbo4 KV: 4 users at full 262K native context (only possible with turbo4)
cd gemma4-26b-a4b && bash start.sh --turbo4

# turbo4 KV: 16 users at 131K/slot (~24 GB, best throughput)
bash start.sh --turbo4 /path/to/model.gguf 10444 $((131072*16)) 16

# f16 KV: 4 users at 32K/slot (simpler, no turbo4 build needed)
bash start.sh
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

# E2B benchmark — TTFT, tok/s, parallel throughput (works with any llama.cpp server)
python3 scripts/bench_e2b.py 10222 --label "Q4_K_M 20p"
python3 scripts/bench_e2b.py 10222 --parallel-levels 1 2 4 8 16 20

# Full API benchmark (TPS, parallel throughput, reasoning quality)
python3 scripts/bench.py 10222

# Automated: start server → benchmark → kill (Qwen3.5-27B GGUF variants)
python3 scripts/run_bench.py [model_filename.gguf]

# Capability test — 20 parallel requests via Anthropic SDK (code gen, debug, tools, reasoning)
.venv/bin/python3 scripts/test_e2b_capability.py
```

Benchmark results go in `results/`. Server logs go in `logs/`.

---

## Claude Code + Local LLM (Gemma 4 E2B)

Use `scripts/claude-local.sh` to launch the Claude Code CLI against any local Gemma model.
It starts `llama-server` if not running and sets all required env vars.

```bash
# Start with Gemma 4 E2B (default, ~6.6 GB VRAM, p=4, 131K/slot)
bash scripts/claude-local.sh

# Other models
bash scripts/claude-local.sh e4b          # Gemma 4 E4B
bash scripts/claude-local.sh 26b          # Gemma 4 26B A4B (f16 KV)
bash scripts/claude-local.sh 26b-t4       # Gemma 4 26B A4B (turbo4 KV)
bash scripts/claude-local.sh e2b --resume # Resume last session
```

**Required env vars** (set automatically by the script):

| Variable | Value | Why |
|---|---|---|
| `ANTHROPIC_BASE_URL` | `http://127.0.0.1:<port>` | Points Claude Code at local server |
| `ANTHROPIC_API_KEY` | `""` (empty string) | Must be explicitly empty — unset causes fallback to Anthropic API |
| `ANTHROPIC_AUTH_TOKEN` | `"local"` | Dummy token accepted by llama-server |
| `ANTHROPIC_DEFAULT_HAIKU_MODEL` | model filename | Prevents fallback to `claude-haiku-4-5-*` |
| `ANTHROPIC_DEFAULT_SONNET_MODEL` | model filename | Prevents fallback to `claude-sonnet-4-5-*` |
| `ANTHROPIC_DEFAULT_OPUS_MODEL` | model filename | Prevents fallback to `claude-opus-4-5-*` |
| `DISABLE_TELEMETRY` | `1` | Stops background calls to `api.anthropic.com` |

> **Root cause of "model may not exist" error:** Claude Code uses three model tiers internally
> (Haiku / Sonnet / Opus). Without all three `ANTHROPIC_DEFAULT_*` vars set, background tasks
> fall back to Anthropic's model names which the local server doesn't recognize → 404 → error.

**llama-server flags required:** `--jinja` for correct Gemma 4 chat template handling.

### Anthropic Python SDK

```python
import anthropic

client = anthropic.Anthropic(
    api_key="local",
    base_url="http://127.0.0.1:10222",
)

msg = client.messages.create(
    model="gemma-4-E2B-it-Q4_K_M.gguf",
    max_tokens=131072,   # use full slot budget; model stops when done
    messages=[{"role": "user", "content": "your prompt here"}],
)

for block in msg.content:
    if block.type == "thinking":
        print("Thinking:", block.thinking[:200])
    elif block.type == "text":
        print("Answer:", block.text)
```

**Important:** `max_tokens` covers thinking + output combined. Use ≥ 4096 or the full
131072 — if too small (e.g. 1024), reasoning exhausts the budget and `text` blocks are empty.

Tool calling works via `tools=` parameter. Install SDK: `pip install anthropic` (use `.venv/`).

---

## Key Findings (see docs/LEARNINGS.md for full detail)

### Gemma 4 26B A4B (llama.cpp) — highest quality
- **MoE:** 25.23B total, ~4B active per token — generation speed close to dense 4B model
- **f16 KV:** ~83 tok/s single, p=4 ceiling (20.5 GB base leaves only ~4 GB for KV); ctx cliff at 131K total
- **turbo4 KV** (`llama-cpp-turboquant-gemma4`): ~114 tok/s, **p=50+ ceiling** (iSWA plateau), 4 users at native 262K context
- turbo4 saves 3.8 GB KV at 256K context — unlocks full native context for multi-user
- Port: **10444**

### Gemma 4 E4B (llama.cpp) — balanced
- **~100 tok/s** single-request, ~467 tok/s peak
- Max parallel at full 131K per slot: **p=8** (~23.4 GB VRAM)
- Port: **10333**

### Gemma 4 E2B (llama.cpp) — fastest / lowest VRAM
- **Q4_K_M: 136 tok/s** single (server), **211 tok/s** raw (llama-bench) — 4× faster than Qwen3.5-27B
- **F16: 93 tok/s** single (server), **108 tok/s** raw — use Q4_K_M instead (see below)
- **iSWA architecture**: sliding window bounded at 1024 tokens (most layers), only 3 full-attention layers → KV cache barely scales with context
- 131K native context; single-slot max is **2M tokens** (21.7 GB VRAM)
- Max parallel at 131K/slot: **p=40** on 24 GB — VRAM plateaus ~24 GB from p=28 onward
- **p=20 at 131K/slot = 19.3 GB VRAM** — recommended sweet spot; 394 tok/s aggregate peak at n=16
- TTFT: ~60ms single-user, ~209ms at n=16 concurrency
- Anthropic SDK: `ThinkingBlock` + `TextBlock` in `msg.content`; use `max_tokens ≥ 4096`
- **SSE note**: thinking tokens stream as `delta.reasoning_content`, not `delta.content` — benchmarks must count both
- **Tool calling works** (single tool per turn); no multi-tool chaining in one response
- **Build requirement**: upstream llama.cpp ≥ ggml v0.9.11 (`gemma4` arch added ~April 2026)

#### F16 vs Q4_K_M (E2B)
Q4_K_M wins on every practical metric — do not use F16 for inference:

| | Q4_K_M | F16 |
|---|---|---|
| Size | 2.88 GiB | 8.66 GiB |
| Single tok/s | **136** | 93 |
| PP tok/s (llama-bench) | **14,115** | 2,838 |
| Max practical slots @ 131K | **20** (4.3 GB free) | 14 (2.6 GB free) |
| Peak agg tok/s | **394** | 383 |

F16 at 18 slots fills VRAM to 97% (97 MiB free) — CUDA compute buffers starve → requests take 48s. Must leave ≥2 GB free. Max practical is p=14.

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

### upstream llama.cpp (Gemma 4, f16 KV)
```bash
cd ~/src/llama.cpp-upstream
cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build-cuda --target llama-server llama-bench -j$(nproc)
```

### llama-cpp-turboquant-gemma4 (Gemma 4 + turbo4 KV, recommended)
```bash
cd ~/src/llama-cpp-turboquant-gemma4  # git clone test1111111111111112/llama-cpp-turboquant-gemma4
cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
cmake --build build-cuda --target llama-server llama-bench -j$(nproc)
# ~90s build time
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
