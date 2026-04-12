# Local LLM — Learnings & Optimization Log

**Hardware:** NVIDIA RTX 4090 (24 GB VRAM), WSL2, CUDA 13.0  
**Stack:** vLLM nightly `0.19.1rc1.dev211+gbd8bd5230`, PyTorch 2.11+cu130  
**Model:** `QuantTrio/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2-AWQ`

---

## Model Notes

Qwen3.5-27B is a **hybrid architecture** — different from standard transformers:
- Alternates between `linear_attention` (GDN / Gated Delta Network) and `full_attention` layers (every 4th layer)
- 64 layers total, 16 full attention + 48 GDN linear attention
- Multimodal (text + vision encoder), even when used text-only
- AWQ INT4 quantization → ~19.77 GiB loaded on GPU

The **Opus-Distilled-v2** fine-tune trains Qwen3.5-27B to reason in the style of Claude Opus 4.6,
using knowledge distillation from Opus reasoning traces.

---

## Setup Issues & Fixes

### 1. TokenizersBackend error ❌ → ✅

```
ValueError: Tokenizer class TokenizersBackend does not exist
```

**Cause:** `tokenizer_config.json` uses `TokenizersBackend`, a class only in `transformers>=5.x`.
vLLM bundles transformers 4.x.

**Fix:** Patch the cached blob directly:

```python
import json
path = '~/.cache/huggingface/hub/models--QuantTrio--Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2-AWQ/blobs/<hash>'
with open(path) as f:
    d = json.load(f)
d['tokenizer_class'] = 'Qwen2TokenizerFast'
with open(path, 'w') as f:
    json.dump(d, f, indent=2)
```

Also required: `--trust-remote-code` flag.

---

### 2. `--limit-mm-per-prompt` format changed ❌ → ✅

Old format `"image=4,video=1"` no longer valid in vLLM nightly.  
**Fix:** Just remove the flag. The default handles multimodal limits.

---

### 3. AWQ requires float16, not bfloat16 ❌ → ✅

```
ValueError: torch.bfloat16 is not supported for quantization method awq. Supported dtypes: [torch.float16]
```

**Fix:** Use `--dtype float16` (not bfloat16).

---

### 4. Vision encoder profiling OOM ❌ → ✅

```
ValueError: No available memory for the cache blocks.
```

**Cause:** On startup vLLM profiles the vision encoder with a max-feature-size image.
This peaks GPU memory above what's available for KV cache blocks, so 0 blocks are allocated.

**Fix:** `--skip-mm-profiling` — skips the multimodal encoder profiling pass.  
This is safe for text-only usage. Vision inputs may under-allocate memory at runtime.

---

### 5. GDN block size forces large KV blocks

vLLM logs: `Setting attention block size to 1568 tokens`

The GDN/Mamba page size forces attention block size to 1568 tokens (vs typical 16-32).
This means each KV cache block is ~49–98 MB depending on dtype.
With a 27B AWQ model using ~19.77 GiB, only ~2–3 GiB remains for KV cache at 0.92 utilization,
giving ~20–30 blocks = enough for 4096–8192 context.

---

### 6. loopback TCP blocked in Claude Code sandbox

The Claude Code Bash tool blocks loopback (`127.0.0.1`) TCP connections.  
**Workaround:** Use the WSL2 bridge IP `172.18.0.1` instead of `localhost`.

---

## Baseline Configuration (Working)

```bash
vllm serve QuantTrio/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2-AWQ \
  --served-model-name qwen35-27b \
  --port 10111 \
  --host 0.0.0.0 \
  --dtype float16 \
  --max-model-len 4096 \
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
```

---

## Baseline Performance

**Single request:** ~14–15 tok/s decode  
**First request:** ~5 tok/s (Triton kernel autotuning on first inference)

| Concurrent requests | Total tok/s | Per-request tok/s | Avg latency |
|--------------------|-------------|-------------------|-------------|
| 1 | 11.8 | 11.8 | 2.0s |
| 2 | 19.1 | 9.6 | 2.4s |
| 4 | 36.9 | 9.2 | 2.5s |
| 6 | 41.5 | 6.9 | 3.3s |
| 8 (queued) | 54.2 | 6.8 | 12.7s |
| 12 (queued) | 50.4 | 4.2 | 17.8s |

**Sweet spot:** 4–6 concurrent requests (~3x throughput vs single)  
**Bottleneck:** `--enforce-eager` disables CUDA graphs — expected ~3–4x performance penalty vs CUDA graph mode

---

## Optimizations Log

### OPT-1a: Partial CUDA graphs (torch.compile + FULL_AND_PIECEWISE) ❌ WORSE
**Result:** 4.7 tok/s single — **3x SLOWER** than baseline  
**Why it failed:** torch.compile (inductor) on GDN hybrid layers takes 66 seconds to compile
and the compiled kernels are slower than PyTorch eager for this hybrid architecture.
CUDA graph capture itself succeeded (`PIECEWISE=7, FULL=3`) but inductor overhead dominates.

```bash
# What was tried:
--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,4,8,16,24,32],"max_cudagraph_capture_size":32}'
# (no --enforce-eager)
```

| n | total tok/s | vs baseline |
|---|-------------|------------|
| 1 | 4.5 | -69% |
| 4 | 8.3 | -78% |
| 8 | 8.7 | -84% |

---

### OPT-1b: No `--enforce-eager`, default CUDA graphs (no torch.compile) ❌ OOM
**Result:** KV cache OOM — max context drops to 2352 tokens  
**Why it failed:** Without `--enforce-eager`, vLLM runs a CUDA graph warmup forward pass
that temporarily peaks GPU memory higher, leaving insufficient space for KV blocks.

---

### OPT-2: Increase `--max-num-seqs` to 8 (standalone) ❌ NOT TESTED
Bundled with OPT-3 below since FP8 is needed to free KV cache memory for more seqs.

---

### OPT-3: FP8 KV cache + 8192 context ✅ WORKS (with caveats)

```bash
--kv-cache-dtype fp8 \
--max-model-len 8192 \
--max-num-seqs 4 \
```

**Result:** Context doubled (4096 → 8192), same single-request speed.  
**Caveat:** First-request warmup is 104 seconds (FP8 Triton kernels compile on first use).
Subsequent requests are normal. Parallel throughput at n=4 slightly worse than baseline.

| n | total tok/s | vs baseline | note |
|---|-------------|------------|------|
| 1 | 12.8 | -9% | |
| 2 | 22.1 | +16% | |
| 4 | 24.6 | -33% | |
| 6 | 38.5 | -7% | |

---

### OPT-2+3: FP8 + max-num-seqs=8 + 7840 context ✅ BEST MULTI-USER CONFIG

```bash
--kv-cache-dtype fp8 \
--max-model-len 7840 \
--max-num-seqs 8 \
--max-num-batched-tokens 2112 \
```

**Result:** 2x the concurrent capacity (8 vs 4), near-baseline throughput, ~2x context vs baseline.
No queueing at n=8 (vs baseline which queued at n>4).

| n | total tok/s | avg latency | vs baseline n=4 |
|---|-------------|-------------|----------------|
| 1 | 15.2 | 1.5s | — |
| 2 | 24.2 | 1.9s | — |
| 4 | 26.1 | 2.4s | -29% total tok/s |
| 6 | 35.4 | 2.9s | — |
| 8 | 35.7 | 3.3s | no queueing ✓ |

**Trade-off:** Total throughput at n=4 drops ~30% vs baseline (FP8 attention overhead),
but n=8 runs without queueing at 3.3s avg latency vs 12.7s queued in baseline.

---

### OPT-4: Context > 8192 (16384+) ❌ NOT VIABLE on 24GB
Even with FP8, the 1568-token block size and 19.77 GiB model weight leave <1 GiB for KV cache.
16K context would require ~1.8 GiB KV with FP8 — not available at 0.92 utilization.
Would need a smaller model or a 48GB GPU.

---

### OPT-5: Speculative decoding (MTP) ❌ OOM
**Result:** MTP draft head adds ~0.8 GiB to model, leaving only 0.04 GiB for KV cache.  
**Why it failed:** `Qwen3_5MTP` draft model shares embeddings/lm_head but still has its own
hidden layers that consume significant VRAM. Not viable on a single 24GB GPU with this model.

```bash
# Tried:
--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
# Error: available KV cache memory (0.04 GiB) < needed (0.93 GiB)
```

---

## Final Recommended Config

### Single-user / low latency (baseline)
```bash
vllm serve QuantTrio/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2-AWQ \
  --served-model-name qwen35-27b \
  --port 10111 --host 0.0.0.0 \
  --dtype float16 \
  --max-model-len 4096 \
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
```
**Performance:** ~14 tok/s single, ~37 tok/s at n=4 | Context: 4096 | Seqs: 4

### Multi-user / larger context (OPT-2+3)
```bash
vllm serve QuantTrio/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2-AWQ \
  --served-model-name qwen35-27b \
  --port 10111 --host 0.0.0.0 \
  --dtype float16 \
  --max-model-len 7840 \
  --gpu-memory-utilization 0.92 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 2112 \
  --quantization awq \
  --kv-cache-dtype fp8 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --enforce-eager \
  --skip-mm-profiling \
  --trust-remote-code
```
**Performance:** ~15 tok/s single, ~36 tok/s at n=8 | Context: 7840 | Seqs: 8  
**Note:** First request after cold start takes ~100s due to FP8 Triton compilation.

---

---

## llama.cpp (GGUF Q4_K_M) — Setup & Evolution

### Initial Setup (q8_0 KV cache)

```bash
# Download
huggingface-cli download Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2-GGUF \
  Qwen3.5-27B.Q4_K_M.gguf --local-dir ./qwen35-27b-gguf

# Run (initial — q8_0 KV, 32K ctx)
/path/to/llama-server \
  --model Qwen3.5-27B.Q4_K_M.gguf \
  --port 10111 --host 0.0.0.0 \
  --ctx-size 32768 \
  --n-gpu-layers 99 \
  --parallel 4 \
  --cont-batching \
  --flash-attn on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --jinja \
  --chat-template-file chat_template.jinja \
  --reasoning-format deepseek \
  --log-disable
```

**Notes:**
- `--reasoning-format deepseek` puts thinking in `reasoning_content` field (not `reasoning`)
- `--flash-attn on` must be explicit (not `-fa`) due to optional-value parsing bug
- No warmup penalty — cold start is instant (~1s first request vs 104s for FP8 vLLM)
- VRAM: ~19.4 GB loaded, 4.8 GB free for KV cache

### Initial Performance (q8_0 / 32K ctx / parallel=4)

| Concurrent | Total tok/s | Per-req tok/s | Latency |
|-----------|-------------|---------------|---------|
| 1 | 39.7 | 39.7 | 3.8s |
| 2 | 69.4 | 34.7 | 4.3s |
| 4 | 109.2 | 27.3 | 5.5s |
| 6 | 91.4 | 15.2 | 6.8s |
| 8 | 113.3 | 14.2 | 7.9s |

### llama.cpp vs vLLM comparison

| Metric | vLLM AWQ (baseline) | vLLM FP8 8-seqs | llama.cpp Q4_K_M |
|--------|--------------------|--------------------|-----------------|
| Single tok/s | 14.3 | 14.6 | **39.6** (+177%) |
| Peak total tok/s | 41.5 @ n=6 | 35.7 @ n=8 | **113.3 @ n=8** |
| Max context | 4096 | 7840 | **32768** |
| VRAM model | 19.77 GB | 19.77 GB | **15.4 GB** |
| VRAM free | ~2 GB | ~2 GB | **~5 GB** |
| Cold start | instant | **~104s** (FP8) | instant |
| API format | OpenAI | OpenAI | OpenAI |
| Reasoning field | `reasoning` | `reasoning` | `reasoning_content` |

### Why llama.cpp is faster here

vLLM's performance is bottlenecked by `--enforce-eager` (mandatory for AWQ+GDN layers).
llama.cpp uses CUDA kernels tuned for GGUF quantization formats and doesn't have
the GDN layer penalty, so it can take full advantage of the RTX 4090's tensor cores.

### llama.cpp trade-offs

- ❌ No native reasoning parser — thinking is embedded in `reasoning_content` (deepseek format), not a structured field  
- ❌ No `tool-call-parser qwen3_coder` equivalent  
- ✅ Far better throughput and VRAM efficiency  
- ✅ 32K context vs 4–8K in vLLM  
- ✅ No warmup penalty

---

## TurboQuant (tq3_0) KV Cache — llama-turboquant-animehacker build

### What it is

`llama-turboquant-animehacker` is a fork implementing Google's TurboQuant algorithm (ICLR 2026):
3.5-bit K+V cache via Lloyd-Max centroid quantization + Walsh-Hadamard Transform rotation.
Achieves **4.57× KV cache compression** vs fp16. Three implementation phases:

1. **Phase 1**: K cache tq3_0 with WHT rotation
2. **Phase 2**: V cache tq3_0 — required non-transposed storage workaround (standard llama.cpp
   only supports quantized V with flash attention, but tq3_0 K disables flash attention)
3. **Phase 3**: Graph-side K/V dequantization before flash attention kernel — re-enables FA,
   eliminating the O(n²) compute buffer wall that capped context at 16K without FA

Binary: `/home/fsabado/src/llama-turboquant-animehacker/build-cuda/bin/llama-server`
Log signal when active: `TQ3_0 K cache with flash_attn - will dequant K/V in attention graph`

### VRAM layout on RTX 4090 (Qwen3.5-27B Q4_K_M)

Key architectural constants:
- Model GPU: ~15088 MiB (Q4_K_M weights)
- RS buffer (GDN recurrent state): **149.6 MiB × parallel** — scales with slot count
- KV buffer (tq3_0): **13.7 bytes/token × ctx_total** — scales with total context
- Compute buffers: ~1100–2200 MiB — scales with ctx (not parallel)
- Baseline (no server): ~1300–1500 MiB

Model native context: **262144 tokens** (256K)

### Parallelism vs context tradeoffs on 24GB

| Parallel | Total ctx | Per-slot ctx | KV MiB | RS MiB | VRAM free | Best for |
|---------|-----------|-------------|--------|--------|-----------|---------|
| 4 | 262K | **65K** | 3584 | 598 | 853 MiB | Deep reasoning, long docs |
| 8 | 262K | **32K** | 3584 | 1197 | 355 MiB | Balanced |
| 16 | 196K | **12K** | 2688 | 2394 | 509 MiB | API server, many users |
| 24 | 131K | **5.6K** | 1848 | 3591 | 556 MiB | High concurrency |
| **32** | 65K | **2K** | 896 | 4788 | 965 MiB | Max concurrency |

**p=32 is the practical ceiling** — p=36+ would need <2K context per slot.

### Throughput at different concurrency levels (tq3_0, p=32 config)

| n | total tok/s | per-req tok/s | avg latency |
|---|-------------|---------------|-------------|
| 1 | 20.9 | 20.9 | 4.8s |
| 2 | 29.0 | 14.5 | 6.9s |
| 4 | 38.6 | 9.6 | 10.4s |
| 8 | 44.1 | 5.5 | 18.1s |
| 16 | 50.6 | 3.2 | 31.6s |
| 24 | 54.8 | 2.3 | 43.7s |
| 32 | 56.7 | 1.8 | 56.5s |

### tq3_0 performance trade-off vs q8_0

tq3_0 single-request is ~44% slower than q8_0 (20–22 vs 39 tok/s).
**Root cause:** WHT dequantization per attention step adds significant overhead per decode step.
The extra throughput from larger parallelism (56.7 tok/s at n=32) can make up for this
if the use case involves many concurrent short requests.

For long conversations actually filling the context window, tq3_0 context headroom
outweighs the per-token overhead.

### Recommended configs (tq3_0)

```bash
LLAMA_BIN="/home/fsabado/src/llama-turboquant-animehacker/build-cuda/bin"

# Single user — max context (65K)
"$LLAMA_BIN/llama-server" --model Qwen3.5-27B.Q4_K_M.gguf \
  --ctx-size 262144 --parallel 4 \
  --flash-attn on --cache-type-k tq3_0 --cache-type-v tq3_0 \
  --n-gpu-layers 99 --cont-batching --jinja \
  --chat-template-file chat_template.jinja --reasoning-format deepseek

# Multi-user — high concurrency (12K/slot)
"$LLAMA_BIN/llama-server" --model Qwen3.5-27B.Q4_K_M.gguf \
  --ctx-size 196608 --parallel 16 \
  --flash-attn on --cache-type-k tq3_0 --cache-type-v tq3_0 \
  --n-gpu-layers 99 --cont-batching --jinja \
  --chat-template-file chat_template.jinja --reasoning-format deepseek

# Max parallelism (2K/slot, 32 concurrent)
"$LLAMA_BIN/llama-server" --model Qwen3.5-27B.Q4_K_M.gguf \
  --ctx-size 65536 --parallel 32 \
  --flash-attn on --cache-type-k tq3_0 --cache-type-v tq3_0 \
  --n-gpu-layers 99 --cont-batching --jinja \
  --chat-template-file chat_template.jinja --reasoning-format deepseek
```

---

## Lower Quantizations: Q3_K_M and Q2_K (mradermacher imatrix, v1)

**Source:** `mradermacher/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-i1-GGUF`  
(Jackrong v2 GGUF only goes down to Q4_K_M; mradermacher quantized v1 with imatrix to Q3/Q2)

**Note on imatrix (i1):** Importance matrix quantization selectively preserves critical weight
regions. The Q3_K_M file uses mixed precisions (353 q3_K + 138 q4_K + 6 q5_K tensors).

**Benchmark config:** tq3_0 KV cache, ctx=65536, parallel=4 (same as Q4_K_M per-slot)

### Setup pitfall: VRAM must be clear before starting

Loading Q3_K_M (12.4 GB) or Q2_K (9.4 GB) requires ~14–12 GB free VRAM respectively.
If a prior llama-server was `terminate()`-d but CUDA context not fully released, the new
server will OOM. **Fix:** `kill -9 <pid>` the orphaned processes and verify VRAM is free
before starting. Use `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader`
to find orphans.

### Q3_K_M results (i1, 12.38 GiB, 3.95 BPW)

| n | total tok/s | per-req tok/s | latency |
|---|-------------|---------------|---------|
| 1 | 23.3 | 23.3 | 6.4s |
| 2 | 23.9 | 12.0 | 12.5s |
| 4 | 23.4 | 5.8 | 25.7s |
| 6 | 23.5 | 3.9 | 30.0s |
| 8 | 23.2 | 2.9 | 38.4s |

Single-request avg: **21.8 tok/s** | Bat-and-ball: **❌ WRONG**

### Q2_K results (i1, 9.43 GiB, 3.03 BPW)

| n | total tok/s | per-req tok/s | latency |
|---|-------------|---------------|---------|
| 1 | 23.9 | 23.9 | 6.2s |
| 2 | 23.8 | 11.9 | 12.6s |
| 4 | 22.3 | 5.6 | 26.9s |
| 6 | 22.6 | 3.8 | 31.3s |
| 8 | 22.2 | 2.8 | 40.6s |

Single-request avg: **25.1 tok/s** | Bat-and-ball: **✅ CORRECT**

### Analysis

**Speed:** Q2_K is ~15% faster single-request than Q3_K_M (smaller model → less memory bandwidth).
Both are within ~5–10% of Q4_K_M's ~22 tok/s — quantization below Q4 gives diminishing speed gains.

**No parallel batching benefit:** Both Q3_K_M and Q2_K show flat total throughput (~22–24 tok/s)
regardless of concurrency. Wall time at n=4 ≈ 4× single-request latency — requests run
essentially sequentially despite `--cont-batching`. This contrasts with Q4_K_M tq3_0 which
achieved 38.6 tok/s at n=4. Root cause unclear — may be related to mradermacher imatrix vs
Jackrong direct quantization, or the 65K vs 262K context size used for Q4_K_M.

**Quality:** Q3_K_M failed the bat-and-ball reasoning test (❌). Q2_K passed (✅). This is
counterintuitive — imatrix quantization at Q2_K may actually preserve reasoning-critical
weight regions better than Q3_K_M's specific tensor selection. Single data point; not conclusive.

**VRAM:** Q3_K_M uses 14.8 GB (model + KV + RS buffers), Q2_K uses 11.8 GB.
Both leave more headroom than Q4_K_M (17.1 GB) on 24 GB VRAM.

---

---

## Gemma 4 E2B (effective 2B, actual 4.65B) — llama.cpp

**Model:** `unsloth/gemma-4-E2B-it-GGUF` Q4_K_M (2.88 GiB)  
**Backend:** `~/src/llama.cpp-upstream` (ggml v0.9.11, commit `66c4f9ded`) — first build with `gemma4` arch support  
**Note:** turboquant and animehacker builds (v8521/v8529) do NOT support `gemma4` — upstream required

---

### Architecture: iSWA (interleaved Sliding Window Attention)

Gemma 4 uses a hybrid attention architecture with two KV caches:

| Cache type | Layers | Cells | Ring-bounded? |
|-----------|--------|-------|--------------|
| Full attention (non-SWA) | 3 | full ctx per slot | No — scales with context |
| Sliding window (SWA) | 12 | 1024 tokens | Yes — bounded regardless of ctx |

**Implication:** Most of the KV budget comes from the 3 full-attention layers.  
Incremental cost per slot at 131K per-slot context ≈ **768 MiB** (vs ~150 MiB/slot for Qwen3.5-27B).  
SWA layers are nearly free (192 MiB total for 12 layers × 16 seqs at 1024 cells).

---

### Raw Throughput (llama-bench, flash-attn on)

| Metric | Value |
|--------|-------|
| pp512 (prompt processing) | **15,484 tok/s** |
| tg128 (token generation) | **216 tok/s** |

API single-request decode: **~155–167 tok/s** (overhead vs raw tg128 is minimal).

---

### API Throughput at Different Configurations

**All configs use flash-attn on, Q4_K_M, cont-batching.**

| Config | ctx total | Per slot | VRAM | n=1 | n=4 | n=8 | n=16 | n=32 | n=64 | Peak total |
|--------|-----------|----------|------|-----|-----|-----|------|------|------|-----------|
| p=4, 131K | 131072 | 32768 | 6271 MiB | 154 | 336 | 351 | 304 | 325 | — | **~350 @ n=8** |
| p=32, 131K | 131072 | 4096 | 6602 MiB | 161 | 328 | 401 | 487 | 448 | — | **~487 @ n=16** |
| p=64, 131K | 131072 | 2048 | 6986 MiB | 134 | 321 | 380 | 440 | 484 | 492 | **~495 @ n=48** |

*(tok/s = total system throughput)*

**High-context configs (131K per slot):**

| Config | ctx total | Per slot | VRAM | Status |
|--------|-----------|----------|------|--------|
| p=4, 524K | 524288 | 131072 | 8,570 MiB | ✅ |
| p=16, 2M | 2097152 | 131072 | 17,930 MiB | ✅ |
| p=24, 3.1M | 3145728 | 131072 | 22,469 MiB | ✅ |
| p=28, 3.7M | 3670016 | 131072 | 24,061 MiB | ✅ |
| p=32, 4.2M | 4194304 | 131072 | 23,967 MiB | ✅ |
| p=36, 4.7M | 4718592 | 131072 | 23,986 MiB | ✅ |
| p=40, 5.2M | 5242880 | 131072 | 23,942 MiB | ✅ **ceiling** |
| p=48, 6.3M | 6291456 | 131072 | — | ❌ OOM |

**Max parallel at native 131K per-slot context on 24GB: p=40** (not p=24 as previously documented)

The VRAM plateaus at ~24 GB from p=28 onward — this is the iSWA effect in action. Each additional
slot above the initial full-attention KV budget only pays for its **1024-token SWA window**, not the
full 131K context. So p=28 through p=40 all fit within ~24 GB even though naive math suggests they
shouldn't.

**Single-slot max context (p=1):**

| ctx | VRAM |
|-----|------|
| 1M | 12,494 MiB |
| 2M (native max) | 21,721 MiB |

---

### Reasoning Quality

Bat-and-ball test (requires `max_tokens=1024` — model uses `reasoning_content` for thinking):

- ✅ **CORRECT** at Q4_K_M with enough output budget
- ⚠️ Gemma 4 E2B uses thinking in `reasoning_content` field (similar to DeepSeek), not in `content`
- `content` is empty if `max_tokens` is too small to complete thinking + answer
- Final answer in `content`; intermediate reasoning in `reasoning_content`

---

### Gemma 4 E2B vs Qwen3.5-27B Q4_K_M (llama.cpp)

| Metric | Qwen3.5-27B Q4_K_M | Gemma 4 E2B Q4_K_M |
|--------|---------------------|---------------------|
| Model VRAM | 15.4 GB | **2.9 GB** |
| Single tok/s | 39.7 | **~160** (+303%) |
| Peak total tok/s | 113 @ n=8 | **~492 @ n=48** (+335%) |
| Max context / slot | 65K (tq3_0) | **131K** (native) |
| Max concurrent (24GB) | 32 (2K/slot, tq3_0) | **64+ (2K/slot) / 40 (131K/slot)** |
| Reasoning field | `reasoning_content` | `reasoning_content` |
| Build required | turboquant-animehacker | upstream llama.cpp ≥ ggml v0.9.11 |

---

### Setup Notes

1. **Build**: turboquant/animehacker builds don't support `gemma4` arch. Must build from `~/src/llama.cpp-upstream`.  
2. **max_tokens**: Needs ≥1024 for thinking-mode reasoning tasks (model thinks in `reasoning_content` before writing `content`).  
3. **Port**: Using 10222 to avoid conflict with Qwen3.5-27B on 10111.

### Recommended Configs

```bash
LLAMA_BIN="/home/fsabado/src/llama.cpp-upstream/build-cuda/bin"
MODEL="/home/fsabado/models/gemma-4-e2b-it/gemma-4-E2B-it-Q4_K_M.gguf"
export LD_LIBRARY_PATH="$LLAMA_BIN:$LD_LIBRARY_PATH"

# Single-user / deep reasoning — full 131K context per session
"$LLAMA_BIN/llama-server" --model "$MODEL" --port 10222 --host 0.0.0.0 \
  -ngl 99 --ctx-size 524288 --parallel 4 \
  --cont-batching --flash-attn on --no-warmup
# VRAM: 8570 MiB | Single: ~155 tok/s | 131K ctx/slot

# Max full-context multi-user — 40 users at native 131K context (iSWA ceiling on 24GB)
"$LLAMA_BIN/llama-server" --model "$MODEL" --port 10222 --host 0.0.0.0 \
  -ngl 99 --ctx-size 5242880 --parallel 40 \
  --cont-batching --flash-attn on --no-warmup
# VRAM: 23942 MiB | 131K ctx/slot | iSWA plateau: p=28–40 all ~24 GB

# Single-user / max context — full 2M native context, 1 session
"$LLAMA_BIN/llama-server" --model "$MODEL" --port 10222 --host 0.0.0.0 \
  -ngl 99 --ctx-size 2097152 --parallel 1 \
  --cont-batching --flash-attn on --no-warmup
# VRAM: 21721 MiB | 2M ctx/slot (native max)

# High-concurrency / short context — 64 users at 2K context
"$LLAMA_BIN/llama-server" --model "$MODEL" --port 10222 --host 0.0.0.0 \
  -ngl 99 --ctx-size 131072 --parallel 64 \
  --cont-batching --flash-attn on --no-warmup
# VRAM: 6986 MiB | Peak: ~492 tok/s @ n=48-64 | 2K ctx/slot
```

---

---

## Gemma 4 E4B (effective 4B, actual 7.52B) — llama.cpp

**Model:** `bartowski/google_gemma-4-E4B-it-GGUF` Q4_K_M (5.02 GiB)  
**Backend:** `~/src/llama.cpp-upstream` (same build as E2B)  
**Architecture:** same iSWA hybrid as E2B — non-SWA + SWA KV caches

---

### Raw Throughput (llama-bench, flash-attn on)

| Metric | E4B | E2B | Δ |
|--------|-----|-----|---|
| pp512 (prompt processing) | 9,869 tok/s | 15,484 tok/s | -36% |
| tg128 (token generation) | **131.8 tok/s** | 216 tok/s | -39% |
| API single-request | **~100–105 tok/s** | ~155–167 tok/s | -35% |

---

### iSWA KV Cache (E4B vs E2B)

| Model | Non-SWA layers | VRAM/slot at 131K ctx | SWA layers | VRAM/slot (SWA) |
|-------|---------------|----------------------|------------|-----------------|
| E2B | 3 | 768 MiB | 12 | ~12 MiB |
| **E4B** | **8** | **2048 MiB** | — | ~40 MiB |

E4B has 2.67× the non-SWA KV cost per slot, which sharply limits full-context parallelism.

---

### API Throughput at Different Configurations

| Config | Per slot | VRAM | n=1 | n=4 | n=8 | n=16 | n=32 | n=64 | Peak |
|--------|---------|------|-----|-----|-----|------|------|------|------|
| p=4, ctx=131K | 32768 | 9292 MiB | 92 | 236 | 202 | 214 | 198 | — | **~236 @ n=4** |
| p=32, ctx=131K | 4096 | 10469 MiB | 96 | 204 | 254 | 421 | **446** | — | **~446 @ n=32** |
| p=64, ctx=131K | 2048 | 11485 MiB | 84 | 217 | 267 | 308 | 433 | **467** | **~467 @ n=64** |

*(tok/s = total system throughput)*

### High-context configs (131K per slot)

| Config | VRAM | Notes |
|--------|------|-------|
| p=4, ctx=524K | 15615 MiB | Max context quality |
| **p=8, ctx=1M** | **23401 MiB** | **Ceiling — 8 full-context users** |
| p=12+ at 131K/slot | OOM (lazy alloc masks real limit) | Use p=8 as safe ceiling |

**Max parallel at native 131K per-slot context on 24GB: p=8**  
(vs E2B's p=24 — E4B's 8 non-SWA layers cost 2.67× more KV per slot)

---

### Reasoning Quality

- ✅ Bat-and-ball **CORRECT** — answered in **486 tokens** (vs E2B's 829 tokens)
- E4B is more concise in reasoning traces
- Same `reasoning_content` field; same `max_tokens ≥ 1024` requirement

---

### Gemma 4 E4B vs E2B vs Qwen3.5-27B (llama.cpp)

| Metric | Qwen3.5-27B Q4_K_M | Gemma 4 E2B Q4_K_M | Gemma 4 E4B Q4_K_M |
|--------|---------------------|---------------------|---------------------|
| Model VRAM | 15.4 GB | 2.9 GB | **5.0 GB** |
| Single tok/s | 39.7 | ~160 | **~103** |
| Peak total tok/s | 113 @ n=8 | ~492 @ n=48 | **~467 @ n=64** |
| Max ctx/slot | 65K (tq3_0) | 131K | **131K** |
| Max parallel at 131K/slot | 32 | 24 | **8** |
| Reasoning tokens used | N/A | 829 | **486 (more efficient)** |
| KV per slot at 131K ctx | ~150 MiB | 768 MiB | **2048 MiB** |

**When to prefer E4B over E2B:** When reasoning quality matters and VRAM headroom is available. E4B answers more concisely (fewer reasoning tokens). At high concurrency with short context, E4B and E2B peak throughput is similar (~460–495 tok/s).

### Recommended Configs

```bash
LLAMA_BIN="/home/fsabado/src/llama.cpp-upstream/build-cuda/bin"
MODEL="/home/fsabado/models/gemma-4-e4b-it/google_gemma-4-E4B-it-Q4_K_M.gguf"
export LD_LIBRARY_PATH="$LLAMA_BIN:$LD_LIBRARY_PATH"

# Single-user / deep reasoning — full 131K context per session
"$LLAMA_BIN/llama-server" --model "$MODEL" --port 10333 --host 0.0.0.0 \
  -ngl 99 --ctx-size 524288 --parallel 4 \
  --cont-batching --flash-attn on --no-warmup
# VRAM: 15615 MiB | Single: ~103 tok/s | 131K ctx/slot

# Multi-user — 8 users at full 131K context (ceiling on 24GB)
"$LLAMA_BIN/llama-server" --model "$MODEL" --port 10333 --host 0.0.0.0 \
  -ngl 99 --ctx-size 1048576 --parallel 8 \
  --cont-batching --flash-attn on --no-warmup
# VRAM: 23401 MiB | Peak: ~446 tok/s @ n=32 | 131K ctx/slot

# High-concurrency / short context — 64 users at 2K context
"$LLAMA_BIN/llama-server" --model "$MODEL" --port 10333 --host 0.0.0.0 \
  -ngl 99 --ctx-size 131072 --parallel 64 \
  --cont-batching --flash-attn on --no-warmup
# VRAM: 11485 MiB | Peak: ~467 tok/s @ n=64 | 2K ctx/slot
```

---

## Gemma 4 26B A4B (effective 4B active, 25.23B total MoE) — llama.cpp

**Model:** `bartowski/google_gemma-4-26B-A4B-it-GGUF` Q4_K_M  
**File:** 15.85 GiB | 25.23B total params, ~4B active per token  
**Binary:** `llama.cpp-upstream` (build b32-66c4f9ded)

### Architecture

- **30 layers**, iSWA pattern: 5 SWA layers then 1 full-attention, repeating
  - SWA layers: 8 KV heads @ head_dim=256, window=1024 tokens
  - Full-attention layers: 2 KV heads @ head_dim=512 (GQA ratio=8)
- **MoE:** 128 experts, 8 active per token (`n_expert=128`, `n_expert_used=8`)
- **Expert FFN dim:** 704 (dense FFN: 2112)
- **Context trained:** 262,144 tokens

The MoE sparsity means token generation speed is close to a dense 4B model (~100 tok/s),
despite 25B total parameters. The iSWA pattern limits KV scaling — SWA layers only store
1024 tokens, full-attention layers use 2 KV heads (very small GQA).

### VRAM Breakdown (Q4_K_M, 24 GB card)

| Config | VRAM | Notes |
|--------|------|-------|
| Model only (ctx=8K, p=1) | 20,539 MiB | ~20 GB base — tight headroom |
| +ctx=131K, p=1 | 22,924 MiB | +2,385 MiB KV |
| +ctx=262K, p=1 | 23,830 MiB | sub-linear due to iSWA |
| ctx=524K, p=4 (131K/slot) | 23,669 MiB | ✅ max useful config |
| ctx=786K, p=6 | OOM | ❌ |
| ctx=1M, p=8 | OOM | ❌ |

**Parallelism ceiling: p=4 at 131K/slot on 24 GB.** This is much lower than E2B (p=40) and
E4B (p=8) due to the large model base (20.5 GB vs 2.9 GB / 5.0 GB).

### Raw Throughput (llama-bench, flash-attn, -ngl 99)

| Test | tok/s |
|------|-------|
| pp512 | 1,977 |
| pp2048 | 4,404 |
| tg128 | 99.5 |
| tg256 | 117.1 |

MoE advantage: tg speed matches a dense ~4B model despite 25B total parameters.

### API Throughput — Context Sweet Spot

**⚠️ Critical finding:** A sharp performance cliff exists above ~131K total context.

| Total ctx | Per-slot | agg c=4 | Single tok/s | Notes |
|-----------|----------|---------|--------------|-------|
| 4,096 | 4K | — | 76 | Very fast |
| 32,768 | 8K | 188 | 80 | ✅ Sweet spot |
| 131,072 | 32K | 188 | 81 | ✅ Best balance |
| 524,288 | 131K | ~13 | 10 | ❌ 7× slowdown |

The cliff at ctx_total > 131K causes >7× throughput degradation. Root cause: full-attention
layers must attend over the entire pre-allocated KV buffer; at 524K allocated tokens the
CUDA compute cost dominates even for short inputs.

### API Throughput — Parallel Scaling (ctx=131K total, p=4, 32K/slot)

| Concurrency | avg tok/s | agg tok/s | avg latency |
|-------------|-----------|-----------|-------------|
| c=1 | 83 | 83 | 3.0s |
| c=2 | 70 | 139 | 3.6s |
| c=4 | 47 | 188 | 5.3s |
| c=8 | 35 | 185 | 8.0s (queued 2 rounds) |

### Reasoning Quality

- Uses `reasoning_content` field (same as E2B/E4B)
- Generates chain-of-thought thinking before final answer
- 26B model produces more detailed reasoning (1,448 char thinking for "17 × 23")
- `max_tokens ≥ 600` recommended for reasoning tasks

### Recommended Configs

```bash
LLAMA_BIN="/home/fsabado/src/llama.cpp-upstream/build-cuda/bin"
MODEL="/home/fsabado/models/gemma-4-26b-a4b-it/google_gemma-4-26B-A4B-it-Q4_K_M.gguf"
export LD_LIBRARY_PATH="$LLAMA_BIN:$LD_LIBRARY_PATH"

# Best balance: 4 users at 32K context each — 188 tok/s peak
"$LLAMA_BIN/llama-server" --model "$MODEL" --port 10444 --host 0.0.0.0 \
  -ngl 99 --ctx-size 131072 --parallel 4 \
  --cont-batching --flash-attn on --no-warmup --log-disable
# VRAM: ~22 GB | Single: ~83 tok/s | agg c=4: ~188 tok/s

# Single-user, short context — max speed
"$LLAMA_BIN/llama-server" --model "$MODEL" --port 10444 --host 0.0.0.0 \
  -ngl 99 --ctx-size 4096 --parallel 1 \
  --cont-batching --flash-attn on --no-warmup --log-disable
# VRAM: ~21 GB | Single: ~90 tok/s

# ❌ AVOID: any config with ctx*parallel > 131072 causes 7x slowdown
```

### Gemma 4 26B A4B vs Smaller Gemma 4 Models

| Metric | E2B (4.65B) | E4B (7.52B) | 26B A4B (25.23B) |
|--------|-------------|-------------|------------------|
| Model VRAM | 2.9 GB | 5.0 GB | **20.5 GB** |
| KV headroom (24 GB) | ~21 GB | ~19 GB | **~4 GB** |
| Parallelism ceiling (131K/slot) | p=40 | p=8 | **p=4** |
| Max safe total ctx | 2M+ | 1M | **131K** |
| Single tok/s | ~155 | ~103 | **~83** |
| Peak agg tok/s | ~492 (p=48) | ~467 (p=64) | **~188 (p=4)** |
| Active params per token | 4.65B (dense) | 7.52B (dense) | **~4B (MoE)** |
| Reasoning quality | Good | Better | **Best** |

**When to prefer 26B A4B:** When answer quality matters most and you can accept lower
throughput and strictly limited parallelism. The 26B total parameter model provides
richer context representations despite similar active-parameter cost per token.

---

## Summary Table

| Config | Single tok/s | Peak total tok/s | Max context | Max seqs | Notes |
|--------|-------------|-----------------|-------------|----------|-------|
| vLLM Baseline | 14.3 | 41.5 (n=6) | 4096 | 4 | Best single-user vLLM |
| vLLM OPT-1a CUDA graphs | 4.7 | 8.7 | 4096 | 4 | ❌ torch.compile overhead |
| vLLM OPT-3 FP8 4-seqs | 13.8 | 38.5 (n=6) | 8192 | 4 | 2x context, slow warmup |
| **vLLM OPT-2+3 FP8 8-seqs** | **14.6** | **35.7 (n=8)** | **7840** | **8** | **Best multi-user vLLM** |
| vLLM OPT-5 MTP | N/A | N/A | N/A | N/A | ❌ OOM |
| llama.cpp q8_0 / 32K | 39.7 | 113.3 (n=8) | 32768 | 8 | Best throughput |
| **llama.cpp tq3_0 / 262K p=4** | **22.2** | **38.6** | **65536/slot** | **4** | **Best context quality** |
| llama.cpp tq3_0 / 196K p=16 | 20.9 | 50.6 | 12288/slot | 16 | Best API concurrency |
| llama.cpp tq3_0 / 65K p=32 | 20.9 | 56.7 | 2048/slot | **32** | Max parallelism |
| llama.cpp Q3_K_M i1 / 65K p=4 | 21.8 | 23.4 (n=4) | 16384/slot | 4 | ❌ no batching; ❌ quality |
| llama.cpp Q2_K i1 / 65K p=4 | 25.1 | 22.3 (n=4) | 16384/slot | 4 | ❌ no batching; ✅ quality |
| **Gemma 4 E2B Q4_K_M p=4 524K** | **~155** | **~350 (n=4)** | **131072/slot** | **4** | Best context quality |
| **Gemma 4 E2B Q4_K_M p=40 5.2M** | **~155** | **~(tbd)** | **131072/slot** | **40** | **iSWA ceiling on 24 GB** |
| **Gemma 4 E2B Q4_K_M p=1 2M** | **~155** | **~155** | **2097152/slot** | **1** | **Max single-slot context (native)** |
| Gemma 4 E2B Q4_K_M p=64 131K | ~150 | ~492 (n=48) | 2048/slot | 64 | Best concurrency, short ctx |
| **Gemma 4 E4B Q4_K_M p=4 524K** | **~103** | **~236 (n=4)** | **131072/slot** | **4** | Best E4B context quality |
| **Gemma 4 E4B Q4_K_M p=8 1M** | **~103** | **~446 (n=32)** | **131072/slot** | **8** | E4B full-ctx ceiling on 24GB |
| Gemma 4 E4B Q4_K_M p=64 131K | ~84 | ~467 (n=64) | 2048/slot | 64 | E4B high concurrency |
| **Gemma 4 26B A4B Q4_K_M p=4 131K** | **~83** | **~188 (n=4)** | **32768/slot** | **4** | **Best quality; MoE 25B total / 4B active** |
| Gemma 4 26B A4B Q4_K_M p=1 4K | ~90 | ~90 | 4096/slot | 1 | Single-user max speed |

### Key Insights

**vLLM:** `--enforce-eager` is mandatory for AWQ + Qwen3.5 GDN hybrid layers.
CUDA graphs and torch.compile both degrade performance or cause OOM on this architecture.

**llama.cpp tq3_0 trade-off:** 4.57× KV compression unlocks huge context (up to 262K native)
but WHT dequantization adds ~44% per-token overhead vs q8_0. Choose based on use case:
- Short interactions needing max throughput → q8_0 / 32K
- Long conversations / document analysis → tq3_0 / 262K p=4 (65K per slot)
- Many concurrent users → tq3_0 / 196K p=16 or tq3_0 / 65K p=32

**Q2/Q3 lower quants:** Speed is similar to Q4_K_M (~20–25 tok/s single), no benefit beyond
saving VRAM. Parallel batching doesn't work (total throughput flat at n>1). Q3_K_M shows
quality degradation on reasoning tasks. Use Q4_K_M or above for reliable reasoning.

**RS buffer is the parallelism limit:** Qwen3.5's GDN recurrent state costs exactly
149.6 MiB per parallel slot, making p=32 the practical ceiling on a single 24GB GPU.

---
