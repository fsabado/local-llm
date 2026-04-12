#!/usr/bin/env python3
"""Benchmark a running vLLM server. Usage: python3 bench.py [port]"""

import json, sys, time, threading, urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 10111
# Use WSL2 bridge IP — loopback is blocked in Claude Code sandbox
BASE = f"http://172.18.0.1:{PORT}/v1"
MODEL = "qwen35-27b"


def post(path, payload=None):
    url = BASE + path
    data = json.dumps(payload).encode() if payload else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def single_request(prompt, max_tokens=200, thinking=False):
    t0 = time.time()
    r = post("/chat/completions", {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": thinking},
    })
    elapsed = time.time() - t0
    msg = r["choices"][0]["message"]
    # vLLM puts thinking in "reasoning", llama.cpp uses "reasoning_content"
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    return {"tokens": r["usage"]["completion_tokens"], "prompt_tokens": r["usage"]["prompt_tokens"],
            "time": elapsed, "tps": r["usage"]["completion_tokens"] / elapsed,
            "content": msg.get("content", ""),
            "reasoning": reasoning,
            "finish_reason": r["choices"][0]["finish_reason"]}


def parallel_benchmark(n, max_tokens=150):
    prompt = "Write a haiku about the ocean."
    results = [None] * n

    def run(i):
        results[i] = single_request(prompt, max_tokens=max_tokens, thinking=False)

    t_start = time.time()
    threads = [threading.Thread(target=run, args=(i,)) for i in range(n)]
    for t in threads: t.start()
    for t in threads: t.join()
    wall = time.time() - t_start

    total_tokens = sum(r["tokens"] for r in results)
    avg_latency = sum(r["time"] for r in results) / n
    return {
        "n": n, "wall": wall, "total_tokens": total_tokens,
        "total_tps": total_tokens / wall,
        "avg_latency": avg_latency,
        "per_req_tps": total_tokens / wall / n,
    }


def run_full_benchmark(label=""):
    print(f"\n{'='*60}")
    if label:
        print(f"CONFIG: {label}")
    print(f"{'='*60}")

    # Check server
    try:
        models = post("/models")
        print(f"Model: {models['data'][0]['id']}\n")
    except Exception as e:
        print(f"ERROR: Cannot connect — {e}")
        return None

    # Warmup
    print("Warming up (first request triggers Triton autotuning)...")
    w = single_request("Say hi.", max_tokens=16)
    print(f"  Warmup: {w['tokens']} tokens in {w['time']:.1f}s ({w['tps']:.1f} tok/s)\n")

    # Single-request decode throughput
    print("--- Single-request decode throughput ---")
    runs = []
    for _ in range(3):
        r = single_request("Count from 1 to 50, one per line.", max_tokens=200)
        runs.append(r)
        print(f"  {r['tokens']} tokens in {r['time']:.2f}s = {r['tps']:.1f} tok/s")
    avg_single = sum(r["tps"] for r in runs) / len(runs)
    print(f"  Avg single-request: {avg_single:.1f} tok/s\n")

    # Parallel throughput
    print("--- Parallel throughput ---")
    parallel_results = {}
    for n in [1, 2, 4, 6, 8]:
        p = parallel_benchmark(n)
        parallel_results[n] = p
        print(f"  n={n}: wall={p['wall']:.1f}s  total={p['total_tps']:.1f} tok/s  "
              f"per-req={p['per_req_tps']:.1f} tok/s  latency={p['avg_latency']:.1f}s")

    # Reasoning quality spot-check
    print("\n--- Reasoning quality ---")
    r = single_request(
        "A bat and a ball cost $1.10. The bat costs $1.00 more than the ball. How much does the ball cost?",
        max_tokens=512, thinking=True
    )
    ans = r["content"].strip()
    correct = "0.05" in ans or "5 cents" in ans.lower() or "five cents" in ans.lower()
    print(f"  Bat-and-ball: {'✅ CORRECT' if correct else '❌ WRONG'} — {ans[:80]}")
    print(f"  Thinking: {len(r['reasoning'])} chars, answer in {r['time']:.1f}s")

    return {"avg_single_tps": avg_single, "parallel": parallel_results}


if __name__ == "__main__":
    run_full_benchmark()
