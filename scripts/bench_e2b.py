#!/usr/bin/env python3
"""
Gemma 4 E2B benchmark — TTFT, single tok/s, parallel throughput.

Usage:
  python3 scripts/bench_e2b.py [port]          # benchmark running server
  python3 scripts/bench_e2b.py --label "F16"   # add label to output
"""

import argparse
import builtins
import json
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass, field

# Force unbuffered output so progress is visible when piped/backgrounded
print = lambda *a, **kw: builtins.print(*a, **{**kw, "flush": True})

PORT = 10222
HOST = "127.0.0.1"


def base(port: int) -> str:
    return f"http://{HOST}:{port}/v1"


def post(url: str, payload: dict, timeout: int = 300) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def stream_request(port: int, payload: dict) -> tuple[float, float, int]:
    """Return (ttft_s, total_s, n_tokens). Uses SSE streaming."""
    url = f"{base(port)}/chat/completions"
    payload = {**payload, "stream": True}
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    ttft = None
    n_tokens = 0
    with urllib.request.urlopen(req, timeout=300) as r:
        for raw_line in r:
            line = raw_line.decode().strip()
            if not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                break
            try:
                obj = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            delta = obj.get("choices", [{}])[0].get("delta", {})
            # Count both thinking and content tokens — both burn compute
            text = delta.get("content") or delta.get("reasoning_content") or ""
            if text:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                n_tokens += 1
    total = time.perf_counter() - t0
    return ttft or total, total, n_tokens


@dataclass
class RequestResult:
    ttft: float
    total: float
    n_tokens: int

    @property
    def tps(self) -> float:
        return self.n_tokens / self.total if self.total > 0 else 0


def single_gen(port: int, prompt: str, max_tokens: int = 256) -> RequestResult:
    payload = {
        "model": "benchmark",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    ttft, total, n = stream_request(port, payload)
    return RequestResult(ttft=ttft, total=total, n_tokens=n)


def parallel_bench(port: int, n: int, max_tokens: int = 512) -> dict:
    prompt = "Write a haiku about the ocean."
    results: list[RequestResult | None] = [None] * n

    def run(i: int) -> None:
        results[i] = single_gen(port, prompt, max_tokens=max_tokens)

    t_start = time.perf_counter()
    threads = [threading.Thread(target=run, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t_start

    valid = [r for r in results if r is not None]
    total_tokens = sum(r.n_tokens for r in valid)
    avg_ttft = sum(r.ttft for r in valid) / len(valid)
    avg_latency = sum(r.total for r in valid) / len(valid)
    return {
        "n": n,
        "wall": wall,
        "total_tokens": total_tokens,
        "aggregate_tps": total_tokens / wall,
        "per_req_tps": total_tokens / wall / n,
        "avg_ttft": avg_ttft,
        "avg_latency": avg_latency,
    }


def get_model_name(port: int) -> str:
    try:
        r = post(f"{base(port)}/models", {})
        return r["data"][0]["id"]
    except Exception:
        return "unknown"


def run_bench(port: int, label: str = "", parallel_levels: list[int] | None = None) -> dict:
    if parallel_levels is None:
        parallel_levels = [1, 2, 4, 8, 16]

    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  {label or 'Benchmark'} — port {port}")
    print(sep)

    model = get_model_name(port)
    print(f"  Model: {model}\n")

    # Warmup
    print("Warmup...")
    w = single_gen(port, "Say hi.", max_tokens=16)
    print(f"  {w.n_tokens} tokens, {w.total:.1f}s ({w.tps:.0f} tok/s)\n")

    # Single-request baseline
    print("── Single-request baseline (3 runs) ──────────────────────────")
    runs = []
    for _ in range(3):
        r = single_gen(port, "Count from 1 to 50, one per line.", max_tokens=256)
        runs.append(r)
        print(f"  TTFT {r.ttft*1000:.0f}ms  total {r.total:.2f}s  {r.tps:.1f} tok/s  ({r.n_tokens} tokens)")
    avg_tps = sum(r.tps for r in runs) / len(runs)
    avg_ttft = sum(r.ttft for r in runs) / len(runs) * 1000
    print(f"  → avg {avg_tps:.1f} tok/s  TTFT {avg_ttft:.0f}ms\n")

    # Parallel throughput
    print("── Parallel throughput ────────────────────────────────────────")
    print(f"  {'n':>4}  {'agg tok/s':>10}  {'per-req tok/s':>14}  {'avg TTFT':>10}  {'avg latency':>12}")
    print(f"  {'-'*4}  {'-'*10}  {'-'*14}  {'-'*10}  {'-'*12}")
    parallel_data = {}
    for n in parallel_levels:
        p = parallel_bench(port, n)
        parallel_data[n] = p
        print(
            f"  {n:>4}  {p['aggregate_tps']:>10.1f}  {p['per_req_tps']:>14.1f}"
            f"  {p['avg_ttft']*1000:>9.0f}ms  {p['avg_latency']:>11.2f}s"
        )

    # Prompt processing
    print("\n── Prompt processing (pp) ─────────────────────────────────────")
    long_prompt = "The quick brown fox jumps over the lazy dog. " * 100
    r_pp = single_gen(port, long_prompt, max_tokens=32)
    # pp speed approximated from TTFT over ~450 input tokens
    approx_input_tokens = 450
    pp_tps = approx_input_tokens / r_pp.ttft if r_pp.ttft > 0 else 0
    print(f"  Input ~{approx_input_tokens} tokens → TTFT {r_pp.ttft*1000:.0f}ms ≈ {pp_tps:.0f} tok/s pp")

    return {
        "label": label,
        "model": model,
        "avg_single_tps": avg_tps,
        "avg_ttft_ms": avg_ttft,
        "parallel": parallel_data,
        "pp_tps_approx": pp_tps,
    }


def print_comparison(results: list[dict]) -> None:
    if len(results) < 2:
        return
    a, b = results[0], results[1]
    print(f"\n{'='*65}")
    print("  COMPARISON SUMMARY")
    print(f"{'='*65}")
    print(f"  {'Metric':<28}  {a['label']:>12}  {b['label']:>12}  {'Ratio':>8}")
    print(f"  {'-'*28}  {'-'*12}  {'-'*12}  {'-'*8}")

    def row(name, va, vb, fmt=".1f", unit=""):
        ratio = vb / va if va > 0 else float("inf")
        print(f"  {name:<28}  {va:>11{fmt}}{unit}  {vb:>11{fmt}}{unit}  {ratio:>7.2f}×")

    row("Single tok/s", a["avg_single_tps"], b["avg_single_tps"])
    row("TTFT (single)", a["avg_ttft_ms"], b["avg_ttft_ms"], unit="ms")
    row("PP speed (approx)", a["pp_tps_approx"], b["pp_tps_approx"])

    all_n = sorted(set(a["parallel"]) & set(b["parallel"]))
    for n in all_n:
        pa, pb = a["parallel"][n], b["parallel"][n]
        row(f"Agg tok/s  n={n}", pa["aggregate_tps"], pb["aggregate_tps"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("port", nargs="?", type=int, default=PORT)
    parser.add_argument("--label", default="")
    parser.add_argument(
        "--parallel-levels", nargs="+", type=int,
        default=[1, 2, 4, 8, 16],
        metavar="N",
    )
    args = parser.parse_args()
    run_bench(args.port, label=args.label, parallel_levels=args.parallel_levels)


if __name__ == "__main__":
    main()
