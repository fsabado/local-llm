#!/usr/bin/env python3
"""
Gemma 4 E2B capability comparison — Q4_K_M vs F16.

Run once per model, saves JSON results. Compare with --compare.

Usage:
  python3 scripts/compare_e2b.py --label q4 --port 10222   # test Q4_K_M
  python3 scripts/compare_e2b.py --label f16 --port 10223  # test F16
  python3 scripts/compare_e2b.py --compare results/compare_e2b_q4_*.json results/compare_e2b_f16_*.json
"""

import argparse
import builtins
import json
import os
import sys
import textwrap
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

print = lambda *a, **kw: builtins.print(*a, **{**kw, "flush": True})

HOST = "127.0.0.1"

# ---------------------------------------------------------------------------
# Tool definitions (for tool-call tests)
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature units"}
            },
            "required": ["city"]
        }
    },
    {
        "name": "calculator",
        "description": "Evaluate a mathematical expression. Returns a number.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression to evaluate, e.g. '2 + 2' or 'sqrt(144)'"}
            },
            "required": ["expression"]
        }
    },
    {
        "name": "web_search",
        "description": "Search the web for current information.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    }
]

# Simulated tool responses
TOOL_RESPONSES = {
    "get_weather": lambda args: json.dumps({
        "city": args.get("city", "unknown"),
        "temperature": 18,
        "units": args.get("units", "celsius"),
        "condition": "partly cloudy",
        "humidity": 65,
        "wind_kph": 12
    }),
    "calculator": lambda args: str(__import__("math").sqrt(
        float(args.get("expression", "0")
              .replace("sqrt(", "").replace(")", "").strip())
    ) if "sqrt" in args.get("expression", "") else eval(
        args.get("expression", "0").replace("^", "**"),
        {"__builtins__": {}, "sqrt": __import__("math").sqrt}
    )),
    "web_search": lambda args: json.dumps({
        "query": args.get("query"),
        "results": [
            {"title": "Wikipedia: " + args.get("query", ""), "snippet": "Comprehensive article on the topic with historical context and recent developments."},
            {"title": "Recent News", "snippet": "Latest updates as of 2026 indicate ongoing research and debate in this field."}
        ]
    })
}

# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------
TESTS = [
    # ── Reasoning ──────────────────────────────────────────────────────────
    {
        "id": "reasoning_crt",
        "category": "reasoning",
        "name": "CRT bat & ball",
        "prompt": "A bat and a ball together cost $1.10. The bat costs exactly $1.00 more than the ball. How much does the ball cost? Show your reasoning step by step.",
        "check": lambda r: "0.05" in r or "5 cents" in r.lower() or "five cents" in r.lower(),
        "check_desc": "answer = $0.05",
        "max_tokens": 131072,
    },
    {
        "id": "reasoning_knights",
        "category": "reasoning",
        "name": "Knights & Knaves",
        "prompt": (
            "On an island, knights always tell the truth and knaves always lie. "
            "You meet a person A who says: 'I am a knave.' "
            "What is A — knight or knave? Explain why."
        ),
        "check": lambda r: "knave" in r.lower() and ("cannot" in r.lower() or "paradox" in r.lower() or "impossible" in r.lower() or "neither" in r.lower() or "not possible" in r.lower() or "can't" in r.lower()),
        "check_desc": "recognises the self-referential paradox",
        "max_tokens": 131072,
    },
    {
        "id": "reasoning_wordproblem",
        "category": "reasoning",
        "name": "Multi-step word problem",
        "prompt": (
            "Alice has twice as many apples as Bob. "
            "Charlie has 5 fewer apples than Alice. "
            "Together they have 55 apples. "
            "How many apples does each person have? Show your work."
        ),
        "check": lambda r: "24" in r and "12" in r and "19" in r,
        "check_desc": "Alice=24, Bob=12, Charlie=19 (correct: 5B-5=55 → B=12)",
        "max_tokens": 131072,
    },
    {
        "id": "reasoning_syllogism",
        "category": "reasoning",
        "name": "Syllogism validity",
        "prompt": (
            "Evaluate whether the following argument is valid or invalid, and explain why:\n"
            "Premise 1: All mammals are warm-blooded.\n"
            "Premise 2: Some warm-blooded animals can fly.\n"
            "Conclusion: Therefore, some mammals can fly.\n"
        ),
        "check": lambda r: "invalid" in r.lower() or "not valid" in r.lower() or "does not follow" in r.lower() or "fallacy" in r.lower() or "cannot conclude" in r.lower(),
        "check_desc": "correctly identifies invalid syllogism",
        "max_tokens": 131072,
    },
    # ── Coding ─────────────────────────────────────────────────────────────
    {
        "id": "coding_mergesort",
        "category": "coding",
        "name": "Implement merge sort",
        "prompt": (
            "Write a Python implementation of merge sort. "
            "Include a docstring, handle edge cases (empty list, single element), "
            "and add 3 assert-based tests at the bottom."
        ),
        "check": lambda r: "def merge" in r and "assert" in r,
        "check_desc": "contains merge function and asserts",
        "max_tokens": 131072,
    },
    {
        "id": "coding_debug",
        "category": "coding",
        "name": "Debug off-by-one",
        "prompt": textwrap.dedent("""\
            Find and fix the bug in this Python function:

            ```python
            def find_max_subarray_sum(arr):
                max_sum = 0
                current_sum = 0
                for i in range(len(arr) - 1):
                    current_sum += arr[i]
                    if current_sum < 0:
                        current_sum = 0
                    if current_sum > max_sum:
                        max_sum = current_sum
                return max_sum
            ```

            Test case that fails: find_max_subarray_sum([-1, -2, -3]) should return -1, but returns 0.
            Explain what's wrong and provide the corrected code.
        """),
        "check": lambda r: "range(len(arr))" in r or "max_sum = arr[0]" in r or "float('-inf')" in r or "negative" in r.lower(),
        "check_desc": "identifies off-by-one and all-negative case",
        "max_tokens": 131072,
    },
    {
        "id": "coding_regex",
        "category": "coding",
        "name": "Explain regex",
        "prompt": (
            "Explain in plain English what this regex does, broken down component by component:\n\n"
            r"`^(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$`"
            "\n\nAlso give two strings that match and two that don't."
        ),
        "check": lambda r: "password" in r.lower() or "uppercase" in r.lower() or "lookahead" in r.lower(),
        "check_desc": "recognises password validation pattern",
        "max_tokens": 131072,
    },
    # ── Tool use ───────────────────────────────────────────────────────────
    {
        "id": "tool_weather",
        "category": "tool_use",
        "name": "Single tool call",
        "prompt": "What is the current weather in Tokyo? Use the get_weather tool.",
        "tools": TOOLS,
        "check": lambda r: "tokyo" in r.lower() and ("18" in r or "partly" in r.lower() or "cloud" in r.lower()),
        "check_desc": "calls tool and uses result in answer",
        "max_tokens": 131072,
    },
    {
        "id": "tool_calculator",
        "category": "tool_use",
        "name": "Calculator tool",
        "prompt": "What is the square root of 1764? Use the calculator tool to find out.",
        "tools": TOOLS,
        "check": lambda r: "42" in r,
        "check_desc": "calls calculator, answer = 42",
        "max_tokens": 131072,
    },
    {
        "id": "tool_search",
        "category": "tool_use",
        "name": "Web search + synthesize",
        "prompt": "Search for information about the transformer architecture in machine learning, then summarize what you find.",
        "tools": TOOLS,
        "check": lambda r: len(r) > 100,
        "check_desc": "calls search and produces synthesis",
        "max_tokens": 131072,
    },
    # ── Instruction following ───────────────────────────────────────────────
    {
        "id": "instruct_json",
        "category": "instruction_following",
        "name": "Strict JSON output",
        "prompt": (
            "Output ONLY valid JSON (no markdown, no explanation, no code fences) "
            "describing a person with these fields: name (string), age (integer), "
            "skills (array of strings, at least 3), active (boolean)."
        ),
        "check": lambda r: _valid_json_check(r),
        "check_desc": "output is parseable JSON with required fields",
        "max_tokens": 131072,
    },
    {
        "id": "instruct_lipogram",
        "category": "instruction_following",
        "name": "Lipogram (no letter e)",
        "prompt": "Write a short poem of exactly 4 lines about the night sky. The poem must not contain the letter 'e' anywhere (uppercase or lowercase).",
        "check": lambda r: _lipogram_check(r),
        "check_desc": "no letter 'e' in any line, exactly 4 lines",
        "max_tokens": 131072,
    },
    {
        "id": "instruct_stepbystep",
        "category": "instruction_following",
        "name": "Labeled reasoning steps",
        "prompt": (
            "Solve this problem using exactly these labeled steps — "
            "Step 1: Restate the problem. Step 2: Identify knowns/unknowns. "
            "Step 3: Choose a method. Step 4: Execute. Step 5: Verify.\n\n"
            "Problem: A train travels 120 km at 60 km/h, then 80 km at 40 km/h. What is the average speed for the whole journey?"
        ),
        "check": lambda r: all(f"step {i}" in r.lower() for i in range(1, 6)),
        "check_desc": "all 5 labeled steps present",
        "max_tokens": 131072,
    },
    # ── Knowledge ──────────────────────────────────────────────────────────
    {
        "id": "knowledge_backprop",
        "category": "knowledge",
        "name": "Explain backpropagation",
        "prompt": "Explain backpropagation in neural networks to a software engineer who knows calculus. Be technically precise, covering the chain rule, gradient flow, and vanishing gradients.",
        "check": lambda r: "chain rule" in r.lower() and "gradient" in r.lower(),
        "check_desc": "covers chain rule and gradients",
        "max_tokens": 131072,
    },
    {
        "id": "knowledge_montyhall",
        "category": "knowledge",
        "name": "Monty Hall problem",
        "prompt": "Explain the Monty Hall problem and give the correct answer. Why do most people get it wrong?",
        "check": lambda r: ("switch" in r.lower() or "switching" in r.lower()) and ("2/3" in r or "66" in r or "two-third" in r.lower() or "frac{2}{3}" in r or "frac{2}" in r),
        "check_desc": "correct answer: switch, 2/3 probability",
        "max_tokens": 131072,
    },
    {
        "id": "knowledge_mirror",
        "category": "knowledge",
        "name": "Mirror reversal puzzle",
        "prompt": "Why does a mirror appear to flip left and right, but not up and down? Give the real physical explanation.",
        "check": lambda r: "front" in r.lower() or "depth" in r.lower() or "axis" in r.lower() or "flip" in r.lower(),
        "check_desc": "addresses the depth/front-back flip explanation",
        "max_tokens": 131072,
    },
    # ── Adversarial ─────────────────────────────────────────────────────────
    {
        "id": "adversarial_liar",
        "category": "adversarial",
        "name": "Liar's paradox",
        "prompt": 'Is the following statement true or false: "This statement is false." Explain your answer carefully.',
        "check": lambda r: "paradox" in r.lower() or "neither" in r.lower() or "self-refer" in r.lower() or "cannot be" in r.lower() or "undecidable" in r.lower(),
        "check_desc": "recognises it as a paradox, not true/false",
        "max_tokens": 131072,
    },
    {
        "id": "adversarial_pronoun",
        "category": "adversarial",
        "name": "Ambiguous pronoun",
        "prompt": (
            "'The city council refused the demonstrators a permit because they feared violence.'\n\n"
            "Who does 'they' refer to? Discuss both possible interpretations and which is more likely."
        ),
        "check": lambda r: "council" in r.lower() and "demonstrat" in r.lower(),
        "check_desc": "discusses both referents",
        "max_tokens": 131072,
    },
    {
        "id": "adversarial_fabrication",
        "category": "adversarial",
        "name": "Hallucination / calibration",
        "prompt": "What is the capital city of Glaronia, and what is its population?",
        "check": lambda r: (
            "don't know" in r.lower() or "doesn't exist" in r.lower() or
            "not a" in r.lower() or "no country" in r.lower() or
            "fictional" in r.lower() or "made up" in r.lower() or
            "not aware" in r.lower() or "cannot find" in r.lower() or
            "unfamiliar" in r.lower() or "not familiar" in r.lower()
        ),
        "check_desc": "admits it doesn't know / flags non-existent country",
        "max_tokens": 131072,
    },
    {
        "id": "creativity_story",
        "category": "creativity",
        "name": "Short story with twist",
        "prompt": (
            "Write a short story (200–300 words) about an astronaut returning home after 10 years in space. "
            "End with an unexpected plot twist that recontextualises everything that came before."
        ),
        "check": lambda r: len(r.split()) >= 150,
        "check_desc": "at least 150 words",
        "max_tokens": 131072,
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_json_check(text: str) -> bool:
    text = text.strip()
    # strip markdown fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        d = json.loads(text)
        return (
            isinstance(d.get("name"), str)
            and isinstance(d.get("age"), int)
            and isinstance(d.get("skills"), list)
            and len(d["skills"]) >= 3
            and isinstance(d.get("active"), bool)
        )
    except Exception:
        return False


def _lipogram_check(text: str) -> bool:
    lines = [l for l in text.strip().splitlines() if l.strip()]
    if len(lines) < 4:
        return False
    poem = "\n".join(lines[:4])
    return "e" not in poem and "E" not in poem


def chat(port: int, messages: list[dict], tools: list | None = None,
         max_tokens: int = 1024) -> tuple[str, float, int]:
    """Send a chat request (non-streaming). Returns (text, elapsed_s, n_tokens)."""
    payload: dict = {
        "model": "benchmark",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    if tools:
        # Convert to OpenAI-style tools for llama.cpp
        payload["tools"] = [
            {"type": "function", "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            }}
            for t in tools
        ]
        payload["tool_choice"] = "auto"

    url = f"http://{HOST}:{port}/v1/chat/completions"
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as r:
        resp = json.loads(r.read())
    elapsed = time.perf_counter() - t0

    choice = resp["choices"][0]
    msg = choice["message"]

    # Handle tool call — execute and do one follow-up turn
    if choice.get("finish_reason") == "tool_calls" or msg.get("tool_calls"):
        tool_results = []
        for tc in msg.get("tool_calls", []):
            fn = tc["function"]
            name = fn["name"]
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except Exception:
                args = {}
            result = TOOL_RESPONSES.get(name, lambda _: "error")(args)
            tool_results.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })

        follow_messages = messages + [msg] + tool_results
        follow_payload = {
            "model": "benchmark",
            "messages": follow_messages,
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        fdata = json.dumps(follow_payload).encode()
        freq = urllib.request.Request(
            url, data=fdata, headers={"Content-Type": "application/json"}
        )
        t1 = time.perf_counter()
        with urllib.request.urlopen(freq, timeout=120) as r2:
            fresp = json.loads(r2.read())
        elapsed += time.perf_counter() - t1

        final_msg = fresp["choices"][0]["message"]
        text = final_msg.get("content") or ""
        n_tokens = (
            resp.get("usage", {}).get("completion_tokens", 0)
            + fresp.get("usage", {}).get("completion_tokens", 0)
        )
        return text, elapsed, n_tokens

    # Normal response — strip thinking prefix if present
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    n_tokens = resp.get("usage", {}).get("completion_tokens", 0)
    return content, elapsed, n_tokens


def run_test(port: int, test: dict) -> dict:
    messages = [{"role": "user", "content": test["prompt"]}]
    tools = test.get("tools")
    try:
        text, elapsed, n_tokens = chat(
            port, messages, tools=tools, max_tokens=test.get("max_tokens", 1024)
        )
        passed = test["check"](text.lower() if text else "")
        tps = n_tokens / elapsed if elapsed > 0 else 0
        return {
            "id": test["id"],
            "passed": passed,
            "text": text,
            "elapsed": round(elapsed, 2),
            "n_tokens": n_tokens,
            "tps": round(tps, 1),
            "error": None,
        }
    except Exception as e:
        return {
            "id": test["id"],
            "passed": False,
            "text": "",
            "elapsed": 0,
            "n_tokens": 0,
            "tps": 0,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Run all tests (4 concurrent to match p=4 server)
# ---------------------------------------------------------------------------

def run_all(port: int, label: str, concurrency: int = 4) -> dict:
    results = {}
    sem = threading.Semaphore(concurrency)
    lock = threading.Lock()
    threads = []

    def worker(test):
        with sem:
            with lock:
                print(f"  [{label}] running: {test['name']} ...", end=" ")
            r = run_test(port, test)
            status = "✓" if r["passed"] else "✗"
            with lock:
                print(f"{status}  ({r['elapsed']}s, {r['n_tokens']} tok, {r['tps']} tok/s)")
            results[test["id"]] = r

    for test in TESTS:
        t = threading.Thread(target=worker, args=(test,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    return results


# ---------------------------------------------------------------------------
# Compare and print report
# ---------------------------------------------------------------------------

def print_report(results_a: dict, label_a: str, results_b: dict, label_b: str):
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  CAPABILITY COMPARISON: {label_a}  vs  {label_b}")
    print(sep)

    categories = {}
    for test in TESTS:
        categories.setdefault(test["category"], []).append(test)

    total_pass_a = total_pass_b = 0

    for cat, tests in categories.items():
        print(f"\n── {cat.upper().replace('_', ' ')} {'─'*(50-len(cat))}")
        print(f"  {'Test':<30}  {label_a:>10}  {label_b:>10}  {'Winner':>8}")
        print(f"  {'-'*30}  {'-'*10}  {'-'*10}  {'-'*8}")
        for test in tests:
            ra = results_a.get(test["id"], {})
            rb = results_b.get(test["id"], {})
            pa = ra.get("passed", False)
            pb = rb.get("passed", False)
            total_pass_a += pa
            total_pass_b += pb
            sa = "✓" if pa else "✗"
            sb = "✓" if pb else "✗"
            if pa and not pb:
                winner = label_a
            elif pb and not pa:
                winner = label_b
            elif pa and pb:
                # Both pass — compare tok/s
                tpa = ra.get("tps", 0)
                tpb = rb.get("tps", 0)
                winner = "tie"
            else:
                winner = "both fail"
            ta = f"{sa} {ra.get('tps', 0):.0f}t/s"
            tb = f"{sb} {rb.get('tps', 0):.0f}t/s"
            print(f"  {test['name']:<30}  {ta:>10}  {tb:>10}  {winner:>8}")

    print(f"\n{sep}")
    print(f"  TOTAL PASS: {label_a} = {total_pass_a}/{len(TESTS)}   {label_b} = {total_pass_b}/{len(TESTS)}")

    # Speed comparison
    tps_a = [r["tps"] for r in results_a.values() if r["tps"] > 0]
    tps_b = [r["tps"] for r in results_b.values() if r["tps"] > 0]
    avg_a = sum(tps_a) / len(tps_a) if tps_a else 0
    avg_b = sum(tps_b) / len(tps_b) if tps_b else 0
    print(f"  AVG tok/s: {label_a} = {avg_a:.1f}   {label_b} = {avg_b:.1f}")
    print(sep)

    # Failures
    for label, results in [(label_a, results_a), (label_b, results_b)]:
        failures = [(t, results[t["id"]]) for t in TESTS
                    if not results.get(t["id"], {}).get("passed") and t["id"] in results]
        if failures:
            print(f"\n── {label} FAILURES")
            for test, r in failures:
                err = f" [ERROR: {r['error']}]" if r.get("error") else ""
                print(f"  ✗ {test['name']}{err}")
                if r.get("text"):
                    snippet = r["text"].replace("\n", " ")[:120]
                    print(f"    → {snippet}…")

    # Detailed diffs for interesting cases
    print(f"\n── RESPONSE SAMPLES (adversarial + tool use)")
    interesting = ["adversarial_fabrication", "adversarial_liar", "tool_weather", "tool_calculator", "instruct_lipogram"]
    for tid in interesting:
        test = next(t for t in TESTS if t["id"] == tid)
        ra = results_a.get(tid, {})
        rb = results_b.get(tid, {})
        print(f"\n  [{tid}] {test['name']}")
        print(f"  Prompt: {test['prompt'][:100]}…" if len(test['prompt']) > 100 else f"  Prompt: {test['prompt']}")
        for label, r in [(label_a, ra), (label_b, rb)]:
            txt = (r.get("text") or "").strip().replace("\n", " ")
            status = "✓" if r.get("passed") else "✗"
            print(f"  {label} {status}: {txt[:200]}{'…' if len(txt) > 200 else ''}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    run_p = sub.add_parser("run", help="Run tests against a server")
    run_p.add_argument("--port", type=int, default=10222)
    run_p.add_argument("--label", required=True)
    run_p.add_argument("--concurrency", type=int, default=4)
    run_p.add_argument("--out-dir", default="results")

    cmp_p = sub.add_parser("compare", help="Compare two saved result files")
    cmp_p.add_argument("file_a")
    cmp_p.add_argument("file_b")

    # Backwards compat: positional --label --port without subcommand
    parser.add_argument("--port", type=int, default=10222)
    parser.add_argument("--label", default="")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--compare", nargs=2, metavar=("FILE_A", "FILE_B"))

    args = parser.parse_args()

    if args.compare:
        fa, fb = args.compare
        with open(fa) as f: data_a = json.load(f)
        with open(fb) as f: data_b = json.load(f)
        print_report(data_a["results"], data_a["label"], data_b["results"], data_b["label"])
        return

    label = args.label or f"port{args.port}"
    print(f"\nRunning {len(TESTS)} tests against port {args.port} (label: {label})")
    print(f"Concurrency: {args.concurrency}\n")

    results = run_all(args.port, label, concurrency=args.concurrency)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"compare_e2b_{label}_{ts}.json"
    with open(out_file, "w") as f:
        json.dump({"label": label, "port": args.port, "timestamp": ts, "results": results}, f, indent=2)

    passed = sum(1 for r in results.values() if r.get("passed"))
    print(f"\nPassed: {passed}/{len(TESTS)}")
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
