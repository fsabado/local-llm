#!/usr/bin/env python3
"""
Gemma 4 E2B capability test — 20 parallel requests via Anthropic Python SDK
max_tokens=131072 per slot (full slot budget — model stops naturally)

Usage:
    .venv/bin/python3 scripts/test_e2b_capability.py
"""

import time
import concurrent.futures
from dataclasses import dataclass, field

import anthropic

BASE_URL = "http://127.0.0.1:10222"
MODEL = "gemma-4-E2B-it-Q4_K_M.gguf"
MAX_TOKENS = 131072
TIMEOUT = 600

client = anthropic.Anthropic(api_key="local", base_url=BASE_URL, timeout=TIMEOUT)

TOOLS = [
    {
        "name": "web_search",
        "description": "Search the web for current information",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "run_python",
        "description": "Execute Python code and return the output",
        "input_schema": {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a file from disk",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
]

TESTS = [
    # ── Code generation ───────────────────────────────────────────────────────
    {"id": "codegen_lru_cache", "cat": "code_gen", "tools": False,
     "prompt": "Implement an LRU Cache class in Python with get(key) and put(key, value), O(1) time. Use OrderedDict. Include full docstring and type hints."},

    {"id": "codegen_async_retry", "cat": "code_gen", "tools": False,
     "prompt": "Write an async Python decorator `retry(max_attempts=3, backoff=1.5, exceptions=(Exception,))` with exponential backoff. Full type hints, usage example."},

    {"id": "codegen_trie", "cat": "code_gen", "tools": False,
     "prompt": "Implement a Trie in Python: insert, search, starts_with, delete, wildcard '.' search. Complexity analysis in docstring."},

    {"id": "codegen_sql_builder", "cat": "code_gen", "tools": False,
     "prompt": "Write a fluent SQL query builder: select(), from_table(), where(), order_by(), limit(). Parameterized output, SQL injection safe."},

    {"id": "codegen_type_safe_dict", "cat": "code_gen", "tools": False,
     "prompt": "Write a type-safe Python dictionary wrapper using TypeVar and Generic enforcing value types at runtime. Support nested typed dicts. Python 3.12+ with test cases."},

    {"id": "codegen_graph_cycle", "cat": "code_gen", "tools": False,
     "prompt": "Implement directed graph cycle detection using DFS with WHITE/GRAY/BLACK coloring. Handle disconnected graphs, return cycle path if found. Include test cases."},

    # ── Debugging ─────────────────────────────────────────────────────────────
    {"id": "debug_kadane", "cat": "debug", "tools": False,
     "prompt": """Is there a bug in this code? Analyze carefully, fix if broken, explain if correct:
```python
def find_max_subarray(arr):
    max_sum = arr[0]; current = arr[0]
    for i in range(len(arr)):
        current = max(arr[i], current + arr[i])
        max_sum = max(max_sum, current)
    return max_sum
print(find_max_subarray([-2,1,-3,4,-1,2,1,-5,4]))  # Should print 6
```"""},

    {"id": "debug_race_condition", "cat": "debug", "tools": False,
     "prompt": """Identify the race condition and fix it. Explain exactly what sequence causes wrong results:
```python
import threading
counter = 0
def increment():
    global counter
    for _ in range(100000): counter += 1
threads = [threading.Thread(target=increment) for _ in range(4)]
for t in threads: t.start()
for t in threads: t.join()
print(counter)  # Expected 400000, gets random lower value
```"""},

    {"id": "debug_memory_leak", "cat": "debug", "tools": False,
     "prompt": """Find all bugs and the memory leak. Provide a fully fixed version:
```python
class EventEmitter:
    _listeners = {}
    def on(self, event, callback):
        if event not in self._listeners: self._listeners[event] = []
        self._listeners[event].append(callback)
    def emit(self, event, *args):
        for cb in self._listeners.get(event, []): cb(*args)
emitter = EventEmitter()
for i in range(10000): emitter.on('data', lambda x: print(x))
```"""},

    # ── Algorithms ────────────────────────────────────────────────────────────
    {"id": "algo_knapsack", "cat": "algorithm", "tools": False,
     "prompt": "0/1 knapsack in Python: bottom-up DP + backtracking to find selected items. O(n*W). Full test cases with expected output."},

    {"id": "algo_dijkstra", "cat": "algorithm", "tools": False,
     "prompt": "Dijkstra's shortest path using min-heap in Python. Handle disconnected graph, path reconstruction, negative edge warning. Complexity analysis + test cases."},

    {"id": "algo_merge_sort_parallel", "cat": "algorithm", "tools": False,
     "prompt": "Parallel merge sort in Python using concurrent.futures. Fall back to sequential for small arrays. Benchmark vs sorted() on a 100K element list."},

    # ── Tool calls ────────────────────────────────────────────────────────────
    {"id": "tool_web_search", "cat": "tool_call", "tools": True,
     "prompt": "Search the web for the current Python version and summarize what's new."},

    {"id": "tool_run_code", "cat": "tool_call", "tools": True,
     "prompt": "Run Python code to compute the first 30 Fibonacci numbers and show their sum."},

    {"id": "tool_read_file", "cat": "tool_call", "tools": True,
     "prompt": "Read /etc/os-release and report what Linux distribution this is."},

    {"id": "tool_multi_step", "cat": "tool_call", "tools": True,
     "prompt": "Do both: 1) search for 'llama.cpp GGUF format', and 2) run Python to show today's date. Use both tools."},

    # ── Reasoning ─────────────────────────────────────────────────────────────
    {"id": "reason_async_vs_threads", "cat": "reasoning", "tools": False,
     "prompt": "Explain Python asyncio vs threads vs multiprocessing. For each: when to use, pitfalls, concrete failure example if wrong choice is made."},

    {"id": "reason_rate_limiter", "cat": "reasoning", "tools": False,
     "prompt": "Design a distributed rate limiter: per-user + per-endpoint, sliding window, Redis backend, handles distributed races. Data model, pseudocode, edge cases."},

    # ── Edge cases ────────────────────────────────────────────────────────────
    {"id": "edge_unicode", "cat": "edge_case", "tools": False,
     "prompt": "Write a Python function that correctly reverses Unicode strings including ZWJ emoji, RTL text, combining chars. Explain why [::-1] fails. 5 test cases."},

    {"id": "edge_float", "cat": "edge_case", "tools": False,
     "prompt": "Explain why `0.1 + 0.2 != 0.3` at the bit level. Write safe float comparison. Compare float vs Decimal vs Fraction with examples of when to use each."},
]

assert len(TESTS) == 20, f"Expected 20 tests, got {len(TESTS)}"


@dataclass
class Result:
    test_id: str
    cat: str
    ok: bool
    text: str = ""
    thinking_chars: int = 0
    tool_calls: list = field(default_factory=list)
    elapsed: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    finish: str = ""
    error: str = ""

    @property
    def tps(self):
        return self.tokens_out / self.elapsed if self.elapsed > 0 and self.tokens_out > 0 else 0


def run_test(test: dict) -> Result:
    t0 = time.time()
    tid, cat = test["id"], test["cat"]
    try:
        kwargs = dict(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            temperature=0.7,
            messages=[{"role": "user", "content": test["prompt"]}],
        )
        if test["tools"]:
            kwargs["tools"] = TOOLS

        msg = client.messages.create(**kwargs)
        elapsed = time.time() - t0

        text = ""
        thinking_chars = 0
        tool_calls = []
        for block in msg.content:
            if block.type == "text":
                text = block.text
            elif block.type == "thinking":
                thinking_chars = len(block.thinking or "")
            elif block.type == "tool_use":
                tool_calls.append({"name": block.name, "input": block.input})

        return Result(
            test_id=tid, cat=cat, ok=True,
            text=text, thinking_chars=thinking_chars, tool_calls=tool_calls,
            elapsed=elapsed,
            tokens_in=msg.usage.input_tokens,
            tokens_out=msg.usage.output_tokens,
            finish=msg.stop_reason or "?",
        )
    except Exception as e:
        return Result(tid, cat, False, error=str(e), elapsed=time.time() - t0)


def run_all():
    print(f"Launching {len(TESTS)} parallel requests  max_tokens={MAX_TOKENS:,}/slot\n")
    t_wall = time.time()

    results: list[Result] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(TESTS)) as ex:
        futs = {ex.submit(run_test, t): t for t in TESTS}
        for fut in concurrent.futures.as_completed(futs):
            r = fut.result()
            status = "✓" if r.ok else "✗"
            warn = " [TRUNCATED]" if r.finish == "max_tokens" else ""
            tc_str = f"  {len(r.tool_calls)} tool(s)" if r.tool_calls else ""
            th_str = f"  think={r.thinking_chars//1000}K" if r.thinking_chars else ""
            print(f"  {status} [{r.cat:12s}] {r.test_id:<32s}  {r.elapsed:6.1f}s  "
                  f"{r.tps:5.1f} tok/s  out={r.tokens_out:5d}{th_str}{tc_str}  [{r.finish}]{warn}")
            if r.error:
                print(f"           ERR: {r.error}")
            results.append(r)

    wall = time.time() - t_wall
    ok = [r for r in results if r.ok]
    fail = [r for r in results if not r.ok]
    tool_ok = [r for r in results if r.cat == "tool_call" and r.tool_calls]
    tool_total = [r for r in results if r.cat == "tool_call"]
    truncated = [r for r in ok if r.finish == "max_tokens"]

    print(f"\nWall time : {wall:.1f}s")
    print("=" * 68)
    print(f"PASS      : {len(ok)}/{len(results)}")
    print(f"FAIL      : {len(fail)}")
    print(f"Truncated : {len(truncated)}  (hit {MAX_TOKENS:,} token ceiling)")
    if ok:
        print(f"Avg lat   : {sum(r.elapsed for r in ok)/len(ok):.1f}s")
        print(f"Avg tok/s : {sum(r.tps for r in ok)/len(ok):.1f}")
        print(f"Total out : {sum(r.tokens_out for r in ok):,} tokens")
        print(f"Thinking  : {sum(r.thinking_chars for r in ok)//1000}K chars total")
    print(f"Tool calls: {len(tool_ok)}/{len(tool_total)} triggered")
    print("=" * 68)

    print("\nPer-category:")
    for cat in sorted(set(r.cat for r in results)):
        cr = [r for r in results if r.cat == cat]
        cok = [r for r in cr if r.ok]
        if not cok:
            print(f"  {cat:14s}  0/{len(cr)} pass")
            continue
        print(f"  {cat:14s}  {len(cok)}/{len(cr)} pass"
              f"  {sum(r.elapsed for r in cok)/len(cok):.1f}s avg"
              f"  {sum(r.tps for r in cok)/len(cok):.1f} tok/s"
              f"  think={sum(r.thinking_chars for r in cok)//len(cok)//1000:.0f}K avg")

    print("\n── Sample outputs ───────────────────────────────────────────────")
    for r in sorted(results, key=lambda x: x.cat + x.test_id):
        print(f"\n[{r.test_id}]  finish={r.finish}  out={r.tokens_out}  think={r.thinking_chars//1000}K")
        if not r.ok:
            print(f"  ERROR: {r.error}")
            continue
        for tc in r.tool_calls:
            inp = str(tc.get("input", ""))[:100]
            print(f"  → tool: {tc['name']}({inp})")
        if r.text:
            print(f"  {r.text[:500]}")
        elif not r.tool_calls:
            print("  (no text content)")


if __name__ == "__main__":
    run_all()
