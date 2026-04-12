#!/usr/bin/env python3
"""
Example: add_docstrings.py

Reads all .py files in a target directory, finds functions without docstrings,
and dispatches each one as a separate task to the local worker pool.
Each worker reads the function, writes a docstring, and patches the file.

Usage:
    uv run examples/add_docstrings.py <target_dir> [--dry-run] [--workers N]

Example:
    uv run examples/add_docstrings.py ~/src/local-llm/scripts --dry-run
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm_workers import LocalWorkerPool, Task, WorkerConfig


def find_undocumented_functions(path: Path) -> list[dict]:
    """Return list of {file, func_name, source, lineno} for functions missing docstrings."""
    found = []
    for py_file in sorted(path.rglob("*.py")):
        try:
            source = py_file.read_text()
            tree = ast.parse(source)
        except Exception:
            continue

        lines = source.splitlines()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            # Skip if already has a docstring
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                continue
            # Extract function source lines
            start = node.lineno - 1
            end = node.end_lineno
            func_source = "\n".join(lines[start:end])
            found.append({
                "file": str(py_file),
                "func_name": node.name,
                "source": func_source,
                "lineno": node.lineno,
            })
    return found


def build_tasks(functions: list[dict], dry_run: bool) -> list[Task]:
    tasks = []
    for fn in functions:
        task_id = f"{Path(fn['file']).name}::{fn['func_name']}"
        if dry_run:
            prompt = (
                f"Write a concise Python docstring for this function. "
                f"Return ONLY the docstring text (no triple quotes, no code block).\n\n"
                f"```python\n{fn['source']}\n```"
            )
            tools = []
        else:
            prompt = (
                f"Add a concise docstring to this Python function. "
                f"Use read_file to get the current file, then write_file to save the updated version.\n\n"
                f"File: {fn['file']}\n"
                f"Function: {fn['func_name']} (line {fn['lineno']})\n\n"
                f"```python\n{fn['source']}\n```\n\n"
                f"Rules:\n"
                f"- Insert the docstring as the first statement of the function body\n"
                f"- Keep the rest of the file exactly the same\n"
                f"- Use Google-style docstrings\n"
                f"- Do not add docstrings to any other functions\n"
                f"- Write the entire updated file content"
            )
            tools = ["read_file", "write_file"]

        tasks.append(Task(id=task_id, prompt=prompt, tools=tools, context=fn))
    return tasks


def main():
    parser = argparse.ArgumentParser(description="Add docstrings to undocumented Python functions")
    parser.add_argument("target_dir", help="Directory to scan for Python files")
    parser.add_argument("--dry-run", action="store_true", help="Generate docstrings but don't write files")
    parser.add_argument("--workers", type=int, default=20, help="Max parallel workers (default: 20)")
    parser.add_argument("--max-tasks", type=int, default=None, help="Limit number of tasks (for testing)")
    args = parser.parse_args()

    target = Path(args.target_dir).expanduser().resolve()
    if not target.exists():
        print(f"ERROR: {target} does not exist")
        sys.exit(1)

    print(f"Scanning {target} for undocumented functions...")
    functions = find_undocumented_functions(target)
    if not functions:
        print("No undocumented functions found.")
        return

    print(f"Found {len(functions)} undocumented function(s)")
    if args.max_tasks:
        functions = functions[: args.max_tasks]
        print(f"  → limiting to {args.max_tasks} tasks")

    tasks = build_tasks(functions, dry_run=args.dry_run)
    mode = "DRY RUN (generate only)" if args.dry_run else "WRITE MODE (patching files)"
    print(f"Mode: {mode}  |  workers: {args.workers}\n")

    config = WorkerConfig(
        max_tokens=4096,
        system=(
            "You are a Python expert. Write concise, accurate docstrings. "
            "When asked to modify files, make only the requested change and preserve everything else exactly."
        ),
    )
    pool = LocalWorkerPool(config=config, max_workers=args.workers)
    results = pool.map(tasks)

    # Print generated docstrings in dry-run mode
    if args.dry_run:
        print("\n── Generated docstrings ─────────────────────────────────────────")
        for r in sorted(results, key=lambda x: x.task_id):
            if r.ok:
                print(f"\n[{r.task_id}]")
                print(f"  {r.content[:300]}")
            else:
                print(f"\n[{r.task_id}] ERROR: {r.error}")

    # Summary
    ok = sum(1 for r in results if r.ok)
    fail = sum(1 for r in results if not r.ok)
    print(f"\nDone: {ok} succeeded, {fail} failed")
    if fail:
        for r in results:
            if not r.ok:
                print(f"  FAIL {r.task_id}: {r.error}")


if __name__ == "__main__":
    main()
