"""
LocalWorkerPool — fans tasks out across parallel llama-server slots.

Usage:
    pool = LocalWorkerPool(config)
    results = pool.map(tasks)          # blocks until all done
    results = pool.map(tasks, n=10)    # limit concurrency to 10 even if pool is larger
"""

from __future__ import annotations

import concurrent.futures
import time
from collections.abc import Callable, Iterable

import anthropic
from rich.console import Console
from rich.live import Live
from rich.table import Table

from . import worker as _worker
from .types import Result, Task, WorkerConfig

console = Console()


class LocalWorkerPool:
    """
    Thread-pool backed worker pool.

    Each call to map() fans the task list across min(len(tasks), max_workers) threads.
    Tasks are consumed from a queue — workers pick up the next task as soon as they finish.
    """

    def __init__(self, config: WorkerConfig | None = None, max_workers: int = 20):
        self.config = config or WorkerConfig()
        self.max_workers = max_workers
        self._client = anthropic.Anthropic(
            api_key="local",
            base_url=self.config.base_url,
            timeout=self.config.timeout,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def map(
        self,
        tasks: list[Task],
        n: int | None = None,
        on_result: Callable[[Result], None] | None = None,
        show_progress: bool = True,
    ) -> list[Result]:
        """
        Dispatch all tasks to the worker pool and return results in completion order.

        Args:
            tasks:         List of Task objects to execute.
            n:             Max concurrency override (default: self.max_workers).
            on_result:     Optional callback invoked as each result arrives.
            show_progress: Print a live progress table while running.
        """
        if not tasks:
            return []

        workers = min(n or self.max_workers, len(tasks))
        results: list[Result] = []
        t_start = time.time()

        def _run(task: Task) -> Result:
            return _worker.run(task, self.config, self._client)

        if show_progress:
            results = self._run_with_progress(tasks, workers, _run, on_result, t_start)
        else:
            results = self._run_plain(tasks, workers, _run, on_result)

        return results

    def run_one(self, task: Task) -> Result:
        """Convenience: run a single task synchronously."""
        return _worker.run(task, self.config, self._client)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _run_plain(
        self,
        tasks: list[Task],
        workers: int,
        fn: Callable,
        on_result: Callable | None,
    ) -> list[Result]:
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(fn, t): t for t in tasks}
            for fut in concurrent.futures.as_completed(futs):
                r = fut.result()
                results.append(r)
                if on_result:
                    on_result(r)
        return results

    def _run_with_progress(
        self,
        tasks: list[Task],
        workers: int,
        fn: Callable,
        on_result: Callable | None,
        t_start: float,
    ) -> list[Result]:
        completed: list[Result] = []
        pending = len(tasks)

        def make_table() -> Table:
            elapsed = time.time() - t_start
            ok = sum(1 for r in completed if r.ok)
            fail = len(completed) - ok
            avg_tps = (
                sum(r.tps for r in completed if r.ok) / len([r for r in completed if r.ok])
                if any(r.ok for r in completed)
                else 0
            )
            t = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 1))
            t.add_column("task", style="dim", min_width=28)
            t.add_column("status", min_width=8)
            t.add_column("elapsed", justify="right", min_width=7)
            t.add_column("tok/s", justify="right", min_width=6)
            t.add_column("out", justify="right", min_width=5)
            t.add_column("tools", justify="right", min_width=5)

            for r in completed[-20:]:  # show last 20 completed
                status = "[green]✓[/]" if r.ok else "[red]✗[/]"
                t.add_row(
                    r.task_id[:28],
                    status,
                    f"{r.elapsed:.1f}s",
                    f"{r.tps:.1f}",
                    str(r.tokens_out),
                    str(len(r.tool_calls)),
                )

            t.caption = (
                f"[bold]{ok}/{len(completed)}[/] done  "
                f"[dim]{pending} pending[/]  "
                f"[cyan]{avg_tps:.1f} tok/s avg[/]  "
                f"[dim]{elapsed:.0f}s elapsed[/]"
            )
            return t

        with Live(make_table(), refresh_per_second=4, console=console) as live:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(fn, t): t for t in tasks}
                for fut in concurrent.futures.as_completed(futs):
                    r = fut.result()
                    completed.append(r)
                    pending -= 1
                    live.update(make_table())
                    if on_result:
                        on_result(r)

        # Final summary line
        wall = time.time() - t_start
        ok = sum(1 for r in completed if r.ok)
        total_out = sum(r.tokens_out for r in completed)
        console.print(
            f"\n[bold green]✓[/] {ok}/{len(completed)} tasks — "
            f"{total_out:,} tokens out — "
            f"{wall:.1f}s wall time"
        )
        return completed
