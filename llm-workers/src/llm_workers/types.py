"""Core dataclasses shared across the worker system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorkerConfig:
    """Configuration for a worker pool."""

    base_url: str = "http://127.0.0.1:10222"
    model: str = "gemma-4-E2B-it-Q4_K_M.gguf"
    max_tokens: int = 8192
    temperature: float = 0.7
    system: str | None = None
    timeout: int = 300
    max_tool_iterations: int = 10  # agentic loop safety cap


@dataclass
class Task:
    """A unit of work to dispatch to a worker slot."""

    id: str
    prompt: str
    tools: list[str] = field(default_factory=list)
    """Tool names to enable. Empty = no tools. Available: run_python, read_file, shell, write_file."""
    system: str | None = None
    """Override pool-level system prompt for this task."""
    context: dict[str, Any] = field(default_factory=dict)
    """Arbitrary data carried alongside the task (not sent to model)."""


@dataclass
class ToolCall:
    """Record of a single tool call made during agentic execution."""

    name: str
    input: dict[str, Any]
    result: str
    ok: bool


@dataclass
class Result:
    """Output from a completed worker task."""

    task_id: str
    ok: bool
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    elapsed: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    thinking_chars: int = 0
    iterations: int = 0
    error: str = ""

    @property
    def tps(self) -> float:
        return self.tokens_out / self.elapsed if self.elapsed > 0 and self.tokens_out > 0 else 0.0
