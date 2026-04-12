"""llm-workers — local LLM worker pool via Anthropic SDK + llama-server."""

from .pool import LocalWorkerPool
from .types import Result, Task, ToolCall, WorkerConfig

__all__ = ["LocalWorkerPool", "Task", "Result", "ToolCall", "WorkerConfig"]
