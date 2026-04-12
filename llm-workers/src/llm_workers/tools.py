"""
Tool implementations executed locally during the agentic loop.

Each tool function takes keyword arguments matching its schema and returns a string result.
The REGISTRY maps tool name → (schema_dict, callable).
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Callable

# ── Tool implementations ──────────────────────────────────────────────────────

_MAX_OUTPUT = 8_000   # chars, truncated beyond this
_MAX_FILE = 64_000    # chars, max file read size


def _truncate(s: str, limit: int = _MAX_OUTPUT) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n... [truncated — {len(s) - limit} chars omitted]"


def run_python(code: str, timeout: int = 30) -> str:
    """Execute Python code in a subprocess and return stdout + stderr."""
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(textwrap.dedent(code))
        tmp = f.name
    try:
        proc = subprocess.run(
            [sys.executable, tmp],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = proc.stdout + proc.stderr
        return _truncate(out.strip() or "(no output)")
    except subprocess.TimeoutExpired:
        return f"ERROR: Python execution timed out after {timeout}s"
    except Exception as e:
        return f"ERROR: {e}"
    finally:
        Path(tmp).unlink(missing_ok=True)


def read_file(path: str) -> str:
    """Read a file from disk and return its contents."""
    p = Path(path).expanduser()
    if not p.exists():
        return f"ERROR: file not found: {path}"
    if not p.is_file():
        return f"ERROR: not a file: {path}"
    try:
        content = p.read_text(errors="replace")
        return _truncate(content, _MAX_FILE)
    except Exception as e:
        return f"ERROR reading {path}: {e}"


def shell(command: str, timeout: int = 30) -> str:
    """Run a shell command and return its output."""
    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = proc.stdout + proc.stderr
        return _truncate(out.strip() or "(no output)")
    except subprocess.TimeoutExpired:
        return f"ERROR: command timed out after {timeout}s"
    except Exception as e:
        return f"ERROR: {e}"


def write_file(path: str, content: str) -> str:
    """Write content to a file on disk."""
    p = Path(path).expanduser()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"OK: wrote {len(content)} chars to {p}"
    except Exception as e:
        return f"ERROR writing {path}: {e}"


# ── Anthropic tool schema definitions ────────────────────────────────────────

SCHEMAS: dict[str, dict] = {
    "run_python": {
        "name": "run_python",
        "description": "Execute Python code in a subprocess. Returns stdout and stderr.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"},
                "timeout": {"type": "integer", "description": "Timeout in seconds (default 30)", "default": 30},
            },
            "required": ["code"],
        },
    },
    "read_file": {
        "name": "read_file",
        "description": "Read a file from disk and return its text contents.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or ~ path to file"},
            },
            "required": ["path"],
        },
    },
    "shell": {
        "name": "shell",
        "description": "Run a shell command and return its output.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run"},
                "timeout": {"type": "integer", "description": "Timeout in seconds (default 30)", "default": 30},
            },
            "required": ["command"],
        },
    },
    "write_file": {
        "name": "write_file",
        "description": "Write text content to a file on disk.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or ~ path to write"},
                "content": {"type": "string", "description": "Text content to write"},
            },
            "required": ["path", "content"],
        },
    },
}

# Map name → callable
CALLABLES: dict[str, Callable[..., str]] = {
    "run_python": run_python,
    "read_file": read_file,
    "shell": shell,
    "write_file": write_file,
}


def execute(name: str, inputs: dict[str, Any]) -> tuple[str, bool]:
    """Execute a tool by name. Returns (result_str, ok)."""
    fn = CALLABLES.get(name)
    if fn is None:
        return f"ERROR: unknown tool '{name}'", False
    try:
        result = fn(**inputs)
        ok = not result.startswith("ERROR:")
        return result, ok
    except Exception as e:
        return f"ERROR: {e}", False


def schemas_for(names: list[str]) -> list[dict]:
    """Return Anthropic tool schema dicts for the given tool names."""
    return [SCHEMAS[n] for n in names if n in SCHEMAS]
