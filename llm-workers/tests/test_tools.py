"""Tests for tool implementations — no network required."""

import tempfile
from pathlib import Path

from llm_workers.tools import execute, read_file, run_python, schemas_for, shell, write_file


def test_run_python_basic():
    result, ok = execute("run_python", {"code": "print(1 + 1)"})
    assert ok
    assert "2" in result


def test_run_python_stderr():
    result, ok = execute("run_python", {"code": "import sys; sys.stderr.write('err\\n')"})
    assert "err" in result


def test_run_python_timeout():
    result, ok = execute("run_python", {"code": "import time; time.sleep(99)", "timeout": 1})
    assert not ok
    assert "timed out" in result


def test_run_python_syntax_error():
    result, ok = execute("run_python", {"code": "def ("})
    assert not ok or "Error" in result


def test_read_file_existing():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("hello world")
        path = f.name
    result = read_file(path)
    assert result == "hello world"
    Path(path).unlink()


def test_read_file_missing():
    result = read_file("/nonexistent/path/file.txt")
    assert result.startswith("ERROR")


def test_write_then_read():
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "out.txt")
        result, ok = execute("write_file", {"path": path, "content": "test content"})
        assert ok
        read_result = read_file(path)
        assert read_result == "test content"


def test_shell_basic():
    result, ok = execute("shell", {"command": "echo hello"})
    assert ok
    assert "hello" in result


def test_shell_timeout():
    result, ok = execute("shell", {"command": "sleep 99", "timeout": 1})
    assert not ok
    assert "timed out" in result


def test_schemas_for():
    schemas = schemas_for(["run_python", "read_file"])
    assert len(schemas) == 2
    names = {s["name"] for s in schemas}
    assert names == {"run_python", "read_file"}


def test_schemas_unknown_tool_ignored():
    schemas = schemas_for(["run_python", "nonexistent"])
    assert len(schemas) == 1


def test_unknown_tool():
    result, ok = execute("does_not_exist", {})
    assert not ok
    assert "unknown tool" in result
