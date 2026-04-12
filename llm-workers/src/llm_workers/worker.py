"""
Single agentic worker — runs one Task to completion via a sequential tool-calling loop.

Flow:
  1. Send initial prompt to model
  2. If stop_reason == "tool_use": execute each tool, feed results back, repeat
  3. If stop_reason == "end_turn": extract text content and return Result
  4. Safety cap: abort after max_tool_iterations loops
"""

from __future__ import annotations

import time

import anthropic

from .tools import execute, schemas_for
from .types import Result, Task, ToolCall, WorkerConfig


def run(task: Task, config: WorkerConfig, client: anthropic.Anthropic) -> Result:
    """Execute a single task with the full agentic loop. Thread-safe — no shared state."""
    t0 = time.time()
    tool_calls: list[ToolCall] = []
    total_in = 0
    total_out = 0
    total_thinking = 0

    active_tools = task.tools or []
    tool_schemas = schemas_for(active_tools)
    system = task.system or config.system

    messages: list[dict] = [{"role": "user", "content": task.prompt}]

    for iteration in range(config.max_tool_iterations + 1):
        kwargs: dict = dict(
            model=config.model,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            messages=messages,
        )
        if system:
            kwargs["system"] = system
        if tool_schemas:
            kwargs["tools"] = tool_schemas

        try:
            msg = client.messages.create(**kwargs)
        except Exception as e:
            return Result(
                task_id=task.id,
                ok=False,
                error=str(e),
                tool_calls=tool_calls,
                elapsed=time.time() - t0,
                tokens_in=total_in,
                tokens_out=total_out,
                iterations=iteration,
            )

        total_in += msg.usage.input_tokens
        total_out += msg.usage.output_tokens

        # Collect thinking chars
        for block in msg.content:
            if block.type == "thinking":
                total_thinking += len(block.thinking or "")

        if msg.stop_reason == "end_turn":
            text = next(
                (block.text for block in msg.content if block.type == "text"),
                "",
            )
            return Result(
                task_id=task.id,
                ok=True,
                content=text,
                tool_calls=tool_calls,
                elapsed=time.time() - t0,
                tokens_in=total_in,
                tokens_out=total_out,
                thinking_chars=total_thinking,
                iterations=iteration + 1,
            )

        if msg.stop_reason == "tool_use":
            # Execute every tool_use block and collect results
            assistant_content = msg.content
            tool_result_blocks: list[dict] = []

            for block in msg.content:
                if block.type != "tool_use":
                    continue

                result_str, ok = execute(block.name, block.input)
                tool_calls.append(ToolCall(name=block.name, input=block.input, result=result_str, ok=ok))
                tool_result_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_str,
                })

            # Feed results back into conversation
            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({"role": "user", "content": tool_result_blocks})
            continue

        # Unexpected stop reason
        return Result(
            task_id=task.id,
            ok=False,
            error=f"unexpected stop_reason={msg.stop_reason!r}",
            tool_calls=tool_calls,
            elapsed=time.time() - t0,
            tokens_in=total_in,
            tokens_out=total_out,
            thinking_chars=total_thinking,
            iterations=iteration + 1,
        )

    return Result(
        task_id=task.id,
        ok=False,
        error=f"hit max_tool_iterations={config.max_tool_iterations} without end_turn",
        tool_calls=tool_calls,
        elapsed=time.time() - t0,
        tokens_in=total_in,
        tokens_out=total_out,
        thinking_chars=total_thinking,
        iterations=config.max_tool_iterations,
    )
