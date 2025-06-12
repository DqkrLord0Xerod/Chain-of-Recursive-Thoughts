from __future__ import annotations

import asyncio
import contextlib
import io

from .base import Tool


class PythonExecutionTool:
    """Execute Python code and return stdout or error."""

    name = "python"
    description = "Run Python code snippets"

    async def run(self, task: str) -> str:
        def _execute() -> str:
            buffer = io.StringIO()
            try:
                with contextlib.redirect_stdout(buffer):
                    exec(task, {})
            except Exception as exc:
                return f"Error: {exc}"
            return buffer.getvalue().strip()

        return await asyncio.to_thread(_execute)

