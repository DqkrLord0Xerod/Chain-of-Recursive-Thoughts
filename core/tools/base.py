from __future__ import annotations

from typing import Protocol, Dict


class Tool(Protocol):
    """Generic tool interface."""

    name: str
    description: str

    async def run(self, task: str) -> str:
        """Execute the tool."""
        ...


class ToolRegistry:
    """Simple registry for tools."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    async def run(self, name: str, task: str) -> str:
        tool = self._tools[name]
        return await tool.run(task)

    def get(self, name: str) -> Tool:
        return self._tools[name]
