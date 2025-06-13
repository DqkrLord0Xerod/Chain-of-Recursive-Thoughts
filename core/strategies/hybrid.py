from __future__ import annotations

import structlog

from core.interfaces import LLMProvider, QualityEvaluator
from core.tools import ToolRegistry

from .adaptive import AdaptiveThinkingStrategy

logger = structlog.get_logger(__name__)


class HybridToolStrategy(AdaptiveThinkingStrategy):
    """Adaptive strategy that can invoke tools before thinking."""

    def __init__(
        self,
        llm: LLMProvider,
        evaluator: QualityEvaluator,
        *,
        tools: ToolRegistry | None = None,
        **kwargs,
    ) -> None:
        super().__init__(llm, evaluator, **kwargs)
        self.tools = tools or ToolRegistry()

    def set_tools(self, tools: ToolRegistry) -> None:
        self.tools = tools

    async def preprocess_prompt(self, prompt: str, engine) -> str:
        lower = prompt.lower()
        if "search:" in lower:
            query = prompt.split("search:", 1)[1].strip()
            result = await engine.run_tool("search", query)
            prompt += f"\nSearch result:\n{result}"
        if "python:" in lower:
            code = prompt.split("python:", 1)[1].strip()
            result = await engine.run_tool("python", code)
            prompt += f"\nPython output:\n{result}"
        return prompt
