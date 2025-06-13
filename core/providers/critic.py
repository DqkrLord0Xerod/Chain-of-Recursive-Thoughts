"""LLM-based critic for scoring responses."""

from __future__ import annotations

import structlog

from core.interfaces import LLMProvider
from monitoring.telemetry import generate_request_id

logger = structlog.get_logger(__name__)


class CriticLLM:
    """Wrapper that uses an LLM to rate responses."""

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    async def score(self, response: str, prompt: str) -> float:
        """Return a normalized score between 0 and 1."""
        critique = (
            "On a scale from 0 to 1, rate how well the following response answers "
            "the prompt. Only reply with the numeric score.\n\n"
            f"Prompt:\n{prompt}\n\nResponse:\n{response}\nScore:"
        )
        messages = [{"role": "user", "content": critique}]
        metadata = {"request_id": generate_request_id()}
        try:
            result = await self.llm.chat(
                messages,
                temperature=0,
                metadata=metadata,
            )
            text = result.content.strip().split()[0]
            value = float(text)
        except Exception as e:  # pragma: no cover - logging
            logger.warning(
                "critic_scoring_failed",
                error=str(e),
                request_id=metadata["request_id"],
            )
            return 0.0
        return max(0.0, min(1.0, value))
