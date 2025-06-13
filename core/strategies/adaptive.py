from __future__ import annotations

from typing import List

import structlog

from core.interfaces import LLMProvider, QualityEvaluator

from .base import ThinkingStrategy

logger = structlog.get_logger(__name__)


class AdaptiveThinkingStrategy(ThinkingStrategy):
    """Adaptive strategy that adjusts based on complexity and quality."""

    def __init__(
        self,
        llm: LLMProvider,
        evaluator: QualityEvaluator,
        *,
        min_rounds: int = 1,
        max_rounds: int = 5,
        quality_threshold: float | None = None,
        improvement_threshold: float = 0.01,
    ) -> None:
        self.llm = llm
        self.evaluator = evaluator
        self.min_rounds = min_rounds
        self.max_rounds = max_rounds
        self.quality_threshold = (
            quality_threshold
            if quality_threshold is not None
            else evaluator.thresholds.get("overall", 0.9)
        )
        self.improvement_threshold = improvement_threshold

    async def determine_rounds(self, prompt: str, *, request_id: str) -> int:
        """Use the LLM to determine optimal number of rounds."""
        meta_prompt = (
            f"Analyze this prompt and determine the optimal number of thinking rounds"
            f" (1-{self.max_rounds}):\n\n{prompt}\n\n"
            "Consider:\n- Complexity of the request\n- Need for accuracy\n"
            "- Type of response required\n\n"
            f"Respond with just a number between {self.min_rounds} "
            f"and {self.max_rounds}."
        )
        response = await self.llm.chat(
            [{"role": "user", "content": meta_prompt}],
            temperature=0.3,
            metadata={"request_id": request_id},
        )

        try:
            rounds = int("".join(filter(str.isdigit, response.content)))
            return max(self.min_rounds, min(rounds, self.max_rounds))
        except (ValueError, TypeError):
            logger.warning(
                "Failed to parse thinking rounds",
                response=response.content,
                request_id=request_id,
            )
            return 3

    async def should_continue(
        self,
        rounds_completed: int,
        quality_scores: List[float],
        responses: List[str],
        *,
        request_id: str,
    ) -> tuple[bool, str]:
        """Determine if thinking should continue."""
        if rounds_completed >= self.max_rounds:
            return False, "max_rounds_reached"

        if not quality_scores:
            return True, "no_scores_yet"

        if quality_scores[-1] >= self.quality_threshold:
            return False, "quality_threshold_met"

        if len(quality_scores) >= 3:
            recent_scores = quality_scores[-3:]
            improvement = max(recent_scores) - min(recent_scores)
            if improvement < self.improvement_threshold:
                return False, "quality_plateau"

        if len(responses) >= 3 and responses[-1] in responses[:-1]:
            return False, "oscillation_detected"

        return True, "continue"
