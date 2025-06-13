from __future__ import annotations

from typing import List

from .base import ThinkingStrategy


class FixedThinkingStrategy(ThinkingStrategy):
    """Strategy that always runs a fixed number of rounds."""

    def __init__(self, rounds: int = 1) -> None:
        self.rounds = rounds

    async def determine_rounds(self, prompt: str, *, request_id: str) -> int:
        return self.rounds

    async def should_continue(
        self,
        rounds_completed: int,
        quality_scores: List[float],
        responses: List[str],
    ) -> tuple[bool, str]:
        if rounds_completed >= self.rounds:
            return False, "fixed_rounds_complete"
        return True, "continue"
