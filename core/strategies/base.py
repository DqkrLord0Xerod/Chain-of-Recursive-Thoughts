from __future__ import annotations

from typing import List, Protocol


class ThinkingStrategy(Protocol):
    """Protocol defining the thinking strategy interface."""

    async def determine_rounds(self, prompt: str, *, request_id: str) -> int:
        """Determine number of thinking rounds needed."""

    async def should_continue(
        self,
        rounds_completed: int,
        quality_scores: List[float],
        responses: List[str],
        *,
        request_id: str,
    ) -> tuple[bool, str]:
        """Return whether to continue and the reason."""
