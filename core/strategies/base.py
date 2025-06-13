from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple


class ThinkingStrategy(ABC):
    """Abstract base class for thinking strategies."""

    @abstractmethod
    async def determine_rounds(self, prompt: str, *, request_id: str) -> int:
        """Return the number of rounds to run for the given prompt."""

    @abstractmethod
    async def should_continue(
        self,
        rounds_completed: int,
        quality_scores: List[float],
        responses: List[str],

        *,
        request_id: str,
    ) -> Tuple[bool, str]:

        """Return whether to continue and the reason."""


class QualityEvaluator(ABC):
    """Abstract base class for quality evaluators."""

    thresholds: Dict[str, float]

    @abstractmethod
    def score(self, response: str, prompt: str) -> float:
        """Return a single overall quality score."""

    @abstractmethod
    def detailed_score(self, response: str, prompt: str) -> Dict[str, float]:
        """Return detailed quality metrics."""


class ImprovementPlanner(ABC):
    """Abstract base class for improvement planners."""

    @abstractmethod
    async def create_plan(self, prompt: str, current_response: str) -> str:
        """Generate a plan to improve the current response."""


class ConvergenceStrategy(ABC):
    """Abstract base class for convergence checking strategies."""

    @abstractmethod
    def add(self, response: str, prompt: str) -> None:
        """Add a response to the history."""

    @abstractmethod
    def update(self, response: str, prompt: str) -> Tuple[bool, str]:
        """Add response and immediately evaluate convergence."""

    @abstractmethod
    def should_continue(self, prompt: str) -> Tuple[bool, str]:
        """Evaluate whether processing should continue."""

    @property
    @abstractmethod
    def rolling_average(self) -> float:
        """Return the rolling average score."""

    @property
    @abstractmethod
    def reason_history(self) -> List[str]:
        """Return the list of recorded convergence reasons."""
