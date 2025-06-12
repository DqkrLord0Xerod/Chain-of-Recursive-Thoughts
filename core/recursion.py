from __future__ import annotations

from typing import Callable, Dict, List, Tuple
from statistics import mean, pstdev
import re


class TrendConvergenceStrategy:
    """Strategy for detecting convergence using rolling trends."""

    def __init__(
        self,
        similarity_threshold: float = 0.95,
        improvement_threshold: float = 0.01,
        oscillation_threshold: float = 0.95,
        window: int = 3,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.improvement_threshold = improvement_threshold
        self.oscillation_threshold = oscillation_threshold
        self.window = window

    def detect_plateau(self, scores: List[float]) -> bool:
        """Return True if score improvements plateau based on moving averages."""
        if len(scores) < self.window * 2:
            return False
        recent = scores[-self.window:]
        previous = scores[-2 * self.window:-self.window]
        recent_avg = mean(recent)
        prev_avg = mean(previous)
        return (recent_avg - prev_avg) < self.improvement_threshold


class StatisticalConvergenceStrategy(TrendConvergenceStrategy):
    """Advanced strategy using simple statistics to detect convergence."""

    def __init__(
        self,
        similarity_threshold: float = 0.95,
        improvement_threshold: float = 0.01,
        oscillation_threshold: float = 0.95,
        window: int = 3,
        stddev_threshold: float = 0.005,
    ) -> None:
        super().__init__(
            similarity_threshold,
            improvement_threshold,
            oscillation_threshold,
            window,
        )
        self.stddev_threshold = stddev_threshold

    def detect_statistical(self, scores: List[float]) -> bool:
        """Detect convergence via slope and variance reduction."""
        if len(scores) < self.window:
            return False
        recent = scores[-self.window:]
        n = len(recent)
        x_mean = mean(range(n))
        y_mean = mean(recent)
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        if denominator == 0:
            return False
        slope = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(recent)) / denominator
        return abs(slope) < self.improvement_threshold and pstdev(recent) < self.stddev_threshold


class ConvergenceTracker:
    """Track response quality using rolling statistics to detect convergence."""

    def __init__(
        self,
        similarity_fn: Callable[[str, str], float],
        score_fn: Callable[[str, str], float],
        strategy: TrendConvergenceStrategy | None = None,
        history_size: int = 5,
    ) -> None:
        self.similarity_fn = similarity_fn
        self.score_fn = score_fn
        self.strategy = strategy or TrendConvergenceStrategy()
        self.history_size = history_size
        self.history: List[Tuple[str, float]] = []
        self.reason_history: List[str] = []

    def add(self, response: str, prompt: str) -> None:
        """Add a response and its quality score to the history."""
        score = self.score_fn(response, prompt)
        self.history.append((response, score))
        if len(self.history) > self.history_size:
            self.history.pop(0)

    @property
    def rolling_average(self) -> float:
        """Return the mean quality score of the recent history."""
        if not self.history:
            return 0.0
        return mean(score for _, score in self.history)

    def update(self, response: str, prompt: str) -> Tuple[bool, str]:
        """Add a new response and immediately evaluate convergence."""
        self.add(response, prompt)
        return self.should_continue(prompt)

    def should_continue(self, prompt: str) -> Tuple[bool, str]:
        if len(self.history) < 2:
            self.reason_history.append("insufficient history")
            return True, "insufficient history"

        prev_resp, _ = self.history[-2]
        curr_resp, _ = self.history[-1]

        similarity = self.similarity_fn(prev_resp, curr_resp)
        if similarity >= self.strategy.similarity_threshold:
            self.reason_history.append("converged")
            return False, "converged"

        scores = [s for _, s in self.history]
        if getattr(self.strategy, "detect_statistical", None):
            if self.strategy.detect_statistical(scores):
                self.reason_history.append("statistical convergence")
                return False, "statistical convergence"

        if self.strategy.detect_plateau(scores):
            self.reason_history.append("quality plateau")
            return False, "quality plateau"

        for old_resp, _ in self.history[:-2]:
            if (
                self.similarity_fn(old_resp, curr_resp)
                >= self.strategy.oscillation_threshold
            ):
                self.reason_history.append("oscillation")
                return False, "oscillation"

        self.reason_history.append("continue")
        return True, "continue"


class QualityAssessor:
    """Compute simple quality metrics for responses."""

    def __init__(self, similarity_fn: Callable[[str, str], float]) -> None:
        self.similarity_fn = similarity_fn

    def relevance(self, prompt: str, response: str) -> float:
        return self.similarity_fn(prompt, response)

    def completeness(self, prompt: str, response: str) -> float:
        words_prompt = set(prompt.lower().split())
        words_resp = set(response.lower().split())
        if not words_prompt:
            return 0.0
        return len(words_prompt & words_resp) / len(words_prompt)

    def clarity(self, response: str) -> float:
        if not response:
            return 0.0
        sentences = re.split(r"[.!?]+", response)
        sentences = [s for s in sentences if s.strip()]
        if not sentences:
            return 0.0
        avg_len = len(response.split()) / len(sentences)
        return max(0.0, 1.0 - (avg_len - 20) / 20)

    def accuracy(self, prompt: str, response: str) -> float:
        return self.similarity_fn(prompt, response)

    def comprehensive_score(self, response: str, prompt: str) -> Dict[str, float]:
        metrics = {
            "relevance": self.relevance(prompt, response),
            "completeness": self.completeness(prompt, response),
            "clarity": self.clarity(response),
            "accuracy": self.accuracy(prompt, response),
        }
        metrics["overall"] = sum(metrics.values()) / len(metrics)
        return metrics


class ConvergenceStrategy:
    """Unified interface wrapping tracker and trend detection."""

    def __init__(
        self,
        similarity_fn: Callable[[str, str], float],
        score_fn: Callable[[str, str], float],
        *,
        similarity_threshold: float = 0.95,
        improvement_threshold: float = 0.01,
        oscillation_threshold: float = 0.95,
        window: int = 3,
        history_size: int = 5,
        advanced: bool = False,
    ) -> None:
        strategy_cls = StatisticalConvergenceStrategy if advanced else TrendConvergenceStrategy
        self._tracker = ConvergenceTracker(
            similarity_fn,
            score_fn,
            strategy=strategy_cls(
                similarity_threshold,
                improvement_threshold,
                oscillation_threshold,
                window,
            ),
            history_size=history_size,
        )

    def add(self, response: str, prompt: str) -> None:
        """Add response to the history."""
        self._tracker.add(response, prompt)

    def update(self, response: str, prompt: str) -> Tuple[bool, str]:
        """Add and immediately check for convergence."""
        return self._tracker.update(response, prompt)

    def should_continue(self, prompt: str) -> Tuple[bool, str]:
        """Evaluate whether processing should continue."""
        return self._tracker.should_continue(prompt)

    @property
    def rolling_average(self) -> float:
        return self._tracker.rolling_average

    @property
    def reason_history(self) -> List[str]:
        return self._tracker.reason_history
