from __future__ import annotations

from typing import Callable, Dict, List, Tuple
import re


class ConvergenceTracker:
    """Track response quality to detect convergence or oscillation."""

    def __init__(
        self,
        similarity_fn: Callable[[str, str], float],
        score_fn: Callable[[str, str], float],
        similarity_threshold: float = 0.95,
        quality_threshold: float = 0.01,
        oscillation_threshold: float = 0.95,
        history_size: int = 5,
    ) -> None:
        self.similarity_fn = similarity_fn
        self.score_fn = score_fn
        self.similarity_threshold = similarity_threshold
        self.quality_threshold = quality_threshold
        self.oscillation_threshold = oscillation_threshold
        self.history_size = history_size
        self.history: List[Tuple[str, float]] = []

    def add(self, response: str, prompt: str) -> None:
        score = self.score_fn(response, prompt)
        self.history.append((response, score))
        if len(self.history) > self.history_size:
            self.history.pop(0)

    def update(self, response: str, prompt: str) -> Tuple[bool, str]:
        """Add a new response and immediately evaluate convergence."""
        self.add(response, prompt)
        return self.should_continue(prompt)

    def should_continue(self, prompt: str) -> Tuple[bool, str]:
        if len(self.history) < 2:
            return True, "insufficient history"

        prev_resp, prev_score = self.history[-2]
        curr_resp, curr_score = self.history[-1]

        similarity = self.similarity_fn(prev_resp, curr_resp)
        if similarity >= self.similarity_threshold:
            return False, "converged"

        improvement = curr_score - prev_score
        if improvement < self.quality_threshold:
            return False, "quality plateau"

        for old_resp, _ in self.history[:-2]:
            if (
                self.similarity_fn(old_resp, curr_resp)
                >= self.oscillation_threshold
            ):
                return False, "oscillation"

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
