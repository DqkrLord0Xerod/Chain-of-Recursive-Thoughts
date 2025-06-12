from __future__ import annotations

from typing import Callable

from core.interfaces import QualityEvaluator
from core.recursion import QualityAssessor


class DefaultQualityEvaluator(QualityEvaluator):
    """Wrap QualityAssessor into QualityEvaluator interface."""

    def __init__(self, similarity_fn: Callable[[str, str], float]) -> None:
        self.assessor = QualityAssessor(similarity_fn)
        self.thresholds = {"overall": 0.9}

    def score(self, response: str, prompt: str) -> float:
        return self.assessor.comprehensive_score(response, prompt)["overall"]
