from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import List, Dict

import structlog


@dataclass
class RunMetrics:
    processing_time: float
    token_usage: int
    num_rounds: int
    convergence_reason: str
    quality_scores: List[float] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class MetricsRecorder:
    """Record and summarize run metrics."""

    def __init__(self) -> None:
        self.runs: List[RunMetrics] = []
        self.logger = structlog.get_logger(__name__)

    def record_run(
        self,
        processing_time: float,
        token_usage: int,
        num_rounds: int,
        convergence_reason: str,
        quality_scores: List[float] | None = None,
    ) -> None:
        run = RunMetrics(
            processing_time=processing_time,
            token_usage=token_usage,
            num_rounds=num_rounds,
            convergence_reason=convergence_reason,
            quality_scores=quality_scores or [],
        )
        self.runs.append(run)
        self.logger.info("run_metrics", **asdict(run))

    def summary(self, last_n: int = 5) -> Dict[str, float | str | int]:
        recent = self.runs[-last_n:]
        if not recent:
            return {}
        avg_time = sum(r.processing_time for r in recent) / len(recent)
        avg_tokens = sum(r.token_usage for r in recent) / len(recent)
        avg_rounds = sum(r.num_rounds for r in recent) / len(recent)
        reasons: Dict[str, int] = {}
        for r in recent:
            reasons[r.convergence_reason] = reasons.get(r.convergence_reason, 0) + 1
        most_common = max(reasons, key=reasons.get)
        return {
            "runs": len(recent),
            "avg_processing_time": avg_time,
            "avg_token_usage": avg_tokens,
            "avg_rounds": avg_rounds,
            "most_common_reason": most_common,
        }


def summarize_recent_runs(recorder: MetricsRecorder, last_n: int = 5) -> str:
    data = recorder.summary(last_n)
    if not data:
        return "No runs recorded."
    return (
        f"Last {data['runs']} runs - Avg time: {data['avg_processing_time']:.2f}s, "
        f"Avg tokens: {data['avg_token_usage']:.0f}, Avg rounds: {data['avg_rounds']:.2f}, "
        f"Most common reason: {data['most_common_reason']}"
    )
