from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class DetailedMetrics:
    session_id: str
    quality_progression: List[float] = field(default_factory=list)
    resource_usage: Dict[str, float] = field(default_factory=dict)
    error_count: int = 0


class AdvancedMetricsCollector:
    """Collect detailed metrics for analysis."""

    def __init__(self) -> None:
        self.sessions: Dict[str, DetailedMetrics] = {}
        self.global_stats = defaultdict(deque)

    def start_session(self, session_id: str) -> None:
        self.sessions[session_id] = DetailedMetrics(session_id)

    def record_round(
        self,
        session_id: str,
        round_num: int,
        quality_score: float,
        tokens_used: int,
        time_elapsed: float,
    ) -> None:
        metrics = self.sessions.get(session_id)
        if not metrics:
            metrics = DetailedMetrics(session_id)
            self.sessions[session_id] = metrics
        metrics.quality_progression.append(quality_score)
        metrics.resource_usage[f"round_{round_num}"] = {
            "tokens": tokens_used,
            "time": time_elapsed,
        }

    def increment_error(self, session_id: str) -> None:
        metrics = self.sessions.get(session_id)
        if metrics:
            metrics.error_count += 1

    def get_progress(self, session_id: str) -> List[float]:
        metrics = self.sessions.get(session_id)
        return metrics.quality_progression if metrics else []
