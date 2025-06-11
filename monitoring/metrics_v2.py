"""Enhanced metrics with real-time analysis and insights."""

from __future__ import annotations

import time
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Deque, Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ThinkingMetrics:
    """Detailed metrics for a thinking session."""
    session_id: str
    start_time: float
    end_time: Optional[float] = None
    rounds_completed: int = 0
    convergence_reason: Optional[str] = None
    quality_scores: List[float] = field(default_factory=list)
    round_durations: List[float] = field(default_factory=list)
    token_usage_per_round: List[int] = field(default_factory=list)
    alternatives_generated: List[int] = field(default_factory=list)
    cache_hits: int = 0
    cache_misses: int = 0
    errors: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def duration(self) -> float:
        """Total duration of thinking process."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return sum(self.token_usage_per_round)

    @property
    def quality_improvement(self) -> float:
        """Quality improvement from first to last round."""
        if len(self.quality_scores) < 2:
            return 0.0
        return self.quality_scores[-1] - self.quality_scores[0]

    @property
    def efficiency_score(self) -> float:
        """
        Efficiency score based on quality improvement per token.
        Higher is better.
        """
        if self.total_tokens == 0:
            return 0.0
        return self.quality_improvement / (self.total_tokens / 1000)

    @property
    def convergence_speed(self) -> float:
        """
        How quickly the system converged to a good solution.
        Based on rounds needed to reach 90% of final quality.
        """
        if len(self.quality_scores) < 2:
            return 1.0

        target_quality = self.quality_scores[0] + 0.9 * self.quality_improvement

        for i, score in enumerate(self.quality_scores):
            if score >= target_quality:
                return 1.0 - (i / len(self.quality_scores))

        return 0.0


class MetricsAnalyzer:
    """
    Real-time analysis of CoRT metrics with insights and anomaly detection.
    """

    def __init__(
        self,
        *,
        window_size: int = 1000,
        anomaly_threshold: float = 3.0,  # Standard deviations
    ):
        self.window_size = window_size
        self.anomaly_threshold = anomaly_threshold

        # Sliding windows for various metrics
        self.sessions: Deque[ThinkingMetrics] = deque(maxlen=window_size)
        self.response_times: Deque[float] = deque(maxlen=window_size)
        self.quality_scores: Deque[float] = deque(maxlen=window_size)
        self.token_usage: Deque[int] = deque(maxlen=window_size)
        self.round_counts: Deque[int] = deque(maxlen=window_size)

        # Convergence tracking
        self.convergence_reasons: Dict[str, int] = defaultdict(int)

        # Provider performance
        self.provider_latencies: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=100)
        )
        self.provider_errors: Dict[str, int] = defaultdict(int)

        # Time-based metrics
        self.hourly_stats: Dict[int, Dict[str, Any]] = {}

        # Anomaly tracking
        self.anomalies: Deque[Dict[str, Any]] = deque(maxlen=100)

    def record_session(self, metrics: ThinkingMetrics) -> Dict[str, Any]:
        """
        Record a thinking session and return insights.

        Returns:
            Dictionary with insights and any detected anomalies
        """
        self.sessions.append(metrics)

        # Update sliding windows
        self.response_times.append(metrics.duration)
        if metrics.quality_scores:
            self.quality_scores.append(metrics.quality_scores[-1])
        self.token_usage.append(metrics.total_tokens)
        self.round_counts.append(metrics.rounds_completed)

        # Update convergence tracking
        if metrics.convergence_reason:
            self.convergence_reasons[metrics.convergence_reason] += 1

        # Check for anomalies
        anomalies = self._detect_anomalies(metrics)
        if anomalies:
            self.anomalies.extend(anomalies)

        # Update hourly stats
        hour = datetime.fromtimestamp(metrics.start_time).hour
        self._update_hourly_stats(hour, metrics)

        # Generate insights
        insights = self._generate_insights(metrics)

        return {
            "insights": insights,
            "anomalies": anomalies,
            "warnings": self._generate_warnings(metrics),
        }

    def _detect_anomalies(self, metrics: ThinkingMetrics) -> List[Dict[str, Any]]:
        """Detect anomalies in the session."""
        anomalies = []

        # Duration anomaly
        if len(self.response_times) > 10:
            mean_duration = np.mean(self.response_times)
            std_duration = np.std(self.response_times)

            if abs(metrics.duration - mean_duration) > self.anomaly_threshold * std_duration:
                anomalies.append({
                    "type": "duration_anomaly",
                    "severity": "high" if metrics.duration > mean_duration else "medium",
                    "value": metrics.duration,
                    "expected_range": (
                        mean_duration - 2 * std_duration,
                        mean_duration + 2 * std_duration
                    ),
                    "description": f"Duration {metrics.duration:.2f}s is unusual (expected {mean_duration:.2f}±{std_duration:.2f}s)",
                })

        # Token usage anomaly
        if len(self.token_usage) > 10:
            mean_tokens = np.mean(self.token_usage)
            std_tokens = np.std(self.token_usage)

            if abs(metrics.total_tokens - mean_tokens) > self.anomaly_threshold * std_tokens:
                anomalies.append({
                    "type": "token_usage_anomaly",
                    "severity": "medium",
                    "value": metrics.total_tokens,
                    "expected_range": (
                        int(mean_tokens - 2 * std_tokens),
                        int(mean_tokens + 2 * std_tokens)
                    ),
                    "description": f"Token usage {metrics.total_tokens} is unusual",
                })

        # Quality degradation
        if metrics.quality_improvement < -0.1:
            anomalies.append({
                "type": "quality_degradation",
                "severity": "high",
                "value": metrics.quality_improvement,
                "description": "Quality decreased during thinking process",
            })

        # Excessive rounds
        if metrics.rounds_completed > 4:
            anomalies.append({
                "type": "excessive_rounds",
                "severity": "low",
                "value": metrics.rounds_completed,
                "description": f"Used {metrics.rounds_completed} rounds (usually ≤4)",
            })

        return anomalies

    def _generate_insights(self, metrics: ThinkingMetrics) -> Dict[str, Any]:
        """Generate insights from the session."""
        insights = {
            "efficiency_score": metrics.efficiency_score,
            "convergence_speed": metrics.convergence_speed,
            "cache_effectiveness": (
                metrics.cache_hits / (metrics.cache_hits + metrics.cache_misses)
                if (metrics.cache_hits + metrics.cache_misses) > 0 else 0
            ),
        }

        # Compare to recent sessions
        if len(self.sessions) > 10:
            recent_efficiency = np.mean([s.efficiency_score for s in list(self.sessions)[-10:]])
            insights["efficiency_trend"] = "improving" if metrics.efficiency_score > recent_efficiency else "declining"

        # Token efficiency insights
        if metrics.rounds_completed > 0:
            tokens_per_round = metrics.total_tokens / metrics.rounds_completed
            insights["tokens_per_round"] = tokens_per_round

            if len(self.sessions) > 10:
                avg_tokens_per_round = np.mean([
                    s.total_tokens / s.rounds_completed
                    for s in list(self.sessions)[-10:]
                    if s.rounds_completed > 0
                ])
                insights["token_efficiency_vs_average"] = (
                    "efficient" if tokens_per_round < avg_tokens_per_round * 0.9
                    else "inefficient" if tokens_per_round > avg_tokens_per_round * 1.1
                    else "normal"
                )

        return insights

    def _generate_warnings(self, metrics: ThinkingMetrics) -> List[str]:
        """Generate warnings based on metrics."""
        warnings = []

        # High token usage
        if metrics.total_tokens > 10000:
            warnings.append(f"High token usage: {metrics.total_tokens} tokens")

        # Slow convergence
        if metrics.convergence_speed < 0.3:
            warnings.append("Slow convergence detected")

        # Many errors
        if len(metrics.errors) > 2:
            warnings.append(f"Multiple errors encountered: {len(metrics.errors)}")

        # Poor cache performance
        cache_total = metrics.cache_hits + metrics.cache_misses
        if cache_total > 10 and metrics.cache_hits / cache_total < 0.3:
            warnings.append(f"Poor cache hit rate: {metrics.cache_hits / cache_total:.1%}")

        return warnings

    def _update_hourly_stats(self, hour: int, metrics: ThinkingMetrics) -> None:
        """Update hourly statistics."""
        if hour not in self.hourly_stats:
            self.hourly_stats[hour] = {
                "request_count": 0,
                "total_duration": 0.0,
                "total_tokens": 0,
                "errors": 0,
                "quality_scores": [],
            }

        stats = self.hourly_stats[hour]
        stats["request_count"] += 1
        stats["total_duration"] += metrics.duration
        stats["total_tokens"] += metrics.total_tokens
        stats["errors"] += len(metrics.errors)
        if metrics.quality_scores:
            stats["quality_scores"].append(metrics.quality_scores[-1])

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics."""
        recent_sessions = list(self.sessions)[-100:]  # Last 100 sessions

        return {
            "total_sessions": len(self.sessions),
            "average_duration": np.mean(self.response_times) if self.response_times else 0,
            "average_quality": np.mean(self.quality_scores) if self.quality_scores else 0,
            "average_tokens": np.mean(self.token_usage) if self.token_usage else 0,
            "average_rounds": np.mean(self.round_counts) if self.round_counts else 0,
            "convergence_distribution": dict(self.convergence_reasons),
            "recent_efficiency": np.mean([s.efficiency_score for s in recent_sessions]),
            "anomaly_rate": len(self.anomalies) / len(self.sessions) if self.sessions else 0,
            "hourly_patterns": self._analyze_hourly_patterns(),
        }

    def _analyze_hourly_patterns(self) -> Dict[str, Any]:
        """Analyze patterns by hour of day."""
        if not self.hourly_stats:
            return {}

        patterns = {}
        for hour, stats in self.hourly_stats.items():
            if stats["request_count"] > 0:
                patterns[f"{hour:02d}:00"] = {
                    "requests": stats["request_count"],
                    "avg_duration": stats["total_duration"] / stats["request_count"],
                    "avg_tokens": stats["total_tokens"] / stats["request_count"],
                    "error_rate": stats["errors"] / stats["request_count"],
                    "avg_quality": (
                        np.mean(stats["quality_scores"])
                        if stats["quality_scores"] else 0
                    ),
                }

        return patterns

    def get_provider_stats(self) -> Dict[str, Any]:
        """Get provider-specific statistics."""
        stats = {}

        for provider, latencies in self.provider_latencies.items():
            if latencies:
                stats[provider] = {
                    "avg_latency": np.mean(latencies),
                    "p95_latency": np.percentile(latencies, 95),
                    "p99_latency": np.percentile(latencies, 99),
                    "error_count": self.provider_errors[provider],
                    "reliability": 1.0 - (
                        self.provider_errors[provider] /
                        (len(latencies) + self.provider_errors[provider])
                    ),
                }

        return stats

    def record_provider_latency(self, provider: str, latency: float) -> None:
        """Record provider latency."""
        self.provider_latencies[provider].append(latency)

    def record_provider_error(self, provider: str, error: str) -> None:
        """Record provider error."""
        self.provider_errors[provider] += 1

    def get_recommendations(self) -> List[Dict[str, Any]]:
        """Get system optimization recommendations."""
        recommendations = []

        if not self.sessions:
            return recommendations

        # Check average duration
        avg_duration = np.mean(self.response_times) if self.response_times else 0
        if avg_duration > 10:
            recommendations.append({
                "type": "performance",
                "priority": "high",
                "title": "High average response time",
                "description": f"Average response time is {avg_duration:.1f}s. Consider reducing thinking rounds or implementing hedging.",
                "impact": "user_experience",
            })

        # Check token usage
        avg_tokens = np.mean(self.token_usage) if self.token_usage else 0
        if avg_tokens > 5000:
            recommendations.append({
                "type": "cost",
                "priority": "medium",
                "title": "High token usage",
                "description": f"Average token usage is {int(avg_tokens)}. Consider optimizing prompts or context management.",
                "impact": "cost_reduction",
            })

        # Check cache performance
        total_cache = sum(s.cache_hits + s.cache_misses for s in list(self.sessions)[-100:])
        total_hits = sum(s.cache_hits for s in list(self.sessions)[-100:])

        if total_cache > 100 and total_hits / total_cache < 0.3:
            recommendations.append({
                "type": "performance",
                "priority": "medium",
                "title": "Poor cache hit rate",
                "description": f"Cache hit rate is only {total_hits/total_cache:.1%}. Consider increasing cache size or TTL.",
                "impact": "performance",
            })

        # Check convergence patterns
        total_convergence = sum(self.convergence_reasons.values())
        if total_convergence > 100:
            quality_plateaus = self.convergence_reasons.get("quality_plateau", 0)
            if quality_plateaus / total_convergence > 0.5:
                recommendations.append({
                    "type": "algorithm",
                    "priority": "low",
                    "title": "Frequent quality plateaus",
                    "description": "Over 50% of sessions end due to quality plateau. Consider adjusting improvement threshold.",
                    "impact": "efficiency",
                })

        return recommendations
