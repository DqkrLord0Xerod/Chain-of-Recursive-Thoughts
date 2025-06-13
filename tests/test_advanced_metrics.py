import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # noqa: E402

from monitoring.advanced_metrics import AdvancedMetricsCollector  # noqa: E402
from monitoring.metrics_v2 import MetricsAnalyzer, ThinkingMetrics  # noqa: E402


def test_record_and_retrieve():
    collector = AdvancedMetricsCollector()
    collector.start_session("s1")
    collector.record_round("s1", 1, 0.5, 10, 0.1)
    collector.record_round("s1", 2, 0.6, 12, 0.2)
    progress = collector.get_progress("s1")
    assert progress == [0.5, 0.6]


def test_metrics_analyzer_stage_latency_and_convergence():
    analyzer = MetricsAnalyzer(window_size=10)
    metrics = ThinkingMetrics(
        session_id="s1",
        start_time=0.0,
        end_time=1.0,
        rounds_completed=1,
        convergence_reason="timeout",
        round_durations=[0.3, 0.5],
        quality_scores=[0.1, 0.2],
        token_usage_per_round=[10, 20],
    )

    analyzer.record_session(metrics)

    assert analyzer.convergence_counters["timeout"] == 1
    assert analyzer.stage_latency["initial"][-1] == 0.3
    assert analyzer.stage_latency["round_1"][-1] == 0.5
    summary = analyzer.get_summary_stats()
    assert "stage_latency" in summary
