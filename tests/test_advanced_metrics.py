import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # noqa: E402

from monitoring.advanced_metrics import AdvancedMetricsCollector  # noqa: E402


def test_record_and_retrieve():
    collector = AdvancedMetricsCollector()
    collector.start_session("s1")
    collector.record_round("s1", 1, 0.5, 10, 0.1)
    collector.record_round("s1", 2, 0.6, 12, 0.2)
    progress = collector.get_progress("s1")
    assert progress == [0.5, 0.6]
