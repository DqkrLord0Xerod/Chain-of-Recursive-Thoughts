import os
import sys
import importlib.util

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
spec = importlib.util.spec_from_file_location(
    "recursion", os.path.join(ROOT, "core", "recursion.py")
)
recursion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recursion)
ConvergenceTracker = recursion.ConvergenceTracker
TrendConvergenceStrategy = recursion.TrendConvergenceStrategy
QualityAssessor = recursion.QualityAssessor


def test_convergence_detection():
    tracker = ConvergenceTracker(lambda a, b: 1.0 if a == b else 0.0, lambda r, p: len(r))
    tracker.add("one", "p")
    tracker.add("one", "p")
    cont, reason = tracker.should_continue("p")
    assert not cont
    assert reason == "converged"


def test_quality_assessor():
    qa = QualityAssessor(lambda a, b: 1.0 if a == b else 0.0)
    score = qa.comprehensive_score("hello", "hello")
    assert score["overall"] >= 1.0


def test_rolling_average_and_plateau_detection():
    strategy = TrendConvergenceStrategy(improvement_threshold=0.05, window=2)
    tracker = ConvergenceTracker(
        lambda a, b: 0.0,
        lambda r, p: float(r),
        strategy=strategy,
    )

    for resp in ["0.1", "0.11", "0.115", "0.116"]:
        tracker.add(resp, "p")

    assert abs(tracker.rolling_average - 0.11025) < 1e-6
    cont, reason = tracker.should_continue("p")
    assert not cont
    assert reason == "quality plateau"
