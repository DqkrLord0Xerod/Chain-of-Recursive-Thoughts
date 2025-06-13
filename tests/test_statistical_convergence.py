import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from core.recursion import (  # noqa: E402
    ConvergenceStrategy,
    StatisticalConvergenceStrategy,
)
from core.chat_v2 import CoRTConfig, create_default_engine  # noqa: E402
from core.model_router import ModelRouter  # noqa: E402
from core.budget import BudgetManager  # noqa: E402


def test_statistical_convergence_detection():
    strat = ConvergenceStrategy(
        lambda a, b: 0.0,
        lambda r, p: float(r),
        max_iterations=10,
        window=3,
        improvement_threshold=0.0005,
        advanced=True,
    )
    seq = ["0.1", "0.11", "0.111", "0.1112", "0.1111", "0.11109"]
    reason = ""
    for resp in seq:
        cont, reason = strat.update(resp, "p")
        if not cont:
            break
    assert not cont
    assert reason in ["statistical convergence", "quality plateau"]


def test_engine_advanced_switch():
    cfg = CoRTConfig(advanced_convergence=True)
    router = ModelRouter.from_config(cfg)
    budget = BudgetManager(cfg.model, token_limit=cfg.budget_token_limit, catalog=[{"id": cfg.model, "pricing": {}}])
    engine = create_default_engine(cfg, router=router, budget_manager=budget)
    assert isinstance(
        engine.convergence_strategy._tracker.strategy,
        StatisticalConvergenceStrategy,
    )
