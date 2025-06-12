"""Wrapper around MetricsRecorder."""

from __future__ import annotations

from typing import Optional

from monitoring.metrics import MetricsRecorder


class MetricsManager:
    """Delegates metrics recording."""

    def __init__(self, recorder: Optional[MetricsRecorder] = None) -> None:
        self.recorder = recorder

    def record(
        self,
        *,
        processing_time: float,
        token_usage: int,
        num_rounds: int,
        convergence_reason: str,
    ) -> None:
        if self.recorder:
            self.recorder.record_run(
                processing_time=processing_time,
                token_usage=token_usage,
                num_rounds=num_rounds,
                convergence_reason=convergence_reason,
            )
