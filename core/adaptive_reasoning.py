from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import List

from core.interfaces import LLMProvider, QualityEvaluator


@dataclass
class ReasoningState:
    current_best: str
    confidence: float
    history: List[float] = field(default_factory=list)
    stability_count: int = 0


class AdaptiveReasoner:
    """Iteratively improve a response with adaptive stopping."""

    def __init__(self, llm: LLMProvider, evaluator: QualityEvaluator) -> None:
        self.llm = llm
        self.evaluator = evaluator

    async def reason(self, prompt: str, *, max_rounds: int = 5) -> str:
        initial = await self.llm.chat([{"role": "user", "content": prompt}])
        score = self.evaluator.score(initial, prompt)
        state = ReasoningState(initial, score, [score])

        for _ in range(1, max_rounds + 1):
            if state.confidence > 0.95 and state.stability_count >= 2:
                break

            num_alts = self._num_alternatives(state.confidence)
            tasks = [
                asyncio.create_task(
                    self.llm.chat(
                        [{"role": "user", "content": f"{prompt}\nImprovement {i}"}]
                    )
                )
                for i in range(num_alts)
            ]
            done, _ = await asyncio.wait(tasks, timeout=30)
            best = state.current_best
            best_score = state.confidence
            for task in done:
                text = task.result()
                score = self.evaluator.score(text, prompt)
                if score > best_score:
                    best = text
                    best_score = score
            if best == state.current_best:
                state.stability_count += 1
            else:
                state.stability_count = 0
            state.current_best = best
            state.confidence = best_score
            state.history.append(best_score)

            if len(state.history) > 3:
                last3 = state.history[-3:]
                if max(last3) - min(last3) < 0.01:
                    break

        return state.current_best

    @staticmethod
    def _num_alternatives(conf: float) -> int:
        if conf > 0.9:
            return 2
        if conf > 0.7:
            return 3
        return 4
