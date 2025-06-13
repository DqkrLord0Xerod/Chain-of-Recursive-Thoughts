"""Controllers for recursive thinking loops."""

from __future__ import annotations

import json
import os
import time

from dataclasses import dataclass, asdict
from typing import AsyncIterator, Dict, List, Optional, TYPE_CHECKING

import aiofiles
import structlog

from core.prompt_evolution import evolve_prompt
from monitoring.telemetry import generate_request_id, record_thinking_metrics

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from core.chat_v2 import ThinkingResult, ThinkingRound

logger = structlog.get_logger(__name__)


SESSION_DIR = "session_logs"


@dataclass
class LoopState:
    """Persisted state of a thinking loop."""

    rounds: List[ThinkingRound]
    scores: List[float]
    convergence_reason: str
    start_time: float
    end_time: float


class LoopController:
    """Execute the recursive thinking loop for an engine."""

    def __init__(self, engine) -> None:
        self.engine = engine

    async def _persist_state(self, session_id: str, state: LoopState) -> None:
        """Persist loop state to session log."""
        os.makedirs(SESSION_DIR, exist_ok=True)
        path = os.path.join(SESSION_DIR, f"{session_id}.json")
        try:
            async with aiofiles.open(path, "r") as f:
                data = json.loads(await f.read())
        except FileNotFoundError:
            data = []

        data.append(asdict(state))

        async with aiofiles.open(path, "w") as f:
            await f.write(json.dumps(data, indent=2))

    async def load_loop_history(self, session_id: str) -> List[LoopState]:
        """Load persisted loop history for a session."""
        from core.chat_v2 import ThinkingRound

        path = os.path.join(SESSION_DIR, f"{session_id}.json")
        try:
            async with aiofiles.open(path, "r") as f:
                raw = json.loads(await f.read())
        except FileNotFoundError:
            return []

        history = []
        for item in raw:
            rounds = [ThinkingRound(**r) for r in item["rounds"]]
            state = LoopState(
                rounds=rounds,
                scores=item.get("scores", []),
                convergence_reason=item.get("convergence_reason", ""),
                start_time=item.get("start_time", 0.0),
                end_time=item.get("end_time", 0.0),
            )
            history.append(state)
        return history

    async def get_convergence_reasons(self, session_id: str) -> List[str]:
        """Return convergence reasons for a session history."""
        history = await self.load_loop_history(session_id)
        return [h.convergence_reason for h in history]

    async def evaluate_step(self, prompt: str, response: str) -> float:
        """Score a response using the engine's evaluator and critic."""
        if hasattr(self.engine, "_score_response"):
            return await self.engine._score_response(response, prompt)
        return self.engine.evaluator.score(response, prompt)

    async def run_loop(
        self,
        prompt: str,
        *,
        context: Optional[List[Dict[str, str]]] = None,
        max_thinking_time: float = 30.0,
        target_quality: float = 0.9,
        metadata: Optional[Dict[str, object]] = None,
    ) -> Dict:
        """Run the optimized thinking loop."""
        start_time = time.time()
        metadata = metadata or {}
        request_id = metadata.get("request_id") or generate_request_id()
        metadata["request_id"] = request_id
        logger.info("loop_start", request_id=request_id, prompt=prompt)

        cached_response = await self.engine._check_semantic_cache(prompt)
        if cached_response:
            return {
                "response": cached_response,
                "cached": True,
                "thinking_time": 0.0,
                "metadata": {"cache_type": "semantic"},
            }

        if self.engine.enable_compression:
            evolved = evolve_prompt(prompt, self.engine.prompt_history)
            compressed = await self.engine._compress_prompt(evolved, context)
        else:
            compressed = prompt

        initial = await self.engine._generate_initial(compressed, context)
        initial_quality = await self.evaluate_step(prompt, initial.content)
        cont_conv, conv_reason = self.engine.convergence_strategy.update(
            initial.content,
            prompt,
        )
        if not cont_conv:
            await self.engine._update_semantic_cache(
                prompt, initial.content, initial_quality
            )
            self.engine.prompt_history.append(prompt)
            return {
                "response": initial.content,
                "cached": False,
                "thinking_time": time.time() - start_time,
                "thinking_rounds": 0,
                "initial_quality": initial_quality,
                "final_quality": initial_quality,
                "metadata": {"early_stop": conv_reason},
            }
        if initial_quality >= target_quality:
            await self.engine._update_semantic_cache(
                prompt, initial.content, initial_quality
            )
            self.engine.prompt_history.append(prompt)
            return {
                "response": initial.content,
                "cached": False,
                "thinking_time": time.time() - start_time,
                "thinking_rounds": 0,
                "initial_quality": initial_quality,
                "final_quality": initial_quality,
                "metadata": {"early_stop": "initial_good_enough"},
            }

        prompt_category = self.engine._categorize_prompt(prompt)
        if getattr(self.engine, "adaptive_optimizer", None) and self.engine.enable_adaptive:
            best_response, candidates, metrics = await self.engine.adaptive_optimizer.think_adaptive(
                prompt,
                initial.content,
                prompt_category,
            )
        elif getattr(self.engine, "parallel_optimizer", None):
            best_response, candidates, metrics = await self.engine.parallel_optimizer.think_parallel(
                prompt,
                initial.content,
            )
        else:
            best_response = initial.content
            candidates = []
            metrics = {"rounds": 0}

        thinking_time = time.time() - start_time
        final_quality = await self.evaluate_step(prompt, best_response)
        cont_conv, conv_reason = self.engine.convergence_strategy.update(
            best_response,
            prompt,
        )
        if not cont_conv:
            await self.engine._update_semantic_cache(prompt, best_response, final_quality)
            self.engine.prompt_history.append(prompt)
            return {
                "response": best_response,
                "cached": False,
                "thinking_time": thinking_time,
                "thinking_rounds": metrics.get("rounds", 0),
                "initial_quality": initial_quality,
                "final_quality": final_quality,
                "improvement": final_quality - initial_quality,
                "candidates_evaluated": len(candidates),
                "metadata": {"early_stop": conv_reason, **metrics},
            }
        await self.engine._update_semantic_cache(prompt, best_response, final_quality)

        record_thinking_metrics(
            metrics.get("rounds", 0),
            thinking_time,
            metrics.get("convergence_reason", "unknown"),
            initial_quality,
            final_quality,
            sum(c.tokens_used for c in candidates) if candidates else 0,
        )

        self.engine.prompt_history.append(prompt)
        logger.info(
            "loop_complete",
            request_id=request_id,
            rounds=metrics.get("rounds", 0),
            duration=thinking_time,
            final_quality=final_quality,
        )
        metrics["request_id"] = request_id
        return {
            "response": best_response,
            "cached": False,
            "thinking_time": thinking_time,
            "thinking_rounds": metrics.get("rounds", 0),
            "initial_quality": initial_quality,
            "final_quality": final_quality,
            "improvement": final_quality - initial_quality,
            "candidates_evaluated": len(candidates),
            "metadata": metrics,
        }

    async def run_stream(
        self, prompt: str, *, context: Optional[List[Dict[str, str]]] = None
    ) -> AsyncIterator[Dict]:
        """Yield progress updates for the thinking loop."""
        start_time = time.time()
        initial = await self.engine._generate_initial(prompt, context)
        quality = await self.evaluate_step(prompt, initial.content)
        yield {
            "stage": "initial",
            "response": initial.content,
            "quality": quality,
            "elapsed": time.time() - start_time,
        }
        if quality >= 0.9:
            return

        current_best = initial.content
        current_quality = quality
        for round_num in range(3):
            messages = [
                {
                    "role": "user",
                    "content": f"Improve: {prompt}\nCurrent: {current_best}",
                }
            ]
            alternative = await self.engine.llm.chat(
                messages, temperature=0.7 - round_num * 0.2
            )
            alt_quality = await self.evaluate_step(prompt, alternative.content)
            if alt_quality > current_quality:
                current_best = alternative.content
                current_quality = alt_quality
                yield {
                    "stage": f"round_{round_num + 1}",
                    "response": current_best,
                    "quality": current_quality,
                    "improvement": current_quality - quality,
                    "elapsed": time.time() - start_time,
                }
            if current_quality >= 0.9:
                break

    async def respond(
        self,
        prompt: str,
        *,
        thinking_rounds: Optional[int] = None,
        alternatives_per_round: int = 3,
        temperature: float = 0.7,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> "ThinkingResult":
        """High level loop used by RecursiveThinkingEngine."""
        from core.chat_v2 import (
            ThinkingRound as _ThinkingRound,
            ThinkingResult,
        )

        start_time = time.time()
        metadata = metadata or {}
        request_id = metadata.get("request_id") or generate_request_id()
        metadata["request_id"] = request_id
        logger.info("loop_start", request_id=request_id, prompt=prompt)

        if hasattr(self.engine.thinking_strategy, "preprocess_prompt"):
            prompt = await self.engine.thinking_strategy.preprocess_prompt(prompt, self.engine)

        memory_messages: List[Dict[str, str]] = []
        if getattr(self.engine, "memory_store", None):
            memory_messages = await self.engine.memory_store.retrieve_messages(prompt)

        history = self.engine.conversation.get()

        rounds = thinking_rounds
        if rounds is None:
            rounds = await self.engine.thinking_strategy.determine_rounds(
                prompt,
                request_id=request_id,
            )

        messages = memory_messages + history + [{"role": "user", "content": prompt}]

        stage_start = time.time()
        resp = await self.engine.cache_manager.chat(
            messages,
            temperature=temperature,
            role="assistant",
            metadata=metadata,
        )

        initial_duration = time.time() - stage_start

        best_response = resp.content
        total_tokens = resp.usage.get("total_tokens", 0)
        quality = await self.evaluate_step(prompt, best_response)
        cont_conv, convergence_reason = self.engine.convergence_strategy.update(best_response, prompt)
        if not cont_conv:
            thinking_history = [
                _ThinkingRound(
                    round_number=0,
                    response=best_response,
                    alternatives=[],
                    selected=True,
                    explanation="initial",
                    quality_score=quality,
                    duration=initial_duration,
                )
            ]
            self.engine.conversation.add("user", prompt)
            self.engine.conversation.add("assistant", best_response)
            processing_time = time.time() - start_time
            metadata["quality_progression"] = [quality]
            metadata["final_quality"] = quality
            self.engine.metrics.record(
                processing_time=processing_time,
                token_usage=total_tokens,
                num_rounds=0,
                convergence_reason=convergence_reason,
            )
            return ThinkingResult(
                response=best_response,
                thinking_rounds=0,
                thinking_history=thinking_history,
                total_tokens=total_tokens,
                processing_time=processing_time,
                convergence_reason=convergence_reason,
                metadata=metadata,
            )

        thinking_history = [
            _ThinkingRound(
                round_number=0,
                response=best_response,
                alternatives=[],
                selected=True,
                explanation="initial",
                quality_score=quality,
                duration=initial_duration,
            )
        ]
        quality_scores = [quality]
        responses = [best_response]
        convergence_reason = "complete"

        for round_num in range(1, rounds + 1):
            improve_prompt = (
                f"Given the prompt:\n{prompt}\nCurrent answer:\n{best_response}\n"
                f"Provide up to {alternatives_per_round} alternatives as JSON with keys 'alternatives', 'selection', 'thinking'."
            )

            messages = memory_messages + history + [{"role": "user", "content": improve_prompt}]
            alt_resp = await self.engine.cache_manager.chat(
                messages,
                temperature=temperature,
                role="assistant",
                metadata=metadata,
            )

            messages = memory_messages + history + [
                {"role": "user", "content": improve_prompt}
            ]
            stage_start = time.time()
            alt_resp = await self.engine.cache_manager.chat(
                messages, temperature=temperature, role="assistant"
            )
            round_duration = time.time() - stage_start

            total_tokens += alt_resp.usage.get("total_tokens", 0)
            try:
                data = json.loads(alt_resp.content)
                alts = data.get("alternatives", [])
                selection = data.get("selection", "current")
                explanation = data.get("thinking", "")
            except json.JSONDecodeError as exc:
                alts = []
                selection = "current"
                explanation = f"JSON parsing failed: {exc}"
                best_response = alt_resp.content
            else:
                if selection != "current":
                    try:
                        idx = int(selection) - 1
                        if 0 <= idx < len(alts):
                            best_response = alts[idx]
                        else:
                            explanation = "selection out of range"
                    except (ValueError, TypeError):
                        explanation = "invalid selection"
            quality = await self.evaluate_step(prompt, best_response)
            thinking_history.append(
                _ThinkingRound(
                    round_number=round_num,
                    response=best_response,
                    alternatives=alts,
                    selected=True,
                    explanation=explanation,
                    quality_score=quality,
                    duration=round_duration,
                )
            )
            quality_scores.append(quality)
            responses.append(best_response)
            cont_conv, conv_reason = self.engine.convergence_strategy.update(
                best_response,
                prompt,
            )
            if not cont_conv:
                convergence_reason = conv_reason
                break

            cont, reason = await self.engine.thinking_strategy.should_continue(
                round_num,
                quality_scores,
                responses,
                request_id=request_id,
            )
            convergence_reason = reason
            if not cont:
                break

        if getattr(self.engine, "planner", None):
            plan = await self.engine.planner.create_plan(prompt, best_response)
            metadata.setdefault("improvement_plans", []).append(plan)

        self.engine.conversation.add("user", prompt)
        self.engine.conversation.add("assistant", best_response)

        processing_time = time.time() - start_time
        metadata["quality_progression"] = quality_scores
        metadata["final_quality"] = quality_scores[-1]
        self.engine.metrics.record(
            processing_time=processing_time,
            token_usage=total_tokens,
            num_rounds=len(thinking_history) - 1,
            convergence_reason=convergence_reason,
            quality_scores=quality_scores,
        )

        loop_state = LoopState(
            rounds=thinking_history,
            scores=quality_scores,
            convergence_reason=convergence_reason,
            start_time=start_time,
            end_time=time.time(),
        )
        if session_id:
            await self._persist_state(session_id, loop_state)

        logger.info(
            "loop_complete",
            request_id=request_id,
            rounds=len(thinking_history) - 1,
            duration=processing_time,
            convergence_reason=convergence_reason,
        )

        from core.chat_v2 import ThinkingResult
        return ThinkingResult(

            response=best_response,
            thinking_rounds=len(thinking_history) - 1,
            thinking_history=thinking_history,
            total_tokens=total_tokens,
            processing_time=processing_time,
            convergence_reason=convergence_reason,
            metadata=metadata,
        )
