"""Parallel thinking optimization for faster convergence."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ThinkingCandidate:
    """A candidate response with metadata."""
    response: str
    quality_score: float
    generation_time: float
    tokens_used: int
    source: str  # "initial", "alternative_1", etc.
    metadata: Dict = None


class ParallelThinkingOptimizer:
    """
    Optimize thinking through parallel generation and early stopping.
    
    Key optimizations:
    1. Generate alternatives in parallel
    2. Stop generation when quality threshold met
    3. Use progressive refinement (start with cheap/fast, escalate if needed)
    4. Batch similar prompts for efficiency
    """
    
    def __init__(
        self,
        llm_provider,
        quality_evaluator,
        *,
        max_parallel: int = 3,
        quality_threshold: float | None = None,
        timeout_per_round: float = 10.0,
        enable_progressive: bool = True,
    ):
        self.llm = llm_provider
        self.evaluator = quality_evaluator
        self.max_parallel = max_parallel
        self.quality_threshold = (
            quality_threshold
            if quality_threshold is not None
            else quality_evaluator.thresholds.get("overall", 0.9)
        )
        self.timeout_per_round = timeout_per_round
        self.enable_progressive = enable_progressive
        
    async def think_parallel(
        self,
        prompt: str,
        initial_response: str,
        *,
        max_rounds: int = 3,
        alternatives_per_round: int = 3,
        temperature_schedule: Optional[List[float]] = None,
    ) -> Tuple[str, List[ThinkingCandidate], Dict]:
        """
        Execute parallel thinking process.
        
        Returns:
            Best response, all candidates, and performance metrics
        """
        start_time = time.time()
        candidates = []
        
        # Initial candidate
        initial_score = self.evaluator.score(initial_response, prompt)
        candidates.append(ThinkingCandidate(
            response=initial_response,
            quality_score=initial_score,
            generation_time=0,
            tokens_used=len(initial_response.split()) * 2,  # Estimate
            source="initial",
        ))
        
        # Check if initial is good enough
        if initial_score >= self.quality_threshold:
            return initial_response, candidates, {
                "rounds": 0,
                "early_stopped": True,
                "total_time": time.time() - start_time,
            }
            
        # Temperature schedule for progressive refinement
        if temperature_schedule is None:
            temperature_schedule = [0.7, 0.5, 0.3] if self.enable_progressive else [0.7] * max_rounds
            
        # Parallel thinking rounds
        best_response = initial_response
        best_score = initial_score
        
        for round_num in range(max_rounds):
            temperature = temperature_schedule[min(round_num, len(temperature_schedule) - 1)]
            
            logger.info(
                "parallel_round_start",
                round=round_num + 1,
                current_best_score=best_score,
                temperature=temperature,
            )
            
            # Generate alternatives in parallel
            round_candidates = await self._generate_parallel_alternatives(
                prompt,
                best_response,
                num_alternatives=alternatives_per_round,
                temperature=temperature,
                round_num=round_num,
            )
            
            candidates.extend(round_candidates)
            
            # Find best candidate
            for candidate in round_candidates:
                if candidate.quality_score > best_score:
                    best_score = candidate.quality_score
                    best_response = candidate.response
                    
            # Early stopping if quality threshold met
            if best_score >= self.quality_threshold:
                logger.info(
                    "parallel_early_stop",
                    round=round_num + 1,
                    score=best_score,
                    threshold=self.quality_threshold,
                )
                break
                
            # Adaptive stopping - if improvement is minimal
            if round_num > 0:
                recent_scores = [c.quality_score for c in candidates[-alternatives_per_round*2:]]
                if max(recent_scores) - min(recent_scores) < 0.02:
                    logger.info("parallel_plateau_stop", round=round_num + 1)
                    break
                    
        metrics = {
            "rounds": round_num + 1,
            "total_candidates": len(candidates),
            "best_score": best_score,
            "improvement": best_score - initial_score,
            "total_time": time.time() - start_time,
            "early_stopped": best_score >= self.quality_threshold,
        }
        
        return best_response, candidates, metrics
        
    async def _generate_parallel_alternatives(
        self,
        prompt: str,
        current_best: str,
        num_alternatives: int,
        temperature: float,
        round_num: int,
    ) -> List[ThinkingCandidate]:
        """Generate alternatives in parallel with timeout."""
        
        tasks = []
        for i in range(min(num_alternatives, self.max_parallel)):
            task = asyncio.create_task(
                self._generate_single_alternative(
                    prompt,
                    current_best,
                    temperature=temperature,
                    variant=i,
                    round_num=round_num,
                )
            )
            tasks.append(task)
            
        # Wait with timeout
        done, pending = await asyncio.wait(
            tasks,
            timeout=self.timeout_per_round,
            return_when=asyncio.ALL_COMPLETED,
        )
        
        # Cancel timed out tasks
        for task in pending:
            task.cancel()
            
        # Collect results
        candidates = []
        for task in done:
            try:
                candidate = task.result()
                if candidate:
                    candidates.append(candidate)
            except Exception as e:
                logger.warning("parallel_generation_failed", error=str(e))
                
        return candidates
        
    async def _generate_single_alternative(
        self,
        prompt: str,
        current_best: str,
        temperature: float,
        variant: int,
        round_num: int,
    ) -> Optional[ThinkingCandidate]:
        """Generate a single alternative response."""
        
        start_time = time.time()
        
        # Vary the prompt slightly for diversity
        refinement_prompts = [
            f"Improve this response to '{prompt}':\n{current_best}\nMake it more accurate and complete.",
            f"Provide a better alternative to this response for '{prompt}':\n{current_best}\nFocus on clarity.",
            f"Rewrite this response to '{prompt}' to be more helpful:\n{current_best}",
            f"Given '{prompt}', enhance this response:\n{current_best}\nAdd missing details.",
        ]
        
        alternative_prompt = refinement_prompts[variant % len(refinement_prompts)]
        
        try:
            # Generate alternative
            messages = [{"role": "user", "content": alternative_prompt}]
            response = await self.llm.chat(
                messages,
                temperature=temperature,
            )
            
            alternative_text = response.content
            
            # Evaluate quality
            quality_score = self.evaluator.score(alternative_text, prompt)
            
            return ThinkingCandidate(
                response=alternative_text,
                quality_score=quality_score,
                generation_time=time.time() - start_time,
                tokens_used=response.usage["total_tokens"],
                source=f"round_{round_num + 1}_alt_{variant + 1}",
                metadata={"temperature": temperature},
            )
            
        except Exception as e:
            logger.error(
                "alternative_generation_error",
                variant=variant,
                error=str(e),
            )
            return None


class BatchThinkingOptimizer:
    """
    Optimize multiple thinking requests through batching.
    
    Useful for handling multiple concurrent requests efficiently.
    """
    
    def __init__(
        self,
        parallel_optimizer: ParallelThinkingOptimizer,
        *,
        batch_size: int = 5,
        batch_timeout: float = 0.1,
    ):
        self.optimizer = parallel_optimizer
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        
        self._pending_requests: List[Tuple[str, asyncio.Future]] = []
        self._batch_lock = asyncio.Lock()
        self._batch_task: Optional[asyncio.Task] = None
        
    async def think(self, prompt: str, initial_response: str) -> Tuple[str, Dict]:
        """
        Add thinking request to batch.
        
        Returns:
            Best response and metrics
        """
        future = asyncio.Future()
        
        async with self._batch_lock:
            self._pending_requests.append((prompt, future))
            
            # Start batch processor if not running
            if self._batch_task is None or self._batch_task.done():
                self._batch_task = asyncio.create_task(self._process_batch())
                
        # Wait for result
        result = await future
        return result
        
    async def _process_batch(self):
        """Process a batch of thinking requests."""
        await asyncio.sleep(self.batch_timeout)
        
        async with self._batch_lock:
            if not self._pending_requests:
                return
                
            # Take up to batch_size requests
            batch = self._pending_requests[:self.batch_size]
            self._pending_requests = self._pending_requests[self.batch_size:]
            
        # Process batch in parallel
        tasks = []
        for prompt, future in batch:
            task = asyncio.create_task(
                self._process_single(prompt, future)
            )
            tasks.append(task)
            
        await asyncio.gather(*tasks, return_exceptions=True)
        
    async def _process_single(self, prompt: str, future: asyncio.Future):
        """Process a single thinking request."""
        try:
            # Generate initial response first
            messages = [{"role": "user", "content": prompt}]
            initial_response = await self.optimizer.llm.chat(messages)
            
            # Run parallel thinking
            best, candidates, metrics = await self.optimizer.think_parallel(
                prompt,
                initial_response.content,
            )
            
            future.set_result((best, metrics))
            
        except Exception as e:
            future.set_exception(e)


class AdaptiveThinkingOptimizer:
    """
    Adaptive optimizer that learns from past performance.
    
    Adjusts parameters based on historical data to minimize
    tokens while maximizing quality.
    """
    
    def __init__(
        self,
        parallel_optimizer: ParallelThinkingOptimizer,
    ):
        self.optimizer = parallel_optimizer
        self.history: List[Dict] = []
        self.parameter_performance: Dict[str, List[float]] = {}
        
    async def think_adaptive(
        self,
        prompt: str,
        initial_response: str,
        prompt_category: Optional[str] = None,
    ) -> Tuple[str, List[ThinkingCandidate], Dict]:
        """
        Think with adaptive parameters based on prompt category.
        
        Args:
            prompt: User prompt
            initial_response: Initial response
            prompt_category: Category for parameter selection
            
        Returns:
            Best response, candidates, and metrics
        """
        # Get optimal parameters for this category
        params = self._get_optimal_parameters(prompt_category)
        
        # Run thinking with adapted parameters
        result = await self.optimizer.think_parallel(
            prompt,
            initial_response,
            max_rounds=params["max_rounds"],
            alternatives_per_round=params["alternatives"],
            temperature_schedule=params["temperatures"],
        )
        
        best_response, candidates, metrics = result
        
        # Record performance
        self._record_performance(prompt_category, params, metrics)
        
        return best_response, candidates, metrics
        
    def _get_optimal_parameters(self, category: Optional[str]) -> Dict:
        """Get optimal parameters based on historical performance."""
        
        # Default parameters
        default = {
            "max_rounds": 3,
            "alternatives": 3,
            "temperatures": [0.7, 0.5, 0.3],
        }
        
        if not category or category not in self.parameter_performance:
            return default
            
        # Analyze historical performance
        perf_data = self.parameter_performance[category]
        if len(perf_data) < 10:
            return default
            
        # Find parameters that maximize quality/token efficiency
        # This is a simplified version - in practice, use more sophisticated optimization
        recent_perfs = perf_data[-20:]
        
        avg_efficiency = sum(p["efficiency"] for p in recent_perfs) / len(recent_perfs)
        
        # Adjust parameters based on efficiency
        if avg_efficiency > 0.8:
            # High efficiency - can be more aggressive
            return {
                "max_rounds": 2,
                "alternatives": 2,
                "temperatures": [0.5, 0.3],
            }
        elif avg_efficiency < 0.4:
            # Low efficiency - need more exploration
            return {
                "max_rounds": 4,
                "alternatives": 4,
                "temperatures": [0.9, 0.7, 0.5, 0.3],
            }
        else:
            return default
            
    def _record_performance(self, category: Optional[str], params: Dict, metrics: Dict):
        """Record performance metrics for learning."""
        
        if not category:
            category = "general"
            
        if category not in self.parameter_performance:
            self.parameter_performance[category] = []
            
        # Calculate efficiency metric
        efficiency = metrics["improvement"] / (metrics["total_candidates"] * 0.001)
        
        self.parameter_performance[category].append({
            "params": params,
            "metrics": metrics,
            "efficiency": efficiency,
            "timestamp": time.time(),
        })
        
        # Keep only recent history
        if len(self.parameter_performance[category]) > 100:
            self.parameter_performance[category] = self.parameter_performance[category][-100:]
