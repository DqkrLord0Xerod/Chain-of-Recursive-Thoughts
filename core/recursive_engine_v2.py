"""Optimized recursive thinking engine with advanced techniques."""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import structlog

from core.interfaces import (
    CacheProvider,
    LLMProvider,
    QualityEvaluator,
    LLMResponse,
)
from core.chat_v2 import CoRTConfig
from core.model_policy import ModelSelector
from api import fetch_models
from core.providers import (
    OpenRouterLLMProvider,
    OpenAILLMProvider,
    InMemoryLRUCache,
    EnhancedQualityEvaluator,
)
from core.optimization.parallel_thinking import (
    ParallelThinkingOptimizer,
    AdaptiveThinkingOptimizer,
)
from core.recursion import ConvergenceStrategy
from monitoring.telemetry import trace_method, record_thinking_metrics


logger = structlog.get_logger(__name__)


class OptimizedRecursiveEngine:
    """
    Highly optimized recursive thinking engine.

    Optimizations:
    1. Parallel alternative generation
    2. Adaptive parameter tuning
    3. Intelligent caching with semantic similarity
    4. Prompt compression and optimization
    5. Early stopping with multiple criteria
    6. Batch processing for efficiency
    """

    def __init__(
        self,
        llm: LLMProvider,
        cache: CacheProvider,
        evaluator: QualityEvaluator,
        *,
        enable_parallel: bool = True,
        enable_adaptive: bool = True,
        enable_compression: bool = True,
        max_cache_size: int = 10000,
        convergence_strategy: Optional[ConvergenceStrategy] = None,
    ):
        self.llm = llm
        self.cache = cache
        self.evaluator = evaluator
        self.convergence_strategy = convergence_strategy or ConvergenceStrategy(
            lambda a, b: evaluator.score(a, b),
            evaluator.score,
        )

        # Optimizers
        self.parallel_optimizer = ParallelThinkingOptimizer(
            llm,
            evaluator,
            max_parallel=3,
            quality_threshold=0.92,
        ) if enable_parallel else None

        self.adaptive_optimizer = AdaptiveThinkingOptimizer(
            self.parallel_optimizer,
        ) if enable_adaptive and self.parallel_optimizer else None

        self.enable_compression = enable_compression

        # Semantic cache for similar prompts
        self.semantic_cache: Dict[str, List[Tuple[str, str, float]]] = {}
        self.max_cache_size = max_cache_size

    @trace_method("recursive_think")
    async def think(
        self,
        prompt: str,
        *,
        context: Optional[List[Dict[str, str]]] = None,
        max_thinking_time: float = 30.0,
        target_quality: float = 0.9,
        enable_streaming: bool = False,
    ) -> Dict:
        """
        Execute optimized recursive thinking.

        Args:
            prompt: User prompt
            context: Conversation context
            max_thinking_time: Maximum time for thinking
            target_quality: Target quality score
            enable_streaming: Stream intermediate results

        Returns:
            Dictionary with response and metadata
        """
        start_time = time.time()

        # Check semantic cache first
        cached_response = await self._check_semantic_cache(prompt)
        if cached_response:
            return {
                "response": cached_response,
                "cached": True,
                "thinking_time": 0.0,
                "metadata": {"cache_type": "semantic"},
            }

        # Compress prompt if needed
        if self.enable_compression:
            compressed_prompt = await self._compress_prompt(prompt, context)
        else:
            compressed_prompt = prompt

        # Generate initial response
        initial_response = await self._generate_initial(compressed_prompt, context)

        # Check if initial is good enough
        initial_quality = self.evaluator.score(initial_response.content, prompt)
        if initial_quality >= target_quality:
            await self._update_semantic_cache(prompt, initial_response.content, initial_quality)
            return {
                "response": initial_response.content,
                "cached": False,
                "thinking_time": time.time() - start_time,
                "thinking_rounds": 0,
                "initial_quality": initial_quality,
                "final_quality": initial_quality,
                "metadata": {"early_stop": "initial_good_enough"},
            }

        # Determine prompt category for adaptive optimization
        prompt_category = self._categorize_prompt(prompt)

        # Run optimized thinking
        if self.adaptive_optimizer and self.enable_adaptive:
            best_response, candidates, metrics = await self.adaptive_optimizer.think_adaptive(
                prompt,
                initial_response.content,
                prompt_category,
            )
        elif self.parallel_optimizer:
            best_response, candidates, metrics = await self.parallel_optimizer.think_parallel(
                prompt,
                initial_response.content,
            )
        else:
            # Fallback to simple sequential thinking
            best_response = initial_response.content
            candidates = []
            metrics = {"rounds": 0}

        # Record metrics
        thinking_time = time.time() - start_time
        final_quality = self.evaluator.score(best_response, prompt)

        await self._update_semantic_cache(prompt, best_response, final_quality)

        record_thinking_metrics(
            rounds=metrics.get("rounds", 0),
            duration=thinking_time,
            convergence_reason=metrics.get("convergence_reason", "unknown"),
            initial_quality=initial_quality,
            final_quality=final_quality,
            total_tokens=sum(c.tokens_used for c in candidates) if candidates else 0,
        )

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

    async def think_stream(
        self,
        prompt: str,
        *,
        context: Optional[List[Dict[str, str]]] = None,
    ):
        """
        Stream thinking progress and intermediate results.

        Yields:
            Dictionary updates with current best response and metadata
        """
        start_time = time.time()

        # Initial response
        initial_response = await self._generate_initial(prompt, context)
        initial_quality = self.evaluator.score(initial_response.content, prompt)

        yield {
            "stage": "initial",
            "response": initial_response.content,
            "quality": initial_quality,
            "elapsed": time.time() - start_time,
        }

        if initial_quality >= 0.9:
            return

        # Stream improvements
        current_best = initial_response.content
        current_quality = initial_quality

        for round_num in range(3):
            # Generate alternatives
            messages = [{
                "role": "user",
                "content": f"Improve: {prompt}\nCurrent: {current_best}",
            }]

            alternative = await self.llm.chat(messages, temperature=0.7 - round_num * 0.2)
            alt_quality = self.evaluator.score(alternative.content, prompt)

            if alt_quality > current_quality:
                current_best = alternative.content
                current_quality = alt_quality

                yield {
                    "stage": f"round_{round_num + 1}",
                    "response": current_best,
                    "quality": current_quality,
                    "improvement": current_quality - initial_quality,
                    "elapsed": time.time() - start_time,
                }

            if current_quality >= 0.9:
                break

    async def _generate_initial(
        self,
        prompt: str,
        context: Optional[List[Dict[str, str]]],
    ) -> 'LLMResponse':
        """Generate initial response with caching."""

        messages = []
        if context:
            messages.extend(context)
        messages.append({"role": "user", "content": prompt})

        # Check exact cache
        cache_key = self._compute_cache_key(messages)
        cached = await self.cache.get(cache_key)
        if cached:
            logger.info("cache_hit", cache_key=cache_key[:8])
            return cached

        # Generate response
        response = await self.llm.chat(messages)

        # Cache response
        await self.cache.set(
            cache_key,
            response,
            ttl=3600,
            tags=["llm_response", "initial"],
        )

        return response

    async def _compress_prompt(
        self,
        prompt: str,
        context: Optional[List[Dict[str, str]]],
    ) -> str:
        """
        Compress prompt while preserving key information.

        Techniques:
        1. Remove redundancy
        2. Summarize verbose sections
        3. Extract key requirements
        """

        if len(prompt) < 500:  # Don't compress short prompts
            return prompt

        compression_prompt = f"""Compress this prompt while preserving ALL key information and requirements:

{prompt}

Rules:
- Keep all specific requirements
- Remove only redundancy and verbose explanations
- Maintain the original intent
- Output only the compressed version"""

        messages = [{"role": "user", "content": compression_prompt}]
        response = await self.llm.chat(messages, temperature=0.3)

        compressed = response.content

        # Verify compression didn't lose critical info
        if len(compressed) < len(prompt) * 0.3:  # Too aggressive
            logger.warning(
                "compression_too_aggressive",
                original_len=len(prompt),
                compressed_len=len(compressed),
            )
            return prompt

        return compressed

    def _categorize_prompt(self, prompt: str) -> str:
        """
        Categorize prompt for adaptive optimization.

        Categories help the system learn optimal parameters
        for different types of requests.
        """

        prompt_lower = prompt.lower()

        # Simple categorization - can be enhanced with ML
        if any(word in prompt_lower for word in ["code", "program", "function", "implement"]):
            return "coding"
        elif any(word in prompt_lower for word in ["explain", "what is", "how does", "why"]):
            return "explanation"
        elif any(word in prompt_lower for word in ["write", "essay", "story", "create"]):
            return "creative"
        elif any(word in prompt_lower for word in ["analyze", "compare", "evaluate"]):
            return "analysis"
        elif any(word in prompt_lower for word in ["solve", "calculate", "compute"]):
            return "problem_solving"
        else:
            return "general"

    async def _check_semantic_cache(self, prompt: str) -> Optional[str]:
        """Check semantic cache for similar prompts."""

        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]

        if prompt_hash not in self.semantic_cache:
            return None

        # Find best matching cached response
        best_match = None
        best_similarity = 0.0

        for cached_prompt, cached_response, cached_quality in self.semantic_cache[prompt_hash]:
            # Simple similarity - in production, use embeddings
            similarity = self._simple_similarity(prompt, cached_prompt)

            if similarity > 0.95 and similarity > best_similarity:
                best_match = cached_response
                best_similarity = similarity

        if best_match:
            logger.info(
                "semantic_cache_hit",
                similarity=best_similarity,
                prompt_hash=prompt_hash,
            )

        return best_match

    async def _update_semantic_cache(
        self,
        prompt: str,
        response: str,
        quality: float,
    ) -> None:
        """Update semantic cache with new response."""

        if quality < 0.7:  # Don't cache low quality
            return

        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]

        if prompt_hash not in self.semantic_cache:
            self.semantic_cache[prompt_hash] = []

        # Add to cache
        self.semantic_cache[prompt_hash].append((prompt, response, quality))

        # Limit cache size per hash
        if len(self.semantic_cache[prompt_hash]) > 10:
            # Keep only best quality responses
            self.semantic_cache[prompt_hash].sort(key=lambda x: x[2], reverse=True)
            self.semantic_cache[prompt_hash] = self.semantic_cache[prompt_hash][:10]

        # Global cache size limit
        total_cached = sum(len(v) for v in self.semantic_cache.values())
        if total_cached > self.max_cache_size:
            # Remove oldest entries
            oldest_hash = min(self.semantic_cache.keys())
            del self.semantic_cache[oldest_hash]

    def _simple_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity for semantic cache."""

        # Normalize
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()

        if text1 == text2:
            return 1.0

        # Token overlap
        tokens1 = set(text1.split())
        tokens2 = set(text2.split())

        if not tokens1 or not tokens2:
            return 0.0

        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        return intersection / union

    def _compute_cache_key(self, messages: List[Dict[str, str]]) -> str:
        """Compute deterministic cache key for messages."""

        content = json.dumps(messages, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    async def get_stats(self) -> Dict:
        """Get engine statistics."""

        cache_stats = await self.cache.stats() if hasattr(self.cache, 'stats') else {}

        semantic_cache_size = sum(len(v) for v in self.semantic_cache.values())

        stats = {
            "cache": cache_stats,
            "semantic_cache": {
                "size": semantic_cache_size,
                "buckets": len(self.semantic_cache),
            },
        }

        if self.adaptive_optimizer:
            stats["adaptive"] = {
                "categories": list(self.adaptive_optimizer.parameter_performance.keys()),
                "total_history": sum(
                    len(v) for v in self.adaptive_optimizer.parameter_performance.values()
                ),
            }

        return stats


def create_optimized_engine(config: CoRTConfig) -> OptimizedRecursiveEngine:
    """Build an :class:`OptimizedRecursiveEngine` from configuration."""

    selector: Optional[ModelSelector] = None
    default_model = config.model

    if config.model_policy:
        metadata = fetch_models()
        selector = ModelSelector(metadata, config.model_policy)
        default_model = selector.model_for_role("assistant")

    if config.provider.lower() == "openai":
        llm = OpenAILLMProvider(
            api_key=config.api_key or os.getenv("OPENAI_API_KEY"),
            model=default_model,
            max_retries=config.max_retries,
        )
    else:
        llm = OpenRouterLLMProvider(
            api_key=config.api_key or os.getenv("OPENROUTER_API_KEY"),
            model=default_model,
            max_retries=config.max_retries,
        )

    cache = InMemoryLRUCache(max_size=config.cache_size)

    evaluator = EnhancedQualityEvaluator()
    convergence = ConvergenceStrategy(evaluator.score, evaluator.score)

    return OptimizedRecursiveEngine(
        llm=llm,
        cache=cache,
        evaluator=evaluator,
        convergence_strategy=convergence,
        enable_parallel=config.enable_parallel_thinking,
    )
