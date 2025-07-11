"""Optimized recursive thinking engine with advanced techniques."""

from __future__ import annotations

import hashlib
import json
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
from core.model_router import ModelRouter
from core.budget import BudgetManager
from api import fetch_models
from core.providers import (
    InMemoryLRUCache,
    EnhancedQualityEvaluator,
    CriticLLM,
)
from core.optimization.parallel_thinking import (
    ParallelThinkingOptimizer,
    AdaptiveThinkingOptimizer,
)
from core.recursion import ConvergenceStrategy
from core.loop_controller import LoopController
from monitoring.telemetry import trace_method
from core.security import CredentialManager


logger = structlog.get_logger(__name__)


credential_manager = CredentialManager()


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
        model_router: Optional[ModelRouter] = None,
        budget_manager: Optional[BudgetManager] = None,
        critic: Optional[CriticLLM] = None,
        enable_parallel: bool = True,
        enable_adaptive: bool = True,
        enable_compression: bool = True,
        max_cache_size: int = 10000,
        convergence_strategy: Optional[ConvergenceStrategy] = None,
    ):

        self.model_router = model_router
        self.budget_manager = budget_manager
        self.llm = llm  # type: ignore[assignment]
        self.cache = cache
        self.evaluator = evaluator
        self.critic = critic
        self.convergence_strategy = convergence_strategy or ConvergenceStrategy(
            lambda a, b: evaluator.score(a, b),
            evaluator.score,
            max_iterations=5,
        )

        # Optimizers
        self.parallel_optimizer = ParallelThinkingOptimizer(
            llm,
            evaluator,
            critic=critic,
            max_parallel=3,
            quality_threshold=evaluator.thresholds.get("overall", 0.92),
        ) if enable_parallel else None

        self.adaptive_optimizer = AdaptiveThinkingOptimizer(
            self.parallel_optimizer,
        ) if enable_adaptive and self.parallel_optimizer else None

        self.enable_compression = enable_compression
        self.enable_adaptive = enable_adaptive
        self.prompt_history: List[str] = []

        # Controller handling the main loop
        self.loop_controller = LoopController(self)

        # Semantic cache for similar prompts
        self.semantic_cache: Dict[str, List[Tuple[str, str, float]]] = {}
        self.max_cache_size = max_cache_size

    async def _score_response(self, response: str, prompt: str) -> float:
        """Score a response using evaluator and optional critic."""
        score = self.evaluator.score(response, prompt)
        if self.critic:
            try:
                score = await self.critic.score(response, prompt)
            except Exception as e:  # pragma: no cover - logging
                logger.warning("critic_error", error=str(e))
        return score

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
        """Delegate to :class:`LoopController` for the main loop."""
        metadata = {"request_id": generate_request_id()}
        return await self.loop_controller.run_loop(
            prompt,
            context=context,
            max_thinking_time=max_thinking_time,
            target_quality=target_quality,
            metadata=metadata,
        )

    @trace_method("think_stream")
    async def think_stream(
        self,
        prompt: str,
        *,
        context: Optional[List[Dict[str, str]]] = None,
    ):
        """Stream progress using :class:`LoopController`."""
        metadata = {"request_id": generate_request_id()}
        async for update in self.loop_controller.run_stream(
            prompt, context=context, metadata=metadata
        ):
            yield update

    @trace_method("generate_initial")
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

    @trace_method("compress_prompt")
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

    @trace_method("categorize_prompt")
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

    @trace_method("check_semantic_cache")
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

    @trace_method("update_semantic_cache")
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

    @trace_method("simple_similarity")
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

    @trace_method("compute_cache_key")
    def _compute_cache_key(self, messages: List[Dict[str, str]]) -> str:
        """Compute deterministic cache key for messages."""

        content = json.dumps(messages, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    @trace_method("get_engine_stats")
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


def create_optimized_engine(
    config: CoRTConfig,
    *,
    router: Optional[ModelRouter] = None,
    budget_manager: Optional[BudgetManager] = None,
) -> OptimizedRecursiveEngine:
    """Build an :class:`OptimizedRecursiveEngine` from configuration."""

    selector: Optional[ModelSelector] = None
    default_model = config.model
    if config.model_policy:
        metadata = fetch_models()
        selector = ModelSelector(metadata, config.model_policy)

    router = ModelRouter(
        provider=config.provider,
        api_key=config.api_key,
        providers=config.providers,
        provider_weights=config.provider_weights,
        model=default_model,
        selector=selector,
        max_retries=config.max_retries,
    )

    critic = None
    if selector:
        try:
            critic = CriticLLM(router.provider_for_role("critic"))
        except Exception:  # pragma: no cover - optional critic
            critic = None
    router = router or ModelRouter.from_config(config, selector)
    llm = router.provider_for_role("assistant")

    critic = None
    try:
        critic_provider = router.provider_for_role("critic")
        critic = CriticLLM(critic_provider)
    except Exception:
        critic = None

    cache = InMemoryLRUCache(max_size=config.cache_size)
    evaluator = EnhancedQualityEvaluator(thresholds=config.quality_thresholds)
    convergence = ConvergenceStrategy(
        evaluator.score,
        evaluator.score,
        max_iterations=5,
        advanced=config.advanced_convergence,
    )

    return OptimizedRecursiveEngine(
        llm=llm,
        cache=cache,
        evaluator=evaluator,
        model_router=router,
        budget_manager=budget_manager,
        critic=critic,
        convergence_strategy=convergence,
        enable_parallel=config.enable_parallel_thinking,
    )
