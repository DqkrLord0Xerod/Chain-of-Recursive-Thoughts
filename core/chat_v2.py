"""Refactored chat engine using dependency injection and clean architecture."""

from __future__ import annotations

import json  # noqa: F401
import time  # noqa: F401
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os  # noqa: F401

import structlog

from core.strategies import ThinkingStrategy, strategy_from_config, load_strategy
from core.interfaces import (
    CacheProvider,
    LLMProvider,
    QualityEvaluator,
)
from core.context_manager import ContextManager
from core.recursion import ConvergenceStrategy
from monitoring.metrics import MetricsRecorder
from core.providers import (  # noqa: F401
    OpenRouterLLMProvider,
    OpenAILLMProvider,
    InMemoryLRUCache,
    EnhancedQualityEvaluator,
    OpenRouterEmbeddingProvider,
)
from core.planning import ImprovementPlanner
from core.model_policy import ModelSelector
from core.model_router import ModelRouter
from core.budget import BudgetManager
from core.cache_manager import CacheManager
from core.metrics_manager import MetricsManager
from core.conversation import ConversationManager
from core.tools import ToolRegistry, SearchTool, PythonExecutionTool  # noqa: F401
from core.memory import FaissMemoryStore
from api import fetch_models  # noqa: F401
from config import settings
from core.security import OutputFilter
import tiktoken  # noqa: F401


logger = structlog.get_logger(__name__)


@dataclass
class ThinkingRound:
    """Represents one round of thinking."""
    round_number: int
    response: str
    alternatives: List[str]
    selected: bool
    explanation: str
    quality_score: float
    duration: float


@dataclass
class ThinkingResult:
    """Result of the recursive thinking process."""
    response: str
    thinking_rounds: int
    thinking_history: List[ThinkingRound]
    total_tokens: int
    processing_time: float
    convergence_reason: str
    metadata: Dict = field(default_factory=dict)
    cost_total: float = 0.0
    cost_this_step: float = 0.0


@dataclass
class CoRTConfig:
    """Configuration for building a default thinking engine."""

    api_key: str | None = field(default_factory=lambda: settings.openrouter_api_key)
    model: str | None = field(default_factory=lambda: settings.model)
    model_policy: Optional[Dict[str, str]] = None
    provider: str = field(default_factory=lambda: settings.llm_provider)
    providers: Optional[List[str]] = None
    provider_weights: Optional[List[float]] = None
    max_context_tokens: int = 2000
    cache_size: int = 128
    max_retries: int = 3
    budget_token_limit: int = 100000
    enable_parallel_thinking: bool = True
    enable_tools: bool = True
    thinking_strategy: str = field(
        default_factory=lambda: settings.thinking_strategy
    )
    quality_thresholds: Optional[Dict[str, float]] = None
    advanced_convergence: bool = False
    memory_dim: int = 1536
    memory_top_k: int = 3


class RecursiveThinkingEngine:
    """Clean, dependency-injected recursive thinking engine."""

    def __init__(
        self,
        llm: LLMProvider,
        cache: CacheProvider,
        evaluator: QualityEvaluator,
        context_manager: ContextManager,
        thinking_strategy: ThinkingStrategy,
        convergence_strategy: Optional[ConvergenceStrategy] = None,
        model_selector: Optional[ModelSelector] = None,
        model_router: Optional[ModelRouter] = None,
        *,
        cache_manager: Optional[CacheManager] = None,
        metrics_manager: Optional[MetricsManager] = None,
        metrics_recorder: Optional[MetricsRecorder] = None,
        budget_manager: Optional["BudgetManager"] = None,
        conversation_manager: Optional[ConversationManager] = None,
        tools: Optional[ToolRegistry] = None,
        planner: Optional["ImprovementPlanner"] = None,
        memory_store: Optional["FaissMemoryStore"] = None,
        output_filter: Optional["OutputFilter"] = None,
    ) -> None:
        self.llm = llm
        self.cache = cache
        self.evaluator = evaluator
        self.context_manager = context_manager
        self.thinking_strategy = thinking_strategy
        self.convergence_strategy = convergence_strategy or ConvergenceStrategy(
            evaluator.score,
            evaluator.score,
            max_iterations=5,
            advanced=False,  # Will be passed explicitly in create_default_engine
        )
        self.model_selector = model_selector
        self.model_router = model_router
        self.budget_manager = budget_manager
        self.cache_manager = cache_manager or CacheManager(
            llm,
            cache,
            budget_manager=budget_manager,
            model_selector=model_selector,
        )
        self.metrics = metrics_manager or MetricsManager(metrics_recorder)
        self.conversation = conversation_manager or ConversationManager(
            llm,
            context_manager,
            budget_manager=budget_manager,
        )
        self.tools = tools or ToolRegistry()
        self.planner = planner
        self.memory_store = memory_store

        from core.loop_controller import LoopController

        self.loop_controller = LoopController(self)
        self.output_filter = output_filter

        if hasattr(self.thinking_strategy, "set_tools"):
            self.thinking_strategy.set_tools(self.tools)

    async def run_tool(self, name: str, task: str) -> str:
        """Execute a registered tool."""
        return await self.tools.run(name, task)

    async def think_and_respond(
        self,
        prompt: str,
        *,
        thinking_rounds: Optional[int] = None,
        alternatives_per_round: int = 3,
        temperature: float = 0.7,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> ThinkingResult:
        """Execute the recursive loop via :class:`LoopController`."""
        result = await self.loop_controller.respond(
            prompt,
            thinking_rounds=thinking_rounds,
            alternatives_per_round=alternatives_per_round,
            temperature=temperature,
            session_id=session_id,
            metadata=metadata,
        )
        if self.output_filter:
            result.response = self.output_filter.filter(result.response)
        return result


def create_default_engine(
    config: CoRTConfig,
    *,
    router: Optional[ModelRouter] = None,
    budget_manager: Optional[BudgetManager] = None,
    output_filter: Optional[OutputFilter] = None,
) -> RecursiveThinkingEngine:
    """Build a :class:`RecursiveThinkingEngine` from configuration."""

    if config.provider.lower() == "openai":
        llm = OpenAILLMProvider(
            api_key=config.api_key or os.getenv("OPENAI_API_KEY"),
            model=config.model,
            max_retries=config.max_retries,
        )
    else:
        llm = OpenRouterLLMProvider(
            api_key=config.api_key or os.getenv("OPENROUTER_API_KEY"),
            model=config.model,
            max_retries=config.max_retries,
        )

    cache = InMemoryLRUCache(max_size=config.cache_size)
    evaluator = EnhancedQualityEvaluator(thresholds=config.quality_thresholds)

    tokenizer = tiktoken.get_encoding("cl100k_base")
    ctx_mgr = ContextManager(config.max_context_tokens, tokenizer)

    strategy = strategy_from_config(config, llm, evaluator)
    strategy = load_strategy(config.thinking_strategy, llm, evaluator)
    convergence = ConvergenceStrategy(
        evaluator.score,
        evaluator.score,
        advanced=config.advanced_convergence,
    )

    tools = ToolRegistry()
    if config.enable_tools:
        tools.register(SearchTool())
        tools.register(PythonExecutionTool())

    return RecursiveThinkingEngine(
        llm=llm,
        cache=cache,
        evaluator=evaluator,
        context_manager=ctx_mgr,
        thinking_strategy=strategy,
        convergence_strategy=convergence,
        tools=tools,
        model_router=router,
        budget_manager=budget_manager,
        tools=tools,
        output_filter=output_filter,
    )
