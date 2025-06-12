"""Refactored chat engine using dependency injection and clean architecture."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os

import structlog

from core.interfaces import (
    CacheProvider,
    LLMProvider,
    QualityEvaluator,
)
from core.context_manager import ContextManager
from core.recursion import ConvergenceStrategy
from core.strategies import ThinkingStrategy, load_strategy
from monitoring.metrics import MetricsRecorder
from core.providers import (
    OpenRouterLLMProvider,
    OpenAILLMProvider,
    InMemoryLRUCache,
    EnhancedQualityEvaluator,
)
from core.model_policy import ModelSelector
from core.budget import BudgetManager
from core.cache_manager import CacheManager
from core.metrics_manager import MetricsManager
from core.conversation import ConversationManager
from core.tools import ToolRegistry, SearchTool, PythonExecutionTool
from api import fetch_models
from config import settings
import tiktoken


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
    max_context_tokens: int = 2000
    cache_size: int = 128
    max_retries: int = 3
    budget_token_limit: int = 100000
    enable_parallel_thinking: bool = True
    enable_tools: bool = True
    thinking_strategy: str = "adaptive"
    quality_thresholds: Optional[Dict[str, float]] = None


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
        *,
        cache_manager: Optional[CacheManager] = None,
        metrics_manager: Optional[MetricsManager] = None,
        metrics_recorder: Optional[MetricsRecorder] = None,
        budget_manager: Optional["BudgetManager"] = None,
        conversation_manager: Optional[ConversationManager] = None,
        tools: Optional[ToolRegistry] = None,
    ) -> None:
        self.llm = llm
        self.cache = cache
        self.evaluator = evaluator
        self.context_manager = context_manager
        self.thinking_strategy = thinking_strategy
        self.convergence_strategy = convergence_strategy or ConvergenceStrategy(
            evaluator.score,
            evaluator.score,
        )
        self.model_selector = model_selector
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

        if hasattr(self.thinking_strategy, "set_tools"):
            self.thinking_strategy.set_tools(self.tools)

    async def run_tool(self, name: str, task: str) -> str:
        """Execute a registered tool."""
        return await self.tools.run(name, task)
        
    async def think_and_respond(
        self,
        user_input: str,
        *,
        thinking_rounds: Optional[int] = None,
        alternatives_per_round: int = 3,
        temperature: float = 0.7,
        metadata: Optional[Dict] = None,
    ) -> ThinkingResult:
        """Execute recursive thinking process."""

        start_time = time.time()
        if self.budget_manager:
            start_cost = self.budget_manager.dollars_spent
        else:
            start_cost = 0.0
        metadata = metadata or {}
        
        logger.info(
            "thinking_start",
            user_input_length=len(user_input),
            override_rounds=thinking_rounds,
            metadata=metadata,
        )

        if hasattr(self.thinking_strategy, "preprocess_prompt"):
            user_input = await self.thinking_strategy.preprocess_prompt(
                user_input, self
            )
        
        # Determine number of rounds
        if thinking_rounds is None:
            thinking_rounds = await self.thinking_strategy.determine_rounds(user_input)
            
        logger.info("thinking_rounds_determined", rounds=thinking_rounds)
        
        # Get initial response
        messages = self.context_manager.optimize(
            self.conversation.get() + [{"role": "user", "content": user_input}]
        )
        
        initial_response = await self.cache_manager.chat(
            messages,
            temperature=temperature,
            role="assistant",
        )

        if self.budget_manager and self.budget_manager.will_exceed_budget(0):
            convergence_reason = "budget_exceeded"
            thinking_rounds = 0
        else:
            convergence_reason = "max_rounds"
        
        current_best = initial_response.content
        thinking_history: List[ThinkingRound] = []
        quality_scores: List[float] = []
        all_responses: List[str] = [current_best]
        total_tokens = initial_response.usage["total_tokens"]
        
        # Initial quality assessment
        initial_quality = self.evaluator.score(current_best, user_input)
        quality_scores.append(initial_quality)
        self.convergence_strategy.add(current_best, user_input)
        
        thinking_history.append(
            ThinkingRound(
                round_number=0,
                response=current_best,
                alternatives=[],
                selected=True,
                explanation="Initial response",
                quality_score=initial_quality,
                duration=time.time() - start_time,
            )
        )
        
        # Recursive thinking rounds
        rounds_completed = 0
        
        for round_num in range(1, thinking_rounds + 1):
            round_start = time.time()

            # Check if we should continue
            should_continue, reason = await self.thinking_strategy.should_continue(
                rounds_completed, quality_scores, all_responses
            )
            
            if not should_continue:
                convergence_reason = reason
                break

            conv_continue, conv_reason = self.convergence_strategy.should_continue(
                user_input
            )
            if not conv_continue:
                convergence_reason = conv_reason
                break
                
            logger.info("thinking_round_start", round=round_num)
            
            # Generate and evaluate alternatives
            best_response, alternatives, explanation, round_tokens = await self._generate_and_evaluate_alternatives(
                current_best,
                user_input,
                alternatives_per_round,
                temperature,
            )
            
            total_tokens += round_tokens

            if self.budget_manager and self.budget_manager.will_exceed_budget(0):
                convergence_reason = "budget_exceeded"
                break
            
            # Record all alternatives
            for i, alt in enumerate(alternatives):
                alt_quality = self.evaluator.score(alt, user_input)
                thinking_history.append(
                    ThinkingRound(
                        round_number=round_num,
                        response=alt,
                        alternatives=[],
                        selected=(alt == best_response),
                        explanation=explanation if alt == best_response else f"Alternative {i+1}",
                        quality_score=alt_quality,
                        duration=time.time() - round_start,
                    )
                )
                
            # Update tracking
            if best_response != current_best:
                current_best = best_response
                
            best_quality = self.evaluator.score(current_best, user_input)
            quality_scores.append(best_quality)
            all_responses.append(current_best)
            rounds_completed += 1

            cont, reason = self.convergence_strategy.update(current_best, user_input)
            if not cont:
                convergence_reason = reason
                break
            
            logger.info(
                "thinking_round_complete",
                round=round_num,
                quality_score=best_quality,
                improved=(best_response != all_responses[-2]),
            )
            
        # Update conversation history
        self.conversation.add("user", user_input)
        self.conversation.add("assistant", current_best)
        
        # Record metrics
        processing_time = time.time() - start_time
        
        self.metrics.record(
            processing_time=processing_time,
            token_usage=total_tokens,
            num_rounds=rounds_completed,
            convergence_reason=convergence_reason,
        )

        if self.budget_manager:
            cost_total = self.budget_manager.dollars_spent
            cost_this_step = cost_total - start_cost
        else:
            cost_total = 0.0
            cost_this_step = 0.0

        result = ThinkingResult(
            response=current_best,
            thinking_rounds=rounds_completed,
            thinking_history=thinking_history,
            total_tokens=total_tokens,
            processing_time=processing_time,
            convergence_reason=convergence_reason,
            metadata={
                **metadata,
                "quality_progression": quality_scores,
                "final_quality": quality_scores[-1] if quality_scores else 0,
            },
            cost_total=cost_total,
            cost_this_step=cost_this_step,
        )
        
        logger.info(
            "thinking_complete",
            rounds_completed=rounds_completed,
            total_tokens=total_tokens,
            processing_time=processing_time,
            convergence_reason=convergence_reason,
            final_quality=quality_scores[-1] if quality_scores else 0,
        )
        
        return result

    async def _generate_and_evaluate_alternatives(
        self,
        current_best: str,
        prompt: str,
        num_alternatives: int,
        temperature: float,
    ) -> tuple[str, List[str], str, int]:
        """Generate alternatives and select the best one."""
        
        # Batch generation prompt
        batch_prompt = f"""Current response to "{prompt}":
{current_best}

Generate {num_alternatives} alternative responses that could be better.
Then evaluate all options (including the current one) and select the best.

Respond in this JSON format:
{{
    "alternatives": ["alt1", "alt2", ...],
    "evaluation": {{
        "current": {{"score": 0-10, "strengths": "...", "weaknesses": "..."}},
        "1": {{"score": 0-10, "strengths": "...", "weaknesses": "..."}},
        "2": {{"score": 0-10, "strengths": "...", "weaknesses": "..."}}
    }},
    "selection": "current" or "1" or "2",
    "thinking": "Why this option is best"
}}"""

        messages = self.context_manager.optimize(
            self.conversation.get() + [{"role": "user", "content": batch_prompt}]
        )
        
        response = await self.cache_manager.chat(
            messages,
            temperature=temperature,
            role="critic",
        )
        
        # Parse response
        try:
            data = json.loads(response.content)
            alternatives = data.get("alternatives", [])[:num_alternatives]
            selection = data.get("selection", "current")
            thinking = data.get("thinking", "No thinking provided")
            
            # Determine selected response
            if selection == "current":
                best = current_best
            else:
                try:
                    idx = int(selection) - 1
                    best = alternatives[idx] if 0 <= idx < len(alternatives) else current_best
                except Exception:
                    best = current_best
                    
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON response", response=response.content[:200])
            # Fallback: treat response as single alternative
            alternatives = [response.content]
            best = response.content
            thinking = "JSON parsing failed, using raw response"

        return best, alternatives, thinking, response.usage["total_tokens"]


def create_default_engine(config: CoRTConfig) -> RecursiveThinkingEngine:
    """Convenience helper to build a thinking engine from a config."""

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

    try:
        tokenizer = tiktoken.get_encoding("cl100k_base")
    except Exception:
        class _SimpleTokenizer:
            def encode(self, text: str) -> List[str]:
                return text.split()

        tokenizer = _SimpleTokenizer()
    context_manager = ContextManager(
        max_tokens=config.max_context_tokens,
        tokenizer=tokenizer,
    )

    evaluator = EnhancedQualityEvaluator(thresholds=config.quality_thresholds)

    strategy = load_strategy(config.thinking_strategy, llm, evaluator)
    convergence = ConvergenceStrategy(evaluator.score, evaluator.score)

    tools = None
    if config.enable_tools:
        tools = ToolRegistry()
        tools.register(SearchTool())
        tools.register(PythonExecutionTool())

    budget = BudgetManager(default_model, token_limit=config.budget_token_limit)
    cache_manager = CacheManager(
        llm,
        cache,
        budget_manager=budget,
        model_selector=selector,
    )
    metrics_manager = MetricsManager(MetricsRecorder())
    conversation_manager = ConversationManager(
        llm,
        context_manager,
        budget_manager=budget,
    )

    return RecursiveThinkingEngine(
        llm=llm,
        cache=cache,
        evaluator=evaluator,
        context_manager=context_manager,
        thinking_strategy=strategy,
        convergence_strategy=convergence,
        model_selector=selector,
        cache_manager=cache_manager,
        metrics_manager=metrics_manager,
        budget_manager=budget,
        conversation_manager=conversation_manager,
        tools=tools,
    )
