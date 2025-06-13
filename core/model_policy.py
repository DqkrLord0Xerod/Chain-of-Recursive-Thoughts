from __future__ import annotations

import logging
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

import asyncio

from core.interfaces import LLMProvider
from exceptions import APIError

logger = logging.getLogger(__name__)


@dataclass
class ModelCostMetrics:
    """Cost and performance metrics for a model."""
    model_id: str
    cost_per_token: float
    avg_quality_score: float
    avg_response_time: float
    reliability_score: float
    cost_efficiency: float  # quality per dollar
    usage_count: int


@dataclass
class CostQualityTradeoff:
    """Analysis of cost vs quality tradeoffs."""
    model_id: str
    cost_score: float  # Lower is better
    quality_score: float  # Higher is better
    efficiency_score: float  # quality/cost ratio
    recommended_use_cases: List[str]


@dataclass
class ModelFallbackChain:
    """Fallback chain optimized for budget constraints."""
    primary_model: str
    fallback_models: List[str]
    cost_thresholds: List[float]
    reasoning: str


class ModelSelector:
    """Select models for roles based on a policy with fallbacks."""

    def __init__(self, metadata: Iterable[Dict[str, Any]], policy: Dict[str, str]):
        self._available = [m.get("id") for m in metadata if m.get("id")]
        self._available_set = set(self._available)
        if not self._available:
            raise ValueError("No model metadata provided")
        self.policy = policy

    def model_for_role(self, role: str) -> str:
        preferred = self.policy.get(role)
        if preferred and preferred in self._available_set:
            return preferred
        default = self.policy.get("default")
        if default and default in self._available_set:
            return default
        return self._available[0]

    def map_roles(self, roles: Iterable[str]) -> Dict[str, str]:
        return {role: self.model_for_role(role) for role in roles}


async def parallel_provider_call(
    providers: List[LLMProvider],
    messages: List[Dict[str, str]],
    *,
    weights: Optional[List[float]] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
) -> Any:
    """Call providers concurrently and return the highest ranked response."""

    if not providers:
        raise ValueError("No providers supplied")

    weights = weights or [1.0] * len(providers)
    if len(weights) != len(providers):
        raise ValueError("weights length must match providers length")

    async def _call(p: LLMProvider):
        try:
            return await p.chat(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:  # pragma: no cover - debug logging
            return exc

    results = await asyncio.gather(*[_call(p) for p in providers])

    scored: List[tuple[float, Any]] = []
    for weight, result in zip(weights, results):
        if isinstance(result, Exception):
            continue
        score = len(result.content) * weight
        scored.append((score, result))

    if not scored:
        raise APIError("All providers failed")

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


class ModelCostAnalyzer:
    """Analyzes cost/quality trade-offs and provides intelligent model selection."""
    
    def __init__(self, model_catalog: List[Dict[str, Any]]):
        self.model_catalog = model_catalog
        self.model_metrics = {}
        self.usage_history = defaultdict(list)
        self.cost_benchmarks = {}
        self._initialize_cost_metrics()
        
    def _initialize_cost_metrics(self) -> None:
        """Initialize cost metrics from model catalog."""
        for model in self.model_catalog:
            model_id = model.get("id")
            if not model_id:
                continue
                
            pricing = model.get("pricing", {})
            prompt_cost = float(pricing.get("prompt", 0))
            completion_cost = float(pricing.get("completion", 0))
            cost_per_token = (prompt_cost + completion_cost) / 1000.0
            
            self.model_metrics[model_id] = ModelCostMetrics(
                model_id=model_id,
                cost_per_token=cost_per_token,
                avg_quality_score=0.5,  # Initialize with neutral score
                avg_response_time=1.0,
                reliability_score=0.9,
                cost_efficiency=0.5 / max(cost_per_token, 0.001),
                usage_count=0
            )
            
        logger.info(f"Initialized cost metrics for {len(self.model_metrics)} models")
    
    def analyze_cost_quality_tradeoffs(self, budget_constraint: Optional[float] = None) -> List[CostQualityTradeoff]:
        """Analyze cost vs quality tradeoffs for available models."""
        tradeoffs = []
        
        for model_id, metrics in self.model_metrics.items():
            if budget_constraint and metrics.cost_per_token > budget_constraint:
                continue
                
            # Normalize scores (0-1 scale)
            cost_score = 1.0 - min(metrics.cost_per_token / 0.01, 1.0)  # Lower cost = higher score
            quality_score = metrics.avg_quality_score
            efficiency_score = metrics.cost_efficiency
            
            # Determine recommended use cases
            use_cases = self._determine_use_cases(metrics)
            
            tradeoffs.append(CostQualityTradeoff(
                model_id=model_id,
                cost_score=cost_score,
                quality_score=quality_score,
                efficiency_score=efficiency_score,
                recommended_use_cases=use_cases
            ))
        
        # Sort by efficiency score (best tradeoff first)
        tradeoffs.sort(key=lambda x: x.efficiency_score, reverse=True)
        return tradeoffs
    
    def _determine_use_cases(self, metrics: ModelCostMetrics) -> List[str]:
        """Determine recommended use cases based on model characteristics."""
        use_cases = []
        
        if metrics.cost_per_token < 0.001:  # Very low cost
            use_cases.extend(["bulk_processing", "initial_drafts", "experimentation"])
        elif metrics.cost_per_token < 0.005:  # Medium cost
            use_cases.extend(["general_purpose", "content_generation", "analysis"])
        else:  # High cost
            use_cases.extend(["high_quality_output", "complex_reasoning", "final_review"])
            
        if metrics.avg_quality_score > 0.8:
            use_cases.append("premium_tasks")
        if metrics.reliability_score > 0.95:
            use_cases.append("production_critical")
            
        return use_cases
    
    def generate_fallback_chain(self,
                              primary_model: str,
                              budget_limit: float,
                              quality_threshold: float = 0.7) -> ModelFallbackChain:
        """Generate optimized fallback chain based on budget constraints."""
        
        if primary_model not in self.model_metrics:
            raise ValueError(f"Model {primary_model} not found in catalog")
        
        primary_metrics = self.model_metrics[primary_model]
        
        # Find suitable fallback models
        fallback_candidates = []
        for model_id, metrics in self.model_metrics.items():
            if (model_id != primary_model and
                metrics.avg_quality_score >= quality_threshold and
                metrics.cost_per_token < primary_metrics.cost_per_token):
                fallback_candidates.append((model_id, metrics))
        
        # Sort by cost efficiency
        fallback_candidates.sort(key=lambda x: x[1].cost_efficiency, reverse=True)
        
        # Build fallback chain with cost thresholds
        fallback_models = []
        cost_thresholds = []
        
        remaining_budget = budget_limit
        for model_id, metrics in fallback_candidates[:3]:  # Max 3 fallbacks
            if remaining_budget > metrics.cost_per_token * 1000:  # Assume 1000 token requests
                fallback_models.append(model_id)
                cost_thresholds.append(remaining_budget * 0.8)  # Use 80% of remaining budget
                remaining_budget *= 0.5  # Reserve half for next tier
        
        reasoning = f"Chain optimized for ${budget_limit:.4f} budget with {quality_threshold:.1f} quality threshold"
        
        return ModelFallbackChain(
            primary_model=primary_model,
            fallback_models=fallback_models,
            cost_thresholds=cost_thresholds,
            reasoning=reasoning
        )
    
    def recommend_model_for_budget(self,
                                 budget_per_request: float,
                                 quality_requirements: Optional[Dict[str, float]] = None,
                                 role: str = "general") -> str:
        """Recommend optimal model for given budget and quality requirements."""
        
        quality_requirements = quality_requirements or {"overall": 0.7}
        min_quality = quality_requirements.get("overall", 0.7)
        
        # Filter models that meet budget and quality requirements
        suitable_models = []
        for model_id, metrics in self.model_metrics.items():
            estimated_cost = metrics.cost_per_token * 1000  # Assume 1000 tokens
            if (estimated_cost <= budget_per_request and
                metrics.avg_quality_score >= min_quality):
                suitable_models.append((model_id, metrics))
        
        if not suitable_models:
            # Fallback to cheapest available model
            cheapest = min(self.model_metrics.items(), key=lambda x: x[1].cost_per_token)
            logger.warning(f"No models meet budget ${budget_per_request:.4f}, using cheapest: {cheapest[0]}")
            return cheapest[0]
        
        # Select best efficiency model
        best_model = max(suitable_models, key=lambda x: x[1].cost_efficiency)
        logger.info(f"Recommended {best_model[0]} for ${budget_per_request:.4f} budget")
        return best_model[0]
    
    def update_model_performance(self,
                               model_id: str,
                               quality_score: float,
                               response_time: float,
                               cost: float,
                               success: bool = True) -> None:
        """Update model performance metrics based on actual usage."""
        
        if model_id not in self.model_metrics:
            logger.warning(f"Unknown model {model_id}, skipping metrics update")
            return
        
        metrics = self.model_metrics[model_id]
        
        # Update usage history
        self.usage_history[model_id].append({
            'timestamp': time.time(),
            'quality_score': quality_score,
            'response_time': response_time,
            'cost': cost,
            'success': success
        })
        
        # Update running averages
        recent_history = self.usage_history[model_id][-50:]  # Last 50 requests
        if recent_history:
            metrics.avg_quality_score = sum(h['quality_score'] for h in recent_history) / len(recent_history)
            metrics.avg_response_time = sum(h['response_time'] for h in recent_history) / len(recent_history)
            metrics.reliability_score = sum(h['success'] for h in recent_history) / len(recent_history)
            
            # Update cost efficiency
            avg_cost = sum(h['cost'] for h in recent_history) / len(recent_history)
            metrics.cost_efficiency = metrics.avg_quality_score / max(avg_cost, 0.001)
            
        metrics.usage_count += 1
        
        logger.debug(f"Updated metrics for {model_id}: quality={metrics.avg_quality_score:.3f}, "
                    f"efficiency={metrics.cost_efficiency:.3f}")
    
    def get_cost_efficiency_report(self) -> Dict[str, Any]:
        """Generate comprehensive cost efficiency report."""
        
        # Calculate overall statistics
        total_usage = sum(metrics.usage_count for metrics in self.model_metrics.values())
        avg_efficiency = sum(metrics.cost_efficiency for metrics in self.model_metrics.values()) / len(self.model_metrics)
        
        # Find best and worst performers
        best_efficiency = max(self.model_metrics.items(), key=lambda x: x[1].cost_efficiency)
        worst_efficiency = min(self.model_metrics.items(), key=lambda x: x[1].cost_efficiency)
        
        # Generate insights
        insights = []
        for model_id, metrics in self.model_metrics.items():
            if metrics.usage_count > 0:
                if metrics.cost_efficiency > avg_efficiency * 1.5:
                    insights.append(f"{model_id} shows excellent cost efficiency")
                elif metrics.cost_efficiency < avg_efficiency * 0.5:
                    insights.append(f"{model_id} may be overpriced for current use cases")
        
        return {
            "total_models": len(self.model_metrics),
            "total_usage": total_usage,
            "avg_cost_efficiency": avg_efficiency,
            "best_efficiency_model": best_efficiency[0],
            "worst_efficiency_model": worst_efficiency[0],
            "model_metrics": {
                model_id: {
                    "cost_per_token": metrics.cost_per_token,
                    "avg_quality": metrics.avg_quality_score,
                    "cost_efficiency": metrics.cost_efficiency,
                    "usage_count": metrics.usage_count,
                    "reliability": metrics.reliability_score
                }
                for model_id, metrics in self.model_metrics.items()
            },
            "insights": insights
        }


class EnhancedModelSelector(ModelSelector):
    """Enhanced ModelSelector with cost analysis integration."""
    
    def __init__(self, metadata: Iterable[Dict[str, Any]], policy: Dict[str, str],
                 cost_analyzer: Optional[ModelCostAnalyzer] = None):
        super().__init__(metadata, policy)
        self.cost_analyzer = cost_analyzer or ModelCostAnalyzer(list(metadata))
        
    def model_for_role_with_budget(self, role: str, budget_per_request: float) -> str:
        """Select model for role considering budget constraints."""
        
        # Get preferred model from policy
        preferred = self.policy.get(role)
        
        if preferred and preferred in self._available_set:
            # Check if preferred model fits budget
            if preferred in self.cost_analyzer.model_metrics:
                metrics = self.cost_analyzer.model_metrics[preferred]
                estimated_cost = metrics.cost_per_token * 1000  # Assume 1000 tokens
                if estimated_cost <= budget_per_request:
                    return preferred
        
        # Use cost analyzer to recommend budget-appropriate model
        return self.cost_analyzer.recommend_model_for_budget(budget_per_request, role=role)
    
    def get_fallback_chain_for_role(self, role: str, budget_limit: float) -> ModelFallbackChain:
        """Get optimized fallback chain for a role within budget."""
        primary_model = self.model_for_role(role)
        return self.cost_analyzer.generate_fallback_chain(primary_model, budget_limit)
