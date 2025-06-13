from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass

import structlog


logger = structlog.get_logger(__name__)

from api import fetch_models

logger = logging.getLogger(__name__)


class BudgetManager:
    """Track token usage and compute costs for a model."""

    def __init__(self, model: str, token_limit: int, catalog: Optional[List[Dict[str, any]]] = None) -> None:
        self.model = model
        self.token_limit = token_limit
        self.tokens_used = 0
        self.dollars_spent = 0.0

        catalog = catalog or self._load_catalog()
        self.pricing = self._find_pricing(catalog)
        self._cost_per_token = self._compute_cost_per_token()

    @staticmethod
    def _load_catalog() -> List[Dict[str, any]]:
        try:
            return fetch_models()
        except Exception:
            return []

    def _find_pricing(self, catalog: List[Dict[str, any]]) -> Dict[str, float]:
        for entry in catalog:
            if entry.get("id") == self.model:
                return entry.get("pricing", {})
        return {}

    def _compute_cost_per_token(self) -> float:
        prompt = float(self.pricing.get("prompt", 0))
        completion = float(self.pricing.get("completion", 0))
        if prompt == 0 and completion == 0:
            return 0.0
        return (prompt + completion) / 1000.0

    def will_exceed_budget(self, next_tokens: int) -> bool:
        """Return True if adding ``next_tokens`` would exceed the limit."""
        return self.tokens_used + next_tokens >= self.token_limit

    def enforce_limit(self, next_tokens: int) -> None:
        """Raise ``TokenLimitError`` if the token budget would be exceeded."""
        from exceptions import TokenLimitError

        if self.will_exceed_budget(next_tokens):
            logger.warning(
                "token_limit_exceeded",
                used=self.tokens_used,
                attempted=next_tokens,
                limit=self.token_limit,
            )
            raise TokenLimitError("Token budget exceeded")

    def record_llm_usage(self, tokens: int) -> None:
        """Record ``tokens`` consumed and update cost statistics."""
        self.tokens_used += tokens
        self.dollars_spent += tokens * self._cost_per_token


@dataclass
class UsagePattern:
    """Represents usage patterns for ML-based predictions."""
    tokens_per_request: List[int]
    request_times: List[float]
    cost_efficiency: List[float]
    quality_scores: List[float]


@dataclass
class CostPrediction:
    """Prediction result for request costs."""
    predicted_tokens: int
    predicted_cost: float
    confidence: float
    reasoning: str


@dataclass
class BudgetAlert:
    """Budget alert configuration and tracking."""
    threshold: float  # 0.9 for 90%, 0.95 for 95%, etc.
    triggered: bool = False
    trigger_time: Optional[float] = None
    callback: Optional[Callable[[float, float], None]] = None


class PredictiveBudgetManager(BudgetManager):
    """Advanced budget manager with ML-based cost prediction and optimization."""
    
    def __init__(
        self,
        model: str,
        token_limit: int,
        catalog: Optional[List[Dict[str, any]]] = None,
        budget_limit: Optional[float] = None,
        enable_alerts: bool = True
    ) -> None:
        super().__init__(model, token_limit, catalog)
        self.budget_limit = budget_limit or (token_limit * self._cost_per_token * 0.8)  # 80% safety margin
        self.enable_alerts = enable_alerts
        
        # ML prediction components
        self.usage_history = deque(maxlen=1000)  # Keep last 1000 requests
        self.usage_patterns = defaultdict(lambda: UsagePattern([], [], [], []))
        self.prediction_accuracy_history = deque(maxlen=100)
        
        # Real-time monitoring
        self.alerts = {
            0.90: BudgetAlert(0.90),
            0.95: BudgetAlert(0.95),
            1.00: BudgetAlert(1.00)
        }
        
        # Cost optimization tracking
        self.cost_efficiency_scores = deque(maxlen=100)
        self.optimization_recommendations = []
        
        logger.info(f"Initialized PredictiveBudgetManager for {model} with budget ${budget_limit:.4f}")
    
    def predict_request_cost(self, prompt: str, context_length: int = 0) -> CostPrediction:
        """Predict cost and token usage for a request using ML models."""
        try:
            # Simple ML-based prediction using historical patterns
            base_tokens = len(prompt.split()) * 1.3  # Rough tokenization estimate
            context_multiplier = 1.0 + (context_length / 10000)  # Context scaling
            
            if self.usage_history:
                # Use historical patterns for prediction
                recent_usage = list(self.usage_history)[-50:]  # Last 50 requests
                if recent_usage:
                    avg_tokens = sum(usage['tokens'] for usage in recent_usage) / len(recent_usage)
                    # Adjust based on prompt characteristics
                    predicted_tokens = int(base_tokens * context_multiplier * 0.3 + avg_tokens * 0.7)
                else:
                    predicted_tokens = int(base_tokens * context_multiplier)
            else:
                predicted_tokens = int(base_tokens * context_multiplier)
            
            predicted_cost = predicted_tokens * self._cost_per_token
            confidence = min(0.9, len(self.usage_history) / 100.0)  # Higher confidence with more data
            
            reasoning = f"Based on {len(self.usage_history)} historical requests"
            if context_length > 0:
                reasoning += f", context length {context_length}"
                
            return CostPrediction(predicted_tokens, predicted_cost, confidence, reasoning)
            
        except Exception as e:
            logger.warning(f"Cost prediction failed: {e}")
            # Fallback to simple estimation
            fallback_tokens = len(prompt.split()) * 2
            return CostPrediction(
                fallback_tokens,
                fallback_tokens * self._cost_per_token,
                0.5,
                "Fallback estimation"
            )
    
    def analyze_usage_patterns(self) -> Dict[str, any]:
        """Analyze usage patterns to identify cost optimization opportunities."""
        if not self.usage_history:
            return {"status": "insufficient_data"}
        
        recent_usage = list(self.usage_history)[-100:]
        
        # Calculate efficiency metrics
        avg_tokens = sum(usage['tokens'] for usage in recent_usage) / len(recent_usage)
        avg_cost = sum(usage['cost'] for usage in recent_usage) / len(recent_usage)
        avg_quality = sum(usage.get('quality', 0.5) for usage in recent_usage) / len(recent_usage)
        
        cost_per_quality = avg_cost / max(avg_quality, 0.1)
        
        # Identify patterns
        patterns = {
            "avg_tokens_per_request": avg_tokens,
            "avg_cost_per_request": avg_cost,
            "avg_quality_score": avg_quality,
            "cost_per_quality_unit": cost_per_quality,
            "total_requests": len(recent_usage),
            "efficiency_trend": self._calculate_efficiency_trend(),
        }
        
        # Generate recommendations
        recommendations = []
        if cost_per_quality > self._get_cost_efficiency_benchmark():
            recommendations.append("Consider using more cost-efficient models for lower-priority requests")
        
        if avg_tokens > self._get_token_efficiency_benchmark():
            recommendations.append("Optimize prompt length to reduce token usage")
            
        patterns["recommendations"] = recommendations
        self.optimization_recommendations = recommendations
        
        return patterns
    
    def _calculate_efficiency_trend(self) -> str:
        """Calculate the trend in cost efficiency over time."""
        if len(self.cost_efficiency_scores) < 10:
            return "insufficient_data"
        
        recent_scores = list(self.cost_efficiency_scores)[-10:]
        older_scores = list(self.cost_efficiency_scores)[-20:-10] if len(self.cost_efficiency_scores) >= 20 else []
        
        if not older_scores:
            return "insufficient_data"
        
        recent_avg = sum(recent_scores) / len(recent_scores)
        older_avg = sum(older_scores) / len(older_scores)
        
        if recent_avg > older_avg * 1.05:
            return "improving"
        elif recent_avg < older_avg * 0.95:
            return "declining"
        else:
            return "stable"
    
    def _get_cost_efficiency_benchmark(self) -> float:
        """Get cost efficiency benchmark for comparison."""
        return 0.001  # $0.001 per quality unit as benchmark
    
    def _get_token_efficiency_benchmark(self) -> float:
        """Get token efficiency benchmark for comparison."""
        return 1000  # 1000 tokens per request as benchmark
    
    def check_budget_alerts(self) -> List[BudgetAlert]:
        """Check and trigger budget alerts if thresholds are exceeded."""
        triggered_alerts = []
        
        if not self.enable_alerts:
            return triggered_alerts
        
        current_usage_ratio = self.dollars_spent / max(self.budget_limit, 0.001)
        
        for threshold, alert in self.alerts.items():
            if current_usage_ratio >= threshold and not alert.triggered:
                alert.triggered = True
                alert.trigger_time = time.time()
                triggered_alerts.append(alert)
                
                logger.warning(f"Budget alert triggered: {threshold*100}% threshold exceeded")
                
                if alert.callback:
                    try:
                        alert.callback(current_usage_ratio, self.dollars_spent)
                    except Exception as e:
                        logger.error(f"Alert callback failed: {e}")
        
        return triggered_alerts
    
    def record_usage(self, tokens: int, quality_score: Optional[float] = None) -> None:
        """Enhanced usage recording with quality tracking."""
        super().record_usage(tokens)
        
        # Record detailed usage for ML analysis
        usage_record = {
            'timestamp': time.time(),
            'tokens': tokens,
            'cost': tokens * self._cost_per_token,
            'quality': quality_score or 0.5,
            'model': self.model
        }
        
        self.usage_history.append(usage_record)
        
        # Calculate and store cost efficiency
        if quality_score:
            cost_efficiency = (tokens * self._cost_per_token) / max(quality_score, 0.1)
            self.cost_efficiency_scores.append(cost_efficiency)
        
        # Check alerts
        self.check_budget_alerts()
        
        # Update prediction accuracy if we have predictions
        self._update_prediction_accuracy(tokens)
    
    def _update_prediction_accuracy(self, actual_tokens: int) -> None:
        """Update prediction accuracy metrics."""
        # This would typically store the last prediction and compare
        # For now, we'll implement a simple version
        if hasattr(self, '_last_prediction') and self._last_prediction:
            error = abs(actual_tokens - self._last_prediction.predicted_tokens) / max(actual_tokens, 1)
            accuracy = max(0, 1 - error)
            self.prediction_accuracy_history.append(accuracy)
            
    def get_optimization_insights(self) -> Dict[str, any]:
        """Get comprehensive cost optimization insights."""
        patterns = self.analyze_usage_patterns()
        
        current_accuracy = 0.5
        if self.prediction_accuracy_history:
            current_accuracy = sum(self.prediction_accuracy_history) / len(self.prediction_accuracy_history)
        
        insights = {
            "usage_patterns": patterns,
            "prediction_accuracy": current_accuracy,
            "budget_utilization": self.dollars_spent / max(self.budget_limit, 0.001),
            "cost_savings_potential": self._estimate_savings_potential(),
            "alert_status": {
                threshold: alert.triggered
                for threshold, alert in self.alerts.items()
            },
            "recommendations": self.optimization_recommendations
        }
        
        return insights
    
    def _estimate_savings_potential(self) -> float:
        """Estimate potential cost savings from optimizations."""
        if not self.usage_history:
            return 0.0
        
        # Simple heuristic: 20-40% savings potential based on efficiency
        recent_usage = list(self.usage_history)[-50:]
        if not recent_usage:
            return 0.0
        
        avg_efficiency = sum(usage.get('quality', 0.5) / max(usage['cost'], 0.001) for usage in recent_usage) / len(recent_usage)
        benchmark_efficiency = 500  # tokens per dollar benchmark
        
        if avg_efficiency < benchmark_efficiency * 0.8:
            return 0.4  # 40% savings potential
        elif avg_efficiency < benchmark_efficiency * 0.9:
            return 0.2  # 20% savings potential
        else:
            return 0.1  # 10% savings potential

    # Backwards compatibility
    record_usage = record_llm_usage

    @property
    def cost_per_token(self) -> float:
        """Return cost per token for the configured model."""
        return self._cost_per_token

    @property
    def remaining_tokens(self) -> int:
        """Return the number of tokens remaining in the budget."""
        return max(self.token_limit - self.tokens_used, 0)
