"""Enhanced MetricsManager with real-time cost monitoring."""

from __future__ import annotations

import logging
import time
from typing import Optional, Dict, List, Callable, Any
from dataclasses import dataclass, field
from collections import deque, defaultdict

from monitoring.metrics import MetricsRecorder

logger = logging.getLogger(__name__)


@dataclass
class CostAlert:
    """Real-time cost alert configuration."""
    threshold: float
    alert_type: str  # 'budget_percentage', 'cost_per_quality', 'absolute_cost'
    callback: Optional[Callable[[Dict[str, Any]], None]] = None
    triggered: bool = False
    trigger_count: int = 0
    last_triggered: Optional[float] = None


@dataclass
class CostMetrics:
    """Real-time cost and quality metrics."""
    timestamp: float
    cost: float
    quality_score: float
    token_usage: int
    model_used: str
    processing_time: float
    cost_per_quality_unit: float = field(init=False)
    
    def __post_init__(self):
        self.cost_per_quality_unit = self.cost / max(self.quality_score, 0.1)


class RealTimeCostMonitor:
    """Real-time cost monitoring with configurable alerts."""
    
    def __init__(self, budget_limit: float = 100.0):
        self.budget_limit = budget_limit
        self.current_cost = 0.0
        self.cost_history = deque(maxlen=1000)
        self.quality_history = deque(maxlen=1000)
        
        # Alert configuration
        self.alerts = {
            'budget_90': CostAlert(0.90, 'budget_percentage'),
            'budget_95': CostAlert(0.95, 'budget_percentage'),
            'budget_100': CostAlert(1.00, 'budget_percentage'),
        }
        
        # Real-time metrics
        self.metrics_buffer = deque(maxlen=100)
        self.cost_efficiency_tracker = deque(maxlen=50)
        
        # Performance tracking
        self.model_performance = defaultdict(lambda: {
            'total_cost': 0.0,
            'total_requests': 0,
            'avg_quality': 0.0,
            'cost_efficiency': 0.0
        })
        
        logger.info(f"Initialized RealTimeCostMonitor with ${budget_limit:.2f} budget")
    
    def add_alert(self, alert_id: str, threshold: float, alert_type: str,
                  callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> None:
        """Add custom alert configuration."""
        self.alerts[alert_id] = CostAlert(threshold, alert_type, callback)
        logger.info(f"Added alert '{alert_id}': {alert_type} threshold {threshold}")
    
    def record_cost_metrics(self, cost: float, quality_score: float, token_usage: int,
                          model_used: str, processing_time: float) -> None:
        """Record real-time cost and quality metrics."""
        
        metrics = CostMetrics(
            timestamp=time.time(),
            cost=cost,
            quality_score=quality_score,
            token_usage=token_usage,
            model_used=model_used,
            processing_time=processing_time
        )
        
        self.current_cost += cost
        self.cost_history.append(cost)
        self.quality_history.append(quality_score)
        self.metrics_buffer.append(metrics)
        
        # Update model performance tracking
        model_perf = self.model_performance[model_used]
        model_perf['total_cost'] += cost
        model_perf['total_requests'] += 1
        
        # Update running averages
        total_requests = model_perf['total_requests']
        model_perf['avg_quality'] = (
            (model_perf['avg_quality'] * (total_requests - 1) + quality_score) / total_requests
        )
        model_perf['cost_efficiency'] = model_perf['avg_quality'] / max(model_perf['total_cost'] / total_requests, 0.001)
        
        # Update cost efficiency tracker
        if quality_score > 0:
            cost_efficiency = cost / quality_score
            self.cost_efficiency_tracker.append(cost_efficiency)
        
        # Check alerts
        self._check_alerts(metrics)
        
        logger.debug(f"Recorded metrics: ${cost:.4f}, quality={quality_score:.3f}, model={model_used}")
    
    def _check_alerts(self, metrics: CostMetrics) -> None:
        """Check and trigger alerts based on current metrics."""
        
        current_time = time.time()
        
        for alert_id, alert in self.alerts.items():
            should_trigger = False
            alert_data = {}
            
            if alert.alert_type == 'budget_percentage':
                usage_ratio = self.current_cost / self.budget_limit
                if usage_ratio >= alert.threshold:
                    should_trigger = True
                    alert_data = {
                        'alert_type': 'budget_percentage',
                        'threshold': alert.threshold,
                        'current_usage': usage_ratio,
                        'current_cost': self.current_cost,
                        'budget_limit': self.budget_limit
                    }
            
            elif alert.alert_type == 'cost_per_quality':
                if metrics.cost_per_quality_unit >= alert.threshold:
                    should_trigger = True
                    alert_data = {
                        'alert_type': 'cost_per_quality',
                        'threshold': alert.threshold,
                        'current_ratio': metrics.cost_per_quality_unit,
                        'cost': metrics.cost,
                        'quality': metrics.quality_score
                    }
            
            elif alert.alert_type == 'absolute_cost':
                if self.current_cost >= alert.threshold:
                    should_trigger = True
                    alert_data = {
                        'alert_type': 'absolute_cost',
                        'threshold': alert.threshold,
                        'current_cost': self.current_cost
                    }
            
            # Trigger alert if conditions met and not recently triggered
            if should_trigger and (not alert.last_triggered or
                                 current_time - alert.last_triggered > 300):  # 5 min cooldown
                
                alert.triggered = True
                alert.trigger_count += 1
                alert.last_triggered = current_time
                
                alert_data.update({
                    'alert_id': alert_id,
                    'trigger_count': alert.trigger_count,
                    'timestamp': current_time
                })
                
                logger.warning(f"Cost alert triggered: {alert_id} - {alert_data}")
                
                if alert.callback:
                    try:
                        alert.callback(alert_data)
                    except Exception as e:
                        logger.error(f"Alert callback failed for {alert_id}: {e}")
    
    def get_real_time_stats(self) -> Dict[str, Any]:
        """Get current real-time cost and performance statistics."""
        
        if not self.metrics_buffer:
            return {"status": "no_data"}
        
        recent_metrics = list(self.metrics_buffer)[-10:]  # Last 10 requests
        
        # Calculate current stats
        avg_cost = sum(m.cost for m in recent_metrics) / len(recent_metrics)
        avg_quality = sum(m.quality_score for m in recent_metrics) / len(recent_metrics)
        avg_cost_per_quality = sum(m.cost_per_quality_unit for m in recent_metrics) / len(recent_metrics)
        
        # Budget utilization
        budget_utilization = self.current_cost / self.budget_limit
        
        # Cost efficiency trend
        efficiency_trend = "stable"
        if len(self.cost_efficiency_tracker) >= 10:
            recent_efficiency = list(self.cost_efficiency_tracker)[-5:]
            older_efficiency = list(self.cost_efficiency_tracker)[-10:-5]
            
            recent_avg = sum(recent_efficiency) / len(recent_efficiency)
            older_avg = sum(older_efficiency) / len(older_efficiency)
            
            if recent_avg < older_avg * 0.9:  # 10% improvement (lower is better)
                efficiency_trend = "improving"
            elif recent_avg > older_avg * 1.1:  # 10% degradation
                efficiency_trend = "declining"
        
        return {
            "timestamp": time.time(),
            "current_cost": self.current_cost,
            "budget_limit": self.budget_limit,
            "budget_utilization": budget_utilization,
            "avg_cost_per_request": avg_cost,
            "avg_quality_score": avg_quality,
            "avg_cost_per_quality_unit": avg_cost_per_quality,
            "efficiency_trend": efficiency_trend,
            "total_requests": len(self.metrics_buffer),
            "active_alerts": [
                alert_id for alert_id, alert in self.alerts.items()
                if alert.triggered
            ],
            "model_performance": dict(self.model_performance)
        }
    
    def get_cost_optimization_recommendations(self) -> List[str]:
        """Generate cost optimization recommendations based on current metrics."""
        recommendations = []
        
        if not self.metrics_buffer:
            return ["Insufficient data for recommendations"]
        
        recent_metrics = list(self.metrics_buffer)[-20:]
        
        # Analyze cost efficiency
        avg_cost_per_quality = sum(m.cost_per_quality_unit for m in recent_metrics) / len(recent_metrics)
        
        if avg_cost_per_quality > 0.01:  # Benchmark threshold
            recommendations.append("Consider using more cost-efficient models for routine tasks")
        
        # Analyze model usage patterns
        model_usage = defaultdict(int)
        model_costs = defaultdict(float)
        
        for metrics in recent_metrics:
            model_usage[metrics.model_used] += 1
            model_costs[metrics.model_used] += metrics.cost
        
        # Find expensive models with low usage
        for model, usage in model_usage.items():
            if usage < len(recent_metrics) * 0.1 and model_costs[model] > avg_cost_per_quality:
                recommendations.append(f"Model '{model}' shows low usage but high cost - consider alternatives")
        
        # Budget utilization recommendations
        budget_utilization = self.current_cost / self.budget_limit
        if budget_utilization > 0.8:
            recommendations.append("Budget utilization is high - consider implementing stricter cost controls")
        
        return recommendations or ["Current usage patterns appear optimized"]


class EnhancedMetricsManager(MetricsManager):
    """Enhanced MetricsManager with real-time cost monitoring."""

    def __init__(self, recorder: Optional[MetricsRecorder] = None,
                 budget_limit: float = 100.0,
                 enable_cost_monitoring: bool = True) -> None:
        super().__init__(recorder)
        self.enable_cost_monitoring = enable_cost_monitoring
        
        if enable_cost_monitoring:
            self.cost_monitor = RealTimeCostMonitor(budget_limit)
        else:
            self.cost_monitor = None
        
        logger.info(f"Enhanced MetricsManager initialized with cost monitoring: {enable_cost_monitoring}")

    def record(
        self,
        *,
        processing_time: float,
        token_usage: int,
        num_rounds: int,
        convergence_reason: str,
        cost: Optional[float] = None,
        quality_score: Optional[float] = None,
        model_used: Optional[str] = None,
    ) -> None:
        """Enhanced recording with cost and quality metrics."""
        
        # Record to original recorder
        if self.recorder:
            self.recorder.record_run(
                processing_time=processing_time,
                token_usage=token_usage,
                num_rounds=num_rounds,
                convergence_reason=convergence_reason,
            )
        
        # Record cost metrics if monitoring enabled
        if self.cost_monitor and cost is not None:
            quality = quality_score or 0.5  # Default quality if not provided
            model = model_used or "unknown"
            
            self.cost_monitor.record_cost_metrics(
                cost=cost,
                quality_score=quality,
                token_usage=token_usage,
                model_used=model,
                processing_time=processing_time
            )
    
    def add_cost_alert(self, alert_id: str, threshold: float, alert_type: str,
                      callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> None:
        """Add custom cost alert."""
        if self.cost_monitor:
            self.cost_monitor.add_alert(alert_id, threshold, alert_type, callback)
        else:
            logger.warning("Cost monitoring not enabled, cannot add alert")
    
    def get_cost_insights(self) -> Dict[str, Any]:
        """Get comprehensive cost insights and recommendations."""
        if not self.cost_monitor:
            return {"status": "cost_monitoring_disabled"}
        
        stats = self.cost_monitor.get_real_time_stats()
        recommendations = self.cost_monitor.get_cost_optimization_recommendations()
        
        return {
            "real_time_stats": stats,
            "optimization_recommendations": recommendations,
            "monitoring_status": "active"
        }
    
    def reset_cost_tracking(self) -> None:
        """Reset cost tracking (useful for new budget periods)."""
        if self.cost_monitor:
            self.cost_monitor.current_cost = 0.0
            self.cost_monitor.cost_history.clear()
            self.cost_monitor.quality_history.clear()
            
            # Reset alert states
            for alert in self.cost_monitor.alerts.values():
                alert.triggered = False
                alert.last_triggered = None
            
            logger.info("Cost tracking reset")


# Maintain backward compatibility
MetricsManager = EnhancedMetricsManager
