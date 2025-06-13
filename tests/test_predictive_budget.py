import pytest
import time
from unittest.mock import Mock, patch
from typing import Dict, List

from core.budget import BudgetManager, PredictiveBudgetManager, CostPrediction, BudgetAlert


@pytest.fixture
def sample_catalog():
    """Sample model catalog for testing."""
    return [
        {
            "id": "test-model-cheap",
            "pricing": {"prompt": 0.001, "completion": 0.002}
        },
        {
            "id": "test-model-expensive", 
            "pricing": {"prompt": 0.01, "completion": 0.02}
        }
    ]


@pytest.fixture
def predictive_budget_manager(sample_catalog):
    """Create PredictiveBudgetManager for testing."""
    return PredictiveBudgetManager(
        model="test-model-cheap",
        token_limit=1000,
        catalog=sample_catalog,
        budget_limit=1.0,
        enable_alerts=True
    )


def test_predictive_budget_manager_initialization(sample_catalog):
    """Test PredictiveBudgetManager initializes correctly."""
    manager = PredictiveBudgetManager(
        model="test-model-cheap",
        token_limit=1000,
        catalog=sample_catalog,
        budget_limit=0.5
    )
    
    assert manager.model == "test-model-cheap"
    assert manager.token_limit == 1000
    assert manager.budget_limit == 0.5
    assert manager.enable_alerts is True
    assert len(manager.alerts) == 3  # 90%, 95%, 100%
    assert 0.90 in manager.alerts
    assert 0.95 in manager.alerts
    assert 1.00 in manager.alerts


def test_predict_request_cost_basic(predictive_budget_manager):
    """Test basic cost prediction functionality."""
    prediction = predictive_budget_manager.predict_request_cost(
        prompt="Hello world test prompt",
        context_length=0
    )
    
    assert isinstance(prediction, CostPrediction)
    assert prediction.predicted_tokens > 0
    assert prediction.predicted_cost > 0
    assert 0 <= prediction.confidence <= 1
    assert len(prediction.reasoning) > 0


def test_predict_request_cost_with_history(predictive_budget_manager):
    """Test cost prediction with usage history."""
    # Add some usage history
    for i in range(10):
        predictive_budget_manager.usage_history.append({
            'timestamp': time.time(),
            'tokens': 100 + i,
            'cost': 0.01 + i * 0.001,
            'quality': 0.8,
            'model': 'test-model-cheap'
        })
    
    prediction = predictive_budget_manager.predict_request_cost(
        prompt="Test prompt with history",
        context_length=500
    )
    
    assert prediction.predicted_tokens > 0
    assert prediction.predicted_cost > 0
    assert prediction.confidence > 0.05  # Should have some confidence with history


def test_analyze_usage_patterns_no_data(predictive_budget_manager):
    """Test usage pattern analysis with no data."""
    patterns = predictive_budget_manager.analyze_usage_patterns()
    assert patterns["status"] == "insufficient_data"


def test_analyze_usage_patterns_with_data(predictive_budget_manager):
    """Test usage pattern analysis with data."""
    # Add usage history
    for i in range(20):
        predictive_budget_manager.usage_history.append({
            'timestamp': time.time(),
            'tokens': 100 + i,
            'cost': 0.01 + i * 0.001,
            'quality': 0.7 + i * 0.01,
            'model': 'test-model-cheap'
        })
    
    patterns = predictive_budget_manager.analyze_usage_patterns()
    
    assert "avg_tokens_per_request" in patterns
    assert "avg_cost_per_request" in patterns
    assert "avg_quality_score" in patterns
    assert "cost_per_quality_unit" in patterns
    assert "recommendations" in patterns
    assert patterns["total_requests"] == 20


def test_budget_alerts_trigger(predictive_budget_manager):
    """Test budget alerts are triggered correctly."""
    # Spend enough to trigger 90% alert
    predictive_budget_manager.record_usage(300, quality_score=0.8)  # Should cost ~0.9
    
    alerts = predictive_budget_manager.check_budget_alerts()
    
    # Should trigger 90% alert
    assert len(alerts) >= 1
    triggered_alert = alerts[0]
    assert triggered_alert.threshold == 0.90
    assert triggered_alert.triggered is True


def test_record_usage_enhanced(predictive_budget_manager):
    """Test enhanced usage recording."""
    initial_tokens = predictive_budget_manager.tokens_used
    initial_cost = predictive_budget_manager.dollars_spent
    
    predictive_budget_manager.record_usage(100, quality_score=0.8)
    
    # Check basic recording
    assert predictive_budget_manager.tokens_used == initial_tokens + 100
    assert predictive_budget_manager.dollars_spent > initial_cost
    
    # Check enhanced features
    assert len(predictive_budget_manager.usage_history) == 1
    assert len(predictive_budget_manager.cost_efficiency_scores) == 1
    
    usage_record = predictive_budget_manager.usage_history[0]
    assert usage_record['tokens'] == 100
    assert usage_record['quality'] == 0.8
    assert usage_record['model'] == 'test-model-cheap'


def test_get_optimization_insights(predictive_budget_manager):
    """Test optimization insights generation."""
    # Add some usage data
    for i in range(15):
        predictive_budget_manager.record_usage(100, quality_score=0.7 + i * 0.01)
    
    insights = predictive_budget_manager.get_optimization_insights()
    
    assert "usage_patterns" in insights
    assert "prediction_accuracy" in insights
    assert "budget_utilization" in insights
    assert "cost_savings_potential" in insights
    assert "alert_status" in insights
    assert "recommendations" in insights
    
    # Check alert status structure
    alert_status = insights["alert_status"]
    assert 0.90 in alert_status
    assert 0.95 in alert_status
    assert 1.00 in alert_status


def test_backward_compatibility():
    """Test that PredictiveBudgetManager maintains backward compatibility."""
    catalog = [{"id": "test-model", "pricing": {"prompt": 0.001, "completion": 0.002}}]
    
    # Should work exactly like original BudgetManager for basic operations
    predictive_manager = PredictiveBudgetManager("test-model", 1000, catalog)
    basic_manager = BudgetManager("test-model", 1000, catalog)
    
    # Test basic operations
    assert predictive_manager.will_exceed_budget(500) == basic_manager.will_exceed_budget(500)
    
    predictive_manager.record_usage(100)
    basic_manager.record_usage(100)
    
    assert predictive_manager.tokens_used == basic_manager.tokens_used
    assert abs(predictive_manager.dollars_spent - basic_manager.dollars_spent) < 0.0001


def test_cost_prediction_fallback(predictive_budget_manager):
    """Test cost prediction fallback mechanism."""
    # Mock an exception in prediction
    with patch.object(predictive_budget_manager, 'usage_history', side_effect=Exception("Test error")):
        prediction = predictive_budget_manager.predict_request_cost("test prompt")
        
        # Should still return a valid prediction
        assert isinstance(prediction, CostPrediction)
        assert prediction.predicted_tokens > 0
        assert prediction.confidence == 0.5  # Fallback confidence
        assert prediction.reasoning == "Fallback estimation"


def test_efficiency_trend_calculation(predictive_budget_manager):
    """Test efficiency trend calculation."""
    # Add cost efficiency scores to simulate trend
    for i in range(25):
        score = 0.01 - i * 0.0002  # Improving trend (lower is better)
        predictive_budget_manager.cost_efficiency_scores.append(score)
    
    trend = predictive_budget_manager._calculate_efficiency_trend()
    assert trend == "improving"
    
    # Test declining trend
    predictive_budget_manager.cost_efficiency_scores.clear()
    for i in range(25):
        score = 0.005 + i * 0.0002  # Declining trend
        predictive_budget_manager.cost_efficiency_scores.append(score)
    
    trend = predictive_budget_manager._calculate_efficiency_trend()
    assert trend == "declining"


def test_alert_callback_execution(predictive_budget_manager):
    """Test that alert callbacks are executed when triggered."""
    callback_called = False
    callback_data = None
    
    def test_callback(usage_ratio, dollars_spent):
        nonlocal callback_called, callback_data
        callback_called = True
        callback_data = (usage_ratio, dollars_spent)
    
    # Set callback for 90% alert
    predictive_budget_manager.alerts[0.90].callback = test_callback
    
    # Trigger alert by spending enough
    predictive_budget_manager.record_usage(300, quality_score=0.8)
    
    assert callback_called
    assert callback_data is not None
    assert callback_data[0] >= 0.9  # Usage ratio
    assert callback_data[1] > 0  # Dollars spent