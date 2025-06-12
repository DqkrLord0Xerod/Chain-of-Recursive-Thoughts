"""Tests for monitoring and telemetry."""

import asyncio
import time
from unittest.mock import patch

import pytest

from monitoring.telemetry import (
    initialize_telemetry,
    get_tracer,
    get_metrics,
    trace_span,
    async_trace_span,
    trace_method,
    record_thinking_metrics,
    record_cache_metrics,
    TelemetryMixin,
)
from monitoring.metrics_v2 import (
    ThinkingMetrics,
    MetricsAnalyzer,
)


class TestTelemetry:
    
    @patch('monitoring.telemetry.start_http_server')
    def test_initialize_telemetry(self, mock_http_server):
        """Test telemetry initialization."""
        initialize_telemetry(
            service_name="test",
            enable_console_export=False,
            enable_prometheus=True,
            prometheus_port=8888,
        )
        
        # Should be able to get tracer and metrics
        tracer = get_tracer()
        metrics = get_metrics()
        
        assert tracer is not None
        assert metrics is not None
        mock_http_server.assert_called_once_with(8888)
        
    def test_trace_span(self):
        """Test trace span context manager."""
        initialize_telemetry(enable_prometheus=False)
        
        with trace_span("test_span", attributes={"key": "value"}) as span:
            assert span is not None
            span.set_attribute("additional", "attribute")
            
    def test_trace_span_exception(self):
        """Test trace span records exceptions."""
        initialize_telemetry(enable_prometheus=False)
        
        with pytest.raises(ValueError):
            with trace_span("test_span"):
                raise ValueError("Test error")
                
    @pytest.mark.asyncio
    async def test_async_trace_span(self):
        """Test async trace span context manager."""
        initialize_telemetry(enable_prometheus=False)
        
        async with async_trace_span("async_test_span") as span:
            assert span is not None
            await asyncio.sleep(0.01)
            
    def test_trace_method_decorator_sync(self):
        """Test trace method decorator on sync functions."""
        initialize_telemetry(enable_prometheus=False)
        
        @trace_method("custom_name")
        def test_function(x, y):
            return x + y
            
        result = test_function(1, 2)
        assert result == 3
        
    @pytest.mark.asyncio
    async def test_trace_method_decorator_async(self):
        """Test trace method decorator on async functions."""
        initialize_telemetry(enable_prometheus=False)
        
        @trace_method()
        async def async_test_function(x, y):
            await asyncio.sleep(0.01)
            return x + y
            
        result = await async_test_function(1, 2)
        assert result == 3
        
    def test_record_thinking_metrics(self):
        """Test recording thinking metrics."""
        initialize_telemetry(enable_prometheus=False)

        record_thinking_metrics(
            rounds=3,
            duration=5.2,
            convergence_reason="quality_threshold",
            initial_quality=0.6,
            final_quality=0.9,
            total_tokens=1500,
            prompt_tokens=600,
            completion_tokens=900,
        )

        # Metrics should be recorded without error

    def test_record_token_breakdown(self):
        """Ensure prompt and completion tokens are recorded."""
        initialize_telemetry(enable_prometheus=False)
        metrics = get_metrics()

        with patch.object(metrics.prompt_tokens, "record") as mock_prompt, \
                patch.object(metrics.completion_tokens, "record") as mock_comp:
            record_thinking_metrics(
                rounds=1,
                duration=1.0,
                convergence_reason="test",
                initial_quality=0.1,
                final_quality=0.2,
                total_tokens=100,
                prompt_tokens=40,
                completion_tokens=60,
            )

            mock_prompt.assert_called_once_with(40)
            mock_comp.assert_called_once_with(60)
        
    def test_record_cache_metrics(self):
        """Test recording cache metrics."""
        initialize_telemetry(enable_prometheus=False)
        
        record_cache_metrics(hit=True, cache_type="memory")
        record_cache_metrics(hit=False, cache_type="disk")
        
        # Metrics should be recorded without error
        
    def test_telemetry_mixin(self):
        """Test TelemetryMixin functionality."""
        initialize_telemetry(enable_prometheus=False)
        
        class TestClass(TelemetryMixin):
            def __init__(self):
                self.model = "test-model"
                self.session_id = "test-session"
                
        obj = TestClass()
        
        # Test creating span attributes
        attrs = obj.create_span_attributes(custom="value")
        assert attrs["cort.class"] == "TestClass"
        assert attrs["cort.model"] == "test-model"
        assert attrs["cort.session_id"] == "test-session"
        assert attrs["custom"] == "value"
        
        # Test recording metrics
        obj.record_metric("request_counter", 1.0, labels={"status": "success"})


class TestThinkingMetrics:
    
    def test_thinking_metrics_properties(self):
        """Test ThinkingMetrics calculated properties."""
        metrics = ThinkingMetrics(
            session_id="test",
            start_time=time.time() - 10,
            end_time=time.time(),
            rounds_completed=3,
            quality_scores=[0.5, 0.7, 0.9],
            token_usage_per_round=[500, 600, 700],
        )
        
        assert metrics.duration == pytest.approx(10, rel=0.1)
        assert metrics.total_tokens == 1800
        assert metrics.quality_improvement == pytest.approx(0.4)
        assert metrics.efficiency_score > 0
        assert 0 <= metrics.convergence_speed <= 1
        
    def test_thinking_metrics_no_improvement(self):
        """Test metrics when quality doesn't improve."""
        metrics = ThinkingMetrics(
            session_id="test",
            start_time=time.time(),
            quality_scores=[0.8],
            token_usage_per_round=[1000],
        )

        assert metrics.quality_improvement == 0
        assert metrics.efficiency_score == 0
        assert metrics.convergence_speed == 1.0

    def test_prompt_completion_tokens(self):
        """Validate total_tokens with prompt/completion fields."""
        metrics = ThinkingMetrics(
            session_id="test",
            start_time=time.time(),
            prompt_tokens=50,
            completion_tokens=75,
        )

        assert metrics.total_tokens == 125


class TestMetricsAnalyzer:
    
    @pytest.fixture
    def analyzer(self):
        return MetricsAnalyzer(window_size=100, anomaly_threshold=2.0)
        
    @pytest.fixture
    def sample_metrics(self):
        return ThinkingMetrics(
            session_id="test",
            start_time=time.time() - 5,
            end_time=time.time(),
            rounds_completed=3,
            convergence_reason="quality_threshold",
            quality_scores=[0.5, 0.7, 0.85],
            round_durations=[1.0, 1.5, 2.0],
            token_usage_per_round=[500, 600, 700],
            cache_hits=5,
            cache_misses=2,
        )
        
    def test_record_session(self, analyzer, sample_metrics):
        """Test recording a session."""
        result = analyzer.record_session(sample_metrics)
        
        assert "insights" in result
        assert "anomalies" in result
        assert "warnings" in result
        
        # Should update internal state
        assert len(analyzer.sessions) == 1
        assert len(analyzer.response_times) == 1
        assert analyzer.convergence_reasons["quality_threshold"] == 1
        
    def test_anomaly_detection(self, analyzer):
        """Test anomaly detection."""
        # Add normal sessions
        for i in range(20):
            metrics = ThinkingMetrics(
                session_id=f"normal_{i}",
                start_time=time.time() - 5,
                end_time=time.time(),
                rounds_completed=3,
                quality_scores=[0.5, 0.7, 0.8],
                token_usage_per_round=[500, 500, 500],
            )
            analyzer.record_session(metrics)
            
        # Add anomalous session
        anomaly_metrics = ThinkingMetrics(
            session_id="anomaly",
            start_time=time.time() - 30,  # Much longer duration
            end_time=time.time(),
            rounds_completed=3,
            quality_scores=[0.8, 0.7, 0.6],  # Quality degradation
            token_usage_per_round=[5000, 5000, 5000],  # High token usage
        )
        
        result = analyzer.record_session(anomaly_metrics)
        
        assert len(result["anomalies"]) > 0
        anomaly_types = [a["type"] for a in result["anomalies"]]
        assert "duration_anomaly" in anomaly_types
        assert "quality_degradation" in anomaly_types
        
    def test_insights_generation(self, analyzer, sample_metrics):
        """Test insight generation."""
        # Record multiple sessions
        for i in range(10):
            analyzer.record_session(sample_metrics)
            
        result = analyzer.record_session(sample_metrics)
        insights = result["insights"]
        
        assert "efficiency_score" in insights
        assert "convergence_speed" in insights
        assert "cache_effectiveness" in insights
        assert insights["cache_effectiveness"] == pytest.approx(0.714, rel=0.01)
        
    def test_warnings_generation(self, analyzer):
        """Test warning generation."""
        # High token usage
        metrics = ThinkingMetrics(
            session_id="high_tokens",
            start_time=time.time(),
            quality_scores=[0.5, 0.6],
            token_usage_per_round=[6000, 6000],
            cache_hits=1,
            cache_misses=20,
        )
        
        result = analyzer.record_session(metrics)
        warnings = result["warnings"]
        
        assert any("High token usage" in w for w in warnings)
        assert any("Poor cache hit rate" in w for w in warnings)
        
    def test_summary_stats(self, analyzer, sample_metrics):
        """Test summary statistics."""
        # Record sessions
        for i in range(50):
            analyzer.record_session(sample_metrics)
            
        stats = analyzer.get_summary_stats()
        
        assert stats["total_sessions"] == 50
        assert "average_duration" in stats
        assert "average_quality" in stats
        assert "convergence_distribution" in stats
        assert stats["convergence_distribution"]["quality_threshold"] == 50
        
    def test_hourly_patterns(self, analyzer):
        """Test hourly pattern analysis."""
        # Create sessions at different hours
        for hour in [9, 9, 10, 10, 10, 14, 14]:
            metrics = ThinkingMetrics(
                session_id=f"hour_{hour}",
                start_time=time.time() - (24 - hour) * 3600,  # Simulate different hours
                end_time=time.time() - (24 - hour) * 3600 + 5,
                quality_scores=[0.8],
                token_usage_per_round=[1000],
            )
            analyzer._update_hourly_stats(hour, metrics)
            
        stats = analyzer.get_summary_stats()
        patterns = stats["hourly_patterns"]
        
        assert "09:00" in patterns
        assert patterns["10:00"]["requests"] == 3
        
    def test_provider_stats(self, analyzer):
        """Test provider statistics tracking."""
        # Record latencies
        for i in range(10):
            analyzer.record_provider_latency("provider_1", 0.5 + i * 0.1)
            analyzer.record_provider_latency("provider_2", 1.0 + i * 0.2)
            
        # Record errors
        analyzer.record_provider_error("provider_1", "timeout")
        analyzer.record_provider_error("provider_2", "rate_limit")
        analyzer.record_provider_error("provider_2", "api_error")
        
        stats = analyzer.get_provider_stats()
        
        assert "provider_1" in stats
        assert "provider_2" in stats
        assert stats["provider_1"]["avg_latency"] < stats["provider_2"]["avg_latency"]
        assert stats["provider_1"]["error_count"] == 1
        assert stats["provider_2"]["error_count"] == 2
        assert stats["provider_1"]["reliability"] > stats["provider_2"]["reliability"]
        
    def test_recommendations(self, analyzer):
        """Test system recommendations."""
        # Create problematic sessions
        for i in range(50):
            metrics = ThinkingMetrics(
                session_id=f"slow_{i}",
                start_time=time.time() - 15,  # 15 second duration
                end_time=time.time(),
                quality_scores=[0.5, 0.6, 0.65, 0.68, 0.69],  # Plateau
                token_usage_per_round=[2000, 2000, 2000, 2000, 2000],
                convergence_reason="quality_plateau",
                cache_hits=1,
                cache_misses=10,
            )
            analyzer.record_session(metrics)
            
        recommendations = analyzer.get_recommendations()
        
        assert len(recommendations) > 0
        
        # Check for expected recommendations
        rec_types = [r["type"] for r in recommendations]
        assert "performance" in rec_types  # High response time
        assert "cost" in rec_types  # High token usage
        
        # Check specific recommendations
        perf_recs = [r for r in recommendations if r["type"] == "performance"]
        assert any("response time" in r["title"] for r in perf_recs)
        assert any("cache hit rate" in r["title"] for r in perf_recs)

        cost_recs = [r for r in recommendations if r["type"] == "cost"]
        assert any("token usage" in r["title"] for r in cost_recs)


def test_metrics_summary_endpoint(monkeypatch):
    """Ensure metrics summary endpoint returns data from analyzer."""
    from starlette.testclient import TestClient
    import recthink_web_v2

    class DummyAnalyzer:
        def get_summary_stats(self):
            return {"total_sessions": 2}

        anomalies = [{"type": "test"}]

    monkeypatch.setattr(recthink_web_v2, "metrics_analyzer", DummyAnalyzer())

    client = TestClient(recthink_web_v2.app)
    resp = client.get("/metrics/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert data["summary"]["total_sessions"] == 2
    assert data["anomalies"][0]["type"] == "test"
