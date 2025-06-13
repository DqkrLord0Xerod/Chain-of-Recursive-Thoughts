"""OpenTelemetry integration for distributed tracing and metrics."""

from __future__ import annotations

import asyncio
import functools
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Callable, Dict, Optional, TypeVar
import uuid
import logging

from opentelemetry import trace, metrics
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from fastapi import FastAPI
from opentelemetry.metrics import Histogram, Counter, UpDownCounter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)
from opentelemetry.trace import Status, StatusCode
from prometheus_client import start_http_server

import structlog

logger = structlog.get_logger(__name__)
audit_logger = structlog.get_logger("audit")

T = TypeVar('T')

# Global instances
_tracer: Optional[trace.Tracer] = None
_meter: Optional[metrics.Meter] = None
_metrics: Optional['CoRTMetrics'] = None


def generate_request_id() -> str:
    """Generate a short request/session identifier."""
    return uuid.uuid4().hex[:8]


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    """Configure structlog for JSON or console output."""
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.CallsiteParameterAdder(
            [structlog.processors.CallsiteParameter.MODULE]
        ),
    ]
    if fmt == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO))
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(level.upper())
        ),
        logger_factory=structlog.PrintLoggerFactory(),
    )


def audit_log(event: str, **kwargs: Any) -> None:
    """Record an audit log event."""
    audit_logger.info(event, **kwargs)


class CoRTMetrics:
    """CoRT-specific metrics collection."""
    
    def __init__(self, meter: metrics.Meter):
        self.meter = meter
        
        # Request metrics
        self.request_counter = meter.create_counter(
            name="cort_requests_total",
            description="Total number of CoRT requests",
            unit="1",
        )
        
        self.request_duration = meter.create_histogram(
            name="cort_request_duration_seconds",
            description="Request duration in seconds",
            unit="s",
        )
        
        # Thinking metrics
        self.thinking_rounds = meter.create_histogram(
            name="cort_thinking_rounds",
            description="Number of thinking rounds per request",
            unit="1",
        )
        
        self.thinking_duration = meter.create_histogram(
            name="cort_thinking_duration_seconds",
            description="Duration of thinking process",
            unit="s",
        )
        
        self.convergence_counter = meter.create_counter(
            name="cort_convergence_total",
            description="Convergence events by reason",
            unit="1",
        )
        
        # Quality metrics
        self.quality_score = meter.create_histogram(
            name="cort_quality_score",
            description="Final quality scores",
            unit="1",
        )
        
        self.quality_improvement = meter.create_histogram(
            name="cort_quality_improvement",
            description="Quality improvement from initial to final",
            unit="1",
        )
        
        # Token usage
        self.token_usage = meter.create_histogram(
            name="cort_token_usage",
            description="Token usage per request",
            unit="1",
        )

        self.prompt_tokens = meter.create_histogram(
            name="cort_prompt_tokens",
            description="Prompt tokens per request",
            unit="1",
        )

        self.completion_tokens = meter.create_histogram(
            name="cort_completion_tokens",
            description="Completion tokens per request",
            unit="1",
        )
        
        self.token_efficiency = meter.create_histogram(
            name="cort_token_efficiency",
            description="Tokens per thinking round",
            unit="1",
        )
        
        # Cache metrics
        self.cache_hits = meter.create_counter(
            name="cort_cache_hits_total",
            description="Cache hit count",
            unit="1",
        )
        
        self.cache_misses = meter.create_counter(
            name="cort_cache_misses_total",
            description="Cache miss count",
            unit="1",
        )
        
        # Error metrics
        self.error_counter = meter.create_counter(
            name="cort_errors_total",
            description="Error count by type",
            unit="1",
        )
        
        # System metrics
        self.active_sessions = meter.create_up_down_counter(
            name="cort_active_sessions",
            description="Number of active chat sessions",
            unit="1",
        )
        
        self.provider_latency = meter.create_histogram(
            name="cort_provider_latency_seconds",
            description="LLM provider latency",
            unit="s",
        )

        self.provider_failures = meter.create_counter(
            name="cort_provider_failures_total",
            description="LLM provider failure count",
            unit="1",
        )


def initialize_telemetry(
    *,
    service_name: str = "cort",
    service_version: str = "1.0.0",
    enable_console_export: bool = False,
    enable_prometheus: bool = True,
    prometheus_port: int = 8080,
    jaeger_endpoint: Optional[str] = None,
    log_level: str = "INFO",
    log_format: str = "json",
) -> None:
    """
    Initialize OpenTelemetry with configured exporters.
    
    Args:
        service_name: Service name for traces and metrics
        service_version: Service version
        enable_console_export: Export traces to console
        enable_prometheus: Enable Prometheus metrics
        prometheus_port: Port for Prometheus metrics server
        jaeger_endpoint: Jaeger collector endpoint
    """
    global _tracer, _meter, _metrics

    configure_logging(log_level, log_format)
    
    # Create resource
    resource = Resource.create({
        "service.name": service_name,
        "service.version": service_version,
    })
    
    # Setup tracing
    trace_provider = TracerProvider(resource=resource)
    
    if enable_console_export:
        trace_provider.add_span_processor(
            BatchSpanProcessor(ConsoleSpanExporter())
        )
        
    if jaeger_endpoint:
        try:
            from opentelemetry.exporter.jaeger.thrift import JaegerExporter
            jaeger_exporter = JaegerExporter(
                agent_host_name=jaeger_endpoint.split(":")[0],
                agent_port=int(jaeger_endpoint.split(":")[1]) if ":" in jaeger_endpoint else 6831,
            )
            trace_provider.add_span_processor(
                BatchSpanProcessor(jaeger_exporter)
            )
        except ImportError:
            logger.warning("Jaeger exporter not available, install opentelemetry-exporter-jaeger")
    
    trace.set_tracer_provider(trace_provider)
    _tracer = trace.get_tracer(service_name, service_version)
    
    # Setup metrics
    if enable_prometheus:
        reader = PrometheusMetricReader()
        metrics_provider = MeterProvider(resource=resource, metric_readers=[reader])
        metrics.set_meter_provider(metrics_provider)
        
        # Start Prometheus HTTP server
        start_http_server(prometheus_port)
        logger.info(f"Prometheus metrics available at http://localhost:{prometheus_port}/metrics")
    else:
        metrics_provider = MeterProvider(resource=resource)
        metrics.set_meter_provider(metrics_provider)
    
    _meter = metrics.get_meter(service_name, service_version)
    _metrics = CoRTMetrics(_meter)
    
    # Auto-instrument HTTP clients
    RequestsInstrumentor().instrument()
    AioHttpClientInstrumentor().instrument()
    
    logger.info(
        "telemetry_initialized",
        service_name=service_name,
        service_version=service_version,
        prometheus_enabled=enable_prometheus,
        jaeger_enabled=bool(jaeger_endpoint),
    )


def get_tracer() -> trace.Tracer:
    """Get the global tracer instance."""
    if _tracer is None:
        raise RuntimeError("Telemetry not initialized. Call initialize_telemetry() first.")
    return _tracer


def get_metrics() -> CoRTMetrics:
    """Get the global metrics instance."""
    if _metrics is None:
        raise RuntimeError("Telemetry not initialized. Call initialize_telemetry() first.")
    return _metrics


@contextmanager
def trace_span(
    name: str,
    *,
    attributes: Optional[Dict[str, Any]] = None,
    record_exception: bool = True,
):
    """
    Context manager for creating trace spans.
    
    Args:
        name: Span name
        attributes: Span attributes
        record_exception: Record exceptions in span
    """
    tracer = get_tracer()
    
    with tracer.start_as_current_span(name) as span:
        if attributes:
            span.set_attributes(attributes)
            
        try:
            yield span
        except Exception as e:
            if record_exception:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
            raise


@asynccontextmanager
async def async_trace_span(
    name: str,
    *,
    attributes: Optional[Dict[str, Any]] = None,
    record_exception: bool = True,
):
    """Async context manager for creating trace spans."""
    tracer = get_tracer()
    
    with tracer.start_as_current_span(name) as span:
        if attributes:
            span.set_attributes(attributes)
            
        try:
            yield span
        except Exception as e:
            if record_exception:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
            raise


def trace_method(
    name: Optional[str] = None,
    *,
    attributes: Optional[Dict[str, Any]] = None,
):
    """
    Decorator for tracing methods.
    
    Args:
        name: Span name (defaults to function name)
        attributes: Additional span attributes
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        span_name = name or f"{func.__module__}.{func.__name__}"
        
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs) -> T:
                async with async_trace_span(
                    span_name,
                    attributes=attributes,
                ):
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs) -> T:
                with trace_span(
                    span_name,
                    attributes=attributes,
                ):
                    return func(*args, **kwargs)
            return sync_wrapper
            
    return decorator


class TelemetryMixin:
    """Mixin class to add telemetry to any class."""
    
    def record_metric(
        self,
        metric_name: str,
        value: float,
        *,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """Record a metric value."""
        metrics = get_metrics()
        metric = getattr(metrics, metric_name, None)
        
        if metric is None:
            logger.warning(f"Unknown metric: {metric_name}")
            return
            
        if isinstance(metric, Counter):
            metric.add(value, labels or {})
        elif isinstance(metric, Histogram):
            metric.record(value, labels or {})
        elif isinstance(metric, UpDownCounter):
            metric.add(value, labels or {})
            
    def create_span_attributes(self, **kwargs) -> Dict[str, Any]:
        """Create span attributes with common fields."""
        attributes = {
            "cort.class": self.__class__.__name__,
        }
        
        # Add any instance attributes that might be useful
        if hasattr(self, "model"):
            attributes["cort.model"] = str(self.model)
        if hasattr(self, "session_id"):
            attributes["cort.session_id"] = str(self.session_id)
            
        attributes.update(kwargs)
        return attributes


# Convenience functions for common metrics
def record_thinking_metrics(
    rounds: int,
    duration: float,
    convergence_reason: str,
    initial_quality: float,
    final_quality: float,
    total_tokens: int,
    *,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
) -> None:
    """Record thinking process metrics."""
    metrics = get_metrics()
    
    metrics.thinking_rounds.record(rounds)
    metrics.thinking_duration.record(duration)
    metrics.convergence_counter.add(1, {"reason": convergence_reason})
    metrics.quality_score.record(final_quality)
    metrics.quality_improvement.record(final_quality - initial_quality)
    metrics.token_usage.record(total_tokens)

    if prompt_tokens is not None:
        metrics.prompt_tokens.record(prompt_tokens)
    if completion_tokens is not None:
        metrics.completion_tokens.record(completion_tokens)
    
    if rounds > 0:
        metrics.token_efficiency.record(total_tokens / rounds)


def record_cache_metrics(hit: bool, cache_type: str = "memory", *, count: int = 1) -> None:
    """Record cache hit/miss metrics."""
    metrics = get_metrics()
    labels = {"cache_type": cache_type}

    if hit:
        metrics.cache_hits.add(count, labels)
    else:
        metrics.cache_misses.add(count, labels)


def record_error(error_type: str, error_message: str) -> None:
    """Record error metrics."""
    metrics = get_metrics()
    metrics.error_counter.add(1, {
        "error_type": error_type,
        "error_category": error_type.split(".")[-1],  # Last part of error type
    })
    
    # Also log for debugging
    logger.error(
        "cort_error_recorded",
        error_type=error_type,
        error_message=error_message,
    )


def record_provider_latency(provider: str, latency: float) -> None:
    """Record latency for a provider."""
    metrics = get_metrics()
    metrics.provider_latency.record(latency, {"provider": provider})


def record_provider_failure(provider: str, error_type: str) -> None:
    """Record a provider failure."""
    metrics = get_metrics()
    metrics.provider_failures.add(1, {"provider": provider, "error_type": error_type})


def instrument_fastapi(app: "FastAPI", *, excluded_urls: str = "/docs,/openapi.json") -> None:
    """Attach OpenTelemetry tracing middleware to a FastAPI app."""
    FastAPIInstrumentor().instrument_app(
        app,
        tracer_provider=trace.get_tracer_provider(),
        excluded_urls=excluded_urls,
    )
