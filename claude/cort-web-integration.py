"""Updated web API using the new architecture."""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import structlog
import uvicorn

# New architecture imports
from config import get_production_config
from core.providers.llm import OpenRouterLLMProvider, MultiProviderLLM
from core.providers.cache import HybridCacheProvider, InMemoryLRUCache, DiskCacheProvider
from core.providers.resilient_llm import ResilientLLMProvider
from core.providers.embeddings import EmbeddingProvider
from core.providers.quality import EnhancedQualityEvaluator
from core.chat_v2 import RecursiveThinkingEngine, AdaptiveThinkingStrategy
from core.context_manager import ContextManager
from core.security.api_security import (
    APIKeyManager,
    SecurityMiddleware,
    SecurityConfig,
    SecurityError,
    ValidationError,
    RateLimitError,
)
from monitoring.telemetry import (
    initialize_telemetry,
    trace_method,
    get_metrics,
    TelemetryMixin,
)
from monitoring.metrics_v2 import MetricsAnalyzer

logger = structlog.get_logger(__name__)

# Global instances
app: FastAPI = None
engine_pool: Dict[str, RecursiveThinkingEngine] = {}
security_middleware: SecurityMiddleware = None
metrics_analyzer: MetricsAnalyzer = None


# Request/Response models
class ChatRequest(BaseModel):
    prompt: str
    context: Optional[List[Dict[str, str]]] = None
    thinking_rounds: Optional[int] = None
    temperature: float = 0.7
    enable_streaming: bool = False
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    thinking_rounds: int
    quality_score: float
    processing_time: float
    cached: bool
    session_id: Optional[str] = None
    metadata: Dict = {}


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime: float
    services: Dict[str, Dict]


# Dependency injection
async def get_security_validator(
    authorization: Optional[str] = Header(None),
    x_session_id: Optional[str] = Header(None),
) -> Dict:
    """Extract and validate security credentials."""
    api_key = None
    if authorization and authorization.startswith("Bearer "):
        api_key = authorization[7:]
        
    return {
        "api_key": api_key,
        "session_id": x_session_id,
    }


async def get_thinking_engine() -> RecursiveThinkingEngine:
    """Get or create thinking engine for request."""
    # In production, might want per-user engines
    if "default" not in engine_pool:
        engine_pool["default"] = await create_thinking_engine()
    return engine_pool["default"]


# Application lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    logger.info("starting_application")
    
    # Load configuration
    config = get_production_config()
    
    # Initialize telemetry
    if config.monitoring.metrics_enabled:
        initialize_telemetry(
            service_name=config.app_name,
            service_version=config.app_version,
            enable_prometheus=True,
            prometheus_port=config.monitoring.prometheus_port,
            jaeger_endpoint=config.monitoring.jaeger_endpoint,
        )
    
    # Initialize security
    global security_middleware
    security_config = SecurityConfig(
        **config.security.dict()
    )
    api_key_manager = APIKeyManager(
        config.security.api_key_master_key.get_secret_value()
    )
    security_middleware = SecurityMiddleware(security_config, api_key_manager)
    
    # Initialize metrics analyzer
    global metrics_analyzer
    metrics_analyzer = MetricsAnalyzer()
    
    # Create default engine
    engine_pool["default"] = await create_thinking_engine()
    
    logger.info("application_started", config=config.app_name)
    
    yield
    
    # Cleanup
    logger.info("shutting_down_application")
    for engine in engine_pool.values():
        if hasattr(engine, 'cleanup'):
            await engine.cleanup()
    
    logger.info("application_stopped")


async def create_thinking_engine() -> RecursiveThinkingEngine:
    """Create a new thinking engine with production configuration."""
    config = get_production_config()
    
    # Create LLM providers
    providers = []
    
    # Primary provider
    primary = OpenRouterLLMProvider(
        api_key=config.llm.primary_api_key.get_secret_value(),
        model=config.llm.primary_model,
        max_retries=config.llm.max_retries,
        timeout=config.llm.timeout,
    )
    providers.append(primary)
    
    # Fallback providers
    for i, (model, api_key) in enumerate(
        zip(config.llm.fallback_models, config.llm.fallback_api_keys)
    ):
        fallback = OpenRouterLLMProvider(
            api_key=api_key.get_secret_value(),
            model=model,
        )
        providers.append(fallback)
    
    # Wrap with resilience
    resilient_llm = ResilientLLMProvider(
        providers,
        enable_hedging=config.llm.enable_hedging,
        hedge_delay=config.llm.hedge_delay,
        max_hedges=config.llm.max_hedges,
    )
    
    # Create cache
    memory_cache = InMemoryLRUCache(
        max_size=config.cache.memory_cache_size
    )
    
    if config.cache.disk_cache_enabled:
        disk_cache = DiskCacheProvider(
            cache_dir=config.cache.disk_cache_path,
            max_size_mb=config.cache.disk_cache_size_mb,
            enable_compression=config.cache.disk_cache_compression,
        )
        cache = HybridCacheProvider(memory_cache, disk_cache)
    else:
        cache = memory_cache
    
    # Create evaluator
    embedding_provider = EmbeddingProvider()  # Would need implementation
    evaluator = EnhancedQualityEvaluator(embedding_provider)
    
    # Create context manager
    import tiktoken
    tokenizer = tiktoken.get_encoding("cl100k_base")
    context_manager = ContextManager(
        max_tokens=config.performance.max_thinking_rounds * 1000,
        tokenizer=tokenizer,
    )
    
    # Create thinking strategy
    strategy = AdaptiveThinkingStrategy(
        resilient_llm,
        max_rounds=config.performance.max_thinking_rounds,
        quality_threshold=config.performance.quality_threshold,
    )
    
    # Create engine
    engine = RecursiveThinkingEngine(
        llm=resilient_llm,
        cache=cache,
        evaluator=evaluator,
        context_manager=context_manager,
        thinking_strategy=strategy,
        metrics_recorder=get_metrics() if config.monitoring.metrics_enabled else None,
    )
    
    return engine


# API endpoints
app = FastAPI(
    title="CoRT API",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure from settings in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/v1/chat", response_model=ChatResponse)
@trace_method()
async def chat(
    request: ChatRequest,
    security: Dict = Depends(get_security_validator),
    engine: RecursiveThinkingEngine = Depends(get_thinking_engine),
) -> ChatResponse:
    """Process chat request with recursive thinking."""
    
    try:
        # Validate request
        validation_result = await security_middleware.validate_request(
            api_key=security.get("api_key"),
            session_id=security.get("session_id"),
            prompt=request.prompt,
            context=request.context,
        )
        
        # Record request
        start_time = time.time()
        
        # Execute thinking
        result = await engine.think(
            prompt=request.prompt,
            context=request.context,
            max_thinking_time=30.0,
            target_quality=0.9,
        )
        
        # Record metrics
        if metrics_analyzer:
            from monitoring.metrics_v2 import ThinkingMetrics
            
            metrics = ThinkingMetrics(
                session_id=security.get("session_id", "anonymous"),
                start_time=start_time,
                end_time=time.time(),
                rounds_completed=result["thinking_rounds"],
                convergence_reason=result["metadata"].get("convergence_reason"),
                quality_scores=[result["initial_quality"], result["final_quality"]],
                token_usage_per_round=[],  # Would need to track this
            )
            
            analysis = metrics_analyzer.record_session(metrics)
            
            # Log any anomalies
            if analysis["anomalies"]:
                logger.warning("thinking_anomalies", anomalies=analysis["anomalies"])
        
        return ChatResponse(
            response=result["response"],
            thinking_rounds=result["thinking_rounds"],
            quality_score=result["final_quality"],
            processing_time=result["thinking_time"],
            cached=result["cached"],
            session_id=security.get("session_id"),
            metadata={
                "improvement": result["improvement"],
                "rate_limit_remaining": validation_result.get("rate_limit_remaining"),
            },
        )
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RateLimitError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except SecurityError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        logger.error("chat_error", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/v1/chat/stream")
@trace_method()
async def chat_stream(
    request: ChatRequest,
    security: Dict = Depends(get_security_validator),
    engine: RecursiveThinkingEngine = Depends(get_thinking_engine),
):
    """Stream chat responses with thinking progress."""
    
    try:
        # Validate request
        await security_middleware.validate_request(
            api_key=security.get("api_key"),
            session_id=security.get("session_id"),
            prompt=request.prompt,
            context=request.context,
        )
        
        async def generate():
            """Generate streaming response."""
            async for update in engine.think_stream(
                prompt=request.prompt,
                context=request.context,
            ):
                yield f"data: {json.dumps(update)}\n\n"
                
            yield "data: [DONE]\n\n"
            
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
        )
        
    except Exception as e:
        logger.error("stream_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    config = get_production_config()
    
    # Check services
    services = {}
    
    # Check LLM providers
    if "default" in engine_pool:
        engine = engine_pool["default"]
        if hasattr(engine.llm, 'health_check'):
            services["llm"] = await engine.llm.health_check()
        else:
            services["llm"] = {"status": "unknown"}
    
    # Check cache
    if "default" in engine_pool:
        engine = engine_pool["default"]
        cache_stats = await engine.cache.stats()
        services["cache"] = {
            "status": "healthy",
            "stats": cache_stats,
        }
    
    # Check metrics
    if metrics_analyzer:
        services["metrics"] = {
            "status": "healthy",
            "summary": metrics_analyzer.get_summary_stats(),
        }
    
    return HealthResponse(
        status="healthy",
        version=config.app_version,
        uptime=time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0,
        services=services,
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    # Prometheus client handles this automatically
    return Response(
        content="",
        media_type="text/plain",
    )


@app.get("/api/v1/stats")
@trace_method()
async def get_stats(
    security: Dict = Depends(get_security_validator),
):
    """Get system statistics."""
    
    # Require API key for stats
    if not security.get("api_key"):
        raise HTTPException(status_code=403, detail="API key required")
    
    stats = {
        "engines": len(engine_pool),
        "metrics": metrics_analyzer.get_summary_stats() if metrics_analyzer else {},
        "recommendations": metrics_analyzer.get_recommendations() if metrics_analyzer else [],
    }
    
    # Add provider stats
    if "default" in engine_pool and hasattr(engine_pool["default"].llm, 'get_metrics'):
        stats["providers"] = engine_pool["default"].llm.get_metrics()
    
    return stats


# Error handlers
@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    return {"error": "validation_error", "detail": str(exc)}, 400


@app.exception_handler(RateLimitError)  
async def rate_limit_error_handler(request: Request, exc: RateLimitError):
    return {"error": "rate_limit", "detail": str(exc)}, 429


@app.exception_handler(SecurityError)
async def security_error_handler(request: Request, exc: SecurityError):
    return {"error": "security_error", "detail": str(exc)}, 403


if __name__ == "__main__":
    # For development only
    uvicorn.run(
        "recthink_web_v2:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )