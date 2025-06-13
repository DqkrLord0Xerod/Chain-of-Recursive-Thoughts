# -*- coding: utf-8 -*-
"""Unified configuration module."""

from __future__ import annotations

import os
from typing import List, Optional

from pydantic import (
    Field,
    HttpUrl,
    PostgresDsn,
    RedisDsn,
    SecretStr,
    field_validator,
)
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Basic application settings."""

    llm_provider: str = "openrouter"
    openrouter_api_key: str | None = None
    openai_api_key: str | None = None
    model: str = "mistralai/mistral-small-3.1-24b-instruct:free"
    api_base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    embed_url: str = "https://openrouter.ai/api/v1/embeddings"
    frontend_url: str = "http://localhost:3000"
    ws_base_url: str = "ws://localhost:8000"
    thinking_strategy: str = Field("adaptive", env="THINKING_STRATEGY")

    class Config:
        env_file = ".env"
        case_sensitive = False


class SecuritySettings(BaseSettings):
    """Security related settings."""

    require_api_key: bool = True
    api_key_master_key: SecretStr = Field(..., env="API_KEY_MASTER_KEY")
    max_prompt_length: int = 10000
    max_context_messages: int = 100
    max_tokens_per_request: int = 100000

    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600

    session_timeout: int = 3600
    max_concurrent_sessions: int = 1000
    session_secret_key: SecretStr = Field(..., env="SESSION_SECRET_KEY")

    cors_allowed_origins: List[HttpUrl] = []
    cors_allow_credentials: bool = True
    cors_max_age: int = 3600

    blocked_patterns: List[str] = Field(default_factory=list)
    enable_output_sanitization: bool = True

    @field_validator("api_key_master_key", "session_secret_key")
    @classmethod
    def _validate_secret(cls, v: SecretStr) -> SecretStr:
        if len(v.get_secret_value()) < 32:
            raise ValueError("Secret keys must be at least 32 characters")
        return v

    class Config:
        case_sensitive = False


class DatabaseSettings(BaseSettings):
    """Database configuration."""

    postgres_url: Optional[PostgresDsn] = Field(None, env="DATABASE_URL")
    postgres_pool_size: int = 20
    postgres_max_overflow: int = 40
    postgres_pool_timeout: float = 30.0
    postgres_echo: bool = False

    redis_url: Optional[RedisDsn] = Field(None, env="REDIS_URL")
    redis_max_connections: int = 50
    redis_decode_responses: bool = True
    redis_socket_keepalive: bool = True
    redis_socket_timeout: int = 30

    mongodb_url: Optional[str] = Field(None, env="MONGODB_URL")
    mongodb_database: str = "cort"

    class Config:
        case_sensitive = False


class LLMSettings(BaseSettings):
    """LLM provider settings."""

    provider: str = Field("openrouter", env="LLM_PROVIDER")
    primary_provider: str = Field("openrouter", env="LLM_PRIMARY_PROVIDER")
    primary_model: str = Field(..., env="LLM_PRIMARY_MODEL")
    primary_api_key: SecretStr = Field(..., env="LLM_PRIMARY_API_KEY")

    fallback_providers: List[str] = []
    fallback_models: List[str] = []
    fallback_api_keys: List[SecretStr] = []

    max_retries: int = 3
    timeout: float = 30.0
    enable_hedging: bool = True
    hedge_delay: float = 0.5
    max_hedges: int = 2

    default_temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    @field_validator("primary_api_key")
    @classmethod
    def _validate_api_key(cls, v: SecretStr) -> SecretStr:
        if not v.get_secret_value():
            raise ValueError("Primary API key must be provided")
        return v

    class Config:
        case_sensitive = False


class CacheSettings(BaseSettings):
    """Cache configuration."""

    cache_backend: str = Field("hybrid", env="CACHE_BACKEND")

    memory_cache_size: int = 1000
    memory_cache_ttl: int = 3600

    disk_cache_enabled: bool = True
    disk_cache_path: str = Field("/var/cache/cort", env="DISK_CACHE_PATH")
    disk_cache_size_mb: int = 5000
    disk_cache_compression: bool = True

    distributed_cache_enabled: bool = False
    distributed_cache_prefix: str = "cort:"

    semantic_cache_enabled: bool = True
    semantic_cache_threshold: float = 0.95
    semantic_cache_max_entries: int = 10000
    semantic_cache_ttl: int = 3600
    semantic_cache_min_hits: int = 0

    class Config:
        case_sensitive = False


class MemorySettings(BaseSettings):
    """Vector memory configuration."""

    backend: str = "faiss"
    index_path: str = "memory.index"
    embedding_dim: int = 1536
    retrieval_top_k: int = 3

    class Config:
        case_sensitive = False


class MonitoringSettings(BaseSettings):
    """Monitoring and observability."""

    metrics_enabled: bool = True
    prometheus_port: int = Field(9090, env="PROMETHEUS_PORT")

    tracing_enabled: bool = True
    jaeger_endpoint: Optional[str] = Field(None, env="JAEGER_ENDPOINT")
    trace_sampling_rate: float = 0.1

    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_format: str = "json"
    log_file: Optional[str] = Field(None, env="LOG_FILE")
    log_rotation: str = "daily"
    log_retention_days: int = 30

    health_check_enabled: bool = True
    health_check_interval: int = 60
    health_check_timeout: int = 10

    alert_webhook_url: Optional[HttpUrl] = Field(None, env="ALERT_WEBHOOK_URL")
    alert_email: Optional[str] = Field(None, env="ALERT_EMAIL")

    @field_validator("log_level")
    @classmethod
    def _validate_level(cls, v: str) -> str:
        valid = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid:
            raise ValueError(f"Log level must be one of {valid}")
        return v.upper()

    class Config:
        case_sensitive = False


class RetentionSettings(BaseSettings):
    """Data retention policies."""

    log_retention_days: int = Field(30, env="LOG_RETENTION_DAYS")
    audit_log_retention_days: int = Field(90, env="AUDIT_LOG_RETENTION_DAYS")
    metrics_retention_days: int = Field(180, env="METRICS_RETENTION_DAYS")

    class Config:
        case_sensitive = False


class PerformanceSettings(BaseSettings):
    """Performance tuning settings."""

    enable_parallel_thinking: bool = True
    max_parallel_alternatives: int = 3
    enable_adaptive_optimization: bool = True
    enable_prompt_compression: bool = True

    max_thinking_rounds: int = 5
    max_thinking_time: float = 30.0
    quality_threshold: float = 0.92

    enable_request_batching: bool = True
    batch_size: int = 5
    batch_timeout: float = 0.1

    aiohttp_connector_limit: int = 100
    aiohttp_connector_ttl: int = 300

    worker_count: int = Field(4, env="WORKER_COUNT")
    worker_class: str = "uvicorn.workers.UvicornWorker"
    worker_timeout: int = 60

    @field_validator("worker_count")
    @classmethod
    def _validate_workers(cls, v: int) -> int:
        cpu_count = os.cpu_count() or 1
        if v < 1 or v > cpu_count * 4:
            raise ValueError(
                f"Worker count should be between 1 and {cpu_count * 4}"
            )
        return v

    class Config:
        case_sensitive = False


class ProductionSettings(BaseSettings):
    """Complete production configuration."""

    app_name: str = Field("CoRT", env="APP_NAME")
    app_version: str = Field("1.0.0", env="APP_VERSION")
    environment: str = Field("production", env="APP_ENV")
    debug: bool = Field(False, env="DEBUG")

    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    api_base_path: str = Field("/api/v1", env="API_BASE_PATH")

    frontend_url: HttpUrl = Field(..., env="FRONTEND_URL")

    security: SecuritySettings = Field(default_factory=SecuritySettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    performance: PerformanceSettings = Field(default_factory=PerformanceSettings)
    retention: RetentionSettings = Field(default_factory=RetentionSettings)

    @field_validator("environment")
    @classmethod
    def _validate_env(cls, v: str) -> str:
        if v not in {"development", "staging", "production"}:
            raise ValueError(
                "Environment must be one of development, staging, production"
            )
        return v

    @field_validator("debug")
    @classmethod
    def _validate_debug(cls, v: bool, info) -> bool:
        env = info.data.get("environment") if isinstance(info.data, dict) else None
        if v and env == "production":
            raise ValueError("Debug must be False in production")
        return v

    class Config:
        case_sensitive = False
        env_file = ".env"
        env_file_encoding = "utf-8"


def load_production_config() -> ProductionSettings:
    """Load and validate production configuration."""
    try:
        data = {
            "environment": os.getenv("APP_ENV", "production"),
            "frontend_url": os.getenv("FRONTEND_URL"),
            "llm": {
                "primary_model": os.getenv("LLM_PRIMARY_MODEL"),
                "primary_api_key": os.getenv("LLM_PRIMARY_API_KEY"),
            },
            "security": {
                "api_key_master_key": os.getenv("API_KEY_MASTER_KEY"),
                "session_secret_key": os.getenv("SESSION_SECRET_KEY"),
            },
            "database": {"postgres_url": os.getenv("DATABASE_URL")},
            "retention": {
                "log_retention_days": os.getenv("LOG_RETENTION_DAYS"),
                "audit_log_retention_days": os.getenv("AUDIT_LOG_RETENTION_DAYS"),
                "metrics_retention_days": os.getenv("METRICS_RETENTION_DAYS"),
            },
        }
        config = ProductionSettings.model_validate(data)
        _validate_runtime_config(config)
        return config
    except Exception as exc:
        import sys
        print(f"FATAL: Failed to load configuration: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def _validate_runtime_config(config: ProductionSettings) -> None:
    """Extra runtime validation after model creation."""
    if config.cache.cache_backend == "redis" and not config.database.redis_url:
        raise ValueError("Redis URL required when using Redis cache backend")
    if config.security.require_api_key and not config.security.api_key_master_key:
        raise ValueError("API key master key required when API keys are enabled")
    if config.environment == "production":
        if not config.monitoring.metrics_enabled:
            raise ValueError("Metrics must be enabled in production")
        if config.security.cors_allowed_origins == ["*"]:
            raise ValueError("CORS must specify allowed origins in production")
        if not config.database.postgres_url and not config.database.redis_url:
            raise ValueError(
                "At least one database must be configured in production"
            )


get_production_config = load_production_config
