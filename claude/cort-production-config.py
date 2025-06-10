"""Production configuration with security and validation."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from pydantic import BaseSettings, Field, validator, SecretStr
from pydantic.networks import HttpUrl, PostgresDsn, RedisDsn


class SecuritySettings(BaseSettings):
    """Security-related settings."""
    
    # API Security
    require_api_key: bool = True
    api_key_master_key: SecretStr = Field(..., env="API_KEY_MASTER_KEY")
    max_prompt_length: int = 10000
    max_context_messages: int = 100
    max_tokens_per_request: int = 100000
    
    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600
    
    # Session Management
    session_timeout: int = 3600
    max_concurrent_sessions: int = 1000
    session_secret_key: SecretStr = Field(..., env="SESSION_SECRET_KEY")
    
    # CORS
    cors_allowed_origins: List[HttpUrl] = []
    cors_allow_credentials: bool = True
    cors_max_age: int = 3600
    
    # Content Security
    blocked_patterns: List[str] = Field(default_factory=list)
    enable_output_sanitization: bool = True
    
    @validator("api_key_master_key", "session_secret_key")
    def validate_secrets(cls, v: SecretStr) -> SecretStr:
        """Ensure secrets are strong enough."""
        if len(v.get_secret_value()) < 32:
            raise ValueError("Secret keys must be at least 32 characters")
        return v
        
    class Config:
        case_sensitive = False


class DatabaseSettings(BaseSettings):
    """Database configuration."""
    
    # PostgreSQL
    postgres_url: Optional[PostgresDsn] = Field(None, env="DATABASE_URL")
    postgres_pool_size: int = 20
    postgres_max_overflow: int = 40
    postgres_pool_timeout: float = 30.0
    postgres_echo: bool = False
    
    # Redis
    redis_url: Optional[RedisDsn] = Field(None, env="REDIS_URL")
    redis_max_connections: int = 50
    redis_decode_responses: bool = True
    redis_socket_keepalive: bool = True
    redis_socket_timeout: int = 30
    
    # MongoDB (if needed)
    mongodb_url: Optional[str] = Field(None, env="MONGODB_URL")
    mongodb_database: str = "cort"
    
    class Config:
        case_sensitive = False


class LLMSettings(BaseSettings):
    """LLM provider settings."""
    
    # Primary provider
    primary_provider: str = Field("openrouter", env="LLM_PRIMARY_PROVIDER")
    primary_model: str = Field(..., env="LLM_PRIMARY_MODEL")
    primary_api_key: SecretStr = Field(..., env="LLM_PRIMARY_API_KEY")
    
    # Fallback providers
    fallback_providers: List[str] = []
    fallback_models: List[str] = []
    fallback_api_keys: List[SecretStr] = []
    
    # Provider settings
    max_retries: int = 3
    timeout: float = 30.0
    enable_hedging: bool = True
    hedge_delay: float = 0.5
    max_hedges: int = 2
    
    # Model parameters
    default_temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    
    @validator("primary_api_key")
    def validate_api_key(cls, v: SecretStr) -> SecretStr:
        """Ensure API key is provided."""
        if not v.get_secret_value():
            raise ValueError("Primary API key must be provided")
        return v
        
    class Config:
        case_sensitive = False


class CacheSettings(BaseSettings):
    """Cache configuration."""
    
    # Cache backend
    cache_backend: str = Field("hybrid", env="CACHE_BACKEND")  # memory, disk, redis, hybrid
    
    # Memory cache
    memory_cache_size: int = 1000
    memory_cache_ttl: int = 3600
    
    # Disk cache
    disk_cache_enabled: bool = True
    disk_cache_path: str = Field("/var/cache/cort", env="DISK_CACHE_PATH")
    disk_cache_size_mb: int = 5000
    disk_cache_compression: bool = True
    
    # Distributed cache
    distributed_cache_enabled: bool = False
    distributed_cache_prefix: str = "cort:"
    
    # Semantic cache
    semantic_cache_enabled: bool = True
    semantic_cache_threshold: float = 0.95
    semantic_cache_max_entries: int = 10000
    
    class Config:
        case_sensitive = False


class MonitoringSettings(BaseSettings):
    """Monitoring and observability settings."""
    
    # Metrics
    metrics_enabled: bool = True
    prometheus_port: int = Field(9090, env="PROMETHEUS_PORT")
    
    # Tracing
    tracing_enabled: bool = True
    jaeger_endpoint: Optional[str] = Field(None, env="JAEGER_ENDPOINT")
    trace_sampling_rate: float = 0.1
    
    # Logging
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_format: str = "json"  # json or console
    log_file: Optional[str] = Field(None, env="LOG_FILE")
    log_rotation: str = "daily"
    log_retention_days: int = 30
    
    # Health checks
    health_check_enabled: bool = True
    health_check_interval: int = 60
    health_check_timeout: int = 10
    
    # Alerts
    alert_webhook_url: Optional[HttpUrl] = Field(None, env="ALERT_WEBHOOK_URL")
    alert_email: Optional[str] = Field(None, env="ALERT_EMAIL")
    
    @validator("log_level")
    def validate_log_level(cls, v: str) -> str:
        """Ensure valid log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()
        
    class Config:
        case_sensitive = False


class PerformanceSettings(BaseSettings):
    """Performance tuning settings."""
    
    # Thinking optimization
    enable_parallel_thinking: bool = True
    max_parallel_alternatives: int = 3
    enable_adaptive_optimization: bool = True
    enable_prompt_compression: bool = True
    
    # Resource limits
    max_thinking_rounds: int = 5
    max_thinking_time: float = 30.0
    quality_threshold: float = 0.92
    
    # Batching
    enable_request_batching: bool = True
    batch_size: int = 5
    batch_timeout: float = 0.1
    
    # Connection pooling
    aiohttp_connector_limit: int = 100
    aiohttp_connector_ttl: int = 300
    
    # Worker settings
    worker_count: int = Field(4, env="WORKER_COUNT")
    worker_class: str = "uvicorn.workers.UvicornWorker"
    worker_timeout: int = 60
    
    @validator("worker_count")
    def validate_worker_count(cls, v: int) -> int:
        """Ensure reasonable worker count."""
        cpu_count = os.cpu_count() or 1
        if v < 1 or v > cpu_count * 4:
            raise ValueError(f"Worker count should be between 1 and {cpu_count * 4}")
        return v
        
    class Config:
        case_sensitive = False


class ProductionSettings(BaseSettings):
    """Complete production configuration."""
    
    # Application
    app_name: str = Field("CoRT", env="APP_NAME")
    app_version: str = Field("1.0.0", env="APP_VERSION")
    environment: str = Field("production", env="APP_ENV")
    debug: bool = Field(False, env="DEBUG")
    
    # API
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    api_base_path: str = Field("/api/v1", env="API_BASE_PATH")
    
    # Frontend
    frontend_url: HttpUrl = Field(..., env="FRONTEND_URL")
    
    # Sub-configurations
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    performance: PerformanceSettings = Field(default_factory=PerformanceSettings)
    
    @validator("environment")
    def validate_environment(cls, v: str) -> str:
        """Ensure valid environment."""
        valid_envs = ["development", "staging", "production"]
        if v not in valid_envs:
            raise ValueError(f"Environment must be one of {valid_envs}")
        return v
        
    @validator("debug")
    def validate_debug(cls, v: bool, values: Dict[str, Any]) -> bool:
        """Ensure debug is off in production."""
        if v and values.get("environment") == "production":
            raise ValueError("Debug must be False in production")
        return v
        
    class Config:
        case_sensitive = False
        env_file = ".env"
        env_file_encoding = "utf-8"
        
        
def load_production_config() -> ProductionSettings:
    """Load and validate production configuration."""
    try:
        config = ProductionSettings()
        
        # Additional runtime validation
        _validate_runtime_config(config)
        
        return config
        
    except Exception as e:
        # Configuration errors should fail fast
        import sys
        print(f"FATAL: Failed to load configuration: {e}", file=sys.stderr)
        sys.exit(1)
        
        
def _validate_runtime_config(config: ProductionSettings) -> None:
    """Additional runtime configuration validation."""
    
    # Ensure required services are configured
    if config.cache.cache_backend == "redis" and not config.database.redis_url:
        raise ValueError("Redis URL required when using Redis cache backend")
        
    # Validate security settings
    if config.security.require_api_key and not config.security.api_key_master_key:
        raise ValueError("API key master key required when API keys are enabled")
        
    # Check for production readiness
    if config.environment == "production":
        # Ensure monitoring is enabled
        if not config.monitoring.metrics_enabled:
            raise ValueError("Metrics must be enabled in production")
            
        # Ensure secure settings
        if config.security.cors_allowed_origins == ["*"]:
            raise ValueError("CORS must specify allowed origins in production")
            
        # Ensure database is configured
        if not config.database.postgres_url and not config.database.redis_url:
            raise ValueError("At least one database must be configured in production")


# Export convenience function
get_production_config = load_production_config
