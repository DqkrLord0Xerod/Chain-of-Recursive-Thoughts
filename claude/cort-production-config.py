"""Deprecated - use ``config.config`` instead."""

from config.config import (
    SecuritySettings,
    DatabaseSettings,
    LLMSettings,
    CacheSettings,
    MonitoringSettings,
    PerformanceSettings,
    ProductionSettings,
    load_production_config,
)

__all__ = [
    "SecuritySettings",
    "DatabaseSettings",
    "LLMSettings",
    "CacheSettings",
    "MonitoringSettings",
    "PerformanceSettings",
    "ProductionSettings",
    "load_production_config",
]
