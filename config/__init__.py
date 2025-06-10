"""Environment-aware settings for RecThink."""

from __future__ import annotations

import os

from .settings import Settings
from .production import ProductionSettings, load_production_config


class Development(Settings):
    """Default development configuration."""

    pass


class Staging(Settings):
    """Settings for a staging deployment."""

    api_base_url: str = "https://staging.example.com/api"
    frontend_url: str = "https://staging.example.com"
    ws_base_url: str = "wss://staging.example.com"


class Production(Settings):
    """Settings for production deployments."""

    api_base_url: str = "https://api.example.com"
    frontend_url: str = "https://example.com"
    ws_base_url: str = "wss://example.com"


_env_map = {
    "development": Development,
    "staging": Staging,
    "production": Production,
}


def get_settings() -> Settings:
    """Return settings based on ``APP_ENV``."""

    env = os.getenv("APP_ENV", "development").lower()
    cls = _env_map.get(env, Development)
    return cls()


settings = get_settings()


__all__ = [
    "Development",
    "Staging",
    "Production",
    "ProductionSettings",
    "load_production_config",
    "settings",
    "get_settings",
]
