from __future__ import annotations

import time
from typing import Any, Dict, List

import requests

from config import settings

_TTL = 300  # 5 minutes
_cache: Dict[str, Any] = {"data": None, "expires_at": 0}


def _models_url() -> str:
    """Return the OpenRouter models endpoint derived from api_base_url."""
    base = settings.api_base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        base = base[: -len("/chat/completions")]
    return f"{base}/models"


def fetch_models() -> List[Dict[str, Any]]:
    """Fetch and cache OpenRouter model metadata."""
    if _cache["data"] is not None and time.time() < _cache["expires_at"]:
        return _cache["data"]

    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.get(_models_url(), headers=headers, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    models = data.get("data") if isinstance(data, dict) else data
    results: List[Dict[str, Any]] = []
    for m in models or []:
        results.append(
            {
                "id": m.get("id"),
                "context_length": m.get("context_length"),
                "pricing": m.get("pricing", {}),
            }
        )

    _cache["data"] = results
    _cache["expires_at"] = time.time() + _TTL
    return results
