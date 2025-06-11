
"""OpenRouter API helpers."""

from .openrouter import (
    sync_chat_completion,
    async_chat_completion,
    get_embeddings,
)
from .model_catalog import fetch_models

__all__ = [
    "sync_chat_completion",
    "async_chat_completion",
    "get_embeddings",
    "fetch_models",
]
