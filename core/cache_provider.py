from __future__ import annotations

from collections import OrderedDict

from core.interfaces import CacheProvider


class InMemoryCache(CacheProvider):
    """Simple LRU in-memory cache."""

    def __init__(self, max_size: int = 128) -> None:
        self.max_size = max_size
        self.store: OrderedDict[str, str] = OrderedDict()

    def get(self, key: str) -> str | None:
        if key in self.store:
            self.store.move_to_end(key)
            return self.store[key]
        return None

    def set(self, key: str, value: str) -> None:
        self.store[key] = value
        self.store.move_to_end(key)
        if len(self.store) > self.max_size:
            self.store.popitem(last=False)
