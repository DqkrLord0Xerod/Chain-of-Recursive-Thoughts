import os
import sys
import types
import asyncio

import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)


# Fake aioredis module used for testing
class FakeRedis:
    def __init__(self):
        self.store = {}
        self.sets = {}
        self.ttl = {}

    async def get(self, key):
        exp = self.ttl.get(key)
        if exp and exp <= asyncio.get_event_loop().time():
            self.store.pop(key, None)
            self.ttl.pop(key, None)
            return None
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key] = value
        if ex is not None:
            self.ttl[key] = asyncio.get_event_loop().time() + ex
        return True

    async def delete(self, *keys):
        removed = 0
        for key in keys:
            if key in self.store:
                self.store.pop(key)
                removed += 1
            self.ttl.pop(key, None)
        return removed

    async def sadd(self, name, *values):
        self.sets.setdefault(name, set()).update(values)
        return len(values)

    async def srem(self, name, *values):
        self.sets.get(name, set()).difference_update(values)
        return len(values)

    async def smembers(self, name):
        return set(self.sets.get(name, set()))

    async def expire(self, name, ttl):
        # TTL handling not needed for tests
        return True

    async def flushdb(self):
        self.store.clear()
        self.sets.clear()
        self.ttl.clear()
        return True

    async def dbsize(self):
        return len(self.store)

    async def info(self, section=None):
        return {"used_memory": len(self.store)}


def setup_fake_aioredis(monkeypatch):
    fake_mod = types.SimpleNamespace(from_url=lambda url: FakeRedis())
    monkeypatch.setitem(sys.modules, "aioredis", fake_mod)


@pytest.mark.asyncio
async def test_set_get(monkeypatch):
    setup_fake_aioredis(monkeypatch)
    from core.providers.cache import RedisCacheProvider

    cache = RedisCacheProvider("redis://localhost")
    await cache.set("k", "v")
    assert await cache.get("k") == "v"


@pytest.mark.asyncio
async def test_delete(monkeypatch):
    setup_fake_aioredis(monkeypatch)
    from core.providers.cache import RedisCacheProvider

    cache = RedisCacheProvider("redis://localhost")
    await cache.set("k", "v")
    await cache.delete("k")
    assert await cache.get("k") is None


@pytest.mark.asyncio
async def test_clear_by_tag(monkeypatch):
    setup_fake_aioredis(monkeypatch)
    from core.providers.cache import RedisCacheProvider

    cache = RedisCacheProvider("redis://localhost")
    await cache.set("a", 1, tags=["t1"])
    await cache.set("b", 2, tags=["t1"])
    await cache.set("c", 3, tags=["t2"])

    removed = await cache.clear(tag="t1")
    assert removed == 2
    assert await cache.get("a") is None
    assert await cache.get("b") is None
    assert await cache.get("c") == 3


@pytest.mark.asyncio
async def test_stats(monkeypatch):
    setup_fake_aioredis(monkeypatch)
    from core.providers.cache import RedisCacheProvider

    cache = RedisCacheProvider("redis://localhost")
    await cache.set("a", 1)
    stats = await cache.stats()
    assert stats["entry_count"] == 1
    assert stats["type"] == "redis"
