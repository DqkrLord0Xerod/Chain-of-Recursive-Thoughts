import sys
import types
import asyncio

import pytest
import importlib.util
import os

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)


# Fake aioredis module used for testing
class FakeRedis:
    def __init__(self):
        self.store = {}
        self.sets = {}
        self.ttl = {}
        self.expire_calls = []

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
        self.expire_calls.append((name, ttl))
        return True

    async def ping(self):
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
    class FakeConnectionPool:
        def __init__(self, url):
            self.url = url

        @classmethod
        def from_url(cls, url):
            return cls(url)

    class FakeRedisWrapper(FakeRedis):
        def __init__(self, connection_pool=None):
            super().__init__()
            self.connection_pool = connection_pool

    fake_mod = types.SimpleNamespace(
        ConnectionPool=FakeConnectionPool,
        Redis=FakeRedisWrapper,
        from_url=lambda url: FakeRedisWrapper(),
    )
    monkeypatch.setitem(sys.modules, "aioredis", fake_mod)


def get_provider_class():
    path = os.path.join(ROOT, "core", "providers", "cache.py")
    name = "cache"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod.RedisCacheProvider


@pytest.mark.asyncio
async def test_set_get(monkeypatch):
    setup_fake_aioredis(monkeypatch)
    RedisCacheProvider = get_provider_class()

    cache = RedisCacheProvider("redis://localhost")
    await cache.set("k", "v")
    assert await cache.get("k") == "v"


@pytest.mark.asyncio
async def test_delete(monkeypatch):
    setup_fake_aioredis(monkeypatch)
    RedisCacheProvider = get_provider_class()

    cache = RedisCacheProvider("redis://localhost")
    await cache.set("k", "v")
    await cache.delete("k")
    assert await cache.get("k") is None


@pytest.mark.asyncio
async def test_clear_by_tag(monkeypatch):
    setup_fake_aioredis(monkeypatch)
    RedisCacheProvider = get_provider_class()

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
    RedisCacheProvider = get_provider_class()

    cache = RedisCacheProvider("redis://localhost")
    await cache.set("a", 1)
    stats = await cache.stats()
    assert stats["entry_count"] == 1
    assert stats["type"] == "redis"


@pytest.mark.asyncio
async def test_ping(monkeypatch):
    setup_fake_aioredis(monkeypatch)
    RedisCacheProvider = get_provider_class()

    cache = RedisCacheProvider("redis://localhost")
    assert await cache.ping() is True


@pytest.mark.asyncio
async def test_tag_ttl_propagation(monkeypatch):
    setup_fake_aioredis(monkeypatch)
    RedisCacheProvider = get_provider_class()

    cache = RedisCacheProvider("redis://localhost")
    await cache.set("k", "v", ttl=10, tags=["t1"])

    key_tags = cache._key_tags("k")
    tag_set = cache._tag_set("t1")
    assert (key_tags, 10) in cache.redis.expire_calls
    assert (tag_set, 10) in cache.redis.expire_calls


def test_connection_pool(monkeypatch):
    setup_fake_aioredis(monkeypatch)
    RedisCacheProvider = get_provider_class()

    cache = RedisCacheProvider("redis://localhost")
    assert cache.pool.url == "redis://localhost"
    assert cache.redis.connection_pool == cache.pool
