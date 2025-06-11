import pytest

from core.providers.cache import InMemoryLRUCache, DiskCacheProvider


@pytest.mark.asyncio
async def test_memory_lru():
    cache = InMemoryLRUCache(max_size=2)
    await cache.set("a", "x")
    await cache.set("b", "y")
    await cache.set("c", "z")
    assert await cache.get("a") is None
    assert await cache.get("b") == "y"


@pytest.mark.asyncio
async def test_disk_persistence(tmp_path):
    cache_dir = tmp_path / "cache"
    cache1 = DiskCacheProvider(str(cache_dir))
    await cache1.set("k", "v")

    cache2 = DiskCacheProvider(str(cache_dir))
    assert await cache2.get("k") == "v"
