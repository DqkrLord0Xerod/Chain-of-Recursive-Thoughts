"""Cache provider implementations with multiple backends."""

from __future__ import annotations

import asyncio
import hashlib
import json
import pickle
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union

import aiofiles
import structlog

logger = structlog.get_logger(__name__)


class CacheProvider(Protocol):
    """Enhanced cache provider interface."""
    
    async def get(self, key: str) -> Optional[Any]:
        """Retrieve value from cache."""
        ...
        
    async def set(
        self,
        key: str,
        value: Any,
        *,
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Store value in cache with optional TTL and tags."""
        ...
        
    async def delete(self, key: str) -> None:
        """Remove value from cache."""
        ...
        
    async def clear(self, *, tag: Optional[str] = None) -> int:
        """Clear cache, optionally by tag. Returns number of entries cleared."""
        ...
        
    async def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        ...


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    value: Any
    created_at: float
    accessed_at: float
    access_count: int = 0
    ttl: Optional[int] = None
    tags: List[str] = field(default_factory=list)
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        return time.time() > self.created_at + self.ttl


class InMemoryLRUCache:
    """Thread-safe LRU cache with TTL and tagging support."""
    
    def __init__(self, max_size: int = 1000) -> None:
        self.max_size = max_size
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        async with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._misses += 1
                return None
                
            if entry.is_expired():
                del self._cache[key]
                self._misses += 1
                return None
                
            # Update LRU order
            self._cache.move_to_end(key)
            entry.accessed_at = time.time()
            entry.access_count += 1
            self._hits += 1
            
            return entry.value
            
    async def set(
        self,
        key: str,
        value: Any,
        *,
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Set value in cache."""
        async with self._lock:
            now = time.time()
            
            # Remove existing entry if present
            if key in self._cache:
                del self._cache[key]
                
            # Add new entry
            self._cache[key] = CacheEntry(
                value=value,
                created_at=now,
                accessed_at=now,
                ttl=ttl,
                tags=tags or [],
            )
            
            # Evict oldest entries if over capacity
            while len(self._cache) > self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._evictions += 1
                
    async def delete(self, key: str) -> None:
        """Delete entry from cache."""
        async with self._lock:
            self._cache.pop(key, None)
            
    async def clear(self, *, tag: Optional[str] = None) -> int:
        """Clear cache entries."""
        async with self._lock:
            if tag is None:
                count = len(self._cache)
                self._cache.clear()
                return count
                
            # Clear by tag
            to_delete = [
                k for k, v in self._cache.items()
                if tag in v.tags
            ]
            for key in to_delete:
                del self._cache[key]
                
            return len(to_delete)
            
    async def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        async with self._lock:
            total_entries = len(self._cache)
            expired_entries = sum(
                1 for e in self._cache.values() if e.is_expired()
            )
            
            return {
                "type": "in_memory_lru",
                "max_size": self.max_size,
                "current_size": total_entries,
                "expired_entries": expired_entries,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / (self._hits + self._misses) if (self._hits + self._misses) > 0 else 0,
                "evictions": self._evictions,
            }


class DiskCacheProvider:
    """Persistent disk-based cache with async I/O."""
    
    def __init__(
        self,
        cache_dir: Union[str, Path],
        *,
        max_size_mb: int = 1000,
        enable_compression: bool = True,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_mb = max_size_mb
        self.enable_compression = enable_compression
        self._index_file = self.cache_dir / ".cache_index.json"
        self._index: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._load_index_sync()
        
    def _load_index_sync(self) -> None:
        """Load cache index synchronously on init."""
        if self._index_file.exists():
            try:
                with open(self._index_file, 'r') as f:
                    self._index = json.load(f)
            except Exception as e:
                logger.warning("Failed to load cache index", error=str(e))
                self._index = {}
                
    async def _save_index(self) -> None:
        """Save cache index to disk."""
        async with aiofiles.open(self._index_file, 'w') as f:
            await f.write(json.dumps(self._index, indent=2))
            
    def _get_cache_path(self, key: str) -> Path:
        """Get file path for cache key."""
        # Use first 2 chars of hash for directory sharding
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        subdir = self.cache_dir / key_hash[:2]
        subdir.mkdir(exist_ok=True)
        return subdir / f"{key_hash}.pkl"
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache."""
        async with self._lock:
            metadata = self._index.get(key)
            if not metadata:
                return None
                
            # Check TTL
            if metadata.get("ttl"):
                if time.time() > metadata["created_at"] + metadata["ttl"]:
                    await self.delete(key)
                    return None
                    
            cache_path = self._get_cache_path(key)
            if not cache_path.exists():
                # Index out of sync
                del self._index[key]
                await self._save_index()
                return None
                
            try:
                async with aiofiles.open(cache_path, 'rb') as f:
                    data = await f.read()
                    
                if self.enable_compression:
                    import zlib
                    data = zlib.decompress(data)
                    
                value = pickle.loads(data)
                
                # Update access metadata
                self._index[key]["accessed_at"] = time.time()
                self._index[key]["access_count"] = metadata.get("access_count", 0) + 1
                
                return value
                
            except Exception as e:
                logger.error("Failed to read cache entry", key=key, error=str(e))
                await self.delete(key)
                return None
                
    async def set(
        self,
        key: str,
        value: Any,
        *,
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Set value in disk cache."""
        async with self._lock:
            cache_path = self._get_cache_path(key)
            
            try:
                data = pickle.dumps(value)
                
                if self.enable_compression:
                    import zlib
                    data = zlib.compress(data)
                    
                async with aiofiles.open(cache_path, 'wb') as f:
                    await f.write(data)
                    
                # Update index
                now = time.time()
                self._index[key] = {
                    "created_at": now,
                    "accessed_at": now,
                    "access_count": 0,
                    "size": len(data),
                    "ttl": ttl,
                    "tags": tags or [],
                }
                
                await self._save_index()
                await self._enforce_size_limit()
                
            except Exception as e:
                logger.error("Failed to write cache entry", key=key, error=str(e))
                raise
                
    async def delete(self, key: str) -> None:
        """Delete entry from disk cache."""
        async with self._lock:
            if key not in self._index:
                return
                
            cache_path = self._get_cache_path(key)
            
            try:
                if cache_path.exists():
                    cache_path.unlink()
                del self._index[key]
                await self._save_index()
                
            except Exception as e:
                logger.error("Failed to delete cache entry", key=key, error=str(e))
                
    async def clear(self, *, tag: Optional[str] = None) -> int:
        """Clear disk cache."""
        async with self._lock:
            if tag is None:
                # Clear everything
                count = len(self._index)
                
                for key in list(self._index.keys()):
                    cache_path = self._get_cache_path(key)
                    try:
                        if cache_path.exists():
                            cache_path.unlink()
                    except Exception:
                        pass
                        
                self._index.clear()
                await self._save_index()
                return count
                
            # Clear by tag
            to_delete = [
                k for k, v in self._index.items()
                if tag in v.get("tags", [])
            ]
            
            for key in to_delete:
                await self.delete(key)
                
            return len(to_delete)
            
    async def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        async with self._lock:
            total_size = sum(
                m.get("size", 0) for m in self._index.values()
            )
            
            return {
                "type": "disk",
                "cache_dir": str(self.cache_dir),
                "max_size_mb": self.max_size_mb,
                "current_size_mb": total_size / (1024 * 1024),
                "entry_count": len(self._index),
                "compression_enabled": self.enable_compression,
            }
            
    async def _enforce_size_limit(self) -> None:
        """Remove oldest entries if over size limit."""
        total_size = sum(m.get("size", 0) for m in self._index.values())
        max_size_bytes = self.max_size_mb * 1024 * 1024
        
        if total_size <= max_size_bytes:
            return
            
        # Sort by access time
        entries = sorted(
            self._index.items(),
            key=lambda x: x[1].get("accessed_at", 0)
        )
        
        # Remove oldest until under limit
        for key, metadata in entries:
            if total_size <= max_size_bytes:
                break
                
            await self.delete(key)
            total_size -= metadata.get("size", 0)


class HybridCacheProvider:
    """Two-tier cache with memory and disk layers."""
    
    def __init__(
        self,
        memory_cache: CacheProvider,
        disk_cache: CacheProvider,
        *,
        promotion_threshold: int = 3,  # Access count before promoting to memory
    ) -> None:
        self.memory_cache = memory_cache
        self.disk_cache = disk_cache
        self.promotion_threshold = promotion_threshold
        
    async def get(self, key: str) -> Optional[Any]:
        """Get from memory first, then disk."""
        # Try memory cache first
        value = await self.memory_cache.get(key)
        if value is not None:
            return value
            
        # Try disk cache
        value = await self.disk_cache.get(key)
        if value is not None:
            # Check if should promote to memory
            disk_stats = await self.disk_cache.stats()
            if key in disk_stats.get("index", {}):
                access_count = disk_stats["index"][key].get("access_count", 0)
                if access_count >= self.promotion_threshold:
                    # Promote to memory cache
                    await self.memory_cache.set(key, value)
                    
        return value
        
    async def set(
        self,
        key: str,
        value: Any,
        *,
        ttl: Optional[int] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Set in both caches."""
        # Always set in disk cache
        await self.disk_cache.set(key, value, ttl=ttl, tags=tags)
        
        # Set in memory cache if it's a "hot" key
        # (This could be based on various heuristics)
        await self.memory_cache.set(key, value, ttl=ttl, tags=tags)
        
    async def delete(self, key: str) -> None:
        """Delete from both caches."""
        await asyncio.gather(
            self.memory_cache.delete(key),
            self.disk_cache.delete(key),
        )
        
    async def clear(self, *, tag: Optional[str] = None) -> int:
        """Clear both caches."""
        results = await asyncio.gather(
            self.memory_cache.clear(tag=tag),
            self.disk_cache.clear(tag=tag),
        )
        return sum(results)
        
    async def stats(self) -> Dict[str, Any]:
        """Get combined statistics."""
        memory_stats, disk_stats = await asyncio.gather(
            self.memory_cache.stats(),
            self.disk_cache.stats(),
        )
        
        return {
            "type": "hybrid",
            "memory": memory_stats,
            "disk": disk_stats,
            "promotion_threshold": self.promotion_threshold,
        }


# Redis cache provider would go here for production use
class RedisCacheProvider:
    """Redis-based cache provider (placeholder for full implementation)."""
    
    def __init__(self, redis_url: str, **kwargs):
        # Would use aioredis in production
        raise NotImplementedError("Redis cache provider requires aioredis")
