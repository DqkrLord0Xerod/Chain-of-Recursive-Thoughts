"""Embedding provider implementations."""

from __future__ import annotations

import hashlib
from typing import Dict, List, Optional

import numpy as np
from api import openrouter
from core.interfaces import EmbeddingProvider as IEmbeddingProvider


class OpenRouterEmbeddingProvider(IEmbeddingProvider):
    """Embedding provider using OpenRouter API."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "openai/text-embedding-ada-002",
        cache_embeddings: bool = True,
    ):
        self.api_key = api_key
        self.model = model
        self.cache_embeddings = cache_embeddings
        self._cache: Dict[str, List[float]] = {}
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts."""
        if not texts:
            return []
            
        # Check cache
        if self.cache_embeddings:
            embeddings = []
            uncached_texts = []
            uncached_indices = []
            
            for i, text in enumerate(texts):
                cache_key = self._get_cache_key(text)
                if cache_key in self._cache:
                    embeddings.append(self._cache[cache_key])
                else:
                    embeddings.append(None)
                    uncached_texts.append(text)
                    uncached_indices.append(i)
                    
            # Generate embeddings for uncached texts
            if uncached_texts:
                new_embeddings = openrouter.get_embeddings(
                    self.headers,
                    uncached_texts,
                )
                
                # Update cache and results
                for idx, embedding in zip(uncached_indices, new_embeddings):
                    embeddings[idx] = embedding
                    if self.cache_embeddings:
                        cache_key = self._get_cache_key(texts[idx])
                        self._cache[cache_key] = embedding
                        
            return embeddings
        else:
            # No caching, generate all
            return openrouter.get_embeddings(self.headers, texts)
            
    async def similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between texts."""
        embeddings = await self.embed([text1, text2])
        
        if len(embeddings) < 2:
            return 0.0
            
        # Cosine similarity
        emb1, emb2 = embeddings[0], embeddings[1]
        
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
        
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.sha256(f"{self.model}:{text}".encode()).hexdigest()


class CachedEmbeddingProvider(IEmbeddingProvider):
    """Wrapper that adds persistent caching to any embedding provider."""
    
    def __init__(
        self,
        base_provider: IEmbeddingProvider,
        cache_provider,  # CacheProvider
    ):
        self.base_provider = base_provider
        self.cache = cache_provider
        
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings with caching."""
        results = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache
        for i, text in enumerate(texts):
            cache_key = f"embedding:{self._text_hash(text)}"
            cached = await self.cache.get(cache_key)
            
            if cached is not None:
                results.append(cached)
            else:
                results.append(None)
                uncached_texts.append(text)
                uncached_indices.append(i)
                
        # Generate missing embeddings
        if uncached_texts:
            new_embeddings = await self.base_provider.embed(uncached_texts)
            
            # Update results and cache
            for idx, embedding in zip(uncached_indices, new_embeddings):
                results[idx] = embedding
                
                # Cache with long TTL
                cache_key = f"embedding:{self._text_hash(texts[idx])}"
                await self.cache.set(
                    cache_key,
                    embedding,
                    ttl=86400 * 7,  # 7 days
                    tags=["embeddings"],
                )
                
        return results
        
    async def similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity with caching."""
        # Check if we have cached similarity
        sim_key = f"similarity:{self._text_hash(text1)}:{self._text_hash(text2)}"
        cached_sim = await self.cache.get(sim_key)
        
        if cached_sim is not None:
            return cached_sim
            
        # Calculate similarity
        similarity = await self.base_provider.similarity(text1, text2)
        
        # Cache result
        await self.cache.set(
            sim_key,
            similarity,
            ttl=86400,  # 1 day
            tags=["similarity"],
        )
        
        return similarity
        
    def _text_hash(self, text: str) -> str:
        """Generate hash for text."""
        return hashlib.md5(text.encode()).hexdigest()


class LocalEmbeddingProvider(IEmbeddingProvider):
    """Local embedding provider using sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize with a sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except ImportError:
            raise ImportError(
                "sentence-transformers required for local embeddings. "
                "Install with: pip install sentence-transformers"
            )
            
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings locally."""
        if not texts:
            return []
            
        # Run in thread pool to avoid blocking
        import asyncio
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self.model.encode(texts).tolist()
        )
        
        return embeddings
        
    async def similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity locally."""
        embeddings = await self.embed([text1, text2])
        
        if len(embeddings) < 2:
            return 0.0
            
        # Cosine similarity
        emb1 = np.array(embeddings[0])
        emb2 = np.array(embeddings[1])
        
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
