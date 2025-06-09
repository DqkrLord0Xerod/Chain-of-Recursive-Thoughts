"""Enhanced interfaces for dependency injection and clean architecture."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Protocol, runtime_checkable


@dataclass
class Message:
    """Chat message with role and content."""
    role: str
    content: str
    metadata: Dict[str, Any] = None


@dataclass
class LLMResponse:
    """Standardized LLM response with metadata."""
    content: str
    model: str
    usage: Dict[str, int]
    cached: bool = False
    latency: float = 0.0
    metadata: Dict[str, Any] = None


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for large language model providers."""
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        metadata: Optional[Dict] = None,
    ) -> LLMResponse:
        """Send chat completion request and return response."""
        ...
    
    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """Stream chat responses."""
        ...


@runtime_checkable
class CacheProvider(Protocol):
    """Protocol for caching backends with advanced features."""
    
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
        """Clear cache, optionally by tag."""
        ...
        
    async def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        ...


@runtime_checkable
class QualityEvaluator(Protocol):
    """Protocol for evaluating response quality."""
    
    def score(self, response: str, prompt: str) -> float:
        """Return quality score between 0 and 1."""
        ...
        
    def detailed_score(self, response: str, prompt: str) -> Dict[str, float]:
        """Return detailed quality metrics."""
        ...


class EmbeddingProvider(Protocol):
    """Protocol for embedding generation."""
    
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts."""
        ...
        
    async def similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between texts."""
        ...


class ContextCompressor(Protocol):
    """Protocol for context compression."""
    
    async def compress(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
    ) -> List[Dict[str, str]]:
        """Compress messages to fit token limit."""
        ...


class MetricsRecorder(Protocol):
    """Protocol for recording metrics."""
    
    def record_run(
        self,
        *,
        processing_time: float,
        token_usage: int,
        num_rounds: int,
        convergence_reason: str,
        **kwargs
    ) -> None:
        """Record metrics for a thinking run."""
        ...
        
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        ...


class ThinkingStrategy(Protocol):
    """Protocol for thinking strategies."""
    
    async def determine_rounds(self, prompt: str) -> int:
        """Determine number of thinking rounds needed."""
        ...
        
    async def should_continue(
        self,
        rounds_completed: int,
        quality_scores: List[float],
        responses: List[str],
    ) -> tuple[bool, str]:
        """Determine if thinking should continue."""
        ...


class SecurityValidator(Protocol):
    """Protocol for security validation."""
    
    async def validate_request(
        self,
        *,
        api_key: Optional[str],
        session_id: Optional[str],
        prompt: str,
        context: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Validate incoming request."""
        ...


# Abstract base classes for extension

class BaseProvider(ABC):
    """Base class for providers with common functionality."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._initialized = False
        
    async def initialize(self) -> None:
        """Initialize provider resources."""
        if not self._initialized:
            await self._setup()
            self._initialized = True
            
    @abstractmethod
    async def _setup(self) -> None:
        """Setup provider-specific resources."""
        pass
        
    async def cleanup(self) -> None:
        """Cleanup provider resources."""
        if self._initialized:
            await self._teardown()
            self._initialized = False
            
    @abstractmethod
    async def _teardown(self) -> None:
        """Teardown provider-specific resources."""
        pass


class BaseLLMProvider(BaseProvider, LLMProvider):
    """Base implementation for LLM providers."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model = config.get("model", "unknown")
        self.default_temperature = config.get("default_temperature", 0.7)
        self.max_retries = config.get("max_retries", 3)
        
    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """Default stream implementation using regular chat."""
        response = await self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        yield response.content


class BaseCacheProvider(BaseProvider, CacheProvider):
    """Base implementation for cache providers."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.default_ttl = config.get("default_ttl", 3600)
        self.max_size = config.get("max_size", 10000)
        
    async def stats(self) -> Dict[str, Any]:
        """Default stats implementation."""
        return {
            "provider": self.__class__.__name__,
            "max_size": self.max_size,
            "default_ttl": self.default_ttl,
        }