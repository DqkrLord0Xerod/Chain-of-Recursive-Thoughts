"""LLM Provider implementations with retry logic and fallbacks."""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol

import aiohttp
import openai
import structlog
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from api import openrouter
from config import settings
from exceptions import APIError, RateLimitError, TokenLimitError


logger = structlog.get_logger(__name__)


class LLMResponse(Protocol):
    """Protocol for LLM responses."""
    content: str
    usage: Dict[str, int]
    model: str
    cached: bool


@dataclass
class StandardLLMResponse:
    """Standard implementation of LLMResponse."""
    content: str
    usage: Dict[str, int]
    model: str
    cached: bool = False


class LLMProvider(Protocol):
    """Enhanced LLM provider interface."""
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        metadata: Optional[Dict] = None,
    ) -> LLMResponse:
        """Send chat request with enhanced response."""
        ...
    
    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ):
        """Stream chat responses."""
        ...


class OpenRouterLLMProvider:
    """Production-ready OpenRouter provider with retry logic."""
    
    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        max_retries: int = 3,
        timeout: float = 30.0,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self._session = session
        self._owned_session = session is None
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": settings.frontend_url,
            "X-Title": "CoRT Enhanced",
            "Content-Type": "application/json",
        }
        
    async def __aenter__(self):
        if self._owned_session:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._owned_session and self._session:
            await self._session.close()
    
    def _request_id(self, messages: List[Dict]) -> str:
        """Generate unique request ID for logging."""
        content = json.dumps(messages, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()[:8]
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((APIError, aiohttp.ClientError)),
        before_sleep=before_sleep_log(logger, "warning"),
    )
    async def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        metadata: Optional[Dict] = None,
    ) -> StandardLLMResponse:
        """Send chat request with comprehensive error handling."""
        
        request_id = self._request_id(messages)
        logger.info(
            "llm_request_start",
            request_id=request_id,
            model=self.model,
            message_count=len(messages),
            metadata=metadata,
        )
        
        if not self._session:
            self._session = aiohttp.ClientSession()
            
        try:
            response_text = await openrouter.async_chat_completion(
                self.headers,
                messages,
                self.model,
                temperature=temperature,
                session=self._session,
            )
            
            # Estimate token usage (actual usage should come from API)
            prompt_tokens = sum(len(m["content"].split()) * 1.3 for m in messages)
            completion_tokens = len(response_text.split()) * 1.3
            
            logger.info(
                "llm_request_success",
                request_id=request_id,
                prompt_tokens=int(prompt_tokens),
                completion_tokens=int(completion_tokens),
            )
            
            return StandardLLMResponse(
                content=response_text,
                usage={
                    "prompt_tokens": int(prompt_tokens),
                    "completion_tokens": int(completion_tokens),
                    "total_tokens": int(prompt_tokens + completion_tokens),
                },
                model=self.model,
                cached=False,
            )
            
        except RateLimitError as e:
            logger.error("llm_rate_limit", request_id=request_id, error=str(e))
            raise
        except TokenLimitError as e:
            logger.error("llm_token_limit", request_id=request_id, error=str(e))
            raise
        except Exception as e:
            logger.error(
                "llm_request_failed",
                request_id=request_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise APIError(f"LLM request failed: {e}") from e
    
    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ):
        """Stream chat responses with proper error handling."""
        # Implementation would handle streaming responses
        # For now, yield the complete response
        response = await self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        yield response.content


class OpenAILLMProvider:
    """LLM provider using OpenAI's official Python package."""

    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        max_retries: int = 3,
        timeout: float = 30.0,
        client: Optional[openai.AsyncOpenAI] = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self._client = client
        self._owned_client = client is None

    async def __aenter__(self):
        if self._owned_client:
            self._client = openai.AsyncOpenAI(api_key=self.api_key, timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._owned_client and self._client:
            await self._client.close()

    def _request_id(self, messages: List[Dict]) -> str:
        content = json.dumps(messages, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()[:8]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((APIError, openai.OpenAIError)),
        before_sleep=before_sleep_log(logger, "warning"),
    )
    async def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        metadata: Optional[Dict] = None,
    ) -> StandardLLMResponse:
        request_id = self._request_id(messages)
        logger.info(
            "llm_request_start",
            request_id=request_id,
            model=self.model,
            message_count=len(messages),
            metadata=metadata,
        )

        if not self._client:
            self._client = openai.AsyncOpenAI(api_key=self.api_key, timeout=self.timeout)

        try:
            resp = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            content = resp.choices[0].message.content.strip()
            usage = resp.usage or {}

            logger.info(
                "llm_request_success",
                request_id=request_id,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
            )

            return StandardLLMResponse(
                content=content,
                usage=usage,
                model=self.model,
                cached=False,
            )

        except openai.RateLimitError as e:
            logger.error("llm_rate_limit", request_id=request_id, error=str(e))
            raise RateLimitError(str(e))
        except openai.BadRequestError as e:
            if "token" in str(e).lower():
                logger.error("llm_token_limit", request_id=request_id, error=str(e))
                raise TokenLimitError(str(e))
            raise APIError(str(e))
        except openai.OpenAIError as e:
            logger.error(
                "llm_request_failed",
                request_id=request_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise APIError(str(e)) from e

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ):
        """Stream chat responses using OpenAI's streaming API."""
        if not self._client:
            self._client = openai.AsyncOpenAI(
                api_key=self.api_key,
                timeout=self.timeout,
            )

        try:
            stream = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content

        except openai.RateLimitError as e:
            raise RateLimitError(str(e))
        except openai.BadRequestError as e:
            if "token" in str(e).lower():
                raise TokenLimitError(str(e))
            raise APIError(str(e))
        except openai.OpenAIError as e:
            raise APIError(str(e)) from e


class MultiProviderLLM:
    """LLM provider with automatic fallback to multiple providers."""
    
    def __init__(
        self,
        providers: List[LLMProvider],
        *,
        selection_strategy: str = "round_robin",  # or "least_latency", "random"
    ) -> None:
        if not providers:
            raise ValueError("At least one provider required")
            
        self.providers = providers
        self.selection_strategy = selection_strategy
        self._current_index = 0
        self._latencies: Dict[int, List[float]] = {i: [] for i in range(len(providers))}
        
    async def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        metadata: Optional[Dict] = None,
    ) -> LLMResponse:
        """Try providers in order until one succeeds."""
        
        errors = []
        start_idx = self._select_provider()
        
        for i in range(len(self.providers)):
            provider_idx = (start_idx + i) % len(self.providers)
            provider = self.providers[provider_idx]
            
            try:
                import time
                start_time = time.time()
                
                response = await provider.chat(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    metadata=metadata,
                )
                
                # Track latency for adaptive selection
                latency = time.time() - start_time
                self._latencies[provider_idx].append(latency)
                if len(self._latencies[provider_idx]) > 100:
                    self._latencies[provider_idx].pop(0)
                    
                logger.info(
                    "multi_provider_success",
                    provider_index=provider_idx,
                    latency=latency,
                    attempt=i + 1,
                )
                
                return response
                
            except Exception as e:
                errors.append((provider_idx, str(e)))
                logger.warning(
                    "multi_provider_fallback",
                    provider_index=provider_idx,
                    error=str(e),
                    attempt=i + 1,
                )
                
                # Special handling for rate limits
                if isinstance(e, RateLimitError) and i < len(self.providers) - 1:
                    await asyncio.sleep(2 ** i)  # Exponential backoff
                    
        # All providers failed
        error_summary = "; ".join(f"Provider {i}: {e}" for i, e in errors)
        raise APIError(f"All providers failed: {error_summary}")
    
    def _select_provider(self) -> int:
        """Select next provider based on strategy."""
        if self.selection_strategy == "round_robin":
            idx = self._current_index
            self._current_index = (self._current_index + 1) % len(self.providers)
            return idx
            
        elif self.selection_strategy == "random":
            return random.randint(0, len(self.providers) - 1)
            
        elif self.selection_strategy == "least_latency":
            # Select provider with lowest average latency
            avg_latencies = []
            for i, latencies in self._latencies.items():
                if latencies:
                    avg_latencies.append((i, sum(latencies) / len(latencies)))
                else:
                    avg_latencies.append((i, float('inf')))
                    
            return min(avg_latencies, key=lambda x: x[1])[0]
            
        return 0
    
    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ):
        """Stream from the first available provider."""
        # Similar fallback logic for streaming
        response = await self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        yield response.content
