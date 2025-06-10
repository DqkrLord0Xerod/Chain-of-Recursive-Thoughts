"""Resilient LLM provider combining circuit breaker, retry, and hedging."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

import structlog

from core.providers.llm import LLMProvider, LLMResponse
from core.resilience.circuit_breaker import CircuitBreaker, CircuitOpenError
from core.resilience.retry_policies import (
    ExponentialBackoffPolicy,
    RetryExecutor,
    HedgingExecutor,
    CombinedExecutor,
)
from exceptions import APIError, RateLimitError, TokenLimitError


logger = structlog.get_logger(__name__)


class ResilientLLMProvider:
    """
    LLM provider with comprehensive resilience patterns.
    
    Features:
    - Circuit breaker per provider
    - Exponential backoff retry
    - Request hedging for low latency
    - Automatic failover between providers
    - Detailed metrics and monitoring
    """
    
    def __init__(
        self,
        providers: List[LLMProvider],
        *,
        # Circuit breaker config
        circuit_failure_threshold: int = 5,
        circuit_timeout: float = 60.0,
        circuit_half_open_limit: int = 2,
        # Retry config
        max_retries: int = 3,
        retry_initial_delay: float = 1.0,
        retry_max_delay: float = 30.0,
        # Hedging config
        enable_hedging: bool = True,
        hedge_delay: float = 0.5,
        max_hedges: int = 2,
        # Monitoring
        enable_metrics: bool = True,
    ):
        if not providers:
            raise ValueError("At least one provider required")
            
        self.providers = providers
        self.enable_hedging = enable_hedging
        self.enable_metrics = enable_metrics
        
        # Create circuit breakers for each provider
        self.circuit_breakers: List[CircuitBreaker] = []
        for i, provider in enumerate(providers):
            cb = CircuitBreaker(
                name=f"provider_{i}_{getattr(provider, 'model', 'unknown')}",
                failure_threshold=circuit_failure_threshold,
                timeout=circuit_timeout,
                half_open_limit=circuit_half_open_limit,
                error_handler=self._is_circuit_breaking_error,
            )
            self.circuit_breakers.append(cb)
            
        # Create retry policy
        retry_policy = ExponentialBackoffPolicy(
            initial_delay=retry_initial_delay,
            max_delay=retry_max_delay,
            jitter=True,
            retryable_exceptions=[APIError, RateLimitError, asyncio.TimeoutError],
        )
        
        # Create executors
        self.retry_executor = RetryExecutor(retry_policy, max_retries)
        self.hedging_executor = HedgingExecutor(
            initial_delay=hedge_delay,
            max_hedges=max_hedges,
        ) if enable_hedging else None
        
        self.combined_executor = CombinedExecutor(
            self.retry_executor,
            self.hedging_executor,
        )
        
        # Metrics
        self._request_count = 0
        self._success_count = 0
        self._failure_count = 0
        self._total_latency = 0.0
        self._provider_usage = {i: 0 for i in range(len(providers))}
        
    def _is_circuit_breaking_error(self, error: Exception) -> bool:
        """Determine if error should trip circuit breaker."""
        # Don't trip on token limits (user error)
        if isinstance(error, TokenLimitError):
            return False
            
        # Trip on infrastructure errors
        if isinstance(error, (APIError, RateLimitError, asyncio.TimeoutError)):
            return True
            
        # Trip on connection errors
        if isinstance(error, (ConnectionError, OSError)):
            return True
            
        return False
        
    async def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        metadata: Optional[Dict] = None,
        timeout: float = 30.0,
        prefer_provider: Optional[int] = None,
    ) -> LLMResponse:
        """
        Send chat request with full resilience.
        
        Args:
            messages: Chat messages
            temperature: Sampling temperature
            max_tokens: Max response tokens
            metadata: Request metadata
            timeout: Request timeout
            prefer_provider: Preferred provider index
            
        Returns:
            LLM response
            
        Raises:
            APIError: If all providers fail
        """
        start_time = time.time()
        self._request_count += 1
        request_id = f"req_{self._request_count}"
        
        logger.info(
            "resilient_llm_request",
            request_id=request_id,
            message_count=len(messages),
            prefer_provider=prefer_provider,
            metadata=metadata,
        )
        
        # Order providers based on preference and circuit state
        provider_order = self._get_provider_order(prefer_provider)
        
        errors = []
        
        for provider_idx in provider_order:
            provider = self.providers[provider_idx]
            circuit_breaker = self.circuit_breakers[provider_idx]
            
            try:
                # Execute through circuit breaker
                async def provider_call():
                    return await self._call_provider_with_timeout(
                        provider,
                        messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        metadata=metadata,
                        timeout=timeout,
                    )
                    
                # Use combined executor for retry + hedging
                response = await self.combined_executor.execute(
                    lambda: circuit_breaker.call(provider_call),
                    use_hedging=self.enable_hedging and provider_idx == provider_order[0],
                )
                
                # Success!
                latency = time.time() - start_time
                self._record_success(provider_idx, latency)
                
                logger.info(
                    "resilient_llm_success",
                    request_id=request_id,
                    provider_index=provider_idx,
                    latency=latency,
                    retries=self.retry_executor.max_attempts,
                )
                
                return response
                
            except CircuitOpenError as e:
                errors.append((provider_idx, f"Circuit open: {e}"))
                logger.warning(
                    "resilient_llm_circuit_open",
                    request_id=request_id,
                    provider_index=provider_idx,
                )
                continue
                
            except Exception as e:
                errors.append((provider_idx, str(e)))
                logger.warning(
                    "resilient_llm_provider_failed",
                    request_id=request_id,
                    provider_index=provider_idx,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                
                # Don't try next provider for non-retryable errors
                if isinstance(e, TokenLimitError):
                    self._failure_count += 1
                    raise
                    
        # All providers failed
        self._failure_count += 1
        error_summary = "; ".join(f"Provider {i}: {e}" for i, e in errors)
        
        logger.error(
            "resilient_llm_all_failed",
            request_id=request_id,
            errors=errors,
        )
        
        raise APIError(f"All providers failed: {error_summary}")
        
    async def _call_provider_with_timeout(
        self,
        provider: LLMProvider,
        messages: List[Dict[str, str]],
        *,
        temperature: float,
        max_tokens: Optional[int],
        metadata: Optional[Dict],
        timeout: float,
    ) -> LLMResponse:
        """Call provider with timeout."""
        try:
            return await asyncio.wait_for(
                provider.chat(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    metadata=metadata,
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            raise APIError(f"Request timed out after {timeout}s")
            
    def _get_provider_order(self, prefer_provider: Optional[int]) -> List[int]:
        """Get provider order based on preference and circuit states."""
        indices = list(range(len(self.providers)))
        
        # Sort by circuit state and usage
        def sort_key(idx: int) -> tuple:
            cb = self.circuit_breakers[idx]
            # Prioritize: closed > half_open > open
            state_priority = {
                "closed": 0,
                "half_open": 1,
                "open": 2,
            }[cb.state.value]
            
            # Secondary sort by usage (load balancing)
            usage = self._provider_usage.get(idx, 0)
            
            return (state_priority, usage)
            
        indices.sort(key=sort_key)
        
        # Move preferred provider to front if specified and available
        if prefer_provider is not None and 0 <= prefer_provider < len(self.providers):
            if self.circuit_breakers[prefer_provider].state.value != "open":
                indices.remove(prefer_provider)
                indices.insert(0, prefer_provider)
                
        return indices
        
    def _record_success(self, provider_idx: int, latency: float) -> None:
        """Record successful request metrics."""
        self._success_count += 1
        self._total_latency += latency
        self._provider_usage[provider_idx] += 1
        
    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ):
        """Stream chat responses with resilience."""
        # For streaming, we can't use hedging effectively
        # Just use the first available provider with retry
        
        provider_order = self._get_provider_order(None)
        
        for provider_idx in provider_order:
            provider = self.providers[provider_idx]
            circuit_breaker = self.circuit_breakers[provider_idx]
            
            try:
                async def stream_call():
                    async for chunk in provider.stream_chat(
                        messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    ):
                        yield chunk
                        
                # Stream through circuit breaker
                async for chunk in circuit_breaker.call(stream_call):
                    yield chunk
                    
                return
                
            except (CircuitOpenError, Exception) as e:
                logger.warning(
                    "resilient_llm_stream_failed",
                    provider_index=provider_idx,
                    error=str(e),
                )
                continue
                
        raise APIError("All providers failed for streaming")
        
    def get_metrics(self) -> Dict:
        """Get resilience metrics."""
        metrics = {
            "requests": {
                "total": self._request_count,
                "successful": self._success_count,
                "failed": self._failure_count,
                "success_rate": (
                    self._success_count / self._request_count
                    if self._request_count > 0 else 0
                ),
            },
            "latency": {
                "average": (
                    self._total_latency / self._success_count
                    if self._success_count > 0 else 0
                ),
            },
            "provider_usage": self._provider_usage,
            "circuit_breakers": {
                i: cb.get_status()
                for i, cb in enumerate(self.circuit_breakers)
            },
        }
        
        return metrics
        
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all providers."""
        health = {
            "healthy": True,
            "providers": [],
            "timestamp": time.time(),
        }
        
        test_messages = [{"role": "user", "content": "Hello"}]
        
        for i, provider in enumerate(self.providers):
            cb = self.circuit_breakers[i]
            provider_health = {
                "index": i,
                "model": getattr(provider, "model", "unknown"),
                "circuit_state": cb.state.value,
                "circuit_stats": cb.stats.__dict__,
                "healthy": False,
                "error": None,
                "latency": None,
            }
            
            if cb.state.value != "open":
                try:
                    start = time.time()
                    await provider.chat(test_messages, temperature=0.1)
                    provider_health["healthy"] = True
                    provider_health["latency"] = time.time() - start
                except Exception as e:
                    provider_health["error"] = str(e)
                    health["healthy"] = False
            else:
                provider_health["error"] = "Circuit breaker open"
                health["healthy"] = False
                
            health["providers"].append(provider_health)
            
        return health
