"""Advanced retry policies and hedging strategies."""

from __future__ import annotations

import asyncio
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Awaitable, Callable, List, Optional, Type, TypeVar, Union

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar('T')


@dataclass
class RetryContext:
    """Context information for retry attempts."""
    attempt: int
    total_attempts: int
    elapsed_time: float
    last_exception: Optional[Exception] = None
    
    @property
    def is_last_attempt(self) -> bool:
        """Check if this is the last retry attempt."""
        return self.attempt >= self.total_attempts


class RetryPolicy(ABC):
    """Abstract base class for retry policies."""
    
    @abstractmethod
    def should_retry(self, exception: Exception, context: RetryContext) -> bool:
        """Determine if request should be retried."""
        pass
        
    @abstractmethod
    def get_delay(self, context: RetryContext) -> float:
        """Get delay before next retry in seconds."""
        pass


class ExponentialBackoffPolicy(RetryPolicy):
    """Exponential backoff with jitter."""
    
    def __init__(
        self,
        *,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Optional[List[Type[Exception]]] = None,
    ):
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions or [Exception]
        
    def should_retry(self, exception: Exception, context: RetryContext) -> bool:
        """Check if exception is retryable."""
        if context.is_last_attempt:
            return False
            
        return any(
            isinstance(exception, exc_type)
            for exc_type in self.retryable_exceptions
        )
        
    def get_delay(self, context: RetryContext) -> float:
        """Calculate exponential backoff delay."""
        delay = min(
            self.initial_delay * (self.exponential_base ** (context.attempt - 1)),
            self.max_delay
        )
        
        if self.jitter:
            # Add random jitter (±25%)
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)
            
        return max(0, delay)


class LinearBackoffPolicy(RetryPolicy):
    """Linear backoff policy."""
    
    def __init__(
        self,
        *,
        delay_increment: float = 1.0,
        max_delay: float = 30.0,
        retryable_exceptions: Optional[List[Type[Exception]]] = None,
    ):
        self.delay_increment = delay_increment
        self.max_delay = max_delay
        self.retryable_exceptions = retryable_exceptions or [Exception]
        
    def should_retry(self, exception: Exception, context: RetryContext) -> bool:
        """Check if exception is retryable."""
        if context.is_last_attempt:
            return False
            
        return any(
            isinstance(exception, exc_type)
            for exc_type in self.retryable_exceptions
        )
        
    def get_delay(self, context: RetryContext) -> float:
        """Calculate linear backoff delay."""
        return min(
            self.delay_increment * context.attempt,
            self.max_delay
        )


class AdaptiveRetryPolicy(RetryPolicy):
    """Adaptive retry policy based on error patterns."""
    
    def __init__(
        self,
        *,
        initial_delay: float = 1.0,
        retryable_exceptions: Optional[List[Type[Exception]]] = None,
    ):
        self.initial_delay = initial_delay
        self.retryable_exceptions = retryable_exceptions or [Exception]
        self._error_history: List[tuple[float, Exception]] = []
        self._success_rate = 1.0
        
    def should_retry(self, exception: Exception, context: RetryContext) -> bool:
        """Adaptively determine if we should retry."""
        if context.is_last_attempt:
            return False
            
        # Record error
        self._error_history.append((time.time(), exception))
        
        # Clean old errors (older than 5 minutes)
        cutoff = time.time() - 300
        self._error_history = [
            (t, e) for t, e in self._error_history if t > cutoff
        ]
        
        # Calculate recent error rate
        if len(self._error_history) > 10:
            # High error rate - be more conservative
            self._success_rate = max(0.1, self._success_rate * 0.9)
        else:
            # Low error rate - be more aggressive
            self._success_rate = min(1.0, self._success_rate * 1.1)
            
        # Retry if exception is retryable and success rate is reasonable
        is_retryable = any(
            isinstance(exception, exc_type)
            for exc_type in self.retryable_exceptions
        )
        
        return is_retryable and self._success_rate > 0.2
        
    def get_delay(self, context: RetryContext) -> float:
        """Adaptive delay based on success rate."""
        base_delay = self.initial_delay * (2 ** (context.attempt - 1))
        
        # Adjust based on success rate
        adjusted_delay = base_delay / self._success_rate
        
        # Cap at reasonable maximum
        return min(adjusted_delay, 120.0)


class RetryExecutor:
    """Execute functions with retry logic."""
    
    def __init__(self, policy: RetryPolicy, max_attempts: int = 3):
        self.policy = policy
        self.max_attempts = max_attempts
        
    async def execute(
        self,
        func: Callable[..., Awaitable[T]],
        *args,
        on_retry: Optional[Callable[[RetryContext], None]] = None,
        **kwargs
    ) -> T:
        """Execute function with retries."""
        start_time = time.time()
        last_exception = None
        
        for attempt in range(1, self.max_attempts + 1):
            context = RetryContext(
                attempt=attempt,
                total_attempts=self.max_attempts,
                elapsed_time=time.time() - start_time,
                last_exception=last_exception,
            )
            
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                context.last_exception = e
                
                if not self.policy.should_retry(e, context):
                    logger.warning(
                        "retry_policy_exhausted",
                        attempt=attempt,
                        error=str(e),
                        elapsed_time=context.elapsed_time,
                    )
                    raise
                    
                delay = self.policy.get_delay(context)
                
                logger.info(
                    "retry_attempt",
                    attempt=attempt,
                    max_attempts=self.max_attempts,
                    delay=delay,
                    error=str(e),
                )
                
                if on_retry:
                    on_retry(context)
                    
                await asyncio.sleep(delay)
                
        # Should not reach here, but just in case
        raise last_exception or Exception("Retry failed")


class HedgingExecutor:
    """Execute requests with hedging (parallel attempts)."""
    
    def __init__(
        self,
        *,
        initial_delay: float = 0.5,
        max_hedges: int = 2,
        hedge_delay_multiplier: float = 2.0,
    ):
        self.initial_delay = initial_delay
        self.max_hedges = max_hedges
        self.hedge_delay_multiplier = hedge_delay_multiplier
        
    async def execute(
        self,
        func: Callable[..., Awaitable[T]],
        *args,
        **kwargs
    ) -> T:
        """
        Execute function with hedged requests.
        
        Starts additional parallel requests if the initial request
        is taking too long, returns the first successful response.
        """
        tasks: List[asyncio.Task] = []
        exceptions: List[Exception] = []
        
        async def wrapper(delay: float) -> T:
            """Wrapper to add delay before execution."""
            if delay > 0:
                await asyncio.sleep(delay)
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                exceptions.append(e)
                raise
                
        # Start hedged requests with increasing delays
        delay = 0.0
        for i in range(self.max_hedges + 1):
            task = asyncio.create_task(wrapper(delay))
            tasks.append(task)
            
            if i < self.max_hedges:
                delay = self.initial_delay * (self.hedge_delay_multiplier ** i)
                
        try:
            # Return first successful response
            done, pending = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            
            # Cancel pending tasks
            for task in pending:
                task.cancel()
                
            # Get the result
            for task in done:
                if not task.cancelled() and task.exception() is None:
                    result = task.result()
                    
                    logger.info(
                        "hedging_success",
                        successful_hedge=tasks.index(task),
                        total_hedges=len(tasks) - 1,
                    )
                    
                    return result
                    
            # All tasks failed
            if exceptions:
                raise exceptions[0]
            else:
                raise Exception("All hedged requests failed")
                
        finally:
            # Ensure all tasks are cleaned up
            for task in tasks:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass


class CombinedExecutor:
    """Combine retry and hedging strategies."""
    
    def __init__(
        self,
        retry_executor: RetryExecutor,
        hedging_executor: Optional[HedgingExecutor] = None,
    ):
        self.retry_executor = retry_executor
        self.hedging_executor = hedging_executor
        
    async def execute(
        self,
        func: Callable[..., Awaitable[T]],
        *args,
        use_hedging: bool = True,
        **kwargs
    ) -> T:
        """Execute with combined retry and hedging."""
        if use_hedging and self.hedging_executor:
            # Wrap the hedged execution with retry
            async def hedged_func():
                return await self.hedging_executor.execute(func, *args, **kwargs)
                
            return await self.retry_executor.execute(hedged_func)
        else:
            # Just use retry
            return await self.retry_executor.execute(func, *args, **kwargs)