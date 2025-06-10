"""Advanced circuit breaker implementation with half-open state."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, Optional, TypeVar, Generic, List

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitStats:
    """Circuit breaker statistics."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    last_failure_time: Optional[float] = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    state_changes: List[tuple[CircuitState, float]] = field(default_factory=list)
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        if self.total_calls == 0:
            return 0.0
        return self.failed_calls / self.total_calls
        
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls


class CircuitBreaker(Generic[T]):
    """Advanced circuit breaker with configurable thresholds."""
    
    def __init__(
        self,
        name: str,
        *,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout: float = 60.0,
        half_open_limit: int = 1,
        error_handler: Optional[Callable[[Exception], bool]] = None,
    ):
        """
        Initialize circuit breaker.
        
        Args:
            name: Circuit breaker name for logging
            failure_threshold: Failures before opening circuit
            success_threshold: Successes in half-open before closing
            timeout: Seconds before attempting half-open
            half_open_limit: Max concurrent requests in half-open state
            error_handler: Function to determine if error should trip circuit
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout
        self.half_open_limit = half_open_limit
        self.error_handler = error_handler or self._default_error_handler
        
        self._state = CircuitState.CLOSED
        self._stats = CircuitStats()
        self._half_open_semaphore = asyncio.Semaphore(half_open_limit)
        self._state_lock = asyncio.Lock()
        self._last_state_change = time.time()
        
    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state
        
    @property
    def stats(self) -> CircuitStats:
        """Get circuit statistics."""
        return self._stats
        
    def _default_error_handler(self, error: Exception) -> bool:
        """Default error handler - all errors trip the circuit."""
        return True
        
    async def _change_state(self, new_state: CircuitState) -> None:
        """Change circuit state with logging."""
        async with self._state_lock:
            if self._state == new_state:
                return
                
            old_state = self._state
            self._state = new_state
            self._last_state_change = time.time()
            self._stats.state_changes.append((new_state, self._last_state_change))
            
            logger.info(
                "circuit_breaker_state_change",
                name=self.name,
                old_state=old_state.value,
                new_state=new_state.value,
                stats={
                    "total_calls": self._stats.total_calls,
                    "failure_rate": self._stats.failure_rate,
                    "consecutive_failures": self._stats.consecutive_failures,
                }
            )
            
    async def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit."""
        return (
            self._state == CircuitState.OPEN and
            time.time() - self._last_state_change >= self.timeout
        )
        
    async def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Call function through circuit breaker.
        
        Raises:
            CircuitOpenError: If circuit is open
            Original exception: If function fails
        """
        # Check if we should attempt reset
        if await self._should_attempt_reset():
            await self._change_state(CircuitState.HALF_OPEN)
            
        # Handle based on current state
        if self._state == CircuitState.OPEN:
            self._stats.rejected_calls += 1
            raise CircuitOpenError(
                f"Circuit breaker '{self.name}' is OPEN. "
                f"Failed {self._stats.consecutive_failures} times."
            )
            
        # Acquire semaphore for half-open state
        semaphore = (
            self._half_open_semaphore
            if self._state == CircuitState.HALF_OPEN
            else None
        )
        
        try:
            if semaphore:
                async with semaphore:
                    return await self._execute_call(func, *args, **kwargs)
            else:
                return await self._execute_call(func, *args, **kwargs)
        except Exception as e:
            # Let the error handler decide if this should trip the circuit
            if self.error_handler(e):
                await self._record_failure()
            raise
            
    async def _execute_call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute the actual function call."""
        self._stats.total_calls += 1
        
        try:
            # Execute function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, func, *args, **kwargs
                )
                
            # Record success
            await self._record_success()
            return result
            
        except Exception:
            # Failure handling is done in the caller
            raise
            
    async def _record_success(self) -> None:
        """Record successful call."""
        self._stats.successful_calls += 1
        self._stats.consecutive_failures = 0
        self._stats.consecutive_successes += 1
        
        # Check state transitions
        if self._state == CircuitState.HALF_OPEN:
            if self._stats.consecutive_successes >= self.success_threshold:
                await self._change_state(CircuitState.CLOSED)
                
    async def _record_failure(self) -> None:
        """Record failed call."""
        self._stats.failed_calls += 1
        self._stats.consecutive_successes = 0
        self._stats.consecutive_failures += 1
        self._stats.last_failure_time = time.time()
        
        # Check state transitions
        if self._state == CircuitState.CLOSED:
            if self._stats.consecutive_failures >= self.failure_threshold:
                await self._change_state(CircuitState.OPEN)
        elif self._state == CircuitState.HALF_OPEN:
            # Single failure in half-open returns to open
            await self._change_state(CircuitState.OPEN)
            
    async def reset(self) -> None:
        """Manually reset the circuit breaker."""
        await self._change_state(CircuitState.CLOSED)
        self._stats.consecutive_failures = 0
        self._stats.consecutive_successes = 0
        
    def get_status(self) -> Dict:
        """Get detailed circuit breaker status."""
        return {
            "name": self.name,
            "state": self._state.value,
            "stats": {
                "total_calls": self._stats.total_calls,
                "successful_calls": self._stats.successful_calls,
                "failed_calls": self._stats.failed_calls,
                "rejected_calls": self._stats.rejected_calls,
                "failure_rate": round(self._stats.failure_rate, 3),
                "success_rate": round(self._stats.success_rate, 3),
                "consecutive_failures": self._stats.consecutive_failures,
                "consecutive_successes": self._stats.consecutive_successes,
                "last_failure_time": self._stats.last_failure_time,
            },
            "config": {
                "failure_threshold": self.failure_threshold,
                "success_threshold": self.success_threshold,
                "timeout": self.timeout,
                "half_open_limit": self.half_open_limit,
            },
            "uptime": time.time() - self._stats.state_changes[0][1] if self._stats.state_changes else 0,
        }


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""
    pass


class CircuitBreakerGroup:
    """Manage multiple circuit breakers."""
    
    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        
    def add_breaker(self, breaker: CircuitBreaker) -> None:
        """Add a circuit breaker to the group."""
        self._breakers[breaker.name] = breaker
        
    def get_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self._breakers.get(name)
        
    async def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for breaker in self._breakers.values():
            await breaker.reset()
            
    def get_all_status(self) -> Dict[str, Dict]:
        """Get status of all circuit breakers."""
        return {
            name: breaker.get_status()
            for name, breaker in self._breakers.items()
        }
        
    def get_summary(self) -> Dict:
        """Get summary of circuit breaker states."""
        states = {"closed": 0, "open": 0, "half_open": 0}
        total_calls = 0
        total_failures = 0
        
        for breaker in self._breakers.values():
            states[breaker.state.value] += 1
            total_calls += breaker.stats.total_calls
            total_failures += breaker.stats.failed_calls
            
        return {
            "total_breakers": len(self._breakers),
            "states": states,
            "total_calls": total_calls,
            "total_failures": total_failures,
            "overall_failure_rate": (
                total_failures / total_calls if total_calls > 0 else 0
            ),
        }
