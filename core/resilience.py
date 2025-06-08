from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass


class APIError(Exception):
    """Base exception for API related errors."""


class RateLimitError(APIError):
    """Raised when the API rate limit is exceeded."""


class TokenLimitError(APIError):
    """Raised when the request exceeds the allowed token limit."""


@dataclass
class RetryState:
    max_retries: int = 3
    failure_count: int = 0


async def with_retry(coro, state: RetryState) -> str:
    for attempt in range(1, state.max_retries + 1):
        try:
            return await coro()
        except (APIError, RateLimitError, TokenLimitError) as e:
            state.failure_count += 1
            if attempt == state.max_retries:
                raise e
            await asyncio.sleep(2 ** (attempt - 1) + random.random())
    raise APIError("unreachable")
