class APIError(Exception):
    """Base exception for API related errors."""


class RateLimitError(APIError):
    """Raised when the API rate limit is exceeded."""


class TokenLimitError(APIError):
    """Raised when the request exceeds the allowed token limit."""
