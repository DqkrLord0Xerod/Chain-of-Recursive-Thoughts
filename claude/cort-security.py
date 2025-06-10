"""Security measures for API protection."""

from __future__ import annotations

import hashlib
import re
import secrets
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from cryptography.fernet import Fernet

logger = structlog.get_logger(__name__)


@dataclass
class SecurityConfig:
    """Security configuration."""
    max_prompt_length: int = 10000
    max_context_messages: int = 100
    max_tokens_per_request: int = 100000
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600  # seconds
    blocked_patterns: List[str] = None
    require_api_key: bool = True
    enable_encryption: bool = True
    session_timeout: int = 3600
    max_concurrent_sessions: int = 1000


class APIKeyManager:
    """Secure API key management with encryption."""

    def __init__(self, master_key: Optional[str] = None):
        """
        Initialize with master key for encryption.

        Args:
            master_key: Base64-encoded Fernet key. If None, generates new key.
        """
        if master_key:
            self.cipher = Fernet(master_key.encode() if isinstance(master_key, str) else master_key)
        else:
            self.cipher = Fernet(Fernet.generate_key())

        self._keys: Dict[str, Dict] = {}  # key_id -> metadata
        self._encrypted_storage: Dict[str, bytes] = {}  # key_id -> encrypted_key

    def create_api_key(
        self,
        *,
        name: str,
        scopes: List[str],
        rate_limit: Optional[int] = None,
        expires_at: Optional[datetime] = None,
    ) -> Dict[str, str]:
        """
        Create new API key with metadata.

        Returns:
            Dictionary with key_id and api_key
        """
        # Generate secure random key
        api_key = secrets.token_urlsafe(32)
        key_id = hashlib.sha256(api_key.encode()).hexdigest()[:16]

        # Encrypt the actual key
        encrypted_key = self.cipher.encrypt(api_key.encode())

        # Store metadata
        self._keys[key_id] = {
            "name": name,
            "scopes": scopes,
            "rate_limit": rate_limit,
            "expires_at": expires_at.isoformat() if expires_at else None,
            "created_at": datetime.utcnow().isoformat(),
            "last_used": None,
            "usage_count": 0,
        }

        # Store encrypted key separately
        self._encrypted_storage[key_id] = encrypted_key

        logger.info(
            "api_key_created",
            key_id=key_id,
            name=name,
            scopes=scopes,
        )

        return {
            "key_id": key_id,
            "api_key": api_key,
            "name": name,
        }

    def validate_api_key(self, api_key: str) -> Optional[Dict]:
        """
        Validate API key and return metadata if valid.

        Args:
            api_key: The API key to validate

        Returns:
            Metadata dict if valid, None otherwise
        """
        # Compute key ID
        key_id = hashlib.sha256(api_key.encode()).hexdigest()[:16]

        if key_id not in self._keys:
            logger.warning("invalid_api_key_attempt", key_id=key_id)
            return None

        metadata = self._keys[key_id]

        # Check expiration
        if metadata.get("expires_at"):
            expires_at = datetime.fromisoformat(metadata["expires_at"])
            if datetime.utcnow() > expires_at:
                logger.warning("expired_api_key", key_id=key_id)
                return None

        # Verify the key matches
        try:
            encrypted_key = self._encrypted_storage.get(key_id)
            if not encrypted_key:
                return None

            decrypted_key = self.cipher.decrypt(encrypted_key).decode()
            if decrypted_key != api_key:
                return None

        except Exception as e:
            logger.error("api_key_validation_error", error=str(e))
            return None

        # Update usage
        metadata["last_used"] = datetime.utcnow().isoformat()
        metadata["usage_count"] += 1

        return metadata

    def revoke_api_key(self, key_id: str) -> bool:
        """Revoke an API key."""
        if key_id in self._keys:
            del self._keys[key_id]
            del self._encrypted_storage[key_id]
            logger.info("api_key_revoked", key_id=key_id)
            return True
        return False

    def list_api_keys(self) -> List[Dict]:
        """List all API keys (without the actual keys)."""
        return [
            {
                "key_id": key_id,
                **{k: v for k, v in metadata.items() if k != "api_key"}
            }
            for key_id, metadata in self._keys.items()
        ]


class InputValidator:
    """Validate and sanitize user inputs."""

    def __init__(self, config: SecurityConfig):
        self.config = config
        self.blocked_patterns = [
            re.compile(pattern)
            for pattern in (config.blocked_patterns or [])
        ]

    def validate_prompt(self, prompt: str) -> tuple[bool, Optional[str]]:
        """
        Validate user prompt.

        Returns:
            (is_valid, error_message)
        """
        # Length check
        if len(prompt) > self.config.max_prompt_length:
            return False, f"Prompt exceeds maximum length of {self.config.max_prompt_length}"

        # Empty check
        if not prompt.strip():
            return False, "Prompt cannot be empty"

        # Blocked patterns
        for pattern in self.blocked_patterns:
            if pattern.search(prompt):
                return False, "Prompt contains blocked content"

        # Check for potential injection attacks
        if self._contains_injection(prompt):
            return False, "Prompt contains potential injection attack"

        return True, None

    def validate_context(self, messages: List[Dict[str, str]]) -> tuple[bool, Optional[str]]:
        """Validate conversation context."""

        # Message count
        if len(messages) > self.config.max_context_messages:
            return False, f"Context exceeds maximum of {self.config.max_context_messages} messages"

        # Validate each message
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                return False, f"Message {i} is not a dictionary"

            if "role" not in msg or "content" not in msg:
                return False, f"Message {i} missing required fields"

            if msg["role"] not in ["system", "user", "assistant"]:
                return False, f"Message {i} has invalid role"

            # Validate content
            is_valid, error = self.validate_prompt(msg["content"])
            if not is_valid:
                return False, f"Message {i}: {error}"

        return True, None

    def sanitize_output(self, text: str) -> str:
        """Sanitize model output before returning to user."""

        # Remove potential sensitive information patterns
        # This is a simple example - extend based on requirements

        # Remove anything that looks like an API key
        text = re.sub(r'[A-Za-z0-9]{32,}', '[REDACTED]', text)

        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)

        # Remove phone numbers (simple pattern)
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)

        return text

    def _contains_injection(self, text: str) -> bool:
        """Check for potential injection attacks."""

        # Check for common injection patterns
        injection_patterns = [
            r'<script[^>]*>.*?</script>',  # Script tags
            r'javascript:',  # JavaScript protocol
            r'on\w+\s*=',  # Event handlers
            r'eval\s*\(',  # Eval function
            r'exec\s*\(',  # Exec function
            r'__import__',  # Python import
            r'subprocess',  # System calls
            r'os\.\w+\(',  # OS module calls
        ]

        for pattern in injection_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                logger.warning("potential_injection_detected", pattern=pattern)
                return True

        return False


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self, rate: int, window: int):
        """
        Initialize rate limiter.

        Args:
            rate: Number of requests allowed
            window: Time window in seconds
        """
        self.rate = rate
        self.window = window
        self.buckets: Dict[str, List[float]] = {}

    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed for given key."""
        now = time.time()

        if key not in self.buckets:
            self.buckets[key] = []

        # Remove old entries
        self.buckets[key] = [
            timestamp for timestamp in self.buckets[key]
            if now - timestamp < self.window
        ]

        # Check rate
        if len(self.buckets[key]) >= self.rate:
            return False

        # Add current request
        self.buckets[key].append(now)
        return True

    def get_remaining(self, key: str) -> int:
        """Get remaining requests for key."""
        now = time.time()

        if key not in self.buckets:
            return self.rate

        # Count recent requests
        recent = [
            t for t in self.buckets[key]
            if now - t < self.window
        ]

        return max(0, self.rate - len(recent))

    def get_reset_time(self, key: str) -> int:
        """Get seconds until rate limit resets."""
        if key not in self.buckets or not self.buckets[key]:
            return 0

        oldest = min(self.buckets[key])
        reset_time = oldest + self.window

        return max(0, int(reset_time - time.time()))


class SessionManager:
    """Secure session management."""

    def __init__(self, config: SecurityConfig):
        self.config = config
        self.sessions: Dict[str, Dict] = {}
        self._cleanup_interval = 300  # 5 minutes
        self._last_cleanup = time.time()

    def create_session(
        self,
        user_id: str,
        metadata: Optional[Dict] = None,
    ) -> str:
        """Create new session."""

        # Clean up old sessions periodically
        if time.time() - self._last_cleanup > self._cleanup_interval:
            self._cleanup_expired_sessions()

        # Check concurrent session limit
        active_sessions = sum(
            1 for s in self.sessions.values()
            if s.get("user_id") == user_id
        )

        if active_sessions >= self.config.max_concurrent_sessions:
            raise ValueError(f"User {user_id} has too many active sessions")

        # Create session
        session_id = secrets.token_urlsafe(32)

        self.sessions[session_id] = {
            "user_id": user_id,
            "created_at": time.time(),
            "last_activity": time.time(),
            "metadata": metadata or {},
        }

        logger.info("session_created", session_id=session_id[:8], user_id=user_id)

        return session_id

    def validate_session(self, session_id: str) -> Optional[Dict]:
        """Validate session and return data if valid."""

        if session_id not in self.sessions:
            return None

        session = self.sessions[session_id]

        # Check expiration
        if time.time() - session["created_at"] > self.config.session_timeout:
            del self.sessions[session_id]
            logger.info("session_expired", session_id=session_id[:8])
            return None

        # Update activity
        session["last_activity"] = time.time()

        return session

    def end_session(self, session_id: str) -> None:
        """End a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info("session_ended", session_id=session_id[:8])

    def _cleanup_expired_sessions(self) -> None:
        """Remove expired sessions."""
        now = time.time()
        expired = [
            sid for sid, session in self.sessions.items()
            if now - session["created_at"] > self.config.session_timeout
        ]

        for sid in expired:
            del self.sessions[sid]

        if expired:
            logger.info("sessions_cleaned_up", count=len(expired))

        self._last_cleanup = now


class SecurityMiddleware:
    """Comprehensive security middleware."""

    def __init__(
        self,
        config: SecurityConfig,
        api_key_manager: APIKeyManager,
    ):
        self.config = config
        self.api_key_manager = api_key_manager
        self.input_validator = InputValidator(config)
        self.rate_limiter = RateLimiter(
            config.rate_limit_requests,
            config.rate_limit_window,
        )
        self.session_manager = SessionManager(config)

    async def validate_request(
        self,
        *,
        api_key: Optional[str],
        session_id: Optional[str],
        prompt: str,
        context: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Validate incoming request.

        Returns:
            Validation result with user info

        Raises:
            SecurityError: If validation fails
        """

        # API key validation
        user_info = {}
        rate_limit_key = "anonymous"

        if self.config.require_api_key:
            if not api_key:
                raise SecurityError("API key required")

            key_info = self.api_key_manager.validate_api_key(api_key)
            if not key_info:
                raise SecurityError("Invalid API key")

            user_info = key_info
            rate_limit_key = f"api_key_{hashlib.sha256(api_key.encode()).hexdigest()[:8]}"

        # Session validation
        if session_id:
            session_info = self.session_manager.validate_session(session_id)
            if not session_info:
                raise SecurityError("Invalid or expired session")

            user_info.update(session_info)
            rate_limit_key = f"session_{session_id[:8]}"

        # Rate limiting
        if not self.rate_limiter.is_allowed(rate_limit_key):
            remaining = self.rate_limiter.get_remaining(rate_limit_key)
            reset_time = self.rate_limiter.get_reset_time(rate_limit_key)

            raise RateLimitError(
                f"Rate limit exceeded. Remaining: {remaining}, Reset in: {reset_time}s"
            )

        # Input validation
        is_valid, error = self.input_validator.validate_prompt(prompt)
        if not is_valid:
            raise ValidationError(f"Invalid prompt: {error}")

        if context:
            is_valid, error = self.input_validator.validate_context(context)
            if not is_valid:
                raise ValidationError(f"Invalid context: {error}")

        return {
            "user_info": user_info,
            "rate_limit_key": rate_limit_key,
            "rate_limit_remaining": self.rate_limiter.get_remaining(rate_limit_key),
        }


class SecurityError(Exception):
    """Base security exception."""
    pass


class ValidationError(SecurityError):
    """Input validation error."""
    pass


class RateLimitError(SecurityError):
    """Rate limit exceeded error."""
    pass
