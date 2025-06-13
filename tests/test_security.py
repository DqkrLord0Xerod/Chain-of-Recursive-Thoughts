"""Comprehensive security tests."""

import os
import time
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from core.security.api_security import (
    APIKeyManager,
    InputValidator,
    RateLimiter,
    SessionManager,
    SecurityMiddleware,
    SecurityConfig,
    SecurityError,
    ValidationError,
    RateLimitError,
)
from core.security import OutputFilter
from core.chat_v2 import RecursiveThinkingEngine, ThinkingStrategy
from core.context_manager import ContextManager
from core.providers.cache import InMemoryLRUCache
from core.interfaces import LLMProvider, QualityEvaluator
from config.config import load_production_config


class TestAPIKeyManager:
    
    @pytest.fixture
    def api_key_manager(self):
        return APIKeyManager()
        
    def test_create_api_key(self, api_key_manager):
        """Test API key creation."""
        result = api_key_manager.create_api_key(
            name="test-key",
            scopes=["read", "write"],
            rate_limit=100,
        )
        
        assert "key_id" in result
        assert "api_key" in result
        assert len(result["api_key"]) >= 32
        assert result["name"] == "test-key"
        
    def test_validate_api_key_success(self, api_key_manager):
        """Test successful API key validation."""
        # Create key
        result = api_key_manager.create_api_key(
            name="test-key",
            scopes=["read"],
        )
        
        # Validate
        metadata = api_key_manager.validate_api_key(result["api_key"])
        
        assert metadata is not None
        assert metadata["name"] == "test-key"
        assert metadata["scopes"] == ["read"]
        assert metadata["usage_count"] == 1
        
    def test_validate_api_key_invalid(self, api_key_manager):
        """Test invalid API key validation."""
        metadata = api_key_manager.validate_api_key("invalid-key")
        assert metadata is None
        
    def test_validate_api_key_expired(self, api_key_manager):
        """Test expired API key validation."""
        # Create key with past expiration
        result = api_key_manager.create_api_key(
            name="expired-key",
            scopes=["read"],
            expires_at=datetime.utcnow() - timedelta(hours=1),
        )
        
        # Should fail validation
        metadata = api_key_manager.validate_api_key(result["api_key"])
        assert metadata is None
        
    def test_revoke_api_key(self, api_key_manager):
        """Test API key revocation."""
        # Create key
        result = api_key_manager.create_api_key(
            name="test-key",
            scopes=["read"],
        )
        
        # Revoke
        success = api_key_manager.revoke_api_key(result["key_id"])
        assert success is True
        
        # Should no longer validate
        metadata = api_key_manager.validate_api_key(result["api_key"])
        assert metadata is None
        
    def test_list_api_keys(self, api_key_manager):
        """Test listing API keys."""
        # Create multiple keys
        for i in range(3):
            api_key_manager.create_api_key(
                name=f"key-{i}",
                scopes=["read"],
            )
            
        keys = api_key_manager.list_api_keys()
        assert len(keys) == 3
        
        # Should not contain actual API keys
        for key_info in keys:
            assert "api_key" not in key_info
            assert "key_id" in key_info
            assert "name" in key_info


class TestInputValidator:
    
    @pytest.fixture
    def validator(self):
        config = SecurityConfig(
            max_prompt_length=1000,
            max_context_messages=10,
            blocked_patterns=[r"<script.*?>", r"DROP TABLE"],
        )
        return InputValidator(config)
        
    def test_validate_prompt_success(self, validator):
        """Test valid prompt validation."""
        is_valid, error = validator.validate_prompt("What is the weather today?")
        assert is_valid is True
        assert error is None
        
    def test_validate_prompt_too_long(self, validator):
        """Test prompt length validation."""
        long_prompt = "a" * 1001
        is_valid, error = validator.validate_prompt(long_prompt)
        assert is_valid is False
        assert "exceeds maximum length" in error
        
    def test_validate_prompt_empty(self, validator):
        """Test empty prompt validation."""
        is_valid, error = validator.validate_prompt("   ")
        assert is_valid is False
        assert "cannot be empty" in error
        
    def test_validate_prompt_blocked_pattern(self, validator):
        """Test blocked pattern detection."""
        is_valid, error = validator.validate_prompt("Run this <script>alert('xss')</script>")
        assert is_valid is False
        assert "blocked content" in error
        
    def test_validate_prompt_injection(self, validator):
        """Test injection attack detection."""
        is_valid, error = validator.validate_prompt("'; DROP TABLE users; --")
        assert is_valid is False
        assert "blocked content" in error
        
    def test_validate_context_success(self, validator):
        """Test valid context validation."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        is_valid, error = validator.validate_context(messages)
        assert is_valid is True
        assert error is None
        
    def test_validate_context_too_many(self, validator):
        """Test context message limit."""
        messages = [
            {"role": "user", "content": f"Message {i}"}
            for i in range(11)
        ]
        is_valid, error = validator.validate_context(messages)
        assert is_valid is False
        assert "exceeds maximum" in error
        
    def test_validate_context_invalid_structure(self, validator):
        """Test invalid message structure."""
        messages = [
            {"role": "user"},  # Missing content
        ]
        is_valid, error = validator.validate_context(messages)
        assert is_valid is False
        assert "missing required fields" in error
        
    def test_sanitize_output(self, validator):
        """Test output sanitization."""
        text = """
        Here's my API key: sk-1234567890abcdef1234567890abcdef
        Contact me at test@example.com or 555-123-4567
        """
        
        sanitized = validator.sanitize_output(text)
        
        assert "[REDACTED]" in sanitized
        assert "[EMAIL]" in sanitized
        assert "[PHONE]" in sanitized
        assert "sk-1234567890" not in sanitized


class TestRateLimiter:
    
    @pytest.fixture
    def rate_limiter(self):
        return RateLimiter(rate=5, window=60)  # 5 requests per minute
        
    def test_rate_limit_allows_initial(self, rate_limiter):
        """Test rate limiter allows initial requests."""
        for i in range(5):
            assert rate_limiter.is_allowed("user1") is True
            
    def test_rate_limit_blocks_excess(self, rate_limiter):
        """Test rate limiter blocks excess requests."""
        # Use up allowance
        for i in range(5):
            rate_limiter.is_allowed("user1")
            
        # Should block
        assert rate_limiter.is_allowed("user1") is False
        
    def test_rate_limit_different_keys(self, rate_limiter):
        """Test rate limiter tracks different keys separately."""
        # Use up user1's allowance
        for i in range(5):
            rate_limiter.is_allowed("user1")
            
        # user2 should still be allowed
        assert rate_limiter.is_allowed("user2") is True
        
    def test_rate_limit_window_reset(self, rate_limiter):
        """Test rate limit window reset."""
        # Use up allowance
        for i in range(5):
            rate_limiter.is_allowed("user1")
            
        # Should be blocked
        assert rate_limiter.is_allowed("user1") is False
        
        # Mock time passing
        with patch('time.time', return_value=time.time() + 61):
            # Should be allowed again
            assert rate_limiter.is_allowed("user1") is True
            
    def test_get_remaining(self, rate_limiter):
        """Test getting remaining requests."""
        assert rate_limiter.get_remaining("user1") == 5
        
        rate_limiter.is_allowed("user1")
        assert rate_limiter.get_remaining("user1") == 4
        
    def test_get_reset_time(self, rate_limiter):
        """Test getting reset time."""
        rate_limiter.is_allowed("user1")
        
        reset_time = rate_limiter.get_reset_time("user1")
        assert 0 < reset_time <= 60


class TestSessionManager:
    
    @pytest.fixture
    def session_manager(self):
        config = SecurityConfig(
            session_timeout=3600,
            max_concurrent_sessions=2,
        )
        return SessionManager(config)
        
    def test_create_session(self, session_manager):
        """Test session creation."""
        session_id = session_manager.create_session(
            user_id="user1",
            metadata={"ip": "127.0.0.1"},
        )
        
        assert session_id is not None
        assert len(session_id) >= 32
        
    def test_validate_session_success(self, session_manager):
        """Test successful session validation."""
        session_id = session_manager.create_session(user_id="user1")
        
        session_data = session_manager.validate_session(session_id)
        assert session_data is not None
        assert session_data["user_id"] == "user1"
        
    def test_validate_session_invalid(self, session_manager):
        """Test invalid session validation."""
        session_data = session_manager.validate_session("invalid-session")
        assert session_data is None
        
    def test_validate_session_expired(self, session_manager):
        """Test expired session validation."""
        session_id = session_manager.create_session(user_id="user1")
        
        # Mock time passing
        with patch('time.time', return_value=time.time() + 3601):
            session_data = session_manager.validate_session(session_id)
            assert session_data is None
            
    def test_concurrent_session_limit(self, session_manager):
        """Test concurrent session limit."""
        # Create max sessions
        session_manager.create_session(user_id="user1")
        session_manager.create_session(user_id="user1")
        
        # Should fail
        with pytest.raises(ValueError):
            session_manager.create_session(user_id="user1")
            
    def test_end_session(self, session_manager):
        """Test ending a session."""
        session_id = session_manager.create_session(user_id="user1")
        
        session_manager.end_session(session_id)
        
        # Should no longer validate
        session_data = session_manager.validate_session(session_id)
        assert session_data is None


class TestSecurityMiddleware:
    
    @pytest.fixture
    def middleware(self):
        config = SecurityConfig(
            require_api_key=True,
            rate_limit_requests=10,
            rate_limit_window=60,
        )
        api_key_manager = APIKeyManager()
        return SecurityMiddleware(config, api_key_manager), api_key_manager
        
    @pytest.mark.asyncio
    async def test_validate_request_no_api_key(self, middleware):
        """Test request validation without API key."""
        security_middleware, _ = middleware
        
        with pytest.raises(SecurityError) as exc_info:
            await security_middleware.validate_request(
                api_key=None,
                session_id=None,
                prompt="Test prompt",
            )
            
        assert "API key required" in str(exc_info.value)
        
    @pytest.mark.asyncio
    async def test_validate_request_invalid_api_key(self, middleware):
        """Test request validation with invalid API key."""
        security_middleware, _ = middleware
        
        with pytest.raises(SecurityError) as exc_info:
            await security_middleware.validate_request(
                api_key="invalid-key",
                session_id=None,
                prompt="Test prompt",
            )
            
        assert "Invalid API key" in str(exc_info.value)
        
    @pytest.mark.asyncio
    async def test_validate_request_success(self, middleware):
        """Test successful request validation."""
        security_middleware, api_key_manager = middleware
        
        # Create valid API key
        key_result = api_key_manager.create_api_key(
            name="test",
            scopes=["read"],
        )
        
        # Validate request
        result = await security_middleware.validate_request(
            api_key=key_result["api_key"],
            session_id=None,
            prompt="What is the weather?",
        )
        
        assert "user_info" in result
        assert "rate_limit_remaining" in result
        assert result["rate_limit_remaining"] == 9  # Used 1 of 10
        
    @pytest.mark.asyncio
    async def test_validate_request_rate_limit(self, middleware):
        """Test rate limiting."""
        security_middleware, api_key_manager = middleware
        
        # Create API key
        key_result = api_key_manager.create_api_key(
            name="test",
            scopes=["read"],
            rate_limit=2,  # Low limit for testing
        )
        
        # Use up rate limit
        for i in range(10):
            try:
                await security_middleware.validate_request(
                    api_key=key_result["api_key"],
                    session_id=None,
                    prompt="Test",
                )
            except RateLimitError:
                break
                
        # Next request should fail
        with pytest.raises(RateLimitError) as exc_info:
            await security_middleware.validate_request(
                api_key=key_result["api_key"],
                session_id=None,
                prompt="Test",
            )
            
        assert "Rate limit exceeded" in str(exc_info.value)
        
    @pytest.mark.asyncio
    async def test_validate_request_invalid_prompt(self, middleware):
        """Test request validation with invalid prompt."""
        security_middleware, api_key_manager = middleware
        
        # Create API key
        key_result = api_key_manager.create_api_key(
            name="test",
            scopes=["read"],
        )
        
        # Empty prompt
        with pytest.raises(ValidationError) as exc_info:
            await security_middleware.validate_request(
                api_key=key_result["api_key"],
                session_id=None,
                prompt="",
            )
            
        assert "Invalid prompt" in str(exc_info.value)


class TestProductionConfig:
    
    @pytest.fixture
    def mock_env(self):
        """Mock environment variables."""
        env = {
            "APP_ENV": "production",
            "API_KEY_MASTER_KEY": "a" * 32,
            "SESSION_SECRET_KEY": "b" * 32,
            "LLM_PRIMARY_MODEL": "gpt-4",
            "LLM_PRIMARY_API_KEY": "test-key",
            "FRONTEND_URL": "https://example.com",
            "DATABASE_URL": "postgresql://user:pass@localhost/db",
        }
        with patch.dict(os.environ, env):
            yield
            
    def test_load_production_config(self, mock_env):
        """Test loading production configuration."""
        config = load_production_config()
        
        assert config.environment == "production"
        assert config.debug is False
        assert config.security.require_api_key is True
        assert config.llm.primary_model == "gpt-4"
        
    def test_production_config_validation(self, mock_env):
        """Test production config validation."""
        # Debug must be false in production
        with patch.dict(os.environ, {"DEBUG": "true"}):
            with pytest.raises(SystemExit):
                load_production_config()


class TestOutputFilter:

    def test_filter_blocks_pattern(self):
        filt = OutputFilter(blocked_patterns=[r"bad"])
        with pytest.raises(ValueError):
            filt.filter("this is bad text")

    def test_filter_masks_pii(self):
        filt = OutputFilter(mask_pii=True)
        text = "Contact me at user@example.com with key abcdefghijklmnopqrstuvwx1234567890 and 555-123-4567"
        result = filt.filter(text)
        assert "[EMAIL]" in result
        assert "[REDACTED]" in result
        assert "[PHONE]" in result


class TestEngineOutputFiltering:

    class DummyLLM(LLMProvider):
        async def chat(self, messages, *, temperature=0.7, **kwargs):
            return type("Resp", (), {
                "content": "Reach me at user@example.com",
                "usage": {"total_tokens": 1},
                "model": "dummy",
                "cached": False,
            })()

    class DummyEvaluator(QualityEvaluator):
        thresholds = {"overall": 0.5}

        def score(self, response: str, prompt: str) -> float:
            return 0.5

    class OneRoundStrategy(ThinkingStrategy):
        async def determine_rounds(self, prompt: str) -> int:
            return 1

        async def should_continue(self, rounds_completed, quality_scores, responses):
            return False, "done"

    @pytest.mark.asyncio
    async def test_engine_applies_filter(self):
        engine = RecursiveThinkingEngine(
            llm=self.DummyLLM(),
            cache=InMemoryLRUCache(max_size=2),
            evaluator=self.DummyEvaluator(),
            context_manager=ContextManager(100, type("T", (), {"encode": lambda self, t: t.split()})()),
            thinking_strategy=self.OneRoundStrategy(),
            model_selector=None,
            output_filter=OutputFilter(mask_pii=True),
        )
        result = await engine.think_and_respond("hi", thinking_rounds=1, alternatives_per_round=1)
        assert "[EMAIL]" in result.response
                
    def test_security_config_validation(self):
        """Test security configuration validation."""
        from config.config import SecuritySettings
        
        # Short secret should fail
        with pytest.raises(ValueError):
            SecuritySettings(
                api_key_master_key="short",
                session_secret_key="x" * 32,
            )
            
    def test_cors_production_validation(self, mock_env):
        """Test CORS validation in production."""
        # Wildcard CORS should fail in production
        with patch.dict(os.environ, {"CORS_ALLOWED_ORIGINS": '["*"]'}):
            with pytest.raises(SystemExit):
                load_production_config()
