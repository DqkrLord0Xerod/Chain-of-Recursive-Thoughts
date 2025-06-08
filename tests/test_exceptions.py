import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import requests  # noqa: E402
import pytest  # noqa: E402

from recursive_thinking_ai import (  # noqa: E402
    EnhancedRecursiveThinkingChat,
    CoRTConfig,
)
from exceptions import APIError, RateLimitError, TokenLimitError  # noqa: E402


class DummyResp:
    def __init__(self, status, data="err"):
        self.status_code = status
        self.data = data

    def iter_lines(self):
        yield b"data: [DONE]"

    def json(self):
        return {"error": self.data}


def test_rate_limit_error(monkeypatch):
    chat = EnhancedRecursiveThinkingChat(CoRTConfig(api_key="x"))

    monkeypatch.setattr(requests, "post", lambda *a, **k: DummyResp(429))

    with pytest.raises(RateLimitError):
        chat._call_api([{"role": "user", "content": "hi"}], stream=False)


def test_token_limit_error(monkeypatch):
    chat = EnhancedRecursiveThinkingChat(CoRTConfig(api_key="x"))

    monkeypatch.setattr(requests, "post", lambda *a, **k: DummyResp(400, "token exceeded"))

    with pytest.raises(TokenLimitError):
        chat._call_api([{"role": "user", "content": "hi"}], stream=False)


def test_circuit_breaker(monkeypatch):
    chat = EnhancedRecursiveThinkingChat(CoRTConfig(api_key="x", max_retries=2))

    monkeypatch.setattr(requests, "post", lambda *a, **k: DummyResp(500))

    with pytest.raises(APIError):
        chat._call_api([{"role": "user", "content": "hi"}], stream=False)
    assert chat.circuit_open

    with pytest.raises(APIError):
        chat._call_api([{"role": "user", "content": "again"}], stream=False)
