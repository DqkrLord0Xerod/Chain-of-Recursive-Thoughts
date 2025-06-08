import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # noqa: E402

from recursive_thinking_ai import EnhancedRecursiveThinkingChat  # noqa: E402


def test_determine_rounds_invalid_response(monkeypatch, caplog):
    chat = EnhancedRecursiveThinkingChat(api_key="test")
    caplog.set_level(logging.ERROR)
    monkeypatch.setattr(chat, "_call_api", lambda *a, **k: "invalid")
    result = chat._determine_thinking_rounds("prompt")
    assert result == 3
    assert "Failed to parse thinking rounds" in caplog.text


def test_determine_rounds_none_response(monkeypatch, caplog):
    chat = EnhancedRecursiveThinkingChat(api_key="test")
    caplog.set_level(logging.ERROR)
    monkeypatch.setattr(chat, "_call_api", lambda *a, **k: None)
    result = chat._determine_thinking_rounds("prompt")
    assert result == 3
    assert "Failed to parse thinking rounds" in caplog.text
