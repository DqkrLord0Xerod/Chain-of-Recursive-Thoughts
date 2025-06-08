import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from recursive_thinking_ai import (  # noqa: E402
    ContextManager,
    EnhancedRecursiveThinkingChat,
)
import requests  # noqa: E402


class DummyTokenizer:
    def encode(self, text):
        return text.split()


def test_optimize_context_basic():
    manager = ContextManager(max_tokens=4, tokenizer=DummyTokenizer())
    history = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"},
        {"role": "assistant", "content": "a3"},
    ]
    trimmed = manager.optimize_context(history)
    assert trimmed[0]["role"] == "system"
    assert trimmed[-1]["content"] == "a3"
    assert {"role": "user", "content": "u1"} not in trimmed
    assert len(trimmed) <= 4


def test_chat_uses_context_manager(monkeypatch):
    chat = EnhancedRecursiveThinkingChat(api_key="x", max_context_tokens=4)
    chat.tokenizer = DummyTokenizer()
    chat.context_manager = ContextManager(4, chat.tokenizer)
    chat.conversation_history = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]

    def make_response(text):
        class Resp:
            def raise_for_status(self):
                pass

            def iter_lines(self):
                line = (
                    'data: {"choices": [{"delta": {"content": "' + text + '"}}]}'
                )
                yield line.encode()
                yield b"data: [DONE]"

            def json(self):
                return {"choices": [{"message": {"content": text}}]}

        return Resp()

    monkeypatch.setattr(requests, "post", lambda *a, **k: make_response("resp"))
    monkeypatch.setattr(chat, "_semantic_similarity", lambda *a, **k: 0.0)
    monkeypatch.setattr(
        chat.quality_assessor,
        "comprehensive_score",
        lambda *a, **k: {
            "relevance": 0,
            "completeness": 0,
            "clarity": 0,
            "accuracy": 0,
            "overall": 0,
        },
    )
    monkeypatch.setattr(chat, "_determine_thinking_rounds", lambda *a, **k: 1)
    monkeypatch.setattr(
        chat, "_batch_generate_and_evaluate", lambda *a, **k: ("resp", [], "r")
    )

    chat.think_and_respond("u3", verbose=False)

    assert chat.conversation_history[0]["role"] == "system"
    assert {"role": "user", "content": "u1"} not in chat.conversation_history
    assert chat.conversation_history[-1]["content"] == "resp"
