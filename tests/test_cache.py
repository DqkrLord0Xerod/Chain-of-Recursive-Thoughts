import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))  # noqa: E402

import requests  # noqa: E402
from recursive_thinking_ai import EnhancedRecursiveThinkingChat  # noqa: E402


def make_response(text):
    class Resp:
        def __init__(self):
            message = (
                'data: {"choices": [{"delta": {"content": "' + text + '"}}]}'
            )
            self.lines = [message.encode(), b"data: [DONE]"]

        def raise_for_status(self):
            pass

        def iter_lines(self):
            for line in self.lines:
                yield line

        def json(self):
            return {"choices": [{"message": {"content": text}}]}

    return Resp()


def test_cache_hits(monkeypatch):
    chat = EnhancedRecursiveThinkingChat(api_key="x")
    calls = []

    def fake_post(*a, **k):
        calls.append(1)
        return make_response("hello")

    monkeypatch.setattr(requests, "post", fake_post)

    messages = [{"role": "user", "content": "hi"}]
    assert chat._call_api(messages, stream=True) == "hello"
    assert chat._call_api(messages, stream=True) == "hello"
    assert len(calls) == 1


def test_cache_disabled(monkeypatch):
    chat = EnhancedRecursiveThinkingChat(api_key="x", caching_enabled=False)
    calls = []

    def fake_post(*a, **k):
        calls.append(1)
        return make_response("hi")

    monkeypatch.setattr(requests, "post", fake_post)

    messages = [{"role": "user", "content": "hi"}]
    chat._call_api(messages, stream=True)
    chat._call_api(messages, stream=True)
    assert len(calls) == 2


def test_disk_cache_persist(tmp_path, monkeypatch):
    path = tmp_path / "cache.pkl"
    chat = EnhancedRecursiveThinkingChat(
        api_key="x", disk_cache_path=str(path)
    )

    monkeypatch.setattr(requests, "post", lambda *a, **k: make_response("hi"))

    messages = [{"role": "user", "content": "hi"}]
    assert chat._call_api(messages, stream=True) == "hi"

    calls = []

    def fake_post(*a, **k):
        calls.append(1)
        return make_response("new")

    monkeypatch.setattr(requests, "post", fake_post)
    chat2 = EnhancedRecursiveThinkingChat(api_key="x", disk_cache_path=str(path))

    assert chat2._call_api(messages, stream=True) == "hi"
    assert not calls


def test_disk_cache_limit(tmp_path, monkeypatch):
    path = tmp_path / "cache.pkl"
    chat = EnhancedRecursiveThinkingChat(
        api_key="x", disk_cache_path=str(path), disk_cache_size=1
    )

    monkeypatch.setattr(requests, "post", lambda *a, **k: make_response("a"))
    chat._call_api([{"role": "user", "content": "a"}], stream=True)

    monkeypatch.setattr(requests, "post", lambda *a, **k: make_response("b"))
    chat._call_api([{"role": "user", "content": "b"}], stream=True)

    calls = []

    def fake_post(*a, **k):
        calls.append(1)
        return make_response("a2")

    monkeypatch.setattr(requests, "post", fake_post)
    chat2 = EnhancedRecursiveThinkingChat(
        api_key="x", disk_cache_path=str(path), disk_cache_size=1
    )

    chat2._call_api([{"role": "user", "content": "a"}], stream=True)
    assert calls == [1]
