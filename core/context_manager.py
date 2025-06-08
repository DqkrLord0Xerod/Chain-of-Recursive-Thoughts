from __future__ import annotations

from typing import Callable, Dict, List


class ContextManager:
    """Advanced context manager preserving important messages."""

    def __init__(
        self,
        max_tokens: int,
        tokenizer,
        summarizer: Callable[[List[Dict[str, str]]], str] | None = None,
    ) -> None:
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer
        self.summarizer = summarizer

    def _count(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def optimize(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        if not messages:
            return []

        system_msgs = [m for m in messages if m.get("role") == "system"]
        convo = [m for m in messages if m.get("role") != "system"]

        def msg_tokens(msg: Dict[str, str]) -> int:
            return self._count(msg.get("content", ""))

        total = sum(msg_tokens(m) for m in messages)
        if total <= self.max_tokens:
            return list(messages)

        preserved = list(system_msgs)
        recent = []
        token_total = sum(msg_tokens(m) for m in preserved)
        for msg in reversed(convo):
            t = msg_tokens(msg)
            if token_total + t > self.max_tokens:
                break
            recent.insert(0, msg)
            token_total += t
        remaining = [m for m in convo if m not in recent]
        summary = ""
        if remaining and self.summarizer is not None:
            summary = self.summarizer(remaining)
            token_total += self._count(summary)
        if token_total > self.max_tokens:
            while recent and token_total > self.max_tokens:
                removed = recent.pop(0)
                token_total -= msg_tokens(removed)
        if summary:
            preserved.append({"role": "system", "content": summary})
        return preserved + recent
