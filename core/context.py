from __future__ import annotations

from typing import Dict, List


class ContextManager:
    """Manage pruning of conversation history to fit token limits."""

    def __init__(self, max_tokens: int, tokenizer, summarizer=None) -> None:
        self.max_tokens = max_tokens
        self.tokenizer = tokenizer
        self.summarizer = summarizer

    def _count(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def optimize_context(self, messages: List[Dict]) -> List[Dict]:
        """Return a trimmed list of messages within the token budget."""
        if not messages:
            return []

        def msg_tokens(msg: Dict) -> int:
            return self._count(msg.get("content", ""))

        total = sum(msg_tokens(m) for m in messages)
        if total <= self.max_tokens:
            return list(messages)

        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        trimmed_sys: List[Dict] = []
        token_total = 0
        for msg in reversed(system_msgs):
            t = msg_tokens(msg)
            if token_total + t <= self.max_tokens:
                trimmed_sys.insert(0, msg)
                token_total += t

        non_system_trim: List[Dict] = []
        pruned: List[Dict] = []
        for msg in reversed(non_system):
            t = msg_tokens(msg)
            if token_total + t > self.max_tokens:
                pruned.insert(0, msg)
                continue
            non_system_trim.insert(0, msg)
            token_total += t

        if pruned and self.summarizer is not None:
            summary_text = self.summarizer(pruned)
            summary_tokens = self._count(summary_text)
            while non_system_trim and summary_tokens + token_total > self.max_tokens:
                removed = non_system_trim.pop(0)
                token_total -= msg_tokens(removed)
                pruned.insert(0, removed)
            if summary_tokens + token_total <= self.max_tokens:
                trimmed_sys.append({"role": "system", "content": summary_text})
                token_total += summary_tokens

        return trimmed_sys + non_system_trim
