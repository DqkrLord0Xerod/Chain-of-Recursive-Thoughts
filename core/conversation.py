"""Manage conversation history for the thinking engine."""

from __future__ import annotations

import json
import time
from typing import Dict, List, Optional

import aiofiles

from core.interfaces import LLMProvider
from core.context_manager import ContextManager
from core.budget import BudgetManager


class ConversationManager:
    """Handles conversation history and persistence."""

    def __init__(
        self,
        llm: LLMProvider,
        context_manager: ContextManager,
        *,
        budget_manager: Optional[BudgetManager] = None,
    ) -> None:
        self.llm = llm
        self.context_manager = context_manager
        self.budget_manager = budget_manager
        self.history: List[Dict[str, str]] = []

    def add(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        self.history = self.context_manager.optimize(self.history)

    def clear(self) -> None:
        self.history.clear()

    def get(self) -> List[Dict[str, str]]:
        return list(self.history)

    async def save(self, filepath: str) -> None:
        data = {
            "conversation": self.history,
            "timestamp": time.time(),
            "metadata": {"model": getattr(self.llm, "model", "unknown")},
        }
        async with aiofiles.open(filepath, "w") as f:
            await f.write(json.dumps(data, indent=2))

    async def load(self, filepath: str) -> None:
        async with aiofiles.open(filepath, "r") as f:
            data = json.loads(await f.read())
        self.history = data.get("conversation", [])

    async def summarize(self) -> str:
        if not self.history:
            return "No conversation yet."
        messages = self.history + [
            {"role": "user", "content": "Summarize the conversation so far in a short paragraph."}
        ]
        response = await self.llm.chat(messages, temperature=0.5)
        if self.budget_manager:
            tokens = response.usage.get("total_tokens", 0)
            self.budget_manager.enforce_limit(tokens)
            self.budget_manager.record_llm_usage(tokens)
        return response.content
