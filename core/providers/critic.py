"""LLM-based critic for evaluating and improving responses."""

from __future__ import annotations

import json
from typing import Dict, Optional

from core.interfaces import LLMProvider


class CriticLLM:
    """Wrapper that uses an LLM to critique responses."""

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    async def review(self, prompt: str, response: str) -> Dict[str, Optional[str]]:
        """Return a score and optional improved response."""
        review_prompt = {
            "role": "user",
            "content": (
                "Given the prompt and response, rate the response from 0 to 1 "
                "for overall quality. If you can improve it, return an "
                "'improved' version. Respond with JSON like: "
                '{"score": 0.5, "improved": "text"}\n\n'
                f"Prompt: {prompt}\nResponse: {response}"
            ),
        }
        result = await self.llm.chat([review_prompt], temperature=0.0)
        try:
            data = json.loads(result.content)
            score = float(data.get("score", 0))
            improved = data.get("improved")
        except Exception:
            score = 0.0
            improved = None
        return {"score": score, "improved": improved}
