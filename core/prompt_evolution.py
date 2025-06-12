from __future__ import annotations

from typing import List


def evolve_prompt(prompt: str, history: List[str]) -> str:
    """Refine the prompt based on previous prompts."""
    if not history:
        return prompt
    previous = history[-1]
    return f"{previous} -> {prompt}"
