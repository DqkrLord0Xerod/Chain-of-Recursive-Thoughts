from __future__ import annotations

from dataclasses import dataclass

from core.interfaces import LLMProvider

from monitoring.telemetry import generate_request_id

from core.strategies.base import ImprovementPlanner as BaseImprovementPlanner



@dataclass
class ImprovementPlanner(BaseImprovementPlanner):
    """Generate improvement plans for responses."""

    llm: LLMProvider

    async def create_plan(self, prompt: str, current_response: str) -> str:
        """Return an actionable improvement plan."""
        plan_prompt = (
            "You are an assistant that suggests how to improve a response. "
            "Given the user's prompt and the current response, "
            "provide a short numbered list of concrete improvements."
        )
        messages = [
            {"role": "system", "content": plan_prompt},
            {
                "role": "user",
                "content": f"Prompt: {prompt}\nResponse: {current_response}",
            },
        ]
        result = await self.llm.chat(
            messages,
            temperature=0.2,
            metadata={"request_id": generate_request_id()},
        )
        return result.content.strip()
