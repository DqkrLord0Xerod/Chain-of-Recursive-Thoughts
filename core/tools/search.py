from __future__ import annotations

import asyncio
import requests
from urllib.parse import quote_plus

from .base import Tool


class SearchTool:
    """Simple web search tool using DuckDuckGo."""

    name = "search"
    description = "Retrieve web results for a query"

    async def run(self, task: str) -> str:
        def _search() -> str:
            url = (
                "https://duckduckgo.com/?q=" + quote_plus(task)
            )
            resp = requests.get(url, timeout=5)
            return resp.text[:200]

        return await asyncio.to_thread(_search)

