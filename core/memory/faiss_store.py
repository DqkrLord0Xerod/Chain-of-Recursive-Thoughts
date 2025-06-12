from __future__ import annotations

from typing import List, Optional, Dict

import faiss
import numpy as np

from core.interfaces import EmbeddingProvider


class FaissMemoryStore:
    """Simple FAISS based vector store for conversation memory."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        dimension: int,
        *,
        top_k: int = 3,
    ) -> None:
        self.embedding_provider = embedding_provider
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.documents: List[Dict[str, str]] = []
        self.top_k = top_k

    async def add(self, text: str, metadata: Optional[Dict[str, str]] = None) -> None:
        embedding = await self.embedding_provider.embed([text])
        vec = np.array(embedding, dtype="float32")
        self.index.add(vec)
        self.documents.append({"text": text, "metadata": metadata or {}})

    async def search(self, query: str, top_k: Optional[int] = None) -> List[str]:
        if self.index.ntotal == 0:
            return []
        embedding = await self.embedding_provider.embed([query])
        vec = np.array(embedding, dtype="float32")
        k = top_k or self.top_k
        distances, indices = self.index.search(vec, k)
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.documents):
                results.append(self.documents[idx]["text"])
        return results

    async def retrieve_messages(
        self, query: str, top_k: Optional[int] = None
    ) -> List[Dict[str, str]]:
        texts = await self.search(query, top_k)
        return [{"role": "system", "content": t} for t in texts]
