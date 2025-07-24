"""Vector store interface + an in-memory implementation.

The interface is small enough that swapping the backend (Qdrant,
Weaviate, pgvector, Pinecone) only requires reimplementing the same
five methods.  In production we rec Qdrant for its mature filters
and HNSW.  For unit tests / local dev the in-memory backend is enough.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Document:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


class VectorStore(Protocol):
    def upsert(self, documents: list[Document], embeddings: np.ndarray) -> None: ...
    def search(self, embedding: np.ndarray, k: int = 5, filter_: dict | None = None) -> list[Document]: ...
    def __len__(self) -> int: ...


@dataclass
class InMemoryVectorStore:
    """Cosine-similarity in-memory store; sufficient for tests + low-volume RAG."""

    _ids: list[str] = field(default_factory=list)
    _texts: list[str] = field(default_factory=list)
    _meta: list[dict] = field(default_factory=list)
    _emb: np.ndarray | None = None

    def upsert(self, documents: list[Document], embeddings: np.ndarray) -> None:
        if len(documents) != len(embeddings):
            raise ValueError("documents and embeddings length mismatch")
        for doc in documents:
            self._ids.append(doc.id)
            self._texts.append(doc.text)
            self._meta.append(doc.metadata)
        embeddings = embeddings / np.maximum(
            np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12
        )
        if self._emb is None:
            self._emb = embeddings
        else:
            self._emb = np.vstack([self._emb, embeddings])

    def search(
        self, embedding: np.ndarray, k: int = 5, filter_: dict | None = None
    ) -> list[Document]:
        if self._emb is None or len(self._ids) == 0:
            return []
        q = embedding / max(float(np.linalg.norm(embedding)), 1e-12)
        sims = self._emb @ q
        order = np.argsort(-sims)
        out: list[Document] = []
        for idx in order:
            if filter_ and not all(self._meta[idx].get(k_) == v for k_, v in filter_.items()):
                continue
            out.append(Document(id=self._ids[idx], text=self._texts[idx], metadata=self._meta[idx]))
            if len(out) >= k:
                break
        return out

    def __len__(self) -> int:
        return len(self._ids)
