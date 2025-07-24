"""EDGAR-specific RAG.

Wraps a VectorStore so callers can ingest 10-K / 10-Q chunks tagged
with (CIK, accession, item) metadata, then ask "what's new in AAPL's
risk factors since the last 10-K?" — retrieving only relevant chunks
and feeding them to a cheap LLM prompt.

Embeddings are produced by a callable ``embed_fn(list[str]) -> np.ndarray``
the caller passes in — keeps this module model-agnostic (sentence-
transformers, Voyage, OpenAI, local model, etc. all work).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .vector_store import Document, VectorStore


@dataclass
class EdgarRAG:
    store: VectorStore
    embed_fn: Callable[[list[str]], np.ndarray]
    chunk_chars: int = 1500
    chunk_overlap: int = 150

    def ingest_filing(
        self,
        cik: str,
        accession: str,
        form: str,
        body: str,
        item: str | None = None,
    ) -> int:
        chunks = list(self._chunk(body))
        if not chunks:
            return 0
        docs = [
            Document(
                id=f"{cik}/{accession}/{i}",
                text=chunk,
                metadata={"cik": cik, "accession": accession, "form": form, "item": item or "", "chunk_idx": i},
            )
            for i, chunk in enumerate(chunks)
        ]
        embeddings = self.embed_fn([d.text for d in docs])
        self.store.upsert(docs, embeddings)
        return len(docs)

    def search(
        self,
        query: str,
        cik: str | None = None,
        form: str | None = None,
        k: int = 5,
    ) -> list[Document]:
        emb = self.embed_fn([query])[0]
        f: dict | None = None
        if cik or form:
            f = {}
            if cik:
                f["cik"] = cik
            if form:
                f["form"] = form
        return self.store.search(emb, k=k, filter_=f)

    def _chunk(self, body: str):
        if not body:
            return
        n = len(body)
        step = max(1, self.chunk_chars - self.chunk_overlap)
        for start in range(0, n, step):
            yield body[start : start + self.chunk_chars]
            if start + self.chunk_chars >= n:
                return
