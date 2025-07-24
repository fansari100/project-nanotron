import numpy as np

from nanotron_ml.llm.rag.edgar_rag import EdgarRAG
from nanotron_ml.llm.rag.vector_store import Document, InMemoryVectorStore
from nanotron_ml.llm.sentiment import FinBERTSentiment, _rule_based


def _toy_embed(texts: list[str]) -> np.ndarray:
    """Deterministic toy embedder — encodes presence of 6 keywords."""
    keywords = ["bank", "tech", "growth", "loss", "crypto", "ai"]
    out = np.zeros((len(texts), len(keywords)), dtype=np.float32)
    for i, t in enumerate(texts):
        lower = t.lower()
        for j, kw in enumerate(keywords):
            out[i, j] = lower.count(kw)
    return out


def test_in_memory_store_round_trip():
    store = InMemoryVectorStore()
    docs = [
        Document(id="1", text="bank earnings beat", metadata={"cik": "1"}),
        Document(id="2", text="tech rally on AI growth", metadata={"cik": "2"}),
    ]
    embs = _toy_embed([d.text for d in docs])
    store.upsert(docs, embs)
    assert len(store) == 2

    q = _toy_embed(["AI growth"])[0]
    hits = store.search(q, k=2)
    assert hits[0].id == "2"


def test_in_memory_store_filter_excludes_non_matching():
    store = InMemoryVectorStore()
    docs = [
        Document(id="a", text="bank x", metadata={"cik": "1"}),
        Document(id="b", text="bank y", metadata={"cik": "2"}),
    ]
    store.upsert(docs, _toy_embed([d.text for d in docs]))
    hits = store.search(_toy_embed(["bank"])[0], k=5, filter_={"cik": "2"})
    assert all(h.metadata["cik"] == "2" for h in hits)


def test_edgar_rag_chunks_and_ingests():
    rag = EdgarRAG(
        store=InMemoryVectorStore(),
        embed_fn=_toy_embed,
        chunk_chars=20,
        chunk_overlap=5,
    )
    text = "x" * 80
    n = rag.ingest_filing(cik="1", accession="A", form="10-K", body=text)
    assert n >= 4


def test_finbert_falls_back_to_rule_based_when_transformers_missing():
    s = FinBERTSentiment()
    out = s.predict(["TSLA beats earnings, strong growth"])
    assert {"negative", "neutral", "positive"} == set(out[0].keys())


def test_rule_based_picks_up_obvious_polarity():
    pos = _rule_based("AAPL beats with strong growth")
    neg = _rule_based("AAPL misses, downgrade and lawsuit")
    neutral = _rule_based("AAPL released a press release")
    assert pos["positive"] > pos["negative"]
    assert neg["negative"] > neg["positive"]
    assert neutral["neutral"] > 0.5
