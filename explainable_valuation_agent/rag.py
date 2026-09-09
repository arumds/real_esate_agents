"""
explainable_valuation_agent/rag.py

Minimal in-memory RAG layer: embeds a small corpus of market reports and
comparable-sale writeups, and retrieves the top-k most relevant chunks for a
given property + valuation context. In production, swap `SimpleVectorStore`
for Chroma/pgvector/Pinecone and point `CORPUS` at a real ingestion pipeline
(MLS comp exports, market report PDFs via a document loader) -- the retrieval
interface (`retrieve`) stays the same, so the agent code doesn't change.

Concept coverage:
  - "RAG": grounds the LLM's narrative in real market context (recent comps,
    neighborhood trends) rather than letting it hallucinate market commentary
    the SHAP values alone can't provide.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from shared.llm_client import embed

# Small demo corpus: comp writeups + market commentary. In production this
# would be populated by an ingestion job (MLS feed, market report scraper).
CORPUS: list[dict] = [
    {
        "id": "comp-1001",
        "text": (
            "123 Oak Ave, Springfield IL sold for $312,000 in June 2026, "
            "1,900 sqft, 3 bed / 2 bath, built 1999, renovated kitchen. "
            "Sold in 11 days, above asking by 2%."
        ),
    },
    {
        "id": "comp-1002",
        "text": (
            "145 Maple St, Springfield IL sold for $278,500 in May 2026, "
            "1,820 sqft, 3 bed / 2 bath, built 1997, original kitchen/baths. "
            "Sold in 34 days, 3% under asking."
        ),
    },
    {
        "id": "market-report-2026-q2",
        "text": (
            "Springfield IL market report Q2 2026: median days-on-market "
            "fell to 18 days, down from 27 in Q1. Inventory remains tight in "
            "the Maple/Oak corridor. Renovated kitchens are commanding a "
            "5-8% premium over comparable unrenovated homes this quarter."
        ),
    },
    {
        "id": "comp-2001",
        "text": (
            "88 Lakeview Dr, Austin TX sold for $645,000 in July 2026, "
            "2,650 sqft, 4 bed / 3 bath, built 2005, pool. Multiple offers, "
            "closed 4% over asking within 9 days."
        ),
    },
    {
        "id": "market-report-austin-2026-q2",
        "text": (
            "Austin TX market report Q2 2026: single-family inventory up "
            "12% YoY easing price growth to 3.1% annually. Homes with a "
            "pool in the Lakeview area continue to outperform, selling "
            "roughly 6% faster than the metro median."
        ),
    },
]


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass
class RetrievedChunk:
    id: str
    text: str
    score: float


class SimpleVectorStore:
    """In-memory embedding index. Swap for Chroma/pgvector at scale."""

    def __init__(self, corpus: list[dict]):
        self.corpus = corpus
        self._embeddings = embed([c["text"] for c in corpus])

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        query_vec = embed([query])[0]
        scored = [
            RetrievedChunk(id=c["id"], text=c["text"], score=_cosine_sim(query_vec, emb))
            for c, emb in zip(self.corpus, self._embeddings)
        ]
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]


_store: SimpleVectorStore | None = None


def get_store() -> SimpleVectorStore:
    """Lazily build the vector store once (embeddings are computed at first use)."""
    global _store
    if _store is None:
        _store = SimpleVectorStore(CORPUS)
    return _store


def retrieve_market_context(query: str, top_k: int = 3) -> list[dict]:
    """
    Retrieve the most relevant comps/market-report snippets for a property
    valuation narrative.

    Args:
        query: free-text description of the property/valuation to ground
            against (e.g. address + city + key features).
        top_k: number of chunks to return.

    Returns:
        list of {"id", "text", "score"} dicts, most relevant first.
    """
    chunks = get_store().retrieve(query, top_k=top_k)
    return [{"id": c.id, "text": c.text, "score": round(c.score, 4)} for c in chunks]
