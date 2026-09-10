"""
real_estate_agents/langchain_rag/embeddings.py

Custom LangChain `Embeddings` adapter around this repo's existing
`shared/llm_client.py::embed()` function. This means the LangChain RAG
pipeline reuses the same OpenAI embeddings call (and offline mock-mode
fallback) already used everywhere else in the project, rather than
introducing `langchain-openai` as a second, separate embeddings provider
path with its own key handling.
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from real_estate_agents.shared.llm_client import embed


class SharedLLMEmbeddings(Embeddings):
    """Adapts shared.llm_client.embed() to LangChain's Embeddings interface."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return embed([text])[0]
