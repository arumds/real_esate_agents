"""
real_estate_agents/langchain_rag/retriever.py

A real LangChain RAG pipeline over the same demo corpus used by the
hand-rolled version (explainable_valuation_agent/rag.py::CORPUS):

  1. RecursiveCharacterTextSplitter chunks each source document. (This
     demo corpus's entries are already short, so most stay as a single
     chunk -- the splitter is still wired in correctly because a real
     corpus of full market reports/comp writeups would need it, and this
     is meant to demonstrate the actual production shape of the pipeline.)
  2. FAISS vectorstore, built with SharedLLMEmbeddings so it reuses this
     repo's existing embeddings call instead of a separate provider.
  3. Standard LangChain similarity search over the resulting index.

This is intentionally the ONLY implementation using LangChain's actual RAG
primitives -- the hand-rolled version (explainable_valuation_agent/rag.py)
deliberately keeps its original simple in-memory cosine-similarity store as
a baseline for comparison. See langchain_rag/tool.py for how this gets
exposed as a single tool consumed by both the CrewAI and ADK versions.
"""

from __future__ import annotations

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from real_estate_agents.explainable_valuation_agent.rag import CORPUS
from real_estate_agents.langchain_rag.embeddings import SharedLLMEmbeddings

_vectorstore: FAISS | None = None


def _build_vectorstore() -> FAISS:
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=40)

    documents: list[Document] = []
    for entry in CORPUS:
        chunks = splitter.split_text(entry["text"])
        for i, chunk in enumerate(chunks):
            documents.append(Document(page_content=chunk, metadata={"id": entry["id"], "chunk": i}))

    return FAISS.from_documents(documents, SharedLLMEmbeddings())


def get_vectorstore() -> FAISS:
    """Lazily build the FAISS index once (embeddings computed at first use)."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = _build_vectorstore()
    return _vectorstore
