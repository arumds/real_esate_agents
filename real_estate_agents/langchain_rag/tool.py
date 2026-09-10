"""
real_estate_agents/langchain_rag/tool.py

The actual cross-framework deliverable: ONE LangChain tool for market-
context retrieval, built once here, then consumed by BOTH the CrewAI and
ADK implementations via each framework's own official LangChain
interoperability bridge:

    ADK:    google.adk.tools.langchain_tool.LangchainTool(retrieve_market_context)
    CrewAI: crewai.tools.base_tool.BaseTool.from_langchain(retrieve_market_context)

Neither framework needs a reimplementation of this tool -- that's the point.
See real_estate_agents/tools.py and the CrewAI adapter's tools.py for where each bridge
is used.
"""

from __future__ import annotations

from langchain_core.tools import tool

from real_estate_agents.langchain_rag.retriever import get_vectorstore


@tool
def retrieve_market_context(query: str, top_k: int = 3) -> list[dict]:
    """
    Retrieve the most relevant comps/market-report snippets for a property
    valuation narrative, using a real LangChain RAG pipeline (FAISS vector
    similarity search over a chunked corpus of comp writeups and market
    reports).

    Args:
        query: free-text description of the property/valuation to ground
            against (e.g. address + city + key features).
        top_k: number of chunks to return.

    Returns:
        list of {"id", "text", "score"} dicts, most relevant first. "score"
        is a FAISS L2 distance -- lower is more similar.
    """
    results = get_vectorstore().similarity_search_with_score(query, k=top_k)
    return [
        {"id": doc.metadata.get("id", "unknown"), "text": doc.page_content, "score": round(float(score), 4)}
        for doc, score in results
    ]
