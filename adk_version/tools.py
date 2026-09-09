"""
adk_version/tools.py

Wraps the SAME underlying functions used by both the hand-rolled and CrewAI
versions as ADK tools via `google.adk.tools.FunctionTool`. No logic
duplication -- pure adapter layer, same as crewai_version/tools.py.

FunctionTool derives the tool's parameter schema from the function's type
hints and its natural-language description from the docstring (verified:
wrapping data_quality_agent.tools.check_field_completeness produces a
correct schema and description with zero modification to that function).

retrieve_market_context is different: it's the actual LangChain tool from
langchain_rag/tool.py (a real FAISS-backed RAG pipeline), imported via
ADK's own first-party LangChain bridge (google.adk.tools.langchain_tool.
LangchainTool) rather than reimplemented as an ADK FunctionTool. The same
LangChain tool object is also consumed by crewai_version/tools.py via
CrewAI's bridge -- built once, used natively by both frameworks.
"""

from __future__ import annotations

from google.adk.tools import FunctionTool
from google.adk.tools.langchain_tool import LangchainTool

from data_quality_agent.tools import (
    check_field_completeness as _check_field_completeness,
    compare_reported_vs_authoritative as _compare_reported_vs_authoritative,
    geocode_and_validate_address as _geocode_and_validate_address,
    lookup_county_assessor_record as _lookup_county_assessor_record,
)
from langchain_rag.tool import retrieve_market_context as _lc_retrieve_market_context

check_field_completeness = FunctionTool(_check_field_completeness)
geocode_and_validate_address = FunctionTool(_geocode_and_validate_address)
lookup_county_assessor_record = FunctionTool(_lookup_county_assessor_record)
compare_reported_vs_authoritative = FunctionTool(_compare_reported_vs_authoritative)
retrieve_market_context = LangchainTool(_lc_retrieve_market_context)


def run_valuation_model_fn(record: dict) -> dict:
    """
    Run the (mocked) valuation ML model + SHAP explainer on a cleaned
    property record. Replace the body with a real call to your model +
    `shap.TreeExplainer(model).shap_values(X)` in production.

    Args:
        record: a cleaned property record (post data-quality checks) with
            at least 'sqft'.

    Returns:
        dict with 'predicted_price', 'base_value', and 'contributions'
        (list of {feature, value, contribution}). Mirrors the same toy
        model used in crewai_version/tools.py and orchestrate_pipeline.py
        so all three implementations are directly comparable.
    """
    base_value = 265000
    sqft_contribution = (record["sqft"] - 1800) * 42
    predicted = base_value + sqft_contribution + 14100 + 9200 - 1500

    return {
        "predicted_price": predicted,
        "base_value": base_value,
        "contributions": [
            {"feature": "living_area_sqft", "value": f"{record['sqft']:,} sqft", "contribution": sqft_contribution},
            {"feature": "neighborhood", "value": "Maple/Oak corridor", "contribution": 14100},
            {"feature": "renovated_kitchen", "value": "yes", "contribution": 9200},
            {"feature": "lot_size", "value": "0.18 acres", "contribution": -1500},
        ],
    }


run_valuation_model = FunctionTool(run_valuation_model_fn)

DATA_QUALITY_TOOLS = [
    check_field_completeness,
    geocode_and_validate_address,
    lookup_county_assessor_record,
    compare_reported_vs_authoritative,
]

VALUATION_EXPLAINER_TOOLS = [
    run_valuation_model,
    retrieve_market_context,
]
