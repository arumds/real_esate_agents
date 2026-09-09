"""
explainable_valuation_agent/mcp_server.py

Exposes the explainable valuation agent as an MCP server so any MCP host
(Claude Desktop, an underwriting dashboard's agent orchestrator, etc.) can
request a grounded narrative explanation for a given valuation model output.

Run:
    pip install mcp
    python -m explainable_valuation_agent.mcp_server
"""

from __future__ import annotations

from explainable_valuation_agent.agent import explain_valuation
from explainable_valuation_agent.rag import retrieve_market_context
from explainable_valuation_agent.shap_utils import FeatureContribution, ValuationExplanationInput
from shared.mcp_base import build_mcp_server


def explain_valuation_tool(
    property_summary: str,
    predicted_price: float,
    base_value: float,
    contributions: list[dict],
    audience: str = "homeowner",
) -> dict:
    """
    Generate a grounded narrative explaining a valuation model's prediction.

    Args:
        property_summary: short description of the property, e.g.
            "123 Oak Ave, Springfield IL, 3bd/2ba, 1900 sqft".
        predicted_price: the model's final predicted value.
        base_value: the model's baseline/average prediction before feature effects.
        contributions: list of {"feature": str, "value": str, "contribution": float}
            SHAP-style attributions.
        audience: one of "underwriter", "homeowner", "appraiser".
    """
    exp_input = ValuationExplanationInput(
        predicted_price=predicted_price,
        base_value=base_value,
        contributions=[
            FeatureContribution(c["feature"], c["value"], c["contribution"]) for c in contributions
        ],
    )
    result = explain_valuation(property_summary, exp_input, audience=audience)  # type: ignore[arg-type]
    return {
        "narrative": result.narrative,
        "grounding_ok": result.grounding_ok,
        "attempts": result.attempts,
        "retrieved_context": result.retrieved_context,
    }


server = build_mcp_server(
    name="real-estate-explainable-valuation-agent",
    tools={
        "retrieve_market_context": retrieve_market_context,
        "explain_valuation": explain_valuation_tool,
    },
)

if __name__ == "__main__":
    server.run()
