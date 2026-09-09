"""
explainable_valuation_agent/mcp_server.py

Exposes the explainable valuation agent as an MCP server so any MCP host
(Claude Desktop, an underwriting dashboard's agent orchestrator, etc.) can
request a grounded narrative explanation for a property's valuation.

`explain_valuation` runs the same multi-agent composition
real_estate_agents/agent.py::build_pipeline() uses for this step: the ADK
valuation_explainer_agent inside a LoopAgent alongside GroundingCheckerAgent,
so a narrative that fails the grounding check gets retried (up to
MAX_GROUNDING_ATTEMPTS) here too -- not just the bare LlmAgent on its own.

CONTRACT CHANGE from the old hand-rolled version: that one took a
caller-supplied predicted_price/base_value/contributions (assuming you'd
already run a valuation model elsewhere) and only narrated them. ADK's
agent is built to call run_valuation_model itself as part of its own tool
loop (see real_estate_agents/agent.py's valuation-explainer instruction), so this
tool now takes a property `record` and runs both the (mocked) model call
and the narrative step together -- this reflects how the ADK agent actually
works, not a like-for-like drop-in. A caller who already has externally
computed SHAP contributions and only wants narration would need a
different tool shape than this one.

Run:
    pip install mcp "google-adk[extensions]"
    export OPENAI_API_KEY=sk-...   # or ADK_MODEL + GOOGLE_API_KEY for Gemini
    python -m explainable_valuation_agent.mcp_server
"""

from __future__ import annotations

from dotenv import load_dotenv
from google.adk.agents import LoopAgent

from real_estate_agents.agent import build_valuation_explainer_agent
from real_estate_agents.grounding_checker import MAX_GROUNDING_ATTEMPTS, GroundingCheckerAgent
from real_estate_agents.runner_utils import parse_output, run_agent
from explainable_valuation_agent.rag import retrieve_market_context
from shared.mcp_base import build_mcp_server

load_dotenv()


def _build_valuation_loop() -> LoopAgent:
    """Same (explainer, grounding_checker) pairing as
    real_estate_agents/agent.py::build_pipeline() -- built fresh per call so
    concurrent MCP requests don't share agent/session state."""
    return LoopAgent(
        name="valuation_explanation_loop",
        sub_agents=[build_valuation_explainer_agent(), GroundingCheckerAgent(name="grounding_checker")],
        max_iterations=MAX_GROUNDING_ATTEMPTS,
    )


def explain_valuation_tool(
    record: dict,
    property_summary: str,
    audience: str = "homeowner",
) -> dict:
    """
    Run the (mocked) valuation model on a property record and generate a
    grounded, audience-tailored narrative explaining the result. Retries
    the narrative (up to MAX_GROUNDING_ATTEMPTS) if it fails the grounding
    check -- see GroundingCheckerAgent.

    Args:
        record: a cleaned property record with at least 'sqft' -- typically
            the 'final_record' from run_data_quality_check.
        property_summary: short description used as the RAG query, e.g.
            "123 Oak Ave, Springfield IL, 3bd/2ba, 1900 sqft".
        audience: one of "underwriter", "homeowner", "appraiser".
    """
    valuation_loop = _build_valuation_loop()
    final_state = run_agent(
        valuation_loop,
        state={
            # The valuation explainer's instruction reads this key expecting
            # a DataQualityDecision shape; seeding disposition="pass" here
            # is what tells it to proceed straight to calling
            # run_valuation_model instead of halting for review.
            "data_quality_decision_raw": {
                "disposition": "pass",
                "corrected_fields": {},
                "reasoning": "",
                "confidence": 1.0,
                "final_record": record,
            },
            "property_summary": property_summary,
            "audience": audience,
        },
        trigger_text=f"Explain the valuation for: {property_summary}",
    )
    explanation = parse_output(final_state, "valuation_explanation_raw") or {}
    return {
        "narrative": explanation.get("narrative", ""),
        "predicted_price": explanation.get("predicted_price", 0.0),
        "base_value": explanation.get("base_value", 0.0),
        "contributions": explanation.get("contributions", []),
        "halted": explanation.get("halted", False),
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
