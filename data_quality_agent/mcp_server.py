"""
data_quality_agent/mcp_server.py

Exposes the data quality agent as an MCP server: the individual validation
tools (for other agents/hosts to compose freely) AND a single high-level
`run_data_quality_check` tool (for hosts that just want the final decision).

`run_data_quality_check` runs the real ADK LlmAgent
(real_estate_agents.agent.build_data_quality_agent) via real_estate_agents/runner_utils.py,
rather than a second hand-rolled agent loop -- one agent implementation,
reused here instead of duplicated. The four raw validators below are still
plain functions from tools.py, the same single source of truth ADK's
FunctionTool wrappers (real_estate_agents/tools.py) wrap underneath.

Run:
    pip install mcp "google-adk[extensions]"
    export OPENAI_API_KEY=sk-...   # or ADK_MODEL + GOOGLE_API_KEY for Gemini
    python -m data_quality_agent.mcp_server

Then point any MCP-compatible host (Claude Desktop config, another agent's
MCP client) at this server via stdio.
"""

from __future__ import annotations

from real_estate_agents.agent import build_data_quality_agent
from real_estate_agents.runner_utils import parse_output, run_agent
from data_quality_agent.tools import (
    check_field_completeness,
    compare_reported_vs_authoritative,
    geocode_and_validate_address,
    lookup_county_assessor_record,
)
from shared.mcp_base import build_mcp_server


def run_data_quality_check_tool(record: dict) -> dict:
    """
    Run the data quality agent on a property record and return its
    disposition (pass / auto_correct / flag_for_review), any corrected
    fields, and the reasoning behind the decision.

    Args:
        record: raw property record (address, parcel_id, sqft, bedrooms,
            bathrooms, year_built, list_price).
    """
    agent = build_data_quality_agent()
    final_state = run_agent(
        agent,
        state={"input_record": record},
        trigger_text=f"Validate and explain this property record: {record}",
    )
    decision = parse_output(final_state, "data_quality_decision_raw") or {}
    return {
        "disposition": decision.get("disposition", "flag_for_review"),
        "corrected_fields": decision.get("corrected_fields", {}),
        "reasoning": decision.get("reasoning", "Agent produced no decision."),
        "confidence": decision.get("confidence", 0.0),
        "final_record": decision.get("final_record", record),
    }


server = build_mcp_server(
    name="real-estate-data-quality-agent",
    tools={
        "check_field_completeness": check_field_completeness,
        "geocode_and_validate_address": geocode_and_validate_address,
        "lookup_county_assessor_record": lookup_county_assessor_record,
        "compare_reported_vs_authoritative": compare_reported_vs_authoritative,
        "run_data_quality_check": run_data_quality_check_tool,
    },
)

if __name__ == "__main__":
    server.run()
