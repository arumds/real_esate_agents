"""
data_quality_agent/mcp_server.py

Exposes the data quality agent as an MCP server: the individual validation
tools (for other agents/hosts to compose freely) AND a single high-level
`run_data_quality_check` tool (for hosts that just want the final decision).

Run:
    pip install mcp
    python -m data_quality_agent.mcp_server

Then point any MCP-compatible host (Claude Desktop config, another agent's
MCP client) at this server via stdio.
"""

from __future__ import annotations

from data_quality_agent.agent import run_data_quality_check
from data_quality_agent.tools import (
    check_field_completeness,
    compare_reported_vs_authoritative,
    geocode_and_validate_address,
    lookup_county_assessor_record,
)
from shared.mcp_base import build_mcp_server


def run_data_quality_check_tool(record: dict) -> dict:
    """
    Run the full data quality agent pipeline on a property record and return
    its disposition (pass / auto_correct / flag_for_review), any corrected
    fields, and the reasoning behind the decision.

    Args:
        record: raw property record (address, parcel_id, sqft, bedrooms,
            bathrooms, year_built, list_price).
    """
    result = run_data_quality_check(record)
    return {
        "disposition": result.disposition.value,
        "corrected_fields": result.corrected_fields,
        "reasoning": result.reasoning,
        "confidence": result.confidence,
        "final_record": result.final_record,
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
