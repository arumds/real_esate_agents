"""
real_estate_agents/shared/mcp_base.py

Thin convenience layer over the official `mcp` Python SDK (FastMCP) so each
agent project can expose its tools as a standard MCP server with minimal
boilerplate.

Why this matters (emerging agent-interoperability pattern):
  Instead of building a bespoke REST API for "check_sqft" or "explain_valuation"
  that only your own orchestrator knows how to call, exposing these as MCP
  tools means ANY MCP-compatible host (Claude Desktop, an internal Claude
  Agent SDK orchestrator, another team's agent) can discover and call them
  using the same protocol -- no custom client code per consumer.

Install:
    pip install mcp

Run a server module directly, e.g.:
    python -m real_estate_agents.data_quality_agent.mcp_server
"""

from __future__ import annotations

from typing import Callable

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover - allows the rest of the repo to be
    # imported/tested even if the `mcp` package isn't installed yet.
    FastMCP = None


def build_mcp_server(name: str, tools: dict[str, Callable]) -> "FastMCP":
    """
    tools: mapping of tool_name -> python callable. The callable's docstring
    and type hints are used by FastMCP to auto-generate the MCP tool schema
    that clients (Claude Desktop, other agents) see.
    """
    if FastMCP is None:
        raise ImportError(
            "The `mcp` package is required to run this as an MCP server. "
            "Install it with: pip install mcp"
        )

    server = FastMCP(name)
    for tool_name, fn in tools.items():
        server.add_tool(fn, name=tool_name)
    return server
