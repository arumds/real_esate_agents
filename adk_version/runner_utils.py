"""
adk_version/runner_utils.py

Small helper for running a single ADK agent as if it were a plain function
call: seed session state, run to completion, return the final state dict.

This is the boundary between ADK's async Runner/SessionService world and
callers that just want "give me a dict back" -- the MCP servers in
data_quality_agent/ and explainable_valuation_agent/, and
eval/run_golden_eval.py, all use this instead of reimplementing the
Runner/session boilerplate that adk_version/main.py has for the full
pipeline demo.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from google.adk.agents import BaseAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

APP_NAME = "real_estate_pipeline"


async def _run_agent_async(agent: BaseAgent, state: dict[str, Any], trigger_text: str) -> dict[str, Any]:
    session_service = InMemorySessionService()
    user_id = "mcp-client"
    session_id = str(uuid.uuid4())

    await session_service.create_session(app_name=APP_NAME, user_id=user_id, session_id=session_id, state=state)

    runner = Runner(app_name=APP_NAME, agent=agent, session_service=session_service)

    async for _ in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=types.Content(role="user", parts=[types.Part(text=trigger_text)]),
    ):
        pass  # side effects land in session state; we only need the final state

    session = await session_service.get_session(app_name=APP_NAME, user_id=user_id, session_id=session_id)
    return dict(session.state)


def run_agent(agent: BaseAgent, state: dict[str, Any], trigger_text: str) -> dict[str, Any]:
    """
    Synchronous entry point: run `agent` once in a fresh session seeded with
    `state`, and return the final session state. FastMCP tool functions
    (and the golden eval loop) are plain sync callables -- this wraps the
    asyncio.run() so callers don't have to.
    """
    return asyncio.run(_run_agent_async(agent, state, trigger_text))


def parse_output(session_state: dict[str, Any], key: str) -> dict[str, Any] | None:
    """output_key values come back as either a JSON string or an already-parsed
    dict depending on ADK's internal serialization path -- normalize both."""
    raw = session_state.get(key)
    if raw is None:
        return None
    return json.loads(raw) if isinstance(raw, str) else raw
