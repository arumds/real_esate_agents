"""
adk_version/main.py

Run the ADK-based pipeline end to end. Requires OPENAI_API_KEY plus
`pip install "google-adk[extensions]"` for the LiteLLM/OpenAI bridge (see
adk_version/agents.py::_default_model), or set ADK_MODEL + GOOGLE_API_KEY to
use a native Gemini model instead.

Run:
    pip install "google-adk[extensions]"
    export OPENAI_API_KEY=sk-...
    python -m adk_version.main
"""

from __future__ import annotations

import asyncio
import json
import uuid

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from adk_version.callbacks import reset_call_log, summarize_call_log
from adk_version.orchestrator import DEFAULT_AUDIENCES, build_multi_audience_pipeline, build_pipeline

APP_NAME = "real_estate_pipeline"

from dotenv import load_dotenv
load_dotenv()
async def run(record: dict, property_summary: str, audience: str = "homeowner") -> None:
    reset_call_log()
    session_service = InMemorySessionService()
    user_id = "demo-user"
    session_id = str(uuid.uuid4())

    # Seed session state with the inputs the agents read via ctx.session.state
    # (see agents.py's instruction referencing 'input_record'/'audience'/
    # 'property_summary', and orchestrator.py reading 'data_quality_decision_raw').
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
        state={
            "input_record": record,
            "audience": audience,
            "property_summary": property_summary,
        },
    )

    pipeline = build_pipeline()
    runner = Runner(app_name=APP_NAME, agent=pipeline, session_service=session_service)

    print("=" * 70)
    print("RUNNING ADK PIPELINE")
    print("=" * 70)

    final_text = None
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=types.Content(
            role="user",
            parts=[types.Part(text=f"Validate and explain this property record: {json.dumps(record)}")],
        ),
    ):
        if event.author:
            snippet = ""
            if event.content and event.content.parts:
                snippet = "".join(p.text or "" for p in event.content.parts)[:200]
            print(f"[{event.author}] {snippet}")
        if event.content and event.content.parts:
            final_text = "".join(p.text or "" for p in event.content.parts) or final_text

    session = await session_service.get_session(app_name=APP_NAME, user_id=user_id, session_id=session_id)

    print("\n" + "=" * 70)
    print("FINAL SESSION STATE")
    print("=" * 70)
    print("data_quality_decision_raw:", session.state.get("data_quality_decision_raw"))
    print("valuation_explanation_raw:", session.state.get("valuation_explanation_raw"))

    print("\n" + "=" * 70)
    print("CALLBACK LOG (tool/model latency)")
    print("=" * 70)
    print(summarize_call_log())


async def run_multi_audience(record: dict, property_summary: str, audiences=DEFAULT_AUDIENCES) -> None:
    """
    Same pipeline, but the valuation-explanation step is a ParallelAgent
    that generates one narrative per audience concurrently instead of a
    single audience read from session state -- see
    orchestrator.py::build_multi_audience_pipeline.
    """
    reset_call_log()
    session_service = InMemorySessionService()
    user_id = "demo-user"
    session_id = str(uuid.uuid4())

    await session_service.create_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
        state={
            "input_record": record,
            "property_summary": property_summary,
        },
    )

    pipeline = build_multi_audience_pipeline(audiences)
    runner = Runner(app_name=APP_NAME, agent=pipeline, session_service=session_service)

    print("=" * 70)
    print(f"RUNNING ADK PIPELINE (parallel narratives for: {', '.join(audiences)})")
    print("=" * 70)

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=types.Content(
            role="user",
            parts=[types.Part(text=f"Validate and explain this property record: {json.dumps(record)}")],
        ),
    ):
        if event.author:
            snippet = ""
            if event.content and event.content.parts:
                snippet = "".join(p.text or "" for p in event.content.parts)[:200]
            print(f"[{event.author}] {snippet}")

    session = await session_service.get_session(app_name=APP_NAME, user_id=user_id, session_id=session_id)

    print("\n" + "=" * 70)
    print("FINAL SESSION STATE (per audience)")
    print("=" * 70)
    print("data_quality_decision_raw:", session.state.get("data_quality_decision_raw"))
    for audience in audiences:
        print(f"valuation_explanation_raw__{audience}:", session.state.get(f"valuation_explanation_raw__{audience}"))

    print("\n" + "=" * 70)
    print("CALLBACK LOG (tool/model latency, all branches)")
    print("=" * 70)
    print(summarize_call_log())


if __name__ == "__main__":
    demo_record = {
        "parcel_id": "PARCEL-10234",
        "address": "123 Maple St, Springfield, IL 62701",
        "sqft": 1840,  # matches the mocked assessor record exactly -> should PASS
        "bedrooms": 3,
        "bathrooms": 2,
        "year_built": 1998,
        "list_price": 289000,
    }
    property_summary = "123 Maple St, Springfield IL, 3bd/2ba, 1840 sqft"

    asyncio.run(run(demo_record, property_summary=property_summary, audience="homeowner"))

    print("\n")
    asyncio.run(run_multi_audience(demo_record, property_summary=property_summary))
