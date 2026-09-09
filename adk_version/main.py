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

from adk_version.orchestrator import build_pipeline

APP_NAME = "real_estate_pipeline"

from dotenv import load_dotenv
load_dotenv()
async def run(record: dict, property_summary: str, audience: str = "homeowner") -> None:
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
    asyncio.run(
        run(
            demo_record,
            property_summary="123 Maple St, Springfield IL, 3bd/2ba, 1840 sqft",
            audience="homeowner",
        )
    )
