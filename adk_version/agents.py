"""
adk_version/agents.py

Two ADK LlmAgents (google.adk.agents.Agent), each with:
  - `tools`: the same underlying functions as the other two implementations
  - `output_schema`: forces the final answer into our pydantic shape (ADK
    2.8+ supports tools and output_schema together -- confirmed against the
    installed package: "exposing tools during the thought loop and
    enforcing structure only on the final output")
  - `output_key`: writes the validated final answer into session state
    under this key, which is how ADK agents in a pipeline hand data to each
    other (read via ctx.session.state[key] in the orchestrator/checker, or
    via a dynamic `instruction` callback for the next LlmAgent)

The valuation explainer's `instruction` is a callable (not a plain string)
so it can read `grounding_feedback` out of session state on retry -- see
grounding_checker.py for where that gets set. This is the ADK-idiomatic
version of the "generate -> critique -> feed feedback back -> regenerate"
loop.
"""

from __future__ import annotations

import os

from google.adk.agents import Agent
from google.adk.agents.readonly_context import ReadonlyContext

from adk_version.schemas import DataQualityDecision, ValuationExplanation
from adk_version.tools import DATA_QUALITY_TOOLS, VALUATION_EXPLAINER_TOOLS


def _default_model():
    """
    ADK defaults to Gemini. To stay consistent with the rest of this repo
    (which uses OPENAI_API_KEY everywhere), default to ADK's LiteLLM
    integration running the same OpenAI model -- requires:
        pip install "google-adk[extensions]"
    Override with the ADK_MODEL env var to use a native Gemini model string
    instead (e.g. "gemini-2.0-flash"), which needs GOOGLE_API_KEY instead.
    """
    override = os.getenv("ADK_MODEL")
    if override:
        return override
    try:
        from google.adk.models.lite_llm import LiteLlm
    except ImportError as e:
        raise ImportError(
            'ADK\'s OpenAI support needs the LiteLLM extra: pip install '
            '"google-adk[extensions]". Alternatively set ADK_MODEL to a '
            "native Gemini model string (e.g. 'gemini-2.0-flash') and use "
            "GOOGLE_API_KEY instead of OPENAI_API_KEY."
        ) from e
    return LiteLlm(model=os.getenv("OPENAI_MODEL", "openai/gpt-4o-mini"))


DATA_QUALITY_INSTRUCTION = """\
You are a data quality analyst for a real estate valuation pipeline.

Validate the property record in session state under key 'input_record'.
First reason about field completeness/plausibility. If a parcel_id is
present, ALWAYS cross-check the reported sqft against the county assessor
record via your tools, even if the record looks superficially fine --
plausible values can still be wrong.

Decide a final disposition:
  - "pass": no material issues remain.
  - "auto_correct": ONLY if a secondary source gives a confident,
    unambiguous replacement value.
  - "flag_for_review": ambiguous evidence, disagreeing sources, or no
    secondary source available to confirm a suspicious value.

Respond with a DataQualityDecision: disposition, corrected_fields (leave
fields null/unset if unchanged -- only set the ones you actually
corrected), reasoning, confidence, and final_record (the full record to
pass downstream, with all fields populated -- corrections applied if
disposition is auto_correct, otherwise identical to the input).
"""


def build_data_quality_agent() -> Agent:
    return Agent(
        name="data_quality_agent",
        model=_default_model(),
        description="Validates and corrects property records before they reach the valuation model.",
        instruction=DATA_QUALITY_INSTRUCTION,
        tools=DATA_QUALITY_TOOLS,
        output_schema=DataQualityDecision,
        output_key="data_quality_decision_raw",
    )


def _make_valuation_instruction(fixed_audience: str | None = None):
    """
    Returns an instruction callable for the valuation explainer agent.

    `fixed_audience=None` is the original single-agent path: audience comes
    from shared session state (key 'audience') and grounding retry feedback
    lives under the shared key 'grounding_feedback'.

    A non-None `fixed_audience` is for the multi-audience ParallelAgent
    branch (see orchestrator.py::build_multi_audience_pipeline): several of
    these agents run concurrently, one per audience, so each needs its
    audience baked in rather than read from a session-state key they'd all
    share, and its own feedback key so one branch's retry doesn't leak into
    another's.
    """

    def _instruction(context: ReadonlyContext) -> str:
        audience = fixed_audience or context.state.get("audience", "homeowner")
        property_summary = context.state.get("property_summary", "")
        feedback_key = f"grounding_feedback__{fixed_audience}" if fixed_audience else "grounding_feedback"

        base = (
            "You are a valuation explainability specialist. Read "
            "'data_quality_decision_raw' from session state -- it is a "
            "DataQualityDecision JSON object.\n\n"
            "If its disposition is 'flag_for_review', respond with a "
            "ValuationExplanation where halted=true, predicted_price=0, "
            "base_value=0, contributions=[], and narrative explains the "
            "pipeline stopped for human review. Do NOT call any tools in this "
            "case.\n\n"
            "Otherwise: call run_valuation_model on the 'final_record' field "
            f"from that decision. Then call retrieve_market_context with this "
            f"property summary for color: '{property_summary}'. Write a "
            f"narrative explanation for this audience: {audience}. Use ONLY "
            "the dollar figures the run_valuation_model tool actually "
            "returned -- never invent or round differently. You may reference "
            "retrieved market context for color, but never state a comp's sale "
            "price as if it were a contribution to THIS property's valuation. "
            "Echo back the exact predicted_price, base_value, and "
            "contributions you used, matching the tool's return value exactly."
        )

        feedback = context.state.get(feedback_key)
        if feedback:
            base += (
                f"\n\nYOUR PREVIOUS ATTEMPT FAILED A GROUNDING CHECK: {feedback} "
                "Rewrite the narrative using ONLY the figures listed as valid."
            )

        return base

    return _instruction


def build_valuation_explainer_agent(audience: str | None = None) -> Agent:
    """
    audience=None (default): single agent, audience read from session state
    -- used by orchestrator.py::build_pipeline.

    audience="homeowner"/"underwriter"/"appraiser": a fixed-audience branch
    with a unique name/output_key, so N of these can run concurrently under
    a ParallelAgent without clobbering each other's session state -- used by
    orchestrator.py::build_multi_audience_pipeline.
    """
    suffix = f"__{audience}" if audience else ""
    return Agent(
        name=f"valuation_explainer_agent{suffix}",
        model=_default_model(),
        description="Runs the valuation model and explains its output for a given audience.",
        instruction=_make_valuation_instruction(audience),
        tools=VALUATION_EXPLAINER_TOOLS,
        output_schema=ValuationExplanation,
        output_key=f"valuation_explanation_raw{suffix}",
    )
