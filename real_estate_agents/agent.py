"""
real_estate_agents/agent.py

The whole ADK multi-agent system in one file: the two LlmAgents, the
deterministic orchestrator that wires them together, and `root_agent` --
the module-level variable ADK's own tooling (`adk web`, `adk run`,
`adk deploy`) auto-discovers by importing this file.

Structure:
  1. Agent builders -- build_data_quality_agent(), build_valuation_explainer_agent()
     Two ADK LlmAgents (google.adk.agents.Agent), each with:
       - `tools`: the same underlying functions as the other two implementations
       - `output_schema`: forces the final answer into our pydantic shape
       - `output_key`: writes the validated final answer into session state
         under this key, which is how ADK agents in a pipeline hand data to
         each other (read via ctx.session.state[key] in the orchestrator/
         checker, or via a dynamic `instruction` callback for the next
         LlmAgent)
       - callbacks (see callbacks.py) for tool/model latency logging, plus
         an input-validation guardrail on the valuation explainer

     The valuation explainer's `instruction` is a callable (not a plain
     string) so it can read `grounding_feedback` out of session state on
     retry -- see grounding_checker.py for where that gets set. This is the
     ADK-idiomatic version of "generate -> critique -> feed feedback back ->
     regenerate".

  2. PipelineOrchestratorAgent -- the manager-equivalent. Unlike an
     LLM-driven manager (e.g. a hierarchical-process delegation pattern,
     where routing is the manager agent's judgment call), this is a plain
     Python `if` statement inside a custom BaseAgent's _run_async_impl:
     "never let a flagged record reach the valuation step" is enforced
     deterministically, while the two subtasks that actually need reasoning
     (data validation judgment, narrative generation) stay LLM-driven. For
     a regulated pipeline, that's arguably the safer choice.

  3. build_pipeline() / build_multi_audience_pipeline() -- assemble the
     LlmAgents + PipelineOrchestratorAgent (+ ParallelAgent, for the
     multi-audience variant) into a runnable multi-agent system.

  4. root_agent -- the single-audience pipeline, exposed at module scope so
     `adk web`/`adk run` can find and drive it without a custom script.
     Those tools seed session state from your chat message, not from a
     Python dict the way `main.py`'s create_session(..., state={...}) does
     -- DATA_QUALITY_INSTRUCTION below tells the model to read the record
     directly from the conversation when 'input_record' isn't already in
     session state, so root_agent works through `adk web`/`adk run` too.
"""

from __future__ import annotations

import json
import os
from typing import AsyncGenerator

from google.adk.agents import Agent, BaseAgent, LoopAgent, ParallelAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.events.event import Event
from google.genai import types

from real_estate_agents.callbacks import (
    after_model_log,
    after_tool_log,
    before_model_log,
    before_tool_log,
    validate_valuation_input,
)
from real_estate_agents.grounding_checker import MAX_GROUNDING_ATTEMPTS, GroundingCheckerAgent
from real_estate_agents.schemas import DataQualityDecision, ValuationExplanation
from real_estate_agents.tools import DATA_QUALITY_TOOLS, VALUATION_EXPLAINER_TOOLS

DEFAULT_AUDIENCES = ("homeowner", "underwriter", "appraiser")


# ---------------------------------------------------------------------------
# 1. Agent builders
# ---------------------------------------------------------------------------


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
If that key isn't set, look for a property record described directly in
the user's message. The record is a JSON object with these fields:
  - address: string, e.g. "123 Oak Ave, Springfield IL"
  - parcel_id: string, e.g. "17-34-567-890"
  - sqft: integer, e.g. 1900
  - bedrooms: integer, e.g. 3
  - bathrooms: integer, e.g. 2
  - year_built: integer, e.g. 1998

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
        before_tool_callback=before_tool_log,
        after_tool_callback=after_tool_log,
        before_model_callback=before_model_log,
        after_model_callback=after_model_log,
    )


def _make_valuation_instruction(fixed_audience: str | None = None):
    """
    Returns an instruction callable for the valuation explainer agent.

    `fixed_audience=None` is the single-agent path: audience comes from
    shared session state (key 'audience') and grounding retry feedback
    lives under the shared key 'grounding_feedback'.

    A non-None `fixed_audience` is for the multi-audience ParallelAgent
    branch (see build_multi_audience_pipeline below): several of these
    agents run concurrently, one per audience, so each needs its audience
    baked in rather than read from a session-state key they'd all share,
    and its own feedback key so one branch's retry doesn't leak into
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
    -- used by build_pipeline().

    audience="homeowner"/"underwriter"/"appraiser": a fixed-audience branch
    with a unique name/output_key, so N of these can run concurrently under
    a ParallelAgent without clobbering each other's session state -- used by
    build_multi_audience_pipeline().
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
        # before_tool_log always runs first (always returns None, so it never
        # short-circuits the chain) and stamps a start time; validate_valuation_input
        # runs second and may short-circuit run_valuation_model if its input is bad.
        before_tool_callback=[before_tool_log, validate_valuation_input],
        after_tool_callback=after_tool_log,
        before_model_callback=before_model_log,
        after_model_callback=after_model_log,
    )


# ---------------------------------------------------------------------------
# 2. Orchestrator -- deterministic routing between the two agents above
# ---------------------------------------------------------------------------


class PipelineOrchestratorAgent(BaseAgent):
    """
    Deterministically runs the data quality agent, then either runs the
    valuation-explanation step or halts -- based on a plain Python check of
    the data quality agent's disposition, not an LLM's judgment call.

    `valuation_loop` is typed as plain BaseAgent (not LoopAgent) because it
    can be either a single LoopAgent (build_pipeline, one audience) or a
    ParallelAgent of several LoopAgents, one per audience
    (build_multi_audience_pipeline) -- this orchestrator's control flow
    doesn't care which, it just runs whatever valuation step it's handed
    after the gate check passes.
    """

    data_quality_agent: Agent
    valuation_loop: BaseAgent

    def __init__(self, name: str, data_quality_agent: Agent, valuation_loop: BaseAgent):
        # Sub-agents invoked directly (self.xxx.run_async(ctx)) must still be
        # declared in `sub_agents` for ADK's lifecycle/introspection features,
        # per ADK's custom-agent guidance, even though this class's own
        # control flow (not automatic sequencing) decides when each runs.
        super().__init__(
            name=name,
            data_quality_agent=data_quality_agent,
            valuation_loop=valuation_loop,
            sub_agents=[data_quality_agent, valuation_loop],
        )

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        async for event in self.data_quality_agent.run_async(ctx):
            yield event

        raw = ctx.session.state.get("data_quality_decision_raw")
        decision = json.loads(raw) if isinstance(raw, str) else raw

        if decision is None or decision.get("disposition") == "flag_for_review":
            yield Event(
                author=self.name,
                content=types.Content(
                    role="model",
                    parts=[types.Part(text=(
                        "Pipeline halted: data quality check returned "
                        f"'flag_for_review'. Reasoning: {decision.get('reasoning') if decision else 'no decision produced'}. "
                        "Valuation step was NOT run."
                    ))],
                ),
            )
            return

        async for event in self.valuation_loop.run_async(ctx):
            yield event


# ---------------------------------------------------------------------------
# 3. Pipeline assembly
# ---------------------------------------------------------------------------


def build_pipeline() -> PipelineOrchestratorAgent:
    data_quality_agent = build_data_quality_agent()
    valuation_explainer_agent = build_valuation_explainer_agent()
    grounding_checker = GroundingCheckerAgent(name="grounding_checker")

    valuation_loop = LoopAgent(
        name="valuation_explanation_loop",
        sub_agents=[valuation_explainer_agent, grounding_checker],
        max_iterations=MAX_GROUNDING_ATTEMPTS,
    )

    return PipelineOrchestratorAgent(
        name="real_estate_pipeline",
        data_quality_agent=data_quality_agent,
        valuation_loop=valuation_loop,
    )


def build_multi_audience_pipeline(audiences: tuple[str, ...] = DEFAULT_AUDIENCES) -> PipelineOrchestratorAgent:
    """
    Variant of build_pipeline() that generates the valuation narrative for
    several audiences concurrently, via a ParallelAgent, instead of one
    audience read from session state.

    Each audience gets its own (valuation_explainer_agent, grounding_checker)
    LoopAgent -- same generate/self-critique/retry logic as the single-agent
    path, just parametrized per branch (build_valuation_explainer_agent and
    GroundingCheckerAgent both take audience-specific keys) so concurrent
    branches don't clobber each other's session state. The gating rule is
    unchanged: PipelineOrchestratorAgent still decides in plain Python
    whether to run this ParallelAgent at all.

    This is a genuine fan-out/fan-in use of ParallelAgent: the branches are
    fully independent (each just needs 'data_quality_decision_raw' and
    'property_summary', both already in session state before this runs) and
    their results don't need to be merged into a single answer -- each
    branch's output is read separately from the audience-keyed
    'valuation_explanation_raw__<audience>' session-state entries afterward.
    """
    data_quality_agent = build_data_quality_agent()

    audience_loops = []
    for audience in audiences:
        explainer = build_valuation_explainer_agent(audience)
        checker = GroundingCheckerAgent(
            name=f"grounding_checker__{audience}",
            valuation_output_key=f"valuation_explanation_raw__{audience}",
            feedback_key=f"grounding_feedback__{audience}",
        )
        audience_loops.append(
            LoopAgent(
                name=f"valuation_explanation_loop__{audience}",
                sub_agents=[explainer, checker],
                max_iterations=MAX_GROUNDING_ATTEMPTS,
            )
        )

    valuation_parallel = ParallelAgent(
        name="valuation_explanation_parallel",
        sub_agents=audience_loops,
    )

    return PipelineOrchestratorAgent(
        name="real_estate_pipeline_multi_audience",
        data_quality_agent=data_quality_agent,
        valuation_loop=valuation_parallel,
    )


# ---------------------------------------------------------------------------
# 4. root_agent -- what `adk web` / `adk run` / `adk deploy` auto-discover
# ---------------------------------------------------------------------------

root_agent = build_pipeline()
