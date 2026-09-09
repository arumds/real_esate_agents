"""
adk_version/orchestrator.py

The manager-equivalent for this pipeline -- but unlike crewai_version, where
routing is an LLM's judgment call (the manager agent decides whether to
delegate the valuation task), here it's a plain Python `if` statement inside
a custom BaseAgent's _run_async_impl.

This is a genuine, worth-noting difference between the two frameworks:
CrewAI's hierarchical process only gives you LLM-driven delegation, so
enforcing "never let a flagged record reach the valuation step" means
trusting a manager agent's judgment (backed by its goal/backstory prompt).
ADK lets you drop into deterministic Python for exactly this kind of
business rule, while still using the LLM agents for the two subtasks that
actually need reasoning (data validation judgment, narrative generation).
For a regulated pipeline, this is arguably the safer choice -- worth
weighing against CrewAI's simplicity if this rule must never be violated.
"""

from __future__ import annotations

import json
from typing import AsyncGenerator

from google.adk.agents import Agent, BaseAgent, LoopAgent, ParallelAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.genai import types

from adk_version.agents import build_data_quality_agent, build_valuation_explainer_agent
from adk_version.grounding_checker import MAX_GROUNDING_ATTEMPTS, GroundingCheckerAgent

DEFAULT_AUDIENCES = ("homeowner", "underwriter", "appraiser")


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
    path, just parametrized per branch (agents.py::build_valuation_explainer_agent
    and grounding_checker.py::GroundingCheckerAgent both take the audience-
    specific keys) so concurrent branches don't clobber each other's session
    state. The gating rule is unchanged: PipelineOrchestratorAgent still
    decides in plain Python whether to run this ParallelAgent at all.

    This is a genuine fan-out/fan-in use of ParallelAgent: the branches are
    fully independent (each just needs 'data_quality_decision_raw' and
    'property_summary', both already in session state before this runs) and
    their results don't need to be merged into a single answer -- unlike
    SequentialAgent's one-child cast, each branch's output is read
    separately from the audience-keyed 'valuation_explanation_raw__<audience>'
    session-state entries afterward.
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
