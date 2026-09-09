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

from google.adk.agents import Agent, BaseAgent, LoopAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.genai import types

from adk_version.agents import build_data_quality_agent, build_valuation_explainer_agent
from adk_version.grounding_checker import MAX_GROUNDING_ATTEMPTS, GroundingCheckerAgent


class PipelineOrchestratorAgent(BaseAgent):
    """
    Deterministically runs the data quality agent, then either runs the
    valuation-explanation loop or halts -- based on a plain Python check of
    the data quality agent's disposition, not an LLM's judgment call.
    """

    data_quality_agent: Agent
    valuation_loop: LoopAgent

    def __init__(self, name: str, data_quality_agent: Agent, valuation_loop: LoopAgent):
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
