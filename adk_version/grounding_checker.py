"""
adk_version/grounding_checker.py

ADK has no declarative "guardrail with automatic retry" the way CrewAI's
Task.guardrail does. The idiomatic ADK pattern for "generate -> check ->
retry on failure" is a LoopAgent containing [generator_agent, checker_agent],
where the checker is a plain (non-LLM) custom BaseAgent that:
  - reads the generator's last output from session state
  - if it passes, yields an Event with actions.escalate=True, which breaks
    the LoopAgent
  - if it fails, writes feedback into session state (read by the
    generator's next instruction, see agents.py::_valuation_instruction)
    and does NOT escalate, so the loop runs the generator again

This is genuinely more manual than CrewAI's one-line `guardrail=` field --
that's a real trade-off of this framework choice, not a limitation of this
implementation. See eval note in README for how this maps to the same
grounding logic used in shap_utils.py and crewai_version/tasks.py.
"""

from __future__ import annotations

import json
import re
from typing import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.genai import types

from adk_version.schemas import ValuationExplanation

MAX_GROUNDING_ATTEMPTS = 3  # combined with LoopAgent's max_iterations


def _check_grounding(parsed: ValuationExplanation) -> tuple[bool, str]:
    """Same logic as shap_utils.check_narrative_grounding / crewai_version's grounding_guardrail."""
    if parsed.halted:
        return True, ""

    known_values = {round(parsed.predicted_price), round(parsed.base_value)}
    for c in parsed.contributions:
        known_values.add(round(abs(c.contribution)))

    mentioned = {int(m.replace(",", "")) for m in re.findall(r"\$([\d,]+)", parsed.narrative)}
    unsupported = [m for m in mentioned if not any(abs(m - k) <= max(1, k * 0.01) for k in known_values)]

    if unsupported:
        return False, (
            f"Narrative references dollar figures not present in the model "
            f"output: {unsupported}. Known valid figures are: "
            f"{sorted(known_values)}."
        )
    return True, ""


class GroundingCheckerAgent(BaseAgent):
    """
    Non-LLM agent: deterministically validates the valuation explainer's
    last output and decides whether the enclosing LoopAgent should stop.
    """

    async def _run_async_impl(self, ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        raw = ctx.session.state.get("valuation_explanation_raw")
        if raw is None:
            # Nothing produced yet (shouldn't happen if this runs after the
            # explainer agent in the loop) -- fail safe by not escalating,
            # letting max_iterations be the backstop.
            yield Event(
                author=self.name,
                content=types.Content(role="model", parts=[types.Part(text="No valuation output found yet.")]),
                actions=EventActions(escalate=False),
            )
            return

        parsed_dict = json.loads(raw) if isinstance(raw, str) else raw
        parsed = ValuationExplanation.model_validate(parsed_dict)

        ok, feedback = _check_grounding(parsed)

        if ok:
            yield Event(
                author=self.name,
                content=types.Content(role="model", parts=[types.Part(text="Grounding check passed.")]),
                actions=EventActions(escalate=True),  # breaks the LoopAgent
            )
            return

        # Not grounded: stash feedback in state for the explainer's next
        # attempt (see _valuation_instruction reading 'grounding_feedback'),
        # and do NOT escalate so the LoopAgent runs the explainer again.
        yield Event(
            author=self.name,
            content=types.Content(role="model", parts=[types.Part(text=f"Grounding check failed: {feedback}")]),
            actions=EventActions(escalate=False, state_delta={"grounding_feedback": feedback}),
        )
