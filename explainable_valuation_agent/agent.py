"""
explainable_valuation_agent/agent.py

Turns a valuation model's raw output (predicted price + SHAP feature
attributions) into a grounded, audience-appropriate narrative explanation.

Pipeline:
    1. RAG: retrieve relevant comps / market-report context for this property.
    2. Format SHAP attributions into a compact, ranked text block.
    3. LLM call #1: draft the narrative, grounded in (SHAP block + RAG context),
       tailored to the requested audience (underwriter vs homeowner).
    4. Self-critique: run a deterministic grounding check (shap_utils) on the
       draft. If it references dollar figures not backed by the SHAP data,
       loop back with a correction instruction (agentic behavior: the agent
       inspects its own output and decides whether to redo the work, up to
       a small retry budget) instead of shipping an ungrounded explanation.

Concept coverage:
  - "LLM APIs (OpenAI)": narrative generation via shared.llm_client.chat
  - "AI agents / agentic frameworks": the retrieve -> draft -> critique ->
    (maybe) redraft loop is agent behavior, not a single prompt-response call
  - "RAG": explainable_valuation_agent/rag.py grounds market commentary
  - "MCP": exposed via mcp_server.py for use by other hosts/agents
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from explainable_valuation_agent.rag import retrieve_market_context
from explainable_valuation_agent.shap_utils import (
    ValuationExplanationInput,
    check_narrative_grounding,
    format_shap_for_prompt,
)
from shared.llm_client import chat

Audience = Literal["underwriter", "homeowner", "appraiser"]

AUDIENCE_GUIDANCE = {
    "underwriter": (
        "Write for a mortgage underwriter: precise, quantitative, flag any "
        "risk factors or unusual attributions, keep it under 180 words, use "
        "a neutral analytical tone."
    ),
    "homeowner": (
        "Write for a homeowner with no real estate background: plain "
        "language, no jargon, warm but factual tone, under 150 words, "
        "explain what actually drove the number in terms they'd recognize "
        "(location, size, condition, recent nearby sales)."
    ),
    "appraiser": (
        "Write for a professional appraiser reviewing this as a second "
        "opinion: technical tone, reference comparable sales explicitly, "
        "note any features the model may be over/under-weighting, under "
        "200 words."
    ),
}

SYSTEM_PROMPT = """\
You are an explainable-AI assistant for a real estate valuation model. You \
turn a model's SHAP feature attributions plus retrieved market context into \
an accurate, honest narrative explaining the predicted value.

Rules:
- Every dollar figure you state must come directly from the SHAP data you \
are given. Do not invent or round dramatically differently than given.
- You may reference the retrieved comps/market context for color and \
context, but do not state a comp's sale price as if it were a feature \
contribution to THIS property's valuation.
- If the SHAP data and retrieved context seem to conflict (e.g. market \
report says prices rising but model shows a large negative market-trend \
contribution), note the discrepancy briefly rather than papering over it.
- Match the tone and technical depth to the requested audience exactly.
"""


@dataclass
class ExplanationResult:
    narrative: str
    grounding_ok: bool
    retrieved_context: list[dict] = field(default_factory=list)
    attempts: int = 1


def explain_valuation(
    property_summary: str,
    valuation_input: ValuationExplanationInput,
    audience: Audience = "homeowner",
    max_attempts: int = 2,
) -> ExplanationResult:
    """
    Generate a grounded narrative explanation for a valuation model's output.

    Args:
        property_summary: short free-text description of the property, used
            as the RAG query (e.g. "123 Maple St, Springfield IL, 3bd/2ba").
        valuation_input: predicted price, base value, and SHAP contributions.
        audience: who the narrative is for -- changes tone/depth.
        max_attempts: how many draft/critique cycles to allow before
            returning the best available draft with grounding_ok=False.
    """
    retrieved = retrieve_market_context(property_summary, top_k=3)
    context_block = "\n".join(f"- [{c['id']}] {c['text']}" for c in retrieved)
    shap_block = format_shap_for_prompt(valuation_input)

    base_user_message = (
        f"Audience: {audience}\n{AUDIENCE_GUIDANCE[audience]}\n\n"
        f"Property: {property_summary}\n\n"
        f"Model output:\n{shap_block}\n\n"
        f"Retrieved market context:\n{context_block}\n\n"
        "Write the narrative explanation now."
    )

    messages = [{"role": "user", "content": base_user_message}]
    narrative = ""
    grounding_ok = False

    for attempt in range(1, max_attempts + 1):
        response = chat(SYSTEM_PROMPT, messages)
        narrative = response.content or ""

        check = check_narrative_grounding(narrative, valuation_input)
        grounding_ok = check["ok"]

        if grounding_ok:
            break

        # Self-critique loop: tell the model exactly what wasn't grounded
        # and ask it to redraft. This is the agentic part -- the pipeline
        # inspects its own output and decides to iterate rather than ship.
        messages.append({"role": "assistant", "content": narrative})
        messages.append(
            {
                "role": "user",
                "content": (
                    "Grounding check failed: the following dollar figures in "
                    f"your draft don't match the supplied SHAP data: "
                    f"{check['unsupported_dollar_figures']}. Rewrite the "
                    "narrative using ONLY the dollar figures given in the "
                    "model output above."
                ),
            }
        )

    return ExplanationResult(
        narrative=narrative,
        grounding_ok=grounding_ok,
        retrieved_context=retrieved,
        attempts=attempt,
    )


if __name__ == "__main__":
    from explainable_valuation_agent.shap_utils import FeatureContribution

    demo_input = ValuationExplanationInput(
        predicted_price=312000,
        base_value=265000,
        contributions=[
            FeatureContribution("living_area_sqft", "1,900 sqft", 18400),
            FeatureContribution("renovated_kitchen", "yes", 9200),
            FeatureContribution("neighborhood", "Maple/Oak corridor", 14100),
            FeatureContribution("days_on_market_trend", "18 days (Q2 2026)", 6800),
            FeatureContribution("lot_size", "0.18 acres", -1500),
        ],
    )

    result = explain_valuation(
        property_summary="123 Oak Ave, Springfield IL, 3bd/2ba, 1900 sqft, renovated kitchen",
        valuation_input=demo_input,
        audience="homeowner",
    )

    print(f"Grounding OK: {result.grounding_ok} (attempts: {result.attempts})")
    print("\n--- Narrative ---\n")
    print(result.narrative)
