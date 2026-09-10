"""
real_estate_agents/explainable_valuation_agent/shap_utils.py

Utilities for turning raw SHAP (or any additive feature-attribution) output
from the valuation ML model into a structured, LLM-friendly format.

In production, `contributions` would come directly from
`shap.TreeExplainer(model).shap_values(X)` (for a tree-based valuation model)
or a KernelExplainer for other model types. This module doesn't care which --
it just expects a base_value + a list of (feature, contribution) pairs.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FeatureContribution:
    feature: str
    value: str  # human-readable raw feature value, e.g. "1,840 sqft"
    contribution: float  # dollar impact on predicted price, signed


@dataclass
class ValuationExplanationInput:
    predicted_price: float
    base_value: float  # the model's average/baseline prediction before feature effects
    contributions: list[FeatureContribution]


def format_shap_for_prompt(exp: ValuationExplanationInput, top_n: int = 6) -> str:
    """
    Render the SHAP breakdown as a compact, ranked text block the LLM can
    ground its narrative in. Ranked by absolute contribution so the most
    influential factors (positive or negative) come first.
    """
    ranked = sorted(exp.contributions, key=lambda c: abs(c.contribution), reverse=True)[:top_n]
    lines = [
        f"Base (average) predicted value: ${exp.base_value:,.0f}",
        f"Final predicted value: ${exp.predicted_price:,.0f}",
        "Top feature contributions (SHAP values, additive to base):",
    ]
    for c in ranked:
        sign = "+" if c.contribution >= 0 else "-"
        lines.append(f"  - {c.feature} ({c.value}): {sign}${abs(c.contribution):,.0f}")

    accounted = sum(c.contribution for c in ranked)
    residual = (exp.predicted_price - exp.base_value) - accounted
    if abs(residual) > 1:
        lines.append(f"  - All other features combined: {'+' if residual >= 0 else '-'}${abs(residual):,.0f}")

    return "\n".join(lines)


def check_narrative_grounding(narrative: str, exp: ValuationExplanationInput) -> dict:
    """
    Lightweight grounding check: verify that every dollar figure the LLM
    wrote in its narrative actually appears (within rounding) among the
    known SHAP contributions or the predicted/base price. Used by the agent
    as a self-critique step before finalizing output -- catches the LLM
    inventing numbers not supported by the model's actual attributions.

    This is a heuristic safety net, not a substitute for a real fact-checking
    pass; it flags narratives that reference dollar amounts absent from the
    supplied SHAP data so a human can review before the explanation ships.
    """
    import re

    known_values = {round(exp.predicted_price), round(exp.base_value)}
    for c in exp.contributions:
        known_values.add(round(abs(c.contribution)))

    mentioned = {int(m.replace(",", "")) for m in re.findall(r"\$([\d,]+)", narrative)}
    unsupported = [m for m in mentioned if not any(abs(m - k) <= max(1, k * 0.01) for k in known_values)]

    return {"ok": len(unsupported) == 0, "unsupported_dollar_figures": unsupported}
