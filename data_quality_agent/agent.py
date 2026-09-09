"""
data_quality_agent/agent.py

Agentic pipeline that sits in front of the valuation/rental ML models and
decides, for each incoming property record, whether it is:
  - PASSED            -> clean, send straight to the model
  - AUTO_CORRECTED    -> a confidently-fixable issue was found and corrected,
                          with a full audit trail
  - FLAGGED_FOR_REVIEW -> ambiguous / low-confidence issue, needs a human

Design notes:
  - Deterministic rule checks (tools.py) run first and are NOT delegated to
    the LLM -- cheap, fast, reproducible checks shouldn't cost a model call.
  - The LLM agent loop (shared.llm_client.run_agent_loop) is only invoked
    for the *judgment* part: given the deterministic findings, should the
    agent pull a secondary source, and if so, does the discrepancy justify
    an auto-correction or a human flag? That's the part that benefits from
    reasoning over ambiguous evidence rather than fixed if/else rules.
  - This mirrors how you'd want to deploy this in a regulated pipeline: keep
    the audit trail explicit, keep deterministic logic deterministic, and
    scope the LLM's authority narrowly (it can recommend/correct, not
    silently overwrite silently).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from data_quality_agent.tools import TOOL_REGISTRY, TOOL_SPECS, check_field_completeness
from shared.llm_client import run_agent_loop

SYSTEM_PROMPT = """\
You are a data quality agent for a real estate valuation pipeline. You are \
given a property record along with deterministic rule-check findings \
(missing/malformed fields). Your job:

1. For any suspicious numeric field (especially sqft), call \
`geocode_and_validate_address` to confirm the address is real, and call \
`lookup_county_assessor_record` (using the record's parcel_id) plus \
`compare_reported_vs_authoritative` to check the reported value against the \
authoritative source.
2. Decide a final disposition for the record:
   - "auto_correct": ONLY if an authoritative secondary source gives a clear, \
confident replacement value (discrepancy check ran and a trustworthy \
authoritative value exists).
   - "flag_for_review": if data is ambiguous, secondary sources disagree, no \
authoritative source was found, or the discrepancy is borderline.
   - "pass": if no material issues remain.
3. Always explain your reasoning briefly, and list exactly which fields (if \
any) you propose correcting and to what values.

Respond with your FINAL answer as a JSON object with keys: \
"disposition" (one of pass/auto_correct/flag_for_review), \
"corrected_fields" (object, empty if none), "reasoning" (string), \
"confidence" (0.0-1.0). Only output the JSON object as your final answer, \
nothing else.
"""


class Disposition(str, Enum):
    PASS = "pass"
    AUTO_CORRECT = "auto_correct"
    FLAG_FOR_REVIEW = "flag_for_review"


@dataclass
class DataQualityResult:
    disposition: Disposition
    corrected_fields: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    confidence: float = 0.0
    rule_findings: dict[str, Any] = field(default_factory=dict)
    transcript: list[dict] = field(default_factory=list)
    final_record: dict[str, Any] = field(default_factory=dict)


def run_data_quality_check(record: dict[str, Any]) -> DataQualityResult:
    """
    Main entry point: validate a raw property record before it reaches the
    valuation/rental ML model.
    """
    rule_findings = check_field_completeness(record)
    has_parcel_id = bool(record.get("parcel_id"))

    # Fast path: nothing wrong at the rule level AND no parcel_id to cross-check
    # against -> skip the LLM call entirely (cost/latency optimization for
    # records with no available secondary source). If a parcel_id IS present,
    # we always route through the agent loop so it can cross-validate
    # self-consistent-but-possibly-wrong values (e.g. a plausible but
    # incorrect sqft) against the authoritative assessor record.
    if not rule_findings["missing_fields"] and not rule_findings["malformed_fields"] and not has_parcel_id:
        return DataQualityResult(
            disposition=Disposition.PASS,
            reasoning="All required fields present and within plausible ranges; no secondary source available to cross-check.",
            confidence=1.0,
            rule_findings=rule_findings,
            final_record=record,
        )

    user_message = (
        "Property record:\n"
        f"{json.dumps(record, indent=2)}\n\n"
        "Deterministic rule-check findings:\n"
        f"{json.dumps(rule_findings, indent=2)}\n\n"
        "Investigate and produce your final disposition JSON."
    )

    final_answer, transcript = run_agent_loop(
        system_prompt=SYSTEM_PROMPT,
        user_message=user_message,
        tool_registry=TOOL_REGISTRY,
        tool_specs=TOOL_SPECS,
    )

    parsed = _safe_parse_json(final_answer)
    disposition = Disposition(parsed.get("disposition", "flag_for_review"))
    corrected_fields = parsed.get("corrected_fields", {})

    final_record = {**record, **corrected_fields} if disposition == Disposition.AUTO_CORRECT else record

    return DataQualityResult(
        disposition=disposition,
        corrected_fields=corrected_fields,
        reasoning=parsed.get("reasoning", final_answer),
        confidence=float(parsed.get("confidence", 0.5)),
        rule_findings=rule_findings,
        transcript=transcript,
        final_record=final_record,
    )


def _safe_parse_json(text: str) -> dict[str, Any]:
    """LLMs occasionally wrap JSON in prose/markdown fences; strip and parse defensively."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.split("json", 1)[-1] if cleaned.lower().startswith("json") else cleaned
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Fall back: try to find the first {...} block
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                pass
        return {"disposition": "flag_for_review", "reasoning": f"Could not parse LLM output: {text}"}


if __name__ == "__main__":
    # Demo record with a suspicious sqft value and a valid parcel_id we mocked
    demo_record = {
        "parcel_id": "PARCEL-10234",
        "address": "123 Maple St, Springfield, IL 62701",
        "sqft": 2450,  # true assessor value is 1840 -> should be flagged/corrected
        "bedrooms": 3,
        "bathrooms": 2,
        "year_built": 1998,
        "list_price": 289000,
    }

    result = run_data_quality_check(demo_record)
    print(f"Disposition: {result.disposition.value}")
    print(f"Confidence:  {result.confidence}")
    print(f"Reasoning:   {result.reasoning}")
    print(f"Corrected:   {result.corrected_fields}")
