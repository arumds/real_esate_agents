"""
eval/test_deterministic.py

Level 1 evals: pure-Python logic with no LLM call, no API key required.
Run with: pytest eval/test_deterministic.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_quality_agent.tools import check_field_completeness, compare_reported_vs_authoritative
from explainable_valuation_agent.shap_utils import (
    FeatureContribution,
    ValuationExplanationInput,
    check_narrative_grounding,
)


def test_missing_field_detected():
    record = {
        "parcel_id": "PARCEL-10234",
        "address": "123 Maple St",
        "sqft": 1840,
        "bedrooms": 3,
        "bathrooms": None,  # missing
        "year_built": 1998,
        "list_price": 289000,
    }
    result = check_field_completeness(record)
    assert "bathrooms" in result["missing_fields"]


def test_implausible_sqft_flagged():
    record = {
        "parcel_id": "PARCEL-10234",
        "address": "123 Maple St",
        "sqft": -50,  # implausible
        "bedrooms": 3,
        "bathrooms": 2,
        "year_built": 1998,
        "list_price": 289000,
    }
    result = check_field_completeness(record)
    assert "sqft" in result["malformed_fields"]


def test_clean_record_passes():
    record = {
        "parcel_id": "PARCEL-10234",
        "address": "123 Maple St",
        "sqft": 1840,
        "bedrooms": 3,
        "bathrooms": 2,
        "year_built": 1998,
        "list_price": 289000,
    }
    result = check_field_completeness(record)
    assert result["missing_fields"] == []
    assert result["malformed_fields"] == {}


def test_discrepancy_within_tolerance():
    result = compare_reported_vs_authoritative(reported_sqft=1850, authoritative_sqft=1840, tolerance_pct=5.0)
    assert result["within_tolerance"] is True


def test_discrepancy_outside_tolerance():
    result = compare_reported_vs_authoritative(reported_sqft=2450, authoritative_sqft=1840, tolerance_pct=5.0)
    assert result["within_tolerance"] is False
    assert result["discrepancy_pct"] > 5.0


def test_grounding_check_accepts_valid_narrative():
    exp = ValuationExplanationInput(
        predicted_price=312000,
        base_value=265000,
        contributions=[FeatureContribution("sqft", "1900 sqft", 18400)],
    )
    narrative = "Your home is valued at $312,000, driven largely by an $18,400 boost from square footage."
    result = check_narrative_grounding(narrative, exp)
    assert result["ok"] is True


def test_grounding_check_rejects_hallucinated_figure():
    exp = ValuationExplanationInput(
        predicted_price=312000,
        base_value=265000,
        contributions=[FeatureContribution("sqft", "1900 sqft", 18400)],
    )
    narrative = "Your home is valued at $312,000, including a $55,000 premium for the pool."
    result = check_narrative_grounding(narrative, exp)
    assert result["ok"] is False
    assert 55000 in result["unsupported_dollar_figures"]


# Note: there used to be a test here for a hand-rolled JSON-parsing fallback
# (data_quality_agent/agent.py::_safe_parse_json, fail-closed on malformed
# LLM output). That code -- and the failure mode it guarded against -- no
# longer exists: data_quality_agent/mcp_server.py now runs the ADK
# LlmAgent (adk_version/agent.py), whose output_schema enforces valid,
# schema-conformant JSON at the model API level, so there's no free-text
# LLM output left to parse defensively.


if __name__ == "__main__":
    import subprocess

    subprocess.run(["pytest", __file__, "-v"])
