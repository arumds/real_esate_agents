"""
real_estate_agents/data_quality_agent/tools.py

Deterministic validators + mocked "secondary source" lookups the agent can
call as tools. In production, `lookup_county_assessor_record` and
`geocode_and_validate_address` would call real APIs (county assessor open-data
portals, USPS/Smarty address validation, a geocoder). They're mocked here with
a small fixture dataset so the whole pipeline is runnable offline.

Every function here is intentionally a plain, well-typed, well-documented
Python function -- FastMCP (see mcp_server.py) turns these directly into MCP
tool schemas, and the same functions are used as OpenAI function-calling
tools in agent.py. Single source of truth for tool behavior.
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Mock "secondary source" datasets, keyed by parcel_id / normalized address.
# Stand-ins for: county assessor records, USPS/geocoder validation.
# ---------------------------------------------------------------------------

_COUNTY_ASSESSOR_RECORDS = {
    "PARCEL-10234": {"sqft": 1840, "year_built": 1998, "bedrooms": 3, "bathrooms": 2},
    "PARCEL-55871": {"sqft": 2600, "year_built": 2004, "bedrooms": 4, "bathrooms": 3},
    "PARCEL-99120": {"sqft": 1120, "year_built": 1975, "bedrooms": 2, "bathrooms": 1},
}

_VALID_ADDRESS_DB = {
    "123 maple st, springfield, il 62701": {
        "standardized": "123 Maple St, Springfield, IL 62701",
        "lat": 39.7817,
        "lon": -89.6501,
    },
    "88 lakeview dr, austin, tx 78701": {
        "standardized": "88 Lakeview Dr, Austin, TX 78701",
        "lat": 30.2672,
        "lon": -97.7431,
    },
    "500 river rd, portland, or 97201": {
        "standardized": "500 River Rd, Portland, OR 97201",
        "lat": 45.5152,
        "lon": -122.6784,
    },
}


def check_field_completeness(record: dict[str, Any]) -> dict[str, Any]:
    """
    Rule-based check for missing or obviously malformed required fields on a
    property record.

    Args:
        record: property record dict, expected keys include address,
            parcel_id, sqft, bedrooms, bathrooms, year_built, list_price.

    Returns:
        dict with 'missing_fields' (list[str]) and 'malformed_fields'
        (dict[str, str] mapping field -> reason).
    """
    required = ["address", "parcel_id", "sqft", "bedrooms", "bathrooms", "year_built", "list_price"]
    missing = [f for f in required if record.get(f) in (None, "", [])]

    malformed = {}
    sqft = record.get("sqft")
    if isinstance(sqft, (int, float)) and (sqft <= 0 or sqft > 50000):
        malformed["sqft"] = f"sqft={sqft} is out of plausible range (0, 50000]"

    year_built = record.get("year_built")
    if isinstance(year_built, (int, float)) and (year_built < 1700 or year_built > 2026):
        malformed["year_built"] = f"year_built={year_built} is implausible"

    beds = record.get("bedrooms")
    if isinstance(beds, (int, float)) and (beds < 0 or beds > 30):
        malformed["bedrooms"] = f"bedrooms={beds} is implausible"

    return {"missing_fields": missing, "malformed_fields": malformed}


def geocode_and_validate_address(address: str) -> dict[str, Any]:
    """
    Validate and standardize a free-text property address against a
    secondary address-verification source (mocked stand-in for
    USPS/Smarty/Google Geocoding).

    Args:
        address: raw address string as entered in the source system.

    Returns:
        dict with 'is_valid' (bool), and if valid, 'standardized_address',
        'lat', 'lon'. If invalid, 'reason'.
    """
    key = re.sub(r"\s+", " ", address.strip().lower())
    match = _VALID_ADDRESS_DB.get(key)
    if match:
        return {"is_valid": True, **match}
    return {
        "is_valid": False,
        "reason": "Address not found in verification source; possible typo, "
        "unit-number mismatch, or unrecognized street.",
    }


def lookup_county_assessor_record(parcel_id: str) -> dict[str, Any]:
    """
    Look up authoritative property characteristics (sqft, year built,
    bed/bath count) from the county assessor's public record for a given
    parcel ID. Mocked stand-in for a real county open-data API call.

    Args:
        parcel_id: the assessor's parcel identifier for the property.

    Returns:
        dict of assessor-of-record fields, or {'found': False} if no match.
    """
    record = _COUNTY_ASSESSOR_RECORDS.get(parcel_id)
    if record is None:
        return {"found": False}
    return {"found": True, **record}


def compare_reported_vs_authoritative(
    reported_sqft: float,
    authoritative_sqft: float,
    tolerance_pct: float = 5.0,
) -> dict[str, Any]:
    """
    Compare a listing-reported value against an authoritative secondary
    source value (e.g. county assessor sqft) and flag if the discrepancy
    exceeds tolerance.

    Args:
        reported_sqft: value as entered by the listing agent / source feed.
        authoritative_sqft: value from the secondary source of record.
        tolerance_pct: allowed percentage discrepancy before flagging.

    Returns:
        dict with 'discrepancy_pct', 'within_tolerance' (bool).
    """
    if authoritative_sqft == 0:
        return {"discrepancy_pct": None, "within_tolerance": False}
    discrepancy_pct = abs(reported_sqft - authoritative_sqft) / authoritative_sqft * 100
    return {
        "discrepancy_pct": round(discrepancy_pct, 2),
        "within_tolerance": discrepancy_pct <= tolerance_pct,
    }


# Tool registry + OpenAI-format tool specs, shared by agent.py and mcp_server.py
TOOL_REGISTRY = {
    "check_field_completeness": check_field_completeness,
    "geocode_and_validate_address": geocode_and_validate_address,
    "lookup_county_assessor_record": lookup_county_assessor_record,
    "compare_reported_vs_authoritative": compare_reported_vs_authoritative,
}

TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "check_field_completeness",
            "description": "Check a property record for missing or malformed required fields.",
            "parameters": {
                "type": "object",
                "properties": {"record": {"type": "object", "description": "The raw property record"}},
                "required": ["record"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "geocode_and_validate_address",
            "description": "Validate and standardize an address against a secondary verification source.",
            "parameters": {
                "type": "object",
                "properties": {"address": {"type": "string"}},
                "required": ["address"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_county_assessor_record",
            "description": "Fetch authoritative property characteristics from the county assessor record.",
            "parameters": {
                "type": "object",
                "properties": {"parcel_id": {"type": "string"}},
                "required": ["parcel_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_reported_vs_authoritative",
            "description": "Compare a reported numeric field against an authoritative secondary-source value.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reported_sqft": {"type": "number"},
                    "authoritative_sqft": {"type": "number"},
                    "tolerance_pct": {"type": "number"},
                },
                "required": ["reported_sqft", "authoritative_sqft"],
            },
        },
    },
]
