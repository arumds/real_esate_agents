"""
eval/golden_dataset.py

A small labeled dataset for evaluating the Data Quality Agent's disposition
accuracy. In a real deployment this would be built from a sample of actual
historical records where a human reviewer's decision is known (the ground
truth), ideally stratified to include real edge cases your pipeline has
hit -- not just synthetic ones like these.

Each case includes `expected_disposition` and, where relevant,
`expected_correction` so precision on auto_correct can be scored too.
"""

from __future__ import annotations

from typing import Any, Optional, TypedDict


class GoldenCase(TypedDict):
    id: str
    record: dict[str, Any]
    expected_disposition: str  # pass | auto_correct | flag_for_review
    expected_correction: Optional[dict[str, Any]]  # only checked if disposition == auto_correct
    notes: str


GOLDEN_DATASET: list[GoldenCase] = [
    {
        "id": "missing-parcel-id",
        "record": {
            "parcel_id": None,
            "address": "1 Nowhere Ln, Nowhere, TX 00000",
            "sqft": 1500,
            "bedrooms": 2,
            "bathrooms": 1,
            "year_built": 1990,
            "list_price": 150000,
        },
        "expected_disposition": "flag_for_review",
        "expected_correction": None,
        "notes": (
            "parcel_id is a required field and is missing here, which also "
            "means sqft can't be cross-checked against any authoritative "
            "source -> flag for review rather than pass unverified. This is "
            "a deliberate design choice: a record with no way to verify its "
            "own numbers shouldn't get a free pass."
        ),
    },
    {
        "id": "clean-with-parcel",
        "record": {
            "parcel_id": "PARCEL-55871",
            "address": "88 Lakeview Dr, Austin, TX 78701",
            "sqft": 2600,  # matches mocked assessor record exactly
            "bedrooms": 4,
            "bathrooms": 3,
            "year_built": 2004,
            "list_price": 645000,
        },
        "expected_disposition": "pass",
        "expected_correction": None,
        "notes": "Has a parcel_id, sqft matches assessor record exactly, address validates -> clean pass after cross-check.",
    },
    {
        "id": "matches-assessor-exactly",
        "record": {
            "parcel_id": "PARCEL-10234",
            "address": "123 Maple St, Springfield, IL 62701",
            "sqft": 1840,  # exactly matches mocked assessor record
            "bedrooms": 3,
            "bathrooms": 2,
            "year_built": 1998,
            "list_price": 289000,
        },
        "expected_disposition": "pass",
        "expected_correction": None,
        "notes": "sqft matches county assessor record exactly -> should pass after cross-check.",
    },
    {
        "id": "large-sqft-discrepancy",
        "record": {
            "parcel_id": "PARCEL-10234",
            "address": "123 Maple St, Springfield, IL 62701",
            "sqft": 2450,  # assessor says 1840 -> 33% off
            "bedrooms": 3,
            "bathrooms": 2,
            "year_built": 1998,
            "list_price": 289000,
        },
        "expected_disposition": "flag_for_review",
        "expected_correction": None,
        "notes": "33% discrepancy is too large to confidently auto-correct -> should flag, not silently overwrite.",
    },
    {
        "id": "missing-required-field",
        "record": {
            "parcel_id": "PARCEL-55871",
            "address": "500 River Rd, Portland, OR 97201",
            "sqft": 2600,
            "bedrooms": None,  # missing
            "bathrooms": 3,
            "year_built": 2004,
            "list_price": 410000,
        },
        "expected_disposition": "flag_for_review",
        "expected_correction": None,
        "notes": "Missing bedrooms count with no way to infer it from available tools -> flag.",
    },
    {
        "id": "unrecognized-address",
        "record": {
            "parcel_id": "PARCEL-99120",
            "address": "999 Fictional Blvd, Atlantis, XX 00001",
            "sqft": 1120,
            "bedrooms": 2,
            "bathrooms": 1,
            "year_built": 1975,
            "list_price": 95000,
        },
        "expected_disposition": "flag_for_review",
        "expected_correction": None,
        "notes": "Address won't validate against the mocked address DB -> flag, don't guess a standardized address.",
    },
]
