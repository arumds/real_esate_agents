"""
real_estate_agents/schemas.py

Same shape as crewai_version/schemas.py (deliberately duplicated, not
imported cross-framework) so each framework adapter directory stands alone
and can be read/deleted independently without touching the other.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class PropertyRecord(BaseModel):
    """
    Fixed field set standing in for the property record. OpenAI's strict
    structured-output mode (used by ADK's output_schema via LiteLLM) can't
    represent an open-ended dict -- every object needs a declared property
    list with additionalProperties: false -- so this replaces the bare
    `dict` that corrected_fields/final_record used to be. All fields are
    optional since corrected_fields only carries the ones that changed.
    """

    parcel_id: Optional[str] = None
    address: Optional[str] = None
    sqft: Optional[float] = None
    bedrooms: Optional[int] = None
    bathrooms: Optional[float] = None
    year_built: Optional[int] = None
    list_price: Optional[float] = None


class DataQualityDecision(BaseModel):
    disposition: str = Field(description="One of: pass, auto_correct, flag_for_review")
    corrected_fields: PropertyRecord = Field(
        default_factory=PropertyRecord, description="Fields changed, all null if none"
    )
    reasoning: str = Field(description="Brief explanation of the decision")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in this disposition")
    final_record: PropertyRecord = Field(description="The record to pass downstream (corrected if applicable)")


class FeatureContributionOut(BaseModel):
    feature: str
    value: str
    contribution: float


class ValuationExplanation(BaseModel):
    halted: bool = Field(description="True if the pipeline stopped before valuation (e.g. data flagged for review)")
    predicted_price: float = Field(default=0.0)
    base_value: float = Field(default=0.0)
    contributions: list[FeatureContributionOut] = Field(default_factory=list)
    narrative: str = Field(description="The audience-tailored explanation, or the halt reason if halted=True")
