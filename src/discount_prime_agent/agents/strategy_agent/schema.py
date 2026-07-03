"""
schema.py — Agent Strategy structured output contract.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DiscountMechanic = Literal[
    "Free Shipping",
    "BXGY",
    "Volume/Bundle",
    "Bundle/Cross-sell",
    "Leave/minimal discount",
    "Avoid deeper discounts",
    "Order-level Discount",
    "Generic Discount",
    "Collect more data",
]


class CampaignProposal(BaseModel):
    """One prioritized, segment-level campaign proposal."""

    product_ids: list[int] = Field(
        description="Product IDs belonging to this proposed campaign segment."
    )
    segment_label: str = Field(
        description='Short human label for the segment, e.g. "high-margin slow movers".'
    )
    discount_mechanic: DiscountMechanic
    rationale: str = Field(
        description="Why this mechanic fits this segment, grounded in the analytics evidence."
    )
    expected_impact: str = Field(
        description="Qualitative expected outcome — no invented numbers."
    )
    priority: int = Field(ge=1, le=5, description="1 = act now, 5 = lowest priority.")
    supporting_evidence_refs: list[str] = Field(default_factory=list)


class StrategyOutput(BaseModel):
    """Final structured output of Agent Strategy."""

    generated_at: str
    summary: str
    proposals: list[CampaignProposal]
