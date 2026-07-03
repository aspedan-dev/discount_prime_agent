"""
tests/test_strategy_schema.py
-------------------------------
Validation tests for Agent Strategy's Pydantic output contract.
No LLM / no ADK Runner involved.
"""

import pytest
from pydantic import ValidationError

from discount_prime_agent.agents.strategy_agent.schema import CampaignProposal, StrategyOutput


def test_valid_strategy_output_parses():
    payload = {
        "generated_at": "2026-07-03T00:00:00Z",
        "summary": "Two segments identified.",
        "proposals": [
            {
                "product_ids": [111, 222],
                "segment_label": "high-margin slow movers",
                "discount_mechanic": "BXGY",
                "rationale": "High margin, low velocity; use BXGY to move stock without eroding margin.",
                "expected_impact": "Likely to increase velocity without materially reducing margin.",
                "priority": 1,
                "supporting_evidence_refs": ["product_id:111", "movement_class:slow"],
            }
        ],
    }
    out = StrategyOutput.model_validate(payload)
    assert len(out.proposals) == 1
    assert out.proposals[0].discount_mechanic == "BXGY"


def test_invalid_discount_mechanic_rejected():
    with pytest.raises(ValidationError):
        CampaignProposal.model_validate(
            {
                "product_ids": [1],
                "segment_label": "test",
                "discount_mechanic": "Not A Real Mechanic",
                "rationale": "x",
                "expected_impact": "x",
                "priority": 1,
            }
        )


def test_priority_out_of_range_rejected():
    with pytest.raises(ValidationError):
        CampaignProposal.model_validate(
            {
                "product_ids": [1],
                "segment_label": "test",
                "discount_mechanic": "Free Shipping",
                "rationale": "x",
                "expected_impact": "x",
                "priority": 6,
            }
        )
