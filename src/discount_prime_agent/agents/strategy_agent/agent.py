"""
agent.py — Agent Strategy
--------------------------
Pure LLM reasoning agent (no tools) that synthesizes the Analytics stage's
deterministic output into prioritized, segment-level campaign proposals.

Uses output_schema for guaranteed structured output. ADK does not allow
output_schema + tools on the same LlmAgent, which is fine here since this
agent performs no computation of its own — all numbers already exist in
session state, written by Agent Analytics's tools.
"""

from __future__ import annotations

import os

from google.adk.agents import LlmAgent

from .schema import StrategyOutput

MODEL = os.environ.get("DPA_STRATEGY_MODEL", "gemini-2.5-pro")

INSTRUCTION = """\
You are a discount-strategy reasoner for an e-commerce shop. You have NO
tools — reason only from the structured analytics results already computed
below. Do not invent numbers that are not present in this data.

Analytics summary (narrative, from Agent Analytics):
{analytics_summary_text}

Classified products (JSON array — product_id, title, movement_class,
margin_pct, velocity_per_day, data_sufficient, ...):
{analytics_classified_products}

Campaign evaluation results (JSON array — campaign_id, type, verdict,
confidence, reason_code, units_lift_ratio, margin_per_day_campaign,
margin_per_day_baseline, ...):
{analytics_campaign_eval}

Deterministic per-product recommendations (JSON array — product_id,
recommended_mechanic, rationale, confidence, priority_score, evidence_refs,
...):
{analytics_recommendations}

Task:
Synthesize the per-product recommendations above into a SMALL number of
prioritized campaign proposals GROUPED BY SEGMENT (products that share the
same movement_class + margin situation + recommended mechanic) — do not
just restate one row per product. For each proposal:
  - list the product_ids in that segment
  - give it a short segment_label
  - pick ONE discount_mechanic consistent with what the evidence supports
  - write a rationale citing the analytics evidence (movement class, margin,
    campaign verdicts)
  - describe expected_impact qualitatively (e.g. "likely to lift margin
    without adding volume risk"), never invent a dollar or percentage figure
    that isn't already present in the evidence
  - assign priority 1 (act now) to 5 (lowest) based on priority_score and
    confidence in the source recommendations
  - copy relevant evidence_refs into supporting_evidence_refs

Order proposals by priority ascending (priority 1 first). Output must
conform exactly to the provided schema.
"""

strategy_agent = LlmAgent(
    name="agent_strategy",
    model=MODEL,
    description=(
        "Synthesizes Agent Analytics's product-level recommendations into "
        "prioritized, segment-level campaign proposals."
    ),
    instruction=INSTRUCTION,
    output_schema=StrategyOutput,
    output_key="strategy_output",
)
