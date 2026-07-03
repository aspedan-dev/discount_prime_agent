"""
agent.py — Agent Analytics
---------------------------
LlmAgent that runs the Phase 1 pipeline as a strict, ordered sequence of
tool calls. It never computes numbers itself — it only orchestrates tools
and narrates their status/count results.

No output_schema here: ADK does not allow output_schema + tools on the
same LlmAgent, and this agent's job is to call tools.
"""

from __future__ import annotations

import os

from google.adk.agents import LlmAgent

from .tools import (
    classify_products_tool,
    compute_product_metrics_tool,
    evaluate_campaigns_tool,
    ingest_data_tool,
    recommend_products_tool,
)

MODEL = os.environ.get("DPA_ANALYTICS_MODEL", "gemini-2.5-flash")

INSTRUCTION = """\
You are a deterministic analytics runner for an e-commerce discount/campaign
analysis pipeline. You have exactly five tools. Call them ONCE EACH, in this
EXACT order, with no deviation:

  1. ingest_data_tool(data_path="{data_path}")
  2. compute_product_metrics_tool()
  3. classify_products_tool(min_units={min_units})
  4. evaluate_campaigns_tool()
  5. recommend_products_tool()

Rules:
- Do not skip a tool. Do not call a tool twice. Do not call them out of order.
- Do not invent, estimate, or restate any specific numeric value yourself —
  every number in your final answer must come directly from a tool's
  returned status dict.
- If any tool returns {{"status": "error", ...}}, stop and report the error
  message verbatim; do not proceed to later tools.
- After all five tools succeed, write a 3-5 sentence plain-English summary,
  for a merchant with no data background, describing only the COUNTS
  returned by the tools (e.g. how many orders/products were processed, how
  many products fell into each movement class, how many campaigns
  succeeded/flopped/were inconclusive, how many recommendations were
  produced). Do not mention dollar figures or percentages that were not
  explicitly returned by a tool.
"""

analytics_agent = LlmAgent(
    name="agent_analytics",
    model=MODEL,
    description=(
        "Runs the deterministic product-metrics / classification / "
        "campaign-evaluation / recommendation pipeline as tool calls."
    ),
    instruction=INSTRUCTION,
    tools=[
        ingest_data_tool,
        compute_product_metrics_tool,
        classify_products_tool,
        evaluate_campaigns_tool,
        recommend_products_tool,
    ],
    output_key="analytics_summary_text",
)
