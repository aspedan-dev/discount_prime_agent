"""
tools.py
--------
ADK FunctionTools wrapping the Phase 1 deterministic pipeline
(discount_prime_agent.pipeline).  All pandas/pydantic math stays here —
the LLM never sees a raw table, only small status/count dicts, so it
cannot hallucinate numbers.

Data flows between tool calls through `tool_context.state` (JSON-safe
records), never through the LLM's own generated text.  State keys are
flat/top-level so they can be referenced directly in agent instructions
via `{state_key}` templating:

    analytics_orders, analytics_lineitems, analytics_campaigns,
    analytics_shop, analytics_products, analytics_classified_products,
    analytics_campaign_eval, analytics_recommendations
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from google.adk.tools.tool_context import ToolContext

from discount_prime_agent.pipeline import (
    build_clean_frames,
    classify_products,
    evaluate_campaigns,
    recommend_for_products,
    run_product_metrics,
)


# ---------------------------------------------------------------------------
# JSON-safety helper
# ---------------------------------------------------------------------------

def _json_safe(value: Any) -> Any:
    """Recursively convert pandas/numpy values to plain JSON-serializable types."""
    if isinstance(value, (pd.Timestamp,)):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, float) and math.isnan(value):
        return None
    if hasattr(value, "item"):  # numpy scalar (int64, float64, bool_, ...)
        return _json_safe(value.item())
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if value is pd.NaT:
        return None
    return value


def json_safe_records(df: pd.DataFrame) -> list[dict]:
    """Convert a DataFrame to a list of JSON-safe record dicts."""
    return [_json_safe(row) for row in df.to_dict(orient="records")]


# ---------------------------------------------------------------------------
# Tool 1 — ingest
# ---------------------------------------------------------------------------

def ingest_data_tool(data_path: str, tool_context: ToolContext) -> dict:
    """Load and clean the shop's order/line-item/campaign data, stripping PII."""
    orders_df, lineitems_df, campaigns_df, shop = build_clean_frames(data_path)

    tool_context.state["analytics_orders"] = json_safe_records(orders_df)
    tool_context.state["analytics_lineitems"] = json_safe_records(lineitems_df)
    tool_context.state["analytics_campaigns"] = json_safe_records(campaigns_df)
    tool_context.state["analytics_shop"] = _json_safe(shop)

    return {
        "status": "ok",
        "orders": len(orders_df),
        "lineitems": len(lineitems_df),
        "campaigns": len(campaigns_df),
    }


# ---------------------------------------------------------------------------
# Tool 2 — product metrics
# ---------------------------------------------------------------------------

def compute_product_metrics_tool(tool_context: ToolContext) -> dict:
    """Compute per-product revenue/margin/velocity metrics from ingested line items."""
    lineitems_records = tool_context.state.get("analytics_lineitems")
    if not lineitems_records:
        return {"status": "error", "message": "Call ingest_data_tool first."}

    lineitems_df = pd.DataFrame(lineitems_records)
    lineitems_df["createdAt"] = pd.to_datetime(lineitems_df["createdAt"], utc=True)

    products_df, _profiles = run_product_metrics(lineitems_df)
    tool_context.state["analytics_products"] = json_safe_records(products_df)

    return {"status": "ok", "products": len(products_df)}


# ---------------------------------------------------------------------------
# Tool 3 — classify
# ---------------------------------------------------------------------------

def classify_products_tool(min_units: int, tool_context: ToolContext) -> dict:
    """Classify products into fast/medium/slow movement tiers."""
    products_records = tool_context.state.get("analytics_products")
    if not products_records:
        return {"status": "error", "message": "Call compute_product_metrics_tool first."}

    products_df = pd.DataFrame(products_records)
    classified_df = classify_products(products_df, min_units=min_units)
    tool_context.state["analytics_classified_products"] = json_safe_records(classified_df)

    counts = classified_df["movement_class"].value_counts().to_dict()
    return {"status": "ok", "class_counts": {k: int(v) for k, v in counts.items()}}


# ---------------------------------------------------------------------------
# Tool 4 — campaign evaluation
# ---------------------------------------------------------------------------

def evaluate_campaigns_tool(tool_context: ToolContext) -> dict:
    """Evaluate every campaign's profit impact vs. its organic baseline."""
    orders_records = tool_context.state.get("analytics_orders")
    lineitems_records = tool_context.state.get("analytics_lineitems")
    campaigns_records = tool_context.state.get("analytics_campaigns")
    if not orders_records or not lineitems_records or not campaigns_records:
        return {"status": "error", "message": "Call ingest_data_tool first."}

    orders_df = pd.DataFrame(orders_records)
    lineitems_df = pd.DataFrame(lineitems_records)
    campaigns_df = pd.DataFrame(campaigns_records)

    lineitems_df["createdAt"] = pd.to_datetime(lineitems_df["createdAt"], utc=True)
    campaigns_df["startAt"] = pd.to_datetime(campaigns_df["startAt"], utc=True)
    campaigns_df["ranAt"] = pd.to_datetime(campaigns_df["ranAt"], utc=True)
    campaigns_df["endAt"] = pd.to_datetime(campaigns_df["endAt"], utc=True)

    campaign_eval_df, _verdicts = evaluate_campaigns(orders_df, lineitems_df, campaigns_df)
    tool_context.state["analytics_campaign_eval"] = json_safe_records(campaign_eval_df)

    verdict_counts = campaign_eval_df["verdict"].value_counts().to_dict()
    return {"status": "ok", "verdict_counts": {k: int(v) for k, v in verdict_counts.items()}}


# ---------------------------------------------------------------------------
# Tool 5 — recommendations
# ---------------------------------------------------------------------------

def recommend_products_tool(tool_context: ToolContext) -> dict:
    """Generate one deterministic recommendation per product from classification + campaign evidence."""
    classified_records = tool_context.state.get("analytics_classified_products")
    campaign_eval_records = tool_context.state.get("analytics_campaign_eval")
    if not classified_records or not campaign_eval_records:
        return {"status": "error", "message": "Call classify_products_tool and evaluate_campaigns_tool first."}

    classified_df = pd.DataFrame(classified_records)
    campaign_eval_df = pd.DataFrame(campaign_eval_records)

    recs_df, _recs = recommend_for_products(classified_df, campaign_eval_df, affinity_df=None)
    tool_context.state["analytics_recommendations"] = json_safe_records(recs_df)

    return {"status": "ok", "recommendations": len(recs_df)}
