"""
pipeline
--------
Phase 1: deterministic pandas/pydantic analytics core. No AI, no ADK.

Re-exports the full public API so callers have one stable import surface:

    from discount_prime_agent.pipeline import build_clean_frames, run_product_metrics, ...
"""

from __future__ import annotations

from .ingest import (
    PII_COLUMN_FRAGMENTS,
    assert_no_pii,
    build_clean_frames,
    load_raw,
    minimize,
    to_campaigns_df,
    to_lineitems_df,
    to_orders_df,
)
from .metrics import (
    allocate_order_cost_to_lines,
    compute_line_revenue,
    compute_product_metrics,
    product_profiles_from_df,
    run_product_metrics,
)
from .classify import (
    MIN_UNITS,
    classify_products,
    product_profiles_from_classified_df,
)
from .campaign_eval import (
    baseline_lineitems_for_campaign,
    campaign_attributed_lineitems,
    evaluate_campaigns,
    prepare_line_margin_df,
)
from .rules import (
    recommend_for_products,
    summarize_campaign_evidence,
)

__all__ = [
    "PII_COLUMN_FRAGMENTS",
    "assert_no_pii",
    "build_clean_frames",
    "load_raw",
    "minimize",
    "to_campaigns_df",
    "to_lineitems_df",
    "to_orders_df",
    "allocate_order_cost_to_lines",
    "compute_line_revenue",
    "compute_product_metrics",
    "product_profiles_from_df",
    "run_product_metrics",
    "MIN_UNITS",
    "classify_products",
    "product_profiles_from_classified_df",
    "baseline_lineitems_for_campaign",
    "campaign_attributed_lineitems",
    "evaluate_campaigns",
    "prepare_line_margin_df",
    "recommend_for_products",
    "summarize_campaign_evidence",
]
