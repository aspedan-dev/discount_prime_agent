"""
metrics.py
----------
Pure pandas, deterministic product metrics computed from the clean
DataFrames produced by ingest.py.  No raw JSON is read here.

Public API
----------
    compute_line_revenue(lineitems_df)            -> pd.DataFrame
    allocate_order_cost_to_lines(lineitems_df)    -> pd.DataFrame
    compute_product_metrics(lineitems_df)         -> pd.DataFrame
    product_profiles_from_df(products_df)         -> list[ProductProfile]

Grain rules (enforced throughout)
----------------------------------
The following columns on lineitems_df are ORDER-GRAIN values duplicated
across every line item of the same order:

    cost_total, revenue_with_cost, order_total_price, order_total_discounts,
    shipping_original_price, shipping_price_charged, shipping_discount_amount,
    is_free_shipping

They are NEVER summed directly across line-item rows here.

cost_total is accessed ONLY through revenue-share allocation:
    allocated_cost_i = cost_total * line_net_revenue_i
                       / sum(line_net_revenue for the same order_id)

Margin caveat
-------------
Per meta.note in the source file, per-line price/compare_at_price/
product_id/variant_id are synthesized deterministically — they are
structural, not ground-truth pricing data.  Product-level margins are
therefore approximate.  Order- and campaign-level numbers are the reliable
grain.

velocity_per_day denominator
-----------------------------
All products share the same denominator:
    window_days = (dataset_last_order_date - dataset_first_order_date).days + 1
                  (minimum 1)

This ensures velocity_per_day values are directly comparable across
products when classify.py assigns tertile-based movement classes.
"""

from __future__ import annotations

import os
import warnings
from typing import Any

import pandas as pd

from discount_prime_agent.schemas import ProductProfile


# ---------------------------------------------------------------------------
# Step 1 — compute line-level revenue columns
# ---------------------------------------------------------------------------

def compute_line_revenue(lineitems_df: pd.DataFrame) -> pd.DataFrame:
    """
    Append two computed revenue columns to a copy of lineitems_df.

    New columns
    -----------
    line_gross_revenue : float
        quantity * price  (before any discount)
    line_net_revenue   : float
        line_gross_revenue - total_discount  (what the customer actually paid
        for this line item)

    No order-grain columns are summed here.
    """
    df = lineitems_df.copy()

    df["line_gross_revenue"] = df["quantity"] * df["price"]
    df["line_net_revenue"] = df["line_gross_revenue"] - df["total_discount"]

    # Sanity: net revenue should never be negative after discount.
    # The data audit confirmed total_discount <= quantity*price everywhere,
    # but we clip defensively rather than raising, and emit a warning.
    neg_mask = df["line_net_revenue"] < 0
    if neg_mask.any():
        n = neg_mask.sum()
        warnings.warn(
            f"[metrics] {n} line item(s) have line_net_revenue < 0 "
            f"(total_discount exceeds line_gross_revenue).  "
            f"Clipping to 0.0 to avoid negative revenue allocation.",
            stacklevel=2,
        )
        df.loc[neg_mask, "line_net_revenue"] = 0.0

    return df


# ---------------------------------------------------------------------------
# Step 2 — allocate order cost to line items by net-revenue share
# ---------------------------------------------------------------------------

def allocate_order_cost_to_lines(lineitems_df: pd.DataFrame) -> pd.DataFrame:
    """
    Append ``allocated_cost`` to a copy of lineitems_df that already contains
    ``line_net_revenue`` (i.e., output of compute_line_revenue).

    Allocation formula
    ------------------
        order_net_revenue = sum(line_net_revenue) for the same order_id
        allocated_cost_i  = cost_total_i * line_net_revenue_i
                            / order_net_revenue

    Edge cases
    ----------
    - If order_net_revenue == 0 for an order, allocated_cost is set to 0.0
      for all line items in that order (avoids division by zero / inf / NaN).
    - cost_total is read from the row value; it is ORDER-GRAIN (same for every
      line item of the same order) but we do NOT sum it — we read it once per
      row and multiply by the revenue share, which produces the correct result.

    Returns
    -------
    pd.DataFrame  with an additional ``allocated_cost`` column (float).
    """
    if "line_net_revenue" not in lineitems_df.columns:
        raise ValueError(
            "lineitems_df must contain 'line_net_revenue'. "
            "Call compute_line_revenue() first."
        )

    df = lineitems_df.copy()

    # Sum of line_net_revenue per order — this IS a line-grain sum, which is
    # correct because line_net_revenue is a per-line value (not an order-grain
    # duplicate).
    order_net_rev = (
        df.groupby("order_id")["line_net_revenue"]
        .sum()
        .rename("order_net_revenue")
    )
    df = df.join(order_net_rev, on="order_id")

    # Revenue share — safe division
    share = df["line_net_revenue"] / df["order_net_revenue"]
    # Where order_net_revenue == 0, share would be NaN/inf → set to 0
    share = share.where(df["order_net_revenue"] > 0, other=0.0)

    # cost_total is ORDER-GRAIN (same value on every row of the same order),
    # so multiplying it by the revenue share for that row is correct — we
    # are NOT summing it across rows.
    df["allocated_cost"] = (df["cost_total"] * share).round(6)

    # Drop the intermediate column
    df = df.drop(columns=["order_net_revenue"])

    zero_net_orders = (
        df.groupby("order_id")["line_net_revenue"].sum() == 0
    ).sum()
    if zero_net_orders > 0:
        warnings.warn(
            f"[metrics] {zero_net_orders} order(s) have zero total net revenue. "
            f"allocated_cost set to 0.0 for all their line items.",
            stacklevel=2,
        )

    return df


# ---------------------------------------------------------------------------
# Step 3 — compute per-product metrics
# ---------------------------------------------------------------------------

def compute_product_metrics(lineitems_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate line-level data to a per-product summary DataFrame.

    Prerequisites
    -------------
    lineitems_df must already contain:
      - line_gross_revenue  (added by compute_line_revenue)
      - line_net_revenue    (added by compute_line_revenue)
      - allocated_cost      (added by allocate_order_cost_to_lines)

    Velocity denominator
    --------------------
    window_days is computed once from the full dataset's createdAt range
    (inclusive: last - first + 1 day, minimum 1).  Every product uses the
    same denominator so velocity_per_day values are directly comparable for
    classify.py tertile assignment.

    Returns
    -------
    pd.DataFrame sorted by velocity_per_day descending, with columns:
        product_id, title,
        units_sold, revenue, allocated_cost, gross_margin_alloc, margin_pct,
        velocity_per_day,
        product_first_order_date, product_last_order_date,
        window_days  (shared dataset-level denominator, same on every row)
    """
    for col in ("line_net_revenue", "allocated_cost"):
        if col not in lineitems_df.columns:
            raise ValueError(
                f"lineitems_df must contain '{col}'. "
                "Call compute_line_revenue() then allocate_order_cost_to_lines() first."
            )

    df = lineitems_df.copy()

    # ── dataset-level velocity window ────────────────────────────────────────
    dataset_first = df["createdAt"].min()
    dataset_last = df["createdAt"].max()
    # Convert to date for a clean day count (strip intraday time)
    first_date = pd.Timestamp(dataset_first).date()
    last_date = pd.Timestamp(dataset_last).date()
    window_days = max((last_date - first_date).days + 1, 1)

    # ── per-product aggregation ──────────────────────────────────────────────
    # Group by (product_id, title) — both are structural identifiers in the data
    products = (
        df.groupby(["product_id", "title"], sort=False)
        .agg(
            units_sold=("quantity", "sum"),
            revenue=("line_net_revenue", "sum"),
            allocated_cost=("allocated_cost", "sum"),
            product_first_order_date=("createdAt", "min"),
            product_last_order_date=("createdAt", "max"),
        )
        .reset_index()
    )

    # ── derived columns ──────────────────────────────────────────────────────
    products["gross_margin_alloc"] = products["revenue"] - products["allocated_cost"]

    # Safe margin_pct: 0.0 when revenue == 0 (avoids inf / NaN)
    products["margin_pct"] = (
        products["gross_margin_alloc"]
        / products["revenue"].replace(0, float("nan"))
    ).fillna(0.0)

    # Round money columns to 4 decimal places
    for col in ("revenue", "allocated_cost", "gross_margin_alloc", "margin_pct"):
        products[col] = products[col].round(4)

    # velocity_per_day uses the shared dataset window
    products["velocity_per_day"] = (
        products["units_sold"] / window_days
    ).round(6)

    # Carry the shared denominator as a context column
    products["window_days"] = window_days

    # ── sort ─────────────────────────────────────────────────────────────────
    products = products.sort_values("velocity_per_day", ascending=False).reset_index(
        drop=True
    )

    return products


# ---------------------------------------------------------------------------
# Step 4 — build typed ProductProfile objects
# ---------------------------------------------------------------------------

def product_profiles_from_df(products_df: pd.DataFrame) -> list[ProductProfile]:
    """
    Convert the products_df DataFrame to a list of ProductProfile Pydantic objects.

    movement_class and data_sufficient are intentionally left as placeholders:
    - movement_class is set to "slow" (the only Literal value that is safe
      as a conservative default; classify.py will overwrite this based on
      velocity_per_day tertiles across the full product population).
    - data_sufficient is set to False (classify.py will evaluate this based
      on minimum order count and window length thresholds).

    The evidence dict is populated with methodological metadata so that any
    downstream consumer can understand how these numbers were produced.

    Raises
    ------
    pydantic.ValidationError  if any row fails the ProductProfile schema.
    """
    profiles: list[ProductProfile] = []

    window_days_val = int(products_df["window_days"].iloc[0]) if len(products_df) > 0 else 0

    for _, row in products_df.iterrows():
        evidence: dict[str, Any] = {
            "margin_method": "allocated_from_order_cost_by_revenue_share",
            "product_level_truth": "structural_not_ground_truth",
            "velocity_denominator": "dataset_window_days",
            "source": "sample-data-mongo.json",
            "window_days": window_days_val,
            "product_first_order_date": str(row["product_first_order_date"]),
            "product_last_order_date": str(row["product_last_order_date"]),
            "units_sold": int(row["units_sold"]),
            "raw_allocated_cost": float(row["allocated_cost"]),
        }

        profile = ProductProfile(
            product_id=int(row["product_id"]),
            title=str(row["title"]),
            units_sold=int(row["units_sold"]),
            revenue=float(row["revenue"]),
            gross_margin_alloc=float(row["gross_margin_alloc"]),
            margin_pct=float(row["margin_pct"]),
            velocity_per_day=float(row["velocity_per_day"]),
            # Placeholder: classify.py assigns the real class via tertiles
            movement_class="slow",
            # Placeholder: classify.py evaluates data sufficiency
            data_sufficient=False,
            evidence=evidence,
        )
        profiles.append(profile)

    return profiles


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def run_product_metrics(
    lineitems_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[ProductProfile]]:
    """
    Run the full three-step product metrics pipeline.

        compute_line_revenue
        -> allocate_order_cost_to_lines
        -> compute_product_metrics
        -> product_profiles_from_df

    Returns
    -------
    products_df   : pd.DataFrame
    profiles      : list[ProductProfile]
    """
    enriched = compute_line_revenue(lineitems_df)
    enriched = allocate_order_cost_to_lines(enriched)
    products_df = compute_product_metrics(enriched)
    profiles = product_profiles_from_df(products_df)
    return products_df, profiles


# ---------------------------------------------------------------------------
# __main__ — verification block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from discount_prime_agent.pipeline.ingest import build_clean_frames

    DATA_PATH = "data/sample-data-mongo.json"
    OUT_DIR = "outputs"

    print("=" * 68)
    print("metrics.py -- verification run")
    print("=" * 68)

    # ── ingest ───────────────────────────────────────────────────────────────
    orders_df, lineitems_df, campaigns_df, shop = build_clean_frames(DATA_PATH)

    # ── compute ──────────────────────────────────────────────────────────────
    products_df, profiles = run_product_metrics(lineitems_df)

    # ── print summary ────────────────────────────────────────────────────────
    print(f"\nNumber of products: {len(products_df)}")
    print("  (expected from data: 18)")
    print()

    display_cols = [
        "product_id", "title",
        "units_sold", "revenue",
        "gross_margin_alloc", "margin_pct",
        "velocity_per_day", "window_days",
    ]
    pd.set_option("display.max_rows", 50)
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", "{:.4f}".format)
    print("Product metrics (sorted by velocity_per_day descending):")
    print(products_df[display_cols].to_string(index=False))

    print()
    print(
        "Product margin is approximate because cost_total is order-level "
        "and allocated by revenue share."
    )

    # ── validate a sample profile ─────────────────────────────────────────────
    print(f"\nProductProfile objects created: {len(profiles)}")
    print(f"Sample profile[0] movement_class placeholder: '{profiles[0].movement_class}'")
    print(f"Sample profile[0] data_sufficient placeholder: {profiles[0].data_sufficient}")
    print(f"Sample profile[0] evidence keys: {list(profiles[0].evidence.keys())}")

    # ── save CSV ──────────────────────────────────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = f"{OUT_DIR}/product_metrics.csv"
    # Datetime columns need string conversion for clean CSV output
    save_df = products_df.copy()
    save_df["product_first_order_date"] = save_df["product_first_order_date"].astype(str)
    save_df["product_last_order_date"] = save_df["product_last_order_date"].astype(str)
    save_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  ({len(save_df)} rows)")

    print("\n[OK] metrics.py verification complete.")
