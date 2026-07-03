"""
classify.py
-----------
Deterministic velocity-tertile classification of products.

Reads the products DataFrame produced by metrics.py, assigns movement
classes via pandas qcut (with a rank-based fallback for ties), and returns
both an enriched DataFrame and a list of typed ProductProfile objects.

Schema contract note
--------------------
ProductProfile.movement_class is typed as Literal["fast", "medium", "slow"]
in schemas.py.  The middle tertile is therefore "medium" (not "mid").
Products with data_sufficient=False receive movement_class="slow" because
"inconclusive" is not in the Literal — classify.py documents this
assignment in the evidence dict so callers can filter on data_sufficient.

Public API
----------
    classify_products(products_df, min_units=MIN_UNITS) -> pd.DataFrame
    product_profiles_from_classified_df(classified_df)  -> list[ProductProfile]

Classification rules (deterministic, no hardcoded thresholds)
--------------------------------------------------------------
1. velocity_per_day is taken directly from products_df (computed in
   metrics.py).  It is NEVER recomputed here.
2. Products with units_sold < MIN_UNITS:
   - data_sufficient = False
   - movement_class  = "slow"  (schema-safe placeholder; caller should
     filter on data_sufficient before using movement_class)
3. For products with data_sufficient=True, tertiles are assigned:
   - Primary method: pandas.qcut(q=3) on velocity_per_day
     labels=["slow", "medium", "fast"]
   - Fallback (duplicate bin edges / ties): rank velocity_per_day
     descending, split into three groups as evenly as possible
     (top third → fast, middle third → medium, bottom third → slow)
   The fallback is also purely data-driven — no hardcoded values.
"""

from __future__ import annotations

import os
import warnings
from typing import Any

import numpy as np
import pandas as pd

from discount_prime_agent.schemas import ProductProfile


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

MIN_UNITS: int = 20
"""Minimum units_sold for a product to have data_sufficient=True."""

# qcut labels ordered from slowest to fastest (matches qcut ascending bins)
_QCUT_LABELS: list[str] = ["slow", "medium", "fast"]


# ---------------------------------------------------------------------------
# Tertile assignment helpers
# ---------------------------------------------------------------------------

def _assign_tertiles_qcut(velocity: pd.Series) -> pd.Series:
    """
    Attempt pandas.qcut tertile assignment.

    Returns a Series of str labels or raises ValueError if bin edges are
    not unique (i.e., too many tied velocity values for qcut to split).
    """
    labels = pd.qcut(velocity, q=3, labels=_QCUT_LABELS, duplicates="raise")
    return labels.astype(str)


def _assign_tertiles_rank(velocity: pd.Series) -> pd.Series:
    """
    Rank-based tertile fallback for when qcut fails due to tied values.

    Products are ranked by velocity_per_day descending (ties get average
    rank).  The ranked list is then split into thirds as evenly as possible:
      - top third    → "fast"
      - middle third → "medium"
      - bottom third → "slow"

    All arithmetic is on the data — no hardcoded thresholds.
    """
    n = len(velocity)
    # Rank ascending (lowest velocity → rank 1), use average for ties
    rank = velocity.rank(method="average", ascending=True)

    # Tertile boundaries (inclusive upper edges for each third)
    third = n / 3.0
    labels = pd.Series(index=velocity.index, dtype="str")
    labels[rank <= third]         = "slow"
    labels[(rank > third) & (rank <= 2 * third)] = "medium"
    labels[rank > 2 * third]      = "fast"
    return labels


def _compute_tertile_classes(velocity: pd.Series) -> pd.Series:
    """
    Assign tertile movement classes to a Series of velocity values.

    Tries qcut first; falls back to rank-based assignment and emits a
    warning so callers know which method was used.
    """
    if len(velocity) == 0:
        return pd.Series(dtype="str")

    # Single product or all identical — assign "slow" universally
    if velocity.nunique() == 1:
        warnings.warn(
            "[classify] All supplied velocity values are identical; "
            "assigning movement_class='slow' to all.",
            stacklevel=3,
        )
        return pd.Series("slow", index=velocity.index, dtype="str")

    try:
        return _assign_tertiles_qcut(velocity)
    except ValueError:
        warnings.warn(
            "[classify] qcut failed due to duplicate bin edges (tied velocities). "
            "Falling back to rank-based tertile assignment.",
            stacklevel=3,
        )
        return _assign_tertiles_rank(velocity)


# ---------------------------------------------------------------------------
# classify_products
# ---------------------------------------------------------------------------

def classify_products(
    products_df: pd.DataFrame,
    min_units: int = MIN_UNITS,
) -> pd.DataFrame:
    """
    Classify products by velocity tertile and assess data sufficiency.

    Parameters
    ----------
    products_df : pd.DataFrame
        Output of metrics.compute_product_metrics() — must contain at
        minimum: product_id, title, units_sold, velocity_per_day,
        revenue, gross_margin_alloc, margin_pct.
    min_units : int
        Minimum units_sold threshold for data_sufficient=True.
        Defaults to the module constant MIN_UNITS (20).

    Returns
    -------
    pd.DataFrame
        A copy of products_df with two new columns:
          movement_class  : str  ("fast" | "medium" | "slow")
          data_sufficient : bool
        and the following columns guaranteed present for downstream use:
          product_id, title, units_sold, revenue, gross_margin_alloc,
          margin_pct, velocity_per_day, movement_class, data_sufficient.

    Classification rules
    --------------------
    - data_sufficient = (units_sold >= min_units)
    - Products with data_sufficient=False → movement_class = "slow"
      (schema-safe; check data_sufficient before using movement_class)
    - Products with data_sufficient=True → tertile by velocity_per_day:
        top third    → "fast"
        middle third → "medium"
        bottom third → "slow"
    """
    required = {
        "product_id", "title", "units_sold",
        "velocity_per_day", "revenue", "gross_margin_alloc", "margin_pct",
    }
    missing = required - set(products_df.columns)
    if missing:
        raise ValueError(
            f"[classify] products_df is missing required columns: {sorted(missing)}"
        )

    df = products_df.copy()

    # ── sufficiency flag ─────────────────────────────────────────────────────
    df["data_sufficient"] = df["units_sold"] >= min_units

    # ── initialise movement_class to "slow" for all rows ────────────────────
    df["movement_class"] = "slow"

    # ── tertile assignment on sufficient products only ───────────────────────
    sufficient_mask = df["data_sufficient"]
    n_sufficient = sufficient_mask.sum()

    if n_sufficient >= 3:
        # Assign tertiles only within the sufficient subset
        sufficient_velocity = df.loc[sufficient_mask, "velocity_per_day"]
        tertile_labels = _compute_tertile_classes(sufficient_velocity)
        df.loc[sufficient_mask, "movement_class"] = tertile_labels
    elif n_sufficient > 0:
        # Fewer than 3 sufficient products — cannot split into 3 tertiles;
        # assign "fast" to highest velocity, "slow" to all others
        warnings.warn(
            f"[classify] Only {n_sufficient} product(s) have data_sufficient=True. "
            "Cannot form 3 tertiles; assigning 'fast' to the highest-velocity "
            "product and 'slow' to the rest.",
            stacklevel=2,
        )
        top_idx = df.loc[sufficient_mask, "velocity_per_day"].idxmax()
        df.loc[sufficient_mask, "movement_class"] = "slow"
        df.loc[top_idx, "movement_class"] = "fast"
    # else: n_sufficient == 0 → all remain "slow"

    # ── validate all movement_class values are schema-legal ─────────────────
    valid_classes = {"fast", "medium", "slow"}
    bad = ~df["movement_class"].isin(valid_classes)
    if bad.any():
        raise ValueError(
            f"[classify] movement_class contains values outside "
            f"{valid_classes}: {df.loc[bad, 'movement_class'].unique().tolist()}"
        )

    return df


# ---------------------------------------------------------------------------
# product_profiles_from_classified_df
# ---------------------------------------------------------------------------

def product_profiles_from_classified_df(
    classified_df: pd.DataFrame,
) -> list[ProductProfile]:
    """
    Convert the classified DataFrame to a list of typed ProductProfile objects.

    Raises pydantic.ValidationError loudly on any schema mismatch (Rule 9).

    Evidence dict includes:
    - velocity_per_day         : float  — from metrics.py
    - units_sold               : int    — raw count
    - margin_method            : str    — allocation method description
    - product_level_truth      : str    — caveat from meta.note
    - classification_method    : str    — "qcut_tertile" or "rank_tertile"
                                         (recorded from the df if present,
                                          else inferred as "tertile")
    - min_units                : int    — threshold used for data_sufficient
    """
    profiles: list[ProductProfile] = []

    for _, row in classified_df.iterrows():
        evidence: dict[str, Any] = {
            "velocity_per_day": float(row["velocity_per_day"]),
            "units_sold": int(row["units_sold"]),
            "margin_method": "allocated_from_order_cost_by_revenue_share",
            "product_level_truth": "structural_not_ground_truth",
            "classification_method": "tertile_on_velocity_per_day",
            "min_units": MIN_UNITS,
        }

        # Propagate evidence columns from metrics.py if present
        for extra_key in (
            "window_days", "product_first_order_date",
            "product_last_order_date", "raw_allocated_cost",
        ):
            if extra_key in row.index and pd.notna(row[extra_key]):
                evidence[extra_key] = str(row[extra_key])

        if not row["data_sufficient"]:
            evidence["data_sufficient_note"] = (
                f"units_sold ({int(row['units_sold'])}) < min_units ({MIN_UNITS}). "
                "movement_class='slow' is a schema-safe placeholder."
            )

        profile = ProductProfile(
            product_id=int(row["product_id"]),
            title=str(row["title"]),
            units_sold=int(row["units_sold"]),
            revenue=float(row["revenue"]),
            gross_margin_alloc=float(row["gross_margin_alloc"]),
            margin_pct=float(row["margin_pct"]),
            velocity_per_day=float(row["velocity_per_day"]),
            movement_class=str(row["movement_class"]),   # "fast"|"medium"|"slow"
            data_sufficient=bool(row["data_sufficient"]),
            evidence=evidence,
        )
        profiles.append(profile)

    return profiles


# ---------------------------------------------------------------------------
# __main__ — verification block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from discount_prime_agent.pipeline.ingest import build_clean_frames
    from discount_prime_agent.pipeline.metrics import (
        compute_line_revenue,
        allocate_order_cost_to_lines,
        compute_product_metrics,
    )

    DATA_PATH = "data/sample-data-mongo.json"
    OUT_DIR = "outputs"

    print("=" * 72)
    print("classify.py -- verification run")
    print("=" * 72)

    # ── ingest ────────────────────────────────────────────────────────────────
    orders_df, lineitems_df, campaigns_df, shop = build_clean_frames(DATA_PATH)

    # ── metrics (using existing pipeline, not recomputing differently) ───────
    enriched = compute_line_revenue(lineitems_df)
    enriched = allocate_order_cost_to_lines(enriched)
    products_df = compute_product_metrics(enriched)

    # ── classify ──────────────────────────────────────────────────────────────
    classified_df = classify_products(products_df, min_units=MIN_UNITS)
    profiles = product_profiles_from_classified_df(classified_df)

    # ── print classification table ────────────────────────────────────────────
    display_cols = [
        "product_id", "title",
        "units_sold", "velocity_per_day",
        "movement_class", "data_sufficient",
    ]
    pd.set_option("display.max_rows", 30)
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", "{:.4f}".format)

    print("\nProduct classification (sorted by velocity_per_day descending):")
    print(
        classified_df[display_cols]
        .sort_values("velocity_per_day", ascending=False)
        .to_string(index=False)
    )

    # ── class counts ──────────────────────────────────────────────────────────
    print("\nClass counts:")
    counts = classified_df["movement_class"].value_counts()
    for cls in ["fast", "medium", "slow"]:
        print(f"  {cls:8s}: {counts.get(cls, 0)}")

    print(f"\ndata_sufficient=False count: {(~classified_df['data_sufficient']).sum()}")

    # ── validate profiles ─────────────────────────────────────────────────────
    print(f"\nProductProfile objects created: {len(profiles)}")
    sample = profiles[0]
    print(f"  Sample profile[0]: {sample.title!r}")
    print(f"    movement_class  : {sample.movement_class!r}")
    print(f"    data_sufficient : {sample.data_sufficient}")
    print(f"    evidence keys   : {list(sample.evidence.keys())}")

    # ── save CSV ──────────────────────────────────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = f"{OUT_DIR}/product_classification.csv"

    save_cols = [
        "product_id", "title",
        "units_sold", "revenue",
        "gross_margin_alloc", "margin_pct",
        "velocity_per_day",
        "movement_class", "data_sufficient",
    ]
    save_df = classified_df[save_cols].copy()
    save_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  ({len(save_df)} rows, {len(save_cols)} cols)")

    print("\nClassification complete")
