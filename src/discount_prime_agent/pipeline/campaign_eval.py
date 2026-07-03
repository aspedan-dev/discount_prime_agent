"""
campaign_eval.py
----------------
Deterministic pandas campaign evaluation.  Judges each campaign on profit
(margin per day) using real attribution signals from the data.

Data findings (confirmed by inspecting data/sample-data-mongo.json)
-------------------------------------------------------------------
Non-shipping campaigns (sc1, sc3, sc4, sc5, sc7, sc10):
    Attributed via  lineitems_df.active_campaign_id == campaign.id
Shipping campaigns (sc2, sc8, sc9):
    Attributed via  orders_df.order_campaigns[].campaignId == campaign.id
    All line items of those orders are included in the campaign window.
Order-type campaign (sc6 "Spend $100, Save $15"):
    Has zero active_campaign_id hits on lineitems; attributed via
    order_campaigns just like shipping campaigns.

Attribution rule used:
    If  lineitems.active_campaign_id  yields > 0 hits for the campaign →
        use lineitem-level attribution.
    Else →
        use order_campaigns attribution (all lineitems of qualifying orders).

This is data-driven: no campaign IDs or names are hardcoded.

Grain rules (strictly enforced)
--------------------------------
The following are ORDER-GRAIN columns duplicated across line items of the
same order — they are NEVER summed directly across lineitems rows:
    cost_total, revenue_with_cost, order_total_price, order_total_discounts,
    shipping_original_price, shipping_price_charged, shipping_discount_amount,
    is_free_shipping

For shipping_cost_impact:
    orders_df.drop_duplicates("order_id"), then sum(shipping_discount_amount)

Public API
----------
    prepare_line_margin_df(lineitems_df)                          -> pd.DataFrame
    campaign_attributed_lineitems(cid, ctype, lineitems_df,
                                  orders_df)                      -> pd.DataFrame
    baseline_lineitems_for_campaign(attributed_df, lineitems_df)  -> pd.DataFrame
    evaluate_campaigns(orders_df, lineitems_df, campaigns_df)
        -> tuple[pd.DataFrame, list[CampaignVerdict]]
"""

from __future__ import annotations

import os
import warnings
from typing import Any

import pandas as pd

from discount_prime_agent.schemas import CampaignVerdict


# ---------------------------------------------------------------------------
# Config constants
# ---------------------------------------------------------------------------

MIN_CAMPAIGN_DAYS: int = 3
MIN_BASELINE_DAYS: int = 3
MIN_CAMPAIGN_UNITS: int = 20
MIN_BASELINE_UNITS: int = 20
PROFIT_LIFT_THRESHOLD: float = 0.20   # campaign margin/day must be 20% above baseline
UNITS_LIFT_THRESHOLD: float = 1.10    # velocity ratio considered meaningful


# ---------------------------------------------------------------------------
# Step 1 — prepare line-level margin DataFrame
# ---------------------------------------------------------------------------

def prepare_line_margin_df(lineitems_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute line-level revenue and allocated cost from lineitems_df.

    Returns a copy with additional columns:
        line_gross_revenue  = quantity * price
        line_net_revenue    = line_gross_revenue - total_discount
        allocated_cost      = cost_total * (line_net_revenue / order_net_revenue)
                              (0.0 when order_net_revenue == 0)
        line_margin_alloc   = line_net_revenue - allocated_cost

    ORDER-GRAIN rule enforced:
        cost_total is accessed per row (same value within an order) and
        multiplied by the revenue share — it is NEVER summed directly.
    """
    df = lineitems_df.copy()

    df["line_gross_revenue"] = df["quantity"] * df["price"]
    df["line_net_revenue"] = df["line_gross_revenue"] - df["total_discount"]

    # Clip negative net revenue (defensive; audit confirmed none in real data)
    neg = df["line_net_revenue"] < 0
    if neg.any():
        warnings.warn(
            f"[campaign_eval] {neg.sum()} line item(s) have line_net_revenue < 0. "
            "Clipping to 0.0.",
            stacklevel=2,
        )
        df.loc[neg, "line_net_revenue"] = 0.0

    # Per-order sum of net revenue (line-grain sum — correct)
    order_net_rev = (
        df.groupby("order_id")["line_net_revenue"]
        .sum()
        .rename("order_net_revenue")
    )
    df = df.join(order_net_rev, on="order_id")

    # Revenue share — safe division (no inf / NaN)
    share = df["line_net_revenue"] / df["order_net_revenue"].replace(0, float("nan"))
    share = share.fillna(0.0)

    # cost_total is ORDER-GRAIN: same on every row of the same order.
    # Multiplying by revenue share gives the correct per-line allocation.
    df["allocated_cost"] = (df["cost_total"] * share).round(6)
    df["line_margin_alloc"] = (df["line_net_revenue"] - df["allocated_cost"]).round(6)

    df = df.drop(columns=["order_net_revenue"])
    return df


# ---------------------------------------------------------------------------
# Step 2 — campaign attribution
# ---------------------------------------------------------------------------

def _order_ids_from_order_campaigns(
    campaign_id: str,
    orders_df: pd.DataFrame,
) -> set[int]:
    """
    Return the set of order_ids where order_campaigns contains the given
    campaignId.  order_campaigns is a list-of-dicts column.
    """
    mask = orders_df["order_campaigns"].apply(
        lambda lst: any(
            isinstance(d, dict) and d.get("campaignId") == campaign_id
            for d in lst
        )
    )
    return set(orders_df.loc[mask, "order_id"].tolist())


def campaign_attributed_lineitems(
    campaign_id: str,
    campaign_type: str,
    lineitems_df: pd.DataFrame,
    orders_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return the subset of lineitems_df attributed to a given campaign.

    Attribution logic (data-driven, no hardcoded IDs):
    1. Try lineitem-level: rows where active_campaign_id == campaign_id.
    2. If that yields 0 rows, fall back to order-level:
       find orders where order_campaigns contains campaignId == campaign_id,
       then return all line items for those order_ids.

    The fallback covers shipping-type and order-type campaigns (sc2, sc6,
    sc8, sc9) which store attribution in order_campaigns, not in
    active_campaign_id on the line item.

    Parameters
    ----------
    campaign_id   : str   e.g. "sc2"
    campaign_type : str   e.g. "shipping" — stored in the returned df metadata
    lineitems_df  : pd.DataFrame  — must contain line_margin_alloc (from
                    prepare_line_margin_df)
    orders_df     : pd.DataFrame  — clean orders from ingest.py
    """
    # -- try line-item level attribution first --------------------------------
    li_attributed = lineitems_df[
        lineitems_df["active_campaign_id"] == campaign_id
    ].copy()

    if len(li_attributed) > 0:
        return li_attributed

    # -- fallback: order_campaigns attribution --------------------------------
    attributed_order_ids = _order_ids_from_order_campaigns(campaign_id, orders_df)
    if not attributed_order_ids:
        return lineitems_df.iloc[0:0].copy()  # empty DataFrame, correct schema

    return lineitems_df[
        lineitems_df["order_id"].isin(attributed_order_ids)
    ].copy()


# ---------------------------------------------------------------------------
# Step 3 — baseline
# ---------------------------------------------------------------------------

def baseline_lineitems_for_campaign(
    attributed_df: pd.DataFrame,
    lineitems_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return the baseline lineitems for a campaign:
    - Same product_ids as the attributed lineitems.
    - Only lineitems where active_campaign_id is null/None/NaN.
    - Exclude order_ids that appear in attributed_df.

    This isolates "organic" sales of the same products, untouched by any
    campaign discount — the natural comparison group.
    """
    if len(attributed_df) == 0:
        return lineitems_df.iloc[0:0].copy()

    relevant_products = set(attributed_df["product_id"].unique())
    excluded_orders = set(attributed_df["order_id"].unique())

    baseline = lineitems_df[
        lineitems_df["product_id"].isin(relevant_products)
        & lineitems_df["active_campaign_id"].isna()
        & ~lineitems_df["order_id"].isin(excluded_orders)
    ].copy()

    return baseline


# ---------------------------------------------------------------------------
# Step 4 — per-campaign metrics and verdict
# ---------------------------------------------------------------------------

def _date_span_days(df: pd.DataFrame, date_col: str = "createdAt") -> int:
    """Inclusive day span between min and max of a datetime column. Minimum 1."""
    if len(df) == 0:
        return 0
    first = df[date_col].min()
    last = df[date_col].max()
    delta = (last - first).days + 1
    return max(int(delta), 1)


def _shipping_cost_impact(
    attributed_df: pd.DataFrame,
    orders_df: pd.DataFrame,
) -> float:
    """
    Sum shipping_discount_amount from distinct attributed orders only.

    shipping_discount_amount is an ORDER-GRAIN field — we must use
    orders_df (or drop_duplicates on order_id) and NEVER sum it from
    lineitems rows directly.
    """
    attributed_order_ids = set(attributed_df["order_id"].unique())
    qualifying_orders = (
        orders_df[orders_df["order_id"].isin(attributed_order_ids)]
        .drop_duplicates("order_id")
    )
    return float(qualifying_orders["shipping_discount_amount"].sum())


def _sufficiency_check(
    campaign_days: int,
    baseline_days: int,
    campaign_units: int,
    baseline_units: int,
    attributed_orders: int,
    baseline_margin_per_day: float,
    campaign_margin_per_day: float,
) -> tuple[bool, list[str]]:
    """
    Return (is_sufficient, list_of_failed_conditions).
    """
    failures: list[str] = []
    if campaign_days < MIN_CAMPAIGN_DAYS:
        failures.append(f"campaign_days={campaign_days}<{MIN_CAMPAIGN_DAYS}")
    if baseline_days < MIN_BASELINE_DAYS:
        failures.append(f"baseline_days={baseline_days}<{MIN_BASELINE_DAYS}")
    if campaign_units < MIN_CAMPAIGN_UNITS:
        failures.append(f"campaign_units={campaign_units}<{MIN_CAMPAIGN_UNITS}")
    if baseline_units < MIN_BASELINE_UNITS:
        failures.append(f"baseline_units={baseline_units}<{MIN_BASELINE_UNITS}")
    if attributed_orders <= 0:
        failures.append("attributed_orders=0")
    if not pd.notna(baseline_margin_per_day) or baseline_days == 0:
        failures.append("baseline_margin_per_day_invalid")
    if not pd.notna(campaign_margin_per_day) or campaign_days == 0:
        failures.append("campaign_margin_per_day_invalid")
    return len(failures) == 0, failures


def _assign_verdict(
    campaign_margin_per_day: float,
    baseline_margin_per_day: float,
    units_lift_ratio: float,
    is_sufficient: bool,
    sufficiency_failures: list[str],
    campaign_type: str,
) -> tuple[str, float, str]:
    """
    Return (verdict, confidence, reason_code).

    Verdict is profit-first; units_lift_ratio is a confidence signal only.
    """
    if not is_sufficient:
        reason = "low_n:" + "|".join(sufficiency_failures)
        return "inconclusive", 0.2, reason

    # baseline must be non-zero for ratio comparison; if zero treat as
    # inconclusive (we can't interpret "infinite lift" reliably)
    if baseline_margin_per_day == 0:
        return "inconclusive", 0.3, "baseline_margin_per_day_zero"

    lift_ratio = campaign_margin_per_day / baseline_margin_per_day

    # -- verdict decision tree ------------------------------------------------
    if lift_ratio >= (1 + PROFIT_LIFT_THRESHOLD):
        verdict = "success"
        # Units lift as confidence modifier
        if units_lift_ratio > UNITS_LIFT_THRESHOLD:
            confidence = 0.85
            reason = "margin_lift|volume_lift"
        elif units_lift_ratio <= 1.0:
            confidence = 0.60
            reason = "margin_gain_without_volume_lift"
        else:
            confidence = 0.75
            reason = "margin_lift"
        if campaign_type == "shipping":
            reason += "|shipping_order_level_attribution_coarse"
            confidence = min(confidence, 0.70)

    elif lift_ratio <= 1.0:
        verdict = "flop"
        if units_lift_ratio > UNITS_LIFT_THRESHOLD:
            confidence = 0.80
            reason = "volume_lift_margin_eroded"
        else:
            confidence = 0.80
            reason = "margin_erosion"
        if campaign_type == "shipping":
            reason += "|shipping_order_level_attribution_coarse"
            confidence = min(confidence, 0.65)

    else:
        # Between 1.0x and 1.2x lift — not enough to call success
        verdict = "inconclusive"
        confidence = 0.50
        reason = "margin_lift_below_threshold"
        if campaign_type == "shipping":
            reason += "|shipping_order_level_attribution_coarse"
            confidence = min(confidence, 0.45)

    return verdict, confidence, reason


def _eval_single_campaign(
    row: "pd.Series",
    lineitems_margin_df: pd.DataFrame,
    orders_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Compute all metrics and verdict for one campaign row from campaigns_df.
    """
    cid: str = row["id"]
    ctype: str = row["type"]

    # -- attributed lineitems --------------------------------------------------
    attributed = campaign_attributed_lineitems(
        cid, ctype, lineitems_margin_df, orders_df
    )
    # -- baseline lineitems ----------------------------------------------------
    baseline = baseline_lineitems_for_campaign(attributed, lineitems_margin_df)

    # -- campaign metrics ------------------------------------------------------
    attributed_orders = int(attributed["order_id"].nunique()) if len(attributed) else 0
    campaign_units = int(attributed["quantity"].sum()) if len(attributed) else 0
    baseline_units = int(baseline["quantity"].sum()) if len(baseline) else 0
    campaign_days = _date_span_days(attributed) if len(attributed) else 0
    baseline_days = _date_span_days(baseline) if len(baseline) else 0

    # margin totals — safe to sum line_margin_alloc (it is LINE-GRAIN)
    campaign_margin_total = float(attributed["line_margin_alloc"].sum()) if len(attributed) else 0.0
    baseline_margin_total = float(baseline["line_margin_alloc"].sum()) if len(baseline) else 0.0

    # shipping cost impact — order-grain, deduplicated
    shipping_cost_impact: float | None = None
    if ctype == "shipping" and len(attributed) > 0:
        shipping_cost_impact = _shipping_cost_impact(attributed, orders_df)
        # Subtract shipping cost from campaign margin before computing per-day
        campaign_margin_total -= shipping_cost_impact

    # per-day rates (safe division)
    campaign_margin_per_day = (
        campaign_margin_total / campaign_days if campaign_days > 0 else 0.0
    )
    baseline_margin_per_day = (
        baseline_margin_total / baseline_days if baseline_days > 0 else 0.0
    )

    # velocity and lift ratio
    campaign_velocity = campaign_units / campaign_days if campaign_days > 0 else 0.0
    baseline_velocity = baseline_units / baseline_days if baseline_days > 0 else 0.0

    if baseline_velocity > 0:
        units_lift_ratio = campaign_velocity / baseline_velocity
    else:
        units_lift_ratio = 0.0  # baseline_velocity==0 → inconclusive signal

    # -- sufficiency + verdict --------------------------------------------------
    is_sufficient, failures = _sufficiency_check(
        campaign_days, baseline_days,
        campaign_units, baseline_units,
        attributed_orders,
        baseline_margin_per_day if baseline_days > 0 else float("nan"),
        campaign_margin_per_day if campaign_days > 0 else float("nan"),
    )

    # If baseline_velocity == 0, override confidence down
    if baseline_velocity == 0 and is_sufficient:
        is_sufficient = False
        failures.append("baseline_velocity_zero")

    verdict, confidence, reason_code = _assign_verdict(
        campaign_margin_per_day,
        baseline_margin_per_day,
        units_lift_ratio,
        is_sufficient,
        failures,
        ctype,
    )

    return {
        "campaign_id": cid,
        "campaign_name": row["name"],
        "type": ctype,
        "status": row["status"],
        "window_days": campaign_days,
        "attributed_orders": attributed_orders,
        "campaign_units": campaign_units,
        "baseline_units": baseline_units,
        "campaign_margin_total": round(campaign_margin_total, 2),
        "baseline_margin_total": round(baseline_margin_total, 2),
        "margin_per_day_campaign": round(campaign_margin_per_day, 4),
        "margin_per_day_baseline": round(baseline_margin_per_day, 4),
        "units_lift_ratio": round(units_lift_ratio, 4),
        "shipping_cost_impact": round(shipping_cost_impact, 2)
        if shipping_cost_impact is not None
        else None,
        "verdict": verdict,
        "confidence": round(confidence, 2),
        "reason_code": reason_code,
        # Cross-check columns (from campaigns_df — not used for verdict)
        "reported_totalRevenue": row["totalRevenue"],
        "reported_marginPct": row["marginPct"],
    }


# ---------------------------------------------------------------------------
# Step 5 — evaluate all campaigns
# ---------------------------------------------------------------------------

def evaluate_campaigns(
    orders_df: pd.DataFrame,
    lineitems_df: pd.DataFrame,
    campaigns_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[CampaignVerdict]]:
    """
    Evaluate all campaigns and return a summary DataFrame and typed verdicts.

    Parameters
    ----------
    orders_df     : clean orders from ingest.py
    lineitems_df  : clean lineitems from ingest.py
    campaigns_df  : clean campaigns from ingest.py

    Returns
    -------
    campaign_eval_df : pd.DataFrame  (one row per campaign)
    verdicts         : list[CampaignVerdict]  (Pydantic-validated)
    """
    # Prepare line-level margins once — shared across all campaign evaluations
    lineitems_margin_df = prepare_line_margin_df(lineitems_df)

    rows: list[dict] = []
    for _, camp_row in campaigns_df.iterrows():
        result = _eval_single_campaign(camp_row, lineitems_margin_df, orders_df)
        rows.append(result)

    campaign_eval_df = pd.DataFrame(rows)

    # Build typed CampaignVerdict objects — raises loudly on schema mismatch
    verdicts: list[CampaignVerdict] = []
    for _, r in campaign_eval_df.iterrows():
        verdict_obj = CampaignVerdict(
            campaign_id=str(r["campaign_id"]),
            type=str(r["type"]),
            window_days=int(r["window_days"]),
            attributed_orders=int(r["attributed_orders"]),
            units_lift_ratio=float(r["units_lift_ratio"]),
            margin_per_day_baseline=float(r["margin_per_day_baseline"]),
            margin_per_day_campaign=float(r["margin_per_day_campaign"]),
            shipping_cost_impact=(
                float(r["shipping_cost_impact"])
                if r["shipping_cost_impact"] is not None
                else None
            ),
            verdict=str(r["verdict"]),
            confidence=float(r["confidence"]),
            reason_code=str(r["reason_code"]),
        )
        verdicts.append(verdict_obj)

    return campaign_eval_df, verdicts


# ---------------------------------------------------------------------------
# __main__ — verification block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from discount_prime_agent.pipeline.ingest import build_clean_frames

    DATA_PATH = "data/sample-data-mongo.json"
    OUT_DIR = "outputs"

    print("=" * 72)
    print("campaign_eval.py -- verification run")
    print("=" * 72)

    orders_df, lineitems_df, campaigns_df, shop = build_clean_frames(DATA_PATH)

    campaign_eval_df, verdicts = evaluate_campaigns(orders_df, lineitems_df, campaigns_df)

    # -- print results table --------------------------------------------------
    display_cols = [
        "campaign_id", "type", "attributed_orders",
        "units_lift_ratio",
        "margin_per_day_baseline", "margin_per_day_campaign",
        "shipping_cost_impact",
        "verdict", "confidence", "reason_code",
    ]
    pd.set_option("display.max_rows", 20)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.4f}".format)

    print("\nCampaign evaluation results:")
    print(campaign_eval_df[display_cols].to_string(index=False))

    # -- cross-check against reported values ----------------------------------
    print("\nCross-check: reported vs computed totals (order-level grain, informational only)")
    xcheck_cols = [
        "campaign_id", "campaign_name",
        "reported_totalRevenue", "reported_marginPct",
        "campaign_margin_total", "verdict",
    ]
    print(campaign_eval_df[xcheck_cols].to_string(index=False))

    # -- verdict counts -------------------------------------------------------
    print("\nVerdict counts:")
    for v in ["success", "flop", "inconclusive"]:
        n = (campaign_eval_df["verdict"] == v).sum()
        print(f"  {v:15s}: {n}")

    # -- validate typed objects -----------------------------------------------
    print(f"\nCampaignVerdict objects created: {len(verdicts)}")
    print(f"  Sample[0]: campaign_id={verdicts[0].campaign_id!r}  "
          f"verdict={verdicts[0].verdict!r}  "
          f"confidence={verdicts[0].confidence}")

    # -- save CSV -------------------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = f"{OUT_DIR}/campaign_eval.csv"
    campaign_eval_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  ({len(campaign_eval_df)} rows)")

    print("\nCampaign evaluation complete")
