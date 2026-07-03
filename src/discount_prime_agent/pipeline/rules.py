"""
rules.py
--------
Deterministic product recommendation engine.  Plain Python + pandas, no LLM.

Turns product classifications (from classify.py) and campaign verdicts
(from campaign_eval.py) into one typed Recommendation per product.

Schema contract note
--------------------
Recommendation (schemas.py) has these fields:
    product_id, title, recommended_mechanic, rationale,
    evidence_refs (list[str]), confidence (float 0-1),
    priority_score (float), expected_effect (str)

There is NO risk_note field.  Risk notes are encoded inside:
    - expected_effect  (human-readable outcome description + risk caveats)
    - evidence_refs    (machine-readable reference tags)

Any extra kwargs would be silently dropped by extra="ignore", so we map
all risk content to the fields that actually exist.

The spec labels confidence as "high/medium/low" for rule design convenience;
we convert to floats before building Recommendation objects:
    high   -> 1.0
    medium -> 0.6
    low    -> 0.3

Public API
----------
    summarize_campaign_evidence(campaign_eval_df)          -> dict
    recommend_for_products(classified_df, campaign_eval_df,
                           affinity_df=None)
        -> tuple[pd.DataFrame, list[Recommendation]]

Recommendation rules (deterministic, no hardcoded names/thresholds)
--------------------------------------------------------------------
Rule 0  — data_sufficient=False or movement_class not in {fast,medium,slow}
           -> "Collect more data"
Step A  — 3x3 matrix: margin_band (low/mid/high) x movement_class (fast/medium/slow)
Step B  — campaign-evidence modifiers applied after Step A
Priority — recomputed from the FINAL mechanic after Step B
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd

from discount_prime_agent.schemas import Recommendation
from discount_prime_agent.pipeline.classify import MIN_UNITS


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Campaign type -> merchant-friendly mechanic label
CAMPAIGN_TYPE_TO_MECHANIC: dict[str, str] = {
    "shipping":        "Free Shipping",
    "buy_one_get_one": "BXGY",
    "volume":          "Volume/Bundle",
    "order":           "Order-level Discount",
    "general":         "Generic Discount",
}

# Effort to implement each mechanic (denominator in priority_score)
MECHANIC_EFFORT: dict[str, float] = {
    "Leave/minimal discount":  1.0,
    "Free Shipping":           2.0,
    "BXGY":                    3.0,
    "Volume/Bundle":           3.0,
    "Bundle/Cross-sell":       4.0,
    "Collect more data":       1.0,
    "Avoid deeper discounts":  1.0,
    "Order-level Discount":    2.0,
    "Generic Discount":        2.0,
}

# Margin opportunity -> numeric value (numerator in priority_score)
MARGIN_OPP_VALUE: dict[str, float] = {
    "high":   3.0,
    "medium": 2.0,
    "low":    1.0,
}

# Confidence label -> float
CONFIDENCE_FLOAT: dict[str, float] = {
    "high":   1.0,
    "medium": 0.6,
    "low":    0.3,
}

# Structural margin caveat (per meta.note in sample data)
_STRUCTURAL_CAVEAT = (
    "Product-level metrics are structural because per-line product fields "
    "are synthesized in the sample."
)
_SHIPPING_CAVEAT = (
    "Shipping attribution is order-level and coarse."
)


# ---------------------------------------------------------------------------
# Priority score
# ---------------------------------------------------------------------------

def _priority_score(
    margin_opportunity: str,
    confidence_label: str,
    mechanic: str,
) -> float:
    """
    priority_score = margin_opportunity_value * confidence_weight / effort

    Never returns NaN or inf: all lookup keys have defaults.
    """
    mo = MARGIN_OPP_VALUE.get(margin_opportunity, 1.0)
    cw = CONFIDENCE_FLOAT.get(confidence_label, 0.3)
    ef = MECHANIC_EFFORT.get(mechanic, 1.0)
    if ef == 0:
        ef = 1.0   # defensive; effort is always > 0 in the table
    return round(mo * cw / ef, 4)


# ---------------------------------------------------------------------------
# Campaign evidence summary
# ---------------------------------------------------------------------------

def summarize_campaign_evidence(campaign_eval_df: pd.DataFrame) -> dict:
    """
    Collapse campaign_eval_df into a lookup dict used by the recommendation
    rules in Step B.

    Returns
    -------
    dict with keys:
        "success_types"    : set[str]   campaign types with >= 1 success verdict
        "flop_types"       : set[str]   campaign types with >= 1 flop verdict
        "inconclusive_types": set[str]  campaign types with only inconclusive
        "flopped_campaigns": list[dict] full rows where verdict=="flop"
        "success_campaigns": list[dict] full rows where verdict=="success"
        "shipping_roi_positive": bool   True if any shipping campaign succeeded
        "order_discount_flopped": bool  True if any order-type campaign flopped
        "volume_lift_margin_eroded": bool  True if any flop has that reason_code
        "raw_df": pd.DataFrame          the full eval df for advanced lookups
    """
    df = campaign_eval_df.copy()

    success_types: set[str] = set(df.loc[df["verdict"] == "success", "type"].tolist())
    flop_types: set[str]    = set(df.loc[df["verdict"] == "flop",    "type"].tolist())
    inconclusive_types: set[str] = set(
        df.loc[df["verdict"] == "inconclusive", "type"].tolist()
    ) - success_types - flop_types

    flopped_campaigns = df[df["verdict"] == "flop"].to_dict("records")
    success_campaigns = df[df["verdict"] == "success"].to_dict("records")

    shipping_roi_positive = "shipping" in success_types

    order_discount_flopped = "order" in flop_types

    volume_lift_margin_eroded = any(
        "volume_lift_margin_eroded" in str(r.get("reason_code", ""))
        for r in flopped_campaigns
    )

    return {
        "success_types":        success_types,
        "flop_types":           flop_types,
        "inconclusive_types":   inconclusive_types,
        "flopped_campaigns":    flopped_campaigns,
        "success_campaigns":    success_campaigns,
        "shipping_roi_positive":    shipping_roi_positive,
        "order_discount_flopped":   order_discount_flopped,
        "volume_lift_margin_eroded": volume_lift_margin_eroded,
        "raw_df": df,
    }


# ---------------------------------------------------------------------------
# Step A — 3x3 base matrix
# ---------------------------------------------------------------------------

def _step_a(
    margin_band: str,
    movement_class: str,
    evidence: dict,
) -> tuple[str, str, str]:
    """
    Return (recommended_mechanic, margin_opportunity, rationale) from the
    complete 3x3 matrix.

    margin_band    : "high" | "mid" | "low"
    movement_class : "fast" | "medium" | "slow"
    evidence       : campaign evidence summary from summarize_campaign_evidence
    """
    shipping_positive = evidence["shipping_roi_positive"]

    # ── high margin ────────────────────────────────────────────────────────
    if margin_band == "high":
        if movement_class == "fast":
            return (
                "Leave/minimal discount",
                "medium",
                "Product already sells well with healthy margin; "
                "avoid unnecessary discounting.",
            )
        if movement_class == "medium":
            return (
                "Volume/Bundle",
                "medium",
                "Product has margin room and moderate velocity; "
                "test a bundle or volume mechanic.",
            )
        # slow
        return (
            "BXGY",
            "high",
            "Product has margin room but low velocity; use BXGY/gift mechanic "
            "to increase movement without blanket discounting.",
        )

    # ── mid margin ─────────────────────────────────────────────────────────
    if margin_band == "mid":
        if movement_class == "fast":
            return (
                "Leave/minimal discount",
                "medium",
                "Product already moves quickly; avoid discounting unless "
                "campaign evidence supports it.",
            )
        if movement_class == "medium":
            return (
                "Leave/minimal discount",
                "low",
                "No specific rule matched; keep discounting minimal until "
                "stronger evidence exists. [no_specific_rule_matched]",
            )
        # slow
        return (
            "Volume/Bundle",
            "medium",
            "Product is slow with some margin room; bundle/volume mechanic "
            "may improve movement.",
        )

    # ── low margin ─────────────────────────────────────────────────────────
    # movement_class == "fast"
    if movement_class == "fast":
        mechanic = "Free Shipping" if shipping_positive else "Leave/minimal discount"
        return (
            mechanic,
            "medium",
            "Fast products with low margin should avoid deeper item discounts; "
            + ("shipping campaign has positive ROI." if shipping_positive
               else "no positive shipping evidence available."),
        )
    if movement_class == "medium":
        return (
            "Avoid deeper discounts",
            "medium",
            "Margin is already low; avoid discount mechanics that erode profit.",
        )
    # slow + low margin
    return (
        "Avoid deeper discounts",
        "low",
        "Product is slow and low-margin; deeper discounts are risky.",
    )


# ---------------------------------------------------------------------------
# Step B — campaign-evidence modifiers
# ---------------------------------------------------------------------------

# Fallback order when evidence vetoes the base mechanic (spec-mandated)
_FALLBACK_ORDER: list[str] = [
    "Volume/Bundle",
    "BXGY",
    "Free Shipping",
    "Avoid deeper discounts",
    "Leave/minimal discount",
]

# Which campaign types correspond to which mechanics
_MECHANIC_TO_CAMPAIGN_TYPE: dict[str, str] = {
    "Free Shipping":       "shipping",
    "BXGY":                "buy_one_get_one",
    "Volume/Bundle":       "volume",
    "Order-level Discount": "order",
    "Generic Discount":    "general",
}


def _mechanic_is_vetoed(mechanic: str, evidence: dict) -> bool:
    """
    Return True if campaign evidence vetoes the mechanic.

    Vetoed when:
    - The mechanic maps to a campaign type that has a flop verdict.
    - OR: mechanic involves deeper discounts AND volume_lift_margin_eroded is True.
    """
    ctype = _MECHANIC_TO_CAMPAIGN_TYPE.get(mechanic)
    if ctype and ctype in evidence["flop_types"]:
        return True
    if (
        evidence["volume_lift_margin_eroded"]
        and mechanic in ("Generic Discount", "Order-level Discount")
    ):
        return True
    return False


def _step_b(
    base_mechanic: str,
    margin_opportunity: str,
    base_confidence: str,
    base_rationale: str,
    evidence: dict,
    margin_band: str,
    is_free_gift_product: bool = False,
) -> tuple[str, str, str, str, list[str]]:
    """
    Apply campaign-evidence modifiers to the base Step A recommendation.

    Returns
    -------
    (final_mechanic, final_margin_opp, final_confidence_label,
     final_rationale, extra_evidence_tags)
    """
    extra_tags: list[str] = []
    final_mechanic = base_mechanic
    final_margin_opp = margin_opportunity
    final_confidence = base_confidence
    final_rationale = base_rationale

    # -- check if base mechanic is vetoed by campaign evidence ----------------
    if _mechanic_is_vetoed(final_mechanic, evidence):
        # Walk the fallback order and pick the first non-vetoed mechanic
        for fallback in _FALLBACK_ORDER:
            if fallback == "Free Shipping" and not evidence["shipping_roi_positive"]:
                continue
            if not _mechanic_is_vetoed(fallback, evidence):
                extra_tags.append(
                    f"base_mechanic_vetoed:{base_mechanic}->fallback:{fallback}"
                )
                final_mechanic = fallback
                final_confidence = "medium"   # evidence override → moderate confidence
                final_rationale += (
                    f" Campaign evidence vetoed '{base_mechanic}'; "
                    f"fallback to '{fallback}'."
                )
                break
        # If all fallbacks are vetoed (unlikely), terminal safe option
        else:
            final_mechanic = "Leave/minimal discount"
            extra_tags.append("all_fallbacks_vetoed:terminal_safe")
            final_confidence = "low"

    # -- success evidence raises confidence -----------------------------------
    ctype_for_final = _MECHANIC_TO_CAMPAIGN_TYPE.get(final_mechanic)
    if ctype_for_final and ctype_for_final in evidence["success_types"]:
        # Raise confidence one step
        if final_confidence == "medium":
            final_confidence = "high"
        elif final_confidence == "low":
            final_confidence = "medium"
        extra_tags.append(f"campaign_success_evidence:{ctype_for_final}")
        if ctype_for_final == "shipping":
            extra_tags.append("shipping_attribution_coarse")
            final_rationale += f" {_SHIPPING_CAVEAT}"

    # -- flop evidence lowers confidence (if not already vetoed/swapped) -----
    if ctype_for_final and ctype_for_final in evidence["flop_types"]:
        if final_confidence == "high":
            final_confidence = "medium"
        elif final_confidence == "medium":
            final_confidence = "low"
        extra_tags.append(f"campaign_flop_evidence:{ctype_for_final}")

    # -- volume_lift_margin_eroded: warn even if mechanic wasn't swapped ------
    if evidence["volume_lift_margin_eroded"] and final_mechanic in (
        "Generic Discount", "Order-level Discount"
    ):
        extra_tags.append("volume_lift_margin_eroded_risk")
        if final_confidence in ("high", "medium"):
            final_confidence = "medium"

    # -- shipping caveat if shipping evidence was used -----------------------
    if final_mechanic == "Free Shipping":
        extra_tags.append("shipping_attribution_coarse")

    return (
        final_mechanic,
        final_margin_opp,
        final_confidence,
        final_rationale,
        extra_tags,
    )


# ---------------------------------------------------------------------------
# Margin band computation
# ---------------------------------------------------------------------------

def _assign_margin_bands(classified_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute margin_band from the actual product margin_pct distribution.
    Uses the 33rd and 67th percentile of this shop's classified products.

    Bands:
        high  >= 67th percentile
        low   <= 33rd percentile
        mid   otherwise
    """
    df = classified_df.copy()

    # Compute percentiles from the data — never hardcoded thresholds
    p33 = df["margin_pct"].quantile(1 / 3)
    p67 = df["margin_pct"].quantile(2 / 3)

    def _band(x: float) -> str:
        if x >= p67:
            return "high"
        if x <= p33:
            return "low"
        return "mid"

    df["margin_band"] = df["margin_pct"].apply(_band)
    df["_margin_p33"] = round(p33, 4)
    df["_margin_p67"] = round(p67, 4)
    return df


# ---------------------------------------------------------------------------
# Build evidence_refs list
# ---------------------------------------------------------------------------

def _build_evidence_refs(
    row: "pd.Series",
    margin_band: str,
    final_mechanic: str,
    extra_tags: list[str],
    evidence: dict,
) -> list[str]:
    """Assemble the evidence_refs list for a Recommendation."""
    refs: list[str] = [
        f"product_id:{int(row['product_id'])}",
        f"movement_class:{row['movement_class']}",
        f"margin_band:{margin_band}",
        f"units_sold:{int(row['units_sold'])}",
        f"velocity_per_day:{float(row['velocity_per_day']):.4f}",
        f"margin_pct:{float(row['margin_pct']):.4f}",
        f"data_sufficient:{bool(row['data_sufficient'])}",
        f"recommended_mechanic:{final_mechanic}",
        _STRUCTURAL_CAVEAT,
    ]
    refs.extend(extra_tags)

    # Attach relevant campaign verdicts
    raw_df: pd.DataFrame = evidence.get("raw_df", pd.DataFrame())
    if not raw_df.empty:
        ctype_for_mechanic = _MECHANIC_TO_CAMPAIGN_TYPE.get(final_mechanic)
        for _, camp_row in raw_df.iterrows():
            if (
                ctype_for_mechanic and camp_row["type"] == ctype_for_mechanic
            ) or camp_row["verdict"] in ("success", "flop"):
                refs.append(
                    f"campaign:{camp_row['campaign_id']}|type:{camp_row['type']}"
                    f"|verdict:{camp_row['verdict']}|rc:{camp_row['reason_code']}"
                )

    return refs


# ---------------------------------------------------------------------------
# Main recommendation function
# ---------------------------------------------------------------------------

def recommend_for_products(
    classified_products_df: pd.DataFrame,
    campaign_eval_df: pd.DataFrame,
    affinity_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[Recommendation]]:
    """
    Generate exactly one Recommendation per product.

    Parameters
    ----------
    classified_products_df : pd.DataFrame
        Output of classify.classify_products().
    campaign_eval_df : pd.DataFrame
        Output of campaign_eval.evaluate_campaigns().
    affinity_df : pd.DataFrame | None
        Optional real affinity data.  If None, affinity rules are skipped.

    Returns
    -------
    recommendations_df : pd.DataFrame
    recommendations     : list[Recommendation]  (Pydantic-validated)
    """
    # Campaign evidence summary — built once, shared across all products
    evidence = summarize_campaign_evidence(campaign_eval_df)

    # Assign margin bands from this shop's data distribution
    df = _assign_margin_bands(classified_products_df)

    rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        pid = int(row["product_id"])
        title = str(row["title"])
        movement_class = str(row["movement_class"])
        margin_band = str(row["margin_band"])
        data_sufficient = bool(row["data_sufficient"])

        # ── Rule 0: insufficient data ───────────────────────────────────────
        if not data_sufficient or movement_class not in ("fast", "medium", "slow"):
            confidence_label = "low"
            confidence_float = CONFIDENCE_FLOAT["low"]
            mechanic = "Collect more data"
            margin_opp = "low"
            rationale = (
                f"Insufficient sample size (units_sold={int(row['units_sold'])}, "
                f"min required={MIN_UNITS}). "
                "Do not overclaim on product-level metrics."
            )
            risk_note = (
                "Insufficient product-level sample size; do not overclaim. "
                + _STRUCTURAL_CAVEAT
            )
            extra_tags = ["rule0_insufficient_data"]
            p_score = _priority_score(margin_opp, confidence_label, mechanic)

            rec_row = {
                "product_id": pid,
                "title": title,
                "margin_band": margin_band,
                "movement_class": movement_class,
                "data_sufficient": data_sufficient,
                "units_sold": int(row["units_sold"]),
                "velocity_per_day": float(row["velocity_per_day"]),
                "margin_pct": float(row["margin_pct"]),
                "recommended_mechanic": mechanic,
                "margin_opportunity": margin_opp,
                "confidence_label": confidence_label,
                "confidence": confidence_float,
                "rationale": rationale,
                "expected_effect": risk_note,
                "priority_score": p_score,
                "evidence_refs": [
                    f"product_id:{pid}",
                    f"units_sold:{int(row['units_sold'])}",
                    "rule0_insufficient_data",
                    _STRUCTURAL_CAVEAT,
                ],
            }
            rows.append(rec_row)
            continue

        # ── Step A: base mechanic from 3x3 matrix ──────────────────────────
        base_mechanic, margin_opp, base_rationale = _step_a(
            margin_band, movement_class, evidence
        )
        base_confidence = "medium"   # default; Step B will adjust

        # Bump up base confidence if product is high-value and clear signal
        if margin_band == "high" and movement_class in ("fast", "slow"):
            base_confidence = "high"
        elif margin_band == "low" and movement_class == "slow":
            base_confidence = "medium"

        # ── Affinity override (only if real data provided) ──────────────────
        affinity_applied = False
        if affinity_df is not None and not affinity_df.empty:
            # Only apply if this product appears in the affinity table
            if "product_id" in affinity_df.columns:
                aff_row = affinity_df[affinity_df["product_id"] == pid]
                if len(aff_row) > 0:
                    base_mechanic = "Bundle/Cross-sell"
                    margin_opp = "high"
                    base_confidence = "high"
                    base_rationale += " Strong affinity pair detected; bundle recommended."
                    affinity_applied = True

        # ── Step B: campaign-evidence modifiers ─────────────────────────────
        (
            final_mechanic,
            final_margin_opp,
            final_confidence_label,
            final_rationale,
            extra_tags,
        ) = _step_b(
            base_mechanic,
            margin_opp,
            base_confidence,
            base_rationale,
            evidence,
            margin_band,
        )

        if affinity_applied:
            extra_tags.append("affinity_override_applied")

        # ── Priority score on FINAL mechanic ────────────────────────────────
        p_score = _priority_score(final_margin_opp, final_confidence_label, final_mechanic)

        # ── Risk note → expected_effect ──────────────────────────────────────
        risk_parts: list[str] = [_STRUCTURAL_CAVEAT]
        if "shipping_attribution_coarse" in extra_tags:
            risk_parts.append(_SHIPPING_CAVEAT)
        if evidence["volume_lift_margin_eroded"]:
            risk_parts.append(
                "Volume-lift-but-margin-eroded pattern detected; "
                "monitor margin closely if running volume promotions."
            )
        expected_effect = " ".join(risk_parts)

        # ── Evidence refs ────────────────────────────────────────────────────
        evidence_refs = _build_evidence_refs(row, margin_band, final_mechanic, extra_tags, evidence)

        confidence_float = CONFIDENCE_FLOAT.get(final_confidence_label, 0.6)

        rec_row = {
            "product_id": pid,
            "title": title,
            "margin_band": margin_band,
            "movement_class": movement_class,
            "data_sufficient": data_sufficient,
            "units_sold": int(row["units_sold"]),
            "velocity_per_day": float(row["velocity_per_day"]),
            "margin_pct": float(row["margin_pct"]),
            "recommended_mechanic": final_mechanic,
            "margin_opportunity": final_margin_opp,
            "confidence_label": final_confidence_label,
            "confidence": confidence_float,
            "rationale": final_rationale,
            "expected_effect": expected_effect,
            "priority_score": p_score,
            "evidence_refs": evidence_refs,
        }
        rows.append(rec_row)

    recommendations_df = pd.DataFrame(rows)

    # ── Build typed Recommendation objects ──────────────────────────────────
    recommendations: list[Recommendation] = []
    for _, r in recommendations_df.iterrows():
        rec = Recommendation(
            product_id=int(r["product_id"]),
            title=str(r["title"]),
            recommended_mechanic=str(r["recommended_mechanic"]),
            rationale=str(r["rationale"]),
            evidence_refs=list(r["evidence_refs"]),
            confidence=float(r["confidence"]),
            priority_score=float(r["priority_score"]),
            expected_effect=str(r["expected_effect"]),
        )
        recommendations.append(rec)

    return recommendations_df, recommendations


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
    from discount_prime_agent.pipeline.classify import classify_products
    from discount_prime_agent.pipeline.campaign_eval import evaluate_campaigns

    DATA_PATH = "data/sample-data-mongo.json"
    OUT_DIR = "outputs"

    print("=" * 72)
    print("rules.py -- verification run")
    print("=" * 72)

    # ── ingest ────────────────────────────────────────────────────────────────
    orders_df, lineitems_df, campaigns_df, shop = build_clean_frames(DATA_PATH)

    # ── metrics ───────────────────────────────────────────────────────────────
    enriched = allocate_order_cost_to_lines(compute_line_revenue(lineitems_df))
    products_df = compute_product_metrics(enriched)

    # ── classify ──────────────────────────────────────────────────────────────
    classified_df = classify_products(products_df)

    # ── campaign evaluation ───────────────────────────────────────────────────
    campaign_eval_df, _ = evaluate_campaigns(orders_df, lineitems_df, campaigns_df)

    # ── recommendations (no affinity data in this dataset) ───────────────────
    recs_df, recs = recommend_for_products(
        classified_df,
        campaign_eval_df,
        affinity_df=None,
    )

    # ── print results ─────────────────────────────────────────────────────────
    display_cols = [
        "product_id", "title",
        "margin_band", "movement_class",
        "recommended_mechanic",
        "confidence_label", "priority_score",
    ]
    pd.set_option("display.max_rows", 30)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", "{:.4f}".format)

    print("\nRecommendations (sorted by priority_score descending):")
    print(
        recs_df[display_cols]
        .sort_values("priority_score", ascending=False)
        .to_string(index=False)
    )

    print(f"\nNumber of recommendations: {len(recs)}")

    print("\nCounts by mechanic:")
    for mechanic, cnt in recs_df["recommended_mechanic"].value_counts().items():
        print(f"  {mechanic:30s}: {cnt}")

    # ── NaN / inf check ───────────────────────────────────────────────────────
    nan_count = recs_df["priority_score"].isna().sum()
    inf_count = (recs_df["priority_score"].abs() == float("inf")).sum()
    print(f"\nNaN in priority_score: {nan_count}")
    print(f"Inf in priority_score: {inf_count}")
    assert nan_count == 0, "priority_score contains NaN!"
    assert inf_count == 0, "priority_score contains Inf!"

    # ── validate Pydantic objects ─────────────────────────────────────────────
    print(f"\nRecommendation objects validated: {len(recs)}")
    sample = recs[0]
    print(f"  Sample[0]: {sample.title!r}")
    print(f"    mechanic     : {sample.recommended_mechanic!r}")
    print(f"    confidence   : {sample.confidence}")
    print(f"    priority     : {sample.priority_score}")

    # ── save CSV ──────────────────────────────────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = f"{OUT_DIR}/recommendations.csv"

    # evidence_refs is a list — serialise to string for CSV
    import json as _json
    save_df = recs_df.copy()
    save_df["evidence_refs"] = save_df["evidence_refs"].apply(_json.dumps)
    save_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}  ({len(save_df)} rows)")

    print("\nRecommendations complete")
