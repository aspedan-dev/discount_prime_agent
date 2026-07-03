"""
schemas.py
----------
Pydantic v2 data contract for data/sample-data-mongo.json.

Design rules
------------
1. Every model sets  model_config = ConfigDict(extra="ignore")
   so that fields present in the raw JSON but not modelled here
   (including PII fields) are silently dropped instead of raising errors.
2. ONLY fields that physically appear in data/sample-data-mongo.json
   are modelled.  No invented fields.
3. PII fields are explicitly excluded:
     shop:   ownerName, ownerEmail, customerEmail
     orders: customer_email
4. Date fields typed as  datetime  (Pydantic parses ISO-8601 strings).
5. Money / percentage fields typed as  float.
6. Integer fields: orderCount, order_number, product_id, variant_id,
   quantity, shippingDiscountType.

Raw input models (match the file structure)
-------------------------------------------
   ShopMinimized     – shop object (PII dropped by extra="ignore")
   Campaign          – one entry in the campaigns array
   CampaignDiscount  – line-item.campaign_discounts + order.applied_campaigns
   OrderCampaign     – order.order_campaigns
   LineItem          – one entry in order.line_items
   Order             – one entry in the orders array  (customer_email dropped)

Derived output models (shapes only; values computed in metrics/classify)
------------------------------------------------------------------------
   ProductProfile    – per-product analytics summary
   CampaignVerdict   – per-campaign effectiveness verdict
   Recommendation    – per-product action recommendation
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Raw input models
# ---------------------------------------------------------------------------

class ShopMinimized(BaseModel):
    """
    Minimal shop record.

    PII fields (ownerName, ownerEmail, customerEmail) are present in the raw
    JSON but are excluded here by design; extra="ignore" silently drops them.
    Only the six analytical fields we actually need are modelled.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    shopifyDomain: str
    country: str
    currency: str
    planName: str
    createdAt: datetime


class Campaign(BaseModel):
    """
    One campaign record from the campaigns array.

    endAt is Optional because active campaigns have  endAt: null.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    shopId: str
    name: str
    type: str
    status: str
    startAt: datetime
    endAt: Optional[datetime] = None     # null when campaign is still running
    ranAt: datetime
    isDeleted: bool
    totalRevenue: float
    orderCount: int
    marginPct: float
    createdAt: datetime


class CampaignDiscount(BaseModel):
    """
    Discount attribution record.

    Appears in:
      - order.line_items[].campaign_discounts
      - order.applied_campaigns
    """

    model_config = ConfigDict(extra="ignore")

    campaignId: str
    campaignName: str
    campaignType: str
    discountAmount: float


class OrderCampaign(BaseModel):
    """
    Campaign touch-point at the order level.

    Appears in order.order_campaigns.

    shippingDiscountType and shippingDiscountValue are present only for
    shipping-type campaigns; non-shipping campaigns omit them entirely.
    """

    model_config = ConfigDict(extra="ignore")

    campaignId: str
    campaignName: str
    campaignType: str
    discountAmount: float
    shippingDiscountType: Optional[int] = None
    shippingDiscountValue: Optional[float] = None


class LineItem(BaseModel):
    """
    One line item within an order.

    active_campaign_id is absent (not null — the key does not exist) on line
    items that have no active campaign.  Optional[str] = None covers both.

    Note from data/meta.note: per-line price, compare_at_price, product_id,
    and variant_id are synthesized deterministically; treat them as
    structural rather than ground-truth pricing.
    """

    model_config = ConfigDict(extra="ignore")

    product_id: int       # large Shopify int, fits Python int
    variant_id: int       # large Shopify int
    title: str
    price: float
    quantity: int
    total_discount: float
    compare_at_price: float
    active_campaign_id: Optional[str] = None          # absent → None
    active_campaign_ids: list[str] = Field(default_factory=list)
    campaign_discounts: list[CampaignDiscount] = Field(default_factory=list)


class Order(BaseModel):
    """
    One order record.

    customer_email is present in the raw JSON but is EXCLUDED by design.
    extra="ignore" ensures it is silently dropped during parsing.
    """

    model_config = ConfigDict(extra="ignore")

    id: int
    shop_id: str
    order_number: int
    currency: str
    subtotal_price: float
    total_discounts: float
    total_price: float
    line_items: list[LineItem] = Field(default_factory=list)
    has_active_campaign: bool
    is_free_shipping: bool
    shipping_original_price: float
    shipping_price_charged: float
    shipping_discount_amount: float
    applied_campaigns: list[CampaignDiscount] = Field(default_factory=list)
    order_campaigns: list[OrderCampaign] = Field(default_factory=list)
    cost_total: float
    revenue_with_cost: float
    has_partial_cost: bool
    createdAt: datetime


# ---------------------------------------------------------------------------
# Derived output models
# ---------------------------------------------------------------------------

class ProductProfile(BaseModel):
    """
    Per-product analytics summary computed by metrics.py.

    Notes
    -----
    - gross_margin_alloc: gross margin dollars allocated to this product.
    - velocity_per_day:   units sold divided by the number of days in window.
    - movement_class:     one of "fast" / "medium" / "slow" — computed from
                          velocity relative to the product population.
    - data_sufficient:    False when the observation window is too short or
                          order count too low to trust the metrics.
    - evidence:           raw supporting numbers used to derive this profile
                          (open dict so callers can attach whatever they need).
    """

    model_config = ConfigDict(extra="ignore")

    product_id: int
    title: str
    units_sold: int
    revenue: float
    gross_margin_alloc: float
    margin_pct: float
    velocity_per_day: float
    movement_class: Literal["fast", "medium", "slow"]
    data_sufficient: bool
    evidence: dict[str, Any] = Field(default_factory=dict)


class CampaignVerdict(BaseModel):
    """
    Effectiveness verdict for one campaign, produced by campaign_eval.py.

    Notes
    -----
    - window_days:              length of the comparison window in days.
    - attributed_orders:        orders where this campaign appears in
                                applied_campaigns or order_campaigns.
    - units_lift_ratio:         (campaign units/day) / (baseline units/day).
                                > 1.0 means uplift; < 1.0 means suppression.
    - margin_per_day_baseline:  average daily margin before/outside the
                                campaign window (from order cost_total).
    - margin_per_day_campaign:  average daily margin during the campaign.
    - shipping_cost_impact:     net shipping discount dollars given away;
                                None for non-shipping campaigns.
    - verdict:                  "success" | "flop" | "inconclusive"
    - confidence:               0–1 float; low when few orders observed.
    - reason_code:              short machine-readable tag explaining verdict
                                (e.g. "low_n", "margin_lift", "margin_erosion").
    """

    model_config = ConfigDict(extra="ignore")

    campaign_id: str
    type: str
    window_days: int
    attributed_orders: int
    units_lift_ratio: float
    margin_per_day_baseline: float
    margin_per_day_campaign: float
    shipping_cost_impact: Optional[float] = None
    verdict: Literal["success", "flop", "inconclusive"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason_code: str


class Recommendation(BaseModel):
    """
    A single actionable recommendation for a product, produced by classify.py.

    Notes
    -----
    - recommended_mechanic: campaign type to apply (e.g. "volume", "shipping",
                            "buy_one_get_one", "general", "order").
    - rationale:            human-readable explanation.
    - evidence_refs:        list of supporting metric keys (e.g. column names
                            or campaign IDs) that drove the recommendation.
    - confidence:           0–1 float; propagated from ProductProfile /
                            CampaignVerdict confidence.
    - priority_score:       higher = act sooner; dimensionless float used for
                            sorting; computed from velocity and margin.
    - expected_effect:      short description of the anticipated outcome.
    """

    model_config = ConfigDict(extra="ignore")

    product_id: int
    title: str
    recommended_mechanic: str
    rationale: str
    evidence_refs: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    priority_score: float
    expected_effect: str


# ---------------------------------------------------------------------------
# Self-test (__main__ block)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    from pathlib import Path

    DATA_PATH = Path("data/sample-data-mongo.json")

    raw = json.loads(DATA_PATH.read_text(encoding="utf-8"))

    # ── Parse shop ──────────────────────────────────────────────────────────
    shop = ShopMinimized.model_validate(raw["shop"])
    print("=== ShopMinimized ===")
    print(f"  id            : {shop.id}")
    print(f"  shopifyDomain : {shop.shopifyDomain}")
    print(f"  country       : {shop.country}")
    print(f"  currency      : {shop.currency}")
    print(f"  planName      : {shop.planName}")
    print(f"  createdAt     : {shop.createdAt}")

    # ── Parse first 3 campaigns ─────────────────────────────────────────────
    print("\n=== Campaigns (first 3) ===")
    campaigns_raw = raw["campaigns"][:3]
    parsed_campaigns = [Campaign.model_validate(c) for c in campaigns_raw]
    for c in parsed_campaigns:
        end_str = str(c.endAt) if c.endAt else "None (still running)"
        print(
            f"  [{c.id}] {c.name!r:40s} "
            f"type={c.type:20s} status={c.status:8s} "
            f"revenue=${c.totalRevenue:,.0f}  orders={c.orderCount:4d}  "
            f"margin={c.marginPct:.0f}%  endAt={end_str}"
        )

    # ── Parse first 3 orders ────────────────────────────────────────────────
    print("\n=== Orders (first 3) ===")
    orders_raw = raw["orders"][:3]
    parsed_orders = [Order.model_validate(o) for o in orders_raw]
    for o in parsed_orders:
        print(
            f"  order #{o.order_number}  "
            f"subtotal=${o.subtotal_price:.2f}  "
            f"discounts=${o.total_discounts:.2f}  "
            f"total=${o.total_price:.2f}  "
            f"cost=${o.cost_total:.2f}  "
            f"items={len(o.line_items)}  "
            f"campaigns_applied={len(o.applied_campaigns)}  "
            f"order_campaigns={len(o.order_campaigns)}  "
            f"free_ship={o.is_free_shipping}"
        )

    # ── PII guard assertions ─────────────────────────────────────────────────
    print("\n=== PII checks ===")

    assert "customer_email" not in Order.model_fields, (
        "FAIL: customer_email must NOT be a modelled field on Order"
    )

    shop_pii = {"ownerEmail", "ownerName", "customerEmail"}
    leaked = shop_pii & set(ShopMinimized.model_fields)
    assert not leaked, (
        f"FAIL: PII fields leaked into ShopMinimized: {leaked}"
    )

    # Also verify the parsed shop object carries no PII as attributes
    for pii_attr in shop_pii:
        assert not hasattr(shop, pii_attr), (
            f"FAIL: parsed shop object has PII attribute '{pii_attr}'"
        )

    print("  PII check passed")
    print("\nAll schemas parsed successfully.")
