"""
ingest.py
---------
Load data/sample-data-mongo.json, strip all PII, validate the top-level
structure via schemas.py, and flatten the nested JSON into three clean
pandas DataFrames.

Public API
----------
    load_raw(path)           -> dict
    minimize(raw)            -> dict          (PII stripped)
    to_orders_df(raw)        -> pd.DataFrame  (one row per order)
    to_lineitems_df(raw)     -> pd.DataFrame  (one row per line item)
    to_campaigns_df(raw)     -> pd.DataFrame  (one row per campaign)
    assert_no_pii(*dfs)      -> None          (raises on any PII hit)
    build_clean_frames(path) -> tuple[DataFrame, DataFrame, DataFrame, dict]

Grain warnings
--------------
lineitems_df carries several ORDER-GRAIN values duplicated across every line
item that belongs to the same order:

    cost_total, revenue_with_cost, order_total_price, order_total_discounts,
    shipping_original_price, shipping_price_charged, shipping_discount_amount,
    is_free_shipping

DO NOT sum these columns directly across line-item rows.  For order-level
aggregation first use  .drop_duplicates("order_id").  For cost allocation use:

    alloc_cost_i = cost_total * line_revenue_i / sum(line_revenue_within_order)

Fields confirmed by reading data/sample-data-mongo.json
--------------------------------------------------------
Order keys (as returned by the API, before PII removal):
  id, shop_id, subtotal_price, total_discounts, total_tax, line_items,
  has_active_campaign, order_name, order_number, currency, total_price,
  original_compare_at_total, has_shipping_discount, is_free_shipping,
  shipping_original_price, shipping_price_charged, shipping_discount_amount,
  shipping_discount_percentage, applied_campaigns, order_campaigns,
  discount_applications, customer_email [PII - DROPPED], cost_total,
  revenue_with_cost, has_partial_cost, unaccounted_discount,
  createdAt, updatedAt

Line-item keys:
  product_id, variant_id, title, name [not modelled - extra="ignore"],
  price, quantity, total_discount, compare_at_price,
  active_campaign_id [absent on non-discounted items, not null],
  active_campaign_ids, campaign_discounts

Campaign keys:
  id, shopId, name, type, status, isImmediately, startAt, endAt [nullable],
  ranAt, isDeleted, totalRevenue, orderCount, marginPct, createdAt, updatedAt

Audit findings (confirmed via Python inspection of the real file):
  - 5 734 total line items across 2 950 orders
  - 4 111 line items have active_campaign_id; 1 623 do not (key absent)
  - 1 329 order_campaigns have shippingDiscountType; 49 do not
  - All 2 950 orders have at least one entry in applied_campaigns
"""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

from discount_prime_agent.schemas import (
    Campaign,
    Order,
    ShopMinimized,
)


# ---------------------------------------------------------------------------
# PII sentinel values
# ---------------------------------------------------------------------------

# Column-name fragments that must never appear in any clean DataFrame
_PII_COL_FRAGMENTS: tuple[str, ...] = (
    "customer_email",
    "owneremail",
    "ownername",
    "customeremail",
    "email",        # catches any accidental email column name
)

# Regex to detect email addresses inside string values
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")

# PII keys to strip from the shop object
_SHOP_PII_KEYS: tuple[str, ...] = ("ownerName", "ownerEmail", "customerEmail")

# PII key to strip from every order object
_ORDER_PII_KEY = "customer_email"

# Public alias — lets callers (e.g. tests) check the PII fragment list
# without importing a private name.
PII_COLUMN_FRAGMENTS: tuple[str, ...] = _PII_COL_FRAGMENTS


# ---------------------------------------------------------------------------
# load_raw
# ---------------------------------------------------------------------------

def load_raw(path: str | Path = "data/sample-data-mongo.json") -> dict:
    """
    Read and JSON-parse the data file.  Returns the top-level dict as-is.

    Raises
    ------
    FileNotFoundError  if the path does not exist.
    ValueError         if the top-level structure is not a dict.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path.resolve()}")

    raw: Any = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(raw, dict):
        raise ValueError(
            f"Expected top-level JSON object (dict), got {type(raw).__name__}."
        )
    for key in ("shop", "campaigns", "orders"):
        if key not in raw:
            raise ValueError(f"Missing required top-level key '{key}' in {path}.")

    return raw


# ---------------------------------------------------------------------------
# minimize
# ---------------------------------------------------------------------------

def minimize(raw: dict) -> dict:
    """
    Return a deep-copied version of *raw* with all PII fields removed.

    Strips
    ------
    shop:   ownerName, ownerEmail, customerEmail
    orders: customer_email (on every order document)

    Asserts that the stripped keys no longer exist after removal.
    """
    cleaned: dict = copy.deepcopy(raw)

    # ── shop PII ────────────────────────────────────────────────────────────
    shop = cleaned["shop"]
    for key in _SHOP_PII_KEYS:
        shop.pop(key, None)

    for key in _SHOP_PII_KEYS:
        assert key not in shop, (
            f"BUG: PII key '{key}' still present in shop after removal."
        )

    # ── order PII ───────────────────────────────────────────────────────────
    for i, order in enumerate(cleaned["orders"]):
        order.pop(_ORDER_PII_KEY, None)

    for i, order in enumerate(cleaned["orders"]):
        assert _ORDER_PII_KEY not in order, (
            f"BUG: '{_ORDER_PII_KEY}' still present on order at index {i}."
        )

    return cleaned


# ---------------------------------------------------------------------------
# to_orders_df
# ---------------------------------------------------------------------------

# Exact order-level keys to keep (derived from the real JSON field audit).
# customer_email is intentionally absent.
_ORDER_KEEP_COLS: tuple[str, ...] = (
    "id",
    "shop_id",
    "order_number",
    "currency",
    "subtotal_price",
    "total_discounts",
    "total_price",
    "has_active_campaign",
    "is_free_shipping",
    "shipping_original_price",
    "shipping_price_charged",
    "shipping_discount_amount",
    "applied_campaigns",
    "order_campaigns",
    "cost_total",
    "revenue_with_cost",
    "has_partial_cost",
    "createdAt",
)


def to_orders_df(raw: dict) -> pd.DataFrame:
    """
    Build a one-row-per-order DataFrame from the minimized raw dict.

    Each row is validated via the  Order  Pydantic schema before being
    added, so any schema mismatch raises loudly.

    applied_campaigns and order_campaigns remain as Python list columns
    (they are serialised to JSON strings only when writing CSVs).

    Parameters
    ----------
    raw : dict
        The minimized raw dict (customer_email already removed).

    Returns
    -------
    pd.DataFrame  with columns defined in _ORDER_KEEP_COLS, plus
                  'order_id' (alias of 'id' kept for explicit join key).
    """
    rows: list[dict] = []
    for order_doc in raw["orders"]:
        # Validate through Pydantic — raises ValidationError on mismatch
        validated: Order = Order.model_validate(order_doc)

        row: dict[str, Any] = {
            "order_id": validated.id,          # explicit join key
            "shop_id": validated.shop_id,
            "order_number": validated.order_number,
            "currency": validated.currency,
            "subtotal_price": validated.subtotal_price,
            "total_discounts": validated.total_discounts,
            "total_price": validated.total_price,
            "has_active_campaign": validated.has_active_campaign,
            "is_free_shipping": validated.is_free_shipping,
            "shipping_original_price": validated.shipping_original_price,
            "shipping_price_charged": validated.shipping_price_charged,
            "shipping_discount_amount": validated.shipping_discount_amount,
            # Keep as native lists; caller serialises when needed
            "applied_campaigns": [
                cd.model_dump() for cd in validated.applied_campaigns
            ],
            "order_campaigns": [
                oc.model_dump() for oc in validated.order_campaigns
            ],
            "cost_total": validated.cost_total,
            "revenue_with_cost": validated.revenue_with_cost,
            "has_partial_cost": validated.has_partial_cost,
            "createdAt": validated.createdAt,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# to_lineitems_df
# ---------------------------------------------------------------------------

def to_lineitems_df(raw: dict) -> pd.DataFrame:
    """
    Build a one-row-per-line-item DataFrame by exploding order.line_items.

    Parent order fields are carried into each row for joins / context.
    These ORDER-GRAIN values are DUPLICATED across line items of the same
    order — do NOT sum them across line-item rows:

        cost_total, revenue_with_cost, order_total_price,
        order_total_discounts, shipping_original_price,
        shipping_price_charged, shipping_discount_amount, is_free_shipping

    Use  order_id  +  drop_duplicates  for order-level aggregation.
    For cost allocation in metrics.py use proportional revenue sharing:

        alloc_cost_i = cost_total × line_revenue_i / Σ line_revenue (same order)

    active_campaign_id may be absent from a line-item document (key does not
    exist; it is NOT stored as null).  We normalise the absent case to None.
    """
    rows: list[dict] = []

    for order_doc in raw["orders"]:
        validated: Order = Order.model_validate(order_doc)

        # Parent order context fields — ORDER GRAIN (do not sum across rows)
        parent_ctx: dict[str, Any] = {
            "order_id": validated.id,
            "order_number": validated.order_number,
            "shop_id": validated.shop_id,
            "createdAt": validated.createdAt,
            "currency": validated.currency,
            "order_total_price": validated.total_price,
            "order_total_discounts": validated.total_discounts,
            "cost_total": validated.cost_total,            # ORDER GRAIN
            "revenue_with_cost": validated.revenue_with_cost,  # ORDER GRAIN
            "shipping_original_price": validated.shipping_original_price,  # ORDER GRAIN
            "shipping_price_charged": validated.shipping_price_charged,    # ORDER GRAIN
            "shipping_discount_amount": validated.shipping_discount_amount, # ORDER GRAIN
            "is_free_shipping": validated.is_free_shipping,                # ORDER GRAIN
        }

        for li in validated.line_items:
            row: dict[str, Any] = {
                **parent_ctx,
                # Line-item fields (per-line grain)
                "product_id": li.product_id,
                "variant_id": li.variant_id,
                "title": li.title,
                "price": li.price,
                "quantity": li.quantity,
                "total_discount": li.total_discount,
                "compare_at_price": li.compare_at_price,
                # active_campaign_id: absent key → None (normalised by Pydantic)
                "active_campaign_id": li.active_campaign_id,
                # Keep as native lists; caller serialises when needed
                "active_campaign_ids": li.active_campaign_ids,
                "campaign_discounts": [
                    cd.model_dump() for cd in li.campaign_discounts
                ],
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# to_campaigns_df
# ---------------------------------------------------------------------------

_CAMPAIGN_KEEP_COLS: tuple[str, ...] = (
    "id",
    "shopId",
    "name",
    "type",
    "status",
    "startAt",
    "endAt",       # nullable — None for still-running campaigns
    "ranAt",
    "totalRevenue",
    "orderCount",
    "marginPct",
    "isDeleted",
)


def to_campaigns_df(raw: dict) -> pd.DataFrame:
    """
    Build a one-row-per-campaign DataFrame.

    endAt is None for campaigns that are still running (endAt: null in JSON).
    """
    rows: list[dict] = []
    for camp_doc in raw["campaigns"]:
        validated: Campaign = Campaign.model_validate(camp_doc)

        row: dict[str, Any] = {
            "id": validated.id,
            "shopId": validated.shopId,
            "name": validated.name,
            "type": validated.type,
            "status": validated.status,
            "startAt": validated.startAt,
            "endAt": validated.endAt,          # None when still running
            "ranAt": validated.ranAt,
            "totalRevenue": validated.totalRevenue,
            "orderCount": validated.orderCount,
            "marginPct": validated.marginPct,
            "isDeleted": validated.isDeleted,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# assert_no_pii
# ---------------------------------------------------------------------------

def assert_no_pii(*dfs: pd.DataFrame) -> None:
    """
    Raise ValueError if any DataFrame contains PII.

    Checks
    ------
    1. Column names: any column whose lowercased name contains a fragment from
       _PII_COL_FRAGMENTS is flagged.
    2. String values: a random sample of up to 500 string cells is scanned for
       email-address patterns.  Full scan is impractical on 5 000+ rows but
       a sample provides a meaningful guard.

    Raises
    ------
    ValueError with a descriptive message on the first PII hit found.
    """
    for idx, df in enumerate(dfs):
        # ── column-name check ────────────────────────────────────────────────
        for col in df.columns:
            col_lower = col.lower()
            for fragment in _PII_COL_FRAGMENTS:
                if fragment in col_lower:
                    raise ValueError(
                        f"PII DETECTED in DataFrame #{idx}: "
                        f"column '{col}' matches PII fragment '{fragment}'."
                    )

        # ── value check (sampled) ────────────────────────────────────────────
        str_cols = [c for c in df.columns if df[c].dtype == object]
        for col in str_cols:
            sample = df[col].dropna().astype(str).sample(
                n=min(500, len(df)), random_state=42
            )
            for val in sample:
                # Skip JSON-serialised list/dict strings — they may contain
                # campaign names that include "@" in theory, but email regex
                # is tight enough to avoid false positives there.
                if _EMAIL_RE.search(val):
                    raise ValueError(
                        f"PII DETECTED in DataFrame #{idx}, column '{col}': "
                        f"value looks like an email address: '{val[:60]}…'"
                    )


# ---------------------------------------------------------------------------
# build_clean_frames
# ---------------------------------------------------------------------------

def build_clean_frames(
    path: str | Path = "data/sample-data-mongo.json",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Full ingest pipeline: load → strip PII → validate → flatten → audit.

    Returns
    -------
    orders_df       : pd.DataFrame  – one row per order, PII-free
    lineitems_df    : pd.DataFrame  – one row per line item, PII-free
    campaigns_df    : pd.DataFrame  – one row per campaign
    shop_minimized  : dict          – shop record with PII keys removed
    """
    # 1. Load
    raw = load_raw(path)

    # 2. Strip PII (deep copy; asserts keys removed)
    clean_raw = minimize(raw)

    # 3. Build DataFrames (each order validated through Pydantic)
    orders_df = to_orders_df(clean_raw)
    lineitems_df = to_lineitems_df(clean_raw)
    campaigns_df = to_campaigns_df(clean_raw)

    # 4. Parse date columns to timezone-aware datetime
    orders_df["createdAt"] = pd.to_datetime(orders_df["createdAt"], utc=True)
    lineitems_df["createdAt"] = pd.to_datetime(lineitems_df["createdAt"], utc=True)
    campaigns_df["startAt"] = pd.to_datetime(campaigns_df["startAt"], utc=True)
    campaigns_df["ranAt"] = pd.to_datetime(campaigns_df["ranAt"], utc=True)
    # endAt is nullable — pd.to_datetime handles None → NaT
    campaigns_df["endAt"] = pd.to_datetime(campaigns_df["endAt"], utc=True)

    # 5. PII audit — raises on any hit
    assert_no_pii(orders_df, lineitems_df, campaigns_df)

    # 6. Build shop_minimized dict (PII already removed by minimize())
    shop_minimized = clean_raw["shop"]

    # Double-check the dict itself carries no PII keys
    for pii_key in _SHOP_PII_KEYS + (_ORDER_PII_KEY,):
        assert pii_key not in shop_minimized, (
            f"BUG: PII key '{pii_key}' found in shop_minimized."
        )

    return orders_df, lineitems_df, campaigns_df, shop_minimized


# ---------------------------------------------------------------------------
# CSV serialisation helper
# ---------------------------------------------------------------------------

def _serialise_list_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of *df* where any column whose dtype is object and whose
    first non-null value is a list or dict is JSON-serialised to a string.

    This prevents csv write failures on list/dict columns.
    """
    df = df.copy()
    for col in df.columns:
        sample = df[col].dropna()
        if sample.empty:
            continue
        first = sample.iloc[0]
        if isinstance(first, (list, dict)):
            df[col] = df[col].apply(
                lambda v: json.dumps(v) if v is not None else None
            )
    return df


# ---------------------------------------------------------------------------
# __main__ — verification block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    DATA_PATH = "data/sample-data-mongo.json"
    OUT_DIR = "outputs"

    print("=" * 60)
    print("ingest.py — verification run")
    print("=" * 60)

    orders_df, lineitems_df, campaigns_df, shop_minimized = build_clean_frames(
        DATA_PATH
    )

    # ── shapes ──────────────────────────────────────────────────────────────
    print(f"\norders_df    shape : {orders_df.shape}")
    print(f"lineitems_df shape : {lineitems_df.shape}")
    print(f"campaigns_df shape : {campaigns_df.shape}")

    # ── shop ────────────────────────────────────────────────────────────────
    print(f"\nshop_minimized keys : {list(shop_minimized.keys())}")

    # ── PII column count (should be 0) ───────────────────────────────────────
    all_cols = (
        list(orders_df.columns)
        + list(lineitems_df.columns)
        + list(campaigns_df.columns)
    )
    pii_cols = [
        c for c in all_cols
        if any(f in c.lower() for f in _PII_COL_FRAGMENTS)
    ]
    print(f"\nPII columns: {len(pii_cols)}")
    if pii_cols:
        print(f"  !! FOUND: {pii_cols}", file=sys.stderr)

    # ── lineitems head ───────────────────────────────────────────────────────
    print("\nlineitems_df.head(3):")
    head_cols = [
        "order_id", "product_id", "title", "quantity", "price",
        "total_discount", "active_campaign_id", "cost_total",
    ]
    print(lineitems_df[head_cols].head(3).to_string(index=False))

    # ── campaign date sample (shows None handling) ───────────────────────────
    print("\ncampaigns_df[id, name, status, endAt]:")
    print(
        campaigns_df[["id", "name", "status", "endAt"]]
        .to_string(index=False)
    )

    # ── save CSVs ────────────────────────────────────────────────────────────
    os.makedirs(OUT_DIR, exist_ok=True)

    _serialise_list_cols(orders_df).to_csv(
        f"{OUT_DIR}/orders_clean.csv", index=False
    )
    _serialise_list_cols(lineitems_df).to_csv(
        f"{OUT_DIR}/lineitems_clean.csv", index=False
    )
    _serialise_list_cols(campaigns_df).to_csv(
        f"{OUT_DIR}/campaigns_clean.csv", index=False
    )

    print(f"\nCSVs written to {OUT_DIR}/")
    print("  orders_clean.csv")
    print("  lineitems_clean.csv")
    print("  campaigns_clean.csv")
    print("\n[OK] ingest.py verification complete.")
