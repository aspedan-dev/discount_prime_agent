"""
tests/test_ingest.py
--------------------
Unit tests for discount_prime_agent.pipeline.ingest.build_clean_frames(),
using a minimal fixture matching the REAL order/line-item/campaign grain
(schemas.Order / LineItem / Campaign), not a customer-grain shape.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from discount_prime_agent.pipeline.ingest import PII_COLUMN_FRAGMENTS, build_clean_frames

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_RAW: dict = {
    "shop": {
        "id": "shop_test_1",
        "shopifyDomain": "test-shop.myshopify.com",
        "country": "US",
        "currency": "USD",
        "planName": "basic",
        "createdAt": "2023-01-01T00:00:00Z",
        # PII fields present in the raw source, must be stripped:
        "ownerName": "Test Owner",
        "ownerEmail": "owner@example.com",
        "customerEmail": "customer@example.com",
    },
    "campaigns": [
        {
            "id": "c1",
            "shopId": "shop_test_1",
            "name": "Test Campaign",
            "type": "volume",
            "status": "active",
            "startAt": "2024-01-01T00:00:00Z",
            "endAt": None,
            "ranAt": "2024-01-01T00:00:00Z",
            "isDeleted": False,
            "totalRevenue": 1000.0,
            "orderCount": 1,
            "marginPct": 30.0,
            "createdAt": "2024-01-01T00:00:00Z",
        }
    ],
    "orders": [
        {
            "id": 1,
            "shop_id": "shop_test_1",
            "order_number": 1001,
            "currency": "USD",
            "subtotal_price": 50.0,
            "total_discounts": 0.0,
            "total_price": 50.0,
            "line_items": [
                {
                    "product_id": 111,
                    "variant_id": 222,
                    "title": "Widget",
                    "price": 50.0,
                    "quantity": 1,
                    "total_discount": 0.0,
                    "compare_at_price": 60.0,
                    "active_campaign_ids": [],
                    "campaign_discounts": [],
                }
            ],
            "has_active_campaign": False,
            "is_free_shipping": False,
            "shipping_original_price": 5.0,
            "shipping_price_charged": 5.0,
            "shipping_discount_amount": 0.0,
            "applied_campaigns": [],
            "order_campaigns": [],
            "cost_total": 20.0,
            "revenue_with_cost": 30.0,
            "has_partial_cost": False,
            "createdAt": "2024-01-05T00:00:00Z",
            # PII field present in the raw source, must be stripped:
            "customer_email": "buyer@example.com",
        }
    ],
}


def _write_temp_json(doc: dict, tmp_path: Path) -> Path:
    p = tmp_path / "test_data.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_build_clean_frames_returns_three_dataframes_and_shop_dict(tmp_path):
    path = _write_temp_json(MINIMAL_RAW, tmp_path)
    orders_df, lineitems_df, campaigns_df, shop = build_clean_frames(path)
    assert isinstance(orders_df, pd.DataFrame)
    assert isinstance(lineitems_df, pd.DataFrame)
    assert isinstance(campaigns_df, pd.DataFrame)
    assert isinstance(shop, dict)
    assert len(orders_df) == 1
    assert len(lineitems_df) == 1
    assert len(campaigns_df) == 1


def test_no_pii_in_orders_or_lineitems(tmp_path):
    path = _write_temp_json(MINIMAL_RAW, tmp_path)
    orders_df, lineitems_df, _campaigns_df, shop = build_clean_frames(path)

    cols = {c.lower() for c in orders_df.columns} | {c.lower() for c in lineitems_df.columns}
    for fragment in PII_COLUMN_FRAGMENTS:
        assert not any(fragment in c for c in cols), f"PII fragment '{fragment}' leaked into a column name"

    for pii_key in ("ownerName", "ownerEmail", "customerEmail", "customer_email"):
        assert pii_key not in shop


def test_missing_required_top_level_key_raises(tmp_path):
    bad_doc = {k: v for k, v in MINIMAL_RAW.items() if k != "campaigns"}
    path = _write_temp_json(bad_doc, tmp_path)
    with pytest.raises(ValueError, match="campaigns"):
        build_clean_frames(path)


def test_file_not_found_raises():
    with pytest.raises(FileNotFoundError):
        build_clean_frames("nonexistent/path/data.json")


def test_line_item_fields_present(tmp_path):
    path = _write_temp_json(MINIMAL_RAW, tmp_path)
    _orders_df, lineitems_df, _campaigns_df, _shop = build_clean_frames(path)
    row = lineitems_df.iloc[0]
    assert row["product_id"] == 111
    assert row["quantity"] == 1
    assert row["price"] == 50.0


def test_real_sample_data():
    """End-to-end test against the actual sample data file."""
    orders_df, lineitems_df, campaigns_df, shop = build_clean_frames("data/sample-data-mongo.json")
    assert len(orders_df) > 0
    assert len(lineitems_df) > 0
    assert len(campaigns_df) > 0
    assert "order_id" in orders_df.columns
    assert "product_id" in lineitems_df.columns
    assert isinstance(shop, dict)
