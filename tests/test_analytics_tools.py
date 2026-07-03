"""
tests/test_analytics_tools.py
------------------------------
Unit tests for the Agent Analytics FunctionTools, called directly against
a fake ToolContext (no LLM / no ADK Runner involved) so these run fast and
without a GOOGLE_API_KEY.
"""

import json

import pytest

from discount_prime_agent.agents.analytics_agent.tools import (
    classify_products_tool,
    compute_product_metrics_tool,
    evaluate_campaigns_tool,
    ingest_data_tool,
    recommend_products_tool,
)


class _FakeToolContext:
    """Minimal stand-in for google.adk.tools.tool_context.ToolContext — only .state is used."""

    def __init__(self):
        self.state: dict = {}


@pytest.fixture()
def ctx():
    return _FakeToolContext()


DATA_PATH = "data/sample-data-mongo.json"


def test_ingest_data_tool_populates_state(ctx):
    result = ingest_data_tool(DATA_PATH, ctx)
    assert result["status"] == "ok"
    assert result["orders"] > 0
    assert result["lineitems"] > 0
    assert result["campaigns"] > 0
    assert ctx.state["analytics_orders"]
    assert ctx.state["analytics_lineitems"]
    assert ctx.state["analytics_campaigns"]
    assert isinstance(ctx.state["analytics_shop"], dict)
    # every value written to state must be JSON-serializable
    json.dumps(ctx.state["analytics_orders"])
    json.dumps(ctx.state["analytics_lineitems"])
    json.dumps(ctx.state["analytics_campaigns"])


def test_full_tool_chain_in_order(ctx):
    ingest_result = ingest_data_tool(DATA_PATH, ctx)
    assert ingest_result["status"] == "ok"

    metrics_result = compute_product_metrics_tool(ctx)
    assert metrics_result["status"] == "ok"
    assert metrics_result["products"] > 0
    json.dumps(ctx.state["analytics_products"])

    classify_result = classify_products_tool(20, ctx)
    assert classify_result["status"] == "ok"
    assert isinstance(classify_result["class_counts"], dict)
    json.dumps(ctx.state["analytics_classified_products"])

    eval_result = evaluate_campaigns_tool(ctx)
    assert eval_result["status"] == "ok"
    assert isinstance(eval_result["verdict_counts"], dict)
    json.dumps(ctx.state["analytics_campaign_eval"])

    recommend_result = recommend_products_tool(ctx)
    assert recommend_result["status"] == "ok"
    assert recommend_result["recommendations"] > 0
    json.dumps(ctx.state["analytics_recommendations"])


def test_compute_product_metrics_tool_without_ingest_errors(ctx):
    result = compute_product_metrics_tool(ctx)
    assert result["status"] == "error"


def test_classify_products_tool_without_metrics_errors(ctx):
    result = classify_products_tool(20, ctx)
    assert result["status"] == "error"


def test_recommend_products_tool_without_prereqs_errors(ctx):
    result = recommend_products_tool(ctx)
    assert result["status"] == "error"
