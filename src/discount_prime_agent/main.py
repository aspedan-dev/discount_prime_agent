"""
main.py
-------
CLI entry point with two modes.

    pipeline (default) — the deterministic Phase 1 pandas pipeline only.
        No API key required.
        python -m discount_prime_agent.main --mode pipeline

    agents — runs Agent Orchestration (Agent Analytics -> Agent Strategy)
        via Google ADK + Gemini. Requires GOOGLE_API_KEY.
        python -m discount_prime_agent.main --mode agents

Usage
-----
    python -m discount_prime_agent.main
    python -m discount_prime_agent.main --mode agents --data data/sample-data-mongo.json --out outputs/
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import pandas as pd

from discount_prime_agent.pipeline import (
    build_clean_frames,
    classify_products,
    evaluate_campaigns,
    recommend_for_products,
    run_product_metrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(df: pd.DataFrame, out_dir: Path, filename: str) -> None:
    """Save a DataFrame to CSV in *out_dir* and print a confirmation line."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    df.to_csv(path, index=False)
    print(f"  -> {path}  ({len(df)} rows, {len(df.columns)} cols)")


def _section(title: str) -> None:
    width = 60
    print(f"\n{'-' * width}")
    print(f"  {title}")
    print(f"{'-' * width}")


# ---------------------------------------------------------------------------
# Mode: pipeline (deterministic, no LLM)
# ---------------------------------------------------------------------------

def _run_pipeline_mode(data_path: str, out_dir: str, min_units: int) -> None:
    """Execute the Phase 1 deterministic analytics pipeline end to end."""

    _section("STEP 1 - Ingest & validate")
    orders_df, lineitems_df, campaigns_df, shop = build_clean_frames(data_path)
    print(f"  orders={len(orders_df)}  lineitems={len(lineitems_df)}  campaigns={len(campaigns_df)}")

    _section("STEP 2 - Product metrics")
    products_df, _profiles = run_product_metrics(lineitems_df)
    print(f"  products={len(products_df)}")

    _section("STEP 3 - Classify products")
    classified_df = classify_products(products_df, min_units=min_units)
    print(classified_df["movement_class"].value_counts().to_string())

    _section("STEP 4 - Evaluate campaigns")
    campaign_eval_df, _verdicts = evaluate_campaigns(orders_df, lineitems_df, campaigns_df)
    print(campaign_eval_df["verdict"].value_counts().to_string())

    _section("STEP 5 - Recommendations")
    recs_df, _recs = recommend_for_products(classified_df, campaign_eval_df, affinity_df=None)
    print(f"  recommendations={len(recs_df)}")

    _section("STEP 6 - Write outputs")
    out = Path(out_dir)

    def _serialise_list_cols(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in df.columns:
            sample = df[col].dropna()
            if sample.empty:
                continue
            if isinstance(sample.iloc[0], (list, dict)):
                df[col] = df[col].apply(lambda v: json.dumps(v) if v is not None else None)
        return df

    _save(_serialise_list_cols(orders_df), out, "orders_clean.csv")
    _save(_serialise_list_cols(lineitems_df), out, "lineitems_clean.csv")
    _save(_serialise_list_cols(campaigns_df), out, "campaigns_clean.csv")
    _save(_serialise_list_cols(products_df), out, "product_metrics.csv")
    _save(_serialise_list_cols(classified_df), out, "product_classification.csv")
    _save(_serialise_list_cols(campaign_eval_df), out, "campaign_eval.csv")
    _save(_serialise_list_cols(recs_df), out, "recommendations.csv")

    _section("DONE")
    print(f"  All outputs written to: {out.resolve()}")


# ---------------------------------------------------------------------------
# Mode: agents (ADK + Gemini)
# ---------------------------------------------------------------------------

async def _run_agents_mode(data_path: str, out_dir: str, min_units: int) -> None:
    from dotenv import load_dotenv

    load_dotenv()

    from discount_prime_agent.agents import run_agent_pipeline

    _section("Running Agent Orchestration (Agent Analytics -> Agent Strategy)")
    result = await run_agent_pipeline(data_path=data_path, min_units=min_units, out_dir=out_dir)

    _section("DONE")
    n_proposals = len((result.get("strategy") or {}).get("proposals", []))
    print(f"  Strategy proposals: {n_proposals}")
    print(f"  Written to: {(Path(out_dir) / 'agent_strategy_output.json').resolve()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prime Growth Agent - deterministic pipeline and/or ADK agent orchestration"
    )
    parser.add_argument(
        "--mode",
        choices=["pipeline", "agents"],
        default="pipeline",
        help="'pipeline' = deterministic pandas only (default, no API key needed). "
             "'agents' = run the ADK Agent Analytics -> Agent Strategy orchestrator (needs GOOGLE_API_KEY).",
    )
    parser.add_argument(
        "--data",
        default="data/sample-data-mongo.json",
        help="Path to the shop's order/campaign JSON export (default: data/sample-data-mongo.json)",
    )
    parser.add_argument(
        "--out",
        default="outputs/",
        help="Directory to write outputs (default: outputs/)",
    )
    parser.add_argument(
        "--min-units",
        type=int,
        default=20,
        help="Minimum units_sold for a product to be classified as data-sufficient (default: 20)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.mode == "pipeline":
        _run_pipeline_mode(data_path=args.data, out_dir=args.out, min_units=args.min_units)
    else:
        asyncio.run(_run_agents_mode(data_path=args.data, out_dir=args.out, min_units=args.min_units))


if __name__ == "__main__":
    main()
