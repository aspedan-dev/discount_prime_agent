"""
run.py
------
Shared async entry point for executing the agent orchestrator
programmatically (used by both `main.py --mode agents` and the MCP server,
so the Runner/session boilerplate lives in exactly one place).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from .orchestrator import root_agent

APP_NAME = "discount_prime_agent"
USER_ID = "cli"


async def run_agent_pipeline(
    data_path: str,
    min_units: int = 20,
    out_dir: str | None = None,
) -> dict[str, Any]:
    """
    Run Agent Orchestration (Analytics -> Strategy) end to end.

    Parameters
    ----------
    data_path : path to the shop's order/campaign JSON export
    min_units : minimum units_sold for a product to be "data sufficient"
    out_dir   : if given, also writes the merged result to
                <out_dir>/agent_strategy_output.json

    Returns
    -------
    dict with keys: analytics_summary, recommendations, campaign_eval, strategy
    """
    session_service = InMemorySessionService()
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        state={"data_path": data_path, "min_units": min_units},
    )

    runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)

    user_message = types.Content(
        role="user",
        parts=[types.Part(text="Run the full analytics and strategy pipeline.")],
    )

    async for _event in runner.run_async(
        user_id=USER_ID, session_id=session.id, new_message=user_message
    ):
        pass  # state is inspected after the run completes; no per-event handling needed

    final_session = await session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=session.id
    )
    state = final_session.state

    merged = {
        "analytics_summary": state.get("analytics_summary_text"),
        "recommendations": state.get("analytics_recommendations"),
        "campaign_eval": state.get("analytics_campaign_eval"),
        "strategy": state.get("strategy_output"),
    }

    if out_dir:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "agent_strategy_output.json").write_text(
            json.dumps(merged, indent=2, default=str), encoding="utf-8"
        )

    return merged
