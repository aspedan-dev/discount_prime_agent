"""
orchestrator.py — Agent Orchestration
---------------------------------------
Root SequentialAgent: runs Agent Analytics, then Agent Strategy, in strict
order. `root_agent` is the name ADK's CLI tooling (`adk web`, `adk run`)
looks for when pointed at this package.
"""

from __future__ import annotations

from google.adk.agents import SequentialAgent

from .analytics_agent import analytics_agent
from .strategy_agent import strategy_agent

root_agent = SequentialAgent(
    name="agent_orchestration",
    description=(
        "Runs Agent Analytics (deterministic pipeline via tools) followed by "
        "Agent Strategy (LLM synthesis into prioritized campaign proposals)."
    ),
    sub_agents=[analytics_agent, strategy_agent],
)
