"""
discount_prime_agent
--------------------
Phase 1 + Phase 2: deterministic pandas analytics pipeline, plus a Google
ADK multi-agent layer on top of it.

Subpackages
-----------
    pipeline  – deterministic pandas/pydantic core (no AI):
                build_clean_frames, run_product_metrics, classify_products,
                evaluate_campaigns, recommend_for_products
    agents    – ADK agents:
                agents.analytics_agent (Agent Analytics, tool-calling)
                agents.strategy_agent  (Agent Strategy, structured LLM output)
                agents.orchestrator    (Agent Orchestration, SequentialAgent,
                                        exports root_agent)
                agents.run_agent_pipeline() – programmatic entry point
    mcp       – MCP server exposing Agent Orchestration as a callable tool

CLI
---
    python -m discount_prime_agent.main --mode pipeline   (no API key needed)
    python -m discount_prime_agent.main --mode agents      (needs GOOGLE_API_KEY)
"""
