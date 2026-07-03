"""
server.py — MCP server exposing Agent Orchestration as a tool.

Run locally (stdio transport):
    python -m discount_prime_agent.mcp.server

This lets any MCP-capable client (Claude Desktop, an MCP Inspector, or in
the future the Prime-Backend NestJS service acting as an MCP client) invoke
the full Analytics -> Strategy pipeline and get back the same JSON that
`main.py --mode agents` writes to outputs/agent_strategy_output.json.

Future direction (not built yet): swap transport="stdio" for an
SSE/streamable-HTTP transport once Prime-Backend needs to call this over
the network as a real endpoint instead of spawning a local subprocess.
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from mcp.server.fastmcp import FastMCP

from discount_prime_agent.agents import run_agent_pipeline

mcp_server = FastMCP("discount-prime-agent")


@mcp_server.tool()
async def get_campaign_recommendations(
    data_path: str = "data/sample-data-mongo.json",
    min_units: int = 20,
) -> dict:
    """Run the Agent Analytics + Agent Strategy pipeline and return prioritized campaign proposals as JSON."""
    return await run_agent_pipeline(data_path=data_path, min_units=min_units, out_dir="outputs")


if __name__ == "__main__":
    mcp_server.run(transport="stdio")
