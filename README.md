# Prime Growth Agent

> **Phase 1** — deterministic Python/pandas analytics pipeline (no AI)
> **Phase 2** — Google ADK multi-agent layer (Agent Analytics → Agent Strategy, orchestrated by Agent Orchestration) + MCP server

---

## Architecture

```
data/sample-data-mongo.json ("MCP order information" JSON)
            │
            ▼
┌─────────────────────────────────────────────────────────────┐
│ Agent Orchestration (SequentialAgent, root_agent)             │
│                                                                 │
│   ┌───────────────────┐        ┌──────────────────────────┐  │
│   │ Agent Analytics     │  ──▶  │ Agent Strategy             │  │
│   │ (LlmAgent + tools)  │       │ (LlmAgent, output_schema)  │  │
│   │                      │       │                            │  │
│   │ tools = pipeline/*   │       │ reads session.state        │  │
│   │  - ingest            │       │ written by Agent Analytics │  │
│   │  - product metrics   │       │ produces prioritized       │  │
│   │  - classify           │       │ campaign proposals          │  │
│   │  - campaign eval      │       │ (StrategyOutput schema)     │  │
│   │  - recommend          │       └──────────────────────────┘  │
│   └───────────────────┘                                          │
└─────────────────────────────────────────────────────────────┘
            │
            ▼
outputs/agent_strategy_output.json     +      MCP tool: get_campaign_recommendations
```

The deterministic pandas/pydantic core (`pipeline/`) never changed behavior —
Agent Analytics only *calls* it as ADK tools. All numbers in the final output
trace back to that deterministic code, never to LLM-generated text. Agent
Strategy is the only LLM-reasoning-only step (no tools), and it is
schema-constrained (`StrategyOutput`) so its output is always valid JSON.

---

## Repository layout

```
discount_prime_agent/
├── data/
│   └── sample-data-mongo.json     ← shop/orders/campaigns export (PII present, stripped on ingest)
├── src/discount_prime_agent/
│   ├── schemas.py                 ← Pydantic v2 data contract (raw + derived models)
│   ├── pipeline/                  ← Phase 1: deterministic pandas core, no AI
│   │   ├── ingest.py              ← load → strip PII → validate → flatten
│   │   ├── metrics.py             ← per-product revenue/margin/velocity
│   │   ├── classify.py            ← fast/medium/slow movement tertiles
│   │   ├── campaign_eval.py       ← campaign profit verdicts vs. baseline
│   │   └── rules.py               ← per-product discount-mechanic recommendations
│   ├── agents/                    ← Phase 2: ADK multi-agent layer
│   │   ├── analytics_agent/       ← Agent Analytics (LlmAgent + 5 FunctionTools)
│   │   ├── strategy_agent/        ← Agent Strategy (LlmAgent, output_schema)
│   │   ├── orchestrator.py        ← Agent Orchestration (SequentialAgent, root_agent)
│   │   └── run.py                 ← run_agent_pipeline(): shared Runner/session helper
│   ├── mcp/
│   │   └── server.py              ← MCP server exposing get_campaign_recommendations
│   └── main.py                    ← CLI: --mode pipeline | --mode agents
├── outputs/                       ← CSV + JSON outputs written at runtime
├── tests/
├── .env.example
├── requirements.txt
└── README.md
```

---

## Data contract

Input JSON (`data/sample-data-mongo.json`) top-level shape:

```json
{ "meta": {...}, "shop": {...}, "campaigns": [...], "orders": [...] }
```

- **shop**: PII fields `ownerName`, `ownerEmail`, `customerEmail` are stripped on ingest.
- **orders**: one order per entry; `customer_email` (PII) is stripped; each order has `line_items[]`, `applied_campaigns[]`, `order_campaigns[]`.
- **campaigns**: shipping / volume / buy_one_get_one / order / general types; `endAt` is `null` while a campaign is still running.

Full field-level contract is in [`schemas.py`](src/discount_prime_agent/schemas.py) (Pydantic v2, `extra="ignore"` so PII/unknown fields are silently dropped, never raised on).

Grain warning (documented in `pipeline/ingest.py`/`metrics.py`/`campaign_eval.py`): several columns on the line-items DataFrame are **order-grain** (duplicated across every line item of the same order) — `cost_total`, `shipping_*`, `is_free_shipping`, etc. They are never summed directly across line-item rows; costs are allocated by revenue share.

---

## Quick start

```bash
# 1. Install dependencies (editable install so `discount_prime_agent` resolves from src/)
pip install -r requirements.txt
pip install -e .

# 2a. Deterministic pipeline only (no API key needed)
python -m discount_prime_agent.main --mode pipeline

# 2b. Full agent pipeline (Agent Analytics -> Agent Strategy)
cp .env.example .env        # then fill in GOOGLE_API_KEY
python -m discount_prime_agent.main --mode agents
```

### `--mode pipeline` outputs (`outputs/`)

| File                          | Description                              |
|--------------------------------|-------------------------------------------|
| `orders_clean.csv`             | PII-stripped orders                       |
| `lineitems_clean.csv`          | PII-stripped line items                   |
| `campaigns_clean.csv`          | Campaign records                          |
| `product_metrics.csv`          | Per-product revenue/margin/velocity       |
| `product_classification.csv`   | + movement_class (fast/medium/slow)       |
| `campaign_eval.csv`            | Campaign verdicts (success/flop/inconclusive) |
| `recommendations.csv`          | One deterministic recommendation/product  |

### `--mode agents` output

| File                              | Description                                      |
|-------------------------------------|---------------------------------------------------|
| `outputs/agent_strategy_output.json` | `{analytics_summary, recommendations, campaign_eval, strategy}` — `strategy` is the `StrategyOutput`-shaped, prioritized campaign proposal list |

---

## Running the agents with the ADK dev UI

```bash
adk web src/discount_prime_agent/agents
```

Opens a browser UI where you can chat with `root_agent`, inspect each tool
call Agent Analytics makes (in strict order), and see the session state
handed off to Agent Strategy.

---

## MCP server

```bash
python -m discount_prime_agent.mcp.server
```

Starts a stdio MCP server exposing one tool:

```python
get_campaign_recommendations(data_path: str = "data/sample-data-mongo.json", min_units: int = 20) -> dict
```

which runs the same Agent Orchestration pipeline and returns the merged JSON
(also writing `outputs/agent_strategy_output.json`). Inspect it manually with:

```bash
mcp dev src/discount_prime_agent/mcp/server.py
```

**Future direction (not built yet):** swap the stdio transport for
SSE/streamable-HTTP so the Prime-Backend NestJS service can call this over
the network as a real endpoint instead of spawning a local subprocess. There
is currently no existing MCP code in Prime-Backend — this would be the first.

---

## Environment variables (`.env`)

See [`.env.example`](.env.example):

| Variable                    | Purpose                                                     |
|-------------------------------|---------------------------------------------------------------|
| `GOOGLE_API_KEY`             | Google AI Studio key (Gemini Developer API, not Vertex AI)   |
| `GOOGLE_GENAI_USE_VERTEXAI`  | Must be `FALSE` to force the AI Studio code path              |
| `DPA_ANALYTICS_MODEL`        | Model for Agent Analytics (default `gemini-2.5-flash`)        |
| `DPA_STRATEGY_MODEL`         | Model for Agent Strategy (default `gemini-2.5-pro`)           |
| `DPA_DATA_PATH`, `DPA_OUT_DIR`, `DPA_MIN_UNITS` | CLI default overrides                    |

---

## Running tests

```bash
pytest tests/ -v
pytest tests/ -v --cov=discount_prime_agent
```

`test_ingest.py` and `test_analytics_tools.py`/`test_strategy_schema.py` run
without any API key (they call pipeline functions and tool functions
directly, or validate Pydantic schemas — no LLM involved).

---

## Phase roadmap

| Phase | Description                                          | Status      |
|-------|--------------------------------------------------------|-------------|
| 1     | Deterministic Python/pandas analytics                 | Done        |
| 2     | ADK agents (Analytics/Strategy/Orchestration) + MCP    | Current     |
| 3     | MCP over SSE/HTTP; Prime-Backend integration           | Planned     |
| 4     | Production deployment & monitoring                    | Planned     |

---

## Business rules reference

Thresholds live in [`pipeline/classify.py`](src/discount_prime_agent/pipeline/classify.py) (`MIN_UNITS`) and [`pipeline/campaign_eval.py`](src/discount_prime_agent/pipeline/campaign_eval.py) (`MIN_CAMPAIGN_DAYS`, `MIN_CAMPAIGN_UNITS`, `PROFIT_LIFT_THRESHOLD`, `UNITS_LIFT_THRESHOLD`, etc.) — all data-driven (percentile-based margin bands, tertile-based velocity classes), no hardcoded product/campaign names.
