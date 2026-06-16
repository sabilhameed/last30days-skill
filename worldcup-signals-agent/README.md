# World Cup Signals Agent

An autonomous agent that uses the [`last30days`](../skills/last30days) skill to
collect the latest signals about the **ongoing World Cup** and distill them into
structured, prediction-relevant intelligence stored in **DuckDB**.

The objective: surface any news or chatter — injuries, suspensions, lineup
moves, form/momentum, tactical match-ups, betting/prediction-market shifts —
that gives an **edge in forecasting upcoming match outcomes**.

## How it works

```
                 ┌─────────────────────────────────────────────┐
   research      │  for each angle:                            │
   angles  ─────▶│   1. run last30days engine (--emit=json)    │
 (config.py)     │   2. flatten ranked evidence into items     │
                 │   3. extract signals (Claude or heuristic)  │
                 │   4. persist run + items + clusters + signals│
                 └───────────────────────┬─────────────────────┘
                                         ▼
                                   DuckDB database
                       runs · raw_items · clusters · signals
                              + match_edges (view)
```

- **Collection** — `engine.py` shells out to the bundled `last30days` engine in
  `--emit=json --agent` mode. That returns the full structured research report
  (ranked candidates, clusters, per-source items) which we parse directly.
- **Extraction** — `extract.py` turns the raw chatter into discrete
  `PredictiveSignal` records. The primary path uses Claude
  (`claude-opus-4-8`) with **structured outputs** (`messages.parse`) for honest
  edge/confidence ratings; with no API key it falls back to a deterministic
  keyword heuristic so the pipeline still produces structured output offline.
- **Storage** — `store.py` writes everything to DuckDB with a stable schema and
  exposes a `match_edges` view that rolls signals up into a per-match edge board.

## Research angles

Defined in `worldcup_agent/config.py` (`DEFAULT_ANGLES`). Each becomes one engine
run with football-specific subreddit / X-handle targeting:

| Angle | What it hunts for |
|---|---|
| `injuries_and_availability` | injuries, fitness doubts, suspensions, returns |
| `starting_lineups` | predicted XI, rotation, rested players |
| `team_form_and_momentum` | form, momentum, morale, sentiment |
| `tactics_and_matchups` | tactical edges, manager decisions |
| `betting_and_prediction_markets` | odds moves, market pricing, upset calls |

## Quick start

```bash
cd worldcup-signals-agent
pip install -r requirements.txt

# Offline demo — runs the engine against fixtures, no keys needed:
python -m worldcup_agent run --mock

# Live collection with heuristic extraction (no Claude key):
python -m worldcup_agent run --no-llm

# Live collection with Claude-powered extraction:
export ANTHROPIC_API_KEY=sk-ant-...
python -m worldcup_agent run

# Inspect what's stored:
python -m worldcup_agent board       # per-match edge board
python -m worldcup_agent signals     # top signals with rationale
python -m worldcup_agent info        # config + row counts
```

## DuckDB schema

| Table | Grain | Key columns |
|---|---|---|
| `runs` | one engine invocation (angle × execution) | `run_id`, `batch_id`, `angle`, `item_count`, `signal_count`, `extract_method` |
| `raw_items` | every ranked candidate | `run_id`, `item_id`, `source`, `title`, `url`, `engagement`, `final_score` |
| `clusters` | engine evidence clusters | `run_id`, `cluster_id`, `score`, `uncertainty` |
| `signals` | extracted prediction signal | `signal_id`, `match`, `teams`, `signal_type`, `favored_team`, `edge_score`, `confidence`, `rationale` |
| `match_edges` (view) | per-match rollup | `match`, `signal_count`, `total_edge`, `avg_edge` |

`edge_score` is `strength_weight × confidence_weight` on a 0..1 scale, so the
edge board ranks fixtures by how much actionable signal has accumulated.

Query it directly:

```sql
-- which upcoming match has the most decision-relevant chatter?
SELECT * FROM match_edges;

-- all high-confidence injury signals
SELECT match, favored_team, rationale FROM signals
WHERE signal_type = 'injury' AND confidence = 'high'
ORDER BY edge_score DESC;
```

## Scheduling

The pipeline is idempotent per batch and append-only across batches, so it is
safe to run on a cron (e.g. every few hours during the tournament). Each run
adds a new `batch_id`, letting you track how signals evolve as matches approach.

## Configuration

See `.env.example`. Everything is environment-overridable; `--mock` and
`--no-llm` flags exist for offline / no-key operation. The underlying engine's
own configuration (extra sources, API keys) is documented in
[`../CONFIGURATION.md`](../CONFIGURATION.md).
