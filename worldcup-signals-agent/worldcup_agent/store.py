"""DuckDB persistence layer.

Schema (all timestamps stored as ISO strings / native TIMESTAMP where possible):

  runs            one row per engine invocation (angle x execution)
  raw_items       every ranked candidate the engine returned for a run
  clusters        engine evidence clusters for a run
  signals         prediction-relevant signals extracted from the run
  match_edges     a VIEW aggregating signals into a per-match edge board

The design is append-only: each `agent run` writes a fresh batch keyed by a
UUID run_id, so historical runs accumulate for trend analysis. Re-running is
safe — primary keys prevent duplicate rows within a run.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from .extract import PredictiveSignal, edge_score

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id        VARCHAR PRIMARY KEY,
    batch_id      VARCHAR NOT NULL,        -- groups all angles from one `agent run`
    angle         VARCHAR NOT NULL,
    topic         VARCHAR NOT NULL,
    signal_focus  VARCHAR,
    created_at    TIMESTAMP NOT NULL,
    generated_at  VARCHAR,                 -- engine's own report timestamp
    range_from    VARCHAR,
    range_to      VARCHAR,
    item_count    INTEGER NOT NULL,
    cluster_count INTEGER NOT NULL,
    signal_count  INTEGER NOT NULL,
    extract_method VARCHAR NOT NULL,
    is_mock       BOOLEAN NOT NULL
);

CREATE TABLE IF NOT EXISTS raw_items (
    run_id        VARCHAR NOT NULL,
    item_id       VARCHAR NOT NULL,
    rank          INTEGER,
    source        VARCHAR,
    sources       VARCHAR,                 -- JSON array
    title         VARCHAR,
    url           VARCHAR,
    snippet       VARCHAR,
    author        VARCHAR,
    published_at  VARCHAR,
    engagement    DOUBLE,
    final_score   DOUBLE,
    PRIMARY KEY (run_id, item_id)
);

CREATE TABLE IF NOT EXISTS clusters (
    run_id        VARCHAR NOT NULL,
    cluster_id    VARCHAR NOT NULL,
    title         VARCHAR,
    score         DOUBLE,
    sources       VARCHAR,                 -- JSON array
    uncertainty   VARCHAR,
    PRIMARY KEY (run_id, cluster_id)
);

CREATE TABLE IF NOT EXISTS signals (
    signal_id      VARCHAR PRIMARY KEY,
    run_id         VARCHAR NOT NULL,
    batch_id       VARCHAR NOT NULL,
    angle          VARCHAR NOT NULL,
    match          VARCHAR,
    teams          VARCHAR,                -- JSON array
    signal_type    VARCHAR,
    favored_team   VARCHAR,
    edge_strength  VARCHAR,
    confidence     VARCHAR,
    edge_score     DOUBLE,
    time_horizon   VARCHAR,
    rationale      VARCHAR,
    source_item_ids VARCHAR,               -- JSON array
    created_at     TIMESTAMP NOT NULL
);
"""

# A convenience view: the per-match "edge board" rolled up from signals.
MATCH_EDGES_VIEW = """
CREATE OR REPLACE VIEW match_edges AS
SELECT
    match,
    count(*)               AS signal_count,
    round(sum(edge_score), 3) AS total_edge,
    round(avg(edge_score), 3) AS avg_edge,
    max(created_at)        AS last_seen
FROM signals
WHERE match IS NOT NULL AND match <> 'unknown'
GROUP BY match
ORDER BY total_edge DESC;
"""


def _now() -> datetime:
    return datetime.now(timezone.utc)


class SignalStore:
    """Wraps a DuckDB connection and the World Cup signal schema."""

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(self.db_path))
        self._init_schema()

    def _init_schema(self) -> None:
        self.con.execute(SCHEMA)
        self.con.execute(MATCH_EDGES_VIEW)

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> "SignalStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def new_batch_id(self) -> str:
        return uuid.uuid4().hex

    def record_run(
        self,
        *,
        batch_id: str,
        angle: str,
        topic: str,
        signal_focus: str,
        report: dict,
        items: list[dict],
        signals: list[PredictiveSignal],
        extract_method: str,
        is_mock: bool,
    ) -> str:
        """Persist one engine run plus its items, clusters and signals.

        Returns the generated run_id.
        """
        run_id = uuid.uuid4().hex
        created = _now()

        self.con.execute(
            """INSERT INTO runs VALUES
               (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                run_id,
                batch_id,
                angle,
                topic,
                signal_focus,
                created,
                report.get("generated_at"),
                report.get("range_from"),
                report.get("range_to"),
                len(items),
                len(report.get("clusters", [])),
                len(signals),
                extract_method,
                is_mock,
            ],
        )

        for it in items:
            self.con.execute(
                """INSERT OR IGNORE INTO raw_items VALUES
                   (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    run_id,
                    it["item_id"],
                    it["rank"],
                    it["source"],
                    json.dumps(it["sources"]),
                    it["title"],
                    it["url"],
                    it["snippet"],
                    it["author"],
                    it["published_at"],
                    it["engagement"],
                    it["final_score"],
                ],
            )

        for cl in report.get("clusters", []):
            self.con.execute(
                """INSERT OR IGNORE INTO clusters VALUES (?, ?, ?, ?, ?, ?)""",
                [
                    run_id,
                    cl.get("cluster_id"),
                    cl.get("title"),
                    float(cl.get("score") or 0.0),
                    json.dumps(cl.get("sources") or []),
                    cl.get("uncertainty"),
                ],
            )

        for sig in signals:
            self.con.execute(
                """INSERT INTO signals VALUES
                   (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    uuid.uuid4().hex,
                    run_id,
                    batch_id,
                    angle,
                    sig.match,
                    json.dumps(sig.teams),
                    sig.signal_type,
                    sig.favored_team,
                    sig.edge_strength,
                    sig.confidence,
                    edge_score(sig),
                    sig.time_horizon,
                    sig.rationale,
                    json.dumps(sig.source_item_ids),
                    created,
                ],
            )
        return run_id

    # --- Read helpers -------------------------------------------------------

    def match_edge_board(self, limit: int = 20) -> list[dict]:
        rows = self.con.execute(
            "SELECT match, signal_count, total_edge, avg_edge, last_seen "
            "FROM match_edges LIMIT ?",
            [limit],
        ).fetchall()
        cols = ["match", "signal_count", "total_edge", "avg_edge", "last_seen"]
        return [dict(zip(cols, r)) for r in rows]

    def top_signals(self, limit: int = 15, batch_id: str | None = None) -> list[dict]:
        sql = (
            "SELECT match, signal_type, favored_team, edge_score, confidence, "
            "angle, rationale FROM signals"
        )
        params: list = []
        if batch_id:
            sql += " WHERE batch_id = ?"
            params.append(batch_id)
        sql += " ORDER BY edge_score DESC, confidence DESC LIMIT ?"
        params.append(limit)
        rows = self.con.execute(sql, params).fetchall()
        cols = ["match", "signal_type", "favored_team", "edge_score", "confidence", "angle", "rationale"]
        return [dict(zip(cols, r)) for r in rows]

    def counts(self) -> dict:
        out = {}
        for table in ("runs", "raw_items", "clusters", "signals"):
            out[table] = self.con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        return out
