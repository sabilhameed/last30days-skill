"""Tests for the DuckDB storage layer."""

from __future__ import annotations

from worldcup_agent.extract import PredictiveSignal
from worldcup_agent.store import SignalStore


def _report() -> dict:
    return {
        "topic": "World Cup injuries",
        "generated_at": "2026-06-16T00:00:00Z",
        "range_from": "2026-05-17",
        "range_to": "2026-06-16",
        "clusters": [
            {"cluster_id": "c1", "title": "Injury cluster", "score": 42.0,
             "sources": ["reddit", "x"], "uncertainty": "thin-evidence"},
        ],
    }


def _items() -> list[dict]:
    return [
        {
            "item_id": "cand-1", "rank": 1, "source": "reddit",
            "sources": ["reddit"], "title": "Star striker doubtful",
            "url": "https://reddit.com/x", "snippet": "fitness doubt ahead of QF",
            "author": None, "published_at": "2026-06-14", "engagement": 80.0,
            "final_score": 50.0,
        },
    ]


def _signals() -> list[PredictiveSignal]:
    return [
        PredictiveSignal(
            match="Brazil vs Croatia", teams=["Brazil", "Croatia"],
            signal_type="injury", favored_team="Croatia",
            edge_strength="high", confidence="medium",
            time_horizon="next match", rationale="Striker doubtful.",
            source_item_ids=["cand-1"],
        ),
    ]


def test_record_run_and_counts(tmp_path):
    db = tmp_path / "t.duckdb"
    with SignalStore(db) as store:
        batch = store.new_batch_id()
        run_id = store.record_run(
            batch_id=batch, angle="injuries_and_availability",
            topic="World Cup injuries", signal_focus="injuries",
            report=_report(), items=_items(), signals=_signals(),
            extract_method="heuristic", is_mock=True,
        )
        assert run_id
        counts = store.counts()
        assert counts == {"runs": 1, "raw_items": 1, "clusters": 1, "signals": 1}


def test_match_edge_board(tmp_path):
    db = tmp_path / "t.duckdb"
    with SignalStore(db) as store:
        batch = store.new_batch_id()
        store.record_run(
            batch_id=batch, angle="injuries_and_availability",
            topic="t", signal_focus="f", report=_report(),
            items=_items(), signals=_signals(),
            extract_method="heuristic", is_mock=True,
        )
        board = store.match_edge_board()
        assert len(board) == 1
        assert board[0]["match"] == "Brazil vs Croatia"
        # high (1.0) * medium (0.66) = 0.66
        assert board[0]["total_edge"] == 0.66


def test_raw_item_idempotent(tmp_path):
    db = tmp_path / "t.duckdb"
    with SignalStore(db) as store:
        batch = store.new_batch_id()
        # Two runs reuse the same item_id under different run_ids -> both kept,
        # but a duplicate within a run is ignored by the primary key.
        rid = store.record_run(
            batch_id=batch, angle="a", topic="t", signal_focus="f",
            report=_report(), items=_items() + _items(), signals=[],
            extract_method="none", is_mock=True,
        )
        assert rid
        # second item with the same (run_id, item_id) collapsed to one row
        assert store.counts()["raw_items"] == 1
