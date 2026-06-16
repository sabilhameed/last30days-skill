"""Tests for the engine wrapper helpers (no subprocess required)."""

from __future__ import annotations

from worldcup_agent.engine import _engagement_total, iter_ranked_items


def test_engagement_total_variants():
    assert _engagement_total(None) == 0.0
    assert _engagement_total(100) == 100.0
    assert _engagement_total({"likes": 10, "comments": 5, "rank": 2.0}) == 17.0
    assert _engagement_total("weird") == 0.0


def test_iter_ranked_items_flattens():
    report = {
        "ranked_candidates": [
            {
                "candidate_id": "c1", "source": "reddit", "sources": ["reddit"],
                "title": " Title ", "url": "u", "snippet": " snip ",
                "final_score": 12.5, "engagement": {"likes": 3},
                "source_items": [{"published_at": "2026-06-10", "author": "bob"}],
            },
            {
                "candidate_id": "c2", "source": "x", "title": "T2", "url": "u2",
                "snippet": "s2", "final_score": 8.0, "engagement": 4,
                "source_items": [],
            },
        ]
    }
    items = iter_ranked_items(report)
    assert len(items) == 2
    assert items[0]["item_id"] == "c1"
    assert items[0]["rank"] == 1
    assert items[0]["title"] == "Title"  # stripped
    assert items[0]["published_at"] == "2026-06-10"
    assert items[0]["author"] == "bob"
    assert items[0]["engagement"] == 3.0


def test_iter_ranked_items_limit():
    report = {"ranked_candidates": [{"candidate_id": f"c{i}", "title": str(i)} for i in range(5)]}
    assert len(iter_ranked_items(report, limit=2)) == 2
