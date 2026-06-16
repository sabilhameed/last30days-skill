"""Tests for the heuristic extractor and edge scoring."""

from __future__ import annotations

from worldcup_agent.config import DEFAULT_ANGLES
from worldcup_agent.extract import (
    PredictiveSignal,
    edge_score,
    extract_heuristic,
    _classify,
    _guess_teams,
)


def test_edge_score_levels():
    s = PredictiveSignal(
        match="A vs B", teams=["A", "B"], signal_type="form",
        favored_team="A", edge_strength="high", confidence="high",
        time_horizon="next match", rationale="r",
    )
    assert edge_score(s) == 1.0
    s2 = s.model_copy(update={"confidence": "low"})
    assert edge_score(s2) == round(1.0 * 0.33, 4)


def test_classify_keywords():
    assert _classify("Star striker picks up an injury") == "injury"
    assert _classify("Predicted starting XI revealed") == "lineup"
    assert _classify("Odds shift toward the favorites") == "betting_odds"
    assert _classify("A perfectly ordinary sentence") == "other"


def test_guess_teams():
    teams = _guess_teams("Brazil face Croatia in the quarter-final")
    assert "Brazil" in teams and "Croatia" in teams


def test_heuristic_skips_irrelevant():
    angle = DEFAULT_ANGLES[0]
    items = [
        {"item_id": "i1", "source": "reddit", "title": "Brazil striker injury doubt",
         "snippet": "fitness concern", "engagement": 60.0},
        {"item_id": "i2", "source": "x", "title": "Cool highlight reel",
         "snippet": "great goals montage", "engagement": 5.0},
    ]
    signals = extract_heuristic(angle, items)
    # only the injury item is prediction-relevant
    assert len(signals) == 1
    assert signals[0].signal_type == "injury"
    assert signals[0].edge_strength == "high"  # engagement >= 50
    assert "Brazil" in signals[0].teams
