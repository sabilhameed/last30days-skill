"""Tests for the heuristic extractor and edge scoring."""

from __future__ import annotations

from worldcup_agent.config import DEFAULT_ANGLES
from worldcup_agent.extract import (
    PredictiveSignal,
    edge_score,
    extract_heuristic,
    extract_polymarket,
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


def test_guess_teams_resolves_players():
    teams = _guess_teams("Will Neymar play in the World Cup?")
    assert teams == ["Brazil"]


def test_polymarket_availability_market():
    item = {
        "item_id": "p1", "source": "polymarket",
        "title": "Will Neymar play in the World Cup?",
        "snippet": "down 8.0% this week", "engagement": 3.0,
    }
    sig = extract_polymarket(item)
    assert sig is not None
    assert sig.signal_type == "injury"        # availability market
    assert sig.teams == ["Brazil"]            # player resolved to nation
    assert sig.edge_strength == "high"        # 8% move
    assert sig.confidence == "medium"         # real-money market move
    assert "8" in sig.rationale


def test_polymarket_win_market_up_favors_team():
    item = {
        "item_id": "p2", "source": "polymarket",
        "title": "Will South Korea win the 2026 FIFA World Cup?",
        "snippet": "up 5.0% today", "engagement": 100.0,
    }
    sig = extract_polymarket(item)
    assert sig.signal_type == "betting_odds"
    assert sig.favored_team == "South Korea"  # "up" favors the subject team
    assert sig.edge_strength == "medium"      # 5% move


def test_polymarket_routed_through_heuristic():
    angle = DEFAULT_ANGLES[-1]
    items = [{
        "item_id": "p3", "source": "polymarket",
        "title": "World Cup Group B Winner",
        "snippet": "down 10.0% this week", "engagement": 3.0,
    }]
    signals = extract_heuristic(angle, items)
    assert len(signals) == 1
    assert signals[0].signal_type == "betting_odds"
    assert signals[0].edge_strength == "high"


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
