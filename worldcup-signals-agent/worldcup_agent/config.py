"""Configuration for the World Cup signals agent.

Resolution order (highest priority first): explicit constructor args >
environment variables > defaults. Nothing here hardcodes secrets — the
Anthropic key comes from ANTHROPIC_API_KEY via the SDK's own resolution.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

# Repo root is two levels up from this file (worldcup-signals-agent/worldcup_agent/config.py).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent

# Default location of the bundled last30days engine inside this repo.
DEFAULT_SKILL_DIR = REPO_ROOT / "skills" / "last30days"


@dataclass(frozen=True)
class ResearchAngle:
    """One research query the agent fans out to the engine.

    `topic` is the search string passed to last30days. `subreddits` and
    `x_related` give the engine football-specific targeting so the headless
    planner does not have to guess. `signal_focus` is a hint surfaced to the
    extraction model about what to look for.
    """

    name: str
    topic: str
    signal_focus: str
    subreddits: tuple[str, ...] = ()
    x_related: tuple[str, ...] = ()


@dataclass(frozen=True)
class Fixture:
    """A specific upcoming match to research.

    `home_win` / `draw` / `away_win` are the reward points for a correct call
    (lower = bookmaker favorite, higher = bigger payout for the upset). Optional
    context the agent surfaces alongside its own signal-based lean.
    """

    home: str
    away: str
    home_win: float | None = None
    draw: float | None = None
    away_win: float | None = None

    @property
    def match(self) -> str:
        return f"{self.home} vs {self.away}"

    def topic(self) -> str:
        return (
            f"{self.home} vs {self.away} World Cup team news injuries "
            "lineup form prediction preview"
        )


# The default angle set. Each maps to one engine run. Tuned to surface the
# information that actually moves match outcomes: availability (injuries /
# suspensions), starting XI news, recent form, tactical match-ups, and the
# betting/prediction-market signal that aggregates everything else.
DEFAULT_ANGLES: tuple[ResearchAngle, ...] = (
    ResearchAngle(
        name="injuries_and_availability",
        topic="World Cup injuries player availability doubts",
        signal_focus="player injuries, fitness doubts, suspensions, and who is ruled out or returning",
        subreddits=("worldcup", "soccer", "football"),
        x_related=("FabrizioRomano", "OptaJoe", "BBCSport"),
    ),
    ResearchAngle(
        name="starting_lineups",
        topic="World Cup predicted starting lineup team news",
        signal_focus="predicted starting XI, rotation, rested players, and confirmed lineups",
        subreddits=("worldcup", "soccer"),
        x_related=("FabrizioRomano", "OptaJoe"),
    ),
    ResearchAngle(
        name="team_form_and_momentum",
        topic="World Cup team form momentum performance reaction",
        signal_focus="recent form, momentum, locker-room morale, and fan/pundit sentiment about each team",
        subreddits=("worldcup", "soccer", "football"),
        x_related=("OptaJoe", "ESPNFC"),
    ),
    ResearchAngle(
        name="tactics_and_matchups",
        topic="World Cup tactical analysis matchup preview",
        signal_focus="tactical match-ups, manager decisions, and stylistic advantages between upcoming opponents",
        subreddits=("soccer", "footballtactics"),
        x_related=("OptaJoe",),
    ),
    ResearchAngle(
        name="betting_and_prediction_markets",
        topic="World Cup odds prediction market favorites upset",
        signal_focus="odds movement, prediction-market pricing, value bets, and upset calls",
        subreddits=("soccer", "sportsbook"),
        x_related=("Polymarket",),
    ),
)


def load_fixtures(path: Path | str) -> list[Fixture]:
    """Load fixtures from a JSON file shaped like fixtures.json."""
    data = json.loads(Path(path).read_text())
    out: list[Fixture] = []
    for f in data.get("fixtures", []):
        out.append(
            Fixture(
                home=f["home"],
                away=f["away"],
                home_win=f.get("home_win"),
                draw=f.get("draw"),
                away_win=f.get("away_win"),
            )
        )
    return out


@dataclass
class AgentConfig:
    """Top-level runtime configuration."""

    # Where the last30days engine lives.
    skill_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("LAST30DAYS_SKILL_DIR", str(DEFAULT_SKILL_DIR))
        )
    )
    # Python interpreter used to run the engine (it requires 3.12+).
    python_bin: str | None = field(
        default_factory=lambda: os.environ.get("LAST30DAYS_PYTHON")
    )
    # DuckDB database file.
    db_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get("WORLDCUP_AGENT_DB", str(PROJECT_ROOT / "data" / "worldcup_signals.duckdb"))
        )
    )
    # Anthropic model for signal extraction. Defaults to the most capable Opus.
    model: str = field(
        default_factory=lambda: os.environ.get("WORLDCUP_AGENT_MODEL", "claude-opus-4-8")
    )
    # Run the engine against deterministic fixtures (no network / API keys).
    mock: bool = field(
        default_factory=lambda: os.environ.get("WORLDCUP_AGENT_MOCK", "").lower()
        in {"1", "true", "yes"}
    )
    # Skip the LLM extraction step and use the heuristic extractor only.
    no_llm: bool = field(
        default_factory=lambda: os.environ.get("WORLDCUP_AGENT_NO_LLM", "").lower()
        in {"1", "true", "yes"}
    )
    # Max ranked items per run fed to the extractor (keeps token cost bounded).
    max_items_per_run: int = 12
    # Engine retrieval depth: "quick" | "default" | "deep".
    depth: str = "default"
    angles: tuple[ResearchAngle, ...] = DEFAULT_ANGLES

    def has_anthropic_key(self) -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
