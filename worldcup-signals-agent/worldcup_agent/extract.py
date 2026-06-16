"""Signal extraction: turn raw engine items into structured prediction signals.

Primary path uses Claude (Anthropic SDK) with structured outputs to classify the
month of chatter into discrete, prediction-relevant signals. When no API key is
available (or extraction is disabled), a deterministic keyword heuristic produces
a coarser set of signals so the pipeline still yields structured output offline.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from .config import AgentConfig, ResearchAngle

SignalType = Literal[
    "injury",
    "suspension",
    "lineup",
    "form",
    "morale",
    "tactical",
    "betting_odds",
    "fan_sentiment",
    "managerial",
    "weather",
    "other",
]
Strength = Literal["low", "medium", "high"]

# Map categorical strength/confidence to a numeric weight for the edge score.
_LEVEL_WEIGHT = {"low": 0.33, "medium": 0.66, "high": 1.0}


class PredictiveSignal(BaseModel):
    """A single prediction-relevant signal distilled from the chatter."""

    match: str = Field(
        description="The fixture this bears on, e.g. 'Brazil vs Argentina'. Use 'unknown' if unclear.",
    )
    teams: list[str] = Field(
        default_factory=list,
        description="National teams involved in or affected by this signal.",
    )
    signal_type: SignalType = Field(description="Category of the signal.")
    favored_team: str = Field(
        description="The team that gains an edge from this signal, or 'none' if neutral/unclear.",
    )
    edge_strength: Strength = Field(
        description="How strongly this shifts the expected outcome.",
    )
    confidence: Strength = Field(
        description="Confidence that the signal is real and well-sourced, not rumor.",
    )
    time_horizon: str = Field(
        description="When it applies, e.g. 'next match', 'group stage', 'knockouts'.",
    )
    rationale: str = Field(
        description="One or two sentences explaining the signal and its predictive relevance.",
    )
    source_item_ids: list[str] = Field(
        default_factory=list,
        description="item_id values of the raw items this signal is drawn from.",
    )


class SignalExtraction(BaseModel):
    """Container the model fills in."""

    signals: list[PredictiveSignal] = Field(default_factory=list)


def edge_score(signal: PredictiveSignal) -> float:
    """Combine strength and confidence into a single 0..1 edge score."""
    return round(
        _LEVEL_WEIGHT[signal.edge_strength] * _LEVEL_WEIGHT[signal.confidence], 4
    )


_SYSTEM_PROMPT = (
    "You are a football (soccer) match-prediction analyst working the ongoing "
    "World Cup. You are given a batch of recent posts and headlines (Reddit, X, "
    "YouTube, prediction markets, web) collected for one research angle. Extract "
    "discrete, prediction-relevant SIGNALS: concrete facts or shifts that would "
    "give an edge in forecasting upcoming match outcomes — injuries, suspensions, "
    "lineup/rotation news, form and momentum, morale, tactical match-ups, "
    "managerial decisions, weather, and betting/prediction-market moves. "
    "Ignore generic hype, highlight reels, and off-pitch noise with no bearing on "
    "results. Be conservative: rate edge_strength and confidence honestly, and use "
    "'unknown'/'none' when the chatter does not pin down a fixture or beneficiary. "
    "Reference the item_id of every item a signal draws from."
)


def _build_user_prompt(angle: ResearchAngle, items: list[dict]) -> str:
    lines = [
        f"Research angle: {angle.name}",
        f"Focus: {angle.signal_focus}",
        "",
        "Items (id | source | engagement | title — snippet):",
    ]
    for it in items:
        snippet = it["snippet"][:280].replace("\n", " ")
        lines.append(
            f"- {it['item_id']} | {it['source']} | eng={int(it['engagement'])} | "
            f"{it['title']} — {snippet}"
        )
    lines.append(
        "\nReturn the structured set of signals. If nothing in this batch is "
        "prediction-relevant, return an empty list."
    )
    return "\n".join(lines)


def extract_with_llm(
    cfg: AgentConfig, angle: ResearchAngle, items: list[dict]
) -> list[PredictiveSignal]:
    """Use Claude structured outputs to extract signals. Raises on SDK errors."""
    import anthropic  # imported lazily so offline/mock runs don't need the dep

    client = anthropic.Anthropic()
    response = client.messages.parse(
        model=cfg.model,
        max_tokens=4096,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_user_prompt(angle, items)}],
        output_format=SignalExtraction,
    )
    parsed = response.parsed_output
    return parsed.signals if parsed else []


# --- Heuristic fallback -----------------------------------------------------

_KEYWORD_TYPES: tuple[tuple[SignalType, tuple[str, ...]], ...] = (
    ("injury", ("injur", "fitness", "doubt", "knock", "strain", "ruled out", "limp")),
    ("suspension", ("suspend", "red card", "ban", "yellow card accumulation")),
    ("lineup", ("lineup", "line-up", "starting xi", "starting 11", "rotation", "rested", "bench")),
    ("managerial", ("manager", "coach", "sacked", "press conference", "tactic change")),
    ("betting_odds", ("odds", "favorite", "favourite", "bet", "market", "polymarket", "upset")),
    ("tactical", ("tactic", "formation", "press", "matchup", "match-up", "counter")),
    ("form", ("form", "momentum", "streak", "winless", "unbeaten")),
    ("morale", ("morale", "locker room", "dressing room", "confidence", "crisis")),
    ("fan_sentiment", ("fans", "supporters", "reaction", "furious", "buzz")),
    ("weather", ("weather", "rain", "heat", "pitch condition")),
)


def _classify(text: str) -> SignalType:
    low = text.lower()
    for signal_type, keywords in _KEYWORD_TYPES:
        if any(k in low for k in keywords):
            return signal_type
    return "other"


def extract_heuristic(
    angle: ResearchAngle, items: list[dict]
) -> list[PredictiveSignal]:
    """Deterministic, no-API extractor. Coarser but always available.

    Prediction-market items (Polymarket) get a dedicated parser that reads the
    implied-probability movement baked into the snippet; everything else falls
    back to keyword classification.
    """
    signals: list[PredictiveSignal] = []
    for it in items:
        if (it.get("source") or "").lower() == "polymarket":
            sig = extract_polymarket(it)
            if sig is not None:
                signals.append(sig)
            continue

        text = f"{it['title']} {it['snippet']}"
        signal_type = _classify(text)
        if signal_type == "other":
            continue  # don't manufacture signal from clearly-irrelevant chatter
        teams = _guess_teams(text)
        # Engagement is a rough proxy for how loud/real a signal is.
        strength: Strength = "high" if it["engagement"] >= 50 else "medium" if it["engagement"] >= 10 else "low"
        signals.append(
            PredictiveSignal(
                match=" vs ".join(teams[:2]) if len(teams) >= 2 else "unknown",
                teams=teams,
                signal_type=signal_type,
                favored_team="none",
                edge_strength=strength,
                confidence="low",  # heuristic has no judgment, so stay humble
                time_horizon="next match",
                rationale=(
                    f"Keyword-classified as {signal_type} from {it['source']} "
                    f"chatter: {it['title'][:120]}"
                ),
                source_item_ids=[it["item_id"]],
            )
        )
    return signals


# --- Polymarket prediction-market extraction --------------------------------

# Probability movement embedded in the snippet, e.g. "down 8.0% this week".
_MOVE_RE = re.compile(
    r"\b(up|down)\s+([0-9]+(?:\.[0-9]+)?)%\s+(today|this week|this month)\b",
    re.IGNORECASE,
)
# Availability markets, e.g. "Will Neymar play in the World Cup?".
_AVAILABILITY_RE = re.compile(
    r"\bwill\s+(.+?)\s+(?:play|feature|start|be (?:fit|available))\b", re.IGNORECASE
)
# "Will <team> win ..." outcome markets.
_WIN_RE = re.compile(r"\bwill\s+(.+?)\s+win\b", re.IGNORECASE)


def _move_strength(pct: float) -> Strength:
    """A bigger probability swing is a stronger market signal."""
    if pct >= 8.0:
        return "high"
    if pct >= 4.0:
        return "medium"
    return "low"


def extract_polymarket(item: dict) -> PredictiveSignal | None:
    """Turn one Polymarket market into a structured signal.

    Reads the probability move from the snippet (direction + magnitude +
    timeframe) and the market subject from the title, mapping players to their
    national teams. Availability markets ('Will X play') are typed as injury
    signals; everything else is a betting_odds signal.
    """
    title = item.get("title") or ""
    snippet = item.get("snippet") or ""
    text = f"{title} {snippet}"
    teams = _guess_teams(text)

    move = _MOVE_RE.search(snippet)
    direction = move.group(1).lower() if move else None
    pct = float(move.group(2)) if move else 0.0
    timeframe = move.group(3).lower() if move else None

    # Classify the market and infer who it favors.
    avail = _AVAILABILITY_RE.search(title)
    win = _WIN_RE.search(title)
    if avail:
        signal_type: SignalType = "injury"
        subject = avail.group(1).strip()
        # "down" on a will-play market = less likely to feature = hurts their side.
        favored_team = "none"
        horizon = "tournament"
        subject_desc = f"availability of {subject}"
    elif win:
        signal_type = "betting_odds"
        subject = win.group(1).strip()
        # "up" = market favors this team more; "down" = favors their rivals.
        favored_team = (
            (teams[0] if teams else subject) if direction == "up" else "none"
        )
        horizon = "tournament"
        subject_desc = f"win odds for {subject}"
    else:
        signal_type = "betting_odds"
        favored_team = "none"
        horizon = "tournament"
        subject_desc = title or "World Cup market"

    # Strength comes from the size of the move; engagement is the fallback proxy.
    if move:
        strength = _move_strength(pct)
        # A real-money market move is a more trustworthy signal than raw chatter.
        confidence: Strength = "medium" if pct >= 4.0 else "low"
        move_desc = f"implied probability {direction} {pct:g}% {timeframe}"
    else:
        strength = "high" if item.get("engagement", 0) >= 50 else "low"
        confidence = "low"
        move_desc = "active market, no recent move parsed"

    return PredictiveSignal(
        match=" vs ".join(teams[:2]) if len(teams) >= 2 else "unknown",
        teams=teams,
        signal_type=signal_type,
        favored_team=favored_team,
        edge_strength=strength,
        confidence=confidence,
        time_horizon=horizon,
        rationale=f"Polymarket {subject_desc}: {move_desc} ({title}).",
        source_item_ids=[item["item_id"]],
    )


# Star players mapped to their national team, so a player-only market still
# resolves to a nation in the fallback extractor.
_PLAYER_NATION: dict[str, str] = {
    "Messi": "Argentina", "Neymar": "Brazil", "Vinicius": "Brazil",
    "Mbappe": "France", "Mbappé": "France", "Kane": "England",
    "Bellingham": "England", "Foden": "England", "Yamal": "Spain",
    "Pedri": "Spain", "Musiala": "Germany", "Ronaldo": "Portugal",
    "Pulisic": "USA", "Modric": "Croatia", "Lukaku": "Belgium",
    "Haaland": "Norway", "Salah": "Egypt", "Son": "South Korea",
}

# A small roster of World Cup nations for naive team detection in the fallback.
_NATIONS = (
    "Argentina", "Brazil", "France", "England", "Spain", "Germany", "Portugal",
    "Netherlands", "Croatia", "Belgium", "Italy", "Morocco", "Japan", "USA",
    "Mexico", "Uruguay", "Colombia", "Senegal", "Switzerland", "Denmark",
    "Poland", "South Korea", "Australia", "Ecuador", "Ghana", "Cameroon",
    "Saudi Arabia", "Qatar", "Canada", "Serbia", "Wales", "Iran",
)


def _guess_teams(text: str) -> list[str]:
    found: list[str] = []
    for nation in _NATIONS:
        if re.search(rf"\b{re.escape(nation)}\b", text, re.IGNORECASE) and nation not in found:
            found.append(nation)
    # Resolve star-player names to their nation (e.g. "Neymar" -> "Brazil").
    for player, nation in _PLAYER_NATION.items():
        if re.search(rf"\b{re.escape(player)}\b", text, re.IGNORECASE) and nation not in found:
            found.append(nation)
    return found


def extract_signals(
    cfg: AgentConfig, angle: ResearchAngle, items: list[dict]
) -> tuple[list[PredictiveSignal], str]:
    """Extract signals, returning (signals, method).

    Prefers the LLM extractor; falls back to the heuristic on missing key,
    disabled LLM, or SDK error.
    """
    if not items:
        return [], "none"
    if not cfg.no_llm and cfg.has_anthropic_key():
        try:
            return extract_with_llm(cfg, angle, items), "llm"
        except Exception as exc:  # noqa: BLE001 - fall back rather than crash the run
            print(f"  [extract] LLM extraction failed ({exc}); using heuristic.")
    return extract_heuristic(angle, items), "heuristic"
