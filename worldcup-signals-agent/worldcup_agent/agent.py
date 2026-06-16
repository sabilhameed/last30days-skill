"""The autonomous orchestrator.

For each configured research angle it: runs the last30days engine, flattens the
ranked evidence, extracts prediction signals (LLM or heuristic), and persists
everything to DuckDB. Returns a summary the CLI renders.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import AgentConfig
from .engine import EngineError, iter_ranked_items, run_angle
from .extract import extract_signals
from .store import SignalStore


@dataclass
class AngleResult:
    angle: str
    items: int
    signals: int
    method: str
    error: str | None = None


@dataclass
class RunSummary:
    batch_id: str
    mock: bool
    angles: list[AngleResult] = field(default_factory=list)
    edge_board: list[dict] = field(default_factory=list)
    top_signals: list[dict] = field(default_factory=list)
    db_counts: dict = field(default_factory=dict)

    @property
    def total_signals(self) -> int:
        return sum(a.signals for a in self.angles)


def run_pipeline(cfg: AgentConfig) -> RunSummary:
    """Execute the full collect -> extract -> store pipeline across all angles."""
    store = SignalStore(cfg.db_path)
    batch_id = store.new_batch_id()
    summary = RunSummary(batch_id=batch_id, mock=cfg.mock)

    try:
        for angle in cfg.angles:
            print(f"[{angle.name}] running engine ({'mock' if cfg.mock else 'live'})...")
            try:
                engine_run = run_angle(cfg, angle)
            except EngineError as exc:
                print(f"  ! engine error: {exc}")
                summary.angles.append(
                    AngleResult(angle=angle.name, items=0, signals=0, method="none", error=str(exc))
                )
                continue

            items = iter_ranked_items(engine_run.report, limit=cfg.max_items_per_run)
            signals, method = extract_signals(cfg, angle, items)
            print(f"  -> {len(items)} items, {len(signals)} signals ({method})")

            store.record_run(
                batch_id=batch_id,
                angle=angle.name,
                topic=angle.topic,
                signal_focus=angle.signal_focus,
                report=engine_run.report,
                items=items,
                signals=signals,
                extract_method=method,
                is_mock=cfg.mock,
            )
            summary.angles.append(
                AngleResult(angle=angle.name, items=len(items), signals=len(signals), method=method)
            )

        summary.edge_board = store.match_edge_board()
        summary.top_signals = store.top_signals(batch_id=batch_id)
        summary.db_counts = store.counts()
    finally:
        store.close()

    return summary
