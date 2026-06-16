"""Command-line entrypoint: `python -m worldcup_agent <command>`.

Commands:
  run     collect signals across all angles and store them in DuckDB
  board   print the current per-match edge board from the database
  signals print the top stored signals
  info    print configuration and database counts
"""

from __future__ import annotations

import argparse
import sys

from .agent import run_fixtures, run_pipeline
from .config import AgentConfig, load_fixtures
from .store import SignalStore


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--db", help="DuckDB path (overrides WORLDCUP_AGENT_DB).")


def _config_from_args(args) -> AgentConfig:
    cfg = AgentConfig()
    if getattr(args, "db", None):
        from pathlib import Path

        cfg.db_path = Path(args.db)
    if getattr(args, "mock", False):
        cfg.mock = True
    if getattr(args, "no_llm", False):
        cfg.no_llm = True
    if getattr(args, "depth", None):
        cfg.depth = args.depth
    return cfg


def _print_board(board: list[dict]) -> None:
    if not board:
        print("  (no per-match edges yet)")
        return
    print(f"  {'MATCH':<28} {'SIGNALS':>7} {'TOTAL':>7} {'AVG':>6}  LAST SEEN")
    for row in board:
        print(
            f"  {str(row['match'])[:28]:<28} {row['signal_count']:>7} "
            f"{row['total_edge']:>7} {row['avg_edge']:>6}  {row['last_seen']}"
        )


def cmd_run(args) -> int:
    cfg = _config_from_args(args)
    if cfg.mock:
        print("Mode: MOCK (engine fixtures, no network).")
    elif not cfg.has_anthropic_key():
        print("Note: no ANTHROPIC_API_KEY found — using heuristic extraction.")
    summary = run_pipeline(cfg)

    print("\n=== Run summary ===")
    for a in summary.angles:
        status = a.error if a.error else f"{a.items} items / {a.signals} signals ({a.method})"
        print(f"  {a.angle:<32} {status}")
    print(f"\nBatch {summary.batch_id} produced {summary.total_signals} signals.")

    print("\n=== Per-match edge board ===")
    _print_board(summary.edge_board)

    print("\n=== Top signals this batch ===")
    if not summary.top_signals:
        print("  (none)")
    for s in summary.top_signals:
        print(
            f"  [{s['edge_score']}] {s['match']} — {s['signal_type']} "
            f"favors {s['favored_team']} (conf={s['confidence']}, {s['angle']})"
        )

    print(f"\nDB rows: {summary.db_counts}")
    return 0


def cmd_fixtures(args) -> int:
    cfg = _config_from_args(args)
    fixtures = load_fixtures(args.file)
    if cfg.mock:
        print("Mode: MOCK (engine fixtures, no network).")
    elif not cfg.has_anthropic_key():
        print("Note: no ANTHROPIC_API_KEY found — using heuristic extraction.")
    print(f"Researching {len(fixtures)} fixtures from {args.file}\n")

    summary = run_fixtures(cfg, fixtures)

    print("\n=== Per-fixture intel ===")
    # Index odds + signals by match for a combined readout.
    by_match: dict[str, list[dict]] = {}
    for s in summary.top_signals:
        by_match.setdefault(s["match"], []).append(s)

    for fx in fixtures:
        sigs = by_match.get(fx.match, [])
        odds = _format_odds(fx)
        print(f"\n● {fx.match}   (reward: {odds})")
        if not sigs:
            print("    no prediction-relevant signals surfaced this run")
            continue
        for s in sorted(sigs, key=lambda x: -x["edge_score"]):
            print(
                f"    [{s['edge_score']}] {s['signal_type']} — favors "
                f"{s['favored_team']} (conf={s['confidence']})"
            )
            print(f"        {s['rationale']}")

    print(f"\nBatch {summary.batch_id}: {summary.total_signals} signals across "
          f"{len(fixtures)} fixtures. DB rows: {summary.db_counts}")
    return 0


def _format_odds(fx) -> str:
    parts = []
    if fx.home_win is not None:
        parts.append(f"{fx.home} {fx.home_win}")
    if fx.draw is not None:
        parts.append(f"draw {fx.draw}")
    if fx.away_win is not None:
        parts.append(f"{fx.away} {fx.away_win}")
    return " / ".join(parts) if parts else "n/a"


def cmd_board(args) -> int:
    cfg = _config_from_args(args)
    with SignalStore(cfg.db_path) as store:
        print("=== Per-match edge board ===")
        _print_board(store.match_edge_board())
    return 0


def cmd_signals(args) -> int:
    cfg = _config_from_args(args)
    with SignalStore(cfg.db_path) as store:
        rows = store.top_signals(limit=args.limit)
        if not rows:
            print("(no signals stored)")
            return 0
        for s in rows:
            print(f"[{s['edge_score']}] {s['match']} — {s['signal_type']} "
                  f"favors {s['favored_team']} (conf={s['confidence']})")
            print(f"    {s['rationale']}")
    return 0


def cmd_info(args) -> int:
    cfg = _config_from_args(args)
    print("World Cup signals agent configuration:")
    print(f"  skill_dir : {cfg.skill_dir}")
    print(f"  db_path   : {cfg.db_path}")
    print(f"  model     : {cfg.model}")
    print(f"  mock      : {cfg.mock}")
    print(f"  no_llm    : {cfg.no_llm}")
    print(f"  has_key   : {cfg.has_anthropic_key()}")
    print(f"  angles    : {', '.join(a.name for a in cfg.angles)}")
    if cfg.db_path.exists():
        with SignalStore(cfg.db_path) as store:
            print(f"  db_counts : {store.counts()}")
    else:
        print("  db_counts : (database not created yet)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="worldcup_agent", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Collect and store World Cup signals.")
    _add_common(p_run)
    p_run.add_argument("--mock", action="store_true", help="Use engine fixtures (offline).")
    p_run.add_argument("--no-llm", dest="no_llm", action="store_true", help="Heuristic extraction only.")
    p_run.add_argument("--depth", choices=["quick", "default", "deep"], help="Engine retrieval depth.")
    p_run.set_defaults(func=cmd_run)

    p_fx = sub.add_parser("fixtures", help="Research a specific set of upcoming matches.")
    _add_common(p_fx)
    p_fx.add_argument("--file", default="fixtures.json", help="JSON file of fixtures to research.")
    p_fx.add_argument("--mock", action="store_true", help="Use engine fixtures (offline).")
    p_fx.add_argument("--no-llm", dest="no_llm", action="store_true", help="Heuristic extraction only.")
    p_fx.add_argument("--depth", choices=["quick", "default", "deep"], help="Engine retrieval depth.")
    p_fx.set_defaults(func=cmd_fixtures)

    p_board = sub.add_parser("board", help="Show the per-match edge board.")
    _add_common(p_board)
    p_board.set_defaults(func=cmd_board)

    p_sig = sub.add_parser("signals", help="Show top stored signals.")
    _add_common(p_sig)
    p_sig.add_argument("--limit", type=int, default=15)
    p_sig.set_defaults(func=cmd_signals)

    p_info = sub.add_parser("info", help="Show configuration and DB counts.")
    _add_common(p_info)
    p_info.set_defaults(func=cmd_info)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
