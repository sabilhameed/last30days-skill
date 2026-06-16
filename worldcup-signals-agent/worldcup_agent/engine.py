"""Thin wrapper around the `last30days` skill engine.

The engine is invoked as a subprocess with `--emit=json --agent`, which makes
it non-interactive and emits the full structured `Report` (topic, range,
clusters, ranked_candidates, items_by_source) to stdout. We parse that JSON and
hand it back; all interpretation happens in `extract.py` / `store.py`.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import AgentConfig, ResearchAngle

# Interpreters to probe, newest first. The engine requires Python 3.12+.
_PYTHON_CANDIDATES = ("python3.14", "python3.13", "python3.12", "python3")
_MIN_VERSION = (3, 12)


class EngineError(RuntimeError):
    """Raised when the engine cannot be located or fails to run."""


def resolve_python(explicit: str | None = None) -> str:
    """Return a Python interpreter that satisfies the engine's 3.12+ floor."""
    candidates = [explicit] if explicit else list(_PYTHON_CANDIDATES)
    for name in candidates:
        if not name:
            continue
        path = shutil.which(name)
        if not path:
            continue
        try:
            out = subprocess.run(
                [path, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (subprocess.SubprocessError, OSError):
            continue
        if out.returncode != 0:
            continue
        try:
            major, minor = (int(x) for x in out.stdout.strip().split("."))
        except ValueError:
            continue
        if (major, minor) >= _MIN_VERSION:
            return path
    raise EngineError(
        f"No Python {_MIN_VERSION[0]}.{_MIN_VERSION[1]}+ interpreter found "
        f"(tried: {', '.join(c for c in candidates if c)}). "
        "Set LAST30DAYS_PYTHON to a suitable interpreter."
    )


def engine_script(skill_dir: Path) -> Path:
    script = skill_dir / "scripts" / "last30days.py"
    if not script.is_file():
        raise EngineError(
            f"last30days engine not found at {script}. "
            "Set LAST30DAYS_SKILL_DIR to the skill directory containing scripts/last30days.py."
        )
    return script


@dataclass
class EngineRun:
    """Result of one engine invocation."""

    angle: ResearchAngle
    report: dict
    mock: bool


def _build_command(
    cfg: AgentConfig, python_bin: str, script: Path, angle: ResearchAngle
) -> list[str]:
    cmd = [
        python_bin,
        str(script),
        angle.topic,
        "--emit=json",
        "--agent",
    ]
    if cfg.mock:
        cmd.append("--mock")
    if cfg.depth == "quick":
        cmd.append("--quick")
    elif cfg.depth == "deep":
        cmd.append("--deep")
    if angle.subreddits:
        cmd.append(f"--subreddits={','.join(angle.subreddits)}")
    if angle.x_related:
        cmd.append(f"--x-related={','.join(angle.x_related)}")
    return cmd


def run_angle(cfg: AgentConfig, angle: ResearchAngle, *, timeout: int = 600) -> EngineRun:
    """Run the engine for a single research angle and return the parsed report."""
    python_bin = resolve_python(cfg.python_bin)
    script = engine_script(cfg.skill_dir)
    cmd = _build_command(cfg, python_bin, script, angle)

    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise EngineError(
            f"engine failed for angle '{angle.name}' (exit {proc.returncode}):\n"
            f"{proc.stderr[-2000:]}"
        )
    try:
        report = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        # The engine writes diagnostics to stderr; surface a slice of stdout too.
        raise EngineError(
            f"could not parse engine JSON for angle '{angle.name}': {exc}\n"
            f"stdout head: {proc.stdout[:500]!r}"
        ) from exc
    return EngineRun(angle=angle, report=report, mock=cfg.mock)


def iter_ranked_items(report: dict, limit: int | None = None) -> list[dict]:
    """Flatten the engine's ranked_candidates into plain item dicts for storage.

    Each candidate carries one or more source_items; we keep the candidate-level
    score/url and merge in the best published date and engagement from its items.
    """
    items: list[dict] = []
    for rank, cand in enumerate(report.get("ranked_candidates", []), start=1):
        source_items = cand.get("source_items") or []
        published_at = None
        author = None
        for si in source_items:
            published_at = published_at or si.get("published_at")
            author = author or si.get("author")
        items.append(
            {
                "item_id": cand.get("candidate_id") or cand.get("item_id") or f"cand-{rank}",
                "rank": rank,
                "source": cand.get("source", "unknown"),
                "sources": cand.get("sources") or [cand.get("source")],
                "title": (cand.get("title") or "").strip(),
                "url": cand.get("url") or "",
                "snippet": (cand.get("snippet") or "").strip(),
                "author": author,
                "published_at": published_at,
                "engagement": _engagement_total(cand.get("engagement")),
                "final_score": float(cand.get("final_score") or 0.0),
            }
        )
        if limit is not None and len(items) >= limit:
            break
    return items


def _engagement_total(engagement) -> float:
    """Normalize the heterogeneous engagement field to a single number."""
    if engagement is None:
        return 0.0
    if isinstance(engagement, (int, float)):
        return float(engagement)
    if isinstance(engagement, dict):
        return float(sum(v for v in engagement.values() if isinstance(v, (int, float))))
    return 0.0


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    from .config import DEFAULT_ANGLES

    config = AgentConfig(mock=True)
    result = run_angle(config, DEFAULT_ANGLES[0])
    print(f"topic={result.report['topic']} items={len(result.report['ranked_candidates'])}")
    json.dump(iter_ranked_items(result.report, limit=3), sys.stdout, indent=2)
