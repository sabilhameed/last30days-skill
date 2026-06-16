"""worldcup_agent — autonomous World Cup signal-collection agent.

Drives the `last30days` skill engine across a set of research angles, extracts
prediction-relevant signals (injuries, lineups, form, betting moves, sentiment),
and persists everything to DuckDB in a structured schema.
"""

__version__ = "0.1.0"
