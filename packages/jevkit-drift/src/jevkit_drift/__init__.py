"""Detect when a new Jev model version changes decisions you depend on.

TypeSafe's docs warn that ``jev-latest`` moves and that tuned confidence
thresholds move with it. This package replays a golden set against a new version
and reports what actually changed, separating flipped decisions from probability
shifts that have not flipped anything yet.
"""

from .compare import (DriftReport, QuestionDelta, RecordDelta, compare_answers,
                      compare_records, compare_sets, total_variation)
from .replay import replay

__version__ = "0.1.0"

__all__ = ["compare_sets", "compare_records", "compare_answers", "replay",
           "DriftReport", "RecordDelta", "QuestionDelta", "total_variation",
           "__version__"]
