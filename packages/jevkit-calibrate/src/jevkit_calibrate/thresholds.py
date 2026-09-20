"""Choosing a confidence threshold from labeled data instead of guessing.

TypeSafe's confidence page recommends three bands: act automatically, proceed
with caution, escalate. It also says plainly that where you draw those lines
depends on your domain and your data. This module draws them from the data.

The trade is always the same. Raising the threshold means acting on fewer cases
but being right more often on the ones you do act on. That is a curve, not a
number, so ``sweep`` returns the whole curve and the ``recommend_*`` functions
pick a point on it against a constraint you state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .metrics import Observation

__all__ = ["ThresholdPoint", "sweep", "recommend_for_accuracy", "recommend_for_coverage"]


@dataclass(frozen=True)
class ThresholdPoint:
    """What happens if you auto-handle everything at or above ``threshold``."""

    threshold: float
    covered: int
    total: int
    correct: int

    @property
    def coverage(self) -> float:
        """Fraction of cases handled automatically."""
        return self.covered / self.total if self.total else 0.0

    @property
    def accuracy(self) -> float:
        """Accuracy on the automatically handled cases.

        Defined as 1.0 when nothing is covered: a threshold that acts on nothing
        is never wrong. Read it together with ``coverage``, never alone.
        """
        return self.correct / self.covered if self.covered else 1.0

    @property
    def escalated(self) -> int:
        return self.total - self.covered

    @property
    def errors(self) -> int:
        """Wrong answers you acted on. Usually the number that actually costs."""
        return self.covered - self.correct


def sweep(observations: Sequence[Observation], steps: int = 101) -> list[ThresholdPoint]:
    """Coverage and accuracy at every threshold from 0 to 1.

    Uses the probability the model put on the outcome it chose, so this works
    for Choice and Score confidence and for a Noul's distance from 0.5 alike, as
    long as the caller is consistent about which it fed in.
    """
    if steps < 2:
        raise ValueError("steps must be at least 2")
    total = len(observations)
    points: list[ThresholdPoint] = []
    for i in range(steps):
        threshold = i / (steps - 1)
        covered = [o for o in observations if o.probability >= threshold]
        points.append(ThresholdPoint(
            threshold=threshold,
            covered=len(covered),
            total=total,
            correct=sum(1 for o in covered if o.correct),
        ))
    return points


def recommend_for_accuracy(
    observations: Sequence[Observation],
    target_accuracy: float,
    *,
    steps: int = 101,
) -> ThresholdPoint | None:
    """Lowest threshold reaching ``target_accuracy``, so coverage stays highest.

    Returns ``None`` when no threshold reaches the target, which is a real
    answer: it means this question cannot be automated at that accuracy and the
    honest move is to change the question rather than the threshold.
    """
    if not 0.0 <= target_accuracy <= 1.0:
        raise ValueError("target_accuracy must be in [0, 1]")
    candidates = [p for p in sweep(observations, steps) if p.covered and p.accuracy >= target_accuracy]
    return min(candidates, key=lambda p: p.threshold) if candidates else None


def recommend_for_coverage(
    observations: Sequence[Observation],
    min_coverage: float,
    *,
    steps: int = 101,
) -> ThresholdPoint | None:
    """Highest threshold still covering ``min_coverage``, so accuracy is best.

    The mirror of ``recommend_for_accuracy``: use it when throughput is the
    binding constraint and you want the most accurate threshold that still keeps
    enough volume out of the review queue.
    """
    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError("min_coverage must be in [0, 1]")
    candidates = [p for p in sweep(observations, steps) if p.coverage >= min_coverage]
    return max(candidates, key=lambda p: p.threshold) if candidates else None
