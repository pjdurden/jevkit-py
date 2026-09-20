"""Calibration metrics over labeled jev answers.

Calibration is Jev's central claim: the probabilities are meant to reflect
real-world frequencies, so that among answers given 0.8, about 80% are right.
TypeSafe measures this across groups of predictions and says plainly that it
does not guarantee any individual answer. This module is how you check the claim
on *your* data, which is the part nobody else can do for you.

Everything here is pure arithmetic over recorded answers. Nothing calls the API,
and there are no dependencies beyond the standard library: a calibration check
that requires a numpy build is a calibration check that does not get run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

__all__ = ["Observation", "Bin", "CalibrationReport", "reliability_bins",
           "expected_calibration_error", "maximum_calibration_error",
           "brier_score", "log_loss", "calibrate"]


@dataclass(frozen=True)
class Observation:
    """One labeled prediction.

    ``probability`` is the model's stated probability for the outcome it chose;
    ``correct`` is whether that outcome matched the label.
    """

    probability: float
    correct: bool
    question_id: str = ""
    request_id: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(f"probability must be in [0, 1], got {self.probability}")


@dataclass
class Bin:
    """One bucket of a reliability diagram."""

    lower: float
    upper: float
    observations: list[Observation] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.observations)

    @property
    def mean_probability(self) -> float:
        """What the model claimed, on average."""
        if not self.observations:
            return 0.0
        return sum(o.probability for o in self.observations) / self.count

    @property
    def accuracy(self) -> float:
        """What actually happened."""
        if not self.observations:
            return 0.0
        return sum(1 for o in self.observations if o.correct) / self.count

    @property
    def gap(self) -> float:
        """Claimed minus observed. Positive means overconfident."""
        return self.mean_probability - self.accuracy

    def label(self) -> str:
        return f"[{self.lower:.2f}, {self.upper:.2f}{']' if self.upper == 1.0 else ')'}"


def reliability_bins(observations: Sequence[Observation], n_bins: int = 10) -> list[Bin]:
    """Bucket observations into equal-width probability bins.

    Equal-width rather than equal-count, because the question being asked is
    "when the model says 0.9, is it right 90% of the time", and that question is
    about fixed probability ranges. Empty bins are kept so the diagram does not
    silently mislead about coverage.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")
    edges = [i / n_bins for i in range(n_bins + 1)]
    bins = [Bin(lower=edges[i], upper=edges[i + 1]) for i in range(n_bins)]
    for obs in observations:
        # The top bin is closed so that a probability of exactly 1.0 lands in it.
        index = min(int(obs.probability * n_bins), n_bins - 1)
        bins[index].observations.append(obs)
    return bins


def expected_calibration_error(observations: Sequence[Observation], n_bins: int = 10) -> float:
    """ECE: average gap between claimed and observed, weighted by bin population.

    0 is perfect. Note that ECE depends on the bin count, so compare ECE values
    only when they were computed with the same ``n_bins``.
    """
    if not observations:
        return 0.0
    total = len(observations)
    return sum(
        (b.count / total) * abs(b.gap) for b in reliability_bins(observations, n_bins) if b.count
    )


def maximum_calibration_error(observations: Sequence[Observation], n_bins: int = 10) -> float:
    """MCE: the worst gap in any populated bin.

    ECE can look healthy while one region is badly wrong. MCE is what catches
    that, and it is the number that matters when a single region is where your
    high-stakes decisions live.
    """
    if not observations:
        return 0.0
    return max((abs(b.gap) for b in reliability_bins(observations, n_bins) if b.count), default=0.0)


def brier_score(observations: Sequence[Observation]) -> float:
    """Mean squared error between probability and outcome. Lower is better.

    Unlike ECE this is a proper scoring rule: it rewards being both calibrated
    and decisive, so a model that always says 0.5 scores poorly even though it
    is perfectly calibrated.
    """
    if not observations:
        return 0.0
    return sum((o.probability - (1.0 if o.correct else 0.0)) ** 2 for o in observations) / len(
        observations
    )


def log_loss(observations: Sequence[Observation], eps: float = 1e-15) -> float:
    """Mean negative log likelihood. Lower is better.

    Punishes confident mistakes far harder than Brier does. ``eps`` clamps the
    probability away from 0 and 1, since an unclamped confident miss is infinite
    and would swamp every other observation.
    """
    if not observations:
        return 0.0
    total = 0.0
    for o in observations:
        p = min(max(o.probability, eps), 1.0 - eps)
        total -= math.log(p) if o.correct else math.log(1.0 - p)
    return total / len(observations)


@dataclass
class CalibrationReport:
    observations: list[Observation]
    n_bins: int = 10

    @property
    def count(self) -> int:
        return len(self.observations)

    @property
    def accuracy(self) -> float:
        if not self.observations:
            return 0.0
        return sum(1 for o in self.observations if o.correct) / self.count

    @property
    def mean_probability(self) -> float:
        if not self.observations:
            return 0.0
        return sum(o.probability for o in self.observations) / self.count

    @property
    def bins(self) -> list[Bin]:
        return reliability_bins(self.observations, self.n_bins)

    @property
    def ece(self) -> float:
        return expected_calibration_error(self.observations, self.n_bins)

    @property
    def mce(self) -> float:
        return maximum_calibration_error(self.observations, self.n_bins)

    @property
    def brier(self) -> float:
        return brier_score(self.observations)

    @property
    def log_loss(self) -> float:
        return log_loss(self.observations)

    @property
    def overconfident(self) -> bool:
        """Claimed probability exceeds observed accuracy overall."""
        return self.mean_probability > self.accuracy

    def diagram(self, width: int = 28) -> str:
        """A text reliability diagram.

        Each row shows a bin's claimed probability against what actually
        happened, so a miscalibrated region is visible without plotting.
        """
        lines = [
            f"{'bin':<14} {'n':>5} {'claimed':>8} {'actual':>8} {'gap':>7}",
            "-" * 46,
        ]
        for b in self.bins:
            if not b.count:
                lines.append(f"{b.label():<14} {0:>5} {'-':>8} {'-':>8} {'-':>7}")
                continue
            bar_len = int(b.accuracy * width)
            marker = int(b.mean_probability * width)
            cells = ["#" if i < bar_len else " " for i in range(width)]
            if 0 <= marker < width:
                cells[marker] = "|" if cells[marker] == " " else "+"
            lines.append(
                f"{b.label():<14} {b.count:>5} {b.mean_probability:>8.3f} "
                f"{b.accuracy:>8.3f} {b.gap:>+7.3f}  {''.join(cells)}"
            )
        lines.append("")
        lines.append("  # observed accuracy, | claimed probability, + both")
        return "\n".join(lines)

    def summary(self) -> str:
        if not self.count:
            return "no labeled observations"
        direction = "overconfident" if self.overconfident else "underconfident"
        return "\n".join([
            f"observations: {self.count}",
            f"accuracy:     {self.accuracy:.4f}",
            f"mean claimed: {self.mean_probability:.4f}  ({direction} "
            f"by {abs(self.mean_probability - self.accuracy):.4f})",
            f"ECE:          {self.ece:.4f}  (over {self.n_bins} bins)",
            f"MCE:          {self.mce:.4f}",
            f"Brier:        {self.brier:.4f}",
            f"log loss:     {self.log_loss:.4f}",
        ])

    def to_dict(self) -> dict[str, object]:
        return {
            "count": self.count,
            "accuracy": self.accuracy,
            "mean_probability": self.mean_probability,
            "overconfident": self.overconfident,
            "ece": self.ece,
            "mce": self.mce,
            "brier": self.brier,
            "log_loss": self.log_loss,
            "n_bins": self.n_bins,
            "bins": [
                {
                    "lower": b.lower, "upper": b.upper, "count": b.count,
                    "mean_probability": b.mean_probability, "accuracy": b.accuracy,
                    "gap": b.gap,
                }
                for b in self.bins
            ],
        }


def calibrate(observations: Iterable[Observation], n_bins: int = 10) -> CalibrationReport:
    return CalibrationReport(observations=list(observations), n_bins=n_bins)
