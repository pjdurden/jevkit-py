"""Comparing two answers to the same request.

TypeSafe's docs warn that ``jev-latest`` moves when a new version ships, and
that confidence thresholds tuned against one version do not automatically hold
on the next. This module answers the question that warning implies: for a set of
requests you already care about, what actually changed?

Two kinds of change are tracked separately, because they mean different things:

- A **flip** is a changed decision. Your code takes a different branch. This is
  what breaks things.
- A **shift** is movement in the probability distribution with the same decision
  on top. Harmless on its own, but it is what moves an answer toward a threshold,
  so a large shift is an early warning even when nothing flipped yet.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from jevkit_core import Answer, Record, parse_answers

__all__ = ["QuestionDelta", "RecordDelta", "DriftReport", "compare_answers",
           "compare_records", "compare_sets"]


def total_variation(a: dict[str, float], b: dict[str, float]) -> float:
    """Total variation distance between two distributions, on 0..1.

    Outcomes present in one distribution and not the other count in full, which
    is what makes this meaningful when a new model version changes the option
    set. Distributions are not renormalized: if the API returns something that
    does not sum to 1, that is reported rather than hidden.
    """
    keys = set(a) | set(b)
    return sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys) / 2.0


@dataclass
class QuestionDelta:
    """What changed for one question between two runs."""

    question_id: str
    type: str
    before: Any
    after: Any
    distribution_shift: float
    confidence_before: float | None
    confidence_after: float | None
    score_before: float | None = None
    score_after: float | None = None

    @property
    def flipped(self) -> bool:
        """The selected outcome changed."""
        return self.before != self.after

    @property
    def confidence_delta(self) -> float | None:
        if self.confidence_before is None or self.confidence_after is None:
            return None
        return self.confidence_after - self.confidence_before

    def describe(self) -> str:
        # JSON rendering rather than repr(), so the Python and JavaScript CLIs
        # emit byte-identical output. repr() quotes with ', JSON with ".
        if self.flipped:
            head = (f"{self.question_id}: FLIP {json.dumps(self.before)} -> "
                    f"{json.dumps(self.after)}")
        else:
            head = f"{self.question_id}: stable ({json.dumps(self.before)})"
        parts = [f"shift={self.distribution_shift:.3f}"]
        delta = self.confidence_delta
        if delta is not None:
            parts.append(f"conf {self.confidence_before:.3f} -> "
                         f"{self.confidence_after:.3f} ({delta:+.3f})")
        return f"{head}  [{', '.join(parts)}]"


@dataclass
class RecordDelta:
    """What changed for one request between two runs."""

    request_id: str
    model_before: str
    model_after: str
    questions: list[QuestionDelta] = field(default_factory=list)

    @property
    def flips(self) -> list[QuestionDelta]:
        return [q for q in self.questions if q.flipped]

    @property
    def max_shift(self) -> float:
        return max((q.distribution_shift for q in self.questions), default=0.0)


def compare_answers(qid: str, before: Answer, after: Answer) -> QuestionDelta:
    if before.type != after.type:
        raise ValueError(
            f"question {qid!r}: type changed from {before.type!r} to {after.type!r}. "
            f"These are not the same question and cannot be compared."
        )
    return QuestionDelta(
        question_id=qid,
        type=before.type,
        before=before.predicted(),
        after=after.predicted(),
        distribution_shift=total_variation(before.probabilities, after.probabilities),
        confidence_before=before.confidence,
        confidence_after=after.confidence,
        score_before=before.score,
        score_after=after.score,
    )


def compare_records(before: Record, after: Record) -> RecordDelta:
    if before.request_id != after.request_id:
        raise ValueError(
            "records describe different requests and cannot be compared "
            f"({before.request_id} vs {after.request_id})"
        )
    old = parse_answers(before.answers)
    new = parse_answers(after.answers)

    missing = sorted(set(old) - set(new))
    added = sorted(set(new) - set(old))
    if missing or added:
        detail = []
        if missing:
            detail.append(f"missing in the new run: {', '.join(missing)}")
        if added:
            detail.append(f"only in the new run: {', '.join(added)}")
        raise ValueError(
            f"answer sets differ for request {before.request_id}: {'; '.join(detail)}"
        )

    return RecordDelta(
        request_id=before.request_id,
        model_before=before.model,
        model_after=after.model,
        questions=[compare_answers(qid, old[qid], new[qid]) for qid in sorted(old)],
    )


@dataclass
class DriftReport:
    """Drift across a whole golden set."""

    deltas: list[RecordDelta] = field(default_factory=list)
    unmatched_before: list[str] = field(default_factory=list)
    unmatched_after: list[str] = field(default_factory=list)

    @property
    def total_questions(self) -> int:
        return sum(len(d.questions) for d in self.deltas)

    @property
    def flips(self) -> list[tuple[str, QuestionDelta]]:
        return [(d.request_id, q) for d in self.deltas for q in d.flips]

    @property
    def flip_rate(self) -> float:
        total = self.total_questions
        return len(self.flips) / total if total else 0.0

    @property
    def max_shift(self) -> float:
        return max((d.max_shift for d in self.deltas), default=0.0)

    @property
    def mean_shift(self) -> float:
        shifts = [q.distribution_shift for d in self.deltas for q in d.questions]
        return sum(shifts) / len(shifts) if shifts else 0.0

    def models(self) -> tuple[set[str], set[str]]:
        return ({d.model_before for d in self.deltas}, {d.model_after for d in self.deltas})

    def summary(self) -> str:
        before, after = self.models()
        lines = [
            f"compared {len(self.deltas)} request(s), {self.total_questions} question(s)",
            f"  {', '.join(sorted(before)) or '?'} -> {', '.join(sorted(after)) or '?'}",
            f"  flips:      {len(self.flips)} ({self.flip_rate:.1%})",
            f"  mean shift: {self.mean_shift:.4f}",
            f"  max shift:  {self.max_shift:.4f}",
        ]
        if self.unmatched_before:
            lines.append(f"  {len(self.unmatched_before)} record(s) in the baseline had no "
                         f"counterpart and were not compared")
        if self.unmatched_after:
            lines.append(f"  {len(self.unmatched_after)} record(s) in the new run had no "
                         f"counterpart and were not compared")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "requests": len(self.deltas),
            "questions": self.total_questions,
            "flips": len(self.flips),
            "flip_rate": self.flip_rate,
            "mean_shift": self.mean_shift,
            "max_shift": self.max_shift,
            "unmatched_before": self.unmatched_before,
            "unmatched_after": self.unmatched_after,
            "details": [
                {
                    "request_id": rid,
                    "question_id": q.question_id,
                    "before": q.before,
                    "after": q.after,
                    "distribution_shift": q.distribution_shift,
                    "confidence_delta": q.confidence_delta,
                }
                for rid, q in self.flips
            ],
        }


def compare_sets(before: list[Record], after: list[Record]) -> DriftReport:
    """Pair two sets of records by request id and compare each pair.

    Records are paired on ``request_id``, the digest of state plus questions with
    the model excluded, which is exactly what makes a baseline comparable to its
    replay on a different model version.
    """
    old = {r.request_id: r for r in before}
    new = {r.request_id: r for r in after}
    shared = sorted(set(old) & set(new))
    return DriftReport(
        deltas=[compare_records(old[rid], new[rid]) for rid in shared],
        unmatched_before=sorted(set(old) - set(new)),
        unmatched_after=sorted(set(new) - set(old)),
    )
