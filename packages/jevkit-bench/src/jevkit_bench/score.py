"""Scoring a labeled `.jevl` suite, and comparing two runs of it.

The question this answers is the one you actually have to defend: for this task,
on my data, is Jev good enough, and what does it cost? That needs accuracy and
cost and latency together, because any one of them alone is easy to win.

Costs are computed from a price per million input tokens, defaulting to Jev's
published $0.042. Output tokens are free on Jev and are reported but not billed.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from jevkit_core import Record, parse_answers

__all__ = ["QuestionResult", "SuiteResult", "score_records", "compare_suites",
           "SuiteComparison", "PRICE_PER_MTOK"]

PRICE_PER_MTOK = 0.042


@dataclass(frozen=True)
class QuestionResult:
    """One scored question from one record."""

    request_id: str
    question_id: str
    type: str
    predicted: Any
    label: Any
    correct: bool
    probability: float
    confidence: float | None
    tags: tuple[str, ...] = ()


@dataclass
class SuiteResult:
    """Everything scored from one run of a suite."""

    results: list[QuestionResult] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    records: int = 0
    unlabeled: int = 0
    models: set[str] = field(default_factory=set)
    price_per_mtok: float = PRICE_PER_MTOK

    @property
    def count(self) -> int:
        return len(self.results)

    @property
    def correct(self) -> int:
        return sum(1 for r in self.results if r.correct)

    @property
    def accuracy(self) -> float:
        return self.correct / self.count if self.count else 0.0

    @property
    def cost(self) -> float:
        """Input-token cost in dollars. Jev bills input only."""
        return self.input_tokens / 1_000_000 * self.price_per_mtok

    @property
    def cost_per_question(self) -> float:
        return self.cost / self.count if self.count else 0.0

    def by_tag(self) -> dict[str, tuple[int, float]]:
        """tag -> (n, accuracy). A result with several tags counts under each."""
        buckets: dict[str, list[QuestionResult]] = defaultdict(list)
        for r in self.results:
            for tag in r.tags:
                buckets[tag].append(r)
        return {
            tag: (len(rs), sum(1 for r in rs if r.correct) / len(rs))
            for tag, rs in sorted(buckets.items())
        }

    def by_question(self) -> dict[str, tuple[int, float]]:
        """question id -> (n, accuracy). Finds the one question dragging the suite."""
        buckets: dict[str, list[QuestionResult]] = defaultdict(list)
        for r in self.results:
            buckets[r.question_id].append(r)
        return {
            qid: (len(rs), sum(1 for r in rs if r.correct) / len(rs))
            for qid, rs in sorted(buckets.items())
        }

    def failures(self) -> list[QuestionResult]:
        """Wrong answers, most confident first: the most interesting bugs."""
        return sorted(
            (r for r in self.results if not r.correct),
            key=lambda r: -r.probability,
        )

    def summary(self) -> str:
        lines = [
            f"records:   {self.records}"
            + (f" ({self.unlabeled} unlabeled, skipped)" if self.unlabeled else ""),
            f"scored:    {self.count} question(s)",
            f"model(s):  {', '.join(sorted(self.models)) or '?'}",
            f"accuracy:  {self.accuracy:.4f}  ({self.correct}/{self.count})",
            f"tokens:    {self.input_tokens:,} in, {self.output_tokens:,} out (out is free)",
            f"cost:      ${self.cost:.6f}  (${self.cost_per_question:.8f}/question "
            f"at ${self.price_per_mtok}/Mtok)",
        ]
        tags = self.by_tag()
        if tags:
            lines.append("by tag:")
            for tag, (n, acc) in tags.items():
                lines.append(f"  {tag:<24} {acc:.4f}  (n={n})")
        questions = self.by_question()
        if len(questions) > 1:
            lines.append("by question:")
            for qid, (n, acc) in questions.items():
                lines.append(f"  {qid:<24} {acc:.4f}  (n={n})")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": self.records,
            "unlabeled": self.unlabeled,
            "scored": self.count,
            "models": sorted(self.models),
            "accuracy": self.accuracy,
            "correct": self.correct,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost": self.cost,
            "by_tag": {k: {"n": n, "accuracy": a} for k, (n, a) in self.by_tag().items()},
            "by_question": {k: {"n": n, "accuracy": a} for k, (n, a) in self.by_question().items()},
        }


def score_records(
    records: Iterable[Record],
    *,
    question_ids: Iterable[str] | None = None,
    price_per_mtok: float = PRICE_PER_MTOK,
) -> SuiteResult:
    """Score every labeled answer in a suite.

    Records with no ``label`` are counted and skipped rather than silently
    dropped, so a suite that quietly lost its labels is visible in the summary
    instead of showing a suspiciously perfect score over three records.
    """
    wanted = set(question_ids) if question_ids is not None else None
    suite = SuiteResult(price_per_mtok=price_per_mtok)

    for record in records:
        suite.records += 1
        suite.models.add(record.model)
        usage = record.usage or {}
        suite.input_tokens += int(usage.get("input_tokens") or 0)
        suite.output_tokens += int(usage.get("output_tokens") or 0)

        if not record.label:
            suite.unlabeled += 1
            continue

        answers = parse_answers(record.answers)
        for qid, label in record.label.items():
            if wanted is not None and qid not in wanted:
                continue
            answer = answers.get(qid)
            if answer is None:
                continue
            suite.results.append(QuestionResult(
                request_id=record.request_id,
                question_id=qid,
                type=answer.type,
                predicted=answer.predicted(),
                label=label,
                correct=answer.is_correct(label),
                probability=answer.top_probability,
                confidence=answer.confidence,
                tags=tuple(record.tags),
            ))
    return suite


@dataclass
class SuiteComparison:
    """Two runs of the same suite, side by side."""

    baseline: SuiteResult
    candidate: SuiteResult
    regressions: list[tuple[str, str]] = field(default_factory=list)
    fixes: list[tuple[str, str]] = field(default_factory=list)

    @property
    def accuracy_delta(self) -> float:
        return self.candidate.accuracy - self.baseline.accuracy

    @property
    def cost_delta(self) -> float:
        return self.candidate.cost - self.baseline.cost

    def summary(self) -> str:
        return "\n".join([
            f"accuracy:  {self.baseline.accuracy:.4f} -> {self.candidate.accuracy:.4f} "
            f"({self.accuracy_delta:+.4f})",
            f"cost:      ${self.baseline.cost:.6f} -> ${self.candidate.cost:.6f} "
            f"({self.cost_delta:+.6f})",
            f"regressed: {len(self.regressions)} (right before, wrong now)",
            f"fixed:     {len(self.fixes)} (wrong before, right now)",
        ])


def compare_suites(baseline: SuiteResult, candidate: SuiteResult) -> SuiteComparison:
    """Compare two scored runs by (request, question).

    Aggregate accuracy can hold steady while the *set* of things you get right
    churns underneath, which matters when a specific case is the one you promised
    someone would work. Regressions and fixes are tracked individually for that
    reason.
    """
    def index(suite: SuiteResult) -> dict[tuple[str, str], QuestionResult]:
        return {(r.request_id, r.question_id): r for r in suite.results}

    before, after = index(baseline), index(candidate)
    shared = set(before) & set(after)
    return SuiteComparison(
        baseline=baseline,
        candidate=candidate,
        regressions=sorted(k for k in shared if before[k].correct and not after[k].correct),
        fixes=sorted(k for k in shared if not before[k].correct and after[k].correct),
    )
