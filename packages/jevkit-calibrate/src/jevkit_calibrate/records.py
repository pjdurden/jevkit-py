"""Turning labeled `.jevl` records into calibration observations."""

from __future__ import annotations

from typing import Iterable

from jevkit_core import Record, parse_answers

from .metrics import Observation

__all__ = ["observations_from_records", "SKIPPED_UNLABELED"]

SKIPPED_UNLABELED = "unlabeled"


def observations_from_records(
    records: Iterable[Record],
    *,
    question_ids: Iterable[str] | None = None,
    use_confidence: bool = False,
) -> list[Observation]:
    """Extract one observation per labeled answer.

    Records with no ``label`` are skipped: calibration needs ground truth, and
    silently treating an unlabeled record as correct or incorrect would poison
    every number downstream.

    ``use_confidence`` picks which quantity is being calibrated. The default,
    ``False``, uses the probability mass on the chosen outcome, which is what
    "when it says 0.8, is it right 80% of the time" means. Setting it to ``True``
    calibrates the API's ``confidence`` statistic instead, which is a different
    question and answers whether your *routing* threshold is well placed. Nouls
    have no confidence, so they fall back to the probability either way.
    """
    wanted = set(question_ids) if question_ids is not None else None
    out: list[Observation] = []

    for record in records:
        if not record.label:
            continue
        answers = parse_answers(record.answers)
        for qid, label in record.label.items():
            if wanted is not None and qid not in wanted:
                continue
            answer = answers.get(qid)
            if answer is None:
                continue
            correct = answer.is_correct(label)
            if use_confidence and answer.confidence is not None:
                probability = answer.confidence
            else:
                probability = answer.top_probability
            out.append(Observation(
                probability=probability,
                correct=correct,
                question_id=qid,
                request_id=record.request_id,
            ))
    return out
