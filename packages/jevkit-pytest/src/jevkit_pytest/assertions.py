"""Assertions over jev answers, with failure messages that say enough.

A bare ``assert answer.choice == "billing"`` tells you nothing about *how*
close the call was. These assertions print the distribution on failure, because
a 0.51/0.49 split and a 0.99/0.01 split are different bugs.
"""

from __future__ import annotations

from typing import Any

from jevkit_core import parse_answer

__all__ = ["assert_answer", "assert_confident", "describe_answer"]


def describe_answer(qid: str, raw: Any) -> str:
    answer = parse_answer(qid, raw)
    probs = ", ".join(
        f"{k}={v:.3f}" for k, v in sorted(answer.probabilities.items(), key=lambda kv: -kv[1])
    )
    parts = [f"predicted={answer.predicted()!r}"]
    if answer.confidence is not None:
        parts.append(f"confidence={answer.confidence:.3f}")
    if answer.score is not None:
        parts.append(f"score={answer.score:.3f}")
    return f"{qid}: {', '.join(parts)}\n    probabilities: {probs}"


def assert_answer(
    raw: Any,
    expected: Any,
    *,
    question_id: str = "answer",
    min_confidence: float | None = None,
    min_probability: float | None = None,
) -> None:
    """Assert an answer selected ``expected``, optionally with enough certainty."""
    answer = parse_answer(question_id, raw)
    predicted = answer.predicted()

    if not answer.is_correct(expected):
        raise AssertionError(
            f"expected {expected!r} but got {predicted!r}\n  {describe_answer(question_id, raw)}"
        )

    if min_probability is not None:
        actual = answer.probability_of(expected)
        if actual < min_probability:
            raise AssertionError(
                f"{expected!r} was selected but carried only {actual:.3f} probability, "
                f"below the required {min_probability:.3f}\n"
                f"  {describe_answer(question_id, raw)}"
            )

    if min_confidence is not None:
        if answer.confidence is None:
            raise AssertionError(
                f"min_confidence was given but a {answer.type} answer carries no "
                f"confidence. Use min_probability instead."
            )
        if answer.confidence < min_confidence:
            raise AssertionError(
                f"{expected!r} was selected but confidence was {answer.confidence:.3f}, "
                f"below the required {min_confidence:.3f}\n"
                f"  {describe_answer(question_id, raw)}"
            )


def assert_confident(raw: Any, minimum: float, *, question_id: str = "answer") -> None:
    """Assert an answer is decisive, without caring which way it went.

    Uses the API's confidence for Choice and Score. A Noul has none, so its
    distance from 0.5 is used and the message says so.
    """
    answer = parse_answer(question_id, raw)
    value = answer.decisiveness
    if value < minimum:
        quantity = "decisiveness (|noul - 0.5| * 2)" if answer.type == "noul" else "confidence"
        raise AssertionError(
            f"{quantity} was {value:.3f}, below the required {minimum:.3f}\n"
            f"  {describe_answer(question_id, raw)}"
        )
