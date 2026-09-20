"""A uniform view of a jev answer, and how it compares to a ground-truth label.

`drift`, `calibrate`, and `bench` all need the same three things from an answer:
what it predicted, how much probability sat on a given outcome, and whether it
matched a label. Those live here so the three packages agree on the definitions
rather than each inventing its own.

The vocabulary is deliberately narrow:

- **predicted** is the outcome the answer selects. For a Choice it is the option
  key. For a Noul it is ``True`` when ``noul`` clears the threshold. For a Score
  it is the index of the most probable level, which is *not* the same as
  rounding ``score``.
- **confidence** is what the API returned, and Nouls do not have one. A Noul's
  distance from 0.5 is a different quantity and is exposed separately as
  ``decisiveness`` rather than pretending it is the same number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["Answer", "parse_answer", "parse_answers", "NOUL_THRESHOLD"]

NOUL_THRESHOLD = 0.5


@dataclass
class Answer:
    """Normalized view over one answer from the API."""

    id: str
    type: str
    raw: dict[str, Any]

    # -- what it said ------------------------------------------------------

    @property
    def probabilities(self) -> dict[str, float]:
        """Outcome -> probability.

        Choice returns its options. Score returns its level indices as strings.
        Noul has no distribution from the API, so the two-outcome distribution
        it implies is synthesized here.
        """
        if self.type == "noul":
            p = float(self.raw.get("noul", 0.0))
            return {"true": p, "false": 1.0 - p}
        probs = self.raw.get("probabilities") or {}
        return {str(k): float(v) for k, v in probs.items()}

    @property
    def confidence(self) -> float | None:
        """As returned by the API. ``None`` for a Noul, which carries none."""
        value = self.raw.get("confidence")
        return None if value is None else float(value)

    @property
    def decisiveness(self) -> float:
        """How far from maximally uncertain this answer is, on 0..1.

        For a Noul this is ``|noul - 0.5| * 2``. This is *not* confidence and is
        not comparable to the API's confidence across question types; it exists
        so Nouls can be thresholded on something with a defined meaning.
        """
        if self.type == "noul":
            return abs(float(self.raw.get("noul", 0.0)) - NOUL_THRESHOLD) * 2.0
        conf = self.confidence
        return 0.0 if conf is None else conf

    def predicted(self, *, noul_threshold: float = NOUL_THRESHOLD) -> Any:
        """The outcome this answer selects."""
        if self.type == "noul":
            return float(self.raw.get("noul", 0.0)) >= noul_threshold
        if self.type == "choice":
            return self.raw.get("choice")
        if self.type == "score":
            probs = self.probabilities
            if not probs:
                return None
            best = max(probs, key=lambda k: probs[k])
            try:
                return int(best)
            except ValueError:
                return best
        return None

    @property
    def score(self) -> float | None:
        """The probability-weighted score, for a Score answer."""
        if self.type != "score":
            return None
        value = self.raw.get("score")
        return None if value is None else float(value)

    # -- how it compares to a label ---------------------------------------

    def _label_key(self, label: Any) -> str:
        if self.type == "noul":
            return "true" if bool(label) else "false"
        return str(label)

    def probability_of(self, label: Any) -> float:
        """Probability this answer assigned to ``label``.

        Returns 0.0 for an outcome the question never offered, which is the
        honest reading: the model could not have selected it.
        """
        return self.probabilities.get(self._label_key(label), 0.0)

    def is_correct(self, label: Any, *, noul_threshold: float = NOUL_THRESHOLD) -> bool:
        predicted = self.predicted(noul_threshold=noul_threshold)
        if self.type == "noul":
            return bool(predicted) is bool(label)
        if self.type == "score":
            try:
                return int(predicted) == int(label)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return predicted == label
        return predicted == label

    @property
    def top_probability(self) -> float:
        """Probability mass on the predicted outcome."""
        probs = self.probabilities
        return max(probs.values()) if probs else 0.0


def parse_answer(qid: str, raw: Any) -> Answer:
    if not isinstance(raw, dict):
        raise ValueError(f"answer {qid!r}: expected an object, got {type(raw).__name__}")
    atype = raw.get("type")
    if atype not in ("choice", "score", "noul"):
        # Infer from shape when the API response omits the discriminator.
        if "choice" in raw:
            atype = "choice"
        elif "noul" in raw:
            atype = "noul"
        elif "score" in raw:
            atype = "score"
        else:
            raise ValueError(
                f"answer {qid!r}: cannot determine type; expected a 'type' field or one "
                f"of 'choice'/'score'/'noul'"
            )
    return Answer(id=qid, type=atype, raw=raw)


def parse_answers(answers: dict[str, Any]) -> dict[str, Answer]:
    return {qid: parse_answer(qid, raw) for qid, raw in answers.items()}
