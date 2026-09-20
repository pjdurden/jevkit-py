"""A uniform view of a jev question.

Questions reach jevkit in three shapes: plain dicts as sent to the HTTP API,
SDK objects (``Choice``/``Score``/``Noul``), and whatever a user's own helper
produces. Every jevkit tool wants the same few facts out of them, so they are
normalized once here rather than re-sniffed in each package.

``instructions`` and ``criteria`` both accept JSON structure, not just
strings, so the normalized form keeps the raw value and exposes the flattened
text separately for anything doing textual analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

__all__ = ["Question", "QuestionType", "normalize_question", "normalize_questions",
           "flatten_text"]

QuestionType = str  # "choice" | "score" | "noul"

_VALID_TYPES = ("choice", "score", "noul")


def flatten_text(value: Any) -> str:
    """Collapse a string / dict / list into one string for textual analysis.

    Dict keys are included: in a Choice, the keys are the option names and
    carry meaning the model sees.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        parts: list[str] = []
        for k, v in value.items():
            parts.append(str(k))
            parts.append(flatten_text(v))
        return " ".join(p for p in parts if p)
    if isinstance(value, (list, tuple)):
        return " ".join(p for p in (flatten_text(v) for v in value) if p)
    return str(value)


@dataclass
class Question:
    """Normalized question. ``raw`` is always the untouched original."""

    id: str
    type: QuestionType
    instructions: Any
    criteria: Any
    raw: Any = field(repr=False, default=None)

    @property
    def instructions_text(self) -> str:
        return flatten_text(self.instructions)

    @property
    def criteria_text(self) -> str:
        return flatten_text(self.criteria)

    @property
    def text(self) -> str:
        """Everything the model reads, as one string."""
        return f"{self.instructions_text} {self.criteria_text}".strip()

    @property
    def options(self) -> list[str]:
        """Choice option keys, or Score level descriptions. Empty otherwise."""
        if self.type == "choice" and isinstance(self.criteria, dict):
            return [str(k) for k in self.criteria.keys()]
        if self.type == "score" and isinstance(self.criteria, (list, tuple)):
            return [flatten_text(v) for v in self.criteria]
        if self.type == "score" and isinstance(self.criteria, dict):
            return [flatten_text(v) for v in self.criteria.values()]
        return []

    def option_descriptions(self) -> Iterator[tuple[str, str]]:
        """(option key, description) pairs for a Choice."""
        if self.type == "choice" and isinstance(self.criteria, dict):
            for k, v in self.criteria.items():
                yield str(k), flatten_text(v)


def _get(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _infer_type(obj: Any) -> str | None:
    declared = _get(obj, "type")
    if isinstance(declared, str) and declared.lower() in _VALID_TYPES:
        return declared.lower()
    # SDK objects carry no "type" field; fall back to the class name.
    cls = type(obj).__name__.lower()
    if cls in _VALID_TYPES:
        return cls
    return None


def normalize_question(qid: str, obj: Any) -> Question:
    qtype = _infer_type(obj)
    if qtype is None:
        raise ValueError(
            f"question {qid!r}: cannot determine type. Expected a 'type' key of "
            f"{_VALID_TYPES}, or a Choice/Score/Noul object."
        )
    return Question(
        id=qid,
        type=qtype,
        instructions=_get(obj, "instructions"),
        criteria=_get(obj, "criteria"),
        raw=obj,
    )


def normalize_questions(questions: dict[str, Any]) -> list[Question]:
    return [normalize_question(qid, obj) for qid, obj in questions.items()]
