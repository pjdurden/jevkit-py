"""Replaying a golden set against a model."""

from __future__ import annotations

from typing import Any, Callable, Iterable

from jevkit_core import Record

__all__ = ["replay", "SystemOneCallable"]

SystemOneCallable = Callable[[Any, dict[str, Any]], Any]


def _extract(response: Any) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
    """Pull (model, answers, usage) out of an SDK response or a plain dict."""
    if isinstance(response, dict):
        model = response.get("model", "")
        answers = response.get("answers", {})
        usage = response.get("usage")
    else:
        model = getattr(response, "model", "")
        answers = getattr(response, "answers", {})
        usage = getattr(response, "usage", None)

    plain: dict[str, Any] = {}
    for qid, answer in (answers or {}).items():
        if isinstance(answer, dict):
            plain[qid] = answer
        elif hasattr(answer, "model_dump"):
            plain[qid] = answer.model_dump()
        elif hasattr(answer, "__dict__"):
            plain[qid] = {k: v for k, v in vars(answer).items() if not k.startswith("_")}
        else:
            raise TypeError(f"answer {qid!r}: cannot convert {type(answer).__name__} to a dict")

    if usage is not None and not isinstance(usage, dict):
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump()
        elif hasattr(usage, "__dict__"):
            usage = {k: v for k, v in vars(usage).items() if not k.startswith("_")}

    return str(model), plain, usage


def replay(
    records: Iterable[Record],
    system_one: SystemOneCallable,
    *,
    tags: Iterable[str] | None = None,
) -> list[Record]:
    """Re-send each record's request and return the new answers as new records.

    ``system_one`` is any callable taking ``(state, questions)`` and returning a
    response, which is the shape both official SDKs already expose. Labels and
    tags carry over from the baseline so a replayed set stays scoreable.
    """
    extra_tags = list(tags or [])
    out: list[Record] = []
    for record in records:
        response = system_one(record.state, record.questions)
        model, answers, usage = _extract(response)
        out.append(Record(
            model=model or record.model,
            state=record.state,
            questions=record.questions,
            answers=answers,
            usage=usage,
            label=record.label,
            tags=list(record.tags) + extra_tags,
            meta={**record.meta, "replayed_from": record.id},
        ))
    return out
