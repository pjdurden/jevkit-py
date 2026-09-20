"""Record and replay jev requests so tests do not hit the API.

A model call in a test is slow, costs money, needs a key in CI, and can change
its answer under you when the alias moves. The fix is the one VCR established
for HTTP: record real responses once, replay them forever, re-record on purpose.

A cassette is a `.jevl` file, the same format `drift`, `bench` and `calibrate`
read, so a recording made by your test suite is also a golden set you can replay
against the next model version.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from jevkit_core import Record, append_record, read_records, record_id

__all__ = ["Cassette", "CassetteMiss", "Mode"]

Mode = str  # "replay" | "record" | "auto" | "passthrough"

_MODES = ("replay", "record", "auto", "passthrough")


class CassetteMiss(LookupError):
    """A request was not on the cassette and the mode forbids recording."""


def _response_to_parts(response: Any) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
    """Pull (model, answers, usage) out of an SDK response or a plain dict."""
    if isinstance(response, dict):
        model, answers, usage = response.get("model", ""), response.get("answers", {}), response.get("usage")
    else:
        model = getattr(response, "model", "")
        answers = getattr(response, "answers", {})
        usage = getattr(response, "usage", None)

    def plain(value: Any) -> Any:
        if isinstance(value, dict):
            return value
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if hasattr(value, "__dict__"):
            return {k: v for k, v in vars(value).items() if not k.startswith("_")}
        return value

    return str(model), {qid: plain(a) for qid, a in (answers or {}).items()}, (
        plain(usage) if usage is not None else None
    )


class ReplayedAnswer(dict):
    """A replayed answer.

    Subclasses ``dict`` so ``answer["choice"]`` works, and mirrors the keys onto
    attributes so ``answer.choice`` works too. Test code written against either
    the SDK objects or the raw JSON keeps working without a shim.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(
                f"answer has no field {name!r}; recorded fields are "
                f"{', '.join(sorted(self)) or '(none)'}"
            ) from exc


class ReplayedResponse:
    """What a cassette hands back in place of a live API response."""

    def __init__(self, record: Record) -> None:
        self._record = record
        self.model = record.model
        self.answers = {qid: ReplayedAnswer(a) for qid, a in record.answers.items()}
        self.usage = record.usage
        self.id = record.id

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"ReplayedResponse(model={self.model!r}, answers={list(self.answers)})"


class Cassette:
    """A recorded set of jev requests, replayed by request digest.

    ``mode`` controls what happens on a miss:

    - ``replay``      raise ``CassetteMiss``. The right default for CI.
    - ``auto``        replay a hit, call through and append on a miss.
    - ``record``      always call through and append, ignoring existing entries.
    - ``passthrough`` never touch the cassette.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        system_one: Callable[..., Any] | None = None,
        *,
        mode: Mode = "replay",
        model: str = "jev-latest",
    ) -> None:
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
        self.path = Path(path)
        self.mode = mode
        self.model = model
        self._system_one = system_one
        self._entries = self._load() if self.path.exists() and mode != "record" else {}
        self.played: list[str] = []
        self.recorded: list[str] = []

    def _load(self) -> dict[str, Record]:
        """Index the file by *requested*-model digest.

        A record stores the model that actually answered, which is what the
        format requires: `jev-latest` is an alias and the response says
        `jev-1.13.0`. But a test asks for the alias, so indexing on the answering
        model would miss every lookup. The requested model is kept in `meta` at
        record time and used to rebuild the index here.

        Later entries win, so re-recording a request appends rather than
        requiring a rewrite.
        """
        entries: dict[str, Record] = {}
        for record in read_records(self.path):
            requested = str(record.meta.get("requested_model") or record.model)
            entries[record_id(requested, record.state, record.questions)] = record
        return entries

    # -- introspection -----------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def unplayed(self) -> list[str]:
        """Entries on the cassette that no test asked for.

        Usually means a test was deleted or a question was reworded, leaving a
        stale recording that will quietly rot.
        """
        return sorted(set(self._entries) - set(self.played))

    def contains(self, state: Any, questions: dict[str, Any], *, model: str | None = None) -> bool:
        return record_id(model or self.model, state, questions) in self._entries

    # -- the call ----------------------------------------------------------

    def system_one(self, state: Any, questions: dict[str, Any], **kwargs: Any) -> Any:
        """Drop-in for a client's ``system_one``.

        Signature matches what both official SDKs expose, so a cassette can be
        passed anywhere a client is expected in a test.
        """
        model = kwargs.pop("model", None) or self.model

        if self.mode == "passthrough":
            return self._call_through(state, questions, model=model, **kwargs)

        key = record_id(model, state, questions)

        if self.mode != "record":
            hit = self._entries.get(key)
            if hit is not None:
                self.played.append(key)
                return ReplayedResponse(hit)

        if self.mode == "replay":
            raise CassetteMiss(
                f"no recording for this request on {self.path}.\n"
                f"  digest: {key}\n"
                f"  Re-run with mode='auto' (or --jev-record) to record it, and commit the "
                f"updated cassette."
            )

        response = self._call_through(state, questions, model=model, **kwargs)
        actual_model, answers, usage = _response_to_parts(response)
        record = Record(
            model=actual_model or model,
            state=state,
            questions=questions,
            answers=answers,
            usage=usage,
            meta={"requested_model": model},
        )
        append_record(self.path, record)
        self._entries[key] = record
        # The live response is returned rather than the record: recording must
        # not change what the code under test sees.
        self.recorded.append(key)
        return response

    __call__ = system_one

    def _call_through(self, state: Any, questions: dict[str, Any], **kwargs: Any) -> Any:
        if self._system_one is None:
            raise CassetteMiss(
                f"cassette {self.path} is in mode {self.mode!r} and needs to call the API, "
                f"but no client was supplied. Pass system_one= when constructing it."
            )
        return self._system_one(state, questions, **kwargs)
