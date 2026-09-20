"""Reading and writing `.jevl` records. See docs/specs/record-format-v1.md."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, IO, Iterable, Iterator

from .canonical import record_id, request_id

__all__ = ["Record", "FORMAT_VERSION", "RecordFormatError",
           "read_records", "write_records", "append_record", "load_cassette"]

FORMAT_VERSION = 1

_REQUIRED = ("v", "id", "ts", "model", "request", "answers")


class RecordFormatError(ValueError):
    """A `.jevl` line is malformed, or its version is unreadable."""


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class Record:
    model: str
    state: Any
    questions: dict[str, Any]
    answers: dict[str, Any]
    id: str = ""
    ts: str = field(default_factory=_utcnow)
    usage: dict[str, Any] | None = None
    label: dict[str, Any] | None = None
    tags: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self.id:
            self.id = record_id(self.model, self.state, self.questions)

    @property
    def request_id(self) -> str:
        """Model-independent digest, used to pair across model versions."""
        return request_id(self.state, self.questions)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "v": FORMAT_VERSION,
            "id": self.id,
            "ts": self.ts,
            "model": self.model,
            "request": {"state": self.state, "questions": self.questions},
            "answers": self.answers,
        }
        if self.usage is not None:
            out["usage"] = self.usage
        if self.label is not None:
            out["label"] = self.label
        if self.tags:
            out["tags"] = list(self.tags)
        if self.meta:
            out["meta"] = dict(self.meta)
        # Unknown keys survive a read/write round trip.
        for k, v in self.extra.items():
            out.setdefault(k, v)
        return out

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, source: str = "<memory>",
                  line_no: int = 0) -> "Record":
        where = f"{source}:{line_no}" if line_no else source
        if not isinstance(raw, dict):
            raise RecordFormatError(f"{where}: record must be a JSON object")

        version = raw.get("v")
        if version is None:
            raise RecordFormatError(f"{where}: missing required key 'v'")
        if not isinstance(version, int):
            raise RecordFormatError(f"{where}: 'v' must be an integer, got {version!r}")
        if version > FORMAT_VERSION:
            raise RecordFormatError(
                f"{where}: record format v{version} is newer than this reader "
                f"(v{FORMAT_VERSION}). Upgrade jevkit-core rather than guessing."
            )

        missing = [k for k in _REQUIRED if k not in raw]
        if missing:
            raise RecordFormatError(f"{where}: missing required key(s): {', '.join(missing)}")

        request = raw["request"]
        if not isinstance(request, dict) or "questions" not in request:
            raise RecordFormatError(f"{where}: 'request' must be an object with 'questions'")

        known = set(_REQUIRED) | {"usage", "label", "tags", "meta"}
        return cls(
            model=raw["model"],
            state=request.get("state"),
            questions=request["questions"],
            answers=raw["answers"],
            id=raw["id"],
            ts=raw["ts"],
            usage=raw.get("usage"),
            label=raw.get("label"),
            tags=list(raw.get("tags") or []),
            meta=dict(raw.get("meta") or {}),
            extra={k: v for k, v in raw.items() if k not in known},
        )


def read_records(path: str | os.PathLike[str] | IO[str]) -> Iterator[Record]:
    """Yield records from a `.jevl` file.

    A line that fails to parse raises. Skipping silently would let a truncated
    golden set look like a passing one.
    """
    if hasattr(path, "read"):
        yield from _read_stream(path, getattr(path, "name", "<stream>"))  # type: ignore[arg-type]
        return
    with open(path, "r", encoding="utf-8") as fh:
        yield from _read_stream(fh, str(path))


def _read_stream(fh: IO[str], source: str) -> Iterator[Record]:
    for line_no, line in enumerate(fh, start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            raw = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise RecordFormatError(f"{source}:{line_no}: invalid JSON: {exc}") from exc
        yield Record.from_dict(raw, source=source, line_no=line_no)


def write_records(path: str | os.PathLike[str], records: Iterable[Record]) -> int:
    count = 0
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def append_record(path: str | os.PathLike[str], record: Record) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def load_cassette(path: str | os.PathLike[str]) -> dict[str, Record]:
    """Map record id -> Record, last occurrence winning.

    Re-recording a request appends rather than rewrites, so the last line for
    an id is the current answer.
    """
    table: dict[str, Record] = {}
    for record in read_records(path):
        table[record.id] = record
    return table
