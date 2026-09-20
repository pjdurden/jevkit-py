"""Diagnostics produced by the linter."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

__all__ = ["Severity", "Diagnostic"]


class Severity(str, Enum):
    ERROR = "error"      # Will very likely produce wrong answers or be rejected.
    WARNING = "warning"  # A documented failure mode is in play.
    INFO = "info"        # Worth a look; may be intentional.

    def __str__(self) -> str:  # pragma: no cover - display only
        return self.value


_ORDER = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}


@dataclass(frozen=True)
class Diagnostic:
    code: str
    severity: Severity
    question_id: str | None
    message: str
    hint: str
    evidence: str = ""

    @property
    def sort_key(self) -> tuple[int, str, str]:
        return (_ORDER[self.severity], self.code, self.question_id or "")

    def format(self, *, color: bool = False) -> str:
        loc = self.question_id or "<request>"
        head = f"{loc}: {self.severity.value} [{self.code}] {self.message}"
        if color:
            tint = {"error": "\033[31m", "warning": "\033[33m", "info": "\033[36m"}
            head = f"{tint[self.severity.value]}{head}\033[0m"
        lines = [head]
        if self.evidence:
            lines.append(f"    found: {self.evidence}")
        lines.append(f"    hint:  {self.hint}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "question_id": self.question_id,
            "message": self.message,
            "hint": self.hint,
            "evidence": self.evidence,
        }
