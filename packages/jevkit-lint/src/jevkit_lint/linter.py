"""The lint entry point."""

from __future__ import annotations

from typing import Any, Iterable

from jevkit_core import normalize_questions

from .diagnostic import Diagnostic, Severity
from .rules import LintContext, rules_for

__all__ = ["lint", "LintResult"]


class LintResult:
    """Diagnostics for one request, ordered most severe first."""

    def __init__(self, diagnostics: list[Diagnostic]) -> None:
        self.diagnostics = sorted(diagnostics, key=lambda d: d.sort_key)

    def __iter__(self):
        return iter(self.diagnostics)

    def __len__(self) -> int:
        return len(self.diagnostics)

    def __bool__(self) -> bool:
        return bool(self.diagnostics)

    def by_severity(self, severity: Severity) -> list[Diagnostic]:
        return [d for d in self.diagnostics if d.severity is severity]

    @property
    def errors(self) -> list[Diagnostic]:
        return self.by_severity(Severity.ERROR)

    @property
    def warnings(self) -> list[Diagnostic]:
        return self.by_severity(Severity.WARNING)

    @property
    def infos(self) -> list[Diagnostic]:
        return self.by_severity(Severity.INFO)

    @property
    def ok(self) -> bool:
        """True when nothing rose to an error."""
        return not self.errors

    def counts(self) -> dict[str, int]:
        return {
            "error": len(self.errors),
            "warning": len(self.warnings),
            "info": len(self.infos),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "counts": self.counts(),
            "diagnostics": [d.to_dict() for d in self.diagnostics],
        }

    def format(self, *, color: bool = False) -> str:
        if not self.diagnostics:
            return "No problems found."
        return "\n".join(d.format(color=color) for d in self.diagnostics)


def lint(
    questions: dict[str, Any],
    state: Any = "",
    *,
    select: Iterable[str] | None = None,
    ignore: Iterable[str] | None = None,
) -> LintResult:
    """Lint a jev request without calling the API.

    ``questions`` accepts SDK objects or plain dicts. ``state`` is optional: omit
    it to check the questions alone, though the budget rules can only report
    meaningfully when the real state is supplied.
    """
    normalized = normalize_questions(questions)
    ctx = LintContext.build(state, normalized)
    ignored = {c.upper() for c in (ignore or ())}

    found: list[Diagnostic] = []
    for rule in rules_for(select):
        if rule.code in ignored:
            continue
        found.extend(rule.check(ctx))
    return LintResult(found)
