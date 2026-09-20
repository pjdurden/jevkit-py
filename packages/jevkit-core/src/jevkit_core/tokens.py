"""Token budget estimation for jev requests.

Jev ingests the state once and evaluates every question against it, so two
separate budgets apply. Both are documented on the model card:

    64k  state + every question combined
    32k  state + the single longest question

These are *estimates*. jevkit deliberately does not bundle a tokenizer: the
dependency is heavy, the vendor does not publish which tokenizer Jev uses,
and a wrong tokenizer is more misleading than an honest approximation. The
estimator is intentionally conservative so that a request jevkit calls safe
is very unlikely to be rejected.
"""

from __future__ import annotations

from typing import Any

from .canonical import canonical_json

__all__ = ["estimate_tokens", "BudgetReport", "check_budget",
           "TOTAL_BUDGET", "STATE_BUDGET"]

TOTAL_BUDGET = 64_000
STATE_BUDGET = 32_000

# Bytes per token. Real-world English on BPE-family tokenizers runs ~4.0;
# 3.5 buys headroom for punctuation-dense JSON without being absurd.
_BYTES_PER_TOKEN = 3.5


def estimate_tokens(value: Any) -> int:
    """Conservative token estimate for any JSON-serializable value."""
    if isinstance(value, str):
        payload = value.encode("utf-8")
    else:
        payload = canonical_json(value)
    return max(1, int(len(payload) / _BYTES_PER_TOKEN) + 1)


class BudgetReport:
    """Where a request sits against both documented budgets."""

    __slots__ = ("state_tokens", "question_tokens", "longest_question_id")

    def __init__(
        self,
        state_tokens: int,
        question_tokens: dict[str, int],
        longest_question_id: str | None,
    ) -> None:
        self.state_tokens = state_tokens
        self.question_tokens = question_tokens
        self.longest_question_id = longest_question_id

    @property
    def total(self) -> int:
        return self.state_tokens + sum(self.question_tokens.values())

    @property
    def longest_pair(self) -> int:
        """state + the single longest question."""
        if self.longest_question_id is None:
            return self.state_tokens
        return self.state_tokens + self.question_tokens[self.longest_question_id]

    @property
    def over_total(self) -> bool:
        return self.total > TOTAL_BUDGET

    @property
    def over_state(self) -> bool:
        return self.longest_pair > STATE_BUDGET

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"BudgetReport(total={self.total}, longest_pair={self.longest_pair}, "
            f"over_total={self.over_total}, over_state={self.over_state})"
        )


def check_budget(state: Any, questions: dict[str, Any]) -> BudgetReport:
    state_tokens = estimate_tokens(state)
    per_question = {qid: estimate_tokens(q) for qid, q in questions.items()}
    longest = max(per_question, key=lambda k: per_question[k]) if per_question else None
    return BudgetReport(state_tokens, per_question, longest)
