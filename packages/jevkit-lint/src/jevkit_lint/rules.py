"""Lint rules.

Every rule here maps to a failure mode TypeSafe documents for `jev-1.13`
(https://docs.typesafe.ai/model-jaggedness/jev-1.13) or to a structural
mistake that makes an answer unusable. The `mode` field on each rule names
the documented failure mode so the CLI can point at the source.

Rules are static. They read your question definitions and never call the API,
which is why jevkit-lint works without an API key.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from jevkit_core import BudgetReport, Question, check_budget

from .diagnostic import Diagnostic, Severity

__all__ = ["LintContext", "Rule", "RULES", "rules_for", "all_codes"]


@dataclass
class LintContext:
    questions: list[Question]
    state: Any
    budget: BudgetReport

    @classmethod
    def build(cls, state: Any, questions: list[Question]) -> "LintContext":
        # Budget estimation canonicalizes its input, so questions are reduced to
        # plain JSON here. SDK objects are not serializable and the raw form is
        # only ever needed by rules that read the normalized view anyway.
        plain = {
            q.id: {"type": q.type, "instructions": q.instructions, "criteria": q.criteria}
            for q in questions
        }
        return cls(questions=questions, state=state, budget=check_budget(state, plain))


@dataclass(frozen=True)
class Rule:
    code: str
    name: str
    mode: str
    check: Callable[[LintContext], list[Diagnostic]]


def _words(text: str) -> str:
    return text.lower()


def _find(patterns: Iterable[str], text: str) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(0).strip()
    return None


# --------------------------------------------------------------------------
# JEV001 — Math and Numbers
# --------------------------------------------------------------------------

_COUNT_PATTERNS = [
    r"\bhow many\b", r"\bnumber of\b", r"\bcount (?:the|of|how)\b", r"\btally\b",
    r"\btotal (?:number|count|of)\b", r"\bsum of\b", r"\baverage\b", r"\bmean of\b",
    r"\bcalculate\b", r"\bcompute the\b", r"\bpercentage of\b", r"\bhow much (?:is|does)\b",
    r"\bmultipl(?:y|ied)\b", r"\bdivide[d]?\b", r"\bsubtract\b",
]


def _check_math(ctx: LintContext) -> list[Diagnostic]:
    out = []
    for q in ctx.questions:
        hit = _find(_COUNT_PATTERNS, q.text)
        if hit:
            out.append(Diagnostic(
                code="JEV001", severity=Severity.ERROR, question_id=q.id,
                message="Question asks the model to count or do arithmetic.",
                evidence=hit,
                hint="jev-1.13 is not a calculator and does not count reliably. "
                     "Iterate the candidates in code, ask one Noul per item, and sum "
                     "the answers yourself.",
            ))
    return out


# --------------------------------------------------------------------------
# JEV002 — Date and time comparison
# --------------------------------------------------------------------------

_DATE_COMPARE_PATTERNS = [
    r"\b(?:before|after|earlier than|later than|prior to)\b[^.?]{0,40}\b(?:date|day|month|year|deadline|timestamp)\b",
    r"\b(?:date|day|month|year|deadline|timestamp)\b[^.?]{0,40}\b(?:before|after|earlier|later)\b",
    r"\bhow (?:long|many days|many months|many years)\b",
    r"\bwithin \d+ (?:day|week|month|year)s?\b",
    r"\b(?:days|weeks|months|years) (?:between|apart|since|until)\b",
    r"\bmost recent\b", r"\bchronologic(?:al|ally)\b",
    r"\b(?:which|what)\b[^.?]{0,30}\b(?:came|comes|happened|occurred|was|is)\s+"
    r"(?:first|last|earliest|latest|more recent)\b",
    r"\b(?:earliest|latest|oldest|newest)\b[^.?]{0,20}"
    r"\b(?:date|day|month|year|deadline|timestamp|event|entry)\b",
    r"\b(?:date|day|month|year|deadline|timestamp)\b[^.?]{0,20}"
    r"\b(?:first|last|earliest|latest)\b",
    r"\bexpired?\b", r"\boverdue\b", r"\bin the (?:past|last|next) \d+\b",
]


def _check_dates(ctx: LintContext) -> list[Diagnostic]:
    out = []
    for q in ctx.questions:
        hit = _find(_DATE_COMPARE_PATTERNS, q.text)
        if hit:
            out.append(Diagnostic(
                code="JEV002", severity=Severity.ERROR, question_id=q.id,
                message="Question compares or measures dates.",
                evidence=hit,
                hint="jev-1.13 reads dates as text, not ordered quantities. Extract the "
                     "parts as Choices over closed sets (12 months, 31 days, a bounded "
                     "year range, plus an explicit 'not stated'), then order and subtract "
                     "in code.",
            ))
    return out


# --------------------------------------------------------------------------
# JEV003 — Generation
# --------------------------------------------------------------------------

_GENERATION_PATTERNS = [
    r"\b(?:write|draft|compose|author) (?:a|an|the|some)\b",
    r"\b(?:generate|produce|create) (?:a|an|the)\s+(?:summary|response|reply|message|list|description|explanation|text|paragraph)\b",
    r"\bsummari[sz]e\b", r"\bparaphrase\b", r"\brephrase\b", r"\brewrite\b",
    r"\bexplain (?:why|how|what)\b", r"\bdescribe (?:in|the|what|how)\b",
    r"\bin your own words\b", r"\bprovide (?:a|an) (?:summary|explanation|rationale|reason)\b",
    r"\bgive (?:a|an) (?:reason|explanation|rationale)\b", r"\btranslate\b",
]


def _check_generation(ctx: LintContext) -> list[Diagnostic]:
    out = []
    for q in ctx.questions:
        hit = _find(_GENERATION_PATTERNS, q.instructions_text)
        if hit:
            out.append(Diagnostic(
                code="JEV003", severity=Severity.ERROR, question_id=q.id,
                message="Question asks the model to generate text.",
                evidence=hit,
                hint="Jev returns typed answers and probabilities, never generated text. "
                     "Use a generative model for this, or restate it as a selection over "
                     "candidates your code already has.",
            ))
    return out


# --------------------------------------------------------------------------
# JEV004 — Literal reading: negation
# --------------------------------------------------------------------------

_NEGATIONS = r"\b(?:not|never|no|none|neither|nor|without|except|unless|excluding|absent|lacks?|fails? to|cannot|can't|doesn't|does not|isn't|is not|aren't|won't)\b"


def _check_negation(ctx: LintContext) -> list[Diagnostic]:
    out = []
    for q in ctx.questions:
        text = q.instructions_text
        hits = re.findall(_NEGATIONS, text, flags=re.IGNORECASE)
        if len(hits) >= 2:
            out.append(Diagnostic(
                code="JEV004", severity=Severity.WARNING, question_id=q.id,
                message=f"Instructions contain {len(hits)} negations, which compounds into "
                        "a double negative.",
                evidence=", ".join(sorted({h.lower() for h in hits})),
                hint="jev-1.13 reads negations literally and loses accuracy on double "
                     "negatives. Restate positively, or split into two literal questions "
                     "and combine them in code.",
            ))
        elif hits and q.type == "noul":
            out.append(Diagnostic(
                code="JEV004", severity=Severity.INFO, question_id=q.id,
                message="Noul instruction is phrased negatively.",
                evidence=hits[0].lower(),
                hint="A Noul returns the probability the statement is true. A negative "
                     "statement inverts the reading of every threshold downstream. Prefer "
                     "the positive form and invert in code if you need it.",
            ))
    return out


# --------------------------------------------------------------------------
# JEV005 — Literal reading: vague scoping
# --------------------------------------------------------------------------

_VAGUE = [
    r"\brelevant\b", r"\bappropriate\b", r"\bimportant\b", r"\bsignificant\b",
    r"\bsuitable\b", r"\bproper\b", r"\bgood\b", r"\bbad\b", r"\bbetter\b",
    r"\breasonable\b", r"\bacceptable\b", r"\bsufficient\b", r"\badequate\b",
    r"\bmeaningful\b", r"\bnoteworthy\b", r"\bproblematic\b",
]


def _check_vague(ctx: LintContext) -> list[Diagnostic]:
    out = []
    for q in ctx.questions:
        found = sorted({
            m.group(0).lower()
            for pattern in _VAGUE
            for m in re.finditer(pattern, q.instructions_text, flags=re.IGNORECASE)
        })
        if found and not q.criteria_text.strip():
            out.append(Diagnostic(
                code="JEV005", severity=Severity.WARNING, question_id=q.id,
                message="Instructions lean on an undefined evaluative word and give no "
                        "criteria to pin it down.",
                evidence=", ".join(found),
                hint="jev-1.13 answers the question you wrote, not the one you meant. "
                     "Define what the word means here in the criteria, including the "
                     "boundary cases.",
            ))
        elif len(found) >= 2:
            out.append(Diagnostic(
                code="JEV005", severity=Severity.INFO, question_id=q.id,
                message="Instructions stack several undefined evaluative words.",
                evidence=", ".join(found),
                hint="Each one is a separate judgment the model has to guess at. Name the "
                     "exact condition, or split into separate questions.",
            ))
    return out


# --------------------------------------------------------------------------
# JEV006 — Indirection
# --------------------------------------------------------------------------

_INDIRECTION_PATTERNS = [
    r"\bthe \w+ of the \w+ of the\b",
    r"\bwhoever\b[^.?]{0,40}\bwhose\b",
    r"\bif .{0,60}\bthen\b.{0,60}\bif\b",
    r"\bwould have (?:been|had)\b",
    r"\bimplies? that\b[^.?]{0,40}\bwhich\b",
    r"\bindirectly\b", r"\btransitively\b",
]


def _check_indirection(ctx: LintContext) -> list[Diagnostic]:
    out = []
    for q in ctx.questions:
        hit = _find(_INDIRECTION_PATTERNS, q.instructions_text)
        if hit:
            out.append(Diagnostic(
                code="JEV006", severity=Severity.WARNING, question_id=q.id,
                message="Instructions require multiple hops of reasoning.",
                evidence=hit,
                hint="Every hop costs accuracy. Resolve the intermediate step in code and "
                     "name the resulting state field directly in the instruction.",
            ))
    return out


# --------------------------------------------------------------------------
# JEV007 — Contradictory instructions and criteria (Noul polarity)
# --------------------------------------------------------------------------

_FALSEY = {"no", "false", "absent", "none", "negative", "not present", "does not", "fails"}
_TRUTHY = {"yes", "true", "present", "positive", "does", "passes"}


def _check_noul_polarity(ctx: LintContext) -> list[Diagnostic]:
    out = []
    for q in ctx.questions:
        if q.type != "noul" or not isinstance(q.criteria, dict):
            continue
        true_side = _words(str(q.criteria.get("true", "")))
        false_side = _words(str(q.criteria.get("false", "")))
        if not true_side and not false_side:
            continue
        inverted = (
            any(tok in true_side for tok in _FALSEY)
            and any(tok in false_side for tok in _TRUTHY)
        )
        if inverted:
            out.append(Diagnostic(
                code="JEV007", severity=Severity.ERROR, question_id=q.id,
                message="Noul criteria invert polarity: 'true' describes a no and 'false' "
                        "describes a yes.",
                evidence=f"true={true_side!r} false={false_side!r}",
                hint="TypeSafe documents this exact shape as a performance loss. Swap the "
                     "two descriptions and rewrite the instruction so 'true' means yes.",
            ))
    return out


# --------------------------------------------------------------------------
# JEV008 — Choice structure
# --------------------------------------------------------------------------

_NO_MATCH = {
    "none", "none_of_the_above", "no_match", "nomatch", "other", "unknown",
    "unclear", "not_stated", "not_applicable", "n_a", "na", "neither",
    "cannot_tell", "insufficient", "uncertain", "ambiguous", "not_specified",
}


def _check_choice_no_match(ctx: LintContext) -> list[Diagnostic]:
    out = []
    for q in ctx.questions:
        if q.type != "choice":
            continue
        keys = {k.lower().replace("-", "_").replace(" ", "_") for k in q.options}
        if keys and not (keys & _NO_MATCH):
            out.append(Diagnostic(
                code="JEV008", severity=Severity.WARNING, question_id=q.id,
                message="Choice has no no-match option.",
                evidence=", ".join(sorted(keys)),
                hint="A Choice always returns one of its options. With nothing meaning "
                     "'none of these', a state that fits no option still produces a "
                     "confident-looking answer. Add an explicit none/unknown option, or "
                     "gate on a separate presence Noul.",
            ))
    return out


def _check_choice_arity(ctx: LintContext) -> list[Diagnostic]:
    out = []
    for q in ctx.questions:
        if q.type != "choice":
            continue
        n = len(q.options)
        if n == 0:
            out.append(Diagnostic(
                code="JEV009", severity=Severity.ERROR, question_id=q.id,
                message="Choice defines no options.",
                hint="Give the Choice a criteria object mapping each option key to a "
                     "description of when it applies.",
            ))
        elif n == 1:
            out.append(Diagnostic(
                code="JEV009", severity=Severity.ERROR, question_id=q.id,
                message="Choice defines a single option, so the answer is predetermined.",
                evidence=q.options[0],
                hint="Use a Noul if the question is really yes/no, or add the competing "
                     "options.",
            ))
    return out


def _check_empty_descriptions(ctx: LintContext) -> list[Diagnostic]:
    out = []
    for q in ctx.questions:
        if q.type != "choice":
            continue
        blank = [key for key, desc in q.option_descriptions() if not desc.strip()]
        if blank:
            out.append(Diagnostic(
                code="JEV010", severity=Severity.WARNING, question_id=q.id,
                message="Choice options have no description.",
                evidence=", ".join(blank),
                hint="The option key alone is all the model gets. Describe when each "
                     "option applies, especially the boundary against its nearest rival.",
            ))
    return out


# --------------------------------------------------------------------------
# JEV011 — Score structure
# --------------------------------------------------------------------------

def _check_score_levels(ctx: LintContext) -> list[Diagnostic]:
    out = []
    for q in ctx.questions:
        if q.type != "score":
            continue
        levels = q.options
        if len(levels) < 2:
            out.append(Diagnostic(
                code="JEV011", severity=Severity.ERROR, question_id=q.id,
                message=f"Score defines {len(levels)} level(s); at least 2 are needed to "
                        "form a scale.",
                hint="Give each level a concrete description of the situation it covers.",
            ))
            continue
        terse = [lv for lv in levels if len(lv.split()) < 2]
        if terse:
            out.append(Diagnostic(
                code="JEV011", severity=Severity.WARNING, question_id=q.id,
                message="Score levels are bare labels rather than descriptions.",
                evidence=", ".join(terse),
                hint="Levels must describe concrete situations and stand on their own. "
                     "'Frustrated but civil' works; 'medium' does not.",
            ))
    return out


# --------------------------------------------------------------------------
# JEV012 — Instructions present and substantive
# --------------------------------------------------------------------------

def _check_instructions(ctx: LintContext) -> list[Diagnostic]:
    out = []
    for q in ctx.questions:
        text = q.instructions_text.strip()
        if not text:
            out.append(Diagnostic(
                code="JEV012", severity=Severity.ERROR, question_id=q.id,
                message="Question has no instructions.",
                hint="The instruction carries the judgment. Without it the model only has "
                     "the criteria to go on.",
            ))
        elif len(text.split()) < 3:
            out.append(Diagnostic(
                code="JEV012", severity=Severity.WARNING, question_id=q.id,
                message="Instructions are too terse to state a condition.",
                evidence=text,
                hint="State the exact condition being judged, in language an average "
                     "reader would resolve the same way you do.",
            ))
    return out


# --------------------------------------------------------------------------
# JEV013 — Question ids are not sent to the model
# --------------------------------------------------------------------------

def _check_id_reference(ctx: LintContext) -> list[Diagnostic]:
    out = []
    ids = {q.id for q in ctx.questions}
    for q in ctx.questions:
        text = q.instructions_text
        referenced = sorted({
            other for other in ids
            if other != q.id
            and re.search(rf"(?<![\w.`]){re.escape(other)}(?![\w`])", text)
        })
        if referenced:
            out.append(Diagnostic(
                code="JEV013", severity=Severity.WARNING, question_id=q.id,
                message="Instructions refer to another question by its id.",
                evidence=", ".join(referenced),
                hint="Question ids are for your code and are not sent to the model. "
                     "Questions in one request are answered in parallel and cannot see "
                     "each other. State the premise explicitly, or split into a second "
                     "request.",
            ))
    return out


# --------------------------------------------------------------------------
# JEV014 / JEV015 — Budgets
# --------------------------------------------------------------------------

def _check_state_budget(ctx: LintContext) -> list[Diagnostic]:
    budget = ctx.budget
    if not budget.over_state:
        return []
    return [Diagnostic(
        code="JEV014", severity=Severity.ERROR, question_id=None,
        message=f"state plus the longest question is about {budget.longest_pair:,} "
                f"tokens, over the 32,000 limit.",
        evidence=f"longest question: {budget.longest_question_id}",
        hint="Retrieve and filter in code so the state carries only the fields this "
             "judgment needs. Estimates are approximate; leave headroom.",
    )]


def _check_total_budget(ctx: LintContext) -> list[Diagnostic]:
    budget = ctx.budget
    if not budget.over_total:
        return []
    return [Diagnostic(
        code="JEV015", severity=Severity.ERROR, question_id=None,
        message=f"state plus all {len(ctx.questions)} questions is about "
                f"{budget.total:,} tokens, over the 64,000 limit.",
        hint="Split into several requests, or drop speculative questions that this "
             "state can never make relevant.",
    )]


# --------------------------------------------------------------------------
# JEV016 — Large state of mostly irrelevant detail
# --------------------------------------------------------------------------

_LARGE_STATE_RATIO = 8.0
_LARGE_STATE_FLOOR = 4_000


def _check_state_size(ctx: LintContext) -> list[Diagnostic]:
    budget = ctx.budget
    if budget.over_state or budget.over_total:
        return []  # Already reported as an error; don't double up.
    question_tokens = sum(budget.question_tokens.values()) or 1
    ratio = budget.state_tokens / question_tokens
    if budget.state_tokens >= _LARGE_STATE_FLOOR and ratio >= _LARGE_STATE_RATIO:
        return [Diagnostic(
            code="JEV016", severity=Severity.INFO, question_id=None,
            message=f"state is about {budget.state_tokens:,} tokens against "
                    f"{question_tokens:,} tokens of questions ({ratio:.0f}x).",
            hint="Accuracy falls as the state grows with content unrelated to the "
                 "decision, and a large state makes a wrong answer hard to attribute. "
                 "Filter in code first, or use a Noul to screen for relevance.",
        )]
    return []


# --------------------------------------------------------------------------
# JEV017 — Numeric representations
# --------------------------------------------------------------------------

_NUMERIC_REPR = [
    r"#[0-9a-fA-F]{6}\b", r"\brgba?\s*\(", r"\bhex(?:adecimal)? (?:value|code|colou?r)\b",
    r"\bbinary (?:value|encoding|representation)\b", r"\bbase64\b",
    r"\bassembly (?:instruction|opcode)\b", r"\bopcode\b", r"\bbytecode\b",
]


def _check_numeric_repr(ctx: LintContext) -> list[Diagnostic]:
    out = []
    for q in ctx.questions:
        hit = _find(_NUMERIC_REPR, q.text)
        if hit:
            out.append(Diagnostic(
                code="JEV017", severity=Severity.WARNING, question_id=q.id,
                message="Question reasons over a machine-oriented numeric representation.",
                evidence=hit,
                hint="jev-1.13 does better on semantic representations than numeric ones: "
                     "colour names beat hex, high-level code beats bytecode. Convert in "
                     "code and pass the name or a named bucket.",
            ))
    return out


# --------------------------------------------------------------------------
# JEV018 — Duplicate judgments
# --------------------------------------------------------------------------

def _check_duplicates(ctx: LintContext) -> list[Diagnostic]:
    seen: dict[str, str] = {}
    out = []
    for q in ctx.questions:
        key = re.sub(r"\W+", " ", q.text.lower()).strip()
        if not key:
            continue
        if key in seen:
            out.append(Diagnostic(
                code="JEV018", severity=Severity.INFO, question_id=q.id,
                message=f"Question is textually identical to {seen[key]!r}.",
                hint="Identical questions in one request cost tokens twice for the same "
                     "answer. If you meant to measure self-consistency, note that they "
                     "are evaluated in one pass and are not independent samples.",
            ))
        else:
            seen[key] = q.id
    return out


RULES: list[Rule] = [
    Rule("JEV001", "math-and-counting", "Math and Numbers", _check_math),
    Rule("JEV002", "date-comparison", "Date and time comparison", _check_dates),
    Rule("JEV003", "generation-request", "Generation", _check_generation),
    Rule("JEV004", "negation", "Literal reading", _check_negation),
    Rule("JEV005", "vague-scoping", "Literal reading", _check_vague),
    Rule("JEV006", "indirection", "Indirection", _check_indirection),
    Rule("JEV007", "noul-polarity", "Contradictory instructions and criteria",
         _check_noul_polarity),
    Rule("JEV008", "choice-no-match", "Common-sense structural invariants",
         _check_choice_no_match),
    Rule("JEV009", "choice-arity", "Common-sense structural invariants",
         _check_choice_arity),
    Rule("JEV010", "option-descriptions", "Literal reading", _check_empty_descriptions),
    Rule("JEV011", "score-levels", "Literal reading", _check_score_levels),
    Rule("JEV012", "instructions-present", "Literal reading", _check_instructions),
    Rule("JEV013", "question-id-reference", "Indirection", _check_id_reference),
    Rule("JEV014", "state-budget", "Large state full of irrelevant detail",
         _check_state_budget),
    Rule("JEV015", "total-budget", "Large state full of irrelevant detail",
         _check_total_budget),
    Rule("JEV016", "state-noise-ratio", "Large state full of irrelevant detail",
         _check_state_size),
    Rule("JEV017", "numeric-representation", "Math and Numbers", _check_numeric_repr),
    Rule("JEV018", "duplicate-questions", "Common-sense structural invariants",
         _check_duplicates),
]


def rules_for(codes: Iterable[str] | None = None) -> list[Rule]:
    if codes is None:
        return list(RULES)
    wanted = {c.upper() for c in codes}
    return [r for r in RULES if r.code in wanted]


def all_codes() -> list[str]:
    return [r.code for r in RULES]
