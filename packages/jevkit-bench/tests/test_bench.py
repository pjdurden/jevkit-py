import pytest
from jevkit_core import Record
from jevkit_bench import compare_suites, score_records


def rec(choice, label, probs=None, state="s", usage=None, tags=()):
    return Record(
        model="jev-1.13.0", state=state,
        questions={"team": {"type": "choice", "instructions": "Which team"}},
        answers={"team": {"type": "choice", "choice": choice,
                          "probabilities": probs or {choice: 1.0}, "confidence": 0.9}},
        label={"team": label}, usage=usage, tags=list(tags),
    )


def test_accuracy_counts_matches():
    suite = score_records([rec("a", "a"), rec("b", "a", state="t")])
    assert suite.count == 2 and suite.correct == 1
    assert suite.accuracy == 0.5


def test_unlabeled_records_are_counted_and_skipped():
    unlabeled = Record(model="m", state="u", questions={"q": {"type": "noul"}},
                       answers={"q": {"type": "noul", "noul": 0.9}})
    suite = score_records([rec("a", "a"), unlabeled])
    assert suite.records == 2 and suite.unlabeled == 1 and suite.count == 1


def test_cost_uses_input_tokens_only():
    suite = score_records([rec("a", "a", usage={"input_tokens": 1_000_000,
                                                "output_tokens": 500_000})])
    assert suite.input_tokens == 1_000_000
    assert suite.output_tokens == 500_000
    assert suite.cost == pytest.approx(0.042)


def test_price_override_is_respected():
    suite = score_records([rec("a", "a", usage={"input_tokens": 1_000_000})],
                          price_per_mtok=1.0)
    assert suite.cost == pytest.approx(1.0)


def test_failures_are_sorted_most_confident_first():
    suite = score_records([
        rec("b", "a", probs={"b": 0.6, "a": 0.4}, state="one"),
        rec("b", "a", probs={"b": 0.99, "a": 0.01}, state="two"),
    ])
    assert [round(f.probability, 2) for f in suite.failures()] == [0.99, 0.6]


def test_by_tag_buckets_results():
    suite = score_records([rec("a", "a", tags=["routing"]),
                           rec("b", "a", state="t", tags=["routing"])])
    assert suite.by_tag()["routing"] == (2, 0.5)


def test_by_question_finds_the_weak_question():
    good = Record(model="m", state="s",
                  questions={"a": {"type": "noul"}, "b": {"type": "noul"}},
                  answers={"a": {"type": "noul", "noul": 0.9},
                           "b": {"type": "noul", "noul": 0.9}},
                  label={"a": True, "b": False})
    by_q = score_records([good]).by_question()
    assert by_q["a"] == (1, 1.0) and by_q["b"] == (1, 0.0)


def test_comparison_tracks_regressions_and_fixes_separately():
    baseline = score_records([rec("a", "a", state="one"), rec("b", "a", state="two")])
    candidate = score_records([rec("b", "a", state="one"), rec("a", "a", state="two")])
    comparison = compare_suites(baseline, candidate)
    # Accuracy is unchanged at 0.5, but the set of correct answers swapped entirely.
    assert comparison.accuracy_delta == 0.0
    assert len(comparison.regressions) == 1
    assert len(comparison.fixes) == 1


def test_empty_suite_reports_zero_rather_than_dividing_by_zero():
    suite = score_records([])
    assert suite.accuracy == 0.0 and suite.cost_per_question == 0.0
