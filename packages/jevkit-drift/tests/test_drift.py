import pytest
from jevkit_core import Record
from jevkit_drift import compare_records, compare_sets, replay, total_variation

QUESTIONS = {"team": {"type": "choice", "instructions": "Which team"}}


def rec(model, choice, probs, confidence=0.5, state="s"):
    return Record(model=model, state=state, questions=QUESTIONS,
                  answers={"team": {"type": "choice", "choice": choice,
                                    "probabilities": probs, "confidence": confidence}})


def test_total_variation_of_identical_distributions_is_zero():
    assert total_variation({"a": 0.5, "b": 0.5}, {"a": 0.5, "b": 0.5}) == 0.0


def test_total_variation_of_disjoint_distributions_is_one():
    assert total_variation({"a": 1.0}, {"b": 1.0}) == pytest.approx(1.0)


def test_total_variation_counts_outcomes_missing_from_one_side():
    assert total_variation({"a": 1.0}, {"a": 0.5, "b": 0.5}) == pytest.approx(0.5)


def test_a_changed_decision_is_a_flip():
    delta = compare_records(
        rec("jev-1.13.0", "billing", {"billing": 0.7, "technical": 0.3}),
        rec("jev-1.14.0", "technical", {"billing": 0.3, "technical": 0.7}),
    )
    assert delta.flips and delta.questions[0].flipped
    assert delta.questions[0].before == "billing"
    assert delta.questions[0].after == "technical"


def test_a_moved_distribution_with_the_same_decision_is_not_a_flip():
    delta = compare_records(
        rec("jev-1.13.0", "billing", {"billing": 0.9, "technical": 0.1}),
        rec("jev-1.14.0", "billing", {"billing": 0.6, "technical": 0.4}),
    )
    assert not delta.flips
    assert delta.max_shift == pytest.approx(0.3)


def test_confidence_delta_is_reported():
    delta = compare_records(
        rec("jev-1.13.0", "billing", {"billing": 0.9}, confidence=0.8),
        rec("jev-1.14.0", "billing", {"billing": 0.9}, confidence=0.5),
    )
    assert delta.questions[0].confidence_delta == pytest.approx(-0.3)


def test_records_for_different_requests_cannot_be_compared():
    with pytest.raises(ValueError, match="different requests"):
        compare_records(rec("m", "a", {"a": 1.0}, state="one"),
                        rec("m", "a", {"a": 1.0}, state="two"))


def test_a_changed_answer_type_is_refused():
    before = Record(model="m", state="s", questions=QUESTIONS,
                    answers={"team": {"type": "choice", "choice": "a", "probabilities": {"a": 1.0}}})
    after = Record(model="m", state="s", questions=QUESTIONS,
                   answers={"team": {"type": "noul", "noul": 1.0}})
    with pytest.raises(ValueError, match="type changed"):
        compare_records(before, after)


def test_a_missing_answer_is_refused_rather_than_ignored():
    before = rec("m", "billing", {"billing": 1.0})
    after = Record(model="m", state="s", questions=QUESTIONS, answers={})
    with pytest.raises(ValueError, match="missing in the new run"):
        compare_records(before, after)


def test_sets_pair_on_request_id_across_model_versions():
    report = compare_sets(
        [rec("jev-1.13.0", "billing", {"billing": 0.7, "technical": 0.3})],
        [rec("jev-1.14.0", "technical", {"billing": 0.3, "technical": 0.7})],
    )
    assert len(report.deltas) == 1
    assert len(report.flips) == 1
    assert report.flip_rate == 1.0


def test_unmatched_records_are_reported_not_silently_dropped():
    report = compare_sets(
        [rec("m", "a", {"a": 1.0}, state="one")],
        [rec("m", "a", {"a": 1.0}, state="two")],
    )
    assert report.deltas == []
    assert len(report.unmatched_before) == 1
    assert len(report.unmatched_after) == 1


def test_replay_carries_labels_and_calls_through():
    baseline = [Record(model="jev-1.13.0", state="s", questions=QUESTIONS,
                       answers={"team": {"type": "choice", "choice": "billing",
                                         "probabilities": {"billing": 1.0}}},
                       label={"team": "billing"}, tags=["routing"])]
    seen = []

    def fake(state, questions):
        seen.append((state, questions))
        return {"model": "jev-1.14.0",
                "answers": {"team": {"type": "choice", "choice": "technical",
                                     "probabilities": {"technical": 1.0}}},
                "usage": {"input_tokens": 5}}

    out = replay(baseline, fake)
    assert len(seen) == 1
    assert out[0].model == "jev-1.14.0"
    assert out[0].label == {"team": "billing"}
    assert "routing" in out[0].tags
    assert out[0].request_id == baseline[0].request_id
