from jevkit_core import STATE_BUDGET, TOTAL_BUDGET, check_budget, estimate_tokens


def test_estimate_grows_with_length():
    assert estimate_tokens("a" * 1000) > estimate_tokens("a" * 10)


def test_estimate_is_never_zero():
    assert estimate_tokens("") >= 1


def test_small_request_is_within_both_budgets():
    report = check_budget("short state", {"q": {"type": "noul", "instructions": "ok?"}})
    assert not report.over_state and not report.over_total


def test_oversized_state_trips_the_state_budget():
    report = check_budget("x" * (STATE_BUDGET * 4), {"q": {"type": "noul", "instructions": "ok?"}})
    assert report.over_state


def test_many_questions_trip_the_total_budget():
    questions = {f"q{i}": {"type": "noul", "instructions": "y" * 2000} for i in range(200)}
    report = check_budget("s", questions)
    assert report.over_total


def test_longest_question_is_identified():
    report = check_budget("s", {
        "small": {"type": "noul", "instructions": "a"},
        "big": {"type": "noul", "instructions": "a" * 500},
    })
    assert report.longest_question_id == "big"


def test_empty_questions_does_not_crash():
    report = check_budget("s", {})
    assert report.longest_question_id is None
    assert report.longest_pair == report.state_tokens
