import pytest
from jevkit_lint import Severity, lint


def test_select_runs_only_the_named_rules():
    questions = {"q": {"type": "noul", "instructions": "How many items are there"}}
    assert [d.code for d in lint(questions, select=["JEV001"])] == ["JEV001"]
    assert lint(questions, select=["JEV002"]).diagnostics == []


def test_ignore_suppresses_a_rule():
    questions = {"q": {"type": "noul", "instructions": "How many items are there"}}
    assert "JEV001" not in [d.code for d in lint(questions, ignore=["JEV001"])]


def test_results_are_sorted_most_severe_first():
    result = lint({
        "gen": {"type": "noul", "instructions": "Summarize this"},
        "ch": {"type": "choice", "instructions": "Pick a team",
               "criteria": {"a": "Team A", "b": "Team B"}},
    })
    severities = [d.severity for d in result]
    assert severities == sorted(severities, key=lambda s: ["error", "warning", "info"].index(s.value))


def test_ok_is_true_when_only_warnings_are_present():
    result = lint({"q": {"type": "choice", "instructions": "Pick a team",
                         "criteria": {"a": "Team A", "b": "Team B"}}})
    assert result.warnings and result.ok


def test_ok_is_false_when_an_error_is_present():
    assert not lint({"q": {"type": "noul", "instructions": "How many items"}}).ok


def test_unknown_question_type_raises_a_clear_error():
    with pytest.raises(ValueError, match="cannot determine type"):
        lint({"q": {"type": "mystery", "instructions": "hello"}})


def test_sdk_style_objects_are_accepted():
    class Noul:
        def __init__(self, instructions):
            self.instructions = instructions
            self.criteria = None

    assert "JEV001" in [d.code for d in lint({"q": Noul("How many items are there")})]


def test_to_dict_is_json_serializable():
    import json
    json.dumps(lint({"q": {"type": "noul", "instructions": "Summarize this"}}).to_dict())
