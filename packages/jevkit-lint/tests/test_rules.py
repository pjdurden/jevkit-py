"""One test per rule: a case that must fire, and a near-miss that must not."""

import pytest
from jevkit_lint import Severity, all_codes, lint


def codes(questions, state=""):
    return [d.code for d in lint(questions, state)]


def noul(instructions, **kw):
    return {"type": "noul", "instructions": instructions, **kw}


def choice(instructions, criteria):
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def score(instructions, criteria):
    return {"type": "score", "instructions": instructions, "criteria": criteria}


GOOD = choice(
    "Which team should handle this request",
    {"billing": "Payment or subscription issues", "technical": "Bugs or integration problems",
     "unknown": "None of the above applies"},
)


def test_a_well_formed_question_is_clean():
    assert codes({"ok": GOOD}) == []


@pytest.mark.parametrize("text", [
    "How many invoices are attached",
    "Count the number of mentions",
    "What is the average order value",
    "Calculate the outstanding balance",
])
def test_jev001_fires_on_arithmetic(text):
    assert "JEV001" in codes({"q": noul(text)})


def test_jev001_ignores_semantic_questions():
    assert "JEV001" not in codes({"q": noul("Does the customer mention a billing problem")})


@pytest.mark.parametrize("text", [
    "Which date came first",
    "Was the invoice sent within 30 days",
    "How long between the two events",
    "Is the subscription overdue",
])
def test_jev002_fires_on_date_reasoning(text):
    assert "JEV002" in codes({"q": noul(text)})


def test_jev002_ignores_date_extraction():
    assert "JEV002" not in codes({
        "q": choice("Which month is named in the message",
                    {"january": "January", "february": "February", "not_stated": "No month named"})
    })


@pytest.mark.parametrize("text", [
    "Summarize the complaint",
    "Write a reply to the customer",
    "Explain why the charge failed",
    "Translate the message",
])
def test_jev003_fires_on_generation(text):
    assert "JEV003" in codes({"q": noul(text)})


def test_jev004_fires_on_double_negation():
    found = codes({"q": noul("The request does not fail to mention a refund, unless it is absent")})
    assert "JEV004" in found


def test_jev004_flags_a_single_negation_on_noul_as_info_only():
    result = lint({"q": noul("The message is not a refund request")})
    (diag,) = [d for d in result if d.code == "JEV004"]
    assert diag.severity is Severity.INFO


def test_jev005_fires_on_vague_word_with_no_criteria():
    assert "JEV005" in codes({"q": noul("The passage is relevant")})


def test_jev005_is_satisfied_by_criteria():
    assert "JEV005" not in codes({
        "q": {"type": "noul", "instructions": "The passage is relevant",
              "criteria": {"true": "It answers the user's question directly",
                           "false": "It does not answer the question"}}
    })


def test_jev006_fires_on_multi_hop_instructions():
    assert "JEV006" in codes({"q": noul("Check the owner of the parent of the account")})


def test_jev007_fires_on_inverted_noul_criteria():
    assert "JEV007" in codes({
        "q": {"type": "noul", "instructions": "Refund requested",
              "criteria": {"true": "No refund was requested", "false": "Yes a refund was requested"}}
    })


def test_jev007_accepts_aligned_criteria():
    assert "JEV007" not in codes({
        "q": {"type": "noul", "instructions": "Refund requested",
              "criteria": {"true": "Yes, a refund is requested", "false": "No refund is requested"}}
    })


def test_jev008_fires_when_no_option_means_none():
    assert "JEV008" in codes({"q": choice("Pick a team", {"a": "Team A", "b": "Team B"})})


def test_jev008_accepts_an_explicit_escape_hatch():
    assert "JEV008" not in codes({"q": GOOD})


def test_jev009_fires_on_a_single_option_choice():
    assert "JEV009" in codes({"q": choice("Pick one", {"only": "The only option"})})


def test_jev009_fires_on_an_empty_choice():
    assert "JEV009" in codes({"q": choice("Pick one", {})})


def test_jev010_fires_on_blank_option_descriptions():
    assert "JEV010" in codes({"q": choice("Pick a team", {"a": "", "b": "Team B", "other": "None"})})


def test_jev011_fires_on_bare_score_labels():
    assert "JEV011" in codes({"q": score("How frustrated", ["low", "medium", "high"])})


def test_jev011_accepts_descriptive_levels():
    assert "JEV011" not in codes({
        "q": score("How frustrated the customer appears",
                   ["Calm, just stating facts", "Frustrated but civil", "Very angry, strong language"])
    })


def test_jev011_fires_on_a_one_level_scale():
    assert "JEV011" in codes({"q": score("How frustrated", ["Calm and stating facts"])})


def test_jev012_fires_on_missing_instructions():
    assert "JEV012" in codes({"q": {"type": "noul", "instructions": ""}})


def test_jev012_fires_on_terse_instructions():
    assert "JEV012" in codes({"q": noul("urgent?")})


def test_jev013_fires_when_one_question_references_another_by_id():
    found = codes({
        "is_refund": noul("The message requests a refund"),
        "amount": noul("If is_refund is true, an amount is stated"),
    })
    assert "JEV013" in found


def test_jev013_ignores_backticked_state_paths():
    assert "JEV013" not in codes({
        "ticket": noul("The message requests a refund"),
        "amount": noul("An amount is stated in `ticket.messages[0].text`"),
    })


def test_jev014_fires_when_state_exceeds_its_budget():
    assert "JEV014" in codes({"q": noul("Is this relevant to billing")}, "x" * 200_000)


def test_jev015_fires_when_the_request_exceeds_the_total_budget():
    questions = {f"q{i}": noul("y" * 3000) for i in range(120)}
    assert "JEV015" in codes(questions, "s")


def test_jev016_flags_a_noisy_state_without_erroring():
    result = lint({"q": noul("Is this about billing")}, "x" * 40_000)
    (diag,) = [d for d in result if d.code == "JEV016"]
    assert diag.severity is Severity.INFO


def test_jev017_fires_on_hex_colours():
    assert "JEV017" in codes({"q": noul("Is #ff0000 close to the brand colour")})


def test_jev018_flags_identical_questions():
    assert "JEV018" in codes({"a": noul("The message requests a refund"),
                              "b": noul("The message requests a refund")})


def test_every_registered_code_has_a_test():
    """Guards against adding a rule and forgetting to cover it."""
    source = open(__file__).read()
    for code in all_codes():
        assert code in source, f"{code} has no test"
