import pytest
from jevkit_pytest import assert_answer, assert_confident, describe_answer

CHOICE = {"type": "choice", "choice": "billing",
          "probabilities": {"billing": 0.55, "technical": 0.45}, "confidence": 0.3}
NOUL = {"type": "noul", "noul": 0.99}


def test_a_matching_answer_passes():
    assert_answer(CHOICE, "billing")


def test_a_mismatch_reports_the_distribution():
    with pytest.raises(AssertionError) as exc:
        assert_answer(CHOICE, "technical")
    assert "probabilities" in str(exc.value)
    assert "billing=0.550" in str(exc.value)


def test_min_probability_catches_a_narrow_win():
    assert_answer(CHOICE, "billing", min_probability=0.5)
    with pytest.raises(AssertionError, match="carried only"):
        assert_answer(CHOICE, "billing", min_probability=0.8)


def test_min_confidence_catches_a_low_confidence_win():
    with pytest.raises(AssertionError, match="confidence was"):
        assert_answer(CHOICE, "billing", min_confidence=0.8)


def test_min_confidence_on_a_noul_explains_the_alternative():
    with pytest.raises(AssertionError, match="carries no\\s+confidence"):
        assert_answer(NOUL, True, min_confidence=0.5)


def test_assert_confident_uses_decisiveness_for_a_noul():
    assert_confident(NOUL, 0.9)
    with pytest.raises(AssertionError, match="decisiveness"):
        assert_confident({"type": "noul", "noul": 0.52}, 0.5)


def test_assert_confident_uses_confidence_for_a_choice():
    with pytest.raises(AssertionError, match="confidence was"):
        assert_confident(CHOICE, 0.9)


def test_describe_orders_probabilities_by_mass():
    out = describe_answer("team", CHOICE)
    assert out.index("billing=") < out.index("technical=")
