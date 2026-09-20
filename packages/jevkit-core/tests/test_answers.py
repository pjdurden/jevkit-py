import pytest
from jevkit_core import parse_answer, parse_answers

CHOICE = {"type": "choice", "choice": "billing",
          "probabilities": {"billing": 0.7, "technical": 0.3}, "confidence": 0.55}
SCORE = {"type": "score", "score": 1.03,
         "probabilities": {"0": 0.1, "1": 0.8, "2": 0.1}, "confidence": 0.84}
NOUL = {"type": "noul", "noul": 0.99}


def test_choice_predicts_its_selection():
    a = parse_answer("q", CHOICE)
    assert a.predicted() == "billing"
    assert a.probability_of("billing") == 0.7
    assert a.is_correct("billing") and not a.is_correct("technical")


def test_noul_predicts_a_bool_against_the_threshold():
    assert parse_answer("q", NOUL).predicted() is True
    assert parse_answer("q", {"type": "noul", "noul": 0.2}).predicted() is False
    assert parse_answer("q", {"type": "noul", "noul": 0.5}).predicted() is True


def test_noul_synthesizes_a_two_outcome_distribution():
    assert parse_answer("q", NOUL).probabilities == {"true": 0.99, "false": pytest.approx(0.01)}


def test_noul_has_no_confidence_but_has_decisiveness():
    a = parse_answer("q", NOUL)
    assert a.confidence is None
    assert a.decisiveness == pytest.approx(0.98)


def test_score_predicts_the_most_probable_level_not_the_rounded_score():
    # score is 1.03 but the mass could sit elsewhere; predicted follows the mass.
    a = parse_answer("q", {"type": "score", "score": 1.03,
                           "probabilities": {"0": 0.45, "1": 0.1, "2": 0.45}})
    assert a.predicted() == 0
    assert parse_answer("q", SCORE).predicted() == 1


def test_score_is_correct_against_an_integer_label():
    assert parse_answer("q", SCORE).is_correct(1)
    assert not parse_answer("q", SCORE).is_correct(2)


def test_probability_of_an_unoffered_outcome_is_zero():
    assert parse_answer("q", CHOICE).probability_of("sales") == 0.0


def test_top_probability_is_the_mass_on_the_prediction():
    assert parse_answer("q", CHOICE).top_probability == 0.7


def test_type_is_inferred_when_the_discriminator_is_missing():
    assert parse_answer("q", {"choice": "a", "probabilities": {"a": 1.0}}).type == "choice"
    assert parse_answer("q", {"noul": 0.4}).type == "noul"


def test_unrecognisable_answer_raises():
    with pytest.raises(ValueError, match="cannot determine type"):
        parse_answer("q", {"mystery": 1})


def test_parse_answers_maps_every_question():
    assert set(parse_answers({"a": CHOICE, "b": NOUL})) == {"a", "b"}
