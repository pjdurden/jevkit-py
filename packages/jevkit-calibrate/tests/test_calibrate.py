import random
import pytest
from jevkit_core import Record
from jevkit_calibrate import (Observation, brier_score, calibrate,
                              expected_calibration_error, log_loss,
                              maximum_calibration_error, observations_from_records,
                              recommend_for_accuracy, recommend_for_coverage, sweep)


def perfect(n=2000, seed=1):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        p = rng.uniform(0.5, 1.0)
        out.append(Observation(probability=p, correct=rng.random() < p))
    return out


def test_probability_outside_the_unit_interval_is_rejected():
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError, match="probability must be"):
            Observation(probability=bad, correct=True)


def test_a_perfectly_calibrated_generator_has_near_zero_ece():
    assert expected_calibration_error(perfect()) < 0.05


def test_a_maximally_overconfident_model_has_ece_near_one():
    obs = [Observation(probability=1.0, correct=False) for _ in range(100)]
    assert expected_calibration_error(obs) == pytest.approx(1.0)


def test_mce_catches_one_bad_region_that_ece_dilutes():
    good = [Observation(probability=0.5, correct=i % 2 == 0) for i in range(998)]
    bad = [Observation(probability=1.0, correct=False) for _ in range(2)]
    obs = good + bad
    assert expected_calibration_error(obs) < 0.01   # diluted
    assert maximum_calibration_error(obs) == pytest.approx(1.0)  # caught


def test_brier_rewards_being_decisive_as_well_as_calibrated():
    always_half = [Observation(probability=0.5, correct=i % 2 == 0) for i in range(100)]
    decisive = [Observation(probability=1.0, correct=True) for _ in range(100)]
    assert brier_score(decisive) < brier_score(always_half)


def test_log_loss_is_finite_on_a_confident_miss():
    assert log_loss([Observation(probability=1.0, correct=False)]) < float("inf")


def test_empty_input_produces_zeros_not_errors():
    assert expected_calibration_error([]) == 0.0
    assert brier_score([]) == 0.0
    assert calibrate([]).summary() == "no labeled observations"


def test_bins_cover_the_unit_interval_and_include_one():
    report = calibrate([Observation(probability=1.0, correct=True)], n_bins=10)
    populated = [b for b in report.bins if b.count]
    assert len(populated) == 1 and populated[0].upper == 1.0


def test_empty_bins_are_kept_so_coverage_is_visible():
    report = calibrate([Observation(probability=0.95, correct=True)], n_bins=10)
    assert len(report.bins) == 10
    assert sum(1 for b in report.bins if b.count == 0) == 9


def test_overconfidence_is_detected():
    obs = [Observation(probability=0.9, correct=i < 50) for i in range(100)]
    assert calibrate(obs).overconfident


def test_diagram_renders_without_a_plotting_library():
    out = calibrate(perfect(200)).diagram()
    assert "claimed" in out and "actual" in out


def test_sweep_is_monotone_in_coverage():
    points = sweep(perfect(500))
    coverages = [p.coverage for p in points]
    assert coverages == sorted(coverages, reverse=True)


def test_threshold_for_accuracy_picks_the_lowest_qualifying_one():
    obs = perfect(3000)
    point = recommend_for_accuracy(obs, 0.85)
    assert point is not None and point.accuracy >= 0.85
    lower = [p for p in sweep(obs) if p.threshold < point.threshold and p.covered]
    assert all(p.accuracy < 0.85 for p in lower)


def test_unreachable_accuracy_returns_none_rather_than_a_bad_threshold():
    obs = [Observation(probability=0.6, correct=False) for _ in range(100)]
    assert recommend_for_accuracy(obs, 0.99) is None


def test_threshold_for_coverage_picks_the_highest_qualifying_one():
    point = recommend_for_coverage(perfect(1000), 0.5)
    assert point is not None and point.coverage >= 0.5


def test_accuracy_is_one_when_nothing_is_covered():
    obs = [Observation(probability=0.1, correct=False)]
    top = sweep(obs)[-1]
    assert top.covered == 0 and top.accuracy == 1.0


def test_records_without_labels_are_skipped():
    unlabeled = Record(model="m", state="s", questions={"q": {"type": "noul"}},
                       answers={"q": {"type": "noul", "noul": 0.9}})
    assert observations_from_records([unlabeled]) == []


def test_observations_are_extracted_from_labeled_records():
    labeled = Record(model="m", state="s", questions={"q": {"type": "choice"}},
                     answers={"q": {"type": "choice", "choice": "a",
                                    "probabilities": {"a": 0.8, "b": 0.2}, "confidence": 0.6}},
                     label={"q": "a"})
    (obs,) = observations_from_records([labeled])
    assert obs.correct and obs.probability == pytest.approx(0.8)


def test_use_confidence_selects_the_other_quantity():
    labeled = Record(model="m", state="s", questions={"q": {"type": "choice"}},
                     answers={"q": {"type": "choice", "choice": "a",
                                    "probabilities": {"a": 0.8, "b": 0.2}, "confidence": 0.6}},
                     label={"q": "a"})
    (obs,) = observations_from_records([labeled], use_confidence=True)
    assert obs.probability == pytest.approx(0.6)
