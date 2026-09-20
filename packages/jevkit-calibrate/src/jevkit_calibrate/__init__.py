"""Verify Jev's calibration on your own data, and pick thresholds from it.

Calibration is the central claim behind a System One model, and it is the one
thing a user cannot check without tooling. This package computes reliability
diagrams, ECE, MCE, Brier and log loss over labeled answers, and turns a
labeled set into a defensible confidence threshold.

Pure standard library. A calibration check that needs a numpy build is a
calibration check that does not get run.
"""

from .metrics import (Bin, CalibrationReport, Observation, brier_score, calibrate,
                      expected_calibration_error, log_loss,
                      maximum_calibration_error, reliability_bins)
from .records import observations_from_records
from .thresholds import (ThresholdPoint, recommend_for_accuracy,
                         recommend_for_coverage, sweep)

__version__ = "0.1.0"

__all__ = ["calibrate", "CalibrationReport", "Observation", "Bin",
           "reliability_bins", "expected_calibration_error",
           "maximum_calibration_error", "brier_score", "log_loss",
           "observations_from_records", "sweep", "ThresholdPoint",
           "recommend_for_accuracy", "recommend_for_coverage", "__version__"]
