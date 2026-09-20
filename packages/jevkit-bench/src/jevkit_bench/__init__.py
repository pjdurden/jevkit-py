"""Score a labeled Jev suite for accuracy and cost, and compare runs.

Answers the question you have to defend: for this task, on my data, is Jev good
enough and what does it cost? Accuracy, tokens and dollars together, because any
one of them alone is easy to win.
"""

from .score import (PRICE_PER_MTOK, QuestionResult, SuiteComparison, SuiteResult,
                    compare_suites, score_records)

__version__ = "0.1.0"

__all__ = ["score_records", "compare_suites", "SuiteResult", "SuiteComparison",
           "QuestionResult", "PRICE_PER_MTOK", "__version__"]
