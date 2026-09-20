"""Shared substrate for the jevkit packages.

Nothing here calls the jev API. This package holds the pieces every other
jevkit tool needs: canonical digests, the `.jevl` record format, a uniform
view of a question, and token budget estimation.
"""

from .answers import NOUL_THRESHOLD, Answer, parse_answer, parse_answers
from .canonical import canonical_json, digest, record_id, request_id
from .question import (Question, flatten_text, normalize_question,
                       normalize_questions)
from .record import (FORMAT_VERSION, Record, RecordFormatError, append_record,
                     load_cassette, read_records, write_records)
from .tokens import (STATE_BUDGET, TOTAL_BUDGET, BudgetReport, check_budget,
                     estimate_tokens)

__version__ = "0.2.0"

__all__ = [
    "Answer", "parse_answer", "parse_answers", "NOUL_THRESHOLD",
    "canonical_json", "digest", "record_id", "request_id",
    "Question", "flatten_text", "normalize_question", "normalize_questions",
    "Record", "RecordFormatError", "FORMAT_VERSION",
    "read_records", "write_records", "append_record", "load_cassette",
    "estimate_tokens", "check_budget", "BudgetReport",
    "TOTAL_BUDGET", "STATE_BUDGET",
    "__version__",
]
