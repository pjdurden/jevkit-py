"""Static linter for TypeSafe Jev questions.

Catches the failure modes TypeSafe documents for jev-1.13 before you spend a
token on them. Needs no API key.
"""

from .diagnostic import Diagnostic, Severity
from .linter import LintResult, lint
from .rules import RULES, Rule, all_codes

__version__ = "0.1.0"

__all__ = ["lint", "LintResult", "Diagnostic", "Severity", "Rule", "RULES",
           "all_codes", "__version__"]
