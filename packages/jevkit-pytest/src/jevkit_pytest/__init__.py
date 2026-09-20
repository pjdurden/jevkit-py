"""Record and replay TypeSafe Jev requests in pytest.

Cassettes are `.jevl` files, the same format the drift, bench and calibrate
packages read, so a recording made by your tests doubles as a golden set.
"""

from .assertions import assert_answer, assert_confident, describe_answer
from .cassette import Cassette, CassetteMiss

__version__ = "0.1.0"

__all__ = ["Cassette", "CassetteMiss", "assert_answer", "assert_confident",
           "describe_answer", "__version__"]
