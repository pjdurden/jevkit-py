# jevkit-pytest

Record and replay TypeSafe Jev requests in pytest.

A model call in a test is slow, costs money, needs a key in CI, and can change
its answer under you when the alias moves. The fix is the one VCR established for
HTTP: record real responses once, replay them forever, re-record on purpose.

> Unofficial and unaffiliated with TypeSafe.

```bash
pip install jevkit-pytest
```

## Use

```python
def test_routes_billing_questions(jev_cassette):
    response = jev_cassette.system_one(
        "I was charged twice for the same order",
        {"team": {"type": "choice", "instructions": "Which team should handle this",
                  "criteria": {"billing": "Payment issues", "technical": "Bugs",
                               "unknown": "None apply"}}},
    )
    assert response.answers["team"]["choice"] == "billing"
```

Record the first time, then never again:

```bash
pytest --jev-record       # record anything missing
pytest                    # replay only; a miss is a failure
pytest --jev-rerecord     # replace every recording
```

To record you need a live client. Supply one by overriding the `jev_client`
fixture in your `conftest.py`:

```python
import pytest
from typesafe_sdk import TypeSafeClient

@pytest.fixture
def jev_client():
    with TypeSafeClient() as client:
        yield client
```

Replay-only runs need no client and no API key, which is the point: CI stays
green without a secret.

## Assertions that explain themselves

```python
from jevkit_pytest import assert_answer, assert_confident

assert_answer(response.answers["team"], "billing", min_probability=0.6)
assert_confident(response.answers["urgency"], 0.8)
```

On failure these print the whole distribution, because a 0.51/0.49 split and a
0.99/0.01 split are different bugs and `assert x == y` cannot tell them apart.

`assert_confident` uses the API's confidence for Choice and Score. A Noul carries
none, so its distance from 0.5 is used and the message says which it used.

## Cassettes are golden sets

A cassette is a `.jevl` file, the same format `jevkit-drift`, `jevkit-bench` and
`jevkit-calibrate` read. So the recordings your tests already make are a golden
set you can replay against the next model version:

```bash
jevkit-drift tests/cassettes/test_routes_billing_questions.jevl candidate.jevl
```

That is the whole reason the format was defined before any of these packages.

## License

MIT
