import json
import pytest
from jevkit_core import canonical_json, digest, record_id, request_id


def test_key_order_does_not_change_canonical_form():
    assert canonical_json({"b": 1, "a": 2}) == canonical_json({"a": 2, "b": 1})


def test_canonical_form_has_no_insignificant_whitespace():
    assert canonical_json({"a": [1, 2]}) == b'{"a":[1,2]}'


def test_canonical_form_is_utf8_not_escaped():
    assert canonical_json({"k": "café"}) == '{"k":"café"}'.encode("utf-8")


def test_non_finite_numbers_are_rejected():
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            canonical_json({"x": bad})
    with pytest.raises(ValueError):
        canonical_json({"nested": {"deep": [float("nan")]}})


def test_digest_is_stable_and_prefixed():
    d = digest({"a": 1})
    assert d.startswith("sha256:") and len(d) == 71
    assert d == digest({"a": 1})


def test_request_id_ignores_model_but_record_id_does_not():
    state, questions = "hello", {"q": {"type": "noul", "instructions": "ok?"}}
    assert record_id("jev-1.13.0", state, questions) != record_id("jev-1.14.0", state, questions)
    assert request_id(state, questions) == request_id(state, questions)


def test_record_id_changes_when_state_changes():
    q = {"q": {"type": "noul", "instructions": "ok?"}}
    assert record_id("m", "a", q) != record_id("m", "b", q)
