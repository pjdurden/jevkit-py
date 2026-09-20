import json
import pytest
from jevkit_core import Record, read_records
from jevkit_pytest import Cassette, CassetteMiss, assert_answer, assert_confident

QUESTIONS = {"team": {"type": "choice", "instructions": "Which team handles this"}}
STATE = "I was charged twice"

LIVE = {"model": "jev-1.13.0",
        "answers": {"team": {"type": "choice", "choice": "billing",
                             "probabilities": {"billing": 0.9, "technical": 0.1},
                             "confidence": 0.8}},
        "usage": {"input_tokens": 12}}


def client(calls):
    def system_one(state, questions, **kw):
        calls.append((state, questions))
        return LIVE
    return system_one


def test_replay_mode_raises_on_a_miss(tmp_path):
    cassette = Cassette(tmp_path / "c.jevl", mode="replay")
    with pytest.raises(CassetteMiss, match="no recording"):
        cassette.system_one(STATE, QUESTIONS)


def test_auto_mode_records_then_replays(tmp_path):
    path = tmp_path / "c.jevl"
    calls = []

    first = Cassette(path, client(calls), mode="auto")
    first.system_one(STATE, QUESTIONS)
    assert len(calls) == 1
    assert path.exists()

    second = Cassette(path, client(calls), mode="auto")
    response = second.system_one(STATE, QUESTIONS)
    assert len(calls) == 1, "second run must not call through"
    assert response.answers["team"]["choice"] == "billing"


def test_recording_returns_the_live_response_unchanged(tmp_path):
    cassette = Cassette(tmp_path / "c.jevl", client([]), mode="auto")
    assert cassette.system_one(STATE, QUESTIONS) is LIVE


def test_replayed_answers_support_both_key_and_attribute_access(tmp_path):
    path = tmp_path / "c.jevl"
    Cassette(path, client([]), mode="auto").system_one(STATE, QUESTIONS)
    response = Cassette(path, mode="replay").system_one(STATE, QUESTIONS)
    assert response.answers["team"]["choice"] == "billing"
    assert response.answers["team"].choice == "billing"
    assert response.answers["team"].confidence == 0.8


def test_unknown_field_gives_a_helpful_attribute_error(tmp_path):
    path = tmp_path / "c.jevl"
    Cassette(path, client([]), mode="auto").system_one(STATE, QUESTIONS)
    response = Cassette(path, mode="replay").system_one(STATE, QUESTIONS)
    with pytest.raises(AttributeError, match="recorded fields are"):
        _ = response.answers["team"].nonsense


def test_a_changed_question_is_a_miss(tmp_path):
    path = tmp_path / "c.jevl"
    Cassette(path, client([]), mode="auto").system_one(STATE, QUESTIONS)
    reworded = {"team": {"type": "choice", "instructions": "Which department handles this"}}
    with pytest.raises(CassetteMiss):
        Cassette(path, mode="replay").system_one(STATE, reworded)


def test_record_mode_always_calls_through(tmp_path):
    path = tmp_path / "c.jevl"
    calls = []
    Cassette(path, client(calls), mode="auto").system_one(STATE, QUESTIONS)
    Cassette(path, client(calls), mode="record").system_one(STATE, QUESTIONS)
    assert len(calls) == 2


def test_passthrough_never_writes(tmp_path):
    path = tmp_path / "c.jevl"
    Cassette(path, client([]), mode="passthrough").system_one(STATE, QUESTIONS)
    assert not path.exists()


def test_recording_without_a_client_explains_itself(tmp_path):
    cassette = Cassette(tmp_path / "c.jevl", mode="auto")
    with pytest.raises(CassetteMiss, match="no client was supplied"):
        cassette.system_one(STATE, QUESTIONS)


def test_invalid_mode_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="mode must be one of"):
        Cassette(tmp_path / "c.jevl", mode="nonsense")


def test_unplayed_entries_are_reported(tmp_path):
    path = tmp_path / "c.jevl"
    Cassette(path, client([]), mode="auto").system_one(STATE, QUESTIONS)
    cassette = Cassette(path, mode="replay")
    assert len(cassette.unplayed) == 1
    cassette.system_one(STATE, QUESTIONS)
    assert cassette.unplayed == []


def test_a_cassette_is_a_valid_jevl_golden_set(tmp_path):
    path = tmp_path / "c.jevl"
    Cassette(path, client([]), mode="auto").system_one(STATE, QUESTIONS)
    (record,) = list(read_records(path))
    assert isinstance(record, Record)
    assert record.model == "jev-1.13.0"
    assert record.usage == {"input_tokens": 12}


def test_contains_reports_membership(tmp_path):
    path = tmp_path / "c.jevl"
    cassette = Cassette(path, client([]), mode="auto")
    assert not cassette.contains(STATE, QUESTIONS)
    cassette.system_one(STATE, QUESTIONS)
    assert Cassette(path, mode="replay").contains(STATE, QUESTIONS)


def test_sdk_style_response_objects_are_recorded(tmp_path):
    class Answer:
        def __init__(self):
            self.type, self.choice = "choice", "billing"
            self.probabilities, self.confidence = {"billing": 1.0}, 0.9

    class Response:
        model = "jev-1.13.0"
        answers = {"team": Answer()}
        usage = None

    path = tmp_path / "c.jevl"
    Cassette(path, lambda s, q, **k: Response(), mode="auto").system_one(STATE, QUESTIONS)
    (record,) = list(read_records(path))
    assert record.answers["team"]["choice"] == "billing"


def test_recording_under_an_alias_replays_under_the_same_alias(tmp_path):
    """Regression: the record stores the answering model (jev-1.13.0) while the
    test asks for the alias (jev-latest). Indexing on the answering model made
    every replay a miss."""
    path = tmp_path / "c.jevl"
    calls = []
    Cassette(path, client(calls), mode="auto", model="jev-latest").system_one(STATE, QUESTIONS)

    replayed = Cassette(path, client(calls), mode="auto", model="jev-latest")
    replayed.system_one(STATE, QUESTIONS)
    assert len(calls) == 1, "replay must not call through"

    (record,) = list(read_records(path))
    assert record.model == "jev-1.13.0", "the record states who actually answered"
    assert record.meta["requested_model"] == "jev-latest"


def test_a_different_requested_model_is_a_separate_entry(tmp_path):
    path = tmp_path / "c.jevl"
    calls = []
    Cassette(path, client(calls), mode="auto", model="jev-latest").system_one(STATE, QUESTIONS)
    Cassette(path, client(calls), mode="auto", model="jev-1.14.0").system_one(STATE, QUESTIONS)
    assert len(calls) == 2
