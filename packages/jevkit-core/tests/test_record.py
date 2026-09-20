import io
import json
import pytest
from jevkit_core import (FORMAT_VERSION, Record, RecordFormatError, load_cassette,
                         read_records, write_records)


def make(model="jev-1.13.0", state="s", answers=None):
    return Record(model=model, state=state,
                  questions={"q": {"type": "noul", "instructions": "ok?"}},
                  answers=answers or {"q": {"type": "noul", "noul": 0.9}})


def test_id_is_derived_when_absent():
    assert make().id.startswith("sha256:")


def test_round_trip_preserves_fields(tmp_path):
    path = tmp_path / "a.jevl"
    original = make()
    original.tags = ["x"]
    original.usage = {"input_tokens": 10}
    assert write_records(path, [original]) == 1
    (restored,) = list(read_records(path))
    assert restored.id == original.id
    assert restored.model == original.model
    assert restored.tags == ["x"]
    assert restored.usage == {"input_tokens": 10}


def test_unknown_keys_survive_round_trip(tmp_path):
    path = tmp_path / "a.jevl"
    raw = make().to_dict()
    raw["future_field"] = {"kept": True}
    path.write_text(json.dumps(raw) + "\n")
    (record,) = list(read_records(path))
    assert record.extra["future_field"] == {"kept": True}
    assert record.to_dict()["future_field"] == {"kept": True}


def test_newer_format_version_is_refused(tmp_path):
    path = tmp_path / "a.jevl"
    raw = make().to_dict()
    raw["v"] = FORMAT_VERSION + 1
    path.write_text(json.dumps(raw) + "\n")
    with pytest.raises(RecordFormatError, match="newer than this reader"):
        list(read_records(path))


def test_malformed_line_raises_rather_than_skipping(tmp_path):
    path = tmp_path / "a.jevl"
    path.write_text(json.dumps(make().to_dict()) + "\nnot json\n")
    with pytest.raises(RecordFormatError, match="invalid JSON"):
        list(read_records(path))


def test_missing_required_key_raises(tmp_path):
    path = tmp_path / "a.jevl"
    raw = make().to_dict()
    del raw["answers"]
    path.write_text(json.dumps(raw) + "\n")
    with pytest.raises(RecordFormatError, match="missing required key"):
        list(read_records(path))


def test_blank_lines_are_skipped(tmp_path):
    path = tmp_path / "a.jevl"
    path.write_text("\n" + json.dumps(make().to_dict()) + "\n\n")
    assert len(list(read_records(path))) == 1


def test_cassette_last_record_wins(tmp_path):
    path = tmp_path / "a.jevl"
    first = make(answers={"q": {"type": "noul", "noul": 0.1}})
    second = make(answers={"q": {"type": "noul", "noul": 0.9}})
    assert first.id == second.id
    write_records(path, [first, second])
    table = load_cassette(path)
    assert len(table) == 1
    assert table[first.id].answers["q"]["noul"] == 0.9


def test_request_id_pairs_records_across_model_versions():
    a, b = make(model="jev-1.13.0"), make(model="jev-1.14.0")
    assert a.id != b.id
    assert a.request_id == b.request_id


def test_reads_from_a_stream():
    text = json.dumps(make().to_dict()) + "\n"
    assert len(list(read_records(io.StringIO(text)))) == 1
