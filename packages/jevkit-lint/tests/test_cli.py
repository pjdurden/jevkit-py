import json
from jevkit_lint.cli import EXIT_FINDINGS, EXIT_OK, EXIT_USAGE, main

CLEAN = {"questions": {"q": {"type": "choice", "instructions": "Which team handles this",
                             "criteria": {"billing": "Payment issues", "unknown": "None apply"}}}}
DIRTY = {"questions": {"q": {"type": "noul", "instructions": "How many items are there"}}}


def write(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return str(path)


def test_clean_file_exits_zero(tmp_path, capsys):
    assert main([write(tmp_path, "a.json", CLEAN)]) == EXIT_OK


def test_file_with_errors_exits_one(tmp_path, capsys):
    assert main([write(tmp_path, "a.json", DIRTY)]) == EXIT_FINDINGS


def test_strict_turns_warnings_into_failure(tmp_path, capsys):
    payload = {"questions": {"q": {"type": "choice", "instructions": "Pick a team",
                                   "criteria": {"a": "Team A", "b": "Team B"}}}}
    path = write(tmp_path, "a.json", payload)
    assert main([path]) == EXIT_OK
    assert main([path, "--strict"]) == EXIT_FINDINGS


def test_json_output_is_parsable(tmp_path, capsys):
    main([write(tmp_path, "a.json", DIRTY), "--format", "json"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["results"][0]["ok"] is False
    assert payload["results"][0]["diagnostics"][0]["code"] == "JEV001"


def test_bare_questions_object_is_accepted(tmp_path, capsys):
    assert main([write(tmp_path, "a.json", DIRTY["questions"])]) == EXIT_FINDINGS


def test_list_rules_exits_zero(capsys):
    assert main(["--list-rules"]) == EXIT_OK
    assert "JEV001" in capsys.readouterr().out


def test_missing_file_is_a_usage_error(capsys):
    assert main(["/nonexistent/path.json"]) == EXIT_USAGE


def test_jevl_input_is_linted(tmp_path, capsys):
    from jevkit_core import Record, write_records
    path = tmp_path / "a.jevl"
    write_records(path, [Record(model="jev-1.13.0", state="s",
                                questions={"q": {"type": "noul",
                                                 "instructions": "How many items are there"}},
                                answers={"q": {"type": "noul", "noul": 0.5}})])
    assert main([str(path)]) == EXIT_FINDINGS
