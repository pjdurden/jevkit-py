"""``jevkit-lint`` command line interface."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from jevkit_core import RecordFormatError, read_records

from .diagnostic import Severity
from .linter import lint
from .rules import RULES

EXIT_OK = 0
EXIT_FINDINGS = 1
EXIT_USAGE = 2


def _load_payload(path: str) -> list[tuple[str, Any, dict[str, Any]]]:
    """Return (label, state, questions) triples from a file.

    Accepts a `.jevl` record file, a request object with `state` and
    `questions`, or a bare questions object.
    """
    if path == "-":
        text = sys.stdin.read()
        source = "<stdin>"
    else:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        source = path

    if path.endswith(".jevl"):
        import io
        out = []
        for i, record in enumerate(read_records(io.StringIO(text))):
            out.append((f"{source}#{i}", record.state, record.questions))
        return out

    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"{source}: expected a JSON object at the top level")

    if "questions" in data and isinstance(data["questions"], dict):
        return [(source, data.get("state", ""), data["questions"])]
    return [(source, "", data)]


def _print_rules() -> None:
    width = max(len(r.name) for r in RULES)
    print("code    name" + " " * (width - 4) + "  documented failure mode")
    print("-" * (8 + width + 2 + 40))
    for rule in RULES:
        print(f"{rule.code}  {rule.name:<{width}}  {rule.mode}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="jevkit-lint",
        description="Statically lint TypeSafe Jev questions against the documented "
                    "jev-1.13 failure modes. Never calls the API.",
    )
    parser.add_argument("files", nargs="*",
                        help="JSON request files, .jevl record files, or - for stdin")
    parser.add_argument("--select", metavar="CODES",
                        help="comma-separated rule codes to run exclusively")
    parser.add_argument("--ignore", metavar="CODES",
                        help="comma-separated rule codes to skip")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--strict", action="store_true",
                        help="exit non-zero on warnings too, not just errors")
    parser.add_argument("--list-rules", action="store_true", help="list rules and exit")
    parser.add_argument("--no-color", action="store_true")
    args = parser.parse_args(argv)

    if args.list_rules:
        _print_rules()
        return EXIT_OK

    if not args.files:
        parser.error("no input files (use - to read stdin, or --list-rules)")

    select = args.select.split(",") if args.select else None
    ignore = args.ignore.split(",") if args.ignore else None
    color = sys.stdout.isatty() and not args.no_color

    payloads: list[tuple[str, Any, dict[str, Any]]] = []
    for path in args.files:
        try:
            payloads.extend(_load_payload(path))
        except (OSError, ValueError, RecordFormatError) as exc:
            print(f"jevkit-lint: {exc}", file=sys.stderr)
            return EXIT_USAGE

    reports = []
    worst = Severity.INFO
    any_error = any_warning = False

    for label, state, questions in payloads:
        try:
            result = lint(questions, state, select=select, ignore=ignore)
        except ValueError as exc:
            print(f"jevkit-lint: {label}: {exc}", file=sys.stderr)
            return EXIT_USAGE
        any_error = any_error or bool(result.errors)
        any_warning = any_warning or bool(result.warnings)
        reports.append((label, result))

    if args.format == "json":
        print(json.dumps(
            {"results": [{"source": label, **result.to_dict()} for label, result in reports]},
            indent=2,
        ))
    else:
        for label, result in reports:
            if len(payloads) > 1:
                print(f"== {label}")
            print(result.format(color=color))
            if len(payloads) > 1:
                print()
        total = {"error": 0, "warning": 0, "info": 0}
        for _, result in reports:
            for key, value in result.counts().items():
                total[key] += value
        if sum(total.values()):
            print(f"\n{total['error']} error(s), {total['warning']} warning(s), "
                  f"{total['info']} info")

    del worst
    if any_error:
        return EXIT_FINDINGS
    if args.strict and any_warning:
        return EXIT_FINDINGS
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
