"""``jevkit-drift`` command line interface."""

from __future__ import annotations

import argparse
import json
import sys

from jevkit_core import RecordFormatError, read_records

from .compare import compare_sets

EXIT_OK = 0
EXIT_DRIFTED = 1
EXIT_USAGE = 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="jevkit-drift",
        description="Compare two .jevl runs of the same requests and report which "
                    "decisions flipped. Never calls the API.",
    )
    parser.add_argument("baseline", help="the golden set")
    parser.add_argument("candidate", help="the same requests answered by another model")
    parser.add_argument("--max-flips", type=int, default=0,
                        help="tolerated number of flipped decisions (default 0)")
    parser.add_argument("--max-shift", type=float, default=None,
                        help="fail if any distribution moves more than this (0..1)")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--quiet", action="store_true", help="only print the summary")
    args = parser.parse_args(argv)

    try:
        before = list(read_records(args.baseline))
        after = list(read_records(args.candidate))
    except (OSError, RecordFormatError) as exc:
        print(f"jevkit-drift: {exc}", file=sys.stderr)
        return EXIT_USAGE

    try:
        report = compare_sets(before, after)
    except ValueError as exc:
        print(f"jevkit-drift: {exc}", file=sys.stderr)
        return EXIT_USAGE

    if args.format == "json":
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.summary())
        if report.flips and not args.quiet:
            print("\nflipped decisions:")
            for request_id, delta in report.flips:
                print(f"  {request_id[7:19]}  {delta.describe()}")

    if not report.deltas and (before or after):
        print("jevkit-drift: no requests matched between the two files. Records pair on "
              "state and questions, so a change to either makes them incomparable.",
              file=sys.stderr)
        return EXIT_USAGE

    if len(report.flips) > args.max_flips:
        return EXIT_DRIFTED
    if args.max_shift is not None and report.max_shift > args.max_shift:
        return EXIT_DRIFTED
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
