"""``jevkit-calibrate`` command line interface."""

from __future__ import annotations

import argparse
import json
import sys

from jevkit_core import RecordFormatError, read_records

from .metrics import calibrate
from .records import observations_from_records
from .thresholds import recommend_for_accuracy, recommend_for_coverage

EXIT_OK = 0
EXIT_FAILED_GATE = 1
EXIT_USAGE = 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="jevkit-calibrate",
        description="Measure how well Jev's probabilities match outcomes on your labeled "
                    "data, and pick a confidence threshold from it. Never calls the API.",
    )
    parser.add_argument("files", nargs="+", help=".jevl files containing labeled records")
    parser.add_argument("--bins", type=int, default=10, help="reliability bins (default 10)")
    parser.add_argument("--question", action="append", dest="questions",
                        help="restrict to this question id (repeatable)")
    parser.add_argument("--use-confidence", action="store_true",
                        help="calibrate the API's confidence rather than the probability "
                             "on the chosen outcome")
    parser.add_argument("--target-accuracy", type=float,
                        help="recommend the lowest threshold reaching this accuracy")
    parser.add_argument("--min-coverage", type=float,
                        help="recommend the highest threshold still covering this fraction")
    parser.add_argument("--max-ece", type=float,
                        help="exit non-zero if ECE exceeds this")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--no-diagram", action="store_true")
    args = parser.parse_args(argv)

    records = []
    try:
        for path in args.files:
            records.extend(read_records(path))
    except (OSError, RecordFormatError) as exc:
        print(f"jevkit-calibrate: {exc}", file=sys.stderr)
        return EXIT_USAGE

    observations = observations_from_records(
        records, question_ids=args.questions, use_confidence=args.use_confidence
    )
    if not observations:
        print("jevkit-calibrate: no labeled observations found. Records need a 'label' "
              "object keyed by question id.", file=sys.stderr)
        return EXIT_USAGE

    report = calibrate(observations, n_bins=args.bins)

    recommendation = None
    if args.target_accuracy is not None:
        recommendation = ("accuracy", args.target_accuracy,
                          recommend_for_accuracy(observations, args.target_accuracy))
    elif args.min_coverage is not None:
        recommendation = ("coverage", args.min_coverage,
                          recommend_for_coverage(observations, args.min_coverage))

    if args.format == "json":
        payload = report.to_dict()
        if recommendation:
            kind, target, point = recommendation
            payload["recommendation"] = {
                "for": kind, "target": target,
                "threshold": point.threshold if point else None,
                "coverage": point.coverage if point else None,
                "accuracy": point.accuracy if point else None,
                "errors": point.errors if point else None,
            }
        print(json.dumps(payload, indent=2))
    else:
        print(report.summary())
        if not args.no_diagram:
            print()
            print(report.diagram())
        if recommendation:
            kind, target, point = recommendation
            print()
            if point is None:
                if kind == "accuracy":
                    print(f"No threshold reaches {target:.1%} accuracy on this data. The "
                          f"question cannot be automated at that bar; change the question "
                          f"rather than the threshold.")
                else:
                    print(f"No threshold covers {target:.1%} of cases.")
            else:
                print(f"threshold {point.threshold:.2f}: covers {point.coverage:.1%} "
                      f"({point.covered}/{point.total}) at {point.accuracy:.1%} accuracy, "
                      f"{point.errors} wrong answer(s) acted on, {point.escalated} escalated")

    if args.max_ece is not None and report.ece > args.max_ece:
        print(f"jevkit-calibrate: ECE {report.ece:.4f} exceeds {args.max_ece}", file=sys.stderr)
        return EXIT_FAILED_GATE
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
