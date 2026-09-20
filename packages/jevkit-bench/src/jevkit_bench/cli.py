"""``jevkit-bench`` command line interface."""

from __future__ import annotations

import argparse
import json
import sys

from jevkit_core import RecordFormatError, read_records

from .score import PRICE_PER_MTOK, compare_suites, score_records

EXIT_OK = 0
EXIT_FAILED_GATE = 1
EXIT_USAGE = 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="jevkit-bench",
        description="Score a labeled .jevl suite for accuracy and cost, and compare two "
                    "runs of it. Never calls the API.",
    )
    parser.add_argument("suite", help="scored .jevl suite")
    parser.add_argument("--baseline", help="a previous run, to compare against")
    parser.add_argument("--question", action="append", dest="questions",
                        help="restrict to this question id (repeatable)")
    parser.add_argument("--price-per-mtok", type=float, default=PRICE_PER_MTOK,
                        help=f"input price per million tokens (default {PRICE_PER_MTOK})")
    parser.add_argument("--min-accuracy", type=float, help="exit non-zero below this")
    parser.add_argument("--max-regressions", type=int,
                        help="exit non-zero above this many regressions (needs --baseline)")
    parser.add_argument("--show-failures", type=int, default=0, metavar="N",
                        help="print the N most confident wrong answers")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)

    if args.max_regressions is not None and not args.baseline:
        print("jevkit-bench: --max-regressions needs --baseline", file=sys.stderr)
        return EXIT_USAGE

    try:
        suite = score_records(read_records(args.suite), question_ids=args.questions,
                              price_per_mtok=args.price_per_mtok)
        baseline = (
            score_records(read_records(args.baseline), question_ids=args.questions,
                          price_per_mtok=args.price_per_mtok)
            if args.baseline else None
        )
    except (OSError, RecordFormatError) as exc:
        print(f"jevkit-bench: {exc}", file=sys.stderr)
        return EXIT_USAGE

    if not suite.count:
        print("jevkit-bench: nothing scored. Records need a 'label' object keyed by "
              "question id.", file=sys.stderr)
        return EXIT_USAGE

    comparison = compare_suites(baseline, suite) if baseline else None

    if args.format == "json":
        payload = suite.to_dict()
        if comparison:
            payload["comparison"] = {
                "baseline_accuracy": comparison.baseline.accuracy,
                "accuracy_delta": comparison.accuracy_delta,
                "cost_delta": comparison.cost_delta,
                "regressions": [list(k) for k in comparison.regressions],
                "fixes": [list(k) for k in comparison.fixes],
            }
        print(json.dumps(payload, indent=2))
    else:
        print(suite.summary())
        if comparison:
            print()
            print(comparison.summary())
            if comparison.regressions:
                print("\nregressions:")
                for rid, qid in comparison.regressions[:20]:
                    print(f"  {rid[7:19]}  {qid}")
        if args.show_failures:
            failures = suite.failures()[: args.show_failures]
            if failures:
                print(f"\nmost confident wrong answers ({len(failures)}):")
                for f in failures:
                    # JSON rendering keeps this byte-identical to the
                    # JavaScript CLI; repr() would quote with '.
                    print(f"  {f.request_id[7:19]}  {f.question_id}: "
                          f"said {json.dumps(f.predicted)}, "
                          f"label {json.dumps(f.label)}, p={f.probability:.3f}")

    if args.min_accuracy is not None and suite.accuracy < args.min_accuracy:
        print(f"jevkit-bench: accuracy {suite.accuracy:.4f} below {args.min_accuracy}",
              file=sys.stderr)
        return EXIT_FAILED_GATE
    if args.max_regressions is not None and comparison is not None:
        if len(comparison.regressions) > args.max_regressions:
            print(f"jevkit-bench: {len(comparison.regressions)} regressions exceed "
                  f"{args.max_regressions}", file=sys.stderr)
            return EXIT_FAILED_GATE
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
