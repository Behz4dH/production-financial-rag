"""Regression gate: compare a fresh eval run against the accepted baseline.

The baseline (eval/baseline.json, committed) is the accepted quality bar; a
run of `make eval` writes data/eval_results.json, and this gate fails (exit 1)
when quality regressed:

- refusal accuracy (the hallucination-resistance safety metric) may not drop
  AT ALL by default;
- overall accuracy may drop at most --overall-tolerance (default 0.03 — one
  flipped question on the 40-item benchmark, absorbing LLM nondeterminism);
- a mode present in the baseline must be present in the run.

Improvements pass; to make them the new bar, re-run with --update-baseline
and commit the changed baseline.json (so the raise is itself reviewed).
"""

import argparse
import json
from pathlib import Path

_DEFAULT_RESULTS = "data/eval_results.json"
_DEFAULT_BASELINE = "eval/baseline.json"


def baseline_from_report(report: dict) -> dict:
    """Shrink an eval_results report to the two gated metrics per mode."""
    return {
        "modes": {
            mode: {
                "overall_accuracy": agg["overall"]["accuracy"],
                "refusal_accuracy": report["refusal_accuracy"][mode],
            }
            for mode, agg in report["modes"].items()
        }
    }


def compare_to_baseline(report: dict, baseline: dict, *,
                        overall_tolerance: float = 0.03,
                        refusal_tolerance: float = 0.0) -> list[str]:
    """Failure messages for every gated metric that regressed; [] passes.

    Modes in the report but not the baseline are ignored — a new mode has no
    accepted bar yet and shouldn't block; it enters the gate when the baseline
    is next updated.
    """
    failures: list[str] = []
    current = baseline_from_report(report)["modes"] if report.get("modes") else {}
    for mode, accepted in baseline["modes"].items():
        got = current.get(mode)
        if got is None:
            failures.append(f"{mode}: mode missing from this eval run "
                            "(baseline still gates it)")
            continue
        drop = accepted["overall_accuracy"] - got["overall_accuracy"]
        if drop > overall_tolerance:
            failures.append(
                f"{mode}: overall accuracy {got['overall_accuracy']:.1%} is "
                f"{drop:.1%} below baseline {accepted['overall_accuracy']:.1%} "
                f"(tolerance {overall_tolerance:.1%})")
        refusal_drop = accepted["refusal_accuracy"] - got["refusal_accuracy"]
        if refusal_drop > refusal_tolerance:
            failures.append(
                f"{mode}: refusal accuracy {got['refusal_accuracy']:.1%} is below "
                f"baseline {accepted['refusal_accuracy']:.1%} - hallucination "
                "resistance regressed")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=_DEFAULT_RESULTS,
                        help="eval report to check (written by `make eval`)")
    parser.add_argument("--baseline", default=_DEFAULT_BASELINE,
                        help="accepted baseline to gate against")
    parser.add_argument("--overall-tolerance", type=float, default=0.03,
                        help="max allowed overall-accuracy drop per mode")
    parser.add_argument("--refusal-tolerance", type=float, default=0.0,
                        help="max allowed refusal-accuracy drop per mode")
    parser.add_argument("--update-baseline", action="store_true",
                        help="accept the current results as the new baseline")
    args = parser.parse_args(argv)

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"gate: {results_path} not found - run `make eval` first")
        return 2
    report = json.loads(results_path.read_text(encoding="utf-8"))

    if args.update_baseline:
        baseline = baseline_from_report(report)
        Path(args.baseline).write_text(
            json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
        print(f"gate: baseline updated from {results_path} -> {args.baseline}")
        return 0

    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        print(f"gate: {baseline_path} not found - create one with "
              "`python -m eval.gate --update-baseline`")
        return 2
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    failures = compare_to_baseline(
        report, baseline,
        overall_tolerance=args.overall_tolerance,
        refusal_tolerance=args.refusal_tolerance)
    if failures:
        print("gate: FAIL - quality regressed vs baseline:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("gate: PASS - no regression vs baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
