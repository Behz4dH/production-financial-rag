"""Run the benchmark across modes; print a table and write eval_results.json."""

import json
import sys
from pathlib import Path

from core.config import get_settings
from eval.golden import load_golden
from eval.metrics import aggregate, refusal_accuracy
from eval.runner import run_mode
from rag.query import answer, build_deps

_MODES = ["basic", "hybrid"]


def build_report(items, modes, deps, answer_fn=answer) -> dict:
    report = {"modes": {}, "refusal_accuracy": {}, "rows": {}}
    for mode in modes:
        rows = run_mode(items, mode, deps, answer_fn=answer_fn)
        report["modes"][mode] = aggregate(rows)
        report["refusal_accuracy"][mode] = refusal_accuracy(rows)
        report["rows"][mode] = [r.model_dump() for r in rows]
    return report


def render_table(report: dict) -> str:
    modes = list(report["modes"])
    categories = sorted({c for m in modes for c in report["modes"][m]["by_category"]})
    width = 16
    header = "category".ljust(width) + "".join(m.ljust(width) for m in modes)
    lines = [header, "-" * len(header)]
    for cat in categories:
        row = cat.ljust(width)
        for mode in modes:
            stats = report["modes"][mode]["by_category"].get(cat)
            cell = f"{stats['accuracy']:.0%} ({stats['correct']}/{stats['total']})" if stats else "-"
            row += cell.ljust(width)
        lines.append(row)
    overall = "OVERALL".ljust(width) + "".join(
        f"{report['modes'][m]['overall']['accuracy']:.0%}".ljust(width) for m in modes)
    lines += ["-" * len(header), overall]
    return "\n".join(lines)


def write_results(report: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    settings = get_settings()
    items = load_golden(str(Path(settings.data_dir).parent / "benchmark" / "answers.json"))
    deps = build_deps(settings)
    report = build_report(items, _MODES, deps)
    print(render_table(report))
    out = str(Path(settings.data_dir).parent / "eval_results.json")
    write_results(report, out)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
