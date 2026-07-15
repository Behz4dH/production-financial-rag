"""eval.gate: the regression gate that fails CI when quality drops."""

import json

import pytest

from eval.gate import baseline_from_report, compare_to_baseline, main


def _report(basic_overall=0.875, basic_refusal=1.0, hybrid_overall=0.85, hybrid_refusal=1.0):
    """An eval_results.json-shaped report (only the parts the gate reads)."""
    return {
        "modes": {
            "basic": {"overall": {"accuracy": basic_overall, "correct": 35, "total": 40}},
            "hybrid": {"overall": {"accuracy": hybrid_overall, "correct": 34, "total": 40}},
        },
        "refusal_accuracy": {"basic": basic_refusal, "hybrid": hybrid_refusal},
    }


def test_baseline_from_report_extracts_per_mode_metrics():
    baseline = baseline_from_report(_report())
    assert baseline["modes"]["basic"] == {"overall_accuracy": 0.875, "refusal_accuracy": 1.0}
    assert baseline["modes"]["hybrid"] == {"overall_accuracy": 0.85, "refusal_accuracy": 1.0}


def test_identical_run_passes():
    baseline = baseline_from_report(_report())
    assert compare_to_baseline(_report(), baseline) == []


def test_improvement_passes():
    baseline = baseline_from_report(_report())
    current = _report(basic_overall=0.95, hybrid_overall=0.9)
    assert compare_to_baseline(current, baseline) == []


def test_overall_drop_within_tolerance_passes():
    baseline = baseline_from_report(_report())
    current = _report(basic_overall=0.85)  # -0.025, one question of 40
    assert compare_to_baseline(current, baseline, overall_tolerance=0.03) == []


def test_overall_drop_beyond_tolerance_fails():
    baseline = baseline_from_report(_report())
    current = _report(basic_overall=0.80)  # -0.075
    failures = compare_to_baseline(current, baseline, overall_tolerance=0.03)
    assert len(failures) == 1
    assert "basic" in failures[0] and "overall" in failures[0]


def test_any_refusal_drop_fails():
    """Refusal accuracy is the safety metric: zero tolerance by default."""
    baseline = baseline_from_report(_report())
    current = _report(hybrid_refusal=0.947)  # one hallucination question flipped
    failures = compare_to_baseline(current, baseline)
    assert len(failures) == 1
    assert "hybrid" in failures[0] and "refusal" in failures[0]


def test_baseline_mode_missing_from_current_run_fails():
    """Dropping a mode silently would make the gate blind to it."""
    baseline = baseline_from_report(_report())
    current = _report()
    del current["modes"]["hybrid"]
    failures = compare_to_baseline(current, baseline)
    assert any("hybrid" in f and "missing" in f for f in failures)


def test_new_mode_not_in_baseline_is_ignored():
    baseline = baseline_from_report(_report())
    current = _report()
    current["modes"]["experimental"] = {"overall": {"accuracy": 0.1}}
    current["refusal_accuracy"]["experimental"] = 0.0
    assert compare_to_baseline(current, baseline) == []


# --- CLI ---

def _write(path, obj):
    path.write_text(json.dumps(obj), encoding="utf-8")


def test_main_passes_and_exits_zero(tmp_path, capsys):
    results = tmp_path / "eval_results.json"
    baseline = tmp_path / "baseline.json"
    _write(results, _report())
    _write(baseline, baseline_from_report(_report()))
    code = main(["--results", str(results), "--baseline", str(baseline)])
    assert code == 0
    assert "PASS" in capsys.readouterr().out


def test_main_fails_and_exits_one_on_regression(tmp_path, capsys):
    results = tmp_path / "eval_results.json"
    baseline = tmp_path / "baseline.json"
    _write(results, _report(basic_refusal=0.9))
    _write(baseline, baseline_from_report(_report()))
    code = main(["--results", str(results), "--baseline", str(baseline)])
    assert code == 1
    assert "FAIL" in capsys.readouterr().out


def test_main_update_baseline_writes_file(tmp_path):
    results = tmp_path / "eval_results.json"
    baseline = tmp_path / "baseline.json"
    _write(results, _report())
    code = main(["--results", str(results), "--baseline", str(baseline), "--update-baseline"])
    assert code == 0
    written = json.loads(baseline.read_text(encoding="utf-8"))
    assert written == baseline_from_report(_report())


def test_main_missing_results_exits_two(tmp_path, capsys):
    baseline = tmp_path / "baseline.json"
    _write(baseline, baseline_from_report(_report()))
    code = main(["--results", str(tmp_path / "nope.json"), "--baseline", str(baseline)])
    assert code == 2
    assert "make eval" in capsys.readouterr().out


def test_main_missing_baseline_exits_two_with_hint(tmp_path, capsys):
    results = tmp_path / "eval_results.json"
    _write(results, _report())
    code = main(["--results", str(results), "--baseline", str(tmp_path / "nope.json")])
    assert code == 2
    assert "--update-baseline" in capsys.readouterr().out
