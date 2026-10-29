"""Audit bounded stage-4 evidence, keeping unmatched numerical failures explicit.

Run after the matrix, numeric, HTTP, pytest, and diagnostic commands in the report.
This reads evidence only; it neither runs the model nor measures performance.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


def main() -> None:
    evidence = Path(__file__).resolve().parents[2] / "docs/c7/evidence"

    def records(filename, prefix):
        return [
            json.loads(line.split(prefix, 1)[1])
            for line in (evidence / filename).read_text().splitlines()
            if prefix in line
        ]

    pytest_results = {}
    for suite, expected in (("cpu", 73), ("execution", 12), ("attention", 22)):
        text = (evidence / f"stage4-{suite}-tests.txt").read_text()
        summary = next(line for line in reversed(text.splitlines()) if " passed" in line)
        assert re.search(rf"\b{expected} passed\b", summary), summary
        assert not re.search(r"\b[1-9]\d* (?:failed|skipped|errors?)\b", summary), summary
        pytest_results[suite] = summary

    matrix = json.loads((evidence / "stage4-matrix-summary.txt").read_text())
    assert matrix["status"] == "PASS" and not matrix["failures"]
    assert matrix["case_count"] == 72 and len(matrix["runs"]) == 9
    assert matrix["request_set_identical"]
    cases = []
    for run in matrix["runs"]:
        assert run["exit_code"] == 0 and run["cases"] == 8
        filename = f"stage4-matrix-{run['tag']}.txt"
        rows = records(filename, "C7_STAGE4_CASE ")
        assert len(rows) == 8 and all(row["status"] == "PASS" for row in rows)
        graph, mixed = (run["summary"]["config"][key] for key in ("graph", "mixed"))
        assert bool(sum(row["graph_count"] for row in rows)) == graph
        assert bool(sum(row["mixed_count"] for row in rows)) == mixed
        if mixed:
            assert not any(row["legacy_extra_finished"] for row in rows)
        cases.extend(rows)

    numeric_file = "stage4-numeric-initial.txt"
    numeric = records(numeric_file, "C7_STAGE4_NUMERIC ")
    numeric_summary = records(numeric_file, "C7_STAGE4_NUMERIC_PASS ")[-1]
    assert len(numeric) == numeric_summary["comparisons"] == 448
    assert numeric_summary["violations"] == 0
    assert Counter(row["kind"] for row in numeric) == {
        "fp32-attention": 420,
        "fp32-head-projection": 14,
        "matched-full-forward-logits": 14,
    }
    for row in numeric:
        assert row["violations"] == 0
        tolerance = (0.02, 0.02) if row["kind"] == "fp32-attention" else (0.1, 0.01)
        assert (row["atol"], row["rtol"]) == tolerance
    for forward in range(15):
        assert {
            row["layer"]
            for row in numeric
            if row["kind"] == "fp32-attention" and row["forward"] == forward
        } == set(range(28))
    assert {row["phase"] for row in numeric} == {"mixed", "prefill", "decode"}

    attention = json.loads((evidence / "stage4-attention-summary.txt").read_text())
    assert attention["comparison_count"] == 39 and attention["violating_elements"] == 0
    http = records("stage4-http-tests.txt", "C7_STAGE4_HTTP_PASS ")[-1]
    assert http["mixed_forwards"] > 0 and len(http["checks"]) == 3
    aborts = records("stage4-http-server.txt", "C7_HTTP_ABORT ")
    assert any(
        row["was_active"] and row["remaining_active"] == 0 and row["free_slots"] == 3
        for row in aborts
    )

    diagnostic_exits = json.loads((evidence / "stage4-diagnostic-exits.txt").read_text())
    baseline = records("stage4-baseline-logits-diagnostic.txt", "BASELINE_ONLY_LOGITS ")
    prefix = records("stage4-prefix-kv-diagnostic.txt", "C7_PREFIX_LOGITS ")
    control = records("stage4-prefix-kv-diagnostic.txt", "C7_CANONICAL_CONTROL ")
    assert len(baseline) == 9 and len(prefix) == len(control) == 3
    baseline_failed = any(row["violations_atol_0.1_rtol_0.01"] for row in baseline)
    prefix_failed = any(row["violations"] for row in prefix)
    assert diagnostic_exits[0]["exit_code"] == 0  # Measurement, not numeric acceptance.
    assert diagnostic_exits[1]["exit_code"] == int(prefix_failed)
    assert all(row["max_abs"] == 0 for row in control)
    assert (evidence / "stage4-dependencies.txt").read_text() == (
        evidence / "dependencies.txt"
    ).read_text()

    result = {
        "bounded_acceptance": "PASS",
        "universal_cross_shape_or_kv_source_logit_acceptance": "NOT_ESTABLISHED",
        "pytest": pytest_results,
        "matrix_cases": len(cases),
        "matrix_user_requests": sum(row["summary"]["request_count"] for row in matrix["runs"]),
        "matrix_churn_requests": sum(row["summary"]["churn_requests"] for row in matrix["runs"]),
        "matrix_mixed_forwards": sum(row["mixed_count"] for row in cases),
        "matrix_graph_forwards": sum(row["graph_count"] for row in cases),
        "legacy_extra_finished_observations": matrix["legacy_extra_finished_observations"],
        "numeric": numeric_summary,
        "attention_regression_comparisons": attention["comparison_count"],
        "http": http,
        "raw_baseline_cross_batch_logits": "FAIL" if baseline_failed else "PASS",
        "raw_prefix_logits": "FAIL" if prefix_failed else "PASS",
        "canonical_kv_diagnostic_control": "PASS",
        "matrix_peak_allocated_bytes": max(
            row["summary"]["peak_allocated_bytes"] for row in matrix["runs"]
        ),
        "matrix_peak_reserved_bytes": max(
            row["summary"]["peak_reserved_bytes"] for row in matrix["runs"]
        ),
        "dependencies_unchanged": True,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
