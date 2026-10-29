"""Run the bounded 2x2x2 correctness matrix sequentially to limit GPU memory."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--waves", type=int, default=12)
    parser.add_argument("--stress-waves", type=int, default=80)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    evidence = root / "docs/c7/evidence"
    environment = dict(os.environ, PYTHONPATH=str(root / "python"))
    environment["PATH"] = (
        str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", "")
    )
    runs, cases, failures = [], [], []
    configurations = [
        (*values, args.waves, False) for values in itertools.product((False, True), repeat=3)
    ]
    configurations.append((True, True, True, args.stress_waves, True))
    for mixed, overlap, graph, waves, stress in configurations:
        tag = f"m{int(mixed)}-o{int(overlap)}-g{int(graph)}" + ("-stress" if stress else "")
        path = evidence / f"stage4-matrix-{tag}.txt"
        command = [
            sys.executable,
            "-u",
            "scripts/c7/stage4_matrix.py",
            "--mixed" if mixed else "--no-mixed",
            "--overlap" if overlap else "--no-overlap",
            "--graph" if graph else "--no-graph",
            "--waves",
            str(waves),
        ]
        try:
            with path.open("w") as output:
                result = subprocess.run(
                    command,
                    cwd=root,
                    env=environment,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    timeout=180,
                )
            status = result.returncode
        except subprocess.TimeoutExpired:
            status = "TIMEOUT"
        records = [
            json.loads(line.split("C7_STAGE4_CASE ", 1)[1])
            for line in path.read_text().splitlines()
            if line.startswith("C7_STAGE4_CASE ")
        ]
        summaries = [
            json.loads(line.split("C7_STAGE4_MATRIX_PASS ", 1)[1])
            for line in path.read_text().splitlines()
            if line.startswith("C7_STAGE4_MATRIX_PASS ")
        ]
        row = {
            "tag": tag,
            "command": command,
            "exit_code": status,
            "cases": len(records),
            "summary": summaries[-1] if summaries else None,
        }
        runs.append(row)
        cases.extend(dict(row, run=tag) for row in records)
        if status != 0 or len(records) != 8 or not summaries:
            failures.append(tag)
        print("C7_STAGE4_RUN " + json.dumps(row), flush=True)
    regular = [row for row in runs if not row["tag"].endswith("-stress") and row["summary"]]
    fingerprints = {row["summary"]["request_fingerprint"] for row in regular}
    if len(regular) != 8 or len(fingerprints) != 1:
        failures.append("request-set-identity")
    natural = [
        row
        for row in cases
        if row["case"].startswith("natural-") and not row["run"].endswith("-stress")
    ]
    for name in ("natural-single-cold", "natural-multi-cold"):
        values = [row["outputs"] for row in natural if row["case"] == name]
        if len(values) != 8 or any(value != values[0] for value in values):
            failures.append(name + "-greedy-equivalence")
    summary = {
        "runs": runs,
        "case_count": len(cases),
        "failures": failures,
        "request_set_identical": len(fingerprints) == 1 and len(regular) == 8,
        "legacy_extra_finished_observations": sum(row["legacy_extra_finished"] for row in cases),
        "status": "FAIL" if failures else "PASS",
    }
    (evidence / "stage4-matrix-summary.txt").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        "C7_STAGE4_MATRIX_SUMMARY "
        + json.dumps({key: value for key, value in summary.items() if key != "runs"}),
        flush=True,
    )
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
