"""Summarize stage-2 pytest numeric evidence, rejecting failed/skipped runs.

Usage: python scripts/c7/summarize_attention.py <pytest-output.txt>
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    lines = args.log.read_text().splitlines()
    summary = next((line for line in reversed(lines) if " passed" in line), "")
    if not summary or re.search(r"\b[1-9]\d* (?:failed|skipped|errors?)\b", summary):
        raise RuntimeError("Expected a completed pytest run with passing tests and no skips/errors")
    numeric, memory = [], []
    for line in lines:
        for prefix, records in [("C7_NUMERIC ", numeric), ("C7_MEMORY ", memory)]:
            if prefix in line:
                records.append(json.loads(line.split(prefix, 1)[1]))
    if not numeric or not memory:
        raise RuntimeError("Missing real numeric or GPU memory evidence")
    if any(row["violations"] or row["atol"] != 0.02 or row["rtol"] != 0.02 for row in numeric):
        raise RuntimeError("Numeric violation or tolerance differs from the stage-1 contract")

    groups = {}
    for row in numeric:
        groups.setdefault(row["name"].split("/", 1)[0], []).append(row)
    result = {
        "log": str(args.log),
        "pytest_summary": summary,
        "comparison_count": len(numeric),
        "tolerance": "abs(actual-reference) <= 0.02 + 0.02 * abs(reference)",
        "violating_elements": sum(row["violations"] for row in numeric),
        "peak_allocated_bytes": max(row["peak_allocated_bytes"] for row in memory),
        "peak_reserved_bytes": max(row["peak_reserved_bytes"] for row in memory),
        "groups": {
            name: {
                "comparisons": len(rows),
                "max_abs": max(row["max_abs"] for row in rows),
                "max_of_per_comparison_abs_p99": max(row["abs_p50_p95_p99"][2] for row in rows),
                "max_relative_floor_1e-6": max(row["max_relative_floor_1e-6"] for row in rows),
            }
            for name, rows in groups.items()
        },
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
