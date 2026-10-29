"""Audit every trial, define missing metrics explicitly, and gate paired ratios."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from common import client_metrics, percentiles, read_json, validate_plan, write_json


def scheduler_metrics(events, requests, client, mode):
    mapped = {
        row["server_uid"]: requests[row["id"]]
        for row in client["results"]
        if row["server_uid"] is not None
    }
    receive = defaultdict(list)
    first_prefill, decode, tokens = {}, defaultdict(list), defaultdict(list)
    forwards, timings = [], []
    for event in events:
        uid = event.get("uid")
        if event["event"] == "receive" and uid in mapped:
            receive[uid].append(event)
        elif event["event"] == "token" and uid in mapped:
            tokens[uid].append(event)
        elif event["event"] == "forward":
            rows = [row for row in event["rows"] if row["uid"] in mapped]
            if not rows:
                continue
            forwards.append(event)
            for row in rows:
                if row["phase"] == "prefill":
                    first_prefill.setdefault(row["uid"], event["submit_t"])
                else:
                    decode[row["uid"]].append(event["submit_t"])
        elif event["event"] == "cuda_forward":
            timings.append(event)
    if not any(event["event"] == "shutdown" for event in events):
        raise ValueError("Missing clean scheduler trace shutdown")
    for uid, request in mapped.items():
        if len(receive[uid]) != 1 or receive[uid][0]["input_sha256"] != request["input_sha256"]:
            raise ValueError("Measured UID/input does not match backend tokenization")
        if receive[uid][0]["max_tokens"] != request["output_tokens"]:
            raise ValueError("Backend request length changed")
    waits = [(stamp - receive[uid][0]["t"]) * 1000 for uid, stamp in first_prefill.items()]
    intervals = [(b - a) * 1000 for stamps in decode.values() for a, b in zip(stamps, stamps[1:])]
    if any(value < 0 for value in waits + intervals):
        raise ValueError("Non-monotonic backend timeline")
    begin, end = client["epoch_perf_counter"], max(
        client["epoch_perf_counter"] + row["end_s"] for row in client["results"]
    )
    # A timed Engine call follows Scheduler input gather; select by the client
    # measurement time interval. Warmup is a separate drained client invocation.
    selected_timings = [row for row in timings if begin <= row["submit_t"] <= end]
    mixed_ms = [row["milliseconds"] for row in selected_timings if row["phase"] == "mixed"]
    graph_count = sum(row["graph"] for row in forwards)
    extra_finished = sum(
        max(0, sum(row["finished"] for row in rows) - 1) for rows in tokens.values()
    )
    if not forwards:
        raise ValueError("No measured backend forwards")
    return {
        "measured_uid_coverage": len(mapped),
        "prefill_wait_backend_ms": percentiles(waits),
        "decode_submit_interval_ms": percentiles(intervals),
        "forward_count": len(forwards),
        "phase_counts": dict(Counter(row["phase"] for row in forwards)),
        "graph_count": graph_count,
        "eager_count": len(forwards) - graph_count,
        "graph_ratio": graph_count / len(forwards),
        "mixed_engine_cuda_ms": percentiles(mixed_ms) if mode == "profile" else None,
        "cuda_timed_forwards": len(selected_timings) if mode == "profile" else None,
        "extra_finished": extra_finished,
        "backend_output_tokens": sum(len(rows) for rows in tokens.values()),
        "resident_kv_pages": percentiles([row["kv_pages"] - row["free_pages"] for row in forwards]),
        "non_evictable_kv_pages": percentiles(
            [row["kv_pages"] - row["free_pages"] - row["evictable_tokens"] for row in forwards]
        ),  # page_size=1 only
        "protected_prefix_tokens": percentiles([row["protected_tokens"] for row in forwards]),
        "allocated_bytes": percentiles([row["allocated_bytes"] for row in forwards]),
        "reserved_bytes": percentiles([row["reserved_bytes"] for row in forwards]),
        "mixed_graph_violations": sum(row["phase"] == "mixed" and row["graph"] for row in forwards),
    }


def summarize_trial(directory, plan):
    run = read_json(directory / "run.json")
    report = dict(
        run=directory.name,
        **run,
        client=None,
        scheduler=None,
        gpu_telemetry=None,
        audit_errors=[],
        performance_comparison_eligible=False,
    )
    if run["plan_sha256"] != plan["sha256"]:
        raise ValueError("Trial used a different frozen plan")
    if not (directory / "client.json").exists():
        report["audit_errors"].append("No measured client results; trial was not completed")
        return report
    client = read_json(directory / "client.json")
    if (
        client["plan_sha256"] != plan["sha256"]
        or client["warmup"]
        or client["scenario"] != run["scenario"]
    ):
        raise ValueError("Measured client provenance mismatch")
    scenario = plan["scenarios"][run["scenario"]]
    report["client"] = client_metrics(
        scenario["requests"], client["results"], scenario["window_s"], plan["slo"]
    )
    actual = read_json(directory / "server-metadata.json")
    expected = dict(
        model=plan["model"],
        dtype="torch.bfloat16",
        tp=1,
        page_size=1,
        attention_backend="fi",
        mixed=run["policy"] == "mixed",
        mode=run["mode"],
        **{key: plan["server"][key] for key in ("context", "slots", "pages", "budget", "cache")},
    )
    for key, value in expected.items():
        if actual[key] != value:
            report["audit_errors"].append(f"Actual serving config mismatch: {key}")
    graph = plan["server"]["graph"]
    expected_graphs = [] if graph == 0 else [1, 2, 4] + list(range(8, graph + 1, 8))
    if actual["graph_sizes"] != expected_graphs:
        report["audit_errors"].append("Actual Graph captures differ from the frozen plan")
    overlap_disabled = actual["environment"].get("MINISGL_DISABLE_OVERLAP_SCHEDULING")
    if overlap_disabled != ("0" if plan["server"]["overlap"] else "1"):
        report["audit_errors"].append("Overlap environment mismatch")
    report["actual_server"] = actual
    if run["mode"] != "primary":
        path = directory / "scheduler.jsonl"
        events = [json.loads(line) for line in path.read_text().splitlines()]
        report["scheduler"] = scheduler_metrics(
            events, {row["id"]: row for row in scenario["requests"]}, client, run["mode"]
        )
        if report["scheduler"]["mixed_graph_violations"]:
            report["audit_errors"].append("Mixed batch incorrectly used CUDA Graph")
        gpu_path = directory / "gpu.csv"
        with gpu_path.open() as stream:
            rows = list(csv.DictReader(stream, skipinitialspace=True))
        samples = [
            float(row["memory.used [MiB]"])
            for row in rows
            if row.get("memory.used [MiB]", "").replace(".", "", 1).isdigit()
        ]
        report["gpu_telemetry"] = {
            "scope": "whole GPU; includes warmup and drain; may include other processes",
            "samples": len(samples),
            "memory_used_mib": percentiles(samples),
        }
        if not samples:
            report["audit_errors"].append("No numeric GPU telemetry; inspect gpu.csv")
    lag = report["client"]["client_dispatch_lag_ms"]["p99"]
    if lag is None or lag > plan["max_dispatch_lag_ms"]:
        report["audit_errors"].append("Client dispatch lag exceeds frozen bound")
    if report["client"]["success_rate"] != 1:
        report["audit_errors"].append(
            "Incomplete/protocol-failed requests; no throughput speedup ratio"
        )
    if run["status"] != "COMPLETE":
        report["audit_errors"].append("Execution did not complete cleanly")
    if not run.get("code_sha256"):
        report["audit_errors"].append("Missing source fingerprint")
    report["performance_comparison_eligible"] = (
        plan["kind"] == "desktop_benchmark"
        and run["mode"] == "primary"
        and not run["nsys"]
        and not report["audit_errors"]
    )
    return report


def paired_comparisons(reports):
    grouped = defaultdict(dict)
    for row in reports:
        key = (row["scenario"], row["repeat"])
        if row["policy"] in grouped[key]:
            raise ValueError("Duplicate trial for scenario/repeat/policy")
        grouped[key][row["policy"]] = row
    comparisons = []
    for (scenario, repeat), pair in grouped.items():
        reason = None
        if set(pair) != {"baseline", "mixed"}:
            reason = "missing baseline or mixed trial"
        elif not all(row["performance_comparison_eligible"] for row in pair.values()):
            reason = "smoke, instrumented, failed, or client-limited trial; inspect raw metrics"
        else:
            baseline, mixed = pair["baseline"], pair["mixed"]
            for key in ("plan_sha256", "code_sha256", "mode", "nsys", "max_clients", "timeout_s"):
                if baseline[key] != mixed[key]:
                    reason = f"trial contract mismatch: {key}"
            for key in (
                "gpu",
                "sm",
                "total_vram_bytes",
                "versions",
                "cuda",
                "graph_sizes",
                "environment",
            ):
                if baseline["actual_server"][key] != mixed["actual_server"][key]:
                    reason = f"environment mismatch: {key}"
        row = {
            "scenario": scenario,
            "repeat": repeat,
            "withheld_reason": reason,
            "cohort_output_ratio_mixed_over_baseline": None,
            "goodput_delta_req_s": None,
        }
        if reason is None:
            base, mixed = pair["baseline"]["client"], pair["mixed"]["client"]
            denominator = base["cohort_output_token_s"]
            row["cohort_output_ratio_mixed_over_baseline"] = (
                mixed["cohort_output_token_s"] / denominator if denominator else None
            )
            row["goodput_delta_req_s"] = (
                mixed["window_goodput_req_s"] - base["window_goodput_req_s"]
            )
        comparisons.append(row)
    return comparisons


def repeat_summary(comparisons):
    grouped = defaultdict(list)
    for row in comparisons:
        grouped[row["scenario"]].append(row)
    return {
        scenario: {
            "paired_trials": len(rows),
            "withheld_trials": sum(row["withheld_reason"] is not None for row in rows),
            "cohort_output_ratio_mixed_over_baseline": percentiles(
                [
                    row["cohort_output_ratio_mixed_over_baseline"]
                    for row in rows
                    if row["withheld_reason"] is None
                ]
            ),
            "interpretation": "per-pair descriptive spread; no significance or benefit conclusion",
        }
        for scenario, rows in grouped.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    args = parser.parse_args()
    plan = read_json(args.experiment / "plan.json")
    validate_plan(plan)
    reports = []
    for path in sorted(args.experiment.glob("*/run.json")):
        try:
            reports.append(summarize_trial(path.parent, plan))
        except (ValueError, KeyError, OSError, json.JSONDecodeError) as error:
            run = read_json(path)
            reports.append(
                dict(
                    run=path.parent.name,
                    **run,
                    client=None,
                    scheduler=None,
                    audit_errors=[str(error)],
                    performance_comparison_eligible=False,
                )
            )
    schedule = read_json(args.experiment / "schedule.json")
    expected = {(row["scenario"], row["policy"], row["repeat"]) for row in schedule}
    observed = {(row["scenario"], row["policy"], row["repeat"]) for row in reports}
    missing = sorted(expected - observed)
    if not reports:
        raise ValueError("No trials; do not summarize an unexecuted experiment as success")
    comparisons = paired_comparisons(reports)
    result = {
        "schema": "c7-analysis-v1",
        "kind": plan["kind"],
        "plan_sha256": plan["sha256"],
        "numerical_limits": "Stage-4 raw cross-shape/prefix logit diagnostics remain FAIL",
        "missing_trials": missing,
        "unexpected_trials": sorted(observed - expected),
        "trials": reports,
        "paired_comparisons": comparisons,
        "repeat_summary": repeat_summary(comparisons),
    }
    write_json(args.experiment / "analysis.json", result)
    with (args.experiment / "metrics.csv").open("w", newline="") as output:
        columns = [
            "run",
            "kind",
            "policy",
            "mode",
            "status",
            "success_rate",
            "window_output_token_s",
            "cohort_output_token_s",
            "window_goodput_req_s",
            "cohort_goodput_req_s",
            "ttft_p50_ms",
            "ttft_p95_ms",
            "itl_p50_ms",
            "itl_p95_ms",
            "itl_p99_ms",
            "tpot_mean_ms",
            "comparison_eligible",
            "audit_errors",
        ]
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        for row in reports:
            client = row["client"] or {}
            writer.writerow(
                dict(
                    **{key: row[key] for key in ("run", "kind", "policy", "mode", "status")},
                    **{key: client.get(key) for key in columns[5:10]},
                    ttft_p50_ms=client.get("ttft_ms", {}).get("p50"),
                    ttft_p95_ms=client.get("ttft_ms", {}).get("p95"),
                    itl_p50_ms=client.get("itl_ms", {}).get("p50"),
                    itl_p95_ms=client.get("itl_ms", {}).get("p95"),
                    itl_p99_ms=client.get("itl_ms", {}).get("p99"),
                    tpot_mean_ms=client.get("tpot_ms", {}).get("mean"),
                    comparison_eligible=row["performance_comparison_eligible"],
                    audit_errors=json.dumps(row["audit_errors"]),
                )
            )
    print(f"Analyzed {len(reports)} trials; missing={len(missing)}; kind={plan['kind']}")
    return int(bool(missing or observed - expected or any(row["audit_errors"] for row in reports)))


if __name__ == "__main__":
    raise SystemExit(main())
