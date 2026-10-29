"""Frozen workload contracts and metric definitions; standard library only."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read_json(path):
    return json.loads(Path(path).read_text())


def source_fingerprint(root):
    return digest(
        {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for directory in (root / "python/minisgl", root / "benchmark/c7")
            for path in sorted(directory.rglob("*.py"))
        }
    )


def percentiles(values):
    values = sorted(values)
    if not values:
        return dict.fromkeys(("count", "mean", "p50", "p95", "p99", "max"))

    def quantile(q):
        pos = (len(values) - 1) * q
        lo = int(pos)
        return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (pos - lo)

    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "p50": quantile(0.5),
        "p95": quantile(0.95),
        "p99": quantile(0.99),
        "max": values[-1],
    }


def validate_plan(plan):
    if plan.get("schema") != "c7-plan-v1":
        raise ValueError("Unsupported plan schema")
    payload = {key: value for key, value in plan.items() if key != "sha256"}
    if plan["sha256"] != digest(payload):
        raise ValueError("Plan fingerprint mismatch")
    config = plan["server"]
    if config["context"] <= 1 or min(config[key] for key in ("slots", "pages", "budget")) <= 0:
        raise ValueError("Invalid serving capacity")
    if config["pages"] < config["context"] or config["cache"] not in ("radix", "naive"):
        raise ValueError("Invalid KV/cache configuration")
    if config["graph"] not in (0,) and config["graph"] < 4:
        raise ValueError("Use graph=0 or >=4; native graph CLI captures [1,2,4] at minimum")
    if plan["kind"] not in ("smoke_correctness", "desktop_benchmark"):
        raise ValueError("Unknown experiment kind")
    for key, value in plan["slo"].items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid SLO: {key}")
    for name, scenario in plan["scenarios"].items():
        rows = scenario["requests"]
        if not rows or scenario["window_s"] <= 0 or not math.isfinite(scenario["window_s"]):
            raise ValueError(f"Invalid window: {name}")
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError("Request IDs must be unique")
        arrivals = [row["arrival_s"] for row in rows]
        if arrivals != sorted(arrivals):
            raise ValueError("Arrival times must be ordered")
        for row in rows + plan["warmup"]:
            if not 0 <= row["arrival_s"] < scenario["window_s"]:
                raise ValueError("Arrival outside the offered-load window")
            if not row["input_ids"] or row["input_sha256"] != digest(row["input_ids"]):
                raise ValueError("Prompt token fingerprint mismatch")
            if row["output_tokens"] <= 0 or (
                len(row["input_ids"]) + row["output_tokens"] > config["context"]
            ):
                raise ValueError("Request would be dropped/clamped by the real scheduler")


def client_metrics(requests, results, window, slo):
    """Window counts are [0,W); cohort counts include drain; no synthetic tokens."""
    expected = {row["id"]: row for row in requests}
    if len(expected) != len(requests) or len({row["id"] for row in results}) != len(results):
        raise ValueError("Duplicate workload/client request ID")
    if set(expected) != {row["id"] for row in results}:
        raise ValueError("Missing or unknown client result")
    ttft, offered_ttft, itl, tpot, lags, e2e = [], [], [], [], [], []
    successes, good, window_good, output, window_output = 0, 0, 0, 0, 0
    errors = Counter()
    for row in results:
        request = expected[row["id"]]
        tokens = row["token_times_s"]
        if not all(math.isfinite(t) for t in [row["sent_s"], row["end_s"], *tokens]):
            raise ValueError("Non-finite client timestamp")
        if row["end_s"] < max([row["sent_s"], *tokens]):
            raise ValueError("Request ended before its token timestamps")
        if tokens != sorted(tokens) or any(t < row["sent_s"] for t in tokens):
            raise ValueError("Non-monotonic token timestamps")
        output += len(tokens)
        window_output += sum(0 <= t < window for t in tokens)
        lag = max(0, row["sent_s"] - request["arrival_s"]) * 1000
        lags.append(lag)
        successful = (
            row["error"] is None
            and row["done_count"] == row["finish_count"] == row["role_count"] == 1
            and len(tokens) == request["output_tokens"]
            and row["server_uid"] is not None
        )
        if not successful:
            errors[row["error"] or "protocol-or-length-mismatch"] += 1
            continue
        successes += 1
        first = (tokens[0] - row["sent_s"]) * 1000
        offered = (tokens[0] - request["arrival_s"]) * 1000
        gaps = [(b - a) * 1000 for a, b in zip(tokens, tokens[1:])]
        per_token = sum(gaps) / len(gaps) if gaps else None
        ttft.append(first)
        offered_ttft.append(offered)
        itl.extend(gaps)
        if per_token is not None:
            tpot.append(per_token)
        e2e.append((row["end_s"] - row["sent_s"]) * 1000)
        meets_slo = (
            offered <= slo["ttft_ms"]
            and (per_token is None or per_token <= slo["tpot_ms"])
            and (not gaps or max(gaps) <= slo["max_itl_ms"])
        )
        good += int(meets_slo)
        window_good += int(meets_slo and 0 <= row["end_s"] < window)
    uids = [row["server_uid"] for row in results if row["server_uid"] is not None]
    if len(uids) != len(set(uids)):
        raise ValueError("Two workload requests mapped to the same server UID")
    duration = max([window] + [row["end_s"] for row in results])
    return {
        "offered_requests": len(requests),
        "successful_requests": successes,
        "success_rate": successes / len(requests),
        "errors": dict(errors),
        "received_output_tokens": output,
        "window_s": window,
        "cohort_with_drain_s": duration,
        "window_output_token_s": window_output / window,
        "cohort_output_token_s": output / duration,
        "window_goodput_req_s": window_good / window,
        "cohort_goodput_req_s": good / duration,
        "slo_satisfied_requests": good,
        "ttft_ms": percentiles(ttft),
        "offered_ttft_ms": percentiles(offered_ttft),
        "itl_ms": percentiles(itl),
        "tpot_ms": percentiles(tpot),
        "e2e_ms": percentiles(e2e),
        "client_dispatch_lag_ms": percentiles(lags),
    }
