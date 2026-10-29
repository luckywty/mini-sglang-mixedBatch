"""CPU checks for accounting, protocol semantics and fair A/B configuration."""

from __future__ import annotations

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmark/c7"))
from analyze import paired_comparisons, scheduler_metrics
from client import one_request, run_load
from common import client_metrics, digest, validate_plan
from run import server_command


def request(uid="r0", arrival=0, count=2):
    return {
        "id": uid,
        "prompt": uid,
        "input_ids": [100, 101],
        "input_sha256": digest([100, 101]),
        "output_tokens": count,
        "arrival_s": arrival,
    }


def result(uid="r0", sent=0, tokens=(0.1, 0.2), end=0.25, server_uid=0):
    return {
        "id": uid,
        "sent_s": sent,
        "end_s": end,
        "token_times_s": list(tokens),
        "server_uid": server_uid,
        "error": None,
        "done_count": 1,
        "finish_count": 1,
        "role_count": 1,
    }


SLO = {"ttft_ms": 400, "tpot_ms": 200, "max_itl_ms": 250}


def plan():
    value = {
        "schema": "c7-plan-v1",
        "kind": "desktop_benchmark",
        "server": {
            "context": 128,
            "slots": 3,
            "pages": 768,
            "budget": 7,
            "graph": 0,
            "cache": "radix",
            "overlap": True,
        },
        "model": "/test/model",
        "slo": SLO,
        "warmup": [],
        "scenarios": {"low": {"window_s": 1, "requests": [request()]}},
    }
    value["sha256"] = digest(value)
    return value


def test_window_and_drain_denominators_and_token_weighting():
    rows = [request(), request("r1", 0.5, 3)]
    outputs = [result(), result("r1", 0.6, (0.8, 0.9, 1.1), 1.2, 1)]
    metrics = client_metrics(rows, outputs, 1, SLO)
    assert metrics["received_output_tokens"] == 5
    assert metrics["window_output_token_s"] == 4
    assert metrics["cohort_output_token_s"] == pytest.approx(5 / 1.2)
    assert metrics["window_goodput_req_s"] == 1
    assert metrics["cohort_goodput_req_s"] == pytest.approx(2 / 1.2)
    assert metrics["ttft_ms"]["p50"] == pytest.approx(150)
    assert metrics["offered_ttft_ms"]["p50"] == pytest.approx(200)
    assert metrics["itl_ms"]["count"] == 3
    assert metrics["tpot_ms"]["mean"] == pytest.approx(125)


def test_client_lag_cannot_hide_slo_violation():
    output = result(sent=0.8, tokens=(0.9, 1.0), end=1.1)
    metrics = client_metrics([request()], [output], 1, SLO)
    assert metrics["ttft_ms"]["mean"] == pytest.approx(100)
    assert metrics["slo_satisfied_requests"] == 0
    assert metrics["client_dispatch_lag_ms"]["max"] == 800


def test_single_token_has_no_invented_itl_or_tpot():
    metrics = client_metrics([request(count=1)], [result(tokens=(0.1,))], 1, SLO)
    assert metrics["tpot_ms"]["mean"] is None
    assert metrics["itl_ms"]["p99"] is None
    assert metrics["slo_satisfied_requests"] == 1


@pytest.mark.parametrize("change", ["length", "duplicate-finish", "timeout"])
def test_failures_remain_in_success_denominator(change):
    bad = result()
    if change == "length":
        bad["token_times_s"].pop()
    elif change == "duplicate-finish":
        bad["finish_count"] = 2
    else:
        bad["error"] = "TimeoutError"
    metrics = client_metrics([request()], [bad], 1, SLO)
    assert metrics["success_rate"] == metrics["window_goodput_req_s"] == 0
    assert sum(metrics["errors"].values()) == 1


@pytest.mark.parametrize("change", ["missing", "duplicate-id", "duplicate-uid", "time"])
def test_reject_corrupted_client_evidence(change):
    requests = [request(), request("r1")]
    outputs = [result(), result("r1", server_uid=1)]
    if change == "missing":
        outputs.pop()
    elif change == "duplicate-id":
        outputs[1]["id"] = "r0"
    elif change == "duplicate-uid":
        outputs[1]["server_uid"] = 0
    else:
        outputs[0]["token_times_s"].reverse()
    with pytest.raises(ValueError):
        client_metrics(requests, outputs, 1, SLO)


def test_baseline_and_mixed_commands_differ_only_in_feature_flag():
    baseline = server_command(plan(), "baseline", 1919)
    assert server_command(plan(), "mixed", 1919) == baseline + ["--enable-mixed-batching"]
    with pytest.raises(ValueError, match="Decode-priority"):
        server_command(plan(), "decode-priority", 1919)


@pytest.mark.parametrize("change", ["hash", "clamp", "negative-arrival", "bad-graph"])
def test_plan_cannot_silently_change_or_clamp_requests(change):
    value = plan()
    validate_plan(value)
    if change == "hash":
        value["server"]["budget"] = 8
    else:
        if change == "clamp":
            value["scenarios"]["low"]["requests"][0]["output_tokens"] = 128
        elif change == "negative-arrival":
            value["scenarios"]["low"]["requests"][0]["arrival_s"] = -1
        else:
            value["server"]["graph"] = 2
        value["sha256"] = digest({k: v for k, v in value.items() if k != "sha256"})
    with pytest.raises(ValueError):
        validate_plan(value)


def test_no_smoke_or_instrumented_speedup_and_no_missing_pair():
    row = {
        "scenario": "low",
        "repeat": 0,
        "policy": "baseline",
        "performance_comparison_eligible": False,
    }
    assert paired_comparisons([row])[0]["withheld_reason"]
    assert (
        paired_comparisons([row, dict(row, policy="mixed")])[0][
            "cohort_output_ratio_mixed_over_baseline"
        ]
        is None
    )


def test_backend_prefill_decode_intervals_and_duplicate_finished():
    events = [
        {
            "event": "receive",
            "uid": 0,
            "t": 10,
            "input_sha256": digest([100, 101]),
            "max_tokens": 2,
        },
        *[
            {
                "event": "forward",
                "submit_t": stamp,
                "phase": phase,
                "graph": False,
                "rows": [{"uid": 0, "phase": phase}],
                "kv_pages": 10,
                "free_pages": 5,
                "evictable_tokens": 1,
                "protected_tokens": 2,
                "allocated_bytes": 100,
                "reserved_bytes": 200,
            }
            for stamp, phase in [(10.1, "prefill"), (10.2, "decode"), (10.4, "decode")]
        ],
        {"event": "token", "uid": 0, "finished": True},
        {"event": "token", "uid": 0, "finished": True},
        {"event": "shutdown"},
    ]
    metrics = scheduler_metrics(
        events, {"r0": request()}, {"results": [result()], "epoch_perf_counter": 10}, "events"
    )
    assert metrics["prefill_wait_backend_ms"]["mean"] == pytest.approx(100)
    assert metrics["decode_submit_interval_ms"]["mean"] == pytest.approx(200)
    assert metrics["extra_finished"] == 1
    assert metrics["mixed_engine_cuda_ms"] is None
    assert metrics["non_evictable_kv_pages"]["mean"] == 4
    with pytest.raises(ValueError, match="shutdown"):
        scheduler_metrics(
            events[:-1],
            {"r0": request()},
            {"results": [result()], "epoch_perf_counter": 10},
            "events",
        )


@pytest.fixture
def http_server():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            mode = body["prompt"]
            if mode == "slow":
                time.sleep(0.1)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            # Empty first content is still a real model token; finish is separate.
            chunks = [
                {
                    "id": "cmpl-7",
                    "choices": [{"delta": {"role": "assistant"}, "finish_reason": None}],
                },
                {
                    "id": "cmpl-7",
                    "choices": [{"delta": {"content": "hello"}, "finish_reason": None}],
                },
                {"id": "cmpl-7", "choices": [{"delta": {}, "finish_reason": "stop"}]},
            ]
            if mode == "uid-change":
                chunks[1]["id"] = "cmpl-8"
            payload = "".join("data: " + json.dumps(row) + "\n\n" for row in chunks)
            if mode != "no-done":
                payload += "data: [DONE]\n\n"
            self.wfile.write(payload.encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join()


def test_actual_sse_parser_counts_empty_token_and_excludes_end_chunks(http_server):
    output = one_request(http_server, request(), {}, time.perf_counter(), 2)
    assert output["error"] is None
    assert len(output["token_times_s"]) == 2
    assert output["server_uid"] == 7
    assert output["done_count"] == output["finish_count"] == output["role_count"] == 1
    assert output["sent_s"] < output["token_times_s"][0] < output["end_s"]


@pytest.mark.parametrize("mode", ["no-done", "uid-change"])
def test_actual_sse_parser_rejects_bad_protocol(http_server, mode):
    assert one_request(http_server, request(mode), {}, time.perf_counter(), 2)["error"]


def test_client_capacity_drops_are_explicit_without_hidden_queue(http_server):
    requests = [request(f"r{i}") for i in range(3)]
    for row in requests:
        row["prompt"] = "slow"
    data = run_load(http_server, requests, {}, timeout=2, max_clients=1)
    assert sum(row["error"] == "client-capacity-exhausted" for row in data["results"]) == 2
    assert sum(row["error"] is None for row in data["results"]) == 1


def test_scheduler_rejects_wrong_input_hash():
    events = [{"event": "receive", "uid": 0, "input_sha256": "wrong"}, {"event": "shutdown"}]
    with pytest.raises(ValueError, match="tokenization"):
        scheduler_metrics(
            events, {"r0": request()}, {"results": [result()], "epoch_perf_counter": 10}, "events"
        )


@pytest.mark.parametrize("change", ["end-before-token", "nan"])
def test_reject_bad_timestamp_bounds(change):
    output = result()
    if change == "nan":
        output["token_times_s"][0] = float("nan")
    else:
        output["end_s"] = 0.15
    with pytest.raises(ValueError):
        client_metrics([request()], [output], 1, SLO)


def comparable_trial(policy):
    return {
        "scenario": "low",
        "repeat": 0,
        "policy": policy,
        "performance_comparison_eligible": True,
        "plan_sha256": "same-plan",
        "code_sha256": "same-code",
        "mode": "primary",
        "nsys": False,
        "max_clients": 256,
        "timeout_s": 120,
        "actual_server": dict.fromkeys(
            ("gpu", "sm", "total_vram_bytes", "versions", "cuda", "graph_sizes", "environment"),
            "same",
        ),
        "client": {"cohort_output_token_s": 10, "window_goodput_req_s": 1},
    }


def test_paired_ratio_and_goodput_delta_have_explicit_direction():
    baseline, mixed = comparable_trial("baseline"), comparable_trial("mixed")
    mixed["client"].update(cohort_output_token_s=12, window_goodput_req_s=0.5)
    pair = paired_comparisons([baseline, mixed])[0]
    assert pair["cohort_output_ratio_mixed_over_baseline"] == 1.2
    assert pair["goodput_delta_req_s"] == -0.5


@pytest.mark.parametrize("key", ["plan_sha256", "code_sha256", "max_clients", "timeout_s"])
def test_paired_ratios_reject_different_trial_contracts(key):
    baseline, mixed = comparable_trial("baseline"), comparable_trial("mixed")
    mixed[key] = "different"
    pair = paired_comparisons([baseline, mixed])[0]
    assert pair["withheld_reason"]
    assert pair["cohort_output_ratio_mixed_over_baseline"] is None


def test_paired_ratios_reject_different_graph_capture_or_environment():
    baseline, mixed = comparable_trial("baseline"), comparable_trial("mixed")
    mixed["actual_server"]["graph_sizes"] = "different"
    assert paired_comparisons([baseline, mixed])[0]["withheld_reason"]
