"""Open-loop HTTP load generator; counts this checkout's one-ack-per-token SSE.

Role belongs to the first token ack. Empty-content token acks count; the separate
finish_reason chunk and [DONE] do not. No usage fields or text re-tokenization.
"""

from __future__ import annotations

import argparse
import http.client
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit

from common import read_json, validate_plan, write_json


def one_request(url, request, sampling, epoch, timeout):
    now = lambda: time.perf_counter() - epoch
    result = {
        "id": request["id"],
        "sent_s": now(),
        "end_s": now(),
        "server_uid": None,
        "token_times_s": [],
        "role_count": 0,
        "finish_count": 0,
        "done_count": 0,
        "error": None,
    }
    connection = None
    try:
        address = urlsplit(url)
        if address.scheme != "http" or address.path not in ("", "/"):
            raise ValueError("Use an http://host:port root URL")
        connection = http.client.HTTPConnection(address.hostname, address.port, timeout=timeout)
        deadline = time.perf_counter() + timeout
        connection.connect()
        sock = connection.sock
        sock.settimeout(max(0.001, deadline - time.perf_counter()))
        payload = dict(
            model="local",
            prompt=request["prompt"],
            max_tokens=request["output_tokens"],
            stream=True,
            **sampling,
        )
        connection.request(
            "POST",
            "/v1/chat/completions",
            json.dumps(payload).encode(),
            {"Content-Type": "application/json", "Connection": "close"},
        )
        sock.settimeout(max(0.001, deadline - time.perf_counter()))
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {response.read(1024)!r}")
        after_done = False
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError("Total request deadline exceeded")
            sock.settimeout(remaining)
            raw = response.readline()
            stamp = now()
            if not raw:
                break
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            if after_done:
                raise ValueError("Data after [DONE]")
            value = line[6:]
            if value == "[DONE]":
                result["done_count"] += 1
                after_done = True
                continue
            chunk = json.loads(value)
            uid = int(chunk["id"].removeprefix("cmpl-"))
            if result["server_uid"] not in (None, uid):
                raise ValueError("Server UID changed within stream")
            result["server_uid"] = uid
            choice = chunk["choices"][0]
            if choice["finish_reason"] is not None:
                result["finish_count"] += 1
                continue
            if result["finish_count"]:
                raise ValueError("Token after finish_reason")
            result["role_count"] += int("role" in choice["delta"])
            result["token_times_s"].append(stamp)
        if len(result["token_times_s"]) != request["output_tokens"]:
            raise ValueError("Output length differs from frozen ignore_eos workload")
        if not result["done_count"] == result["finish_count"] == result["role_count"] == 1:
            raise ValueError("Missing or repeated stream terminator/role")
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        result["end_s"] = now()
        if connection is not None:
            connection.close()
    return result


def run_load(url, requests, sampling, *, timeout, max_clients):
    epoch = time.perf_counter()
    slots = threading.BoundedSemaphore(max_clients)
    results, futures = [], []

    def worker(request):
        try:
            return one_request(url, request, sampling, epoch, timeout)
        finally:
            slots.release()

    with ThreadPoolExecutor(max_workers=max_clients) as pool:
        for request in requests:
            target = epoch + request["arrival_s"]
            while (remaining := target - time.perf_counter()) > 0:
                time.sleep(min(remaining, 0.05))
            if not slots.acquire(blocking=False):
                # Never turn overload into a hidden client-side closed-loop queue.
                stamp = time.perf_counter() - epoch
                results.append(
                    {
                        "id": request["id"],
                        "sent_s": stamp,
                        "end_s": stamp,
                        "server_uid": None,
                        "token_times_s": [],
                        "role_count": 0,
                        "finish_count": 0,
                        "done_count": 0,
                        "error": "client-capacity-exhausted",
                    }
                )
            else:
                futures.append(pool.submit(worker, request))
        results.extend(future.result() for future in futures)
    return {"epoch_perf_counter": epoch, "results": sorted(results, key=lambda row: row["id"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-clients", type=int, default=256)
    args = parser.parse_args()
    plan = read_json(args.plan)
    validate_plan(plan)
    if args.timeout <= 0 or args.max_clients < 1:
        parser.error("Timeout and max-clients must be positive")
    requests = plan["warmup"] if args.warmup else plan["scenarios"][args.scenario]["requests"]
    data = run_load(
        args.url, requests, plan["sampling"], timeout=args.timeout, max_clients=args.max_clients
    )
    data.update(plan_sha256=plan["sha256"], scenario=args.scenario, warmup=args.warmup)
    write_json(args.output, data)
    return int(any(row["error"] for row in data["results"]))


if __name__ == "__main__":
    raise SystemExit(main())
