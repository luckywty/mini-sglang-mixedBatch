"""Bounded correctness matrix worker; one GPU process per configuration.

No latency/throughput measurements. Run all 2x2x2 configurations separately.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path


def load_driver():
    spec = importlib.util.spec_from_file_location(
        "c7_stage4_driver",
        Path(__file__).resolve().parents[2] / "tests/engine/test_mixed_execution.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Driver, module.torch, module.AbortBackendMsg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overlap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--waves", type=int, default=12)
    args = parser.parse_args()
    Driver, torch, Abort = load_driver()
    driver = Driver(mixed=args.mixed, graph_bs=[1, 2, 3] if args.graph else [], budget=7)
    completed = []
    request_log = []
    receive = driver.receive_msg

    def record_requests(blocking=False):
        messages = receive(blocking)
        for message in messages:
            row = {"kind": type(message).__name__, "uid": message.uid}
            if hasattr(message, "input_ids"):
                row.update(
                    input_ids=message.input_ids.tolist(), sampling=asdict(message.sampling_params)
                )
            request_log.append(row)
        return messages

    driver.receive_msg = record_requests

    def reset(*, keep_cache=False):
        driver.reset(overlap=args.overlap, keep_cache=keep_cache)

    def force(uid, ignored):
        # The legacy sampler also samples intermediate chunks. Derive the golden
        # generation number from sequence position, not sampler invocation order.
        req = next(req for req in driver.current_batch.reqs if req.uid == uid)
        return uid * 100 + max(0, req.cached_len - len(driver.prompts[uid]))

    def check(name, expected, *, aborted=()):
        driver.check_resources()
        actual = defaultdict(list)
        finished = Counter()
        for message in driver.events:
            actual[message.uid].append(message.next_token)
            finished[message.uid] += int(message.finished)
        assert dict(actual) == expected, (name, dict(actual), expected)
        for uid in aborted:
            assert uid not in actual
        if args.mixed:
            assert all(finished[uid] == 1 for uid in expected), (name, finished)
            assert all(not row["graph"] for row in driver.traces if row["phase"] == "mixed")
        if not args.graph:
            assert not any(row["graph"] for row in driver.traces)
        report = {
            "case": name,
            "config": vars(args),
            "status": "PASS",
            "outputs": dict(actual),
            "forward_count": len(driver.traces),
            "mixed_count": sum(row["phase"] == "mixed" for row in driver.traces),
            "graph_count": sum(row["graph"] for row in driver.traces),
            "empty_sampling_forwards": sum(row["sample_count"] == 0 for row in driver.traces),
            "prefill_cached_lengths": [
                list(row["cached_lens"])
                for row in driver.traces
                if row["phase"] in ("prefill", "mixed")
            ],
            "legacy_extra_finished": sum(max(0, count - 1) for count in finished.values()),
        }
        completed.append(report)
        print("C7_STAGE4_CASE " + json.dumps(report), flush=True)

    try:
        reset()
        prompt = driver.tokenizer.encode("The capital of France is")
        driver.add(1, prompt, 4)
        driver.drain()
        natural_single = [message.next_token for message in driver.events]
        assert len(natural_single) == 4
        check("natural-single-cold", {1: natural_single})

        reset()
        long_prompt = driver.tokenizer.encode("Repeat the word hello. " * 12)
        driver.add(1, prompt, 6)
        driver.add(2, long_prompt, 3)
        driver.drain()
        natural = {uid: [m.next_token for m in driver.events if m.uid == uid] for uid in (1, 2)}
        assert len(natural[1]) == 6 and len(natural[2]) == 3
        if args.mixed:
            assert any(row["phase"] == "mixed" for row in driver.traces)
        check("natural-multi-cold", natural)

        reset()
        driver.forced = force
        shared = list(range(200, 219))
        driver.add(3, shared, 3)
        driver.add(4, shared, 2)
        driver.drain()
        check("shared-prefix-cold", {3: [300, 301, 302], 4: [400, 401]})
        reset(keep_cache=True)
        driver.forced = force
        driver.add(5, shared, 3)
        driver.drain()
        assert driver.traces[0]["cached_lens"] == (len(shared) - 1,)
        check("shared-prefix-warm", {5: [500, 501, 502]})

        reset()
        driver.forced = force
        driver.add(6, list(range(300, 427)), 1)
        driver.drain()
        if args.mixed:
            assert any(row["sample_count"] == 0 for row in driver.traces)
        check("input-127-output-1", {6: [600]})

        reset()
        driver.forced = force
        driver.add(7, list(range(400, 525)), 9)  # max_seq_len=128 clamps output to 3
        driver.drain()
        check("max-seq-output-clamp", {7: [700, 701, 702]})

        reset()
        driver.forced = force
        driver.add(8, [500, 501], 3)
        driver.add(9, [510, 511], 3)
        driver.step()
        driver.prefill_budget = 1
        driver.add(10, [520, 521, 522], 1)
        driver.drain()
        if args.mixed:
            assert any(
                row["uids"] == (8, 9) and sum(row["query_lens"]) == 2 for row in driver.traces
            )
        check("decode-over-budget", {8: [800, 801, 802], 9: [900, 901, 902], 10: [1000]})

        reset()
        driver.forced = force
        rng = random.Random(20261029)
        golden, aborted = {}, []
        for wave in range(args.waves):
            base = 20 + wave * 3
            ids = [rng.randrange(1000, 2000) for _ in range(17 + wave % 5)]
            driver.add(base, ids, 4)
            driver.step()  # intermediate prefill has no user-visible output in either policy
            driver.inbox.append(Abort(base))
            driver.add(base + 1, ids[:4] + [2000 + wave], 2)
            driver.step()
            driver.inbox.append(Abort(base))
            driver.add(base + 2, ids[:6] + [2100 + wave], 3)
            driver.drain()
            aborted.append(base)
            golden[base + 1] = [(base + 1) * 100 + i for i in range(2)]
            golden[base + 2] = [(base + 2) * 100 + i for i in range(3)]
            driver.check_resources()
        check("staggered-abort-churn", golden, aborted=aborted)
        print(
            "C7_STAGE4_MATRIX_PASS "
            + json.dumps(
                {
                    "config": vars(args),
                    "cases": len(completed),
                    "churn_requests": args.waves * 3,
                    "request_count": sum(row["kind"] == "UserMsg" for row in request_log),
                    "request_fingerprint": hashlib.sha256(
                        json.dumps(request_log, sort_keys=True).encode()
                    ).hexdigest(),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                }
            ),
            flush=True,
        )
    finally:
        driver.shutdown()


if __name__ == "__main__":
    main()
