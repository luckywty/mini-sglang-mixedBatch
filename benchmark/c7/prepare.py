"""Prepare immutable, tokenizer-verified open-loop A/B workloads without CUDA."""

from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path

from common import digest, validate_plan, write_json


def model_fingerprint(model):
    files = sorted(
        path
        for path in model.iterdir()
        if path.is_file()
        and (
            path.suffix == ".safetensors"
            or path.name.endswith(".json")
            or path.name == "merges.txt"
        )
    )
    if not any(path.suffix == ".safetensors" for path in files):
        raise ValueError("Use a complete local safetensors model directory")
    result = {}
    for path in files:
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(block)
        result[path.name] = {"bytes": path.stat().st_size, "sha256": hasher.hexdigest()}
    return result


def exact_prompt(tokenizer, count, rng):
    special = set(tokenizer.all_special_ids)
    ids = []
    for _ in range(128):
        ids = ids[:count]
        while len(ids) < count:
            token = rng.randrange(100, tokenizer.vocab_size // 2)
            if token not in special:
                ids.append(token)
        prompt = tokenizer.decode(ids)
        ids = tokenizer.encode(prompt)  # Same default add_special_tokens as the real server.
        if len(ids) == count:
            return prompt, ids
    raise ValueError(f"Unable to round-trip exact input length {count}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preset", choices=("smoke", "desktop"), default="desktop")
    parser.add_argument("--requests", type=int, default=96)
    parser.add_argument("--reference-rate", type=float, required=True)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument(
        "--warmup-requests", type=int, help="Default: 8 on desktop, 2 for correctness smoke"
    )
    parser.add_argument("--cache", choices=("radix", "naive"), default="radix")
    parser.add_argument("--budget", type=int)
    parser.add_argument("--graph", type=int)
    parser.add_argument(
        "--overlap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Default off for both policies: native baseline HTTP finishes early with overlap",
    )
    parser.add_argument("--ttft-slo-ms", type=float, default=1000)
    parser.add_argument("--tpot-slo-ms", type=float, default=100)
    parser.add_argument("--max-itl-slo-ms", type=float, default=200)
    parser.add_argument("--max-dispatch-lag-ms", type=float, default=20)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new frozen plan path")
    if args.requests < 4 or args.reference_rate <= 0 or args.max_dispatch_lag_ms <= 0:
        parser.error("Use >=4 requests, positive rate and positive dispatch-lag bound")
    from transformers import AutoTokenizer

    model = args.model.resolve()
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    smoke = args.preset == "smoke"
    rng = random.Random(args.seed)
    server = {
        "context": 128 if smoke else 2048,
        "slots": 3 if smoke else 8,
        "pages": 768 if smoke else 8192,
        "budget": args.budget if args.budget is not None else (7 if smoke else 128),
        "graph": args.graph if args.graph is not None else (0 if smoke else 8),
        "overlap": args.overlap,
        "cache": args.cache,
    }

    def request(uid, long=False):
        bounds = (
            (31, 47) if smoke and long else (5, 12) if smoke else (512, 768) if long else (32, 128)
        )
        count = rng.randint(*bounds)
        prompt, ids = exact_prompt(tokenizer, count, rng)
        return {
            "id": uid,
            "prompt": prompt,
            "input_ids": ids,
            "input_sha256": digest(ids),
            "output_tokens": (
                rng.randint(8, 12)
                if smoke
                else rng.randint(16, 64) if long else rng.randint(64, 128)
            ),
            "cohort": "long-prefill" if long else "decode-stream",
            "arrival_s": 0.0,
        }

    warmup_count = args.warmup_requests if args.warmup_requests is not None else (2 if smoke else 8)
    if warmup_count < 2:
        parser.error("Use at least two warmup requests to exercise short and long prefill")
    warmup = [request(f"warmup-{i}", i % 4 == 1) for i in range(warmup_count)]
    base = [request(f"r{i:04d}", i % 5 == 3) for i in range(args.requests)]
    arrivals, current = [], 0.0
    for _ in base:
        arrivals.append(current)
        current += rng.expovariate(args.reference_rate)
    scenarios = {}
    for name, multiplier in (("low", 0.25), ("near", 0.85), ("overload", 1.3)):
        scenarios[name] = {
            "classification": "candidate; confirm using baseline saturation/SLO pilot",
            "rate_multiplier": multiplier,
            "window_s": current / multiplier,
            "requests": [dict(row, arrival_s=t / multiplier) for row, t in zip(base, arrivals)],
        }
    # Start several long-output short prompts together, then inject long prefills
    # while continuing short-prompt arrivals. Same immutable payloads as above.
    ordered = sorted(base, key=lambda row: row["cohort"] == "long-prefill")
    interference = []
    for i, row in enumerate(ordered):
        arrival = 0.0 if i < min(3, args.requests - 1) else (i - 2) / args.reference_rate
        if row["cohort"] == "long-prefill":
            arrival = (0.1 + 0.05 * len(interference)) / args.reference_rate
        interference.append(dict(row, arrival_s=arrival))
    interference.sort(key=lambda row: row["arrival_s"])
    scenarios["interference"] = {
        "classification": "long-prefill injection with continuing decode-stream arrivals",
        "rate_multiplier": None,
        "window_s": max(row["arrival_s"] for row in interference) + 1 / args.reference_rate,
        "requests": interference,
    }
    plan = {
        "schema": "c7-plan-v1",
        "kind": "smoke_correctness" if smoke else "desktop_benchmark",
        "model": str(model),
        "model_files": model_fingerprint(model),
        "seed": args.seed,
        "reference_rate": args.reference_rate,
        "server": server,
        "sampling": {"temperature": 0, "top_k": 1, "top_p": 1.0, "ignore_eos": True},
        "slo": {
            "ttft_ms": args.ttft_slo_ms,
            "tpot_ms": args.tpot_slo_ms,
            "max_itl_ms": args.max_itl_slo_ms,
        },
        "max_dispatch_lag_ms": args.max_dispatch_lag_ms,
        "warmup": warmup,
        "scenarios": scenarios,
    }
    plan["sha256"] = digest(plan)
    validate_plan(plan)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, plan)
    print(f"Prepared {plan['kind']} plan {plan['sha256']} at {args.output}")


if __name__ == "__main__":
    main()
