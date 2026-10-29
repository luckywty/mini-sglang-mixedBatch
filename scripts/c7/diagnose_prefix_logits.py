"""Strict prefix-hit baseline/mixed numerical diagnostic; exits 1 on mismatch.

This known stage-3 validation gap is separate from lifecycle/Token assertions.
It reuses the real-model test driver; requires the project's pytest environment.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def main() -> int:
    spec = importlib.util.spec_from_file_location(
        "c7_execution_probe",
        Path(__file__).resolve().parents[2] / "tests/engine/test_mixed_execution.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    driver = module.Driver()
    torch = module.torch
    prompt = [100, 101, 102, 103, 104, 105]
    sampler = driver.engine.sampler.sample
    prepare = driver._prepare_batch
    snapshots = []
    canonical_indices = None
    coherent_control = False

    def gather(indices):
        cache = driver.engine.kv_cache
        return torch.stack(
            [
                torch.stack([cache.k_cache(layer)[indices], cache.v_cache(layer)[indices]])
                for layer in range(28)
            ]
        )

    def record_kv(logits, args):
        for req in driver.current_batch.reqs:
            if req.uid == 20:
                indices = driver.engine.page_table[req.table_idx, : req.cached_len].clone()
                snapshots.append(gather(indices))
        return sampler(logits, args)

    def prepare_control(batch):
        # A diagnostic ablation only: before the third output, give the disabled
        # baseline the same canonical prompt pages that the safe mixed path uses.
        for req in batch.reqs:
            if coherent_control and req.uid == 20 and req.cached_len == 7:
                driver.engine.page_table[req.table_idx, :6].copy_(canonical_indices)
        return prepare(batch)

    driver.engine.sampler.sample = record_kv
    driver._prepare_batch = prepare_control

    def run(mixed, overlap, *, control=False):
        nonlocal canonical_indices, coherent_control
        coherent_control = control
        driver.reset(overlap=overlap)
        snapshots.clear()
        driver.enable_mixed_batching = mixed
        driver.prefill_budget = 24
        for uid in (1, 2, 3):
            driver.add(uid, prompt, 3)
        driver.drain()
        driver.events.clear()
        handle = driver.cache_manager.prefix_cache.match_prefix(
            torch.tensor(prompt, dtype=torch.int32)
        ).cuda_handle
        canonical_indices = handle.get_matched_indices()
        canonical = gather(canonical_indices)
        driver.add(20, prompt, 3)
        driver.drain()
        return (
            [message.next_token for message in driver.events],
            [value.float().cpu() for value in driver.logits[20]],
            [value.cpu() for value in snapshots],
            canonical.cpu(),
        )

    try:
        baseline_tokens, baseline_logits, baseline_kv, baseline_canonical = run(False, False)
        mixed_tokens, mixed_logits, mixed_kv, mixed_canonical = run(True, True)
        print(
            "C7_PREFIX_TOKENS " + json.dumps({"baseline": baseline_tokens, "mixed": mixed_tokens}),
            flush=True,
        )
        violations = 0
        for i, (actual, expected) in enumerate(zip(mixed_logits, baseline_logits)):
            error = (actual - expected).abs()
            count = int((error > 0.1 + 0.01 * expected.abs()).sum())
            violations += count
            print(
                "C7_PREFIX_LOGITS "
                + json.dumps(
                    {
                        "output": i,
                        "max_abs": error.max().item(),
                        "violations": count,
                        "atol": 0.1,
                        "rtol": 0.01,
                    }
                ),
                flush=True,
            )
        print("C7_PREFIX_STRICT_NUMERICAL " + ("FAIL" if violations else "PASS"), flush=True)
        print(
            "C7_PREFIX_KV_SOURCE "
            + json.dumps(
                {
                    "initial_canonical_equal": torch.equal(baseline_canonical, mixed_canonical),
                    "mixed_third_uses_canonical_prompt": torch.equal(
                        mixed_kv[2][:, :, :6], mixed_canonical
                    ),
                    "baseline_third_keeps_recomputed_prompt": torch.equal(
                        baseline_kv[2][:, :, :6], baseline_kv[0]
                    ),
                    "prompt_kv_recomputed_vs_canonical_max_abs": (
                        baseline_kv[0].float() - baseline_canonical.float()
                    )
                    .abs()
                    .max()
                    .item(),
                }
            ),
            flush=True,
        )
        for step, (baseline, mixed) in enumerate(zip(baseline_kv, mixed_kv)):
            per_position = (baseline.float() - mixed.float()).abs().amax(dim=(0, 1, 3, 4, 5))
            print(
                "C7_PREFIX_KV_DIFF "
                + json.dumps(
                    {
                        "output": step,
                        "different_positions": torch.where(per_position > 0)[0].tolist(),
                        "max_abs": per_position.max().item(),
                    }
                ),
                flush=True,
            )
        _, control_logits, _, _ = run(False, False, control=True)
        for step, (actual, expected) in enumerate(zip(mixed_logits, control_logits)):
            print(
                "C7_CANONICAL_CONTROL "
                + json.dumps({"output": step, "max_abs": (actual - expected).abs().max().item()}),
                flush=True,
            )
            torch.testing.assert_close(actual, expected, atol=0.1, rtol=0.01)
        return int(violations > 0 or baseline_tokens != mixed_tokens)
    finally:
        driver.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
